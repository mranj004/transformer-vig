"""Evaluate a train.py checkpoint without refitting preprocessing or the model.

Produces field and pixel R²/MSE, per-field predictions, and optional GeoTIFF maps.
Also provides the validation functions imported by train.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import RapeseedDataset, collate_rapeseed
from model import ModelConfig, RapeseedTransformerViG


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_rows(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty predictions/history table")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def choose_device(name="auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu, cuda, cuda:N, or auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but this PyTorch environment cannot use it")
    return device


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


class RegressionAccumulator:
    """Float64 SSE and stable pooled target variance, independent of batch cuts."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.sse = 0.0

    def update(self, target, prediction):
        y = torch.as_tensor(target).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        pred = torch.as_tensor(prediction).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        if y.shape != pred.shape:
            raise ValueError("Metrics require matching target and prediction shapes")
        if not torch.isfinite(y).all() or not torch.isfinite(pred).all():
            raise ValueError("Metrics received nonfinite valid targets or predictions")
        n = y.numel()
        if not n:
            return
        mean = y.mean().item()
        delta = mean - self.mean
        self.m2 += (y - mean).square().sum().item() + delta * delta * self.n * n / (self.n + n)
        self.mean += delta * n / (self.n + n)
        self.n += n
        self.sse += (pred - y).square().sum().item()

    def result(self):
        reason = None
        if self.n < 2:
            reason = "fewer_than_two_targets"
        elif self.m2 <= 0:
            reason = "constant_target"
        return dict(n=self.n, mse=self.sse / self.n if self.n else None,
                    r2=1 - self.sse / self.m2 if reason is None else None,
                    r2_undefined_reason=reason)


def save_prediction_map(directory, batch, outputs, index):
    # Rasterio is supplied by data.py's existing dependencies.
    import rasterio