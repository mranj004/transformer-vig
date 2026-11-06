"""Train the dual-head rapeseed Transformer + ViG model using a frozen split.

Uses only train/validation loaders. Test predictions are produced separately by
evaluate.py after model selection. See training_README.md for commands.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import RapeseedDataset, collate_rapeseed, fit_band_stats
from model import ModelConfig, RapeseedTransformerViG, multitask_mse
from evaluate import (choose_device, evaluate_loader, move_batch, sha256_file,
                       write_json, write_rows)


DEFAULT_ROOT = r"C:\Users\Owner\OneDrive\Desktop\UCR\backup-chenlab\other_germany\Germany_raw"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def atomic_checkpoint(path, checkpoint):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train_epoch(model, loader, optimizer, device, field_weight, pixel_weight, grad_clip):
    model.train()
    totals = dict(loss=0., field_mse=0., pixel_mse=0.)
    samples = 0
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model.forward_batch(batch)
        losses = multitask_mse(outputs, batch, field_weight, pixel_weight)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(f"Nonfinite training loss for {batch['sample_id']}")
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        n = batch["images"].shape[0]
        samples += n
        for key in totals:
            totals[key] += float(losses[key].detach()) * n
    if not samples:
        raise ValueError("Training loader is empty")
    return {key: value / samples for key, value in totals.items()}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, default=Path("runs/rapeseed_seed42"))
    p.add_argument("--normalization", type=Path, help="Optional existing matching train-only band-statistics JSON")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.)
    p.add_argument("--field-weight", type=float, default=1.)
    p.add_argument("--pixel-weight", type=float, default=1.)
    p.add_argument("--bands", type=int, nargs="+", default=list(range(1, 13)))
    p.add_argument("--max-dates", type=int, default=24, help="0 keeps all usable dates")
    p.add_argument("--max-days-after-sowing", type=int)
    p.add_argument("--clear-scl", type=int, nargs="+", default=[4, 5])