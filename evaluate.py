"""Evaluate a train.py checkpoint without refitting preprocessing or the model.

Produces field and pixel R-squared/MSE, per-field predictions, and optional GeoTIFF maps.
Also provides the validation functions imported by train.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import RapeseedDataset, collate_rapeseed
from model import ModelConfig, RapeseedTransformerViG, multitask_mse
from paths import DEFAULT_DATA_ROOT


ARCHITECTURE = "spatial_vit_spatial_gnn_v1"


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

    h, w = batch["original_sizes"][index]
    prediction = outputs["pixel_pred"][index, 0, :h, :w].detach().cpu().numpy()
    valid = outputs["pixel_prediction_mask"][index, 0, :h, :w].detach().cpu().numpy()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{Path(batch['sample_id'][index]).name}.tif"
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1,
                       dtype="float32", crs=batch["crs"][index],
                       transform=batch["transform"][index], nodata=np.nan,
                       compress="deflate") as destination:
        destination.write(np.where(valid, prediction, np.nan).astype(np.float32), 1)


@torch.no_grad()
def evaluate_loader(model, loader, device, field_weight=1.0, pixel_weight=1.0,
                    map_directory=None):
    model.eval()
    field_metrics, pixel_metrics = RegressionAccumulator(), RegressionAccumulator()
    totals = dict(loss=0., field_mse=0., pixel_mse=0.)
    rows = []
    samples = covered_pixels = target_pixels = 0
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        outputs = model.forward_batch(batch)
        losses = multitask_mse(outputs, batch, field_weight, pixel_weight)
        n = batch["images"].shape[0]
        samples += n
        for key in totals:
            totals[key] += float(losses[key]) * n
        valid = batch["pixel_valid_mask"] & outputs["pixel_prediction_mask"]
        field_metrics.update(batch["field_yield"], outputs["field_pred"])
        pixel_metrics.update(batch["pixel_yield"][valid], outputs["pixel_pred"][valid])
        covered_pixels += int(valid.sum())
        target_pixels += int(batch["pixel_valid_mask"].sum())
        for i in range(n):
            per_field = RegressionAccumulator()
            per_field.update(batch["pixel_yield"][i][valid[i]], outputs["pixel_pred"][i][valid[i]])
            metrics = per_field.result()
            rows.append(dict(sample_id=batch["sample_id"][i],
                             physical_field_id=batch["physical_field_id"][i],
                             field_target=float(batch["field_yield"][i, 0]),
                             field_prediction=float(outputs["field_pred"][i, 0]),
                             pixel_mse=metrics["mse"], pixel_r2=metrics["r2"],
                             pixel_count=metrics["n"],
                             pixel_coverage=metrics["n"] / int(batch["pixel_valid_mask"][i].sum())))
            if map_directory is not None:
                save_prediction_map(map_directory, cpu_batch, outputs, i)
    if not samples:
        raise ValueError("Evaluation loader is empty")
    metrics = {key: value / samples for key, value in totals.items()}
    metrics.update(field=field_metrics.result(), pixel=pixel_metrics.result(),
                   pixel_prediction_coverage=covered_pixels / target_pixels,
                   samples=samples)
    return metrics, rows


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Checkpoint is not a spatial ViT + spatial GNN model; retrain with train.py")
    model = RapeseedTransformerViG(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-maps", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.cpu_threads < 1:
        parser.error("Batch size/CPU threads must be positive and workers nonnegative")
    torch.set_num_threads(args.cpu_threads)
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if sha256_file(args.manifest) != checkpoint["manifest_sha256"]:
        raise ValueError("Evaluation manifest must match the frozen training split")
    dataset = RapeseedDataset(args.root, args.manifest, split=args.split,
                              normalization=checkpoint["normalization"],
                              **checkpoint["data_config"])
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=partial(collate_rapeseed, pad_multiple=model.config.patch_size))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Evaluation output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, rows = evaluate_loader(model, loader, device,
                                    **checkpoint["loss_weights"],
                                    map_directory=args.output_dir / "maps" if args.save_maps else None)
    write_json(args.output_dir / "metrics.json", dict(split=args.split, **metrics))
    write_rows(args.output_dir / "predictions.csv", rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
