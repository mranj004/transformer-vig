"""Train multispectral spatial ViT -> spatial graph -> GNN -> yield prediction.

Uses only train/validation loaders. Test predictions are produced separately by
evaluate.py after model selection. See README.md for commands.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import RapeseedDataset, collate_rapeseed, fit_band_stats
from model import ModelConfig, RapeseedTransformerViG, multitask_mse
from paths import DEFAULT_DATA_ROOT
from evaluate import (ARCHITECTURE, choose_device, evaluate_loader, move_batch, sha256_file,
                       write_json, write_rows)


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
    p.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT)
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
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--offset", type=float, default=0.0)
    defaults = ModelConfig()
    for name in ("embed_dim", "patch_size", "vit_layers", "vit_heads", "temporal_layers",
                 "temporal_heads", "graph_layers", "graph_k", "mlp_ratio", "knn_chunk_size"):
        p.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    p.add_argument("--dropout", type=float, default=defaults.dropout)
    p.add_argument("--min-observed-fraction", type=float, default=defaults.min_observed_fraction)
    args = p.parse_args()
    for name in ("epochs", "patience", "batch_size", "cpu_threads"):
        if getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if args.num_workers < 0:
        p.error("num_workers must be nonnegative")
    for name in ("lr", "grad_clip", "weight_decay", "min_delta", "field_weight", "pixel_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name in {"lr", "grad_clip"} and value == 0):
            p.error(f"Invalid {name}")
    if args.field_weight + args.pixel_weight == 0:
        p.error("At least one loss weight must be positive")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    device = choose_device(args.device)
    config = ModelConfig(in_channels=len(args.bands), **{
        name: getattr(args, name) for name in ModelConfig.__dataclass_fields__ if name != "in_channels"
    })
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ValueError("Run directory must be empty; choose a new --run-dir")
    data_config = dict(bands=args.bands, max_dates=args.max_dates,
                       max_days_after_sowing=args.max_days_after_sowing, clear_scl=args.clear_scl,
                       radiometric_scale=args.scale, radiometric_offset=args.offset)
    train_data = RapeseedDataset(args.root, args.manifest, split="train",
                                 normalization=args.normalization, **data_config)
    if args.normalization is None:
        print("Fitting normalization on training observations...", flush=True)
        stats = fit_band_stats(train_data)
        train_data = RapeseedDataset(args.root, args.manifest, split="train",
                                     normalization=stats, **data_config)
    else:
        stats = train_data.normalization
    val_data = RapeseedDataset(args.root, args.manifest, split="val",
                               normalization=stats, **data_config)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.run_dir / "normalization.json", stats)
    write_json(args.run_dir / "config.json", {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    })
    loader_config = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         collate_fn=partial(collate_rapeseed, pad_multiple=config.patch_size),
                         worker_init_fn=seed_worker, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_data, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), **loader_config)
    val_loader = DataLoader(val_data, shuffle=False, **loader_config)
    model = RapeseedTransformerViG(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_weights = dict(field_weight=args.field_weight, pixel_weight=args.pixel_weight)
    manifest_sha256 = sha256_file(args.manifest)
    best_loss, stale, history = float("inf"), 0, []
    print(f"Training spatial ViT -> spatial graph -> GNN on {device}; "
          f"{len(train_data)} train / {len(val_data)} validation samples", flush=True)
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        train_metrics = train_epoch(model, train_loader, optimizer, device,
                                     **loss_weights, grad_clip=args.grad_clip)
        val_metrics, val_rows = evaluate_loader(model, val_loader, device, **loss_weights)
        if not math.isfinite(val_metrics["loss"]):
            raise FloatingPointError("Nonfinite validation loss")
        improved = val_metrics["loss"] < best_loss - args.min_delta
        if improved:
            best_loss, stale = val_metrics["loss"], 0
        else:
            stale += 1
        checkpoint = dict(architecture=ARCHITECTURE, epoch=epoch,
                          model_config=model.get_config(), model_state=model.state_dict(),
                          optimizer_state=optimizer.state_dict(), normalization=stats,
                          data_config=train_data.config, manifest_sha256=manifest_sha256,
                          loss_weights=loss_weights, best_val_loss=best_loss, seed=args.seed)
        atomic_checkpoint(args.run_dir / "last.pt", checkpoint)
        if improved:
            atomic_checkpoint(args.run_dir / "best.pt", checkpoint)
            write_json(args.run_dir / "best_val_metrics.json", val_metrics)
            write_rows(args.run_dir / "best_val_predictions.csv", val_rows)
        history.append(dict(epoch=epoch, seconds=time.monotonic() - start,
                            **{f"train_{key}": value for key, value in train_metrics.items()},
                            **{f"val_{key}": val_metrics[key] for key in ("loss", "field_mse", "pixel_mse")}))
        write_rows(args.run_dir / "history.csv", history)
        print(f"Epoch {epoch}: train={train_metrics['loss']:.6f} "
              f"val={val_metrics['loss']:.6f} best={best_loss:.6f}", flush=True)
        if stale >= args.patience:
            print("Early stopping after validation loss stopped improving.")
            break
    print(f"Saved best checkpoint: {args.run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
