from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from torch.utils.data import DataLoader, Dataset


YIELD_MAP = "mean_scaled_yield_masked_regional_statistical_outlier.tif"
DEFAULT_BAND = tuple(range, 13) #Rasterio uses ONE-based band indexes

def _date(value: str):
    return datetime.strptime(value, "%d.%m.%Y").date()

def _grid(path: Path) -> dict:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Missing raster CRS: {path}")
        return dict(crs=src.crs, transform=src.transform,
                    height=src.height, width=src.width)

def _read_aligned(path: Path, grid: dict, indexes=(1,),
                  resampling=Resampling.nearest) -> np.ndarray:
    """Read C,H,W float32; use NaN for nodata and align in geographic space."""
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Missing raster CRS: {path}")
        values = src.read(list(indexes), out_dtype="float32", masked=True)
        values = np.asarray(values.filled(np.nan), dtype=np.float32)
        same_grid = (src.crs == grid["crs"] and
                     src.transform == grid["transform"] and
                     src.height == grid["height"] and
                     src.width == grid["width"])
        if same_grid:
            return values
        aligned = np.full((len(indexes), grid["height"], grid["width"]),
                          np.nan, dtype=np.float32)
        for i in range(len(indexes)):
            reproject(source=values[i], destination=aligned[i],
                      src_transform=src.transform, src_crs=src.crs,
                      src_nodata=np.nan, dst_transform=grid["transform"],
                      dst_crs=grid["crs"], dst_nodata=np.nan,
                      resampling=resampling)
        return aligned

def _read_manifest(root: Path, manifest: Path) -> list[dict]:
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"folder", "physical_field_id", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest needs columns: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("The split manifest is empty.")
    seen_folders, groups = set(), {}
    for row in rows:
        for key in ("folder", "physical_field_id", "split"):
            row[key] = (row[key] or "").strip()
            if not row[key]:
                raise ValueError(f"Empty {key} in manifest")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {row['split']}")
        folder = (root / row["folder"]).resolve()
        if not folder.is_relative_to(root):
            raise ValueError(f"Folder is outside data root: {folder}")
        if folder in seen_folders:
            raise ValueError(f"Duplicate folder: {folder}")
        seen_folders.add(folder)
        group = row["physical_field_id"]
        if group in groups and groups[group] != row["split"]:
            raise ValueError(f"Physical field occurs in multiple splits: {group}")
        groups[group] = row["split"]
        meta_path = folder / f"metadata-{folder.name}.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if str(metadata.get("crop", "")).lower() != "rapeseed":
            raise ValueError(f"Non-rapeseed record in manifest: {folder.name}")
        row["path"] = folder
        row["metadata"] = metadata
    if not any(r["split"] == "train" for r in rows):
        raise ValueError("Manifest has no training records.")
    return rows
def collate_rapeseed(samples: list[dict], pad_multiple: int = 4) -> dict:

    """
    Pad T,H,W, keeping observations, spatial padding and target masks distinct.

    Boolean masks use True = VALID. PyTorch Transformer key-padding masks use
    True = IGNORE, so invert the relevant validity mask when passing it there.
    spatial_mask denotes the original raster rectangle, NOT a parcel boundary.
    """

    if not samples or pad_multiple < 1:
        raise ValueError("Need samples and a positive pad_multiple.")
    if any(s["band_names"] != samples[0]["band_names"] for s in samples):
        raise ValueError("Cannot batch different band schemas.")
    field_masks = [s["field_mask"] is not None for s in samples]
    if any(field_masks) and not all(field_masks):
        raise ValueError("Provide independent field masks for all samples or none.")
    shapes = [s["images"].shape for s in samples]
    b, t, c = len(samples), max(s[0] for s in shapes), shapes[0][1]
    h = math.ceil(max(s[2] for s in shapes) / pad_multiple) * pad_multiple
    w = math.ceil(max(s[3] for s in shapes) / pad_multiple) * pad_multiple
    batch = {
        "images": torch.zeros(b, t, c, h, w),
        "observation_mask": torch.zeros(b, t, 1, h, w, dtype=torch.bool),
        "time_mask": torch.zeros(b, t, dtype=torch.bool),
        "spatial_mask": torch.zeros(b, 1, h, w, dtype=torch.bool),
        "days_since_sowing": torch.zeros(b, t),
        "field_yield": torch.stack([s["field_yield"] for s in samples]),
        "pixel_yield": torch.zeros(b, 1, h, w),
        "pixel_valid_mask": torch.zeros(b, 1, h, w, dtype=torch.bool),
        "field_mask": torch.zeros(b, 1, h, w, dtype=torch.bool) if all(field_masks) else None,
        "original_sizes": [(s[2], s[3]) for s in shapes],
        "band_names": samples[0]["band_names"],
    }
    for key in ("sample_id", "physical_field_id", "crs", "transform", "acquisition_dates"):
        batch[key] = [s[key] for s in samples]
    for i, sample in enumerate(samples):
        ti, _, hi, wi = sample["images"].shape
        batch["images"][i, :ti, :, :hi, :wi] = sample["images"]
        batch["observation_mask"][i, :ti, :, :hi, :wi] = sample["observation_mask"]
        batch["days_since_sowing"][i, :ti] = sample["days_since_sowing"]
        batch["time_mask"][i, :ti] = True
        batch["spatial_mask"][i, :, :hi, :wi] = True
        for key in ("pixel_yield", "pixel_valid_mask"):
            batch[key][i, :, :hi, :wi] = sample[key]
        if all(field_masks):
            batch["field_mask"][i, :, :hi, :wi] = sample["field_mask"]
    return batch


def fit_band_stats(dataset: RapeseedDataset, output_path=None) -> dict:
    """Fit pixel-weighted band statistics from valid TRAIN observations only."""
    if dataset.split != "train" or dataset.normalization is not None:
        raise ValueError("Fit statistics on an unnormalized training dataset only.")
    count = 0
    mean = np.zeros(len(dataset.bands), dtype=np.float64)
    m2 = np.zeros_like(mean)
    # Combine per-sample moments without cancellation from sum(x*x)-sum(x)^2.
    for index in range(len(dataset)):
        sample = dataset[index]
        x = sample["images"].numpy().transpose(1, 0, 2, 3).reshape(len(mean), -1)
        valid = sample["observation_mask"].numpy()[:, 0].reshape(-1)
        values = x[:, valid].astype(np.float64)
        n = values.shape[1]
        if n == 0:
            continue
        current_mean = values.mean(axis=1)
        delta = current_mean - mean
        m2 += ((values - current_mean[:, None]) ** 2).sum(axis=1) + delta**2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    if not count:
        raise ValueError("No valid training observations for normalization.")
    std = np.sqrt(m2 / count)
    std = np.where(std > 1e-8, std, 1.0)
    result = dict(mean=mean.tolist(), std=std.tolist(), count_per_band=count,
                  config=dataset.config, train_signature=dataset.train_signature,
                  band_names=list(dataset.band_names))
    if output_path is not None:
        Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--max-dates", type=int, default=24)
    parser.add_argument("--max-days-after-sowing", type=int)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--stats", help="Existing train-only normalization JSON")
    parser.add_argument("--fit-stats", help="Write train-only normalization JSON")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.stats and args.fit_stats:
        parser.error("Use --stats or --fit-stats, not both.")
    options = dict(root=args.root, manifest=args.manifest, split=args.split,
                   max_dates=args.max_dates, max_days_after_sowing=args.max_days_after_sowing,
                   radiometric_scale=args.scale, radiometric_offset=args.offset)
    dataset = RapeseedDataset(**options, normalization=args.stats)
    if args.fit_stats:
        print("Computing training-only normalization; reading each training sequence...", flush=True)
        stats = fit_band_stats(dataset, args.fit_stats)
        dataset = RapeseedDataset(**options, normalization=stats)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_rapeseed)
    batch = next(iter(loader))
    print(f"{args.split}: {len(dataset)} field-seasons")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {tuple(value.shape)} {value.dtype}")
    print("sample_id:", batch["sample_id"])
    print("field_mask:", "independent parcel masks" if batch["field_mask"] is not None else "not supplied")
    print("Loaded a batch successfully. No model training was performed.")


if __name__ == "__main__":
    main()