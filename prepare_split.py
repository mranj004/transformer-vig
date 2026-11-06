"""Create a reproducible rapeseed-only grouped 60/20/20 split for data.py.

Uses only the Python standard library. Run --help or see README.md.
First create a grouping template, fill in verified physical_field_id values, then
create the split. Alternatively --by-farm groups every record from a farm together
for an explicitly coarser, unseen-farm experiment.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from paths import DEFAULT_DATA_ROOT


SPLITS = ("train", "val", "test")


def discover_rapeseed(root: Path) -> tuple[list[dict], list[str]]:
    """Discover by metadata crop, not the presence of 'rapeseed' in a filename."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Data root does not exist: {root}")
    records, skipped = [], []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        metadata_path = folder / f"metadata-{folder.name}.json"
        if not metadata_path.is_file():
            # A known rapeseed-looking directory must not silently disappear.
            if "rapeseed" in folder.name.lower():
                raise ValueError(f"Missing metadata: {metadata_path}")
            skipped.append(folder.name)
            continue
        meta = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if str(meta.get("crop", "")).strip().lower() != "rapeseed":
            continue
        if not list((folder / "s2_images").glob("S2_L2A_*.tif")):
            raise ValueError(f"No Sentinel-2 images: {folder.name}")
        target = folder / "yield_masks" / "mean_scaled_yield_masked_regional_statistical_outlier.tif"
        if not target.is_file():
            raise ValueError(f"Missing pixel target: {target}")
        try:
            field_yield = float(meta["yield_ground_truth"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing/invalid field target: {folder.name}") from exc
        if not math.isfinite(field_yield):
            raise ValueError(f"Nonfinite field target: {folder.name}")
        farm = str(meta.get("farm_identifier", "")).strip()
        if not farm:
            raise ValueError(f"Missing farm_identifier: {folder.name}")
        records.append({
            "folder": folder.name,
            "farm_identifier": farm,
            "year": str(meta["year"]),
            "centroid_longitude_wgs84": meta.get("centroid_longitude_wgs84", ""),
            "centroid_latitude_wgs84": meta.get("centroid_latitude_wgs84", ""),
        })
    if not records:
        raise ValueError("No rapeseed records found.")
    return records, skipped


def _csv_text(rows: list[dict], columns: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _check_output(path: Path, overwrite: bool):
    if path.exists() and not overwrite:
        raise ValueError(f"Output already exists: {path}. Use --overwrite to replace it explicitly.")
    if path.is_dir():
        raise ValueError(f"Output path is a directory: {path}")


def write_group_template(records: list[dict], path: Path, overwrite=False):
    _check_output(path, overwrite)
    rows = [dict(r, physical_field_id="", field_mask="") for r in records]
    columns = ["folder", "physical_field_id", "farm_identifier", "year",
               "centroid_longitude_wgs84", "centroid_latitude_wgs84", "field_mask"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_csv_text(rows, columns), encoding="utf-8")


def load_physical_groups(records: list[dict], path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"folder", "physical_field_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Grouping CSV requires folder and physical_field_id columns.")
        mapping = {}
        for row in reader:
            folder = (row.get("folder") or "").strip()
            group = (row.get("physical_field_id") or "").strip()
            if not folder or not group:
                raise ValueError(f"Fill every physical_field_id before splitting. Incomplete row: {folder!r}")
            if folder in mapping:
                raise ValueError(f"Duplicate folder in grouping CSV: {folder}")
            mapping[folder] = {"physical_field_id": group,
                               "field_mask": (row.get("field_mask") or "").strip()}
    expected = {r["folder"] for r in records}
    missing, extra = expected - mapping.keys(), mapping.keys() - expected
    if missing or extra:
        raise ValueError(f"Grouping CSV must cover exactly the discovered rapeseed records. "
                         f"Missing: {sorted(missing)[:5]}; unknown: {sorted(extra)[:5]}")
    grouped = [dict(r, **mapping[r["folder"]]) for r in records]
    mask_presence = [bool(r["field_mask"]) for r in grouped]
    if any(mask_presence) and not all(mask_presence):
        raise ValueError("Provide optional field_mask paths for all records or leave all blank.")
    return grouped


def target_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if len(ratios) != 3 or any(not math.isfinite(x) or x <= 0 for x in ratios):
        raise ValueError("Specify three finite, positive split ratios.")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-8):
        raise ValueError("Split ratios must sum to 1.")
    desired = [n * x for x in ratios]
    counts = [math.floor(x) for x in desired]
    remainder_order = sorted(range(3), key=lambda i: (-(desired[i] - counts[i]), i))
    for i in remainder_order[:n - sum(counts)]:
        counts[i] += 1
    return tuple(counts)


def grouped_random_split(records: list[dict], ratios=(0.6, 0.2, 0.2),
                         seed: int = 42, trials: int = 512) -> tuple[list[dict], dict]:
    """Shuffle whole groups; choose cuts close to requested RECORD proportions.

    Examine seeded random permutations and retain the one with smallest summed
    squared deviation from requested record counts. No yield values are used.
    This is a count-balanced random grouped split, not uniform sampling over all
    possible partitions, and not a guarantee of a globally optimal partition.
    """
    if trials < 1:
        raise ValueError("trials must be positive.")
    if len({r['folder'] for r in records}) != len(records):
        raise ValueError("Duplicate record folders.")
    by_group = defaultdict(list)
    for record in records:
        group = record.get("physical_field_id", "")
        if not group:
            raise ValueError("Every record requires a nonempty physical_field_id.")
        by_group[group].append(record)
    groups = sorted(by_group)
    if len(groups) < 3:
        raise ValueError("At least three independent groups are required for three nonempty splits.")
    n = len(records)
    desired = target_counts(n, tuple(ratios))
    rng = random.Random(seed)
    best, best_score, attempts = None, float("inf"), 0
    for attempt in range(trials):
        attempts = attempt + 1
        order = groups.copy()
        rng.shuffle(order)
        cumulative = [0]
        for group in order:
            cumulative.append(cumulative[-1] + len(by_group[group]))
        # For each train cut, the optimum second cut under squared error lies
        # nearest this midpoint. Compare both neighboring attainable counts.
        for i in range(1, len(order) - 1):
            ideal_second = (n + cumulative[i] + desired[1] - desired[2]) / 2
            position = bisect.bisect_left(cumulative, ideal_second, i + 1, len(order))
            choices = sorted({max(i + 1, min(len(order) - 1, position)),
                              max(i + 1, min(len(order) - 1, position - 1))})
            for j in choices:
                counts = (cumulative[i], cumulative[j] - cumulative[i], n - cumulative[j])
                score = sum((a - b) ** 2 for a, b in zip(counts, desired))
                if score < best_score:
                    best_score = score
                    best = (order.copy(), i, j)
        if best_score == 0:
            break
    order, i, j = best
    assignment = {}
    for split, selected in zip(SPLITS, (order[:i], order[i:j], order[j:])):
        assignment.update({group: split for group in selected})
    output = [dict(r, split=assignment[r["physical_field_id"]])
              for r in sorted(records, key=lambda r: r["folder"])]
    group_sets = {s: {r["physical_field_id"] for r in output if r["split"] == s}
                  for s in SPLITS}
    if any(group_sets[a] & group_sets[b] for a, b in
           (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("Internal error: a physical group crossed splits.")
    report = dict(seed=seed, requested_ratios=dict(zip(SPLITS, ratios)),
                  target_record_counts=dict(zip(SPLITS, desired)),
                  total_records=n, total_groups=len(groups),
                  search_trials_requested=trials, search_trials_used=attempts,
                  squared_count_error=best_score,
                  algorithm="seeded_group_permutations_with_record_count_balancing",
                  splits={})
    for split in SPLITS:
        subset = [r for r in output if r["split"] == split]
        report["splits"][split] = dict(
            records=len(subset), groups=len(group_sets[split]), fraction=len(subset) / n,
            by_year=dict(sorted(Counter(r["year"] for r in subset).items())),
            by_farm=dict(sorted(Counter(r["farm_identifier"] for r in subset).items())),
        )
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-group-template", type=Path, metavar="CSV",
                      help="List rapeseed records with blank physical_field_id values; no split is made")
    mode.add_argument("--groups", type=Path, metavar="CSV",
                      help="Reviewed folder-to-physical-field mapping")
    mode.add_argument("--by-farm", action="store_true",
                      help="Coarser alternative: keep all records from each farm together")
    parser.add_argument("--output", type=Path, default=Path("rapeseed_split.csv"))
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.6, 0.2, 0.2),
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        records, skipped = discover_rapeseed(args.root)
        print(f"Found {len(records)} rapeseed field-season records.")
        if args.make_group_template:
            write_group_template(records, args.make_group_template, args.overwrite)
            print(f"Wrote grouping template: {args.make_group_template.resolve()}")
            print("Fill physical_field_id using verified parcel identity across years.")
            print("Coordinates are review aids, not proof of parcel identity. No split was created.")
            return
        summary_path = args.output.with_suffix(".summary.json")
        _check_output(args.output, args.overwrite)
        _check_output(summary_path, args.overwrite)
        if args.groups:
            if args.output.resolve() == args.groups.resolve() or summary_path.resolve() == args.groups.resolve():
                raise ValueError("Split outputs must not overwrite the grouping input.")
            grouped = load_physical_groups(records, args.groups)
            provenance = dict(grouping="user_supplied_physical_field_ids",
                              groups_file=str(args.groups.resolve()),
                              groups_file_sha256=hashlib.sha256(args.groups.read_bytes()).hexdigest())
        else:
            grouped = [dict(r, physical_field_id=f"farm_group:{r['farm_identifier']}", field_mask="")
                       for r in records]
            provenance = dict(grouping="farm", groups_file=None, groups_file_sha256=None)
            print("Using farm groups: this evaluates unseen farms, not just unseen fields.")
        rows, report = grouped_random_split(grouped, tuple(args.ratios), args.seed, args.trials)
        columns = ["folder", "physical_field_id", "split", "farm_identifier", "year"]
        if any(r.get("field_mask") for r in rows):
            columns.append("field_mask")
        manifest_text = _csv_text(rows, columns)
        report.update(provenance)
        report.update(data_root=str(args.root.resolve()),
                      manifest=str(args.output.resolve()),
                      manifest_sha256=hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
                      skipped_directories_without_metadata=skipped,
                      notes=["Group identities are not inferred or verified by this script.",
                             "Ratios target record counts; indivisible groups may prevent exact counts.",
                             "Year/farm counts are reported, not explicitly stratified.",
                             "Yield values are checked for validity but not used to select the split."])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(manifest_text, encoding="utf-8", newline="")
        summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        for split in SPLITS:
            stats = report["splits"][split]
            print(f"{split:5}: {stats['records']:3} records ({stats['fraction']:.1%}), "
                  f"{stats['groups']} groups; target {report['target_record_counts'][split]}")
        print(f"Saved: {args.output.resolve()}")
        print(f"Report: {summary_path.resolve()}")
        print("Reuse this manifest for all models and both yield heads. Refit normalization if the split changes.")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
