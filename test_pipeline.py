"""Run CPU regression checks with: python -m unittest -v test_pipeline."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
from torch.nn import functional as F

from data import RapeseedDataset, YIELD_MAP, collate_rapeseed
from evaluate import load_checkpoint
from model import ModelConfig, RapeseedTransformerViG, SpatialPatchGraph, multitask_mse


def small_config(**kwargs):
    config = dict(in_channels=2, embed_dim=16, patch_size=2, vit_layers=1,
                  vit_heads=2, temporal_layers=1, temporal_heads=2,
                  graph_layers=2, graph_k=2, dropout=0.)
    config.update(kwargs)
    return ModelConfig(**config)


def synthetic_batch(h=5, w=7):
    return dict(images=torch.randn(1, 3, 2, h, w),
                observation_mask=torch.ones(1, 3, 1, h, w, dtype=torch.bool),
                time_mask=torch.ones(1, 3, dtype=torch.bool),
                spatial_mask=torch.ones(1, 1, h, w, dtype=torch.bool),
                days_since_sowing=torch.tensor([[1., 10., 30.]]),
                field_yield=torch.ones(1, 1), pixel_yield=torch.ones(1, 1, h, w),
                pixel_valid_mask=torch.ones(1, 1, h, w, dtype=torch.bool))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)

    def test_spatial_edges_exclude_missing_and_self(self):
        graph = SpatialPatchGraph(small_config(graph_k=1, knn_chunk_size=1))
        valid = torch.tensor([[True, False, True, True]])
        xy = torch.tensor([[0., 0.], [0.01, 0.], [1., 0.], [4., 0.]])
        ids, neighbors = graph(valid, xy)[0]
        self.assertEqual(ids.tolist(), [0, 2, 3])
        self.assertEqual(ids[neighbors].tolist(), [[2], [0], [2]])
        ids, neighbors = graph(torch.tensor([[True, False, False, False]]), xy)[0]
        self.assertEqual(tuple(neighbors.shape), (1, 0))

    def test_vit_context_and_backprop_through_all_stages(self):
        model = RapeseedTransformerViG(small_config())
        batch = synthetic_batch()
        batch["images"].requires_grad_()
        captured = []
        hook = model.vit_norm.register_forward_hook(lambda module, inputs, output: captured.append(output))
        result = model.forward_batch(batch)
        hook.remove()
        # The first patch's contextual feature must depend on a different patch.
        gradient = torch.autograd.grad(captured[0][0, 0, 0], batch["images"], retain_graph=True)[0]
        self.assertGreater(float(gradient[0, 0, :, :2, 2:4].abs().sum()), 0.)
        multitask_mse(result, batch)["loss"].backward()
        for name in ("patch_embedding.weight", "vit_encoder.0.self_attn.in_proj_weight",
                     "temporal_encoder.0.self_attn.in_proj_weight", "graph_blocks.0.message.0.weight",
                     "field_head.4.weight", "pixel_head.2.weight"):
            grad = dict(model.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(float(grad.abs().sum()), 0., name)
        self.assertEqual(tuple(result["pixel_pred"].shape), (1, 1, 5, 7))

    def test_masked_values_padding_and_targets_do_not_change_predictions(self):
        model = RapeseedTransformerViG(small_config()).eval()
        batch = synthetic_batch()
        batch["observation_mask"][:, 1] = False  # Entirely cloudy date.
        batch["observation_mask"][:, :, :, :2, :2] = False  # Missing graph node.
        with torch.no_grad():
            original = model.forward_batch(batch)
            changed = {key: value.clone() for key, value in batch.items()}
            changed["images"].masked_fill_(~changed["observation_mask"].expand_as(changed["images"]), float("nan"))
            changed["pixel_yield"].fill_(10000)
            changed["field_yield"].fill_(-10000)
            changed["pixel_valid_mask"].fill_(False)
            altered = model.forward_batch(changed)
            for key in ("field_pred", "pixel_pred"):
                torch.testing.assert_close(original[key], altered[key])
            for key in ("images", "observation_mask", "spatial_mask", "pixel_yield", "pixel_valid_mask"):
                changed[key] = F.pad(changed[key], (0, 5, 0, 3), value=0)
            padded = model.forward_batch(changed)
            torch.testing.assert_close(original["field_pred"], padded["field_pred"], atol=2e-6, rtol=1e-5)
            torch.testing.assert_close(original["pixel_pred"], padded["pixel_pred"][..., :5, :7], atol=2e-6, rtol=1e-5)
            self.assertFalse(original["pixel_prediction_mask"][..., :2, :2].any())

    def test_single_node_ablation_and_empty_input(self):
        for layers in (0, 2):
            model = RapeseedTransformerViG(small_config(graph_layers=layers))
            batch = synthetic_batch(1, 1)
            result = model.forward_batch(batch)
            self.assertTrue(torch.isfinite(result["field_pred"]).all())
            multitask_mse(result, batch)["loss"].backward()
            batch["observation_mask"].fill_(False)
            with self.assertRaisesRegex(ValueError, "No usable patches"):
                model.forward_batch(batch)


def write_raster(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[1], width=values.shape[2],
                       count=values.shape[0], dtype=str(values.dtype), crs="EPSG:32633",
                       transform=from_origin(100, 200, 10, 10)) as destination:
        destination.write(values)


class WorkflowTests(unittest.TestCase):
    def test_training_checkpoint_evaluation_and_geotiff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i, split in enumerate(("train", "val", "test")):
                folder = root / f"field_{split}"
                folder.mkdir()
                metadata = dict(crop="rapeseed", seeding_date="01.01.2020", harvesting_date="31.12.2020",
                                yield_ground_truth=2. + i)
                (folder / f"metadata-{folder.name}.json").write_text(json.dumps(metadata), encoding="utf-8")
                for day in ("20200102", "20200202"):
                    values = np.arange(70, dtype=np.float32).reshape(2, 5, 7) + i
                    write_raster(folder / "s2_images" / f"S2_L2A_{day}.tif", values)
                    write_raster(folder / "scl_masks" / f"S2_L2A_SCL_{day}.tif", np.full((1, 5, 7), 4, dtype=np.uint8))
                write_raster(folder / "yield_masks" / YIELD_MAP, np.full((1, 5, 7), 2. + i, dtype=np.float32))
                rows.append(dict(folder=folder.name, physical_field_id=str(i), split=split))
            manifest = root / "split.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            run = root / "run"
            project = Path(__file__).resolve().parent
            commands = [
                ["train.py", "--root", str(root), "--manifest", str(manifest), "--run-dir", str(run),
                 "--epochs", "1", "--batch-size", "1", "--bands", "1", "2", "--embed-dim", "16",
                 "--patch-size", "2", "--vit-layers", "1", "--temporal-layers", "1", "--graph-layers", "1",
                 "--cpu-threads", "2", "--device", "cpu"],
                ["evaluate.py", "--root", str(root), "--manifest", str(manifest), "--checkpoint", str(run / "best.pt"),
                 "--output-dir", str(root / "evaluation"), "--save-maps", "--device", "cpu", "--num-workers", "1"],
            ]
            for command in commands:
                process = subprocess.run([sys.executable, *command], cwd=project,
                                         capture_output=True, text=True, timeout=90)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            model, checkpoint = load_checkpoint(run / "best.pt", torch.device("cpu"))
            model.eval()
            dataset = RapeseedDataset(root, manifest, "test", normalization=checkpoint["normalization"],
                                      **checkpoint["data_config"])
            batch = collate_rapeseed([dataset[0]], pad_multiple=2)
            expected = model.forward_batch(batch)["pixel_pred"][0, 0, :5, :7].detach().numpy()
            with rasterio.open(root / "evaluation/maps/field_test.tif") as source:
                self.assertEqual(source.shape, (5, 7))
                self.assertEqual(source.transform, from_origin(100, 200, 10, 10))
                np.testing.assert_allclose(source.read(1), expected, atol=2e-6, rtol=1e-5)
            metrics = json.loads((root / "evaluation/metrics.json").read_text())
            self.assertEqual(metrics["pixel_prediction_coverage"], 1.0)
            self.assertEqual(metrics["samples"], 1)
            self.assertEqual(metrics["field"]["r2_undefined_reason"], "fewer_than_two_targets")
            # A copied dataset must reuse the same train-only statistics on another host/path.
            import shutil
            moved = root / "relocated"
            moved.mkdir()
            for row in rows:
                shutil.copytree(root / row["folder"], moved / row["folder"])
            relocated = RapeseedDataset(moved, manifest, "test", normalization=checkpoint["normalization"],
                                        **checkpoint["data_config"])
            torch.testing.assert_close(dataset[0]["images"], relocated[0]["images"])


if __name__ == "__main__":
    unittest.main()
