"""Multispectral patches -> spatial ViT -> spatial graph -> GNN -> yield.

Compatible with data.py's RapeseedDataset/collate_rapeseed. Requires PyTorch 2.x.
This is a compact, independently implemented ViG-style adaptation, not the exact
ImageNet ViG architecture or a pretrained checkpoint. See README.md.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 12
    embed_dim: int = 96
    patch_size: int = 4
    vit_layers: int = 2
    vit_heads: int = 4
    temporal_layers: int = 2
    temporal_heads: int = 4
    graph_layers: int = 2       # Set to 0 for a transformer-only ablation.
    graph_k: int = 9
    mlp_ratio: int = 2
    dropout: float = 0.1
    min_observed_fraction: float = 0.25
    knn_chunk_size: int = 256

    def __post_init__(self):
        for name in ("in_channels", "embed_dim", "patch_size", "temporal_layers",
                     "temporal_heads", "vit_layers", "vit_heads", "graph_k",
                     "mlp_ratio", "knn_chunk_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.graph_layers, int) or self.graph_layers < 0:
            raise ValueError("graph_layers must be a nonnegative integer")
        if self.embed_dim % self.temporal_heads:
            raise ValueError("embed_dim must be divisible by temporal_heads")
        if self.embed_dim % self.vit_heads:
            raise ValueError("embed_dim must be divisible by vit_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 < self.min_observed_fraction <= 1:
            raise ValueError("min_observed_fraction must be in (0, 1]")


class SpatialPatchGraph(nn.Module):
    """Connect each valid patch to its k nearest patch centers in raster space.

    Edges depend only on spatial distance, never spectral/learned similarity.
    Each pair stores global valid-node IDs and neighbor indices local to them.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.k = config.graph_k
        self.chunk_size = config.knn_chunk_size

    @torch.no_grad()
    def forward(self, valid: Tensor, coordinates: Tensor) -> list[tuple[Tensor, Tensor]]:
        graphs = []
        for sample_valid in valid:
            ids = sample_valid.nonzero(as_tuple=False).flatten()
            points = coordinates[ids].float()
            n = ids.numel()
            k = min(self.k, max(0, n - 1))
            if k == 0:
                neighbors = torch.empty((n, 0), dtype=torch.long, device=valid.device)
            else:
                chunks = []
                with torch.autocast(device_type=valid.device.type, enabled=False):
                    for start in range(0, n, self.chunk_size):
                        stop = min(start + self.chunk_size, n)
                        distances = torch.cdist(points[start:stop], points,
                                                compute_mode="donot_use_mm_for_euclid_dist")
                        local = torch.arange(stop - start, device=valid.device)
                        distances[local, local + start] = float("inf")
                        # Stable ordering breaks equal-distance ties by raster order.
                        chunks.append(distances.argsort(dim=1, stable=True)[:, :k])
                neighbors = torch.cat(chunks, dim=0)
            graphs.append((ids, neighbors))
        return graphs


class SpatialGraphBlock(nn.Module):
    """Max-relative neighbor aggregation, residual update and channel FFN."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.embed_dim
        self.project = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.message = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(),
                                     nn.Linear(2 * d, d), nn.Dropout(config.dropout))
        self.ffn = nn.Sequential(nn.LayerNorm(d),
                                 nn.Linear(d, config.mlp_ratio * d), nn.GELU(),
                                 nn.Dropout(config.dropout),
                                 nn.Linear(config.mlp_ratio * d, d),
                                 nn.Dropout(config.dropout))

    def forward(self, nodes: Tensor, graphs: list[tuple[Tensor, Tensor]]) -> Tensor:
        projected = self.project(nodes)
        results = []
        for b, (ids, neighbor_ids) in enumerate(graphs):
            if ids.numel() == 0:
                results.append(torch.zeros_like(nodes[b]))
                continue
            center = projected[b, ids]
            if neighbor_ids.shape[1]:
                relative = (center[neighbor_ids] - center[:, None]).amax(dim=1)
            else:
                relative = torch.zeros_like(center)
            updated = nodes[b, ids] + self.message(torch.cat((center, relative), dim=-1))
            updated = updated + self.ffn(updated)
            # Differentiable scatter; all invalid nodes remain zero.
            canvas = updated.new_zeros(nodes[b].shape).index_copy(0, ids, updated)
            results.append(canvas)
        return torch.stack(results)


class RapeseedTransformerViG(nn.Module):
    """Predict field yield [B,1] and yield maps [B,1,H,W] in input-grid order.

    Inputs are normalized imagery and INPUT-derived masks only. Boolean validity
    masks use True=valid. Call forward_batch(batch) to explicitly select the
    permitted inputs from the data.py dictionary; targets are never forwarded.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        d, p = cfg.embed_dim, cfg.patch_size
        # An extra channel exposes which within-patch pixels were observed.
        self.patch_embedding = nn.Conv2d(cfg.in_channels + 1, d, p, stride=p)
        self.spatial_embedding = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.vit_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d, nhead=cfg.vit_heads,
                                       dim_feedforward=cfg.mlp_ratio * d,
                                       dropout=cfg.dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(cfg.vit_layers)
        ])
        self.vit_norm = nn.LayerNorm(d)
        self.time_embedding = nn.Sequential(nn.Linear(7, d), nn.GELU(), nn.Linear(d, d))
        # Construct each layer independently rather than cloning identical weights.
        self.temporal_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d, nhead=cfg.temporal_heads,
                                       dim_feedforward=cfg.mlp_ratio * d,
                                       dropout=cfg.dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(cfg.temporal_layers)
        ])
        self.temporal_norm = nn.LayerNorm(d)
        self.graph_construction = SpatialPatchGraph(cfg)
        self.graph_blocks = nn.ModuleList([SpatialGraphBlock(cfg) for _ in range(cfg.graph_layers)])
        self.final_norm = nn.LayerNorm(d)
        self.field_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                        nn.Dropout(cfg.dropout), nn.Linear(d, 1))
        # Predict p*p separate pixels for each node. PixelShuffle preserves the
        # native raster layout without resizing to a fixed square.
        self.pixel_head = nn.Sequential(nn.Conv2d(d, d, 1), nn.GELU(),
                                        nn.Conv2d(d, p * p, 1), nn.PixelShuffle(p))

    def get_config(self) -> dict:
        return asdict(self.config)

    def forward_batch(self, batch: dict) -> dict[str, Tensor]:
        return self(
            images=batch["images"], observation_mask=batch["observation_mask"],
            days_since_sowing=batch["days_since_sowing"],
            time_mask=batch["time_mask"], spatial_mask=batch["spatial_mask"],
            field_mask=batch.get("field_mask"),
        )

    def _check_inputs(self, images, observation_mask, days_since_sowing,
                      time_mask, spatial_mask, field_mask):
        if images.ndim != 5 or not images.is_floating_point():
            raise ValueError("images must be floating point [B,T,C,H,W]")
        b, t, c, h, w = images.shape
        if min(b, t, c, h, w) < 1 or c != self.config.in_channels:
            raise ValueError("Empty input dimensions or incorrect number of bands")
        expected = [("observation_mask", observation_mask, (b, t, 1, h, w)),
                    ("time_mask", time_mask, (b, t)),
                    ("spatial_mask", spatial_mask, (b, 1, h, w))]
        if field_mask is not None:
            expected.append(("field_mask", field_mask, (b, 1, h, w)))
        for name, value, shape in expected:
            if tuple(value.shape) != shape or value.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean with shape {shape}")
            if value.device != images.device:
                raise ValueError(f"{name} and images must be on the same device")
        if tuple(days_since_sowing.shape) != (b, t) or days_since_sowing.device != images.device:
            raise ValueError("days_since_sowing must have shape [B,T] on the input device")
        if not torch.isfinite(days_since_sowing[time_mask]).all():
            raise ValueError("Real dates must have finite days_since_sowing")

    def forward(self, images: Tensor, observation_mask: Tensor,
                days_since_sowing: Tensor, time_mask: Tensor,
                spatial_mask: Tensor, field_mask: Tensor | None = None) -> dict[str, Tensor]:
        self._check_inputs(images, observation_mask, days_since_sowing,
                           time_mask, spatial_mask, field_mask)
        b, t, c, original_h, original_w = images.shape
        d, p = self.config.embed_dim, self.config.patch_size
        domain = spatial_mask if field_mask is None else spatial_mask & field_mask
        observed = observation_mask & time_mask[:, :, None, None, None] & domain[:, None]
        if not torch.isfinite(images.masked_select(observed.expand_as(images))).all():
            raise ValueError("Nonfinite imagery at a location marked observed")
        # torch.where also removes NaNs/Infs in invalid inputs, unlike multiplying by zero.
        cleaned = torch.where(observed, images, 0)
        pad_h, pad_w = (-original_h) % p, (-original_w) % p
        cleaned = F.pad(cleaned, (0, pad_w, 0, pad_h))
        observed = F.pad(observed, (0, pad_w, 0, pad_h), value=False)
        domain_padded = F.pad(domain, (0, pad_w, 0, pad_h), value=False)
        h, w = cleaned.shape[-2:]
        gh, gw = h // p, w // p
        n = gh * gw
        observed_flat = observed.reshape(b * t, 1, h, w)
        image_flat = cleaned.reshape(b * t, c, h, w)
        embedded = self.patch_embedding(torch.cat((image_flat, observed_flat.to(images.dtype)), dim=1))
        # Fractions exclude spatial padding and independent out-of-field pixels.
        domain_count = F.avg_pool2d(domain_padded.float(), p, p) * (p * p)
        observed_count = F.avg_pool2d(observed_flat.float(), p, p).reshape(b, t, 1, gh, gw) * (p * p)
        fraction = observed_count / domain_count[:, None].clamp_min(1)
        date_valid = ((fraction >= self.config.min_observed_fraction) &
                      (observed_count > 0)).reshape(b, t, n).transpose(1, 2)
        node_valid = date_valid.any(dim=-1)  # B,N
        if not node_valid.any(dim=1).all():
            missing = (~node_valid.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"No usable patches in batch samples {missing}; inspect masks or coverage threshold")

        # Spatial ViT: attend over patches within each image, before graph creation.
        # Pixel-based positions are invariant to other samples' batch padding.
        yy, xx = torch.meshgrid(
            (torch.arange(gh, device=images.device, dtype=torch.float32) + 0.5) * p / 128,
            (torch.arange(gw, device=images.device, dtype=torch.float32) + 0.5) * p / 128,
            indexing="ij",
        )
        xy = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
        spatial_features = torch.cat((xy, (2 * math.pi * xy).sin(), (2 * math.pi * xy).cos()), dim=-1)
        tokens = embedded.flatten(2).transpose(1, 2)
        tokens = tokens + self.spatial_embedding(spatial_features.to(images.dtype))[None]
        image_valid = date_valid.transpose(1, 2).reshape(b * t, n)
        active_images = image_valid.any(dim=1).nonzero(as_tuple=False).flatten()
        # Entirely cloudy/padded dates never enter attention; invalid keys are masked.
        spatial_sequence = torch.where(image_valid[..., None], tokens, 0)[active_images]
        for layer in self.vit_encoder:
            spatial_sequence = layer(spatial_sequence,
                                     src_key_padding_mask=~image_valid[active_images])
        spatial_sequence = torch.where(image_valid[active_images, :, None],
                                       self.vit_norm(spatial_sequence), 0)
        tokens = spatial_sequence.new_zeros((b * t, n, d)).index_copy(
            0, active_images, spatial_sequence)
        tokens = tokens.reshape(b, t, n, d).permute(0, 2, 1, 3)
        # Fuse contextual features across dates for one graph node per patch.
        safe_days = torch.where(time_mask, days_since_sowing, 0).to(images.dtype)
        time_features = [safe_days / 365.25]
        for period in (365.25, 90.0, 30.0):
            phase = safe_days * (2 * math.pi / period)
            time_features.extend((phase.sin(), phase.cos()))
        time_position = self.time_embedding(torch.stack(time_features, dim=-1))
        tokens = tokens + time_position[:, None]
        tokens = torch.where(date_valid[..., None], tokens, 0)
        flat_tokens = tokens.reshape(b * n, t, d)
        flat_valid = date_valid.reshape(b * n, t)
        active = node_valid.reshape(-1).nonzero(as_tuple=False).flatten()
        sequence = flat_tokens[active]
        valid_dates = flat_valid[active]
        # Never send an entirely masked sequence into temporal attention.
        for layer in self.temporal_encoder:
            sequence = layer(sequence, src_key_padding_mask=~valid_dates)
        sequence = self.temporal_norm(sequence)
        sequence = torch.where(valid_dates[..., None], sequence, 0)
        pooled = sequence.sum(dim=1) / valid_dates.sum(dim=1, keepdim=True).to(sequence.dtype)
        nodes = pooled.new_zeros((b * n, d)).index_copy(0, active, pooled).reshape(b, n, d)

        # Construct spatial edges once; GNN layers propagate over this same graph.
        graphs = self.graph_construction(node_valid, xy) if self.graph_blocks else []
        for graph in self.graph_blocks:
            nodes = graph(nodes, graphs)
        nodes = torch.where(node_valid[..., None], self.final_norm(nodes), 0)

        # Area-weighted pooling handles partial edge/parcel patches.
        weights = domain_count.flatten(1).to(nodes.dtype) * node_valid.to(nodes.dtype)
        field_features = (nodes * weights[..., None]).sum(dim=1) / weights.sum(dim=1, keepdim=True)
        field_pred = self.field_head(field_features)
        feature_grid = nodes.transpose(1, 2).reshape(b, d, gh, gw)
        pixel_pred = self.pixel_head(feature_grid)[..., :original_h, :original_w]
        patch_mask = node_valid.reshape(b, 1, gh, gw)
        pixel_support = patch_mask.repeat_interleave(p, -2).repeat_interleave(p, -1)
        pixel_support = pixel_support[..., :original_h, :original_w] & domain
        pixel_pred = torch.where(pixel_support, pixel_pred, 0)
        return dict(field_pred=field_pred, pixel_pred=pixel_pred,
                    pixel_prediction_mask=pixel_support, patch_valid_mask=patch_mask)


def multitask_mse(outputs: dict, batch: dict, field_weight: float = 1.0,
                  pixel_weight: float = 1.0) -> dict[str, Tensor]:
    """Field MSE plus per-field-averaged pixel MSE on valid supported targets.

    Keep this pixel mask for evaluation too, and report prediction coverage. No
    target or target mask participates in model.forward_batch(). All tensors must
    already be on the model device. Returns differentiable losses for training.
    """
    if any(not math.isfinite(x) or x < 0 for x in (field_weight, pixel_weight)):
        raise ValueError("Loss weights must be finite and nonnegative")
    if field_weight + pixel_weight == 0:
        raise ValueError("At least one loss weight must be positive")
    field_pred, pixel_pred = outputs["field_pred"], outputs["pixel_pred"]
    field_target, pixel_target = batch["field_yield"], batch["pixel_yield"]
    if field_pred.shape != field_target.shape or pixel_pred.shape != pixel_target.shape:
        raise ValueError("Prediction and target shapes must match exactly")
    target_valid = batch["pixel_valid_mask"]
    prediction_valid = outputs["pixel_prediction_mask"]
    for mask in (target_valid, prediction_valid):
        if mask.shape != pixel_pred.shape or mask.dtype != torch.bool:
            raise ValueError("Pixel masks must be boolean with the same shape as pixel predictions")
    mask = target_valid & prediction_valid
    counts = mask.sum(dim=(1, 2, 3))
    if (counts == 0).any():
        raise ValueError("At least one field has no overlapping valid pixel targets and predictions")
    if not torch.isfinite(field_target).all() or not torch.isfinite(pixel_target[mask]).all():
        raise ValueError("Nonfinite valid yield targets")
    # Float32 reductions avoid squared-error overflow under mixed precision.
    field_loss = (field_pred.float() - field_target.float()).square().mean()
    safe_target = torch.where(mask, pixel_target.float(), 0)
    safe_pred = torch.where(mask, pixel_pred.float(), 0)
    error = (safe_pred - safe_target).square()
    pixel_loss = (error.sum(dim=(1, 2, 3)) / counts).mean()
    return dict(loss=field_weight * field_loss + pixel_weight * pixel_loss,
                field_mse=field_loss, pixel_mse=pixel_loss,
                valid_pixel_count=counts.sum())


def main():
    """Small synthetic shape/backprop check; does not train on user data."""
    torch.manual_seed(42)
    model = RapeseedTransformerViG()
    batch = {
        "images": torch.randn(2, 4, 12, 12, 16),
        "observation_mask": torch.ones(2, 4, 1, 12, 16, dtype=torch.bool),
        "time_mask": torch.ones(2, 4, dtype=torch.bool),
        "spatial_mask": torch.ones(2, 1, 12, 16, dtype=torch.bool),
        "days_since_sowing": torch.tensor([[10., 40., 80., 120.]]).repeat(2, 1),
        "field_yield": torch.full((2, 1), 3.0),
        "pixel_yield": torch.full((2, 1, 12, 16), 3.0),
        "pixel_valid_mask": torch.ones(2, 1, 12, 16, dtype=torch.bool),
    }
    batch["observation_mask"][0, :, :, :4, :4] = False
    result = model.forward_batch(batch)
    losses = multitask_mse(result, batch)
    losses["loss"].backward()
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    for name, value in result.items():
        print(name, tuple(value.shape), value.dtype)
    print("Synthetic forward/backward passed. No trained weights or performance scores were produced.")


if __name__ == "__main__":
    main()
