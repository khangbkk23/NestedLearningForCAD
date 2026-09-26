# models/feature_extractors/cadic_vit_v1.py
"""Strict feature-extractor adapter for the CADIC protocol foundation.

This adapter intentionally has no fallback backbone.  The CADIC paper names a
ViT-Base-Patch8-224 pretrained on ImageNet-21k, but does not publish a
checkpoint identifier in the local paper.  A run therefore has to supply an
explicit checkpoint and identity metadata instead of silently substituting
DINOv2, ViT-B/16, or random weights.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CADICViTConfig:
    model_name: str = "vit_base_patch8_224"
    checkpoint_path: str = ""
    checkpoint_identity: str = ""
    pretraining: str = "ImageNet-21k"
    image_size: int = 224
    patch_size: int = 8
    embed_dim: int = 768
    layer_number: int = 9
    layer_indexing: str = "one_based"
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


class CADICViTFeatureExtractor(nn.Module):
    """Frozen ViT feature extractor with explicit geometry assertions."""

    def __init__(self, config: CADICViTConfig, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        if config.model_name != "vit_base_patch8_224":
            raise ValueError("CADIC v1 requires model_name=vit_base_patch8_224")
        if config.image_size != 224 or config.patch_size != 8 or config.embed_dim != 768:
            raise ValueError("CADIC v1 requires 224px / patch8 / dim768 geometry")
        if config.layer_number < 1 or config.layer_number > 12:
            raise ValueError("layer_number must be in 1..12")
        if config.layer_indexing not in {"one_based", "zero_based"}:
            raise ValueError("layer_indexing must be one_based or zero_based")
        if not config.checkpoint_path or not Path(config.checkpoint_path).is_file():
            raise FileNotFoundError(
                "CADIC exact mode requires an explicit ViT-B/8 checkpoint; "
                f"not found: {config.checkpoint_path!r}"
            )
        if not config.checkpoint_identity:
            raise ValueError("checkpoint_identity is required for an auditable run")
        if config.pretraining.lower() != "imagenet-21k":
            raise ValueError("CADIC v1 requires ImageNet-21k metadata")

        try:
            import timm  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency-specific
            raise RuntimeError(
                "CADIC exact mode requires timm to instantiate ViT-B/8"
            ) from exc

        self.model = timm.create_model(
            config.model_name,
            pretrained=False,
            num_classes=0,
            img_size=config.image_size,
        )
        checkpoint = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must contain a state-dict mapping")
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        blocks = getattr(self.model, "blocks", None)
        if blocks is None or len(blocks) != 12:
            raise RuntimeError("loaded model does not expose exactly 12 ViT blocks")
        self.block_index = (
            config.layer_number - 1
            if config.layer_indexing == "one_based"
            else config.layer_number
        )
        if not 0 <= self.block_index < len(blocks):
            raise RuntimeError(f"invalid selected block index {self.block_index}")
        self._captured: Optional[torch.Tensor] = None
        self._hook = blocks[self.block_index].register_forward_hook(self._capture)

    def _capture(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        if isinstance(output, (tuple, list)):
            output = output[0]
        self._captured = output

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.extract_patch_features(images)

    @torch.no_grad()
    def extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        if images.ndim != 4 or tuple(images.shape[-2:]) != (224, 224):
            raise ValueError(f"expected [B,3,224,224], got {tuple(images.shape)}")
        self._captured = None
        _ = self.model(images)
        if self._captured is None:
            raise RuntimeError("selected ViT block produced no captured tokens")
        tokens = self._captured
        if tokens.ndim != 3 or tokens.shape[1] != 785 or tokens.shape[2] != 768:
            raise RuntimeError(
                f"CADIC geometry assertion failed: expected [B,785,768], got {tuple(tokens.shape)}"
            )
        return tokens[:, 1:, :]

    def protocol_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.config.model_name,
            "checkpoint_path": os.path.abspath(self.config.checkpoint_path),
            "checkpoint_identity": self.config.checkpoint_identity,
            "pretraining": self.config.pretraining,
            "image_size": self.config.image_size,
            "patch_size": self.config.patch_size,
            "embed_dim": self.config.embed_dim,
            "patch_grid": [28, 28],
            "layer_number": self.config.layer_number,
            "layer_indexing": self.config.layer_indexing,
            "block_index": self.block_index,
            "feature_normalization": "unspecified_by_paper",
            "preprocessing": {
                "resize": [224, 224],
                "mean": list(self.config.mean),
                "std": list(self.config.std),
            },
        }

    def close(self) -> None:
        if getattr(self, "_hook", None) is not None:
            self._hook.remove()
            self._hook = None

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown
        try:
            self.close()
        except Exception:
            pass
