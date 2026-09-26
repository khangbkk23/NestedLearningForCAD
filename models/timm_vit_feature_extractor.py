"""Timm ViT adapter that exposes a selected intermediate transformer block."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn


class TimmViTFeatureExtractor(nn.Module):
    """Expose CLS and patch tokens from a one-based ViT block number.

    Intermediate-block post-normalization is not specified by CADIC. The
    `apply_final_norm` option therefore makes that reproduction choice explicit.
    The default returns the selected block output before the model's final norm.
    """

    def __init__(
        self,
        model_name: str,
        *,
        feature_layer: int = 9,
        apply_final_norm: bool = False,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        # Load Pillow before timm imports torchvision. On Windows this avoids
        # native image-DLL load-order conflicts in some Pixi environments.
        from PIL import Image as _PillowImage  # noqa: F401

        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("The timm dependency is required for the configured timm backbone.") from exc

        self.model_name = str(model_name)
        self.feature_layer = int(feature_layer)
        self.apply_final_norm = bool(apply_final_norm)
        if self.feature_layer < 1:
            raise ValueError("feature_layer is one-based and must be >= 1.")

        self.model = timm.create_model(self.model_name, pretrained=pretrained, num_classes=0)
        if not all(hasattr(self.model, name) for name in ("patch_embed", "blocks", "_pos_embed")):
            raise TypeError(f"Backbone {self.model_name!r} is not a compatible timm VisionTransformer.")
        if self.feature_layer > len(self.model.blocks):
            raise ValueError(
                f"Requested block {self.feature_layer}, but {self.model_name!r} has {len(self.model.blocks)} blocks."
            )

    def forward(self, images: torch.Tensor):
        model = self.model
        tokens = model.patch_embed(images)
        tokens = model._pos_embed(tokens)
        if hasattr(model, "patch_drop"):
            tokens = model.patch_drop(tokens)
        if hasattr(model, "norm_pre"):
            tokens = model.norm_pre(tokens)

        for block_index, block in enumerate(model.blocks, start=1):
            tokens = block(tokens)
            if block_index == self.feature_layer:
                break

        if self.apply_final_norm and hasattr(model, "norm"):
            tokens = model.norm(tokens)
        return SimpleNamespace(last_hidden_state=tokens)
    
