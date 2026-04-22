from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .titans_memory import TITANSMemory


class ACCGatingHead(nn.Module):
    """Small embedding classifier used by ACC gating."""

    def __init__(self, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)


class DinoNSP2(nn.Module):
    """
    DINO backbone wrapper with Visual Prompt Tuning and NSP2 instrumentation.

    Key features:
      - Frozen DINO/ViT backbone
      - Trainable visual prompts at early transformer blocks
      - ACC gating head for shift-vs-defect routing
      - Optional TITANS memory integration for Surprise Scalar inference
      - Extraction of ``Q_X W_k^T`` and ``S_P`` tensors for NSP2 covariance
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        img_size: int = 256,
        prompt_length: int = 8,
        prompt_layers: int = 4,
        prompt_dropout: float = 0.0,
        freeze_backbone: bool = True,
        gating_hidden_dim: int = 256,
        gating_dropout: float = 0.1,
        gating_threshold: float = 0.55,
        use_torchhub_fallback: bool = True,
        torchhub_model: str = "dinov2_vits14",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = bool(pretrained)
        self.img_size = int(img_size)
        self.prompt_length = int(prompt_length)
        self.prompt_layers = int(prompt_layers)
        self.gating_threshold = float(gating_threshold)

        self.backbone = self._build_backbone(
            model_name=model_name,
            pretrained=pretrained,
            img_size=img_size,
            use_torchhub_fallback=use_torchhub_fallback,
            torchhub_model=torchhub_model,
        )

        if not hasattr(self.backbone, "blocks"):
            raise RuntimeError("Backbone must expose transformer blocks via .blocks")

        self.embed_dim = int(getattr(self.backbone, "num_features"))
        num_blocks = len(self.backbone.blocks)
        self.num_prompt_layers = min(self.prompt_layers, num_blocks)

        if self.prompt_length <= 0:
            raise ValueError("prompt_length must be positive")

        self.prompt_embeddings = nn.Parameter(
            torch.zeros(self.num_prompt_layers, self.prompt_length, self.embed_dim)
        )
        nn.init.trunc_normal_(self.prompt_embeddings, std=0.02)
        self.prompt_dropout = nn.Dropout(prompt_dropout)

        self.acc_gating = ACCGatingHead(
            embed_dim=self.embed_dim,
            hidden_dim=gating_hidden_dim,
            dropout=gating_dropout,
        )

        # Proxy anomaly head used for differentiable supervision.
        self.proxy_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 1),
        )

        self._titans_memory: Optional[TITANSMemory] = None

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    def _build_backbone(
        self,
        model_name: str,
        pretrained: bool,
        img_size: int,
        use_torchhub_fallback: bool,
        torchhub_model: str,
    ) -> nn.Module:
        try:
            backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                img_size=img_size,
            )
            return backbone
        except Exception as exc:
            if not use_torchhub_fallback:
                raise RuntimeError(f"Unable to load timm model '{model_name}': {exc}") from exc

            hub_model = torch.hub.load(
                "facebookresearch/dinov2",
                torchhub_model,
                pretrained=pretrained,
            )
            if not hasattr(hub_model, "blocks"):
                raise RuntimeError(
                    "TorchHub fallback model does not expose transformer blocks."
                )
            return hub_model

    def set_titans_memory(self, memory: Optional[TITANSMemory]) -> None:
        """Attach (or detach) TITANS memory used during forward routing."""
        self._titans_memory = memory

    def _prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.backbone.patch_embed(x)

        if hasattr(self.backbone, "_pos_embed"):
            tokens = self.backbone._pos_embed(patch_tokens)
        else:
            cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
            tokens = torch.cat([cls_token, patch_tokens], dim=1)
            if hasattr(self.backbone, "pos_embed"):
                pos_embed = self.backbone.pos_embed[:, : tokens.shape[1], :]
                tokens = tokens + pos_embed
            if hasattr(self.backbone, "pos_drop"):
                tokens = self.backbone.pos_drop(tokens)

        if hasattr(self.backbone, "patch_drop"):
            tokens = self.backbone.patch_drop(tokens)
        if hasattr(self.backbone, "norm_pre"):
            tokens = self.backbone.norm_pre(tokens)

        return tokens

    def _extract_affinity_and_aggregation(
        self,
        block: nn.Module,
        tokens_with_prompts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract NSP2 tensors:
          - affinity: ``Q_X W_k^T`` approximation (B, N_image, P)
          - aggregation: prompt states ``S_P`` (B, P, C)
        """
        normed = block.norm1(tokens_with_prompts) if hasattr(block, "norm1") else tokens_with_prompts

        prompt_slice = slice(1, 1 + self.prompt_length)
        image_slice = slice(1 + self.prompt_length, normed.shape[1])

        if (
            hasattr(block, "attn")
            and hasattr(block.attn, "qkv")
            and callable(block.attn.qkv)
        ):
            qkv = block.attn.qkv(normed)
            batch_size, num_tokens, full_dim = qkv.shape
            num_heads = int(getattr(block.attn, "num_heads", 1))
            head_dim = full_dim // (3 * num_heads)

            qkv = qkv.reshape(batch_size, num_tokens, 3, num_heads, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q = qkv[0]
            k = qkv[1]

            q_x = q[:, :, image_slice, :]
            k_p = k[:, :, prompt_slice, :]
            affinity = torch.einsum("bhid,bhjd->bij", q_x, k_p)
            affinity = affinity / math.sqrt(max(head_dim, 1))
        else:
            img_tokens = normed[:, image_slice, :]
            prompt_tokens = normed[:, prompt_slice, :]
            affinity = torch.matmul(img_tokens, prompt_tokens.transpose(1, 2))
            affinity = affinity / math.sqrt(max(normed.shape[-1], 1))

        aggregation = normed[:, prompt_slice, :]
        return affinity.detach(), aggregation.detach()

    def prompt_distribution_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return LayerNorm-based prompt distribution moments (mu, sigma)."""
        prompt_flat = self.prompt_embeddings.reshape(-1, self.embed_dim)
        normalized = F.layer_norm(prompt_flat, (self.embed_dim,))
        mu = normalized.mean()
        sigma = normalized.std(unbiased=False)
        return mu, sigma

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_head"):
            try:
                pooled = self.backbone.forward_head(tokens, pre_logits=True)
            except TypeError:
                pooled = self.backbone.forward_head(tokens)
            if pooled.ndim == 3:
                pooled = pooled[:, 0]
            return pooled

        if hasattr(self.backbone, "norm"):
            tokens = self.backbone.norm(tokens)

        return tokens[:, 0]

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        batch_size = x.shape[0]
        tokens = self._prepare_tokens(x)

        qxwk_mats: List[torch.Tensor] = []
        sp_mats: List[torch.Tensor] = []

        for layer_idx, block in enumerate(self.backbone.blocks):
            if layer_idx < self.num_prompt_layers:
                prompts = self.prompt_embeddings[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)
                prompts = self.prompt_dropout(prompts)

                cls_token = tokens[:, :1, :]
                patch_tokens = tokens[:, 1:, :]
                tokens = torch.cat([cls_token, prompts, patch_tokens], dim=1)

                affinity, aggregation = self._extract_affinity_and_aggregation(block, tokens)
                qxwk_mats.append(affinity)
                sp_mats.append(aggregation)

            tokens = block(tokens)

            if layer_idx < self.num_prompt_layers:
                cls_token = tokens[:, :1, :]
                patch_tokens = tokens[:, 1 + self.prompt_length :, :]
                tokens = torch.cat([cls_token, patch_tokens], dim=1)

        if hasattr(self.backbone, "norm"):
            tokens = self.backbone.norm(tokens)

        embeddings = self._pool_tokens(tokens)

        acc_logits = self.acc_gating(embeddings)
        shift_probability = torch.softmax(acc_logits, dim=-1)[:, 1]

        proxy_logit = self.proxy_head(embeddings).squeeze(-1)
        proxy_score = torch.sigmoid(proxy_logit)

        if self._titans_memory is not None:
            surprise = self._titans_memory.compute_surprise(embeddings)
            route_to_titans = shift_probability <= self.gating_threshold
            image_score = torch.where(route_to_titans, surprise, torch.zeros_like(surprise))
            image_score = image_score.clamp(1e-6, 1 - 1e-6)
            image_logit = torch.logit(image_score)
        else:
            image_logit = proxy_logit
            image_score = proxy_score.clamp(1e-6, 1 - 1e-6)

        anomaly_map = image_score.view(-1, 1, 1, 1).expand(-1, 1, x.shape[-2], x.shape[-1])
        prompt_mu, prompt_sigma = self.prompt_distribution_stats()

        return {
            "features": embeddings,
            "image_score": image_score,
            "image_logit": image_logit,
            "anomaly_map": anomaly_map,
            "patch_features": [],
            "acc_logits": acc_logits,
            "shift_probability": shift_probability,
            "proxy_logit": proxy_logit,
            "proxy_score": proxy_score,
            "qxwk_mats": qxwk_mats,
            "sp_mats": sp_mats,
            "prompt_mu": prompt_mu,
            "prompt_sigma": prompt_sigma,
        }

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["features"]
