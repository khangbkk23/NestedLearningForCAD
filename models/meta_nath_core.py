"""
meta_nath_core.py
-----------------
MetaNATHCore - Phase 1 + 2 system core for Meta-NATH CAD.

Components:
    Eyes       = Frozen vision backbone (DINOv2 now, DINOv3-ready adapter)
    Fast Memory = TTTEngine (TITANS Delta Rule)
    Gate       = ACCGating (Social Welfare)
    Slow Memory = CADICCoreset (Unified Memory Bank)

Design notes:
    - the backbone always stays in eval mode, even when model.train() is called
    - TTTEngine / ACCGating / CADICCoreset are not nn.Module instances
      and require a custom full_state_dict() for complete checkpoints
    - forward() runs Phase 1 (TTT) + Phase 2 (consolidation)
      while scoring is handled separately by score_image()
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .titans_memory import TTTEngine
from .acc_gating import ACCGating
from .cadic_coreset import CADICCoreset

logger = logging.getLogger(__name__)


class _FallbackBackbone(nn.Module):
    """Lightweight local backbone used when the gated HF checkpoint is unavailable."""

    def __init__(self, d: int, patch_size: int = 16):
        super().__init__()
        self.d = d
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, d, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor):
        patches = self.patch_embed(x)                       # [B, d, H', W']
        patches = patches.flatten(2).transpose(1, 2)        # [B, N, d]
        patches = self.norm(patches)
        cls = self.cls_token.expand(x.shape[0], -1, -1)     # [B, 1, d]
        hidden_states = torch.cat([cls, patches], dim=1)    # [B, 1 + N, d]
        return SimpleNamespace(last_hidden_state=hidden_states)


class MetaNATHCore(nn.Module):

    def __init__(
        self,
        d: int = 768,
        tau_acc: float = 0.25,
        max_coreset_size: int = 1000,
        n_patch: Optional[int] = None,
        store_images: bool = False,
        device: str | None = None,
        backbone_name: str = "facebook/dinov2-base",
    ):
        """
        Args:
            d:                Embedding dimension (768 for DINOv2-base/current prototype).
            tau_acc:          ACC Gating threshold (instruction_CAD section 5.8, default 0.25).
            max_coreset_size: Maximum number of CADIC entries.
            n_patch:          Number of patch tokens. None infers from backbone output.
            store_images:     Store raw image tensors for N2B-NC Phase 3.
            device:           'cuda' or 'cpu'; auto-detected when None.
            backbone_name:    HuggingFace checkpoint for the main backbone.
        """
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_str = device
        self.d = d
        self.backbone_name = backbone_name
        self.patch_grid: Optional[Tuple[int, int]] = None

        # ----------------------------------------------------------------
        # 1. Eyes - frozen vision backbone.
        #    Current prototype uses HuggingFace DINOv2; adapter also supports DINOv3 dict outputs.
        # ----------------------------------------------------------------
        logger.info(f"[MetaNATH] Loading backbone ({backbone_name})...")
        try:
            self.backbone = AutoModel.from_pretrained(backbone_name)
        except Exception as exc:
            logger.warning(
                "[MetaNATH] Could not load gated HF backbone; using local fallback backbone instead. "
                f"Reason: {exc}"
            )
            self.backbone = _FallbackBackbone(d=d)
            
        self.backbone = self.backbone.to(device)
        self.backbone.eval()
        # Keep the backbone fully frozen in Phase 1-2.
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # ----------------------------------------------------------------
        # 2. Fast Memory - TITANS Delta Rule.
        # ----------------------------------------------------------------
        self.ttt_engine = TTTEngine(d=d, device=device)

        # ----------------------------------------------------------------
        # 3. Gate - ACC Gating.
        # ----------------------------------------------------------------
        self.gating = ACCGating(tau=tau_acc)

        # ----------------------------------------------------------------
        # 4. Slow Memory - CADIC Incremental Coreset.
        # ----------------------------------------------------------------
        self.coreset = CADICCoreset(
            max_size=max_coreset_size,
            d=d,
            n_patch=n_patch,
            store_images=store_images,
            device=device,
        )

    # ------------------------------------------------------------------
    # Override train() so the backbone always stays in eval mode.
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "MetaNATHCore":
        """
        Force the backbone to remain in eval mode, even if a Trainer calls train().
        """
        super().train(mode)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        return self

    # ------------------------------------------------------------------
    # Forward - Phase 1 (TTT) + Phase 2 (Consolidation)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        task_id: int = 0,
        update_coreset: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the Phase 1 + 2 pipeline for an image batch.

        Args:
            x:              [B, 3, 224, 224]
            task_id:        Current task id for CADIC logging.
            update_coreset: False for pure inference without Slow Memory updates.

        Returns:
            dict with keys:
                z_cls:            [B, d]    - original CLS token from the backbone
                z_updated:        [B, d]    - CLS token after TTT adaptation
                z_patches:        [B, N, d] - patch tokens for anomaly scoring
                surprise:         float
                acc_score:        float
                approved:         bool      - ACCGating approval flag
                coreset_updated:  bool      - whether Coreset changed
        """
        # --- Step 1: feature extraction with the frozen backbone. ---
        z_cls, z_patches, patch_grid = self.extract_features(x)

        # --- Step 2: Test-Time Adaptation with TITANS Fast Memory. ---
        z_updated, surprise = self.ttt_engine.process(z_cls)

        # --- Step 3: gate the adapted representation. ---
        approved, acc_score = self.gating.should_consolidate(z_updated, z_cls)

        # --- Step 4: consolidate approved samples into Slow Memory. ---
        coreset_updated = False
        if update_coreset and approved:
            # Store adapted embeddings, because z_updated reflects the batch
            # after Fast Memory has absorbed it.
            n_updated = self.coreset.update_batch(
                cls_embs=z_updated.detach(),
                patch_embs_batch=z_patches.detach(),
                images=x.detach() if self.coreset.store_images else None,
                task_id=task_id,
            )
            coreset_updated = n_updated > 0

        return {
            "z_cls":           z_cls,
            "z_updated":       z_updated,
            "z_patches":       z_patches,
            "surprise":        surprise,
            "acc_score":       acc_score,
            "approved":        approved,
            "coreset_updated": coreset_updated,
            "coreset_n_updated": n_updated if update_coreset and approved else 0,
            "patch_grid":      patch_grid,
        }

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        Return normalized CLS token, patch tokens, and patch grid from the backbone.

        Supports HuggingFace DINOv2 (`last_hidden_state`) and the DINOv3-style
        dict format (`x_norm_clstoken`, `x_norm_patchtokens`) for later upgrade.
        """
        out = self.backbone(x)

        if isinstance(out, dict) and "x_norm_clstoken" in out and "x_norm_patchtokens" in out:
            z_cls = out["x_norm_clstoken"]
            z_patches = out["x_norm_patchtokens"]
        else:
            if hasattr(out, "last_hidden_state"):
                hidden_states = out.last_hidden_state
            elif isinstance(out, dict) and "last_hidden_state" in out:
                hidden_states = out["last_hidden_state"]
            else:
                raise RuntimeError(
                    "[MetaNATH] Unexpected backbone output format. "
                    "Expected `last_hidden_state` or DINO token dict."
                )

            z_cls = hidden_states[:, 0, :]
            z_patches = hidden_states[:, 1:, :]

        if z_cls.shape[-1] != self.d or z_patches.shape[-1] != self.d:
            raise RuntimeError(
                f"[MetaNATH] Backbone dim mismatch: expected d={self.d}, "
                f"got cls={z_cls.shape[-1]}, patches={z_patches.shape[-1]}."
            )

        n_patch = z_patches.shape[1]
        grid = math.isqrt(n_patch)
        if grid * grid != n_patch:
            raise RuntimeError(
                f"[MetaNATH] Patch count must be square for anomaly maps, got {n_patch}."
            )

        if self.coreset.n_patch is None:
            self.coreset.n_patch = n_patch
            self.coreset.patch_grid = (grid, grid)
        elif self.coreset.n_patch != n_patch:
            raise RuntimeError(
                f"[MetaNATH] Configured n_patch={self.coreset.n_patch}, "
                f"but backbone produced {n_patch}. Set model.n_patch: null "
                "or align dataset.img_size with the backbone patch size."
            )

        self.patch_grid = (grid, grid)
        if self.coreset.patch_grid is None:
            self.coreset.patch_grid = self.patch_grid

        return z_cls, z_patches, self.patch_grid

    # ------------------------------------------------------------------
    # Inference - Anomaly scoring, called separately from forward().
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_image(
        self,
        x: torch.Tensor,
        b: int = 2,
    ) -> Dict[str, Any]:
        """
        Compute anomaly scores for test images with CADIC Coreset.

        Does not update TTT or Coreset; this is pure inference.

        Args:
            x: [1, 3, 224, 224] or [B, 3, 224, 224]
            b: neighborhood size for Eq. 9 (default 2)

        Returns:
            dict with:
                s_img:       float or List[float]    - image-level score
                s_pix:       [N_patch] tensor        - pixel-level scores
                anomaly_map: [H, W] tensor           - upsampled to input size
        """
        if len(self.coreset) == 0:
            raise RuntimeError(
                "[MetaNATH] Coreset is empty. Run at least one forward() "
                "with update_coreset=True before calling score_image()."
            )

        _, z_patches, patch_grid = self.extract_features(x)

        # --- Batch Scoring (CADIC Eq. 8-9) ---
        # Use optimized batch scoring.
        s_img_batch, s_pix_batch = self.coreset.compute_anomaly_score(
            patch_embs_batch=z_patches,
            b=b,
        )

        # --- Batch upsampling. ---
        B = z_patches.shape[0]
        H_patch, W_patch = patch_grid
        
        # [B, 1, 16, 16]
        s_pix_reshaped = s_pix_batch.reshape(B, 1, H_patch, W_patch).to(self.device_str)
        
        # Interpolate the full batch in one GPU operation.
        anomaly_maps_batch = F.interpolate(
            s_pix_reshaped,
            size=(x.shape[-2], x.shape[-1]),
            mode="bilinear",
            align_corners=False,
        ) # [B, 1, H, W]

        # Package results.
        results = []
        for i in range(B):
            results.append({
                "s_img":       s_img_batch[i].item(),
                "s_pix":       s_pix_batch[i],
                "anomaly_map": anomaly_maps_batch[i].squeeze().cpu(),
            })

        if B == 1:
            return results[0]
        return {"batch": results}

    # ------------------------------------------------------------------
    # Checkpointing, custom because TTTEngine/ACCGating/CADICCoreset are not nn.Module.
    # ------------------------------------------------------------------

    def full_state_dict(self, include_backbone: bool = True, include_images: bool = True) -> dict:
        """
        Save backbone weights, TITANS memory, gating history, and Coreset.

        Use this instead of model.state_dict() for complete checkpoints.
        """
        return {
            "backbone":   self.backbone.state_dict() if include_backbone else None,
            "backbone_name": self.backbone_name,
            "ttt_engine": self.ttt_engine.state_dict(),
            "gating":     self.gating.state_dict(),
            "coreset":    self.coreset.state_dict(include_images=include_images),
            "config": {
                "d":                self.d,
                "device":           self.device_str,
                "patch_grid":       self.patch_grid,
            },
        }

    def load_full_state_dict(self, sd: dict) -> None:
        """Load a checkpoint produced by full_state_dict()."""
        if sd.get("backbone") is not None:
            self.backbone.load_state_dict(sd["backbone"])
        self.backbone_name = sd.get("backbone_name", self.backbone_name)
        self.ttt_engine.load_state_dict(sd["ttt_engine"])
        self.gating.load_state_dict(sd["gating"])
        self.coreset.load_state_dict(sd["coreset"])
        patch_grid = sd.get("config", {}).get("patch_grid")
        self.patch_grid = tuple(patch_grid) if patch_grid else self.coreset.patch_grid
        # Keep the backbone frozen after loading.
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def refresh_coreset_embeddings(self, batch_size: int = 32) -> dict:
        """
        Recompute stored Coreset embeddings with the current backbone.

        Phase 3 updates the backbone. Without this refresh, test embeddings are
        produced by the updated backbone while the coreset still lives in the
        old feature space.
        """
        if len(self.coreset) == 0:
            return {"refreshed": False, "reason": "empty_coreset", "entries": 0}
        if not self.coreset.images or any(img is None for img in self.coreset.images):
            return {"refreshed": False, "reason": "missing_images", "entries": len(self.coreset)}

        self.backbone.eval()
        cls_embeddings = []
        patch_embeddings = []
        batch_size = max(1, int(batch_size))

        for start in range(0, len(self.coreset.images), batch_size):
            batch = torch.stack(self.coreset.images[start:start + batch_size]).to(self.device_str)
            z_cls, z_patches, _ = self.extract_features(batch)
            cls_embeddings.extend([z.detach() for z in z_cls])
            patch_embeddings.extend([z.detach() for z in z_patches])

        self.coreset.replace_all_embeddings(cls_embeddings, patch_embeddings)
        return {
            "refreshed": True,
            "reason": "ok",
            "entries": len(self.coreset),
            "batch_size": batch_size,
            "patch_grid": list(self.coreset.patch_grid) if self.coreset.patch_grid else None,
        }

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def log_status(self) -> None:
        """Log a compact system status summary."""
        logger.info(
            f"[MetaNATH] "
            f"coreset={len(self.coreset)}/{self.coreset.max_size} | "
            f"approval_rate={self.gating.approval_rate():.2%} | "
            f"avg_acc={self.gating.running_avg_acc():.4f} | "
            f"|M|_F={self.ttt_engine.memory.M.norm().item():.4f}"
        )

    @property
    def coreset_size(self) -> int:
        return len(self.coreset)
