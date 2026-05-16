"""
meta_nath_core.py
-----------------
MetaNATHCore - Lõi hệ thống Phase 1 + 2 của Meta-NATH CAD.

Kết nối:
    Nhà Thông Thái  = Frozen vision backbone (DINOv2 now, DINOv3-ready adapter)
    Sổ Nháp         = TTTEngine (TITANS Delta Rule)
    Cầu Chì         = ACCGating (Social Welfare)
    Tủ Hồ Sơ        = CADICCoreset (Unified Memory Bank)

Lưu ý thiết kế:
    - backbone LUÔN ở eval mode dù model.train() được gọi
    - TTTEngine / ACCGating / CADICCoreset không phải nn.Module
      -> state_dict() custom để checkpoint đầy đủ
    - forward() chỉ chạy Phase 1 (TTT) + Phase 2 (consolidation)
      Scoring riêng dùng score_image()
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
            d:                Chiều embedding (768 for DINOv2-base/current prototype)
            tau_acc:          Ngưỡng ACC Gating (instruction_CAD §5.8, default 0.25)
            max_coreset_size: Số entry tối đa CADIC (default 1000)
            n_patch:          Số patch tokens. None để infer từ backbone output.
            store_images:     Lưu raw image tensor cho N2B-NC Phase 3
            device:           'cuda' hoặc 'cpu' (auto-detect nếu None)
            backbone_name:    HuggingFace checkpoint của backbone chính
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
        # Đóng băng tuyệt đối — không bao giờ train backbone ở Phase 1-2
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # ----------------------------------------------------------------
        # 2. Sổ Nháp - TITANS Fast Memory
        # ----------------------------------------------------------------
        self.ttt_engine = TTTEngine(d=d, device=device)

        # ----------------------------------------------------------------
        # 3. Cầu Chì - ACC Gating
        # ----------------------------------------------------------------
        self.gating = ACCGating(tau=tau_acc)

        # ----------------------------------------------------------------
        # 4. Tủ Hồ Sơ - CADIC Incremental Coreset
        # ----------------------------------------------------------------
        self.coreset = CADICCoreset(
            max_size=max_coreset_size,
            d=d,
            n_patch=n_patch,
            store_images=store_images,
            device=device,
        )

    # ------------------------------------------------------------------
    # Override train() để backbone LUÔN ở eval mode
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "MetaNATHCore":
        """
        Override bắt buộc: backbone phải luôn ở eval mode
        dù Trainer gọi model.train() bất cứ lúc nào.
        """
        super().train(mode)
        self.backbone.eval()          # Force backbone về eval
        for p in self.backbone.parameters():
            p.requires_grad_(False)   # Double-check không có gradient nào lọt qua
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
        Pipeline Phase 1 + 2 cho một batch ảnh.

        Args:
            x:              [B, 3, 224, 224]
            task_id:        task hiện tại (cho logging CADIC)
            update_coreset: False khi inference thuần túy (không cập nhật Slow Memory)

        Returns:
            dict với keys:
                z_cls:            [B, d]   - CLS token gốc từ backbone
                z_updated:        [B, d]   - sau TTT adaptation
                z_patches:        [B, N, d] - patch tokens (cho AnomalyDecoder)
                surprise:         float
                acc_score:        float
                approved:         bool     - ACCGating có duyệt không
                coreset_updated:  bool     - Coreset có thực sự thay đổi không
        """
        # --- Bước 1: Feature Extraction (Nhà Thông Thái) ---
        z_cls, z_patches, patch_grid = self.extract_features(x)

        # --- Bước 2: Test-Time Adaptation (Sổ Nháp) ---
        z_updated, surprise = self.ttt_engine.process(z_cls)

        # --- Bước 3: Kiểm duyệt (Cầu Chì) ---
        approved, acc_score = self.gating.should_consolidate(z_updated, z_cls)

        # --- Bước 4: Consolidation vào Slow Memory ---
        coreset_updated = False
        if update_coreset and approved:
            # Lưu z_updated (embedding đã thích nghi) vào Coreset,
            # không phải z_cls gốc - vì z_updated là biểu diễn tốt hơn
            # sau khi Fast Memory đã "tiêu hóa" batch này.
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
        Backbone adapter trả về CLS token, patch tokens và patch grid thống nhất.

        Hiện tại hỗ trợ HuggingFace DINOv2 (`last_hidden_state`) và format dict
        của DINOv3 (`x_norm_clstoken`, `x_norm_patchtokens`) cho lần nâng cấp sau.
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
    # Inference — Anomaly Scoring (gọi riêng, không phải forward)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_image(
        self,
        x: torch.Tensor,
        b: int = 2,
    ) -> Dict[str, Any]:
        """
        Tính anomaly score cho ảnh test dùng CADIC Coreset.

        KHÔNG cập nhật TTT hay Coreset —-chỉ inference thuần túy.

        Args:
            x: [1, 3, 224, 224] hoặc [B, 3, 224, 224]
            b: neighborhood size cho Eq. 9 (default 2)

        Returns:
            dict với:
                s_img:       float hoặc List[float]  - image-level score
                s_pix:       [N_patch] tensor        - pixel-level scores
                anomaly_map: [H, W] tensor           - upsample về input size
        """
        if len(self.coreset) == 0:
            raise RuntimeError(
                "[MetaNATH] Coreset rỗng. Cần chạy ít nhất 1 forward() "
                "với update_coreset=True trước khi gọi score_image()."
            )

        _, z_patches, patch_grid = self.extract_features(x)

        # --- Batch Scoring (CADIC Eq. 8-9) ---
        # Gọi hàm Batch Scoring mới đã tối ưu
        s_img_batch, s_pix_batch = self.coreset.compute_anomaly_score(
            patch_embs_batch=z_patches,
            b=b,
        )

        # --- Batch Upsampling (Nội suy hàng loạt) ---
        B = z_patches.shape[0]
        H_patch, W_patch = patch_grid
        
        # [B, 1, 16, 16]
        s_pix_reshaped = s_pix_batch.reshape(B, 1, H_patch, W_patch).to(self.device_str)
        
        # Nội suy cả batch cùng lúc trên GPU
        anomaly_maps_batch = F.interpolate(
            s_pix_reshaped,
            size=(x.shape[-2], x.shape[-1]),
            mode="bilinear",
            align_corners=False,
        ) # [B, 1, H, W]

        # Đóng gói kết quả
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
    # Checkpoint (custom vì TTTEngine/ACCGating/CADICCoreset không phải nn.Module)
    # ------------------------------------------------------------------

    def full_state_dict(self, include_backbone: bool = True, include_images: bool = True) -> dict:
        """
        Lưu đầy đủ: backbone weights + TITANS memory + gating history + coreset.

        Dùng thay cho model.state_dict() khi checkpoint.
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
        """Load từ full_state_dict()."""
        if sd.get("backbone") is not None:
            self.backbone.load_state_dict(sd["backbone"])
        self.backbone_name = sd.get("backbone_name", self.backbone_name)
        self.ttt_engine.load_state_dict(sd["ttt_engine"])
        self.gating.load_state_dict(sd["gating"])
        self.coreset.load_state_dict(sd["coreset"])
        patch_grid = sd.get("config", {}).get("patch_grid")
        self.patch_grid = tuple(patch_grid) if patch_grid else self.coreset.patch_grid
        # Đảm bảo backbone vẫn frozen sau khi load
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
        """In tóm tắt trạng thái hệ thống."""
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
