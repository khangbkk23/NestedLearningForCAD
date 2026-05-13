"""
cadic_coreset.py
----------------
CADIC Incremental Coreset — Slow Memory (Tủ Hồ Sơ) của Meta-NATH CAD.

Là xương sống của toàn hệ thống:
    - Unified Memory Bank duy nhất cho toàn bộ task (không phân mảnh)
    - Max ~1000 entries, bounded VRAM
    - Incremental update theo CADIC Eq. 1-6
    - Anomaly scoring (image-level + pixel-level) theo CADIC Eq. 8-9
    - Lưu patch_embeddings để AnomalyDecoder tính pixel-level map

Tham chiếu:
    CADIC (arXiv:2511.08634) — Gen Yang et al.
    instruction_CAD.md §3.3, §2.1

Ước tính VRAM:
    1000 entries × 256 patches × 768 dim × 4 bytes ≈ 786 MB
    → An toàn trên RTX 3050 Ti (4GB) và Jetson Orin NX (8-16GB)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CADICCoreset:
    """
    Incremental Coreset theo triết lý CADIC.

    Unified Memory Bank: một coreset duy nhất cho toàn bộ lịch sử task.
    Không phân mảnh theo task_id. Mỗi entry đại diện cho một vùng
    feature space, được cập nhật khi có điểm mới "xa hơn".

    Storage per entry:
        cls_embedding:   [d]           →   3 KB
        patch_embeddings: [N_patch, d] → 786 KB  (N_patch=256, d=768)
        utility_score:   float         →   ~0 KB
        image (optional): [C, H, W]   →  ~2.4 MB (chỉ dùng cho N2B-NC)
    """

    def __init__(
        self,
        max_size: int = 1000,
        d: int = 768,
        n_patch: int = 256,
        store_images: bool = True,
        device: str | None = None,
    ):
        """
        Args:
            max_size:     Số entry tối đa (default 1000 theo instruction_CAD §5.13)
            d:            Chiều embedding (768 cho DINOv3-B/14, 1024 cho DINOv3-L/14)
            n_patch:      Số patch tokens per image (256 = 16×16 cho 224×224 input)
            store_images: Có lưu raw image tensor không (cần cho N2B-NC Phase 3)
            device:       'cuda' hoặc 'cpu'
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.max_size     = max_size
        self.d            = d
        self.n_patch      = n_patch
        self.store_images = store_images
        self.device       = device

        # --- Storage ---
        self.cls_embeddings:   List[torch.Tensor] = []   # List of [d]
        self.patch_embeddings: List[torch.Tensor] = []   # List of [N_patch, d]
        self.images:           List[torch.Tensor] = []   # List of [C, H, W]
        self.utilities:        List[float]        = []   # utility score per entry
        self.task_ids:         List[int]          = []   # task_id per entry (for logging)

        self._update_count = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cls_embeddings)

    @property
    def is_full(self) -> bool:
        return len(self) >= self.max_size

    # ------------------------------------------------------------------
    # Core: Incremental Coreset Update (CADIC Eq. 1-6)
    # ------------------------------------------------------------------

    def update(
        self,
        cls_emb: torch.Tensor,
        patch_embs: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        task_id: int = 0,
    ) -> bool:
        """
        Cập nhật Coreset với một embedding mới.

        Nếu Coreset chưa đầy → add trực tiếp.
        Nếu đầy → áp dụng CADIC Eq. 6:
            Nếu d_max(x_new, C) > |C|_min → thay thế điểm gần nhất
            trong cặp gần nhất nhau bằng x_new.

        Args:
            cls_emb:    [d]           — CLS embedding từ DINOv3
            patch_embs: [N_patch, d]  — patch tokens từ DINOv3
            image:      [C, H, W]     — raw image (optional, cho N2B-NC)
            task_id:    int           — task hiện tại (chỉ để logging)

        Returns:
            updated: bool — True nếu Coreset thực sự thay đổi
        """
        cls_emb    = cls_emb.detach().to(self.device)
        patch_embs = patch_embs.detach().to(self.device)

        # --- Chưa đầy: add trực tiếp ---
        if not self.is_full:
            self._add_entry(cls_emb, patch_embs, image, task_id)
            logger.debug(f"[CADIC] Added entry. Size: {len(self)}/{self.max_size}")
            return True

        # --- Đầy: CADIC Eq. 1-6 ---
        C = torch.stack(self.cls_embeddings)   # [M, d]
        x = cls_emb.unsqueeze(0)               # [1, d]

        # Eq. 1-2: d_max = khoảng cách từ x mới đến điểm gần nhất trong C
        dists_to_C = torch.cdist(x, C)         # [1, M]
        d_max      = dists_to_C.min().item()

        # Eq. 4-5: |C|_min = khoảng cách nhỏ nhất giữa 2 điểm trong C
        C_dists = torch.cdist(C, C)            # [M, M]
        C_dists.fill_diagonal_(float("inf"))
        c_min_val = C_dists.min().item()
        # Lấy index của điểm trong cặp gần nhất (sẽ bị thay thế)
        flat_idx  = C_dists.argmin().item()
        c_min_idx = flat_idx // C.shape[0]     # row index

        # Eq. 6: điều kiện thay thế
        if d_max > c_min_val:
            self._replace_entry(c_min_idx, cls_emb, patch_embs, image, task_id)
            self._update_count += 1
            logger.debug(
                f"[CADIC] Replaced entry[{c_min_idx}]. "
                f"d_max={d_max:.4f} > c_min={c_min_val:.4f}. "
                f"Total updates: {self._update_count}"
            )
            return True

        logger.debug(
            f"[CADIC] No update. d_max={d_max:.4f} ≤ c_min={c_min_val:.4f}"
        )
        return False

    def update_batch(
        self,
        cls_embs: torch.Tensor,
        patch_embs_batch: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        task_id: int = 0,
    ) -> int:
        """
        Cập nhật Coreset với một batch embeddings.

        Args:
            cls_embs:         [B, d]
            patch_embs_batch: [B, N_patch, d]
            images:           [B, C, H, W] (optional)
            task_id:          int

        Returns:
            n_updated: số lần Coreset thực sự thay đổi
        """
        B = cls_embs.shape[0]
        n_updated = 0
        for i in range(B):
            img = images[i] if images is not None else None
            updated = self.update(
                cls_emb=cls_embs[i],
                patch_embs=patch_embs_batch[i],
                image=img,
                task_id=task_id,
            )
            if updated:
                n_updated += 1
        return n_updated

    # ------------------------------------------------------------------
    # Anomaly Scoring (CADIC Eq. 8-9)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_anomaly_score(
        self,
        patch_embs_batch: torch.Tensor,
        b: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tính anomaly score cho một BATCH ảnh test (Đã tối ưu Batch Processing).
        
        Args:
            patch_embs_batch: [B, N_patch, d]
            b:                số lân cận gần nhất cho Neighborhood Softmax
            
        Returns:
            s_img:   [B]            — image-level scores
            s_pix:   [B, N_patch]   — pixel-level scores
        """
        if len(self) == 0:
            raise RuntimeError("[CADIC] Coreset rỗng.")

        B, N, D = patch_embs_batch.shape
        patch_embs_batch = patch_embs_batch.to(self.device)
        C_patch = self.get_all_patch_embs()    # [M * N_patch, d]

        # --- 1. Pixel-level scores (Eq. 211) ---
        # Tính khoảng cách cho toàn bộ batch: [B, N, M*N_patch]
        # Để tiết kiệm VRAM trên RTX 3050 Ti, tính s_pix qua từng ảnh trong batch nhưng tối ưu K-NN
        s_pix_list = []
        for i in range(B):
            # torch.cdist [N, M*N_patch]
            dists_i = torch.cdist(patch_embs_batch[i], C_patch)
            s_pix_i = dists_i.min(dim=1).values # [N]
            s_pix_list.append(s_pix_i)
        
        s_pix = torch.stack(s_pix_list) # [B, N]

        # --- 2. Image-level score (Eq. 8-9) ---
        # Tìm patch "tệ nhất" x* cho mỗi ảnh trong batch
        s_star_vals, s_star_idxs = s_pix.max(dim=1) # [B], [B]
        
        # Lấy feature của x* : [B, d]
        x_stars = patch_embs_batch[torch.arange(B), s_star_idxs] 

        # Khoảng cách từ x* đến CLS embeddings trong Coreset
        C_cls = torch.stack(self.cls_embeddings) # [M, d]
        # nb_dists: [B, M]
        nb_dists = torch.cdist(x_stars, C_cls)

        # Lấy b lân cận gần nhất
        b_actual = min(b, len(self))
        top_k_dists, _ = torch.topk(nb_dists, k=b_actual, dim=1, largest=False)
        c_star_dists = nb_dists.min(dim=1).values # [B]

        # Numerical Stability for Softmax (subtract min)
        # weight = 1 - exp(d_min) / sum(exp(d_topk))
        #        = 1 - 1 / sum(exp(d_topk - d_min))
        diffs = top_k_dists - c_star_dists.unsqueeze(1)
        exp_sum = torch.exp(-diffs).sum(dim=1) # Dùng -diffs vì d_topk >= d_min
        
        # Eq. 9: weight = 1 - 1 / Σ exp(d_min - d_i)
        weight = 1.0 - (1.0 / exp_sum)
        s_img = weight * s_star_vals

        return s_img.cpu(), s_pix.cpu()

    # ------------------------------------------------------------------
    # Utility Management (cho N2B-NC Phase 3)
    # ------------------------------------------------------------------

    def get_top_k_by_utility(self, k: int) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Trả về top-k entries có utility cao nhất.

        Returns:
            images: [k, C, H, W] nếu store_images=True, else None
            embs:   [k, d]
        """
        if len(self) == 0:
            raise RuntimeError("[CADIC] Coreset rỗng.")

        k       = min(k, len(self))
        indices = sorted(
            range(len(self.utilities)),
            key=lambda i: self.utilities[i],
            reverse=True,
        )[:k]

        embs = torch.stack([self.cls_embeddings[i] for i in indices])   # [k, d]

        if self.store_images and self.images and self.images[0] is not None:
            imgs = torch.stack([self.images[i] for i in indices])        # [k, C, H, W]
        else:
            imgs = None

        return imgs, embs

    def update_utility(self, idx: int, new_score: float) -> None:
        """Cập nhật utility score của entry tại index idx."""
        if 0 <= idx < len(self.utilities):
            self.utilities[idx] = float(new_score)

    def decay_all_utilities(self, decay: float = 0.99) -> None:
        """
        Giảm nhẹ utility của toàn bộ entries theo thời gian.
        Entries cũ sẽ dần có utility thấp hơn, dễ bị thay thế hơn.
        """
        self.utilities = [u * decay for u in self.utilities]

    # ------------------------------------------------------------------
    # Patch Embeddings Access (cho AnomalyDecoder)
    # ------------------------------------------------------------------

    def get_all_patch_embs(self) -> torch.Tensor:
        """
        Trả về toàn bộ patch embeddings dưới dạng tensor 2D.

        Returns:
            [M * N_patch, d] — flatten tất cả patches của tất cả entries.
            BẮT BUỘC dùng hàm này thay vì truy cập self.patch_embeddings trực tiếp.

        Raises:
            RuntimeError: nếu Coreset rỗng
        """
        if not self.patch_embeddings:
            raise RuntimeError("[CADIC] Không có patch embeddings nào trong Coreset.")
        return torch.cat(self.patch_embeddings, dim=0)   # [M * N_patch, d]

    # ------------------------------------------------------------------
    # Stats & Debug
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Trả về dict thống kê để logging/monitoring."""
        if len(self) == 0:
            return {"size": 0, "is_full": False}

        task_counts: dict[int, int] = {}
        for tid in self.task_ids:
            task_counts[tid] = task_counts.get(tid, 0) + 1

        avg_utility = sum(self.utilities) / len(self.utilities)
        vram_mb     = len(self) * self.n_patch * self.d * 4 / (1024 ** 2)

        return {
            "size":          len(self),
            "max_size":      self.max_size,
            "is_full":       self.is_full,
            "update_count":  self._update_count,
            "avg_utility":   round(avg_utility, 4),
            "task_counts":   task_counts,
            "vram_patch_mb": round(vram_mb, 1),
        }

    def log_stats(self) -> None:
        s = self.stats()
        logger.info(
            f"[CADIC] size={s['size']}/{s['max_size']} | "
            f"updates={s['update_count']} | "
            f"avg_utility={s['avg_utility']} | "
            f"VRAM(patch)≈{s['vram_patch_mb']}MB | "
            f"tasks={s['task_counts']}"
        )

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "cls_embeddings":   [e.cpu() for e in self.cls_embeddings],
            "patch_embeddings": [e.cpu() for e in self.patch_embeddings],
            "images":           [e.cpu() if e is not None else None for e in self.images],
            "utilities":        list(self.utilities),
            "task_ids":         list(self.task_ids),
            "max_size":         self.max_size,
            "d":                self.d,
            "n_patch":          self.n_patch,
            "_update_count":    self._update_count,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.cls_embeddings   = [e.to(self.device) for e in sd["cls_embeddings"]]
        self.patch_embeddings = [e.to(self.device) for e in sd["patch_embeddings"]]
        self.images           = [
            e.to(self.device) if e is not None else None
            for e in sd["images"]
        ]
        self.utilities     = list(sd["utilities"])
        self.task_ids      = list(sd["task_ids"])
        self.max_size      = sd["max_size"]
        self.d             = sd["d"]
        self.n_patch       = sd["n_patch"]
        self._update_count = sd["_update_count"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_entry(
        self,
        cls_emb: torch.Tensor,
        patch_embs: torch.Tensor,
        image: Optional[torch.Tensor],
        task_id: int,
    ) -> None:
        self.cls_embeddings.append(cls_emb)
        self.patch_embeddings.append(patch_embs)
        self.images.append(image.cpu() if image is not None else None)
        self.utilities.append(1.0)
        self.task_ids.append(task_id)

    def _replace_entry(
        self,
        idx: int,
        cls_emb: torch.Tensor,
        patch_embs: torch.Tensor,
        image: Optional[torch.Tensor],
        task_id: int,
    ) -> None:
        self.cls_embeddings[idx]   = cls_emb
        self.patch_embeddings[idx] = patch_embs
        self.images[idx]           = image.cpu() if image is not None else None
        self.utilities[idx]        = 1.0
        self.task_ids[idx]         = task_id
