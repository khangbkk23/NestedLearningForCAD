"""
cadic_coreset.py
----------------
CADIC Incremental Coreset — Slow Memory của Meta-NATH CAD.

Backbone của toàn hệ thống:
    - Unified Memory Bank duy nhất cho toàn bộ tasks (không phân mảnh)
    - Max ~1000 entries, bounded VRAM
    - Incremental update theo CADIC Eq. 1-6
    - Anomaly scoring (image-level + pixel-level) theo CADIC Eq. 8-9
    - Lưu patch_embeddings để AnomalyDecoder tính pixel-level map

Ước tính VRAM:
    1000 entries × 256 patches × 768 dim × 4 bytes ≈ 786 MB
    -> An toàn trên RTX 3050 Ti (4GB) và Jetson Orin NX (8-16GB)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CADICCoreset:
    """
    Incremental Coreset theo CADIC.

    Unified Memory Bank: Một coreset duy nhất cho toàn bộ lịch sử task.
    Không phân mảnh theo task_id. Mỗi entry đại diện cho một vùng
    feature space, được cập nhật khi có điểm mới xa hơn.

    Storage per entry:
        cls_embedding:   [d]           ->   3 KB
        patch_embeddings: [N_patch, d] -> 786 KB  (N_patch=256, d=768)
        utility_score:   float         ->   ~0 KB
        image (optional): [C, H, W]    ->  ~2.4 MB (chỉ dùng cho N2B-NC)
    """

    def __init__(
        self,
        max_size: int = 1000,
        d: int = 768,
        n_patch: Optional[int] = None,
        store_images: bool = False,
        device: str | None = None,
    ):
        """
        Args:
            max_size:     Số entry tối đa (default 1000)
            d:            Chiều embedding backbone (768 cho DINOv2-base prototype)
            n_patch:      Số patch tokens per image (256 = 16×16 cho 224×224 input)
            store_images: Có lưu raw image tensor không (cần cho N2B-NC Phase 3)
            device:       'cuda' hoặc 'cpu'
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.max_size     = max_size
        self.d            = d
        self.n_patch      = n_patch
        self.patch_grid: Optional[Tuple[int, int]] = None
        self.store_images = store_images
        self.device       = device

        # --- Storage ---
        self.cls_embeddings:   List[torch.Tensor] = []   # List of [d]
        self.patch_embeddings: List[torch.Tensor] = []   # List of [N_patch, d]
        self.images:           List[torch.Tensor] = []   # List of [C, H, W]
        self.utilities:        List[float]        = []   # utility score per entry
        self.task_ids:         List[int]          = []   # task_id per entry (for logging)

        self._update_count = 0
        self._closest_pair_cache: Optional[Tuple[float, int]] = None
        self._patch_bank_cache: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cls_embeddings)

    @property
    def is_full(self) -> bool:
        return len(self) >= self.max_size

    # ------------------------------------------------------------------
    # Core: Incremental Coreset Update (CADIC)
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

        Nếu Coreset chưa đầy -> add trực tiếp.
        Nếu đầy -> áp dụng CADIC Eq. 6:
            Nếu d_max(x_new, C) > |C|_min -> thay thế điểm gần nhất
            trong cặp gần nhất nhau bằng x_new.

        Args:
            cls_emb:    [d]           - CLS embedding từ backbone
            patch_embs: [N_patch, d]  - patch tokens từ backbone
            image:      [C, H, W]     - raw image (optional, cho N2B-NC)
            task_id:    int           - task hiện tại (chỉ để logging)

        Returns:
            updated: bool - True nếu Coreset thực sự thay đổi
        """
        cls_emb    = cls_emb.detach().to(self.device)
        patch_embs = patch_embs.detach().to(self.device)
        self._ensure_patch_geometry(patch_embs)

        # --- Chưa đầy: add trực tiếp ---
        if not self.is_full:
            self._add_entry(cls_emb, patch_embs, image, task_id)
            logger.debug(f"[CADIC] Added entry. Size: {len(self)}/{self.max_size}")
            return True

        # --- Đầy: CADIC ---
        C = torch.stack(self.cls_embeddings)   # [M, d]
        x = cls_emb.unsqueeze(0)               # [1, d]

        # Eq. 1-2: d_max = khoảng cách từ x mới đến điểm gần nhất trong C
        dists_to_C = torch.cdist(x, C)         # [1, M]
        d_max      = dists_to_C.min().item()

        # Eq. 4-5: |C|_min = khoảng cách nhỏ nhất giữa 2 điểm trong C
        c_min_val, c_min_idx = self._get_closest_pair()

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
    # Anomaly Scoring (CADIC)
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
            s_img:   [B]            - image-level scores
            s_pix:   [B, N_patch]   - pixel-level scores
        """
        if len(self) == 0:
            raise RuntimeError("[CADIC] Coreset rỗng.")

        if patch_embs_batch.ndim == 2:
            patch_embs_batch = patch_embs_batch.unsqueeze(0)

        B, N, D = patch_embs_batch.shape
        patch_embs_batch = patch_embs_batch.to(self.device)
        if D != self.d:
            raise ValueError(f"[CADIC] Expected patch dim {self.d}, got {D}.")
        if self.n_patch is not None and N != self.n_patch:
            raise ValueError(f"[CADIC] Expected {self.n_patch} patches, got {N}.")

        C_patch = self.get_all_patch_embs()    # [M * N_patch, d]

        # --- 1. Pixel-level scores ---
        # Tính min distance theo chunk trên patch bank để tránh tensor [N, M*N_patch] quá lớn.
        s_pix_list = []
        nearest_patch_idx_list = []
        for i in range(B):
            s_pix_i, nearest_idx_i = self._min_patch_dists(
                patch_embs_batch[i],
                C_patch,
            )
            s_pix_list.append(s_pix_i)
            nearest_patch_idx_list.append(nearest_idx_i)
        
        s_pix = torch.stack(s_pix_list) # [B, N]
        nearest_patch_idxs = torch.stack(nearest_patch_idx_list) # [B, N]

        # --- 2. Image-level score (Eq. 8-9) ---
        # Tìm patch "tệ nhất" x* cho mỗi ảnh trong batch
        s_star_vals, s_star_idxs = s_pix.max(dim=1) # [B], [B]

        s_img_list = []
        b_actual = min(max(int(b), 1), C_patch.shape[0])
        for i in range(B):
            x_star = patch_embs_batch[i, s_star_idxs[i]]
            d_min = s_star_vals[i]
            top_k_dists = self._topk_patch_dists(x_star, C_patch, k=b_actual)

            # Stable PatchCore/CADIC-style weighting in the same patch space:
            # weight = 1 - exp(d_min) / sum(exp(top_k_dists))
            #        = 1 - 1 / sum(exp(top_k_dists - d_min))
            exp_sum = torch.exp(top_k_dists - d_min).sum().clamp_min(1e-12)
            weight = 1.0 - (1.0 / exp_sum)
            s_img_list.append(weight * d_min)

        s_img = torch.stack(s_img_list)

        return s_img.cpu(), s_pix.cpu()

    # ------------------------------------------------------------------
    # Utility Management (cho N2B-NC Phase 3)
    # ------------------------------------------------------------------

    def get_top_k_by_utility(
        self,
        k: int,
        balanced_by_task: bool = False,
        return_metadata: bool = False,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor] | Tuple[Optional[torch.Tensor], torch.Tensor, Dict[str, Any]]:
        """
        Trả về top-k entries có utility cao nhất.

        Returns:
            images: [k, C, H, W] nếu store_images=True, else None
            embs:   [k, d]
        """
        if len(self) == 0:
            raise RuntimeError("[CADIC] Coreset rỗng.")

        k = min(k, len(self))
        indices = self._select_anchor_indices(k, balanced_by_task=balanced_by_task)

        embs = torch.stack([self.cls_embeddings[i] for i in indices])   # [k, d]

        if self.store_images and self.images and self.images[0] is not None:
            imgs = torch.stack([self.images[i] for i in indices])        # [k, C, H, W]
        else:
            imgs = None

        if return_metadata:
            task_counts: Dict[int, int] = {}
            for i in indices:
                task_id = int(self.task_ids[i])
                task_counts[task_id] = task_counts.get(task_id, 0) + 1
            return imgs, embs, {
                "balanced_by_task": bool(balanced_by_task),
                "indices": indices,
                "task_counts": task_counts,
            }

        return imgs, embs

    def _select_anchor_indices(self, k: int, balanced_by_task: bool = False) -> List[int]:
        if not balanced_by_task:
            return sorted(
                range(len(self.utilities)),
                key=lambda i: self.utilities[i],
                reverse=True,
            )[:k]

        by_task: Dict[int, List[int]] = {}
        for idx, task_id in enumerate(self.task_ids):
            by_task.setdefault(int(task_id), []).append(idx)
        for indices in by_task.values():
            indices.sort(key=lambda i: self.utilities[i], reverse=True)

        selected: List[int] = []
        task_order = sorted(by_task)
        while len(selected) < k:
            added = False
            for task_id in task_order:
                candidates = by_task[task_id]
                if candidates:
                    selected.append(candidates.pop(0))
                    added = True
                    if len(selected) >= k:
                        break
            if not added:
                break
        return selected

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

    def replace_all_embeddings(
        self,
        cls_embeddings: List[torch.Tensor],
        patch_embeddings: List[torch.Tensor],
    ) -> None:
        """
        Re-index Coreset embeddings after backbone evolution.

        Phase 3 may update the backbone. When that happens, stored patch/CLS
        embeddings must be recomputed in the new feature space while keeping
        images, utilities, and task ownership unchanged.
        """
        if len(cls_embeddings) != len(self) or len(patch_embeddings) != len(self):
            raise ValueError(
                "[CADIC] Replacement embeddings must match current coreset size: "
                f"got cls={len(cls_embeddings)}, patch={len(patch_embeddings)}, size={len(self)}."
            )

        new_cls = []
        new_patches = []
        for cls_emb, patch_embs in zip(cls_embeddings, patch_embeddings):
            cls_emb = cls_emb.detach().to(self.device)
            patch_embs = patch_embs.detach().to(self.device)
            if cls_emb.ndim != 1 or cls_emb.shape[0] != self.d:
                raise ValueError(f"[CADIC] Expected cls embedding [{self.d}], got {tuple(cls_emb.shape)}.")
            self._ensure_patch_geometry(patch_embs)
            new_cls.append(cls_emb)
            new_patches.append(patch_embs)

        self.cls_embeddings = new_cls
        self.patch_embeddings = new_patches
        self._invalidate_caches()

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
        if self._patch_bank_cache is None:
            self._patch_bank_cache = torch.cat(self.patch_embeddings, dim=0)   # [M * N_patch, d]
        return self._patch_bank_cache

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
        n_patch = int(self.n_patch or 0)
        vram_mb = len(self) * n_patch * self.d * 4 / (1024 ** 2)

        return {
            "size":          len(self),
            "max_size":      self.max_size,
            "is_full":       self.is_full,
            "update_count":  self._update_count,
            "avg_utility":   round(avg_utility, 4),
            "task_counts":   task_counts,
            "vram_patch_mb": round(vram_mb, 1),
            "patch_grid":    self.patch_grid,
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

    def state_dict(self, include_images: bool = True) -> dict:
        return {
            "cls_embeddings":   [e.cpu() for e in self.cls_embeddings],
            "patch_embeddings": [e.cpu() for e in self.patch_embeddings],
            "images":           [e.cpu() if e is not None else None for e in self.images] if include_images else [],
            "utilities":        list(self.utilities),
            "task_ids":         list(self.task_ids),
            "max_size":         self.max_size,
            "d":                self.d,
            "n_patch":          self.n_patch,
            "patch_grid":       self.patch_grid,
            "_update_count":    self._update_count,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.cls_embeddings   = [e.to(self.device) for e in sd["cls_embeddings"]]
        self.patch_embeddings = [e.to(self.device) for e in sd["patch_embeddings"]]
        self.images = [
            e.to(self.device) if e is not None else None
            for e in sd.get("images", [])
        ]
        if not self.images:
            self.images = [None for _ in self.cls_embeddings]
        self.utilities     = list(sd["utilities"])
        self.task_ids      = list(sd["task_ids"])
        self.max_size      = sd["max_size"]
        self.d             = sd["d"]
        self.n_patch       = sd.get("n_patch")
        self.patch_grid    = tuple(sd["patch_grid"]) if sd.get("patch_grid") else None
        self._update_count = sd["_update_count"]
        self._closest_pair_cache = None
        self._patch_bank_cache = None

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
        self._invalidate_caches()

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
        self._invalidate_caches()

    def _ensure_patch_geometry(self, patch_embs: torch.Tensor) -> None:
        if patch_embs.ndim != 2:
            raise ValueError(f"[CADIC] patch_embs must be [N_patch, d], got {tuple(patch_embs.shape)}.")
        n_patch, d = patch_embs.shape
        if d != self.d:
            raise ValueError(f"[CADIC] Expected patch dim {self.d}, got {d}.")
        grid = math.isqrt(n_patch)
        if grid * grid != n_patch:
            raise ValueError(f"[CADIC] Patch count must be square for anomaly maps, got {n_patch}.")
        if self.n_patch is None:
            self.n_patch = n_patch
            self.patch_grid = (grid, grid)
        elif self.n_patch != n_patch:
            raise ValueError(f"[CADIC] Expected {self.n_patch} patches, got {n_patch}.")
        elif self.patch_grid is None:
            self.patch_grid = (grid, grid)

    def _invalidate_caches(self) -> None:
        self._closest_pair_cache = None
        self._patch_bank_cache = None

    def _get_closest_pair(self) -> Tuple[float, int]:
        if self._closest_pair_cache is not None:
            return self._closest_pair_cache

        C = torch.stack(self.cls_embeddings)
        C_dists = torch.cdist(C, C)
        C_dists.fill_diagonal_(float("inf"))
        c_min_val = C_dists.min().item()
        flat_idx = C_dists.argmin().item()
        c_min_idx = flat_idx // C.shape[0]
        self._closest_pair_cache = (c_min_val, c_min_idx)
        return self._closest_pair_cache

    def _min_patch_dists(
        self,
        query_patches: torch.Tensor,
        patch_bank: torch.Tensor,
        chunk_size: int = 8192,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        min_dists = torch.full(
            (query_patches.shape[0],),
            float("inf"),
            device=query_patches.device,
        )
        nearest_idxs = torch.zeros(
            query_patches.shape[0],
            dtype=torch.long,
            device=query_patches.device,
        )

        for start in range(0, patch_bank.shape[0], chunk_size):
            chunk = patch_bank[start:start + chunk_size]
            dists = torch.cdist(query_patches, chunk)
            vals, idxs = dists.min(dim=1)
            better = vals < min_dists
            min_dists[better] = vals[better]
            nearest_idxs[better] = start + idxs[better]

        return min_dists, nearest_idxs

    def _topk_patch_dists(
        self,
        query_patch: torch.Tensor,
        patch_bank: torch.Tensor,
        k: int,
        chunk_size: int = 8192,
    ) -> torch.Tensor:
        best = torch.empty(0, device=query_patch.device)
        query = query_patch.unsqueeze(0)

        for start in range(0, patch_bank.shape[0], chunk_size):
            chunk = patch_bank[start:start + chunk_size]
            vals = torch.cdist(query, chunk).squeeze(0)
            candidates = vals if best.numel() == 0 else torch.cat([best, vals], dim=0)
            best = torch.topk(candidates, k=min(k, candidates.numel()), largest=False).values

        return best
