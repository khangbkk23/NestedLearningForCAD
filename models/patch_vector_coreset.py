"""Patch-vector CADIC memory bank for the reproduction track."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from .cadic_coreset import CADICCoreset
from .distance import pairwise_distance, validate_distance_metric


class PatchVectorCADICCoreset(CADICCoreset):
    """A fixed-capacity bank whose unit is one patch feature vector.

    The existing CADICCoreset remains available as the Phase 1 image-entry
    format so its checkpoints and reported runs keep their original semantics.
    This class implements CADIC's batch update on individual patch vectors.
    """

    mode = "patch_vectors"

    def __init__(
        self,
        max_size: int = 10_000,
        d: int = 768,
        n_patch: Optional[int] = None,
        store_images: bool = False,
        device: str | None = None,
        distance_metric: str = "euclidean",
        distance_chunk_size: int = 512,
    ):
        if store_images:
            raise ValueError(
                "Patch-vector CADIC mode does not store replay images. Keep image replay in a separate budget."
            )
        super().__init__(
            max_size=max_size,
            d=d,
            n_patch=n_patch,
            store_images=False,
            device=device,
            distance_metric=distance_metric,
        )
        if self.max_size < 2:
            raise ValueError("Patch-vector coreset capacity must be at least 2.")
        self.distance_metric = validate_distance_metric(distance_metric)
        self.distance_chunk_size = max(1, int(distance_chunk_size))
        self.patch_vectors = torch.empty((0, self.d), dtype=torch.float32, device=self.device)
        self._closest_pair_cache: Optional[Tuple[float, int, int]] = None

    def __len__(self) -> int:
        return int(self.patch_vectors.shape[0])

    def update(
        self,
        cls_emb: torch.Tensor,
        patch_embs: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        task_id: int = 0,
    ) -> bool:
        del cls_emb
        if image is not None:
            raise ValueError("Patch-vector coreset accepts patch features only; images use a separate memory budget.")
        if patch_embs.ndim != 2:
            raise ValueError(f"patch_embs must be [N,D], got {tuple(patch_embs.shape)}.")
        if self.n_patch is None:
            n = int(patch_embs.shape[0])
            grid = math.isqrt(n)
            if grid * grid != n:
                raise ValueError(f"Patch count must form a square grid, got {n}.")
            self.n_patch = n
            self.patch_grid = (grid, grid)
        self._ensure_patch_geometry(patch_embs)
        return self.update_patch_vectors(patch_embs, task_id=task_id) > 0

    def update_batch(
        self,
        cls_embs: Optional[torch.Tensor],
        patch_embs_batch: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        task_id: int = 0,
    ) -> int:
        del cls_embs
        if images is not None:
            raise ValueError("Patch-vector coreset does not retain raw images.")
        if patch_embs_batch.ndim == 3:
            if patch_embs_batch.shape[0] == 0:
                return 0
            if self.n_patch is None:
                n = int(patch_embs_batch.shape[1])
                grid = math.isqrt(n)
                if grid * grid != n:
                    raise ValueError(f"Patch count must form a square grid, got {n}.")
                self.n_patch = n
                self.patch_grid = (grid, grid)
            self._ensure_patch_geometry(patch_embs_batch[0])
            candidates = patch_embs_batch.reshape(-1, patch_embs_batch.shape[-1])
        elif patch_embs_batch.ndim == 2:
            candidates = patch_embs_batch
        else:
            raise ValueError(
                f"patch_embs_batch must be [B,N,D] or [N,D], got {tuple(patch_embs_batch.shape)}."
            )
        return self.update_patch_vectors(candidates, task_id=task_id)

    @torch.no_grad()
    def update_patch_vectors(self, candidates: torch.Tensor, task_id: int = 0) -> int:
        if candidates.ndim != 2 or candidates.shape[-1] != self.d:
            raise ValueError(f"Expected candidate patch matrix [N,{self.d}], got {tuple(candidates.shape)}.")
        candidates = candidates.detach().to(device=self.device, dtype=torch.float32)
        if candidates.shape[0] == 0:
            return 0

        updated = 0
        if len(self) < self.max_size:
            count = min(self.max_size - len(self), candidates.shape[0])
            self.patch_vectors = torch.cat((self.patch_vectors, candidates[:count]), dim=0)
            self.task_ids.extend([int(task_id)] * count)
            self.utilities.extend([1.0] * count)
            candidates = candidates[count:]
            updated += count
            self._closest_pair_cache = None

        if candidates.shape[0] == 0 or len(self) < 2:
            return updated

        # Cache each incoming vector's exact nearest-bank distance. After a
        # replacement, the bank loses one vector and gains one vector. Most
        # candidates keep their nearest vector, so only distances to the new
        # vector need updating; candidates whose nearest vector was removed are
        # recomputed against the bank. This preserves the greedy CADIC update
        # while avoiding a full candidate-by-bank scan on every replacement.
        nearest_distances, nearest_indices = self._nearest_bank_vectors(candidates)
        active = torch.ones(candidates.shape[0], dtype=torch.bool, device=self.device)

        while bool(active.any()):
            candidate_distances = nearest_distances.masked_fill(~active, float("-inf"))
            dmax, candidate_index_tensor = candidate_distances.max(dim=0)
            candidate_index = int(candidate_index_tensor.item())
            cmin, replace_index, _ = self._get_closest_pair()
            if not bool(dmax > cmin):
                break

            # The selected candidate joins the bank and stops competing.
            active[candidate_index] = False
            removed_nearest = active & (nearest_indices == replace_index)
            self.patch_vectors[replace_index].copy_(candidates[candidate_index])
            self.task_ids[replace_index] = int(task_id)
            self.utilities[replace_index] = 1.0

            # Update against the inserted vector; recompute candidates whose
            # previous nearest vector was removed so their distances stay exact.
            new_vector_distances = pairwise_distance(
                candidates, self.patch_vectors[replace_index:replace_index + 1], self.distance_metric
            ).squeeze(1)
            improved = active & (new_vector_distances < nearest_distances)
            nearest_distances[improved] = new_vector_distances[improved]
            nearest_indices[improved] = replace_index
            if bool(removed_nearest.any()):
                refreshed_distances, refreshed_indices = self._nearest_bank_vectors(candidates[removed_nearest])
                nearest_distances[removed_nearest] = refreshed_distances
                nearest_indices[removed_nearest] = refreshed_indices

            self._update_count += 1
            updated += 1
            # The chosen replacement is one endpoint of the cached closest
            # pair, so that cached minimum is no longer valid.
            self._closest_pair_cache = None

        return updated

    @torch.no_grad()
    def _nearest_bank_vectors(self, candidates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return exact nearest-bank distances and indices for candidate rows."""
        bank = self.patch_vectors
        chunk = self.distance_chunk_size
        all_distances = torch.empty(candidates.shape[0], dtype=torch.float32, device=self.device)
        all_indices = torch.empty(candidates.shape[0], dtype=torch.long, device=self.device)
        for start in range(0, candidates.shape[0], chunk):
            candidate_block = candidates[start:start + chunk]
            min_distances = torch.full((candidate_block.shape[0],), float("inf"), device=self.device)
            min_indices = torch.zeros(candidate_block.shape[0], dtype=torch.long, device=self.device)
            for bank_start in range(0, bank.shape[0], chunk):
                bank_block = bank[bank_start:bank_start + chunk]
                distances = pairwise_distance(candidate_block, bank_block, self.distance_metric)
                values, indices = distances.min(dim=1)
                better = values < min_distances
                min_distances[better] = values[better]
                min_indices[better] = bank_start + indices[better]
            block_end = start + candidate_block.shape[0]
            all_distances[start:block_end] = min_distances
            all_indices[start:block_end] = min_indices
        return all_distances, all_indices

    @torch.no_grad()
    def _get_closest_pair(self) -> Tuple[float, int, int]:
        if self._closest_pair_cache is not None:
            return self._closest_pair_cache
        bank = self.patch_vectors
        chunk = self.distance_chunk_size
        size = bank.shape[0]
        block_min_distances = []
        block_left_indices = []
        block_right_indices = []
        for row_start in range(0, size, chunk):
            row_end = min(row_start + chunk, size)
            rows = bank[row_start:row_end]
            for col_start in range(row_start, size, chunk):
                col_end = min(col_start + chunk, size)
                cols = bank[col_start:col_end]
                distances = pairwise_distance(rows, cols, self.distance_metric)
                if row_start == col_start:
                    upper_triangle = torch.ones_like(distances, dtype=torch.bool).triu_(diagonal=1)
                    distances = distances.masked_fill(~upper_triangle, float("inf"))
                local_value, local_index = distances.reshape(-1).min(dim=0)
                block_min_distances.append(local_value)
                block_left_indices.append(row_start + local_index // distances.shape[1])
                block_right_indices.append(col_start + local_index % distances.shape[1])

        distances = torch.stack(block_min_distances)
        best_block = int(distances.argmin().item())
        best_distance = float(distances[best_block].item())
        best_pair = (
            int(block_left_indices[best_block].item()),
            int(block_right_indices[best_block].item()),
        )
        if not math.isfinite(best_distance):
            raise RuntimeError("Could not find a finite closest pair in the patch coreset.")
        self._closest_pair_cache = (best_distance, best_pair[0], best_pair[1])
        return self._closest_pair_cache

    def get_all_patch_embs(self) -> torch.Tensor:
        if len(self) == 0:
            raise RuntimeError("[CADIC] Patch-vector coreset is empty.")
        return self.patch_vectors

    def get_top_k_by_utility(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "Patch-vector CADIC has no image-anchor API. Phase 3 must define a separate image/anchor memory first."
        )

    def replace_all_embeddings(self, *args, **kwargs) -> None:
        del args, kwargs
        raise RuntimeError(
            "Backbone refresh is not supported for patch-vector-only memory without stored source images."
        )

    def stats(self) -> dict:
        if len(self) == 0:
            return {"size": 0, "max_size": self.max_size, "capacity_unit": "patch_vectors", "is_full": False}
        task_counts: dict[int, int] = {}
        for task_id in self.task_ids:
            task_counts[int(task_id)] = task_counts.get(int(task_id), 0) + 1
        storage_bytes = self.patch_vectors.numel() * self.patch_vectors.element_size()
        return {
            "size": len(self),
            "max_size": self.max_size,
            "capacity_unit": "patch_vectors",
            "is_full": self.is_full,
            "update_count": self._update_count,
            "avg_utility": sum(self.utilities) / len(self.utilities),
            "task_counts": task_counts,
            "patch_bank_bytes": int(storage_bytes),
            "patch_grid": self.patch_grid,
        }

    def state_dict(self, include_images: bool = True) -> dict:
        del include_images
        return {
            "mode": self.mode,
            "patch_vectors": self.patch_vectors.detach().cpu(),
            "utilities": list(self.utilities),
            "task_ids": list(self.task_ids),
            "max_size": self.max_size,
            "d": self.d,
            "n_patch": self.n_patch,
            "patch_grid": self.patch_grid,
            "distance_metric": self.distance_metric,
            "distance_chunk_size": self.distance_chunk_size,
            "_update_count": self._update_count,
        }

    def load_state_dict(self, sd: dict) -> None:
        if sd.get("mode") != self.mode:
            raise ValueError(
                f"Checkpoint coreset mode {sd.get('mode')!r} does not match configured mode {self.mode!r}."
            )
        if int(sd.get("d", self.d)) != self.d:
            raise ValueError(f"Checkpoint feature dimension {sd.get('d')} does not match {self.d}.")
        if int(sd.get("max_size", self.max_size)) != self.max_size:
            raise ValueError(
                f"Checkpoint patch capacity {sd.get('max_size')} does not match configured {self.max_size}."
            )
        saved_metric = validate_distance_metric(sd.get("distance_metric", self.distance_metric))
        if saved_metric != self.distance_metric:
            raise ValueError(
                f"Checkpoint distance metric {saved_metric!r} does not match configured {self.distance_metric!r}."
            )
        saved_n_patch = sd.get("n_patch")
        if self.n_patch is not None and saved_n_patch is not None and int(saved_n_patch) != self.n_patch:
            raise ValueError(
                f"Checkpoint patch count {saved_n_patch} does not match configured {self.n_patch}."
            )
        vectors = sd["patch_vectors"].to(device=self.device, dtype=torch.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.d or vectors.shape[0] > self.max_size:
            raise ValueError(f"Invalid patch-vector bank shape in checkpoint: {tuple(vectors.shape)}.")
        self.patch_vectors = vectors.contiguous()
        self.utilities = list(sd.get("utilities", [1.0] * len(self)))
        self.task_ids = list(sd.get("task_ids", [0] * len(self)))
        if len(self.utilities) != len(self) or len(self.task_ids) != len(self):
            raise ValueError("Checkpoint patch metadata length does not match patch-vector bank.")
        self.n_patch = sd.get("n_patch", self.n_patch)
        self.patch_grid = tuple(sd["patch_grid"]) if sd.get("patch_grid") else None
        self.distance_metric = saved_metric
        self._update_count = int(sd.get("_update_count", 0))
        self._closest_pair_cache = None

    def _min_patch_dists(self, *args, **kwargs):
        # Inherited scorer accesses the same patch-vector bank and distance mode.
        return super()._min_patch_dists(*args, **kwargs)
