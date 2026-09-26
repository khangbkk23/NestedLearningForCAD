"""Patch-vector CADIC coreset, isolated from the experimental Meta-NATH path.

The implementation follows equations (1)--(7) of the local CADIC paper:
incoming patch features are processed as a batch, the farthest incoming
feature is selected, and the closest existing pair is reduced by replacing
one member when the CADIC inequality holds.  The budget is the number of
stored patch vectors, never the number of images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class CADICPatchCoresetConfig:
    budget: int = 10_000
    dim: int = 768
    dtype: str = "float32"
    distance: str = "euclidean"
    chunk_size: int = 2048
    image_neighbors: int = 9


class CADICPatchCoresetV1:
    """A bounded, deterministic, Euclidean patch-feature memory."""

    schema = "cadic_patch_coreset_v1"

    def __init__(
        self,
        config: CADICPatchCoresetConfig | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config or CADICPatchCoresetConfig()
        if self.config.budget < 1:
            raise ValueError("budget must be positive")
        if self.config.dim < 1:
            raise ValueError("dim must be positive")
        if self.config.distance.lower() != "euclidean":
            raise ValueError("CADIC v1 is defined with Euclidean distance")
        if self.config.dtype != "float32":
            raise ValueError("CADIC v1 currently requires float32 feature storage")
        self.device = torch.device(device)
        self.features = torch.empty(
            (0, self.config.dim), dtype=torch.float32, device=self.device
        )
        self.seen_features = 0
        self.accepted_features = 0
        self.replaced_features = 0
        self.rejected_features = 0
        self.update_batches = 0

    @property
    def count(self) -> int:
        return int(self.features.shape[0])

    def __len__(self) -> int:
        return self.count

    @property
    def is_full(self) -> bool:
        return self.count >= self.config.budget

    @property
    def feature_bytes(self) -> int:
        return int(self.features.numel() * self.features.element_size())

    @property
    def memory_bytes(self) -> int:
        """Persistent feature bytes, excluding Python/object/checkpoint overhead."""
        return self.feature_bytes

    def update(self, patch_features: torch.Tensor) -> Dict[str, int]:
        """Update from `[N, D]` or `[B, N, D]` patch features.

        Rows are consumed in input order. Once full, the global farthest row
        is selected repeatedly exactly as in CADIC's batch rule. This makes
        the result deterministic for a fixed feature tensor and seed.
        """
        x = patch_features.detach().to(self.device, dtype=torch.float32)
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        if x.ndim != 2 or x.shape[1] != self.config.dim:
            raise ValueError(
                f"expected [N,{self.config.dim}] or [B,N,{self.config.dim}], got {tuple(patch_features.shape)}"
            )
        if x.shape[0] == 0:
            return {"incoming": 0, "accepted": 0, "replaced": 0, "rejected": 0}

        self.update_batches += 1
        self.seen_features += int(x.shape[0])
        accepted = 0
        replaced = 0
        rejected = 0

        # Initialization/fill is deterministic. The paper does not document
        # a special initializer, so this behavior is recorded in metadata.
        if not self.is_full:
            room = self.config.budget - self.count
            take = min(room, x.shape[0])
            if take:
                self.features = torch.cat((self.features, x[:take]), dim=0)
                accepted += take
            x = x[take:]

        # Eq. (1)-(6): process the remaining batch as a mutable candidate set.
        while x.numel() and self.is_full:
            nearest, _ = self._nearest(x, self.features)
            farthest_value, farthest_row = torch.max(nearest, dim=0)
            closest_value, closest_row = self._closest_pair()
            if not bool(farthest_value > closest_value):
                rejected += int(x.shape[0])
                break
            self.features[closest_row] = x[farthest_row]
            replaced += 1
            # Remove only the selected row; the rest of X remains available
            # for the required repeated batch procedure.
            keep = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            keep[farthest_row] = False
            x = x[keep]

        self.accepted_features += accepted
        self.replaced_features += replaced
        self.rejected_features += rejected
        return {
            "incoming": int(patch_features.reshape(-1, patch_features.shape[-1]).shape[0])
            if patch_features.ndim >= 2 else 0,
            "accepted": accepted,
            "replaced": replaced,
            "rejected": rejected,
        }

    @torch.no_grad()
    def pixel_scores(
        self,
        query_patches: torch.Tensor,
        *,
        b: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pixel scores, nearest indices, and image scores.

        Equation (9) is implemented literally: ``N_b(c*)`` consists of the
        memory vectors nearest to the matched memory vector ``c*``. The paper
        does not provide a numeric ``b``; callers must declare it explicitly.
        """
        if self.count == 0:
            raise RuntimeError("cannot score with an empty CADIC coreset")
        q = query_patches.detach().to(self.device, dtype=torch.float32)
        if q.ndim == 2:
            q = q.unsqueeze(0)
        if q.ndim != 3 or q.shape[-1] != self.config.dim:
            raise ValueError(f"expected [B,N,{self.config.dim}], got {tuple(query_patches.shape)}")
        k = int(self.config.image_neighbors if b is None else b)
        if k < 1:
            raise ValueError("b must be positive")
        k = min(k, self.count)

        all_pixel = []
        all_indices = []
        all_image = []
        for image in q:
            pixel, indices = self._nearest(image, self.features)
            star_row = int(torch.argmax(pixel).item())
            star_score = pixel[star_row]
            c_star_index = int(indices[star_row].item())
            c_star = self.features[c_star_index]
            c_dist, _ = self._nearest(c_star.unsqueeze(0), self.features)
            # `_nearest` returns only the minimum. Obtain the support indices
            # with a direct chunked distance pass to c*.
            support_indices = self._topk_indices(c_star, k)
            support = self.features[support_indices]
            support_dist = torch.linalg.vector_norm(support - image[star_row], dim=1)
            weight = 1.0 - torch.exp(star_score) / torch.exp(support_dist).sum().clamp_min(1e-12)
            all_pixel.append(pixel)
            all_indices.append(indices)
            all_image.append(weight * star_score)

        return torch.stack(all_pixel), torch.stack(all_indices), torch.stack(all_image)

    def score(self, query_patches: torch.Tensor, *, b: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        pixel, _, image = self.pixel_scores(query_patches, b=b)
        return image, pixel

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "config": self.config.__dict__.copy(),
            "features": self.features.detach().cpu(),
            "seen_features": self.seen_features,
            "accepted_features": self.accepted_features,
            "replaced_features": self.replaced_features,
            "rejected_features": self.rejected_features,
            "update_batches": self.update_batches,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("schema") != self.schema:
            raise ValueError(f"unsupported coreset schema: {state.get('schema')!r}")
        saved = CADICPatchCoresetConfig(**state["config"])
        if saved != self.config:
            raise ValueError(f"coreset config mismatch: saved={saved}, current={self.config}")
        features = state["features"].to(self.device, dtype=torch.float32)
        if features.ndim != 2 or features.shape[1] != self.config.dim or features.shape[0] > self.config.budget:
            raise ValueError("invalid saved feature tensor")
        self.features = features
        self.seen_features = int(state.get("seen_features", 0))
        self.accepted_features = int(state.get("accepted_features", 0))
        self.replaced_features = int(state.get("replaced_features", 0))
        self.rejected_features = int(state.get("rejected_features", 0))
        self.update_batches = int(state.get("update_batches", 0))

    def stats(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "count": self.count,
            "budget": self.config.budget,
            "dim": self.config.dim,
            "dtype": str(self.features.dtype).replace("torch.", ""),
            "feature_bytes": self.feature_bytes,
            "seen_features": self.seen_features,
            "accepted_features": self.accepted_features,
            "replaced_features": self.replaced_features,
            "rejected_features": self.rejected_features,
            "update_batches": self.update_batches,
        }

    @torch.no_grad()
    def _nearest(self, query: torch.Tensor, bank: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        best = torch.full((query.shape[0],), float("inf"), device=query.device)
        indices = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        chunk = max(1, int(self.config.chunk_size))
        for start in range(0, bank.shape[0], chunk):
            part = bank[start:start + chunk]
            distances = torch.cdist(query, part, p=2)
            values, local = torch.min(distances, dim=1)
            mask = values < best
            best[mask] = values[mask]
            indices[mask] = start + local[mask]
        return best, indices

    @torch.no_grad()
    def _closest_pair(self) -> Tuple[torch.Tensor, int]:
        if self.count < 2:
            return torch.tensor(float("inf"), device=self.device), 0
        best_value = torch.tensor(float("inf"), device=self.device)
        best_row = 0
        chunk = max(1, int(self.config.chunk_size))
        for start in range(0, self.count, chunk):
            rows = self.features[start:start + chunk]
            distances = torch.cdist(rows, self.features, p=2)
            for local in range(rows.shape[0]):
                global_row = start + local
                distances[local, global_row] = float("inf")
            value, flat = torch.min(distances.reshape(-1), dim=0)
            if bool(value < best_value):
                best_value = value
                best_row = start + int(flat.item() % self.count)
        return best_value, best_row

    @torch.no_grad()
    def _topk_indices(self, query: torch.Tensor, k: int) -> torch.Tensor:
        values = []
        indices = []
        chunk = max(1, int(self.config.chunk_size))
        for start in range(0, self.count, chunk):
            part = self.features[start:start + chunk]
            values.append(torch.linalg.vector_norm(part - query, dim=1))
            indices.append(torch.arange(start, start + part.shape[0], device=self.device))
        distances = torch.cat(values)
        all_indices = torch.cat(indices)
        return all_indices[torch.topk(distances, k=k, largest=False).indices]
