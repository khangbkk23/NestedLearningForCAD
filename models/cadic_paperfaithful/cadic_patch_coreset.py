# models/cadic_patch_coreset_v1.py
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import torch


@torch.no_grad()
def euclidean_distance_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return FP32 Euclidean distances using CADIC Eq. (7).

    The caller may tile ``x`` and ``y`` to bound temporary memory.  Keeping
    this primitive explicit makes the GEMM based distance path auditable and
    avoids the size-dependent non-MM ``torch.cdist`` implementation.
    """
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("euclidean_distance_mm requires [N,D] and [M,D] tensors")
    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32, device=x.device)
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1).unsqueeze(0)
    d2 = x2 + y2 - 2.0 * (x @ y.transpose(0, 1))
    return d2.clamp_min_(0.0).sqrt()

@dataclass(frozen=True)
class CADICPatchCoresetConfig:
    budget: int = 10_000
    dim: int = 768
    dtype: str = "float32"
    distance: str = "euclidean"
    chunk_size: int = 2048
    image_neighbors: int = 9
    query_chunk_size: int = 256
    pair_chunk_size: int = 256

class CADICPatchCoresetV1:
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
        if min(
            self.config.chunk_size,
            self.config.query_chunk_size,
            self.config.pair_chunk_size,
        ) < 1:
            raise ValueError("distance chunk sizes must be positive")

        self.device = torch.device(device)
        self.features = torch.empty(
            (0, self.config.dim), dtype=torch.float32, device=self.device
        )
        self.seen_features = 0
        self.accepted_features = 0
        self.replaced_features = 0
        self.rejected_features = 0
        self.update_batches = 0

        # Diagnostics belong to the profiler, not to the scientific model
        # state. Scoring may increase these maxima without changing memory.
        self._distance_profile = {
            "pair_max_shape": [0, 0],
            "pair_max_elements": 0,
            "nearest_max_shape": [0, 0],
            "nearest_max_elements": 0,
            "pair_max_bytes": 0,
            "nearest_max_bytes": 0,
        }

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
                f"expected [N,{self.config.dim}] or [B,N,{self.config.dim}], "
                f"got {tuple(patch_features.shape)}"
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
            "incoming": (
                int(
                    patch_features.reshape(
                        -1, patch_features.shape[-1]
                    ).shape[0]
                )
                if patch_features.ndim >= 2
                else 0
            ),
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
            raise ValueError(
                f"expected [B,N,{self.config.dim}], got {tuple(query_patches.shape)}"
            )

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

            # `_nearest` returns only the minimum. Obtain the support indices
            # with a direct chunked distance pass to c*.
            support_indices = self._topk_indices(c_star, k)
            support = self.features[support_indices]
            support_dist = euclidean_distance_mm(
                image[star_row].reshape(1, -1), support
            ).squeeze(0)

            log_den = torch.logsumexp(support_dist, dim=0)
            ratio = torch.exp(star_score - log_den)
            weight = 1.0 - ratio

            all_pixel.append(pixel)
            all_indices.append(indices)
            all_image.append(weight * star_score)

        return (
            torch.stack(all_pixel),
            torch.stack(all_indices),
            torch.stack(all_image),
        )

    def score(
        self,
        query_patches: torch.Tensor,
        *,
        b: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pixel, _, image = self.pixel_scores(query_patches, b=b)
        return image, pixel

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "config": self.config.__dict__.copy(),
            "features": self.features.detach().cpu().clone(),
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
            raise ValueError(
                f"coreset config mismatch: saved={saved}, current={self.config}"
            )

        features = state["features"].to(
            self.device, dtype=torch.float32
        ).clone()
        if (
            features.ndim != 2
            or features.shape[1] != self.config.dim
            or features.shape[0] > self.config.budget
        ):
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

    def distance_profile(self) -> Dict[str, Any]:
        """Actual largest allocated distance block, with configured bounds."""
        return {
            **self._distance_profile,
            "pair_chunk_size": self.config.pair_chunk_size,
            "query_chunk_size": self.config.query_chunk_size,
            "bank_chunk_size": self.config.chunk_size,
            "pair_bound_elements": self.config.pair_chunk_size ** 2,
            "nearest_bound_elements": (
                self.config.query_chunk_size * self.config.chunk_size
            ),
            "pair_max_bytes": self._distance_profile["pair_max_bytes"],
            "nearest_max_bytes": self._distance_profile["nearest_max_bytes"],
            "distance_compute_mode": "explicit_fp32_gemm_eq7",
            "distance_dtype": "float32",
        }

    def _record_distance_block(
        self,
        kind: str,
        distances: torch.Tensor,
    ) -> None:
        count = int(distances.numel())
        if count > self._distance_profile[f"{kind}_max_elements"]:
            self._distance_profile[f"{kind}_max_elements"] = count
            self._distance_profile[f"{kind}_max_shape"] = list(distances.shape)
            self._distance_profile[f"{kind}_max_bytes"] = (
                count * distances.element_size()
            )

    @torch.no_grad()
    def _nearest(
        self,
        query: torch.Tensor,
        bank: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            query.ndim != 2
            or bank.ndim != 2
            or query.shape[1] != bank.shape[1]
            or not bank.shape[0]
        ):
            raise ValueError(
                "nearest requires [Q,D] and a nonempty [M,D] bank"
            )

        best = torch.full(
            (query.shape[0],),
            float("inf"),
            dtype=torch.float32,
            device=query.device,
        )
        indices = torch.zeros(
            query.shape[0], dtype=torch.long, device=query.device
        )

        for query_start in range(
            0, query.shape[0], self.config.query_chunk_size
        ):
            query_part = query[
                query_start : query_start + self.config.query_chunk_size
            ]
            best_part = best[
                query_start : query_start + len(query_part)
            ]
            index_part = indices[
                query_start : query_start + len(query_part)
            ]

            for start in range(
                0, bank.shape[0], self.config.chunk_size
            ):
                part = bank[start : start + self.config.chunk_size]

                distances = euclidean_distance_mm(query_part, part)
                self._record_distance_block("nearest", distances)

                values, local = torch.min(distances, dim=1)

                # Strict improvement preserves the lowest global bank index
                # on ties, both within a block and across successive blocks.
                mask = values < best_part
                best_part[mask] = values[mask]
                index_part[mask] = start + local[mask]

        return best, indices

    @torch.no_grad()
    def _closest_pair(self) -> Tuple[torch.Tensor, int]:
        if self.count < 2:
            return torch.tensor(
                float("inf"), dtype=torch.float32, device=self.device
            ), 0

        best_value = torch.tensor(
            float("inf"), dtype=torch.float32, device=self.device
        )
        best_pair = (self.count, self.count)
        chunk = self.config.pair_chunk_size

        # Only unordered pairs i < j are eligible. Tie policy is the
        # lexicographically first global (i,j), replacing its i member.
        for row_start in range(0, self.count, chunk):
            rows = self.features[
                row_start : row_start + chunk
            ]

            for col_start in range(
                row_start, self.count, chunk
            ):
                cols = self.features[
                    col_start : col_start + chunk
                ]
                distances = euclidean_distance_mm(rows, cols)
                self._record_distance_block("pair", distances)

                if row_start == col_start:
                    # Mask self distances and symmetric duplicates in place;
                    # each temporary remains bounded by pair_chunk_size².
                    for local in range(rows.shape[0]):
                        distances[local, : local + 1] = float("inf")

                value, flat = torch.min(
                    distances.reshape(-1), dim=0
                )
                flat_idx = int(flat.item())
                local_row = flat_idx // cols.shape[0]
                col = flat_idx % cols.shape[0]
                pair = (
                    row_start + local_row,
                    col_start + col,
                )

                if (
                    bool(value < best_value)
                    or (
                        bool(value == best_value)
                        and pair < best_pair
                    )
                ):
                    best_value = value
                    best_pair = pair

        return best_value, best_pair[0]

    @torch.no_grad()
    def _topk_indices(
        self,
        query: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        values = []
        indices = []
        chunk = max(1, int(self.config.chunk_size))

        for start in range(0, self.count, chunk):
            part = self.features[start : start + chunk]
            values.append(euclidean_distance_mm(query.reshape(1, -1), part).squeeze(0))
            indices.append(
                torch.arange(
                    start,
                    start + part.shape[0],
                    device=self.device,
                )
            )

        distances = torch.cat(values)
        all_indices = torch.cat(indices)

        return all_indices[
            torch.argsort(distances, stable=True)[:k]
        ]
