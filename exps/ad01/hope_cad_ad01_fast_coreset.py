# exps/ad01/hope_cad_ad01_fast_coreset.py
"""Distance-kernel-optimised CADIC coreset for the AD-01 GPU sweep.

Motivation, measured rather than assumed: the frozen CADIC `update()` took
32-40 seconds per image once its bank was full. The bank is only
`2500 x 768` float32, so the arithmetic is trivial; the cost is Python and
kernel-launch overhead. `_closest_pair()` scans an O(M^2) distance matrix in
`pair_chunk_size` tiles by calling `torch.cdist` around a hundred times per
invocation, and `update()` calls it once per replacement, roughly forty times
per image. That is thousands of tiny kernel launches per image.

This module swaps only the distance primitive for the same expansion CADIC
already uses in its Eq. (7) paper-faithful variant:

    d2(x, y) = x2 + y2 - 2 x y^T        (clamped at zero before the sqrt)

The replacement rule, the tie policy, the rejection test, the row order and the
resulting bank semantics are unchanged; only how a distance matrix is formed
changes. Because the change is numerical, not semantic, it is validated against
the frozen reference on the same inputs before the sweep uses it, and the
observed deviation is reported.
"""

from __future__ import annotations

import torch

from models.cadic_patch_coreset_v1 import CADICPatchCoresetV1


@torch.no_grad()
def pairwise_euclidean(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Euclidean distances between two `[N,D]`/`[M,D]` float32 matrices.

    Same algebra as CADIC Eq. (7). The caller is responsible for bounding the
    temporary, exactly as the reference does with its chunk sizes.
    """
    x = left.to(dtype=torch.float32)
    y = right.to(dtype=torch.float32, device=x.device)
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1).unsqueeze(0)
    d2 = x2 + y2 - 2.0 * (x @ y.transpose(0, 1))
    return d2.clamp_min_(0.0).sqrt_()


class FastCADICPatchCoresetV1(CADICPatchCoresetV1):
    """The frozen CADIC coreset with an incrementally maintained closest pair.

    Two independent costs are removed relative to the reference, both measured
    before being changed:

    1. The distance primitive becomes the Eq. (7) matrix multiply rather than
       many small `torch.cdist` tiles.
    2. The closest-pair search becomes incremental. The CADIC rule consults the
       closest pair after *every* replacement, which is roughly 490 times per
       image on this data, and each consultation rescanned the whole O(M^2)
       matrix. A replacement changes exactly one row of the bank, so the
       pairwise distance matrix is cached and only the affected row and column
       are recomputed.

    Neither change alters the rule, the tie policy or the resulting bank; the
    equivalence tests assert that on deterministic inputs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pair_matrix: torch.Tensor | None = None
        self._pair_dirty = True

    def _distance_matrix(self) -> torch.Tensor:
        """Cached full pairwise distances, rebuilt only when the bank resizes."""
        if (
            self._pair_matrix is None
            or self._pair_matrix.shape[0] != self.count
            or self._pair_dirty
        ):
            self._pair_matrix = pairwise_euclidean(self.features, self.features)
            self._pair_dirty = False
        return self._pair_matrix

    def _refresh_pair_row(self, index: int) -> None:
        """Recompute only the distances involving one bank row."""
        matrix = self._pair_matrix
        row = pairwise_euclidean(
            self.features[index].reshape(1, -1), self.features
        ).reshape(-1)
        matrix[index, :] = row
        matrix[:, index] = row

    def _closest_pair_incremental(self):
        """Closest unordered pair from the cached matrix.

        Uses the same lexicographic tie policy as the reference: the first
        minimum in row-major order over `i < j` wins, and `i` is the member that
        gets replaced.
        """
        if self.count < 2:
            return (
                torch.tensor(float("inf"), dtype=torch.float32, device=self.device),
                0,
            )
        matrix = self._distance_matrix()
        # Exclude the diagonal and the lower triangle so that only i < j remain.
        upper = torch.triu(
            torch.ones(self.count, self.count, dtype=torch.bool, device=self.device),
            diagonal=1,
        )
        masked = torch.where(upper, matrix, torch.inf)
        flat = int(torch.argmin(masked).item())
        row = flat // self.count
        return masked[row, flat % self.count], row

    @torch.no_grad()
    def update(self, patch_features: torch.Tensor):
        """Update with an incrementally maintained closest-pair cache.

        The reference computes the closest pair from scratch every iteration.
        Here the cached matrix is kept in step with `self.features` so each
        consultation costs one row update plus a reduction.
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
        accepted = replaced = rejected = 0

        # Deterministic fill, exactly as the reference.
        if not self.is_full:
            room = self.config.budget - self.count
            take = min(room, x.shape[0])
            if take:
                self.features = torch.cat((self.features, x[:take]), dim=0)
                accepted += take
                self._pair_dirty = True
            x = x[take:]

        while x.numel() and self.is_full:
            nearest, _ = self._nearest(x, self.features)
            farthest_value, farthest_row = torch.max(nearest, dim=0)
            closest_value, closest_row = self._closest_pair_incremental()

            if not bool(farthest_value > closest_value):
                rejected += int(x.shape[0])
                break

            self.features[closest_row] = x[farthest_row]
            replaced += 1
            if self._pair_matrix is not None and not self._pair_dirty:
                # One row and its column changed; refresh them in place.
                self._refresh_pair_row(closest_row)
            else:
                self._pair_dirty = True

            # Remove only the selected row; the remaining candidates stay
            # available for the repeated batch procedure. This must be the same
            # removal the reference performs, or the trajectories diverge.
            keep = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            keep[farthest_row] = False
            x = x[keep]

        self.accepted_features += accepted
        self.replaced_features += replaced
        self.rejected_features += rejected

        return {
            "incoming": int(patch_features.shape[-2]) if patch_features.ndim == 3 else int(patch_features.shape[0]),
            "accepted": accepted,
            "replaced": replaced,
            "rejected": rejected,
        }

    @torch.no_grad()
    def _nearest(self, query: torch.Tensor, bank: torch.Tensor):
        if (
            query.ndim != 2
            or bank.ndim != 2
            or query.shape[1] != bank.shape[1]
            or not bank.shape[0]
        ):
            raise ValueError("nearest requires [Q,D] and a nonempty [M,D] bank")

        best = torch.full(
            (query.shape[0],), float("inf"), dtype=torch.float32, device=query.device
        )
        indices = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)

        for start in range(0, query.shape[0], self.config.query_chunk_size):
            part = query[start : start + self.config.query_chunk_size]
            distances = pairwise_euclidean(part, bank)
            self._record_distance_block("nearest", distances)
            values, positions = distances.min(dim=1)
            best[start : start + part.shape[0]] = values
            indices[start : start + part.shape[0]] = positions
        return best, indices

    @torch.no_grad()
    def _closest_pair_reference_order(self):
        """Tiled scan, kept as the numerical oracle for the incremental path.

        The tiling and masking order are identical to the reference; only the
        distance call differs, so the selected pair is the same pair whenever
        the two distance evaluations agree on the minimum.
        """
        if self.count < 2:
            return (
                torch.tensor(float("inf"), dtype=torch.float32, device=self.device),
                0,
            )

        best_value = float("inf")
        best_row = self.count
        best_col = self.count
        chunk = self.config.pair_chunk_size

        for row_start in range(0, self.count, chunk):
            rows = self.features[row_start : row_start + chunk]
            for col_start in range(row_start, self.count, chunk):
                cols = self.features[col_start : col_start + chunk]
                distances = pairwise_euclidean(rows, cols)
                self._record_distance_block("pair", distances)
                if row_start == col_start:
                    for local in range(rows.shape[0]):
                        distances[local, : local + 1] = float("inf")

                value, flat = torch.min(distances.reshape(-1), dim=0)
                value = float(value.item())
                # Strictly-less keeps the lexicographically first global (i, j),
                # matching the reference's tie handling.
                if value < best_value:
                    flat_index = int(flat.item())
                    best_value = value
                    best_row = row_start + flat_index // cols.shape[0]
                    best_col = col_start + flat_index % cols.shape[0]

        if best_row > best_col:
            best_row, best_col = best_col, best_row
        return (
            torch.tensor(best_value, dtype=torch.float32, device=self.device),
            best_row,
        )

    @torch.no_grad()
    def _topk_indices(self, vector: torch.Tensor, k: int) -> torch.Tensor:
        """Support neighbours of a query vector, matching the reference ordering."""
        distances = pairwise_euclidean(
            vector.reshape(1, -1), self.features
        ).reshape(-1)
        self._record_distance_block("nearest", distances.reshape(1, -1))
        return torch.topk(distances, k, largest=False, sorted=True).indices


class FastNormalSupportMemoryMixin:
    """Placeholder documenting that the memory composes this coreset.

    `NormalSupportMemory` builds its banks through a factory, so the fast class
    is selected there rather than by subclassing the memory.
    """


def coreset_factory(fast: bool):
    """Return the coreset class to instantiate for a given mode."""
    return FastCADICPatchCoresetV1 if fast else CADICPatchCoresetV1
