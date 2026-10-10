# exps/ad02/hope_cad_ad02_memory.py
"""AD-02 instrumented normal-support memory.

The memory is AD-01's `NormalSupportMemory` with three deliberate changes, none
of which touches the scoring path or the replacement rule:

1. **Exact provenance.** Every group bank is a
   `TrackedFastCADICPatchCoresetV1`, so the originating task, image and patch
   position of every surviving row are recorded where the replacement decision is
   made. This replaces AD-01's one-dimensional bin-id coreset, which applied the
   CADIC rule in the wrong space and could not record task origin at all.
2. **Arbitrary partitions.** `assignment` supplies the group id of every one of
   the 784 lattice positions, so the spatial grids and the balanced non-spatial
   random control run through exactly the same code path. Group count and group
   quotas are still derived from the AD-01 rule.
3. **Correct occupancy accounting.** The global arm reports one bank, and every
   per-group view is explicitly disjoint. AD-01's diagnostic listed 16 pointers
   to the *same* bank for a global allocation, so `sum(bin count)` was 16x the
   true size.

Scoring is **global** in every AD-02 arm: a query patch may match any stored
exemplar, whatever group it lives in. Only where capacity is spent differs
between arms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from exps.ad01.hope_cad_ad01_fast_coreset import pairwise_euclidean
from exps.ad01.hope_cad_ad01_normal_support import (
    ALLOCATION_GLOBAL,
    ALLOCATION_SPATIAL,
    PATCHES,
    NormalSupportMemory,
    per_bin_quota,
)
from exps.ad02.hope_cad_ad02_tracked_coreset import TrackedFastCADICPatchCoresetV1

from models.cadic_patch_coreset_v1 import CADICPatchCoresetConfig

QUANTILES = (0.0, 5.0, 25.0, 50.0, 75.0, 95.0, 100.0)


def summarise(values: np.ndarray) -> dict[str, float]:
    """Quantile summary used for every reported distribution."""
    finite = values[np.isfinite(values)]
    payload = {
        f"q{int(q)}": (float(np.percentile(finite, q)) if finite.size else float("nan"))
        for q in QUANTILES
    }
    payload["mean"] = float(finite.mean()) if finite.size else float("nan")
    payload["std"] = float(finite.std()) if finite.size else float("nan")
    payload["n"] = int(values.size)
    payload["n_nonfinite"] = int(values.size - finite.size)
    return payload


@dataclass
class InstrumentedNormalSupportMemory(NormalSupportMemory):
    """NormalSupportMemory with provenance, arbitrary partitions and cost probes."""

    assignment: torch.Tensor | None = None
    arm_label: str = ""
    task_names: Sequence[str] = ()
    partition_kind: str = "spatial"

    # ------------------------------------------------------------------ setup

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.assignment is not None:
            assignment = self.assignment.detach().to(self.device, dtype=torch.long)
            if assignment.numel() != PATCHES:
                raise ValueError(
                    f"assignment must cover {PATCHES} positions, got {assignment.numel()}"
                )
            n_groups = int(assignment.max().item()) + 1
            if n_groups != self.n_bins:
                raise ValueError(
                    f"assignment has {n_groups} groups but the memory was built "
                    f"for {self.n_bins}; set grid*grid accordingly"
                )
            self._bins = assignment
            self.n_bins = n_groups

        def config(quota: int) -> CADICPatchCoresetConfig:
            return CADICPatchCoresetConfig(
                budget=quota,
                dim=self.dim,
                dtype="float32",
                distance="euclidean",
                chunk_size=self.chunk_size,
                image_neighbors=self.image_neighbors,
                query_chunk_size=self.query_chunk_size,
                pair_chunk_size=self.pair_chunk_size,
            )

        if self.allocation == ALLOCATION_SPATIAL:
            self._banks = [
                TrackedFastCADICPatchCoresetV1(config(quota), device=self.device)
                for quota in self.quotas
            ]
        elif self.allocation == ALLOCATION_GLOBAL:
            tracked = TrackedFastCADICPatchCoresetV1(
                config(self.budget), device=self.device
            )
            self._global = tracked
            self._banks = [tracked]
            # The AD-01 global arm carried a second 1-D coreset over bin ids to
            # guess provenance. It is not needed and is not faithful, so the
            # instrumented global arm runs the same single bank as arm A.
            self._provenance = None
        else:
            raise ValueError(f"unknown allocation {self.allocation!r}")

        self._insert_counts = [0] * self.n_bins
        self._stage_events: list[dict[str, Any]] = []
        self._current_stage = -1
        # `_bin_of_position` in the parent is unused; keep the attribute defined.
        self._bin_of_position = []

    # ----------------------------------------------------------------- update

    @torch.no_grad()
    def update(
        self,
        patch_features: torch.Tensor,
        *,
        task_id: int = -1,
        image_index: int = -1,
        global_step: int = -1,
    ) -> dict[str, int]:
        """Consume one image's patches, recording where every row came from."""
        x = patch_features.detach().to(self.device, dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"expected [N,{self.dim}], got {tuple(x.shape)}")
        if x.shape[0] != PATCHES:
            raise ValueError(f"expected exactly {PATCHES} patches, got {x.shape[0]}")

        positions = torch.arange(PATCHES, dtype=torch.long, device=self.device)
        origins_all = torch.stack(
            [
                torch.full_like(positions, int(task_id)),
                torch.full_like(positions, int(image_index)),
                positions,
                torch.full_like(positions, int(global_step)),
            ],
            dim=1,
        )

        totals = {"incoming": 0, "accepted": 0, "replaced": 0, "rejected": 0}
        if self.allocation == ALLOCATION_GLOBAL:
            # One bank, no partition: the same call AD-01 arm A makes, with
            # provenance attached.
            current = self._banks[0].update(x, origins=origins_all)
            for key in totals:
                totals[key] += int(current[key])
            self._insert_counts[0] += PATCHES
            return totals

        for bin_id in range(self.n_bins):
            selector = self._bins == bin_id
            part = x[selector]
            if part.shape[0] == 0:
                continue
            current = self._banks[bin_id].update(
                part, origins=origins_all[selector]
            )
            for key in totals:
                totals[key] += int(current[key])
            self._insert_counts[bin_id] += int(part.shape[0])
        return totals

    def begin_stage(self, stage: int) -> None:
        self._current_stage = int(stage)
        for bank in self._banks:
            bank.drain_events()

    def end_stage(self) -> dict[str, Any]:
        """Drain the stage's insertion/eviction events across all groups."""
        inserted: dict[int, int] = {}
        evicted: dict[int, int] = {}
        insertions = evictions = 0
        for bank in self._banks:
            payload = bank.drain_events()
            for key, value in payload["inserted_by_origin"].items():
                inserted[key] = inserted.get(key, 0) + value
            for key, value in payload["evicted_by_origin"].items():
                evicted[key] = evicted.get(key, 0) + value
            insertions += payload["insertions_total"]
            evictions += payload["evictions_total"]
        payload = {
            "stage": self._current_stage,
            "inserted_by_origin": inserted,
            "evicted_by_origin": evicted,
            "insertions_total": insertions,
            "evictions_total": evictions,
        }
        self._stage_events.append(payload)
        return payload

    # -------------------------------------------------------------- provenance

    def _combined_origin_rows(self) -> dict[str, torch.Tensor] | None:
        """Per-row provenance in the row order of the global scorer's matrix."""
        banks = self._banks
        tasks, groups, images, positions, steps = [], [], [], [], []
        for group, bank in enumerate(banks):
            count = int(bank.count)
            if count == 0:
                continue
            rows = bank.origin_rows()
            tasks.append(rows["task"])
            images.append(rows["image"])
            positions.append(rows["position"])
            steps.append(rows["insert_step"])
            groups.append(torch.full((count,), group, dtype=torch.long))
        if not tasks:
            return None
        return {
            "task": torch.cat(tasks),
            "group": torch.cat(groups),
            "image": torch.cat(images),
            "position": torch.cat(positions),
            "insert_step": torch.cat(steps),
        }

    def slot_counts_by_origin(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for bank in self._banks:
            for origin, value in bank.slot_counts_by_origin().items():
                counts[origin] = counts.get(origin, 0) + value
        return counts

    def slot_counts_by_group_and_origin(self) -> list[dict[int, int]]:
        return [bank.slot_counts_by_origin() for bank in self._banks]

    def age_stats_by_origin(self, current_step: int) -> dict[int, dict[str, float]]:
        rows = self._combined_origin_rows()
        if rows is None:
            return {}
        ages = (int(current_step) - rows["insert_step"]).numpy().astype(np.float64)
        tasks = rows["task"].numpy()
        out: dict[int, dict[str, float]] = {}
        for task in np.unique(tasks):
            values = ages[tasks == task]
            out[int(task)] = {
                "count": int(values.size),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        return out

    # --------------------------------------------------------------- coverage

    @torch.no_grad()
    def coverage_raw(
        self, query_patches: torch.Tensor, *, task_id: int, chunk: int = 1024
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Raw per-patch `(d_any, d_own, own_slots)` for a task's normal patches.

        `d_any` is the distance to the nearest stored exemplar of *any* task, i.e.
        what the deployed global scorer sees. `d_own` is the distance to the
        nearest surviving exemplar *of this task*, i.e. how much of this task's own
        normal support is still in memory. If every exemplar of the task has been
        evicted, `d_own` is infinite and `own_slots` is zero.

        The two are equal at the boundary where the task was just learned, so the
        per-patch ratio `d_own(b) / d_own(t)` is a scale-free measure of how much
        of that task's own normal support has been lost.
        """
        features = self._combined_bank()
        rows = self._combined_origin_rows()
        queries = query_patches.detach().to(self.device, dtype=torch.float32)
        if queries.ndim == 3:
            queries = queries.reshape(-1, queries.shape[-1])

        if rows is None:
            empty = np.full(queries.shape[0], np.inf)
            return empty, empty.copy(), 0

        own_mask = (rows["task"] == int(task_id)).to(self.device)
        d_any = torch.empty(queries.shape[0], dtype=torch.float32, device=self.device)
        d_own = torch.empty(queries.shape[0], dtype=torch.float32, device=self.device)
        for start in range(0, queries.shape[0], chunk):
            part = queries[start : start + chunk]
            distances = pairwise_euclidean(part, features)
            d_any[start : start + part.shape[0]] = distances.min(dim=1).values
            if bool(own_mask.any()):
                d_own[start : start + part.shape[0]] = distances[:, own_mask].min(
                    dim=1
                ).values
            else:
                d_own[start : start + part.shape[0]] = float("inf")

        return (
            d_any.detach().cpu().numpy().astype(np.float64),
            d_own.detach().cpu().numpy().astype(np.float64),
            int(own_mask.sum().item()),
        )

    # ------------------------------------------------------------ diagnostics

    def occupancy(self) -> dict[str, Any]:
        """Disjoint per-group occupancy. Sums are valid in every allocation.

        AD-01's global branch returned 16 references to one bank, so summing the
        rows reported 40,000 stored vectors for a 2,500 budget. Here a global
        allocation reports exactly one bank with `partitioned = False`, and a
        partitioned allocation reports one row per group.
        """
        if self.allocation == ALLOCATION_GLOBAL:
            bank = self._global
            return {
                "allocation": self.allocation,
                "partitioned": False,
                "grid": self.grid,
                "n_groups": 1,
                "total_budget": self.budget,
                "total_count": int(bank.count),
                "capacity_total": int(self.budget),
                "bins_are_disjoint": True,
                "sum_bins_is_total": True,
                "bins": [
                    {
                        "bin_id": 0,
                        "capacity": int(bank.config.budget),
                        "count": int(bank.count),
                        "fill_ratio": bank.count / max(1, bank.config.budget),
                        "inserts": int(self._insert_counts[0]),
                        "seen": int(bank.seen_features),
                        "replaced": int(bank.replaced_features),
                        "rejected": int(bank.rejected_features),
                        "accepted": int(bank.accepted_features),
                    }
                ],
                "min_fill_ratio": bank.count / max(1, bank.config.budget),
                "max_fill_ratio": bank.count / max(1, bank.config.budget),
            }

        rows = []
        for bin_id, bank in enumerate(self._banks):
            rows.append(
                {
                    "bin_id": bin_id,
                    "capacity": int(bank.config.budget),
                    "count": int(bank.count),
                    "fill_ratio": bank.count / max(1, bank.config.budget),
                    "inserts": int(self._insert_counts[bin_id]),
                    "seen": int(bank.seen_features),
                    "replaced": int(bank.replaced_features),
                    "rejected": int(bank.rejected_features),
                    "accepted": int(bank.accepted_features),
                }
            )
        return {
            "allocation": self.allocation,
            "partitioned": True,
            "grid": self.grid,
            "n_groups": self.n_bins,
            "total_budget": self.budget,
            "total_count": int(self.count),
            "capacity_total": int(sum(r["capacity"] for r in rows)),
            "bins_are_disjoint": True,
            "sum_bins_is_total": True,
            "bins": rows,
            "min_fill_ratio": min(r["fill_ratio"] for r in rows),
            "max_fill_ratio": max(r["fill_ratio"] for r in rows),
        }

    def transient_bytes(self) -> dict[str, int]:
        """Analytic transient cost of the incremental closest-pair cache.

        `FastCADICPatchCoresetV1` keeps the bank's full pairwise distance matrix to
        avoid rescanning it after every replacement. For one bank of `M` rows that
        is `M^2` float32 values; partitioned allocations keep one such matrix per
        group, so the cost depends on how the same total budget is split. This is
        working memory, not persistent storage.
        """
        per_bank = [
            int(bank.config.budget) ** 2 * 4 for bank in self._banks
        ]
        return {
            "closest_pair_matrix_bytes_total": int(sum(per_bank)),
            "closest_pair_matrix_bytes_max_bank": int(max(per_bank)),
            "n_banks": len(per_bank),
        }

    def persistent_metadata_bytes(self) -> int:
        """Bytes needed to reconstruct the partition at inference time.

        A spatial grid is fully described by `grid`, so it costs nothing beyond
        the code. A random partition is not derivable from a rule, so its
        784-entry group map must be stored: 784 int8 values.
        """
        if self.allocation == ALLOCATION_GLOBAL or self.partition_kind != "random":
            return 0
        # A random partition is not derivable from a rule, so its 784-entry group
        # map has to be persisted: one int8 group id per lattice position.
        return int(self.assignment.numel()) if self.assignment is not None else 0

    def provenance_snapshot(self) -> dict[str, Any]:
        rows = self._combined_origin_rows()
        if rows is None:
            return {"rows": 0}
        return {
            "rows": int(rows["task"].numel()),
            "slots_by_origin": {
                str(k): int(v) for k, v in self.slot_counts_by_origin().items()
            },
            "slots_by_group_and_origin": [
                {str(k): int(v) for k, v in counts.items()}
                for counts in self.slot_counts_by_group_and_origin()
            ],
        }

    def verify_origins(self, cache: dict[tuple[int, int], torch.Tensor]) -> dict[str, Any]:
        """Bitwise check that recorded provenance matches the source patches."""
        banks = self._banks
        rows = checked = missing = mismatch = 0
        for bank in banks:
            payload = bank.verify_origins(cache)
            rows += payload["rows"]
            checked += payload["checked"]
            missing += payload["missing_source"]
            mismatch += payload["bitwise_mismatch"]
        return {
            "rows": rows,
            "checked": checked,
            "missing_source": missing,
            "bitwise_mismatch": mismatch,
            "exact": bool(mismatch == 0 and missing == 0 and rows == checked),
        }
