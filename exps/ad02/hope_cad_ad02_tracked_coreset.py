# exps/ad02/hope_cad_ad02_tracked_coreset.py
"""AD-02 provenance-tracking CADIC coreset.

AD-01 could not answer its own mediation question because the archived records
never stored *which task* each surviving exemplar came from. Its global
allocation tried to recover that by running a second, one-dimensional coreset
over the bin ids, which applies the CADIC replacement rule in *bin-id* space
rather than in feature space; the two banks therefore need not agree row for row.

This module records provenance where the decision is actually made. The update
loop is the AD-01 `FastCADICPatchCoresetV1` loop, reproduced line for line, with
metadata written alongside every row that is inserted or replaced:

    origin_task[row]      originating task id
    origin_image[row]     originating image index inside that task
    origin_position[row]  row-major patch position in the 28x28 lattice
    insert_step[row]      global image counter at insertion time

The metadata is written only on rows the rule selected, so it cannot influence
the rule. Metadata is held in Python containers rather than device tensors: the
replacement loop fires hundreds of times per image and a device scalar read per
event would synchronise the GPU that many times. Equivalence with the untracked
AD-01 coreset is asserted bitwise by `test_hope_cad_ad02_tracked_coreset.py`
and, on the real stream, by the exact replay of the archived AD-01 records.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from exps.ad01.hope_cad_ad01_fast_coreset import FastCADICPatchCoresetV1


class TrackedFastCADICPatchCoresetV1(FastCADICPatchCoresetV1):
    """`FastCADICPatchCoresetV1` plus exact per-row provenance."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        budget = int(self.config.budget)
        self.origin_task: list[int] = [-1] * budget
        self.origin_image: list[int] = [-1] * budget
        self.origin_position: list[int] = [-1] * budget
        self.insert_step: list[int] = [-1] * budget
        self.slot_counts: dict[int, int] = {}
        # Aggregates for the stage in progress; the caller drains them at each
        # task boundary.
        self.inserted_by_origin: dict[int, int] = {}
        self.evicted_by_origin: dict[int, int] = {}
        self.insertions_total = 0
        self.evictions_total = 0

    # ------------------------------------------------------------- provenance

    def _write_metadata(self, row: int, origin: Sequence[int]) -> None:
        self.origin_task[row] = int(origin[0])
        self.origin_image[row] = int(origin[1])
        self.origin_position[row] = int(origin[2])
        self.insert_step[row] = int(origin[3])

    def _note_insert(self, task: int) -> None:
        self.inserted_by_origin[task] = self.inserted_by_origin.get(task, 0) + 1
        self.slot_counts[task] = self.slot_counts.get(task, 0) + 1
        self.insertions_total += 1

    def _note_evict(self, row: int) -> None:
        task = self.origin_task[row]
        self.evicted_by_origin[task] = self.evicted_by_origin.get(task, 0) + 1
        if self.slot_counts.get(task, 0) > 0:
            self.slot_counts[task] -= 1
        self.evictions_total += 1

    def drain_events(self) -> dict[str, Any]:
        """Return and reset the stage's insertion/eviction aggregates."""
        payload = {
            "inserted_by_origin": dict(self.inserted_by_origin),
            "evicted_by_origin": dict(self.evicted_by_origin),
            "insertions_total": self.insertions_total,
            "evictions_total": self.evictions_total,
        }
        self.inserted_by_origin = {}
        self.evicted_by_origin = {}
        self.insertions_total = 0
        self.evictions_total = 0
        return payload

    def origin_rows(self) -> dict[str, torch.Tensor]:
        count = self.count
        return {
            "task": torch.tensor(self.origin_task[:count], dtype=torch.long),
            "image": torch.tensor(self.origin_image[:count], dtype=torch.long),
            "position": torch.tensor(self.origin_position[:count], dtype=torch.long),
            "insert_step": torch.tensor(self.insert_step[:count], dtype=torch.long),
        }

    def slot_counts_by_origin(self) -> dict[int, int]:
        return {k: v for k, v in self.slot_counts.items() if v > 0}

    # ----------------------------------------------------------------- update

    @torch.no_grad()
    def update(  # type: ignore[override]
        self, patch_features: torch.Tensor, *, origins: torch.Tensor | None = None
    ):
        """AD-01 `FastCADICPatchCoresetV1.update` with provenance bookkeeping.

        `origins` is an optional `[N,4]` long tensor of
        `(task, image, position, global_step)` for the incoming rows, in the same
        order as `patch_features`.
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

        rows = int(x.shape[0])
        if origins is None:
            origin_rows: list[list[int]] = [[-1, -1, -1, -1]] * rows
        else:
            if tuple(origins.shape) != (rows, 4):
                raise ValueError(
                    f"origins must be [{rows},4], got {tuple(origins.shape)}"
                )
            # One host transfer per image, then plain Python indexing.
            origin_rows = origins.detach().cpu().tolist()

        self.update_batches += 1
        self.seen_features += rows
        accepted = replaced = rejected = 0

        # Deterministic fill, exactly as the reference.
        if not self.is_full:
            room = self.config.budget - self.count
            take = min(room, rows)
            if take:
                self.features = torch.cat((self.features, x[:take]), dim=0)
                for offset in range(take):
                    row = self.count - take + offset
                    origin = origin_rows[offset]
                    self._write_metadata(row, origin)
                    self._note_insert(origin[0])
                accepted += take
                self._pair_dirty = True
            x = x[take:]
            origin_rows = origin_rows[take:]

        while x.numel() and self.is_full:
            nearest, _ = self._nearest(x, self.features)
            farthest_value, farthest_row = torch.max(nearest, dim=0)
            closest_value, closest_row = self._closest_pair_incremental()

            if not bool(farthest_value > closest_value):
                rejected += int(x.shape[0])
                break

            selected = int(farthest_row)
            origin = origin_rows[selected]
            self._note_evict(closest_row)
            self.features[closest_row] = x[selected]
            self._write_metadata(closest_row, origin)
            self._note_insert(origin[0])
            replaced += 1
            if self._pair_matrix is not None and not self._pair_dirty:
                self._refresh_pair_row(closest_row)
            else:
                self._pair_dirty = True

            keep = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            keep[selected] = False
            x = x[keep]
            origin_rows = [
                origin for index, origin in enumerate(origin_rows) if index != selected
            ]

        self.accepted_features += accepted
        self.replaced_features += replaced
        self.rejected_features += rejected

        return {
            "incoming": int(patch_features.shape[-2])
            if patch_features.ndim == 3
            else int(patch_features.shape[0]),
            "accepted": accepted,
            "replaced": replaced,
            "rejected": rejected,
        }

    # ----------------------------------------------------------- verification

    @torch.no_grad()
    def verify_origins(
        self, cache: dict[tuple[int, int], torch.Tensor]
    ) -> dict[str, Any]:
        """Check that every stored row really is the recorded source patch.

        `cache[(task, image)]` is that image's `[784, D]` feature block. The check
        is bitwise, so it validates the provenance record against the data rather
        than against another bookkeeping path.
        """
        count = self.count
        mismatch = 0
        missing = 0
        for row in range(count):
            key = (self.origin_task[row], self.origin_image[row])
            block = cache.get(key)
            if block is None:
                missing += 1
                continue
            expected = block[self.origin_position[row]].to(
                self.features.device, dtype=torch.float32
            )
            if not bool(torch.equal(expected, self.features[row])):
                mismatch += 1
        return {
            "rows": count,
            "checked": count - missing,
            "missing_source": missing,
            "bitwise_mismatch": mismatch,
            "exact": bool(mismatch == 0 and missing == 0),
        }

    @torch.no_grad()
    def age_summary(self) -> dict[str, Any]:
        """Age (in images) of surviving rows, per origin task."""
        count = self.count
        out: dict[int, list[int]] = {}
        for row in range(count):
            if self.insert_step[row] < 0:
                continue
            out.setdefault(self.origin_task[row], []).append(self.insert_step[row])
        return out

    # -------------------------------------------------------------- lifecycle

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        payload = super().state_dict()
        count = self.count
        payload["provenance"] = {
            "schema": "ad02_provenance_v1",
            "origin_task": self.origin_task[:count],
            "origin_image": self.origin_image[:count],
            "origin_position": self.origin_position[:count],
            "insert_step": self.insert_step[:count],
        }
        return payload

    def load_state_dict(self, state: dict[str, Any]) -> None:  # type: ignore[override]
        super().load_state_dict(state)
        provenance = state.get("provenance")
        if provenance is None:
            raise ValueError("tracked coreset state is missing provenance rows")
        count = self.count
        for key, field in (
            ("origin_task", self.origin_task),
            ("origin_image", self.origin_image),
            ("origin_position", self.origin_position),
            ("insert_step", self.insert_step),
        ):
            values = list(provenance[key])
            if len(values) != count:
                raise ValueError(f"provenance {key} length mismatch")
            field[:count] = values
        self.slot_counts = {}
        for task in self.origin_task[:count]:
            self.slot_counts[task] = self.slot_counts.get(task, 0) + 1
