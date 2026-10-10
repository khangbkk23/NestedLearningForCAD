# exps/hope_cad_ad01_normal_support.py
"""AD-01 four-arm normal-support memory and anomaly readout.

Two factors are varied independently under a matched total exemplar budget:

    allocation : where bank capacity lives
        GLOBAL  - one unified bank, CADIC Eq.(1)-(6) farthest-point replacement
        SPATIAL - one independent bank per spatial bin, equal capacity per bin

    scoring : which bank entries a query patch may match
        GLOBAL  - the whole bank
        LOCAL   - only the entries allocated to the query patch's spatial bin

That yields the four predeclared arms:

    A = GLOBAL  allocation + GLOBAL scoring   (CADIC reference)
    B = GLOBAL  allocation + LOCAL  scoring
    C = SPATIAL allocation + GLOBAL scoring
    D = SPATIAL allocation + LOCAL  scoring

`B - A` isolates scoring locality with the stored support held fixed.
`C - A` isolates allocation locality with the scorer held fixed.
`D - A` is their joint effect.

Both the allocation binning and the scoring neighborhood are the same fixed
`grid x grid` partition of the 28x28 patch lattice, so the two "spatial" factors
use one consistent notion of location. Coordinates are used only to index the
bank; no task identity is ever available to `fit_task` scoring, and no anomaly
label is read by any fitting, admission, eviction, or normalization step.

The replacement rule, distance kernel, support-density image score, and
pixel-map contract are inherited unchanged from the frozen CADIC reference so
that the comparison isolates the two declared factors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Sequence

import torch

from models.cadic_patch_coreset_v1 import CADICPatchCoresetV1, CADICPatchCoresetConfig

GRID_SIDE = 28
PATCHES = GRID_SIDE * GRID_SIDE

ALLOCATION_GLOBAL = "global"
ALLOCATION_SPATIAL = "spatial"
SCORING_GLOBAL = "global"
SCORING_LOCAL = "local"


def bin_assignment(grid: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return the bin id of every patch in row-major 28x28 order.

    The 28-patch side is split into `grid` equal parts. `grid` must divide 28 so
    every bin has identical patch count and the partition is exact.
    """
    if GRID_SIDE % grid:
        raise ValueError(f"grid must divide {GRID_SIDE}, got {grid}")
    stride = GRID_SIDE // grid
    rows = torch.arange(GRID_SIDE) // stride
    cols = torch.arange(GRID_SIDE) // stride
    bin_of = (rows[:, None] * grid + cols[None, :]).reshape(-1)
    return bin_of.to(torch.long).to(device)


def per_bin_quota(total: int, n_bins: int) -> list[int]:
    """Split `total` across `n_bins` as evenly as possible, remainder first."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if total < n_bins:
        raise ValueError(f"budget {total} is smaller than {n_bins} bins")
    base, remainder = divmod(total, n_bins)
    return [base + (1 if index < remainder else 0) for index in range(n_bins)]


@dataclass
class NormalSupportMemory:
    """Spatially indexed normal-support memory with declared allocation.

    `allocation=global` delegates every insert to a single CADIC coreset.
    `allocation=spatial` maintains one CADIC coreset per spatial bin. In both
    cases the *same* CADIC farthest-point replacement rule is used; only the
    scope of the bank differs. Total capacity is identical across arms.
    """

    budget: int = 2500
    grid: int = 4
    allocation: str = ALLOCATION_GLOBAL
    dim: int = 768
    chunk_size: int = 256
    query_chunk_size: int = 256
    pair_chunk_size: int = 256
    image_neighbors: int = 9
    device: str | torch.device = "cpu"
    _bins: torch.Tensor = field(init=False, repr=False)
    _banks: list[CADICPatchCoresetV1] = field(init=False, repr=False, default_factory=list)
    _global: CADICPatchCoresetV1 | None = field(init=False, repr=False, default=None)
    _bin_of_position: list[list[int]] = field(init=False, repr=False, default_factory=list)
    _insert_counts: list[int] = field(init=False, repr=False, default_factory=list)
    _combined_owner: list[int] | None = field(init=False, repr=False, default=None)
    _bank_bin: torch.Tensor | None = field(init=False, repr=False, default=None)
    _provenance: CADICPatchCoresetV1 | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.allocation not in (ALLOCATION_GLOBAL, ALLOCATION_SPATIAL):
            raise ValueError(f"unknown allocation {self.allocation!r}")
        self.device = torch.device(self.device)
        self._bins = bin_assignment(self.grid, device=self.device)
        self.n_bins = self.grid * self.grid

        def make(budget: int) -> CADICPatchCoresetV1:
            return CADICPatchCoresetV1(
                CADICPatchCoresetConfig(
                    budget=budget,
                    dim=self.dim,
                    dtype="float32",
                    distance="euclidean",
                    chunk_size=self.chunk_size,
                    image_neighbors=self.image_neighbors,
                    query_chunk_size=self.query_chunk_size,
                    pair_chunk_size=self.pair_chunk_size,
                ),
                device=self.device,
            )

        if self.allocation == ALLOCATION_GLOBAL:
            self._global = make(self.budget)
            self._banks = [self._global]
        else:
            quotas = per_bin_quota(self.budget, self.n_bins)
            self._banks = [make(quota) for quota in quotas]
            self.quotas = quotas
            # Arm C needs one scorer over the *union* of all bin banks. It is
            # never updated; it only exposes the combined feature matrix and the
            # ownership map back to the bin that actually holds each entry.
            self._global = make(self.budget)
        if self.allocation == ALLOCATION_GLOBAL:
            # A one-dimensional CADIC coreset over lattice positions. It sees
            # exactly the same patch stream with the same replacement rule, so
            # its row order matches the scored bank's row order entry for entry.
            self._provenance = CADICPatchCoresetV1(
                CADICPatchCoresetConfig(
                    budget=self.budget,
                    dim=1,
                    dtype="float32",
                    distance="euclidean",
                    chunk_size=self.chunk_size,
                    image_neighbors=1,
                    query_chunk_size=self.query_chunk_size,
                    pair_chunk_size=self.pair_chunk_size,
                ),
                device=self.device,
            )
        else:
            self._provenance = None
        self._insert_counts = [0] * self.n_bins

    # ---------------------------------------------------------------- update

    @torch.no_grad()
    def update(self, patch_features: torch.Tensor) -> dict[str, int]:
        """Consume one image's patch features `[N, D]` in row-major patch order."""
        x = patch_features.detach().to(self.device, dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"expected [N,{self.dim}], got {tuple(x.shape)}")
        if x.shape[0] != PATCHES:
            raise ValueError(f"expected exactly {PATCHES} patches, got {x.shape[0]}")

        if self.allocation == ALLOCATION_GLOBAL:
            bank = self._global
            result = bank.update(x)
            # `_provenance` applies the identical CADIC replacement rule to the
            # same patch stream but records each row's originating lattice
            # position. It is the authoritative source of bank-row bins, so the
            # scored bank itself stays byte-identical to the reference.
            self._provenance.update(
                self._bins.to(torch.float32).reshape(PATCHES, 1)
            )
            self._bank_bin = self._provenance.features[:, 0].clone()
            for bin_id in range(self.n_bins):
                self._insert_counts[bin_id] += int((self._bins == bin_id).sum())
            return result

        totals = {"incoming": 0, "accepted": 0, "replaced": 0, "rejected": 0}
        for bin_id in range(self.n_bins):
            selector = self._bins == bin_id
            part = x[selector]
            current = self._banks[bin_id].update(part)
            for key in totals:
                totals[key] += int(current[key])
            self._insert_counts[bin_id] += int(part.shape[0])
        return totals

    # --------------------------------------------------------------- reading

    @property
    def count(self) -> int:
        return sum(bank.count for bank in self._banks)

    def is_full(self) -> bool:
        return all(bank.is_full for bank in self._banks)

    @property
    def feature_bytes(self) -> int:
        return sum(bank.feature_bytes for bank in self._banks)

    @property
    def memory_bytes(self) -> int:
        return sum(bank.memory_bytes for bank in self._banks)

    def bin_banks(self) -> list[CADICPatchCoresetV1]:
        if self.allocation == ALLOCATION_SPATIAL:
            return self._banks
        # A global bank is addressed positionally for LOCAL scoring by taking the
        # same physical entries; capacity is *not* partitioned, so a bin can be
        # empty or hold the entire bank. This is what makes arm B a pure scorer
        # change relative to arm A because the stored support is untouched.
        return [self._global] * self.n_bins

    def bank_for(self, bin_id: int) -> CADICPatchCoresetV1:
        return self.bin_banks()[bin_id]

    # --------------------------------------------------------------- scoring

    @torch.no_grad()
    def score(
        self,
        query_patches: torch.Tensor,
        *,
        scoring: str,
        b: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(image_scores, pixel_scores)` for a `[B, N, D]` batch.

        Pixel scores are always the distance to the nearest permitted bank
        entry. Image scores use the reference support-density weighting on the
        worst patch, computed against the same permitted bank entry set.
        """
        if scoring not in (SCORING_GLOBAL, SCORING_LOCAL):
            raise ValueError(f"unknown scoring {scoring!r}")
        q = query_patches.detach().to(self.device, dtype=torch.float32)
        if q.ndim != 3 or q.shape[1] != PATCHES or q.shape[2] != self.dim:
            raise ValueError(f"expected [B,{PATCHES},{self.dim}], got {tuple(q.shape)}")
        b = self.image_neighbors if b is None else b
        combined = self._combined_bank()
        owners = self._combined_owner

        all_pixel: list[torch.Tensor] = []
        all_indices: list[torch.Tensor] = []
        all_image: list[torch.Tensor] = []
        for image in q:
            if scoring == SCORING_GLOBAL:
                pixel, indices = self._nearest_global(image, combined)
                origin = None
            else:
                pixel, indices, origin = self._nearest_local(image)
            star_row = int(torch.argmax(pixel).item())
            star_score = pixel[star_row]
            if scoring == SCORING_GLOBAL:
                # `indices[star_row]` is a row of the searched matrix, which is
                # what `owners` is indexed by. `star_row` itself is a patch
                # index and is not a valid bank row.
                matched_row = int(indices[star_row].item())
                owner = int(owners[matched_row]) if owners is not None else 0
                bank, local = self._support_bank(matched_row, owner)
            else:
                # LOCAL scoring already reports the exact bank row it matched,
                # so no index arithmetic is needed here.
                bin_id, local = origin[star_row]
                bank = self.bin_banks()[bin_id]
            if bank.count:
                c_star = bank.features[local]
                support_indices = bank._topk_indices(c_star, min(b, bank.count))
                support = bank.features[support_indices]
                support_dist = torch.linalg.vector_norm(support - image[star_row], dim=1)
                log_den = torch.logsumexp(support_dist, dim=0)
                weight = 1.0 - torch.exp(star_score - log_den)
            else:
                # No exemplar was ever admitted for this position. The patch
                # distance is already infinite; keep the weight finite and
                # record the degenerate case through occupancy diagnostics.
                weight = torch.ones_like(star_score)
            all_pixel.append(pixel)
            all_indices.append(indices)
            all_image.append(weight * star_score)
        return torch.stack(all_image), torch.stack(all_pixel)

    def _support_bank(self, matched_row: int, owner: int) -> tuple[CADICPatchCoresetV1, int]:
        """Bank holding a matched entry plus that entry's local row index.

        `matched_row` is a row in whatever feature matrix the scorer searched,
        not a patch index. Under global allocation that matrix is the single
        bank, so the row index is already local. Under spatial allocation a
        global search returns a row of the concatenation, which must be mapped
        back to the owning bin before the support neighbours can be read.
        """
        if self.allocation == ALLOCATION_GLOBAL:
            return self._global, matched_row
        return self._banks[owner], matched_row - self._owner_offset[owner]

    @property
    def _owner_offset(self) -> list[int]:
        offsets, running = [], 0
        for bank in self._banks:
            offsets.append(running)
            running += int(bank.count)
        return offsets

    def _combined_bank(self) -> torch.Tensor:
        """Feature matrix that a GLOBAL scorer searches.

        Global allocation exposes its single bank unchanged. Spatial allocation
        exposes the concatenation of all bin banks, so arm C differs from arm A
        only in *where capacity was spent*, never in how many candidates a query
        patch may match.
        """
        if self.allocation == ALLOCATION_GLOBAL:
            self._combined_owner = None
            return self._global.features
        if not self.count:
            self._combined_owner = None
            return self._global.features
        self._combined_owner = [
            bin_id
            for bin_id, bank in enumerate(self._banks)
            for _ in range(int(bank.count))
        ]
        return torch.cat([bank.features for bank in self._banks if bank.count], dim=0)

    def _nearest_global(
        self, image: torch.Tensor, bank_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._global._nearest(image, bank_features)

    def _nearest_local(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        """Nearest match restricted to exemplars of the patch's own bin.

        `origin[patch]` records `(bin_id, bank_row)` so the support neighbours
        can afterwards be read from the exact same bank. Under global allocation
        the restriction is a mask over one shared bank; under spatial allocation
        it is simply the bin's own bank.
        """
        pixel = torch.empty(PATCHES, dtype=torch.float32, device=self.device)
        local_index = torch.zeros(PATCHES, dtype=torch.long, device=self.device)
        origin: list[tuple[int, int]] = [(0, 0)] * PATCHES
        banks = self.bin_banks()
        bank_bin = self._bank_bin

        for bin_id in range(self.n_bins):
            selector = self._bins == bin_id
            bank = banks[bin_id]
            patches_in_bin = torch.nonzero(selector, as_tuple=False).reshape(-1).tolist()

            if self.allocation == ALLOCATION_GLOBAL:
                # A mask over the single shared bank: the stored support is
                # untouched, only the permitted candidate set shrinks.
                rows = torch.nonzero(bank_bin == bin_id, as_tuple=False).reshape(-1)
                features = bank.features.index_select(0, rows)
            else:
                # The bin's own bank already contains exactly its exemplars.
                rows = None
                features = bank.features

            if features.shape[0] == 0:
                pixel[selector] = float("inf")
                local_index[selector] = 0
                continue

            part_pixel, part_indices = bank._nearest(image[selector], features)
            pixel[selector] = part_pixel
            local_index[selector] = part_indices
            for position, patch in enumerate(patches_in_bin):
                chosen = int(part_indices[position].item())
                bank_row = chosen if rows is None else int(rows[chosen].item())
                origin[patch] = (bin_id, bank_row)
        return pixel, local_index, origin

    # ----------------------------------------------------------- diagnostics

    def occupancy(self) -> dict[str, Any]:
        """Per-bin capacity, fill, replacement and eviction accounting."""
        banks = self.bin_banks()
        rows = []
        for bin_id in range(self.n_bins):
            bank = banks[bin_id]
            rows.append(
                dict(
                    bin_id=bin_id,
                    capacity=int(bank.config.budget),
                    count=int(bank.count),
                    fill_ratio=bank.count / max(1, bank.config.budget),
                    inserts=int(self._insert_counts[bin_id]),
                    seen=int(bank.seen_features),
                    replaced=int(bank.replaced_features),
                    rejected=int(bank.rejected_features),
                    accepted=int(bank.accepted_features),
                )
            )
        return {
            "allocation": self.allocation,
            "grid": self.grid,
            "n_bins": self.n_bins,
            "total_budget": self.budget,
            "total_count": self.count,
            "bins": rows,
            "min_fill_ratio": min(r["fill_ratio"] for r in rows),
            "max_fill_ratio": max(r["fill_ratio"] for r in rows),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "ad01_normal_support_v1",
            "budget": self.budget,
            "grid": self.grid,
            "allocation": self.allocation,
            "dim": self.dim,
            "bins": [bank.state_dict() for bank in self._banks],
            "insert_counts": list(self._insert_counts),
            "provenance": None if self._provenance is None else self._provenance.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema") != "ad01_normal_support_v1":
            raise ValueError(f"unexpected schema {state.get('schema')!r}")
        if int(state["budget"]) != self.budget or str(state["allocation"]) != self.allocation:
            raise ValueError("state does not match this memory configuration")
        banks = state["bins"]
        if len(banks) != len(self._banks):
            raise ValueError("bank count mismatch")
        for bank, payload in zip(self._banks, banks):
            bank.load_state_dict(payload)
        self._insert_counts = [int(v) for v in state.get("insert_counts", self._insert_counts)]
        provenance = state.get("provenance")
        if self._provenance is not None:
            if provenance is None:
                raise ValueError("global-allocation state is missing provenance rows")
            self._provenance.load_state_dict(provenance)
            self._bank_bin = self._provenance.features[:, 0].clone()


def arm_definitions() -> Sequence[dict[str, str]]:
    """The four predeclared arms. Declared before any endpoint evaluation."""
    return (
        {"arm": "A", "allocation": ALLOCATION_GLOBAL, "scoring": SCORING_GLOBAL,
         "role": "CADIC reference under matched budget"},
        {"arm": "B", "allocation": ALLOCATION_GLOBAL, "scoring": SCORING_LOCAL,
         "role": "scoring-locality effect at fixed stored support"},
        {"arm": "C", "allocation": ALLOCATION_SPATIAL, "scoring": SCORING_GLOBAL,
         "role": "allocation-locality effect at fixed scorer"},
        {"arm": "D", "allocation": ALLOCATION_SPATIAL, "scoring": SCORING_LOCAL,
         "role": "joint scoring and allocation locality"},
    )


def arm(*, allocation: str, scoring: str, **kwargs: Any) -> NormalSupportMemory:
    return NormalSupportMemory(allocation=allocation, **kwargs), scoring
