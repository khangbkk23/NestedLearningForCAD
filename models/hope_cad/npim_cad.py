# models/hope_cad/npim_cad.py
"""NPIM-CAD detector: multi-clock predictive innovation memory with a read-only readout.

EXP-NPIM-00 scope (handoff sections 3.4, 4, 5 and 8):

* three rank-64 linear predictors at image-event periods 1 / 16 / 128 for the
  hypothesis arm (the same class also represents the one-level / same-frequency
  matched controls);
* fixed, registered mixture weights over *ready* levels only;
* L2 innovation score ``mean_j (y_j - yhat_j)^2`` by default;
* optional fused-residual diagonal normalisation, disabled unless a normal-only
  correctness gate is explicitly requested;
* ``score()`` is read-only: it must not change weights, pending statistics,
  ready flags, counters or RNG state;
* ``commit_normal_image()`` is an atomic normal-only transaction with snapshot
  rollback on failure.

This module never receives category IDs, image labels, masks, task boundaries
or evaluation-specific calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from models.hope_cad.npim_context import (
    EMBED_DIM,
    GRID,
    PATCHES,
    build_keys,
    make_projection,
    positional_coordinates,
    sha256_tensor,
)
from models.hope_cad.npim_proximal import ProximalConfig, ProximalLevel
from models.hope_cad.npim_state import (
    deterministic_state_hash,
    ledger as state_ledger,
    load_state_file,
    save_state_atomic,
    serialized_state_bytes,
)

SCHEMA = "npim_00_state_v1"


class UncalibratedDetectorError(RuntimeError):
    """Raised when no level has committed yet: no calibrated normality score exists."""


@dataclass(frozen=True)
class NPIMConfig:
    d: int = EMBED_DIM
    r: int = 64
    periods: Tuple[int, int, int] = (1, 16, 128)
    lambdas: Tuple[float, float, float] = (0.1, 1.0, 10.0)
    weights: Tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    projection_seed: int = 20261021
    gamma: float = 0.0
    eps: float = 1e-8
    use_diag: bool = False
    diag_floor: float = 1e-6
    dtype: torch.dtype = torch.float32

    def validate(self) -> None:
        if self.d != EMBED_DIM:
            raise ValueError("NPIM-00 fixes d=768")
        if self.r <= 0:
            raise ValueError("r must be positive")
        if not (len(self.periods) == len(self.lambdas) == len(self.weights)) or len(self.periods) < 1:
            raise ValueError("periods, lambdas and weights must share one positive length")
        if any(int(p) <= 0 for p in self.periods):
            raise ValueError("periods must be positive image counts")
        if any(float(l) <= 0 for l in self.lambdas):
            raise ValueError("lambdas must be positive")
        if any(float(w) <= 0 for w in self.weights):
            raise ValueError("mixture weights must be positive")
        if not torch.isfinite(torch.tensor(sum(self.weights))):
            raise ValueError("invalid mixture weights")
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative")
        if not 0.0 < self.eps < 1.0:
            raise ValueError("eps must be a small positive clamp")
        if not self.diag_floor > 0:
            raise ValueError("diag_floor must be positive")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "d": int(self.d),
            "r": int(self.r),
            "periods": [int(p) for p in self.periods],
            "lambdas": [float(l) for l in self.lambdas],
            "weights": [float(w) for w in self.weights],
            "projection_seed": int(self.projection_seed),
            "gamma": float(self.gamma),
            "eps": float(self.eps),
            "use_diag": bool(self.use_diag),
            "diag_floor": float(self.diag_floor),
        }


class NPIMDetector:
    """Three-clock proximal predictive memory with an immutable scoring readout."""

    def __init__(self, config: NPIMConfig, device: str | torch.device = "cpu") -> None:
        config.validate()
        self.config = config
        self.device = torch.device(device)
        self.projection = make_projection(config.r, config.d, config.projection_seed).to(self.device)
        self.coordinates = (
            positional_coordinates(GRID, config.d).to(self.device) if config.gamma != 0.0 else None
        )
        self.levels: List[ProximalLevel] = [
            ProximalLevel(
                ProximalConfig(d=config.d, r=config.r, period=int(period), lam=float(lam), dtype=config.dtype),
                device=self.device,
            )
            for period, lam in zip(config.periods, config.lambdas)
        ]
        self.completed_events = 0
        self.variance = torch.zeros(config.d, dtype=torch.float32, device=self.device)
        self.var_count = 0

    # -- context / keys ------------------------------------------------------
    def build_keys(self, Z: torch.Tensor) -> torch.Tensor:
        """Keys from ``Z`` ONLY; ``Y`` is never accepted here."""

        if Z.ndim != 2 or tuple(Z.shape) != (PATCHES, self.config.d):
            raise ValueError(f"expected Z of shape [{PATCHES},{self.config.d}], got {tuple(Z.shape)}")
        return build_keys(
            Z,
            self.projection,
            gamma=self.config.gamma,
            coordinates=self.coordinates,
            eps=self.config.eps,
            grid=GRID,
        )

    # -- readout -------------------------------------------------------------
    def _mixture(self) -> List[Tuple[ProximalLevel, float]]:
        ready = [(level, float(w)) for level, w in zip(self.levels, self.config.weights) if level.ready]
        if not ready:
            return []
        total = sum(weight for _, weight in ready)
        return [(level, weight / total) for level, weight in ready]

    @torch.no_grad()
    def predict(self, K: torch.Tensor) -> torch.Tensor | None:
        """Fixed normalised mixture over ready levels, or ``None`` if uncalibrated."""

        _check_keys(K, self.config.r)
        mixture = self._mixture()
        if not mixture:
            return None
        output = torch.zeros(K.shape[0], self.config.d, dtype=K.dtype, device=K.device)
        for level, weight in mixture:
            output = output + weight * (K @ level.W.T)
        if not torch.isfinite(output).all():
            raise RuntimeError("non-finite fused prediction")
        return output

    @torch.no_grad()
    def score_keys(self, K: torch.Tensor, Y: torch.Tensor) -> Dict[str, Any]:
        """Patch/image innovation scores for one image; pure readout."""

        _check_keys(K, self.config.r)
        _check_targets(Y, self.config.d, K.shape[0])
        prediction = self.predict(K)
        if prediction is None:
            raise UncalibratedDetectorError(
                "no level is ready; the zero predictor is not a calibrated normality model"
            )
        residual = Y - prediction
        if self.config.use_diag:
            if self.var_count <= 0:
                raise UncalibratedDetectorError("diagonal readout requested before normal calibration")
            denominator = self.variance.clamp_min(self.config.diag_floor)
            patch_scores = (residual.square() / denominator).mean(dim=-1)
            score_type = "diag"
        else:
            patch_scores = residual.square().mean(dim=-1)
            score_type = "l2"
        if not torch.isfinite(patch_scores).all():
            raise RuntimeError("non-finite innovation score")
        return {
            "patch_scores": patch_scores,
            "image_score": float(patch_scores.max().item()),
            "residual_l2": float(residual.norm(dim=-1).mean().item()),
            "score_type": score_type,
        }

    @torch.no_grad()
    def score(self, Z: torch.Tensor, Y: torch.Tensor) -> Dict[str, Any]:
        """Read-only scoring; verifies that no persistent state or RNG moved."""

        from models.hope_cad.npim_state import rng_fingerprint

        before_state = self.state_hash()
        before_rng = rng_fingerprint()
        K = self.build_keys(Z)
        output = self.score_keys(K, Y)
        if self.state_hash() != before_state:
            raise RuntimeError("scoring mutated persistent NPIM state")
        if rng_fingerprint() != before_rng:
            raise RuntimeError("scoring mutated the RNG state")
        return output

    # -- normal-only transaction --------------------------------------------
    @torch.no_grad()
    def commit_keys(
        self,
        K: torch.Tensor,
        Y: torch.Tensor,
        writer_identity: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Atomically consume one completed normal image.

        Candidate boundary solves are prepared for all levels first; persistent
        state is mutated only after every candidate and finiteness check
        passes.  Any exception restores the exact pre-event snapshot.
        """

        _check_keys(K, self.config.r)
        _check_targets(Y, self.config.d, K.shape[0])
        snapshot = self.state_dict()
        try:
            next_event = self.completed_events + 1
            pre_event_prediction = self.predict(K) if any(level.ready for level in self.levels) else None
            staged_variance = (
                (Y - pre_event_prediction).square().mean(dim=0)
                if pre_event_prediction is not None
                else None
            )
            prepared: List[Tuple[ProximalLevel, torch.Tensor, torch.Tensor, int, bool, torch.Tensor | None]] = []
            for level in self.levels:
                delta_kk, delta_yk, pairs = level.stage_image(K, Y)
                accumulated_kk = level.Skk + delta_kk
                accumulated_yk = level.Syk + delta_yk
                accumulated_pairs = level.pair_count + pairs
                due = (next_event % level.config.period == 0)
                candidate = (
                    level.candidate_on_boundary(accumulated_kk, accumulated_yk, accumulated_pairs)
                    if due
                    else None
                )
                prepared.append((level, accumulated_kk, accumulated_yk, accumulated_pairs, due, candidate))

            # Commit points: no I/O and no throwing user code between these writes.
            for level, accumulated_kk, accumulated_yk, accumulated_pairs, due, candidate in prepared:
                if due:
                    if candidate is None:  # pragma: no cover - defensive
                        raise RuntimeError("due level produced no candidate")
                    level.apply_commit(candidate)
                else:
                    level.stage_pending(accumulated_kk, accumulated_yk, accumulated_pairs)

            if self.config.use_diag and staged_variance is not None:
                self.var_count += 1
                self.variance.add_((staged_variance - self.variance) / float(self.var_count))
            self.completed_events = next_event
            self._validate_live_state()
        except Exception:
            self.load_state_dict(snapshot)
            raise
        return {
            "event": int(self.completed_events),
            "writer_identity": dict(writer_identity or {}),
            "due_levels": [int(i) for i, (_, _, _, _, due, _) in enumerate(prepared) if due],
            "ready": [bool(level.ready) for level in self.levels],
            "update_counts": [int(level.update_count) for level in self.levels],
            "pending_images": [int(level.pending_images) for level in self.levels],
            "pair_counts": [int(level.pair_count) for level in self.levels],
            "state_hash": self.state_hash(),
        }

    @torch.no_grad()
    def commit_normal_image(
        self, Z: torch.Tensor, Y: torch.Tensor, writer_identity: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        return self.commit_keys(self.build_keys(Z), Y, writer_identity=writer_identity)

    # -- persistence ---------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "config": self.config.as_dict(),
            "projection": self.projection.detach().to("cpu", copy=True),
            "levels": [level.state_dict() for level in self.levels],
            "completed_events": int(self.completed_events),
            "variance": self.variance.detach().to("cpu", copy=True),
            "var_count": int(self.var_count),
        }

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        if state.get("schema") != SCHEMA:
            raise ValueError(f"state schema mismatch: {state.get('schema')!r}")
        if state.get("config") != self.config.as_dict():
            raise ValueError("persisted NPIM configuration differs from the target detector")
        if tuple(state["projection"].shape) != (self.config.r, self.config.d):
            raise ValueError("projection geometry mismatch")
        if not torch.isfinite(state["projection"]).all():
            raise ValueError("non-finite projection")
        if int(state.get("var_count", 0)) < 0:
            raise ValueError("invalid variance count")
        if tuple(state["variance"].shape) != (self.config.d,):
            raise ValueError("variance geometry mismatch")
        self.projection.copy_(state["projection"].to(device=self.device, dtype=self.config.dtype))
        if len(state["levels"]) != len(self.levels):
            raise ValueError("level count mismatch")
        for level, saved in zip(self.levels, state["levels"]):
            level.load_state_dict(saved)
        self.completed_events = int(state["completed_events"])
        self.variance.copy_(state["variance"].to(device=self.device, dtype=torch.float32))
        self.var_count = int(state["var_count"])

    def state_hash(self) -> str:
        return deterministic_state_hash(self.state_dict())

    def _validate_live_state(self) -> None:
        """Check the live persistent tensors without allocating CPU clones."""

        for level in self.levels:
            for name in ("W", "Skk", "Syk"):
                if not torch.isfinite(getattr(level, name)).all():
                    raise RuntimeError(f"non-finite {name} after commit")
            if level.pair_count < 0 or level.pending_images < 0 or level.update_count < 0:
                raise RuntimeError("negative NPIM counter")
        if not torch.isfinite(self.variance).all() or not torch.isfinite(self.projection).all():
            raise RuntimeError("non-finite variance or projection")

    def ledger(self) -> Dict[str, Any]:
        report = state_ledger(self.state_dict())
        report["serialized_state_bytes"] = serialized_state_bytes(self.state_dict())
        report["online_tensor_bytes"] = int(self.projection.numel() * self.projection.element_size()) + sum(
            sum(level.tensor_bytes().values()) for level in self.levels
        )
        report["counters"] = {
            "completed_events": int(self.completed_events),
            "ready": [bool(level.ready) for level in self.levels],
            "update_counts": [int(level.update_count) for level in self.levels],
            "pending_images": [int(level.pending_images) for level in self.levels],
            "pair_counts": [int(level.pair_count) for level in self.levels],
            "var_count": int(self.var_count),
        }
        return report

    def summary(self) -> Dict[str, Any]:
        return {
            "completed_events": int(self.completed_events),
            "ready": [bool(level.ready) for level in self.levels],
            "update_counts": [int(level.update_count) for level in self.levels],
            "pending_images": [int(level.pending_images) for level in self.levels],
            "pair_counts": [int(level.pair_count) for level in self.levels],
            "state_hash": self.state_hash(),
            "projection_sha256": sha256_tensor(self.projection),
        }

    # -- explicit persistence helpers ---------------------------------------
    def save(self, path: str) -> Dict[str, Any]:
        return save_state_atomic(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(load_state_file(path))


def _check_keys(K: torch.Tensor, r: int) -> None:
    if K.ndim != 2 or K.shape[1] != r:
        raise ValueError(f"K must be [n,{r}], got {tuple(K.shape)}")
    if not torch.isfinite(K).all():
        raise ValueError("non-finite keys")


def _check_targets(Y: torch.Tensor, d: int, n: int) -> None:
    if Y.ndim != 2 or Y.shape[0] != n or Y.shape[1] != d:
        raise ValueError(f"Y must be [{n},{d}], got {tuple(Y.shape)}")
    if not torch.isfinite(Y).all():
        raise ValueError("non-finite targets")


def native_map(patch_scores: torch.Tensor, hw: Sequence[int]) -> torch.Tensor:
    """Exact P1 geometry: 28x28 patch grid -> native bilinear map (FP32)."""

    if not isinstance(patch_scores, torch.Tensor) or tuple(patch_scores.shape) != (PATCHES,):
        raise ValueError("native_map expects 784 patch scores")
    if not torch.isfinite(patch_scores).all():
        raise ValueError("non-finite patch scores")
    if len(hw) != 2 or min(int(hw[0]), int(hw[1])) <= 0:
        raise ValueError("invalid native geometry")
    return F.interpolate(
        patch_scores.float().reshape(1, 1, GRID, GRID),
        size=(int(hw[0]), int(hw[1])),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )[0, 0]


def image_score(patch_scores: torch.Tensor) -> float:
    if not isinstance(patch_scores, torch.Tensor) or tuple(patch_scores.shape) != (PATCHES,):
        raise ValueError("image_score expects 784 patch scores")
    if not torch.isfinite(patch_scores).all():
        raise ValueError("non-finite patch scores")
    return float(patch_scores.max().item())
