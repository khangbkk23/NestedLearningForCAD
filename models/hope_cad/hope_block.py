# models/hope_cad/hope_block.py
"""Thin, transactional composition of the locked SMT and CMS cores."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

import torch
from torch import nn

from .continuum_memory import CMSCommitResult, ContinuumMemorySystem, Objective
from .self_modifying_titans import SMTProjectionResult, SelfModifyingTitans
from .state import persistent_tensor_bytes, validate_finite_state


_SCHEMA_VERSION = 1
_CMS_DEFAULT_PERIODS = (1, 8)


@dataclass(frozen=True)
class HOPECommitResult:
    """Compact result from one committed normal image."""

    output: torch.Tensor
    smt_representation: torch.Tensor
    event_id: int
    due_levels: tuple[int, ...]
    objective_values: tuple[float, ...]
    pending_counts: tuple[int, ...]
    update_counts: tuple[int, ...]
    eta: torch.Tensor
    alpha: torch.Tensor
    cms_level_outputs: tuple[torch.Tensor, ...]


class HopeBlock(nn.Module):
    """Canonical ``Self-Modifying Titans -> CMS`` representation core.

    The wrapper owns no online tensor state.  SMT owns the fast causal memory
    state and CMS owns the scheduled level state; this class only validates
    inputs, orchestrates their calls, and provides atomic image commits.
    """

    def __init__(
        self,
        dim: int,
        *,
        adaptive_q: bool = False,
        memory_chunk_size: int = 16,
        auxiliary_memory_chunk_size: int = 16,
        local_conv_kernel: int = 4,
        normalization_eps: float = 1e-8,
        cms_update_periods: Sequence[int] = _CMS_DEFAULT_PERIODS,
        cms_learning_rates: Sequence[float] | None = None,
        cms_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if cms_learning_rates is None:
            cms_learning_rates = tuple(1e-3 for _ in cms_update_periods)
        periods = tuple(cms_update_periods)
        rates = tuple(cms_learning_rates)
        self.smt = SelfModifyingTitans(
            dim,
            adaptive_q=adaptive_q,
            memory_chunk_size=memory_chunk_size,
            auxiliary_memory_chunk_size=auxiliary_memory_chunk_size,
            local_conv_kernel=local_conv_kernel,
            normalization_eps=normalization_eps,
        )
        self.cms = ContinuumMemorySystem(
            dim,
            periods,
            rates,
            hidden_dim=cms_hidden_dim,
        )
        self.dim = int(dim)
        self._evaluation_frozen = False

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "dim": self.dim,
            "smt": self.smt.get_extra_state(),
            "cms": self.cms.get_extra_state(),
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if state != self.get_extra_state():
            raise ValueError("HOPE state configuration or schema does not match this module")

    def _validate_input(self, x: torch.Tensor, *, one_image: bool = False) -> None:
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("input must have rank 3 [B,N,D]")
        if x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] != self.dim:
            raise ValueError(f"input must have positive B,N and final dimension {self.dim}")
        if one_image and x.shape[0] != 1:
            raise ValueError("one image is required; use commit_batch for transport batches")
        if not torch.is_floating_point(x):
            raise TypeError("input must have a floating-point dtype")
        if not torch.isfinite(x).all().item():
            raise ValueError("input must contain only finite values")
        reference = next(self.parameters(), None)
        if reference is not None:
            if x.device != reference.device:
                raise ValueError("input device does not match the HOPE module")
            if x.dtype != reference.dtype:
                raise ValueError("input dtype does not match the HOPE module")

    @staticmethod
    def _validate_objectives(objectives: Sequence[Objective], levels: int) -> None:
        if not isinstance(objectives, Sequence) or isinstance(objectives, (str, bytes)):
            raise ValueError("objectives must be a sequence with one callable per CMS level")
        if len(objectives) != levels:
            raise ValueError("objectives must contain one callable per CMS level")
        if any(not callable(objective) for objective in objectives):
            raise ValueError("every CMS objective must be callable")

    @staticmethod
    def _validate_metadata(metadata: Mapping[str, Any] | None) -> None:
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping or None")

    @staticmethod
    def _representation(result: SMTProjectionResult, shape: torch.Size) -> torch.Tensor:
        representation = result.memory_prediction
        if not isinstance(representation, torch.Tensor) or representation.shape != shape:
            raise ValueError("SMT did not return a [B,N,D] memory prediction")
        if not torch.isfinite(representation).all().item():
            raise ValueError("SMT returned a non-finite memory prediction")
        return representation

    def _snapshot_online_state(self) -> dict[str, Any]:
        smt = {
            "memories": {
                name: memory.weight.detach().clone()
                for name, memory in self.smt.memories.items()
            },
            "memory_update_count": self.smt.memory_update_count.detach().clone(),
            "auxiliary_update_count": self.smt.auxiliary_update_count.detach().clone(),
            "online_update_count": self.smt.online_update_count.detach().clone(),
        }
        cms_levels = []
        for level in self.cms.levels:
            values: dict[str, torch.Tensor] = {}
            for name in (
                "fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias",
                "grad_accum_fc1_weight", "grad_accum_fc1_bias",
                "grad_accum_fc2_weight", "grad_accum_fc2_bias",
                "pending_count", "update_count",
            ):
                values[name] = getattr(level, name).detach().clone()
            cms_levels.append(values)
        return {
            "smt": smt,
            "cms": {
                "levels": cms_levels,
                "completed_events": self.cms.completed_events.detach().clone(),
            },
        }

    def snapshot_online_state(self) -> dict[str, Any]:
        """Return an independent snapshot of mutable continuation state."""
        return self._snapshot_online_state()

    def _restore_online_state(self, snapshot: Mapping[str, Any]) -> None:
        with torch.no_grad():
            smt_snapshot = snapshot["smt"]
            for name, memory in self.smt.memories.items():
                memory.weight.copy_(smt_snapshot["memories"][name])
            for name in (
                "memory_update_count", "auxiliary_update_count", "online_update_count"
            ):
                getattr(self.smt, name).copy_(smt_snapshot[name])
            cms_snapshot = snapshot["cms"]
            for level, saved in zip(self.cms.levels, cms_snapshot["levels"]):
                for name, value in saved.items():
                    getattr(level, name).copy_(value)
            self.cms.completed_events.copy_(cms_snapshot["completed_events"])

    def restore_online_state(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`snapshot_online_state`."""
        self._restore_online_state(snapshot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read the complete SMT-to-CMS chain without mutating state."""
        self._validate_input(x)
        smt_result = self.smt.forward(x, update=False)
        representation = self._representation(smt_result, x.shape)
        output = self.cms.forward(representation)
        if output.shape != x.shape or not torch.isfinite(output).all().item():
            raise ValueError("CMS returned an invalid HOPE representation")
        return output

    def commit_image(
        self,
        x: torch.Tensor,
        objectives: Sequence[Objective],
        metadata: Mapping[str, Any] | None = None,
    ) -> HOPECommitResult:
        """Commit exactly one normal image atomically across SMT and CMS."""
        if self._evaluation_frozen:
            raise RuntimeError("evaluation clone cannot commit image events")
        self._validate_input(x, one_image=True)
        self._validate_objectives(objectives, self.cms.K)
        self._validate_metadata(metadata)
        before = self._snapshot_online_state()
        try:
            smt_result = self.smt.forward(x, update=True)
            representation = self._representation(smt_result, x.shape)
            cms_inspection = self.cms.inspect(representation)
            cms_result = self.cms.commit_image(representation, objectives, metadata)
            output = cms_result.output
            if output.shape != x.shape or not torch.isfinite(output).all().item():
                raise ValueError("CMS returned an invalid HOPE representation")
            return HOPECommitResult(
                output=output.detach(),
                smt_representation=representation.detach(),
                event_id=cms_result.event_id,
                due_levels=cms_result.due_levels,
                objective_values=cms_result.objective_values,
                pending_counts=cms_result.pending_counts,
                update_counts=cms_result.update_counts,
                eta=smt_result.eta.detach(),
                alpha=smt_result.alpha.detach(),
                cms_level_outputs=tuple(value.detach() for value in cms_inspection.level_outputs),
            )
        except Exception:
            self._restore_online_state(before)
            raise

    def commit_batch(
        self,
        x: torch.Tensor,
        objectives: Sequence[Objective],
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> tuple[HOPECommitResult, ...]:
        """Process a transport batch as ordered singleton image events."""
        self._validate_input(x)
        self._validate_objectives(objectives, self.cms.K)
        if metadata is not None:
            if isinstance(metadata, (str, bytes)) or len(metadata) != x.shape[0]:
                raise ValueError("metadata must contain one entry per image")
            for item in metadata:
                self._validate_metadata(item)
        return tuple(
            self.commit_image(
                x[index:index + 1],
                objectives,
                None if metadata is None else metadata[index],
            )
            for index in range(x.shape[0])
        )

    def clone_for_evaluation(self) -> "HopeBlock":
        """Return a fully isolated working copy for one test image."""
        clone = deepcopy(self)
        clone._evaluation_frozen = True
        clone.cms._evaluation_frozen = True
        return clone

    def evaluate_image(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt SMT causally in an isolated clone while CMS remains frozen."""
        self._validate_input(x, one_image=True)
        working = self.clone_for_evaluation()
        with torch.no_grad():
            smt_result = working.smt.forward(x, update=True)
            representation = working._representation(smt_result, x.shape)
            output = working.cms.forward(representation)
        if output.shape != x.shape or not torch.isfinite(output).all().item():
            raise ValueError("evaluation produced an invalid HOPE representation")
        return output

    def reset_state(self) -> None:
        """Reset both child online systems without reinitializing parameters."""
        self.smt.reset_state()
        self.cms.reset_state()

    def memory_stats(self) -> dict[str, int]:
        """Return tensor-byte accounting without duplicating shared storage."""
        smt_full = persistent_tensor_bytes(self.smt.state_dict())
        cms_full = persistent_tensor_bytes(self.cms.state_dict())
        online = self._snapshot_online_state()
        smt_online = persistent_tensor_bytes(online["smt"])
        cms_online = persistent_tensor_bytes(online["cms"])
        cms_static = sum(
            int(parameter.numel() * parameter.element_size())
            for name, parameter in self.cms.named_parameters()
            if name.startswith("levels.") and ".initial_" in name
        )
        cms_current = 0
        cms_accum = 0
        cms_counters = int(self.cms.completed_events.numel() * self.cms.completed_events.element_size())
        for level in self.cms.levels:
            for name in ("fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"):
                tensor = getattr(level, name)
                cms_current += int(tensor.numel() * tensor.element_size())
            for name in (
                "grad_accum_fc1_weight", "grad_accum_fc1_bias",
                "grad_accum_fc2_weight", "grad_accum_fc2_bias",
            ):
                tensor = getattr(level, name)
                cms_accum += int(tensor.numel() * tensor.element_size())
            for name in ("pending_count", "update_count"):
                tensor = getattr(level, name)
                cms_counters += int(tensor.numel() * tensor.element_size())
        return {
            "smt_full_tensor_bytes": smt_full,
            "smt_online_tensor_bytes": smt_online,
            "smt_static_tensor_bytes": smt_full - smt_online,
            "cms_full_tensor_bytes": cms_full,
            "cms_static_tensor_bytes": cms_static,
            "cms_current_tensor_bytes": cms_current,
            "cms_gradient_accumulator_bytes": cms_accum,
            "cms_counter_bytes": cms_counters,
            "cms_online_tensor_bytes": cms_online,
            "hope_full_tensor_bytes": smt_full + cms_full,
            "hope_online_tensor_bytes": smt_online + cms_online,
        }


__all__ = ["HOPECommitResult", "HopeBlock"]
