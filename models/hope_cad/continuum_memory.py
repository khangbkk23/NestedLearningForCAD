# models/hope_cad/continuum_memory.py
"""Sequential, scheduled MLP memories with explicit image-event commits.
The MLP geometry, image clock, and plain-gradient update are project choices.
The CMS equations leave the task objective and optimizer error term open.
"""
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F

Objective = Callable[[int, torch.Tensor, torch.Tensor, Mapping[str, Any] | None], torch.Tensor]
_WEIGHT_NAMES = ("fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias")
_SCHEMA_VERSION = 1

@dataclass(frozen=True)
class CMSCommitResult:
    """Pre-event representation and compact evidence of one committed image."""
    output: torch.Tensor
    event_id: int
    due_levels: tuple[int, ...]
    objective_values: tuple[float, ...]
    pending_counts: tuple[int, ...]
    update_counts: tuple[int, ...]

class _CMSLevel(nn.Module):
    """Residual two-layer MLP with slow initial parameters and fast buffers."""
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        shapes = {
            "fc1_weight": (hidden_dim, dim),
            "fc1_bias": (hidden_dim,),
            "fc2_weight": (dim, hidden_dim),
            "fc2_bias": (dim,),
        }
        for name, shape in shapes.items():
            initial = nn.Parameter(torch.empty(shape))
            if name.endswith("weight"):
                nn.init.xavier_uniform_(initial)
            else:
                nn.init.zeros_(initial)
            self.register_parameter(f"initial_{name}", initial)
            self.register_buffer(name, initial.detach().clone())
            self.register_buffer(f"grad_accum_{name}", torch.zeros_like(initial))
        self.register_buffer("pending_count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("update_count", torch.zeros((), dtype=torch.int64))

    def current_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(getattr(self, name) for name in _WEIGHT_NAMES)

    def gradient_accumulators(self) -> tuple[torch.Tensor, ...]:
        return tuple(getattr(self, f"grad_accum_{name}") for name in _WEIGHT_NAMES)

    @staticmethod
    def evaluate(x: torch.Tensor, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        w1, b1, w2, b2 = tensors
        return x + F.linear(F.gelu(F.linear(x, w1, b1)), w2, b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.evaluate(x, self.current_tensors())

    def objective_gradient(
        self,
        level_index: int,
        level_input: torch.Tensor,
        objective: Objective,
        metadata: Mapping[str, Any] | None,
    ) -> tuple[float, tuple[torch.Tensor, ...]]:
        """Evaluate a local objective without writing persistent gradients."""
        with torch.enable_grad():
            working = tuple(t.detach().clone().requires_grad_(True) for t in self.current_tensors())
            local_input = level_input.detach().clone()
            local_output = self.evaluate(local_input, working)
            loss = objective(level_index, local_input, local_output, metadata)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise ValueError("objective must return one scalar tensor")
            if not torch.is_floating_point(loss) or not torch.isfinite(loss).item():
                raise ValueError("objective must return a finite floating-point scalar")
            if loss.requires_grad:
                raw = torch.autograd.grad(loss, working, allow_unused=True)
            else:
                raw = (None,) * len(working)
            gradients = tuple(
                torch.zeros_like(weight) if grad is None else grad.detach()
                for weight, grad in zip(working, raw)
            )
            if any(not torch.isfinite(grad).all().item() for grad in gradients):
                raise ValueError("objective produced non-finite gradients")
            return float(loss.detach().item()), gradients

    def reset_state(self) -> None:
        with torch.no_grad():
            for name in _WEIGHT_NAMES:
                getattr(self, name).copy_(getattr(self, f"initial_{name}").detach())
                getattr(self, f"grad_accum_{name}").zero_()
            self.pending_count.zero_()
            self.update_count.zero_()

class ContinuumMemorySystem(nn.Module):
    """Generic sequential CMS whose read and image-event mutation are distinct."""
    def __init__(
        self,
        dim: int,
        update_periods: Sequence[int],
        learning_rates: Sequence[float],
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(dim, bool) or not isinstance(dim, Integral) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if hidden_dim is None:
            hidden_dim = int(dim)
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, Integral) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if not isinstance(update_periods, Sequence) or isinstance(update_periods, (str, bytes)):
            raise ValueError("update_periods must be a non-empty sequence")
        periods = tuple(update_periods)
        if not periods or any(isinstance(p, bool) or not isinstance(p, Integral) or p <= 0 for p in periods):
            raise ValueError("update_periods must contain positive integers")
        if tuple(sorted(periods)) != periods:
            raise ValueError("update_periods must be nondecreasing in configured level order")
        if not isinstance(learning_rates, Sequence) or isinstance(learning_rates, (str, bytes)):
            raise ValueError("learning_rates must be a sequence with one value per level")
        rates = tuple(learning_rates)
        if len(rates) != len(periods) or any(isinstance(rate, bool) or not isinstance(rate, Real) or not isfinite(float(rate)) or rate <= 0 for rate in rates):
            raise ValueError("learning_rates must contain one positive finite value per level")

        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.K = len(periods)
        self.update_periods = tuple(int(p) for p in periods)
        self.learning_rates = tuple(float(rate) for rate in rates)
        self.levels = nn.ModuleList(_CMSLevel(self.dim, self.hidden_dim) for _ in periods)
        self.register_buffer("completed_events", torch.zeros((), dtype=torch.int64))
        self._evaluation_frozen = False

    def _validate_input(self, x: torch.Tensor, *, one_image: bool = False) -> None:
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("input must have rank 3 [B,N,D]")
        if x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] != self.dim:
            raise ValueError(f"input must have positive B,N and final dimension {self.dim}")
        if one_image and x.shape[0] != 1:
            raise ValueError("commit_image requires B == 1; use commit_batch for transport batches")
        if not torch.is_floating_point(x) or not torch.isfinite(x).all().item():
            raise ValueError("input must contain finite floating-point values")

    def _read_chain(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        values = [x]
        for level in self.levels:
            values.append(level(values[-1]))
        return tuple(values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the representation without mutating any persistent state."""
        self._validate_input(x)
        return self._read_chain(x)[-1]

    def commit_image(self, x: torch.Tensor, objectives: Sequence[Objective],metadata: Mapping[str, Any] | None = None) -> CMSCommitResult:
        if self._evaluation_frozen:
            raise RuntimeError("evaluation clone cannot commit image events")
        self._validate_input(x, one_image=True)
        if not isinstance(objectives, Sequence) or len(objectives) != len(self.levels):
            raise ValueError("objectives must contain one callable per level")
        if any(not callable(objective) for objective in objectives):
            raise ValueError("every objective must be callable")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping or None")
        if int(self.completed_events.item()) == torch.iinfo(torch.int64).max:
            raise OverflowError("completed_events exceeds int64 range")

        # All reads and local contributions precede every persistent write.
        stages = self._read_chain(x)
        event_id = int(self.completed_events.item()) + 1
        read_only_metadata = MappingProxyType(dict(metadata)) if metadata is not None else None
        prepared: list[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...] | None]] = []
        objective_values: list[float] = []
        due_levels: list[int] = []

        for index, (level, period, rate, objective) in enumerate(
            zip(self.levels, self.update_periods, self.learning_rates, objectives)
        ):
            value, gradient = level.objective_gradient(index, stages[index], objective, read_only_metadata)
            objective_values.append(value)
            accumulators = tuple(
                current + addition for current, addition in zip(level.gradient_accumulators(), gradient)
            )
            if any(not torch.isfinite(acc).all().item() for acc in accumulators):
                raise ValueError("gradient accumulation produced non-finite state")
            due = event_id % period == 0
            candidate = None
            if due:
                due_levels.append(index)
                candidate = tuple(
                    current - rate * acc for current, acc in zip(level.current_tensors(), accumulators)
                )
                if any(not torch.isfinite(t).all().item() for t in candidate):
                    raise ValueError("gradient update produced non-finite state")
            prepared.append((accumulators, candidate))

        with torch.no_grad():
            for level, (accumulators, candidate) in zip(self.levels, prepared):
                if candidate is None:
                    for current, accumulated in zip(level.gradient_accumulators(), accumulators):
                        current.copy_(accumulated)
                    level.pending_count.add_(1)
                else:
                    for current, next_value in zip(level.current_tensors(), candidate):
                        current.copy_(next_value)
                    for accumulator in level.gradient_accumulators():
                        accumulator.zero_()
                    level.pending_count.zero_()
                    level.update_count.add_(1)
            self.completed_events.fill_(event_id)

        return CMSCommitResult(
            output=stages[-1],
            event_id=event_id,
            due_levels=tuple(due_levels),
            objective_values=tuple(objective_values),
            pending_counts=tuple(int(level.pending_count.item()) for level in self.levels),
            update_counts=tuple(int(level.update_count.item()) for level in self.levels),
        )

    def commit_batch(
        self,
        x: torch.Tensor,
        objectives: Sequence[Objective],
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> tuple[CMSCommitResult, ...]:
        """Transport batch: commit each image in input order as its own event."""
        self._validate_input(x)
        if metadata is not None and len(metadata) != x.shape[0]:
            raise ValueError("metadata must have one entry per image")
        return tuple(
            self.commit_image(x[index:index + 1], objectives, None if metadata is None else metadata[index])
            for index in range(x.shape[0])
        )

    def reset_state(self) -> None:
        """Restore current memories and scheduler without new random draws."""
        for level in self.levels:
            level.reset_state()
        with torch.no_grad():
            self.completed_events.zero_()

    def clone_for_evaluation(self) -> "ContinuumMemorySystem":
        """Return an isolated read-only working copy of the current CMS state."""
        clone = deepcopy(self)
        clone._evaluation_frozen = True
        return clone

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "update_periods": self.update_periods,
            "learning_rates": self.learning_rates,
            "activation": "gelu",
            "bias": True,
            "normalization": None,
            "residual": True,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if state != self.get_extra_state():
            raise ValueError("CMS state configuration or schema does not match this module")

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        if state_dict.get("_extra_state") != self.get_extra_state():
            raise ValueError("CMS state configuration or schema does not match this module")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
