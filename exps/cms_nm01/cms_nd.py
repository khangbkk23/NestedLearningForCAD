# exps/cms_nm01/cms_nd.py
"""CMS-ND-01 neural density memory.

This module is experiment-specific.  The locked production HOPE modules are
not imported or modified.  Each level is an independent affine-coupling flow
whose parameters are updated from normal-only patch likelihood.  The state
contains current parameters, anchors and detached gradient accumulators.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn


DIM = 768
LATENT_DIM = 64
PATCHES = 784


@dataclass(frozen=True)
class CMSCommit:
    """Evidence from one normal image event."""

    event_id: int
    due_levels: tuple[int, ...]
    updates: tuple[int, ...]
    losses: tuple[float, ...]


def make_orthonormal_projection(
    input_dim: int = DIM,
    latent_dim: int = LATENT_DIM,
    *,
    seed: int = 2401,
) -> torch.Tensor:
    """Create a deterministic row-orthonormal projection [latent, input]."""

    if input_dim < latent_dim or input_dim < 1 or latent_dim < 1:
        raise ValueError("projection dimensions must satisfy input_dim >= latent_dim > 0")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    raw = torch.randn(input_dim, latent_dim, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.transpose(0, 1).contiguous()


class FixedProjection(nn.Module):
    """Seeded, frozen projection used by every CMS-NM-01 arm."""

    def __init__(
        self,
        input_dim: int = DIM,
        latent_dim: int = LATENT_DIM,
        *,
        seed: int = 2401,
    ) -> None:
        super().__init__()
        matrix = make_orthonormal_projection(input_dim, latent_dim, seed=seed)
        self.register_buffer("matrix", matrix)
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.seed = int(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected last dimension {self.input_dim}, got {x.shape[-1]}")
        return torch.matmul(x, self.matrix.transpose(0, 1))

    def digest(self) -> str:
        return hashlib.sha256(self.matrix.detach().cpu().numpy().tobytes()).hexdigest()

    def orthonormal_error(self) -> float:
        gram = self.matrix @ self.matrix.transpose(0, 1)
        eye = torch.eye(self.latent_dim, dtype=gram.dtype, device=gram.device)
        return float((gram - eye).abs().max().item())


class _CouplingBlock(nn.Module):
    """Affine coupling with a fixed half swap after every block."""

    def __init__(self, latent_dim: int, hidden_dim: int, scale_limit: float) -> None:
        super().__init__()
        if latent_dim % 2:
            raise ValueError("latent_dim must be even")
        half = latent_dim // 2
        self.net = nn.Sequential(
            nn.Linear(half, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.scale_limit = float(scale_limit)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left, right = x.chunk(2, dim=-1)
        raw = self.net(left)
        raw_scale, shift = raw.chunk(2, dim=-1)
        scale = self.scale_limit * torch.tanh(raw_scale / self.scale_limit)
        right_new = right * torch.exp(scale) + shift
        output = torch.cat((right_new, left), dim=-1)
        return output, scale.sum(dim=-1), scale.square().mean()


class _AffineFlow(nn.Module):
    """Small affine coupling flow and diagonal Gaussian base."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = 128,
        blocks: int = 4,
        scale_limit: float = 2.0,
    ) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("blocks must be positive")
        self.latent_dim = int(latent_dim)
        self.scale_limit = float(scale_limit)
        self.blocks = nn.ModuleList(
            _CouplingBlock(latent_dim, hidden_dim, scale_limit) for _ in range(blocks)
        )
        self.base_mean = nn.Parameter(torch.zeros(latent_dim))
        self.base_logstd = nn.Parameter(torch.zeros(latent_dim))

    def log_prob(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = x
        logdet = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
        scale_penalty = torch.zeros((), dtype=x.dtype, device=x.device)
        for block in self.blocks:
            value, block_logdet, block_penalty = block(value)
            logdet = logdet + block_logdet
            scale_penalty = scale_penalty + block_penalty
        logstd = self.base_logstd.clamp(-5.0, 5.0)
        standardized = (value - self.base_mean) * torch.exp(-logstd)
        log_base = -0.5 * (
            standardized.square() + 2.0 * logstd + math.log(2.0 * math.pi)
        ).sum(dim=-1)
        return log_base + logdet, scale_penalty / len(self.blocks), value

    def scale_saturation_fraction(self, x: torch.Tensor) -> float:
        value = x
        saturated = 0
        total = 0
        with torch.no_grad():
            for block in self.blocks:
                left, right = value.chunk(2, dim=-1)
                raw = block.net(left)
                raw_scale, shift = raw.chunk(2, dim=-1)
                scale = block.scale_limit * torch.tanh(raw_scale / block.scale_limit)
                saturated += int((scale.abs() >= 0.99 * block.scale_limit).sum().item())
                total += int(scale.numel())
                value = torch.cat((right * torch.exp(scale) + shift, left), dim=-1)
        return float(saturated / total) if total else 0.0


class _MemoryLevel(nn.Module):
    """One CMS level, including current, anchor and accumulated-gradient state."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        blocks: int,
        scale_limit: float,
        period: int,
        learning_rate: float,
        retention: float,
        max_step_norm: float,
    ) -> None:
        super().__init__()
        self.flow = _AffineFlow(latent_dim, hidden_dim, blocks, scale_limit)
        self.period = int(period)
        self.learning_rate = float(learning_rate)
        self.retention = float(retention)
        self.max_step_norm = float(max_step_norm)
        self._names = tuple(name for name, _ in self.flow.named_parameters())
        for index, (_, parameter) in enumerate(self.flow.named_parameters()):
            self.register_buffer(f"anchor_{index}", parameter.detach().clone())
            self.register_buffer(f"grad_accum_{index}", torch.zeros_like(parameter))
        self.register_buffer("pending_count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("update_count", torch.zeros((), dtype=torch.int64))

    def current_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.flow.parameters())

    def anchors(self) -> tuple[torch.Tensor, ...]:
        return tuple(getattr(self, f"anchor_{i}") for i in range(len(self._names)))

    def accumulators(self) -> tuple[torch.Tensor, ...]:
        return tuple(getattr(self, f"grad_accum_{i}") for i in range(len(self._names)))

    def accumulate(self, loss: torch.Tensor) -> None:
        loss.backward()
        with torch.no_grad():
            for parameter, accumulator in zip(self.current_parameters(), self.accumulators()):
                if parameter.grad is not None:
                    accumulator.add_(parameter.grad.detach())
                    parameter.grad = None
            self.pending_count.add_(1)

    def propose_step(self) -> tuple[torch.Tensor, ...]:
        if int(self.pending_count.item()) == 0:
            return tuple(torch.zeros_like(parameter) for parameter in self.current_parameters())
        pending = float(self.pending_count.item())
        raw_steps = tuple(
            self.learning_rate
            * (accumulator / pending + self.retention * (parameter.detach() - anchor))
            for parameter, accumulator, anchor in zip(
                self.current_parameters(), self.accumulators(), self.anchors()
            )
        )
        norm = torch.sqrt(sum(step.square().sum() for step in raw_steps))
        norm_value = float(norm.item())
        if norm_value > self.max_step_norm:
            factor = self.max_step_norm / max(norm_value, 1e-12)
            return tuple(step * factor for step in raw_steps)
        return raw_steps

    def clear_accumulators(self) -> None:
        with torch.no_grad():
            for accumulator in self.accumulators():
                accumulator.zero_()
            self.pending_count.zero_()

    def commit_step(self, step: Sequence[torch.Tensor]) -> None:
        with torch.no_grad():
            for parameter, delta in zip(self.current_parameters(), step):
                parameter.sub_(delta)
            self.clear_accumulators()
            self.update_count.add_(1)

    def copy_anchor_from(self, other: "_MemoryLevel") -> None:
        with torch.no_grad():
            for target, source in zip(self.anchors(), other.current_parameters()):
                target.copy_(source.detach())


class CMSNormalDensityMemory(nn.Module):
    """Three-clock CMS normal density memory for CMS-NM-01."""

    def __init__(
        self,
        *,
        levels: Sequence[int] = (1, 8, 64),
        projection_seed: int = 2401,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = 128,
        blocks: int = 4,
        scale_limit: float = 2.0,
        learning_rates: Sequence[float] | None = None,
        retention: Sequence[float] | None = None,
        max_step_norm: float = 0.05,
        seed: int = 7101,
    ) -> None:
        super().__init__()
        periods = tuple(int(value) for value in levels)
        if not periods or any(value < 1 for value in periods):
            raise ValueError("levels must contain positive clocks")
        if tuple(sorted(periods)) != periods:
            raise ValueError("levels must be nondecreasing")
        if learning_rates is None:
            learning_rates = tuple(1e-3 for _ in periods)
        if retention is None:
            retention = tuple(1e-3 for _ in periods)
        if len(learning_rates) != len(periods) or len(retention) != len(periods):
            raise ValueError("learning_rates and retention must match levels")
        self.periods = periods
        self.projection_seed = int(projection_seed)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.blocks = int(blocks)
        self.scale_limit = float(scale_limit)
        self.learning_rates = tuple(float(value) for value in learning_rates)
        self.retention = tuple(float(value) for value in retention)
        self.max_step_norm = float(max_step_norm)
        self.projection = FixedProjection(DIM, latent_dim, seed=projection_seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.levels = nn.ModuleList(
                _MemoryLevel(
                    latent_dim,
                    hidden_dim,
                    blocks,
                    scale_limit,
                    period,
                    rate,
                    retain,
                    max_step_norm,
                )
                for period, rate, retain in zip(periods, self.learning_rates, self.retention)
            )
        self.register_buffer("completed_events", torch.zeros((), dtype=torch.int64))
        self.register_buffer("initial_projection_rank", torch.zeros((), dtype=torch.int64))

    def project(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3 or patches.shape[-1] != DIM:
            raise ValueError("patches must have shape [B,N,768]")
        return self.projection(patches)

    def _level_log_probs(self, patches: torch.Tensor) -> tuple[torch.Tensor, ...]:
        latent = self.project(patches)
        flattened = latent.reshape(-1, self.latent_dim)
        return tuple(level.flow.log_prob(flattened)[0].reshape(latent.shape[:-1]) for level in self.levels)

    def log_probs(self, patches: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Read per-level patch log probabilities without mutating state."""
        self._validate_patches(patches)
        with torch.no_grad():
            return tuple(value.detach() for value in self._level_log_probs(patches))

    @staticmethod
    def mixture_nll_from_log_probs(log_probs: Sequence[torch.Tensor]) -> torch.Tensor:
        if not log_probs:
            raise ValueError("at least one level is required")
        stack = torch.stack(tuple(log_probs), dim=0)
        return -torch.logsumexp(stack, dim=0) + math.log(len(log_probs))

    def score_patches(self, patches: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return exact equal-weight mixture NLL and individual level NLLs."""
        log_probs = self.log_probs(patches)
        mixture = self.mixture_nll_from_log_probs(log_probs)
        return {
            "mixture_nll": mixture,
            "level_nll": tuple(-value for value in log_probs),
            "log_probs": log_probs,
        }

    def _validate_patches(self, patches: torch.Tensor) -> None:
        if not isinstance(patches, torch.Tensor) or patches.ndim != 3:
            raise ValueError("patches must have shape [B,N,768]")
        if patches.shape[0] < 1 or patches.shape[1] < 1 or patches.shape[2] != DIM:
            raise ValueError("patches must have positive B,N and dimension 768")
        if not torch.is_floating_point(patches) or not torch.isfinite(patches).all().item():
            raise ValueError("patches must be finite floating-point values")

    def _losses_with_graph(self, patches: torch.Tensor) -> tuple[torch.Tensor, ...]:
        latent = self.project(patches)
        flattened = latent.reshape(-1, self.latent_dim)
        losses = []
        for level in self.levels:
            logp, scale_penalty, _ = level.flow.log_prob(flattened)
            losses.append(-logp.mean() + 1e-5 * scale_penalty)
        return tuple(losses)

    def commit_image(self, patches: torch.Tensor) -> CMSCommit:
        """Consume one normal image and commit exactly due levels."""
        self._validate_patches(patches)
        if patches.shape[0] != 1:
            raise ValueError("commit_image requires one image")
        losses = self._losses_with_graph(patches)
        for level, loss in zip(self.levels, losses):
            level.accumulate(loss)
        event_id = int(self.completed_events.item()) + 1
        due = tuple(index for index, period in enumerate(self.periods) if event_id % period == 0)
        proposals = {index: self.levels[index].propose_step() for index in due}
        if not all(torch.isfinite(loss).item() for loss in losses) or not all(
            torch.isfinite(value).all().item()
            for level in self.levels for value in level.accumulators()
        ) or not all(torch.isfinite(value).all().item() for step in proposals.values() for value in step):
            raise FloatingPointError("non-finite normal loss, accumulated gradient or proposed step")
        with torch.no_grad():
            for index in due:
                self.levels[index].commit_step(proposals[index])
            # Anchor propagation is deliberately after all coincident commits.
            if 1 in due:
                self.levels[0].copy_anchor_from(self.levels[1])
            if 2 in due:
                self.levels[1].copy_anchor_from(self.levels[2])
            self.completed_events.fill_(event_id)
        return CMSCommit(
            event_id=event_id,
            due_levels=due,
            updates=tuple(int(level.update_count.item()) for level in self.levels),
            losses=tuple(float(loss.detach().item()) for loss in losses),
        )

    def clone_read_only(self) -> "CMSNormalDensityMemory":
        clone = CMSNormalDensityMemory(
            levels=self.periods,
            projection_seed=self.projection_seed,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            blocks=self.blocks,
            scale_limit=self.scale_limit,
            learning_rates=self.learning_rates,
            retention=self.retention,
            max_step_norm=self.max_step_norm,
        )
        clone.load_state_dict(self.state_dict())
        return clone

    def state_snapshot(self) -> dict[str, Any]:
        return {
            "state_dict": {key: value.detach().cpu().clone() for key, value in self.state_dict().items()},
            "periods": self.periods,
            "projection_seed": self.projection_seed,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "blocks": self.blocks,
            "scale_limit": self.scale_limit,
            "learning_rates": self.learning_rates,
            "retention": self.retention,
            "max_step_norm": self.max_step_norm,
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "CMSNormalDensityMemory":
        model = cls(
            levels=tuple(snapshot["periods"]),
            projection_seed=int(snapshot["projection_seed"]),
            latent_dim=int(snapshot["latent_dim"]),
            hidden_dim=int(snapshot["hidden_dim"]),
            blocks=int(snapshot["blocks"]),
            scale_limit=float(snapshot["scale_limit"]),
            learning_rates=tuple(snapshot["learning_rates"]),
            retention=tuple(snapshot["retention"]),
            max_step_norm=float(snapshot["max_step_norm"]),
        )
        model.load_state_dict(snapshot["state_dict"])
        return model

    def persistent_bytes(self) -> int:
        total = 0
        for value in self.state_dict().values():
            if torch.is_floating_point(value):
                total += int(value.numel() * value.element_size())
            elif value.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                total += int(value.numel() * value.element_size())
        return total

    def saturation_and_rank(self, patches: torch.Tensor) -> dict[str, Any]:
        self._validate_patches(patches)
        with torch.no_grad():
            latent = self.project(patches).reshape(-1, self.latent_dim)
            centered = latent - latent.mean(dim=0, keepdim=True)
            singular = torch.linalg.svdvals(centered)
            tol = max(float(singular.max().item()) * 1e-6, 1e-12)
            rank = int((singular > tol).sum().item())
            top_energy = float((singular[0].square() / singular.square().sum().clamp_min(1e-12)).item())
            energy = singular.square() / singular.square().sum().clamp_min(1e-12)
            effective_rank = float(torch.exp(-(energy * energy.clamp_min(1e-30).log()).sum()).item())
            saturation = [level.flow.scale_saturation_fraction(latent) for level in self.levels]
            return {
                "projection_rank": rank,
                "projection_effective_rank": effective_rank,
                "projection_top1_energy": top_energy,
                "level_scale_saturation": saturation,
                "finite": bool(torch.isfinite(latent).all().item()),
            }


def mixture_nll(log_probs: Sequence[torch.Tensor]) -> torch.Tensor:
    """Functional exact equal-weight mixture NLL."""
    return CMSNormalDensityMemory.mixture_nll_from_log_probs(log_probs)

