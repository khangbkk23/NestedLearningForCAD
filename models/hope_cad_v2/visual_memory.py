"""Five-map visual memory for Nested Learning for CAD, architecture decision v2 §3.

Written from the project's frozen mathematical contract / VisionHOPE v1 paper.
No upstream CUDA extension or backbone implementation is imported. Working maps
are functional values: there is no cross-image state and no implicit commit.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .scan_geometry import chunk_spans, scan_routes


@dataclass(frozen=True)
class VisualMemoryState:
    """Leading dimensions identify independent images, directions and heads."""

    content: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    eta: torch.Tensor
    alpha: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.content, self.key, self.value, self.eta, self.alpha)


@dataclass(frozen=True)
class GuardedStep:
    executed: torch.Tensor
    injection_limit: torch.Tensor
    spectral_limit: torch.Tensor


@dataclass(frozen=True)
class ChunkDiagnostics:
    start: int
    end: int
    # Detached snapshots; diagnostics never retain an outer-training graph.
    gains: torch.Tensor
    boundary_norms: torch.Tensor
    final_norms: torch.Tensor
    key_norm_squared: torch.Tensor
    alpha: torch.Tensor
    executed_eta: torch.Tensor


@dataclass(frozen=True)
class ScanResult:
    output: torch.Tensor
    final_state: VisualMemoryState
    chunks: tuple[ChunkDiagnostics, ...] = ()


@dataclass(frozen=True)
class VisualMemoryResult:
    output: torch.Tensor
    directional_outputs: torch.Tensor  # [B, 4, N, memory_dim], restored coordinates
    final_state: VisualMemoryState  # [B, 4, heads, rows, head_dim]
    # Each group batches routes with the same chunk length (all four for a square).
    scan_groups: tuple[tuple[tuple[str, ...], tuple[ChunkDiagnostics, ...]], ...]


def _finite(name: str, *tensors: torch.Tensor) -> None:
    if any(not bool(torch.isfinite(t).all()) for t in tensors):
        raise ValueError(f"non-finite {name}")


def validate_scan_inputs(z, q, initial, chunk_size) -> None:
    if z.ndim < 2 or z.shape != q.shape or any(n < 1 for n in z.shape):
        raise ValueError("z and q must have identical nonempty [..., tokens, head_dim] shapes")
    if z.dtype not in (torch.float32, torch.float64):
        raise ValueError("visual recurrence requires float32 or float64")
    if q.dtype != z.dtype or q.device != z.device:
        raise ValueError("z/q dtype and device must agree")
    chunk_spans(z.shape[-2], chunk_size)
    leading, d = z.shape[:-2], z.shape[-1]
    for index, tensor in enumerate(initial.tensors()):
        rows = d if index < 3 else 1
        if tensor.shape != (*leading, rows, d):
            raise ValueError("memory shape does not match independent scan/head dimensions")
        if tensor.dtype != z.dtype or tensor.device != z.device:
            raise ValueError("memory dtype/device must match inputs")
    _finite("scan inputs or initial maps", z, q, *initial.tensors())


def stability_matched_step(
    key: torch.Tensor, delta: torch.Tensor, raw_eta: torch.Tensor, alpha: torch.Tensor
) -> GuardedStep:
    """Spec safeguards, including finite saturation and differentiable ULP rounding.

    Inputs are already generated/validated by the scan. Controls have shape
    [..., tokens], vectors [..., tokens, d]. Backward of nextafter is the identity
    on the pre-rounding spectral limit; the clamp still differentiates normally.
    Both first- and higher-order derivatives of that continuous mapping survive.
    """
    radius = (delta.square().sum(-1) + 1e-12).sqrt() + 1e-6
    injection = (1.0 - alpha) * (1.0 - 1e-3) / radius
    # Do not evaluate eta / 0, even in the branch rejected by torch.where.
    safe_injection = torch.where(injection > 0, injection, torch.ones_like(injection))
    soft_cap = injection * (-torch.expm1(-raw_eta / safe_injection))
    key_norm_squared = key.square().sum(-1).clamp_min(1e-20)
    inverse_key_norm = torch.rsqrt(key_norm_squared)
    spectral = (2.0 * alpha * (1.0 - 1e-6)) * inverse_key_norm.square()
    rounded = torch.nextafter(spectral.detach(), torch.zeros_like(spectral))
    spectral = spectral + (rounded - spectral).detach()
    return GuardedStep(torch.minimum(soft_cap, spectral), injection, spectral)


def scan_chunked(
    z: torch.Tensor,
    q: torch.Tensor,
    initial: VisualMemoryState,
    chunk_size: int,
    *,
    capture: bool = False,
) -> ScanResult:
    """Vectorized boundary projections + rank-one shared-gain recurrence.

    Vectorizes all leading dimensions and all projections/readouts in a chunk.
    The token recurrence remains ordered. Rank-one G @ A + B avoids forming A/B
    or doing a dense d³ product at every token. Nothing is detached on the main
    path; this function supports the higher-order graph needed by W4.
    """
    validate_scan_inputs(z, q, initial, chunk_size)
    with torch.autocast(device_type=z.device.type, enabled=False):
        state = initial
        outputs, diagnostics = [], []
        d = z.shape[-1]
        identity = torch.eye(d, dtype=z.dtype, device=z.device)
        for start, end in chunk_spans(z.shape[-2], chunk_size):
            tokens, queries = z[..., start:end, :], q[..., start:end, :]
            boundary = state
            k_raw = tokens @ boundary.key.transpose(-1, -2)
            key = k_raw / (k_raw.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            value = tokens @ boundary.value.transpose(-1, -2)
            eta = 0.025 * F.softplus((tokens @ boundary.eta.transpose(-1, -2)).squeeze(-1))
            alpha = torch.sigmoid((tokens @ boundary.alpha.transpose(-1, -2)).squeeze(-1) + math.log(9.0))
            delta = key - value
            step = stability_matched_step(key, delta, eta, alpha)
            outputs.append(queries @ boundary.content.transpose(-1, -2))
            gain = identity.expand(*z.shape[:-2], d, d)
            gains = []
            for t in range(end - start):
                k, diff = key[..., t, :], delta[..., t, :]
                direction = (gain @ k.unsqueeze(-1)).squeeze(-1) + diff
                gain = (
                    alpha[..., t, None, None] * gain
                    - step.executed[..., t, None, None] * direction.unsqueeze(-1) * k.unsqueeze(-2)
                )
                if capture:
                    gains.append(gain.detach().clone())
            state = VisualMemoryState(*(matrix @ gain for matrix in boundary.tensors()))
            if capture:
                diagnostics.append(ChunkDiagnostics(
                    start, end, torch.stack(gains, dim=-3),
                    torch.stack([t.detach().norm(dim=(-2, -1)) for t in boundary.tensors()], -1),
                    torch.stack([t.detach().norm(dim=(-2, -1)) for t in state.tensors()], -1),
                    key.detach().square().sum(-1), alpha.detach().clone(), step.executed.detach().clone(),
                ))
        output = torch.cat(outputs, dim=-2)
        _finite("scan result", output, *state.tensors())
        return ScanResult(output, state, tuple(diagnostics))


class FourScanVisualMemory(nn.Module):
    """Pure per-view visual operator V; output feeds the subsequent CMS in W4/5.

    Defaults are canonical. Smaller dimensions/grids are useful for mathematical
    fixtures; they are not new scientific configurations of the frozen method.
    """

    def __init__(self, dim: int = 128, head_dim: int = 16, grid_size=(28, 28)):
        super().__init__()
        if any(type(n) is not int or n < 1 for n in (dim, head_dim)) or dim % head_dim:
            raise ValueError("dim must be a positive multiple of head_dim")
        if len(grid_size) != 2:
            raise ValueError("grid_size must have two dimensions")
        scan_routes(*grid_size)
        self.dim, self.head_dim, self.heads = dim, head_dim, dim // head_dim
        self.grid_size = tuple(grid_size)
        self.query = nn.Linear(dim, dim, bias=False)
        self.initial_content = nn.Parameter(torch.empty(4, self.heads, head_dim, head_dim))
        self.initial_key = nn.Parameter(torch.empty_like(self.initial_content))
        self.initial_value = nn.Parameter(torch.empty_like(self.initial_content))
        self.initial_eta = nn.Parameter(torch.empty(4, self.heads, 1, head_dim))
        self.initial_alpha = nn.Parameter(torch.empty_like(self.initial_eta))
        self.fusion = nn.Parameter(torch.empty(4, dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query.weight)
        for matrix in (self.initial_content, self.initial_key, self.initial_value):
            nn.init.trunc_normal_(matrix, std=0.01, a=-0.02, b=0.02)
        nn.init.zeros_(self.initial_eta)
        nn.init.zeros_(self.initial_alpha)
        nn.init.constant_(self.fusion, 0.25)

    def initial_state(self) -> VisualMemoryState:
        """Learned reset source only; not a mutable online-state API."""
        return VisualMemoryState(self.initial_content, self.initial_key, self.initial_value,
                                 self.initial_eta, self.initial_alpha)

    def get_extra_state(self):
        return {"schema": "hope_cad_v2_visual_w3_v1", "dim": self.dim,
                "head_dim": self.head_dim, "grid_size": self.grid_size}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("visual checkpoint geometry/schema mismatch")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self._run(z, capture=False).output

    def inspect(self, z: torch.Tensor) -> VisualMemoryResult:
        """Return route outputs/final working maps and detached stability diagnostics."""
        return self._run(z, capture=True)

    def _run(self, z: torch.Tensor, *, capture: bool) -> VisualMemoryResult:
        h, w = self.grid_size
        if z.ndim != 3 or z.shape[0] < 1 or z.shape[1:] != (h * w, self.dim):
            raise ValueError(f"expected [B,{h * w},{self.dim}] with nonempty B")
        if z.dtype not in (torch.float32, torch.float64) or z.dtype != self.query.weight.dtype:
            raise ValueError("input and parameters must share float32 or float64 dtype")
        if z.device != self.query.weight.device:
            raise ValueError("input and parameters must share device")
        with torch.autocast(device_type=z.device.type, enabled=False):
            routes = scan_routes(h, w, device=z.device)
            q = self.query(z).reshape(z.shape[0], h * w, self.heads, self.head_dim)
            q = F.normalize(q, dim=-1, eps=1e-6).flatten(-2)
            # Square grids batch all four routes. Rectangles need two chunk lengths.
            groups = {}
            for index, route in enumerate(routes):
                groups.setdefault(route.chunk_size, []).append(index)
            restored, final_maps, inspections = {}, {}, []
            for chunk_size, indices in groups.items():
                def route_heads(tensor):
                    ordered = torch.stack([routes[i].scan(tensor) for i in indices], dim=1)
                    return ordered.reshape(z.shape[0], len(indices), h * w, self.heads, self.head_dim).transpose(2, 3)
                initial = VisualMemoryState(*(
                    matrix[indices].unsqueeze(0).expand(z.shape[0], -1, -1, -1, -1)
                    for matrix in self.initial_state().tensors()
                ))
                result = scan_chunked(route_heads(z), route_heads(q), initial, chunk_size, capture=capture)
                for local, index in enumerate(indices):
                    tokens = result.output[:, local].transpose(1, 2).reshape(z.shape[0], h * w, self.dim)
                    restored[index] = routes[index].restore(tokens)
                    final_maps[index] = tuple(t[:, local] for t in result.final_state.tensors())
                if capture:
                    inspections.append((tuple(routes[i].name for i in indices), result.chunks))
            directional = torch.stack([restored[i] for i in range(4)], dim=1)
            output = (directional * self.fusion[None, :, None, :]).sum(dim=1)
            _finite("visual fusion result", output)
            final = VisualMemoryState(*(torch.stack([final_maps[i][m] for i in range(4)], dim=1) for m in range(5)))
            return VisualMemoryResult(output, directional, final, tuple(inspections))
