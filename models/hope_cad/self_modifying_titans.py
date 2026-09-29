# models/hope_cad/self_modifying_titans.py
"""Causal linear self-modifying memories from Nested Learning Eq. 83–93.

The residual MLP variant is deferred. Linear states use the printed Eq. 93
recurrence, including its data-dependent retention and fixed-chunk gradients.
"""
from dataclasses import dataclass
from typing import Iterator, Mapping
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class SMTProjectionResult:
    """Current-call projections, causal representation, and distinct residuals."""
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    eta: torch.Tensor
    alpha: torch.Tensor
    memory_prediction: torch.Tensor | None = None
    memory_residual: torch.Tensor | None = None
    associative_loss_per_token: torch.Tensor | None = None
    associative_loss: torch.Tensor | None = None
    write_residual: torch.Tensor | None = None
    write_loss_per_token: torch.Tensor | None = None
    instantaneous_surprise: Mapping[str, torch.Tensor] | None = None

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self.keys():
            raise KeyError(key)
        return getattr(self, key)

    def keys(self) -> tuple[str, ...]:
        return ("q", "k", "v", "eta", "alpha")

    def values(self) -> Iterator[torch.Tensor]:
        return iter((self.q, self.k, self.v, self.eta, self.alpha))

    def items(self) -> Iterator[tuple[str, torch.Tensor]]:
        return iter(zip(self.keys(), self.values()))

@dataclass(frozen=True)
class LinearUpdateInspection:
    """Temporary one-stream chunk evidence; never serialized as online state."""
    image: int
    name: str
    positions: tuple[int, ...]
    prior_weight: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    target: torch.Tensor
    eta: torch.Tensor
    alpha: torch.Tensor
    surprise: torch.Tensor
    post_weight: torch.Tensor

@dataclass(frozen=True)
class SMTInspection:
    """Explicit debug result; normal forward does not retain the update trace."""

    result: SMTProjectionResult
    updates: tuple[LinearUpdateInspection, ...]

class _LinearMemory(nn.Module):
    """A slow initialization parameter and its full persistent fast matrix."""

    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.initial_weight = nn.Parameter(torch.empty(out_dim, dim))
        nn.init.xavier_uniform_(self.initial_weight)
        self.register_buffer("weight", self.initial_weight.detach().clone())

    def reset_state(self) -> None:
        with torch.no_grad():
            self.weight.copy_(self.initial_weight)

@dataclass(frozen=True)
class _Token:
    position: int
    k: torch.Tensor
    v: torch.Tensor
    eta: torch.Tensor
    alpha: torch.Tensor

class SelfModifyingTitans(nn.Module):
    """Linear SMT with independent memory and auxiliary pending streams.

    Dense row-major Conv1d uses window four, zero padding (left=1, right=2),
    intentionally mixing raster row boundaries. Scalar control postprocessors
    are sigmoid for both eta and alpha, an explicit project mapping.

    Fixed Wq is canonical; adaptive Mq is a paper-inferred optional path. Each
    map has a full fast matrix, never a base-plus-delta projection. Eq. 93 has
    no separate momentum state. Slow initial matrices are reset sources, not
    extra reset-only buffers. Outer-loop differentiation through online writes
    is outside this core; fixed Wq and convolution remain differentiable.
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
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        if local_conv_kernel != 4:
            raise ValueError("local_conv_kernel must be 4")
        if memory_chunk_size < 1 or auxiliary_memory_chunk_size < 1:
            raise ValueError("chunk sizes must be positive")
        if normalization_eps <= 0:
            raise ValueError("normalization_eps must be positive")
        self.dim = int(dim)
        self.adaptive_q = bool(adaptive_q)
        self.memory_chunk_size = int(memory_chunk_size)
        self.auxiliary_memory_chunk_size = int(auxiliary_memory_chunk_size)
        self.local_conv_kernel = int(local_conv_kernel)
        self.local_conv_padding = (1, 2)
        self.normalization_eps = float(normalization_eps)
        self.local_conv = nn.Conv1d(dim, dim, 4, padding=0, bias=True)
        self.base_q = None if adaptive_q else nn.Linear(dim, dim, bias=False)
        shapes = {"k": dim, "v": dim, "eta": 1, "alpha": 1, "memory": dim}
        if adaptive_q:
            shapes["q"] = dim
        self.memories = nn.ModuleDict({
            name: _LinearMemory(dim, out_dim) for name, out_dim in shapes.items()
        })
        self.auxiliary_names = tuple(name for name in shapes if name != "memory")
        for name in ("memory_update_count", "auxiliary_update_count", "online_update_count"):
            self.register_buffer(name, torch.zeros((), dtype=torch.int64))

    @staticmethod
    def chunk_spans(length: int, chunk_size: int) -> tuple[tuple[int, int], ...]:
        """Return full chunks followed by the shorter final remainder."""
        if length < 0 or chunk_size < 1:
            raise ValueError("length must be non-negative and chunk_size positive")
        return tuple(
            (start, min(start + chunk_size, length))
            for start in range(0, length, chunk_size)
        )

    def get_extra_state(self) -> dict:
        """Checkpoint structural/config metadata needed for continuation."""
        return {
            "dim": self.dim, "adaptive_q": self.adaptive_q,
            "memory_chunk_size": self.memory_chunk_size,
            "auxiliary_memory_chunk_size": self.auxiliary_memory_chunk_size,
            "normalization_eps": self.normalization_eps,
            "control_mapping": "sigmoid",
        }

    def set_extra_state(self, state: dict) -> None:
        if state["dim"] != self.dim or state["adaptive_q"] != self.adaptive_q:
            raise ValueError("checkpoint memory geometry/query path differs")
        if state["control_mapping"] != "sigmoid":
            raise ValueError("checkpoint control mapping differs")
        self.memory_chunk_size = state["memory_chunk_size"]
        self.auxiliary_memory_chunk_size = state["auxiliary_memory_chunk_size"]
        self.normalization_eps = state["normalization_eps"]

    def memory_state(self) -> dict[str, torch.Tensor]:
        """Full fast matrices keyed by memory family."""
        return {name: memory.weight for name, memory in self.memories.items()}

    def online_state(self) -> dict[str, torch.Tensor]:
        result = self.memory_state()
        for name in ("memory_update_count", "auxiliary_update_count", "online_update_count"):
            result[name] = getattr(self, name)
        return result

    def reset_state(self) -> None:
        """Reset from current slow initialization parameters without RNG use."""
        for memory in self.memories.values():
            memory.reset_state()
        with torch.no_grad():
            self.memory_update_count.zero_()
            self.auxiliary_update_count.zero_()
            self.online_update_count.zero_()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        return self.local_conv(F.pad(x.transpose(1, 2), (1, 2))).transpose(1, 2)

    def _project(self, h: torch.Tensor, weights: Mapping[str, torch.Tensor]) -> SMTProjectionResult:
        q_raw = F.linear(h, weights["q"]) if self.adaptive_q else self.base_q(h)
        return SMTProjectionResult(
            q=F.normalize(q_raw, dim=-1, eps=self.normalization_eps),
            k=F.normalize(F.linear(h, weights["k"]), dim=-1, eps=self.normalization_eps),
            v=F.linear(h, weights["v"]),
            eta=torch.sigmoid(F.linear(h, weights["eta"])),
            alpha=torch.sigmoid(F.linear(h, weights["alpha"])),
        )

    def project(self, x: torch.Tensor, *, update: bool = False) -> SMTProjectionResult:
        """Read current projections without advancing either clock."""
        del update
        weights = {name: memory.weight.detach().clone() for name, memory in self.memories.items()}
        return self._project(self.preprocess(x), weights)

    def associative_memory_loss(
        self, k: torch.Tensor, v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return prediction, raw M(k)-v residual, and mean squared L2 loss."""
        prediction = F.linear(k, self.memories["memory"].weight.detach().clone())
        residual = prediction - v
        return prediction, residual, residual.square().sum(dim=-1).mean()

    @staticmethod
    def compute_instantaneous_surprise(
        k: torch.Tensor, target: torch.Tensor, *, weight: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient of half squared L2 with fixed key/target and no .grad writes.

        Pass one token to obtain Eq. 93's momentary gradient; multiple tokens
        produce the sum of their gradients. Half squared L2 matches the
        residual outer-product explicitly printed in the primary recurrence.
        """
        with torch.enable_grad():
            working = weight.detach().clone().requires_grad_(True)
            loss = 0.5 * (F.linear(k.detach(), working) - target.detach()).square().sum()
            gradient = torch.autograd.grad(loss, working, create_graph=False)[0]
        return gradient.detach()

    def forward(self, x: torch.Tensor, update: bool = False) -> SMTProjectionResult:
        return self._run(x, update=update, capture=False).result

    def inspect(self, x: torch.Tensor, update: bool = False) -> SMTInspection:
        """Opt-in trace of actual inputs and candidates for both update clocks."""
        return self._run(x, update=update, capture=True)

    def _prepare_update(
        self, name: str, prior: torch.Tensor, pending: list[_Token],
        *, image: int, capture: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, LinearUpdateInspection | None]:
        # Targets and every gradient refer to the start of THIS stream's chunk.
        key = torch.cat([token.k for token in pending], dim=0).detach()
        value = torch.cat([token.v for token in pending], dim=0).detach()
        eta = torch.cat([token.eta for token in pending], dim=0).detach()
        alpha = torch.cat([token.alpha for token in pending], dim=0).detach()
        target = F.linear(value, prior).detach()
        candidate = prior.detach().clone()
        gradients = []
        total = torch.zeros_like(prior)
        with torch.no_grad():
            for i in range(len(pending)):
                gradient = self.compute_instantaneous_surprise(
                    key[i:i + 1], target[i:i + 1], weight=prior,
                )
                # Eq. 93: W_i = W_(i-1)(alpha I-eta kk^T)-eta G_i.
                # Rank-one algebra avoids materializing a D-by-D identity.
                candidate = (
                    alpha[i, 0] * candidate
                    - eta[i, 0] * torch.outer(candidate @ key[i], key[i])
                    - eta[i, 0] * gradient
                )
                total.add_(gradient)
                if capture:
                    gradients.append(gradient)
        if not torch.isfinite(candidate).all().item():
            raise ValueError(f"non-finite linear memory update: {name}")
        trace = None
        if capture:
            trace = LinearUpdateInspection(
                image, name, tuple(token.position for token in pending),
                prior.detach().clone(), key, value, target, eta, alpha,
                torch.stack(gradients), candidate.detach().clone(),
            )
        return candidate.detach(), total.detach(), trace

    def _run(self, x: torch.Tensor, *, update: bool, capture: bool) -> SMTInspection:
        h = self.preprocess(x)
        results = []
        traces = []
        latest_surprise = {}
        for image in range(h.shape[0]):
            memory_pending: list[_Token] = []
            auxiliary_pending: list[_Token] = []
            memory_prior = None
            auxiliary_prior = {}
            for position in range(h.shape[1]):
                if not memory_pending:
                    memory_prior = self.memories["memory"].weight.detach().clone()
                if not auxiliary_pending:
                    auxiliary_prior = {
                        name: self.memories[name].weight.detach().clone()
                        for name in self.auxiliary_names
                    }
                p = self._project(h[image:image + 1, position:position + 1], auxiliary_prior)
                output = F.linear(p.q, memory_prior)
                raw_residual = F.linear(p.k, memory_prior) - p.v
                write_residual = F.linear(p.k, memory_prior) - F.linear(p.v, memory_prior)
                results.append(SMTProjectionResult(
                    **dict(p.items()), memory_prediction=output,
                    memory_residual=raw_residual,
                    associative_loss_per_token=raw_residual.square().sum(-1),
                    write_residual=write_residual,
                    write_loss_per_token=0.5 * write_residual.square().sum(-1),
                ))
                token = _Token(position, *(getattr(p, name).detach().reshape(1, -1)
                                         for name in ("k", "v", "eta", "alpha")))
                memory_pending.append(token)
                auxiliary_pending.append(token)
                end = position + 1
                memory_boundary = end % self.memory_chunk_size == 0 or end == h.shape[1]
                auxiliary_boundary = end % self.auxiliary_memory_chunk_size == 0 or end == h.shape[1]
                candidates = {}
                if memory_boundary:
                    candidates["memory"] = self._prepare_update(
                        "memory", memory_prior, memory_pending, image=image, capture=capture,
                    )
                if auxiliary_boundary:
                    for name in self.auxiliary_names:
                        candidates[name] = self._prepare_update(
                            name, auxiliary_prior[name], auxiliary_pending, image=image, capture=capture,
                        )
                # All coincident candidates are prepared before any state write.
                for name, (candidate, surprise, trace) in candidates.items():
                    latest_surprise[name] = surprise
                    if trace is not None:
                        traces.append(trace)
                    if update:
                        with torch.no_grad():
                            self.memories[name].weight.copy_(candidate)
                if update:
                    with torch.no_grad():
                        self.memory_update_count.add_(int(memory_boundary))
                        self.auxiliary_update_count.add_(int(auxiliary_boundary))
                        self.online_update_count.add_(int(memory_boundary) + int(auxiliary_boundary))
                if memory_boundary:
                    memory_pending.clear()
                if auxiliary_boundary:
                    auxiliary_pending.clear()
        fields = {}
        for name in ("q", "k", "v", "eta", "alpha", "memory_prediction", "memory_residual",
                     "associative_loss_per_token", "write_residual", "write_loss_per_token"):
            token_values = torch.cat([getattr(result, name) for result in results], dim=1)
            fields[name] = token_values.reshape(h.shape[0], h.shape[1], *token_values.shape[2:])
        result = SMTProjectionResult(
            **fields, associative_loss=fields["associative_loss_per_token"].mean(),
            instantaneous_surprise=latest_surprise,
        )
        return SMTInspection(result, tuple(traces))

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError(f"expected x with rank 3 [B,N,D], got {tuple(x.shape)}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected final dimension {self.dim}, got {x.shape[-1]}")
        if x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError("B and N must be positive")
        if not torch.is_floating_point(x):
            raise TypeError("x must have a floating-point dtype")
        if not torch.isfinite(x).all().item():
            raise ValueError("x must contain only finite values")
