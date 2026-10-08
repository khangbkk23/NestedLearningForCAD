# exps/hope_image_synchronous_memory.py
"""Snapshot-based linear memory experiments and immutable function probes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from models.hope_cad.self_modifying_titans import SelfModifyingTitans
from exps.hope_update_stabilization import UpdateMapping, run_update_smt


EPS = 1e-12
H = 0.02
METHODS = ("P0", "P1", "P2", "FROZEN")
FP32_TOL = 256 * torch.finfo(torch.float32).eps
COUNTERS = ("memory_update_count", "auxiliary_update_count", "online_update_count")


def fingerprint(values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode())
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def synchronized_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


@dataclass(frozen=True)
class MemorySnapshot:
    weights: Mapping[str, torch.Tensor]
    counters: Mapping[str, torch.Tensor]
    event: int
    identity: str


@dataclass(frozen=True)
class PatchQuantities:
    spatial: torch.Tensor
    queries: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    gates: torch.Tensor
    coordinates: torch.Tensor
    source_identity: str

    @property
    def count(self) -> int:
        return self.keys.shape[0]


@dataclass(frozen=True)
class ImageStatistics:
    C: torch.Tensor
    D: torch.Tensor
    count: int
    source_identity: str


@dataclass(frozen=True)
class EventProposal:
    weights: Mapping[str, torch.Tensor]
    counters: Mapping[str, torch.Tensor]
    quantities: PatchQuantities
    transition: torch.Tensor | None
    source_identity: str
    event: int
    causal_output: torch.Tensor
    metrics: Mapping[str, float]
    timings: Mapping[str, float]


@torch.no_grad()
def aggregate_image_statistics(
    quantities: PatchQuantities,
    order: torch.Tensor | None = None,
    partitions: Sequence[int] | None = None,
) -> ImageStatistics:
    """Mean-normalized reduction of complete, already spatialized tuples."""
    n, dim = quantities.keys.shape
    if n < 1:
        raise ValueError("at least one valid patch is required")
    if order is None:
        order = torch.arange(n, device=quantities.keys.device)
    order = order.to(quantities.keys.device)
    if order.shape != (n,) or not torch.equal(torch.sort(order).values, torch.arange(n, device=order.device)):
        raise ValueError("order must permute every coordinate exactly once")
    sizes = tuple(partitions) if partitions is not None else (n,)
    if any(size < 1 for size in sizes) or sum(sizes) != n:
        raise ValueError("partitions must cover all valid patches")
    C = quantities.keys.new_zeros(dim, dim)
    D = torch.zeros_like(C)
    start = 0
    for size in sizes:
        ids = order[start:start + size]
        keys, values = quantities.keys[ids], quantities.values[ids]
        weighted_keys = keys * (quantities.gates[ids] / n)
        C.add_(keys.T @ weighted_keys)
        D.add_((keys - values).T @ weighted_keys)
        start += size
    return ImageStatistics(C.detach(), D.detach(), n, quantities.source_identity)


@torch.no_grad()
def propose_transition(statistics: ImageStatistics, *, controlled: bool, h: float = H):
    """Construct the full SR-DGD operator; control uses exact dense SVD."""
    identity = torch.eye(statistics.C.shape[0], device=statistics.C.device, dtype=statistics.C.dtype)
    raw = identity - h * (statistics.C + statistics.D)
    metrics: dict[str, float] = {}
    start = synchronized_time(raw.device)
    if controlled:
        # CPU LAPACK can fail to converge on the deliberately ill-conditioned
        # FP32 transition.  The controller is an exact dense-SVD oracle, so
        # evaluating that oracle in FP64 is the smallest diagnostic-only
        # robustness measure; the committed transition retains model dtype.
        control_raw = raw.double() if raw.dtype == torch.float32 else raw
        u, s, vh = torch.linalg.svd(control_raw, full_matrices=False)
        clipped_s = s.clamp(max=1.0)
        safe = ((u * clipped_s.unsqueeze(0)) @ vh).to(raw.dtype)
        metrics.update({
            "raw_operator_norm": float(s.max()),
            # The projected spectrum is the controller's exact definition of
            # ||T_safe||_2.  Re-SVD of the reconstructed FP32 matrix can exceed
            # one by a few ulps and would reject a mathematically valid cap.
            "safe_operator_norm": float(clipped_s.max()),
            "realized_safe_operator_norm": float(torch.linalg.svdvals(safe).max()),
            "clipped_fraction": float((s > 1.0).double().mean()),
            "maximum_singular_excess": float((s - 1.0).clamp(min=0).max()),
            "transition_distortion": float((safe - raw).double().norm() / ((raw - identity).double().norm() + EPS)),
            "raw_update_norm": float((raw - identity).double().norm()),
            "controlled_update_norm": float((safe - identity).double().norm()),
        })
    else:
        safe = raw
        metrics.update({"transition_distortion": 0.0, "clipped_fraction": 0.0})
    elapsed = synchronized_time(raw.device) - start
    return safe.detach(), metrics, elapsed


@torch.no_grad()
def local_objective(weight: torch.Tensor, prior: torch.Tensor, quantities: PatchQuantities) -> dict[str, float]:
    """Complete fixed local quadratic; references never follow a new state."""
    target = F.linear(quantities.values, prior)
    prediction = F.linear(quantities.keys, weight)
    w = quantities.gates.double() / quantities.count
    residual = prediction.double() - target.double()
    residual_energy = 0.5 * (residual.square() * w).sum()
    rank_energy = 0.5 * (prediction.double().square() * w).sum()
    return {
        "J": float(residual_energy + rank_energy),
        "self_target_residual": float(residual_energy),
        "fixed_association_error": float((residual.square() * w).sum() / ((target.double().square() * w).sum() + EPS)),
    }


def comparison(reference: torch.Tensor, evaluation: torch.Tensor) -> dict[str, float]:
    a, b = reference.detach().double(), evaluation.detach().double()
    norm = a.norm()
    return {
        "relative_l2": float((b - a).norm() / (norm + EPS)),
        "cosine": float((a * b).sum() / (norm * b.norm() + EPS)),
        "absolute_residual": float((b - a).norm()),
        "rms_ratio": float(b.square().mean().sqrt() / (a.square().mean().sqrt() + EPS)),
    }


@torch.no_grad()
def fixed_association_read_errors(
    weight: torch.Tensor, keys: torch.Tensor, queries: torch.Tensor, targets: torch.Tensor,
) -> dict[str, float]:
    """Separate fitting write keys from accessing associations via read queries."""
    denominator = targets.detach().double().square().sum() + EPS
    key_prediction = F.linear(keys, weight)
    query_prediction = F.linear(queries, weight)
    return {
        "key_error": float((key_prediction.double() - targets.double()).square().sum() / denominator),
        "query_error": float((query_prediction.double() - targets.double()).square().sum() / denominator),
        "key_query_cosine": float(F.cosine_similarity(keys.double(), queries.double(), dim=-1, eps=EPS).mean()),
    }


def geometry(rows: torch.Tensor, *, spectrum: bool = False) -> dict[str, float | bool]:
    x = rows.detach().double()
    centered = x - x.mean(0, keepdim=True)
    result: dict[str, float | bool] = {
        "rms": float(x.square().mean().sqrt()),
        "variance": float(centered.square().mean()),
        "norm": float(x.norm()),
        "finite": bool(torch.isfinite(x).all()),
    }
    if spectrum:
        s = torch.linalg.svdvals(centered)
        p = s / (s.sum() + EPS)
        result["rank"] = float(torch.exp(-(p * torch.log(p + EPS)).sum()))
        result["top1_energy"] = float(s[0].square() / (s.square().sum() + EPS))
    return result


@torch.no_grad()
def visual_structure(frozen: torch.Tensor, memory: torch.Tensor, initial_memory: torch.Tensor) -> dict[str, float]:
    """Correspondence is same-image same-coordinate across state interventions."""
    n = frozen.shape[0]
    ids = torch.linspace(0, n - 1, min(128, n), device=frozen.device).long()
    z = F.normalize(frozen[ids].double(), dim=-1, eps=EPS)
    m = F.normalize(memory[ids].double(), dim=-1, eps=EPS)
    init = F.normalize(initial_memory[ids].double(), dim=-1, eps=EPS)
    combined = F.normalize((frozen[ids] + memory[ids]).double(), dim=-1, eps=EPS)
    width = min(5, len(ids) - 1)
    eye = torch.eye(len(ids), device=z.device, dtype=torch.bool)
    zn = (z @ z.T).masked_fill(eye, -float("inf")).topk(width, dim=1).indices
    def overlap(features: torch.Tensor) -> float:
        nn = (features @ features.T).masked_fill(eye, -float("inf")).topk(width, dim=1).indices
        return float((zn.unsqueeze(-1) == nn.unsqueeze(-2)).any(-1).double().mean())
    return {
        "memory_neighbor_overlap": overlap(m),
        "combined_neighbor_overlap": overlap(combined),
        "initial_memory_neighbor_overlap": overlap(init),
        "memory_coordinate_retrieval": float(((m @ init.T).argmax(1) == torch.arange(len(ids), device=m.device)).double().mean()),
        "combined_coordinate_retrieval": float(((combined @ z.T).argmax(1) == torch.arange(len(ids), device=m.device)).double().mean()),
        "memory_initial_cosine": comparison(initial_memory, memory)["cosine"],
        "memory_initial_absolute_drift": comparison(initial_memory, memory)["absolute_residual"],
        "combined_frozen_cosine": comparison(frozen, frozen + memory)["cosine"],
    }


class ImageSynchronousMemory:
    """An isolated image lifecycle; only commit_event changes live state."""

    def __init__(self, initial_state: Mapping[str, Any], method: str, *, device="cpu"):
        if method not in METHODS:
            raise ValueError("unsupported memory method")
        extra = initial_state["_extra_state"]
        if extra["adaptive_q"] or extra["memory_chunk_size"] != 16 or extra["auxiliary_memory_chunk_size"] != 16:
            raise ValueError("fixed query and aligned 16-token clocks are required")
        with torch.random.fork_rng(devices=[]):
            self.smt = SelfModifyingTitans(int(initial_state["_extra_state"]["dim"]), memory_chunk_size=16, auxiliary_memory_chunk_size=16)
        self.smt.to(dtype=initial_state["memories.memory.weight"].dtype)
        self.smt.load_state_dict(deepcopy(initial_state))
        self.smt.to(device).eval()
        for parameter in self.smt.parameters():
            parameter.requires_grad_(False)
        self.method = method
        self.completed_events = torch.zeros((), dtype=torch.int64, device=device)

    def state_values(self) -> dict[str, torch.Tensor]:
        return {
            **{f"M_{name}": m.weight for name, m in self.smt.memories.items()},
            **{name: getattr(self.smt, name) for name in COUNTERS},
            "completed_events": self.completed_events,
        }

    def state_fingerprint(self) -> str:
        return fingerprint(self.state_values())

    def snapshot_state(self) -> MemorySnapshot:
        return MemorySnapshot(
            {name: m.weight.detach().clone() for name, m in self.smt.memories.items()},
            {name: getattr(self.smt, name).detach().clone() for name in COUNTERS},
            int(self.completed_events), self.state_fingerprint(),
        )

    @torch.no_grad()
    def generate_update_quantities(self, image: torch.Tensor, snapshot: MemorySnapshot) -> PatchQuantities:
        self.smt._validate_input(image)
        if image.shape[0] != 1:
            raise ValueError("one image is required")
        spatial = self.smt.preprocess(image).detach()[0]
        queries = F.normalize(self.smt.base_q(spatial), dim=-1, eps=self.smt.normalization_eps)
        return PatchQuantities(
            spatial.clone(), queries.clone(),
            F.normalize(F.linear(spatial, snapshot.weights["k"]), dim=-1, eps=self.smt.normalization_eps),
            F.linear(spatial, snapshot.weights["v"]),
            torch.sigmoid(F.linear(spatial, snapshot.weights["eta"])),
            torch.arange(spatial.shape[0], device=spatial.device), snapshot.identity,
        )

    @torch.no_grad()
    def read_from_snapshot(self, quantities: PatchQuantities, snapshot: MemorySnapshot) -> torch.Tensor:
        return F.linear(quantities.queries, snapshot.weights["memory"]).detach().clone()

    @torch.no_grad()
    def evaluate_read_only(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        snapshot = self.snapshot_state()
        quantities = self.generate_update_quantities(image, snapshot)
        memory = self.read_from_snapshot(quantities, snapshot)
        return {"frozen": image[0].detach().clone(), "memory": memory, "combined": (image[0] + memory).detach().clone()}

    @torch.no_grad()
    def propose_event(self, image: torch.Tensor, *, order: torch.Tensor | None = None) -> EventProposal:
        device = image.device
        start = synchronized_time(device)
        snapshot = self.snapshot_state()
        quantities = self.generate_update_quantities(image, snapshot)
        pre_read = self.read_from_snapshot(quantities, snapshot)
        read_end = synchronized_time(device)
        timings = {"snapshot_and_read_seconds": read_end - start, "statistics_seconds": 0.0, "controller_seconds": 0.0}
        transition = None
        metrics: dict[str, float] = {}
        if self.method == "P0":
            working = deepcopy(self.smt)
            spatial = quantities.spatial if order is None else quantities.spatial[order]
            # Spatial preprocessing is already complete on the original grid.
            working.preprocess = lambda unused: spatial.unsqueeze(0).detach().clone()
            causal, _, _ = run_update_smt(
                working, image,
                UpdateMapping("P0", eta_kind="horizon_sigmoid", eta_h=H, horizon=quantities.count),
                capture_trace=False,
            )
            if order is not None:
                causal = causal[:, torch.argsort(order)]
            weights = {name: m.weight.detach().clone() for name, m in working.memories.items()}
            counters = {name: getattr(working, name).detach().clone() for name in COUNTERS}
            causal_output = causal[0].detach().clone()
        elif self.method in ("P1", "P2"):
            stats = aggregate_image_statistics(quantities, order)
            stat_end = synchronized_time(device)
            timings["statistics_seconds"] = stat_end - read_end
            transition, metrics, controlled_seconds = propose_transition(stats, controlled=self.method == "P2")
            timings["controller_seconds"] = controlled_seconds
            weights = {name: (value @ transition).detach() for name, value in snapshot.weights.items()}
            counters = {name: value + (2 if name == "online_update_count" else 1) for name, value in snapshot.counters.items()}
            causal_output = pre_read
        else:
            weights = {name: value.clone() for name, value in snapshot.weights.items()}
            counters = {name: value.clone() for name, value in snapshot.counters.items()}
            causal_output = pre_read
        metrics["event_update_ratio"] = float((weights["memory"] - snapshot.weights["memory"]).double().norm() / (snapshot.weights["memory"].double().norm() + EPS))
        timings["proposal_seconds"] = synchronized_time(device) - start
        return EventProposal(weights, counters, quantities, transition, snapshot.identity, snapshot.event + 1, causal_output, metrics, timings)

    def validate_transition(self, proposal: EventProposal) -> None:
        if proposal.source_identity != self.state_fingerprint() or proposal.event != int(self.completed_events) + 1:
            raise ValueError("stale or unrelated image proposal")
        if set(proposal.weights) != set(self.smt.memories) or set(proposal.counters) != set(COUNTERS):
            raise ValueError("proposal schema differs")
        for name, value in proposal.weights.items():
            ref = self.smt.memories[name].weight
            if value.shape != ref.shape or value.device != ref.device or value.dtype != ref.dtype:
                raise ValueError("proposal geometry differs")
            if value.requires_grad or value.grad_fn is not None or not torch.isfinite(value).all():
                raise ValueError("proposal contains invalid memory")
            if self.method == "FROZEN" and not torch.equal(value, ref):
                raise ValueError("frozen memory proposal changes a weight")
        count = (proposal.quantities.count + 15) // 16 if self.method == "P0" else (0 if self.method == "FROZEN" else 1)
        for name, value in proposal.counters.items():
            reference = getattr(self.smt, name)
            increment = 2 * count if name == "online_update_count" else count
            if value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError("proposal counter geometry differs")
            if not torch.equal(value, reference + increment):
                raise ValueError("proposal changes the update clock")
        if not torch.isfinite(proposal.causal_output).all():
            raise ValueError("non-finite representation")
        if self.method == "P2" and proposal.metrics["safe_operator_norm"] > 1 + FP32_TOL:
            raise ValueError("controlled transition violates operator bound")

    @torch.no_grad()
    def commit_event(self, proposal: EventProposal) -> float:
        self.validate_transition(proposal)
        start = synchronized_time(self.completed_events.device)
        snapshot = self.snapshot_state()
        if self.method == "FROZEN":
            self.completed_events.fill_(proposal.event)
            return synchronized_time(self.completed_events.device) - start
        try:
            for name, value in proposal.weights.items():
                self.smt.memories[name].weight.copy_(value)
            for name, value in proposal.counters.items():
                getattr(self.smt, name).copy_(value)
            self.completed_events.fill_(proposal.event)
        except BaseException:
            for name, value in snapshot.weights.items():
                self.smt.memories[name].weight.copy_(value)
            for name, value in snapshot.counters.items():
                getattr(self.smt, name).copy_(value)
            self.completed_events.fill_(snapshot.event)
            raise
        return synchronized_time(self.completed_events.device) - start

    def serialize_state(self, path: Path | None = None) -> dict[str, Any]:
        payload = {
            "schema_version": 1, "method": self.method, "h": H,
            "smt": {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else deepcopy(value) for key, value in self.smt.state_dict().items()},
            "completed_events": int(self.completed_events), "state_hash": self.state_fingerprint(),
        }
        if path is not None:
            torch.save(payload, path)
        return payload

    @classmethod
    def deserialize_state(cls, payload: Mapping[str, Any], *, device="cpu"):
        if payload["schema_version"] != 1 or payload["h"] != H:
            raise ValueError("incompatible experiment state")
        module = cls(payload["smt"], payload["method"], device=device)
        module.completed_events.fill_(payload["completed_events"])
        if module.state_fingerprint() != payload["state_hash"]:
            raise ValueError("serialized state fingerprint differs")
        return module

    def memory_stats(self) -> dict[str, Any]:
        values = self.state_values()
        full = {key: value for key, value in self.smt.state_dict().items() if isinstance(value, torch.Tensor)}
        online = sum(value.numel() * value.element_size() for value in values.values())
        return {
            "mutable_bytes": online,
            "full_tensor_bytes": sum(value.numel() * value.element_size() for value in full.values()) + 8,
            "controller_persistent_bytes": 0, "model_probe_bytes": 0,
            "frozen_read_only_weight_bytes": online - 32 if self.method == "FROZEN" else 0,
            "mutable_scientific_bytes": 8 if self.method == "FROZEN" else online,
            "online_tensor_keys": len(values),
            "schema": {key: [list(value.shape), str(value.dtype)] for key, value in values.items()},
            "finite": all(bool(torch.isfinite(value).all()) for value in values.values()),
            "persistent_graph": any(value.grad_fn is not None or value.requires_grad for value in values.values()),
        }


@torch.no_grad()
def enumeration_check(memory: ImageSynchronousMemory, proposal: EventProposal, *, seed: int) -> list[dict[str, Any]]:
    """Permute tuples after the one original-grid spatial computation."""
    if memory.method not in ("P1", "P2"):
        raise ValueError("aggregate method required")
    q = proposal.quantities
    generator = torch.Generator().manual_seed(seed)
    n = q.count
    orders = {
        "reverse": torch.arange(n - 1, -1, -1),
        "random_a": torch.randperm(n, generator=generator),
        "random_b": torch.randperm(n, generator=generator),
        "uneven_microbatches": torch.arange(n),
    }
    rows = []
    base = aggregate_image_statistics(q)
    state_hash = memory.state_fingerprint()
    snapshot = memory.snapshot_state()
    for name, ids in orders.items():
        parts = (1, min(17, n - 2), n - 1 - min(17, n - 2)) if name == "uneven_microbatches" and n > 2 else None
        stats = aggregate_image_statistics(q, ids, parts)
        transition, _, _ = propose_transition(stats, controlled=memory.method == "P2")
        diffs = {
            "C_max_abs": float((stats.C - base.C).abs().max()),
            "D_max_abs": float((stats.D - base.D).abs().max()),
            "transition_max_abs": float((transition - proposal.transition).abs().max()),
            "state_max_abs": max(float((snapshot.weights[key] @ transition - value).abs().max()) for key, value in proposal.weights.items()),
        }
        passed = torch.allclose(stats.C, base.C, atol=FP32_TOL, rtol=FP32_TOL) and torch.allclose(stats.D, base.D, atol=FP32_TOL, rtol=FP32_TOL)
        passed = passed and torch.allclose(transition, proposal.transition, atol=FP32_TOL, rtol=FP32_TOL)
        passed = passed and all(torch.allclose(snapshot.weights[key] @ transition, value, atol=FP32_TOL, rtol=FP32_TOL) for key, value in proposal.weights.items())
        rows.append({"enumeration": name, "passed": bool(passed), **diffs})
    assert memory.state_fingerprint() == state_hash
    if not all(row["passed"] for row in rows):
        raise ValueError("tuple enumeration invariance failed")
    return rows
