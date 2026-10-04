# exps/hope_retention_stabilization.py
"""Isolated retention/control mappings and diagnostics for HOPE experiments.

This module intentionally does not modify the canonical Self-Modifying Titans
implementation.  It replays its locked linear recurrence from cloned state so
that alternative eta/alpha parameterizations can be compared against PM0.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from models.hope_cad.continuum_memory import ContinuumMemorySystem
from models.hope_cad.self_modifying_titans import SelfModifyingTitans
from models.hope_cad.state import persistent_tensor_bytes, validate_finite_state


DTYPE_EPS = float(torch.finfo(torch.float32).eps)
RANK_EPS = 1e-12
STREAM_CHECKPOINTS = (1, 2, 4, 8, 16, 32, 40, 41, 50)
GATE2_CANDIDATE_NAMES = (
    "PM0",
    "R1_lambda_0.0005_eta_max_0.02",
    "R1_lambda_0.0005_eta_max_0.1",
    "R1_lambda_0.001_eta_max_0.02",
)
HORIZON_TOKENS = 784
GATE2B_CANDIDATE_NAMES = (
    "PM0",
    "C0_alpha_one_eta_max_0.02",
    "H1_lambda_h_0.002_eta_max_0.02",
    "H2_lambda_h_0.005_eta_max_0.02",
    "H3_lambda_h_0.010_eta_max_0.02",
)


@dataclass(frozen=True)
class RetentionMapping:
    """An experiment-only eta/alpha transform."""

    name: str
    alpha_kind: str = "pm0"
    alpha_scale: float = 0.0
    eta_kind: str = "sigmoid"
    eta_scale: float = 1.0
    horizon: int = HORIZON_TOKENS

    def controls(self, raw_eta: torch.Tensor, raw_alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eta = torch.sigmoid(raw_eta)
        if self.eta_kind == "scaled_sigmoid":
            eta = self.eta_scale * eta
        elif self.eta_kind != "sigmoid":
            raise ValueError(f"unknown eta mapping: {self.eta_kind}")

        if self.alpha_kind == "pm0":
            alpha = torch.sigmoid(raw_alpha)
        elif self.alpha_kind == "fixed_one":
            alpha = torch.ones_like(raw_alpha)
        elif self.alpha_kind == "near_one":
            alpha = 1.0 - self.alpha_scale * torch.sigmoid(raw_alpha)
        elif self.alpha_kind == "horizon_near_one":
            if self.horizon <= 0:
                raise ValueError("horizon must be positive")
            alpha = 1.0 - (self.alpha_scale / self.horizon) * torch.sigmoid(raw_alpha)
        elif self.alpha_kind == "residual_one":
            alpha = 1.0 + self.alpha_scale * torch.tanh(raw_alpha)
        else:
            raise ValueError(f"unknown alpha mapping: {self.alpha_kind}")
        return eta, alpha


def candidate_mappings() -> tuple[RetentionMapping, ...]:
    """Return the fixed Gate-1 candidate list, independent of observations."""
    candidates: list[RetentionMapping] = [RetentionMapping("PM0")]
    for lam in (0.0005, 0.001, 0.002, 0.005):
        suffix = f"{lam:g}"
        candidates.extend(
            (
                RetentionMapping(f"R1_lambda_{suffix}_eta_sigmoid", "near_one", lam),
                RetentionMapping(
                    f"R1_lambda_{suffix}_eta_max_0.1",
                    "near_one", lam, "scaled_sigmoid", 0.1,
                ),
                RetentionMapping(
                    f"R1_lambda_{suffix}_eta_max_0.02",
                    "near_one", lam, "scaled_sigmoid", 0.02,
                ),
            )
        )
    for delta in (0.0005, 0.001):
        candidates.append(RetentionMapping(f"R2_delta_{delta:g}", "residual_one", delta))
    return tuple(candidates)


def gate2_mappings() -> tuple[RetentionMapping, ...]:
    """Return exactly the four mappings locked for the 50-image gate."""
    available = {mapping.name: mapping for mapping in candidate_mappings()}
    return tuple(available[name] for name in GATE2_CANDIDATE_NAMES)


def gate2b_mappings() -> tuple[RetentionMapping, ...]:
    """Return exactly the horizon-normalized Gate-2B candidates."""
    candidates = {
        "PM0": RetentionMapping("PM0"),
        "C0_alpha_one_eta_max_0.02": RetentionMapping(
            "C0_alpha_one_eta_max_0.02", "fixed_one", 0.0, "scaled_sigmoid", 0.02,
        ),
        "H1_lambda_h_0.002_eta_max_0.02": RetentionMapping(
            "H1_lambda_h_0.002_eta_max_0.02", "horizon_near_one", 0.002,
            "scaled_sigmoid", 0.02, HORIZON_TOKENS,
        ),
        "H2_lambda_h_0.005_eta_max_0.02": RetentionMapping(
            "H2_lambda_h_0.005_eta_max_0.02", "horizon_near_one", 0.005,
            "scaled_sigmoid", 0.02, HORIZON_TOKENS,
        ),
        "H3_lambda_h_0.010_eta_max_0.02": RetentionMapping(
            "H3_lambda_h_0.010_eta_max_0.02", "horizon_near_one", 0.010,
            "scaled_sigmoid", 0.02, HORIZON_TOKENS,
        ),
    }
    return tuple(candidates[name] for name in GATE2B_CANDIDATE_NAMES)


def clone_smt_from_state(
    state: Mapping[str, Any], dim: int, device: str | torch.device = "cpu",
) -> SelfModifyingTitans:
    """Create an independent canonical SMT clone from a captured state."""
    module = SelfModifyingTitans(
        dim=dim,
        adaptive_q=False,
        memory_chunk_size=16,
        auxiliary_memory_chunk_size=16,
    )
    module.load_state_dict(state)
    return module.to(device)


def clone_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    """Clone tensor leaves while preserving module extra-state metadata."""
    return {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in state.items()
    }


def clone_cms_from_state(
    state: Mapping[str, Any], dim: int, device: str | torch.device = "cpu",
) -> ContinuumMemorySystem:
    """Create an independent canonical K=2 CMS clone from a captured state."""
    module = ContinuumMemorySystem(
        dim,
        update_periods=(1, 8),
        learning_rates=(1e-3, 1e-3),
        hidden_dim=dim,
    )
    module.load_state_dict(state)
    return module.to(device)


def six_stats(values: torch.Tensor) -> dict[str, float]:
    """Summarize tensor values with overflow-safe diagnostic reductions."""
    flat = values.detach().double().reshape(-1)
    if flat.numel() == 0:
        return {key: float("nan") for key in ("min", "p01", "median", "mean", "p99", "max")}
    quantiles = torch.quantile(flat, flat.new_tensor([0.01, 0.5, 0.99]))
    return {
        "min": float(flat.min().item()),
        "p01": float(quantiles[0].item()),
        "median": float(quantiles[1].item()),
        "mean": float(flat.mean().item()),
        "p99": float(quantiles[2].item()),
        "max": float(flat.max().item()),
    }


def effective_rank(features: torch.Tensor) -> float:
    """Entropy effective rank over all rows after row-centering."""
    matrix = features.detach().to(device="cpu", dtype=torch.float64)
    if matrix.ndim == 3:
        if matrix.shape[0] != 1:
            raise ValueError("effective_rank expects one image when input is rank 3")
        matrix = matrix[0]
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError("effective_rank expects [N,D]")
    if not torch.isfinite(matrix).all().item():
        return float("nan")
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    probabilities = singular / (singular.sum() + RANK_EPS)
    return float(torch.exp(-(probabilities * torch.log(probabilities + RANK_EPS)).sum()).item())


def cosine_summary(left: torch.Tensor, right: torch.Tensor, pairs: torch.Tensor | None = None) -> dict[str, float]:
    """Summarize safe sampled pairwise row cosines."""
    left_rows = left.detach().double().reshape(-1, left.shape[-1])
    right_rows = right.detach().double().reshape(-1, right.shape[-1])
    if pairs is None:
        count = min(left_rows.shape[0], 256)
        indices = torch.arange(count, device=left_rows.device)
        pairs = torch.stack((indices, torch.roll(indices, shifts=-1)), dim=1)
    values = F.cosine_similarity(left_rows[pairs[:, 0]], right_rows[pairs[:, 1]], dim=-1, eps=1e-12)
    return {**six_stats(values), "std": float(values.std(unbiased=False).item())}


def tensor_geometry(features: torch.Tensor, pairs: torch.Tensor | None = None) -> dict[str, float | bool]:
    """Measure geometry without changing the FP32 scientific computation."""
    value = features.detach().double()
    rows = value.reshape(-1, value.shape[-1])
    centered = rows - rows.mean(dim=0, keepdim=True)
    result: dict[str, float | bool] = {
        "norm": float(value.norm().item()),
        "rms": float(value.square().mean().sqrt().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "average_patch_norm": float(rows.norm(dim=-1).mean().item()),
        "centered_patch_variance": float(centered.square().mean().item()),
        "effective_rank": effective_rank(value),
        "finite": bool(torch.isfinite(value).all().item()),
    }
    result.update({f"cosine_{key}": val for key, val in cosine_summary(rows, rows, pairs).items()})
    return result


def _state_metrics(module: SelfModifyingTitans, initial: Mapping[str, torch.Tensor]) -> dict[str, dict[str, float | int | bool]]:
    result: dict[str, dict[str, float | int | bool]] = {}
    for name, memory in module.memories.items():
        value = memory.weight.detach()
        baseline = initial[name].detach()
        norm = float(value.double().norm().item())
        baseline_norm = float(baseline.double().norm().item())
        result[name] = {
            "norm": norm,
            "max_abs": float(value.abs().max().item()),
            "nonzero_count": int(torch.count_nonzero(value).item()),
            "finite": bool(torch.isfinite(value).all().item()),
            "relative_norm": norm / baseline_norm if baseline_norm else float("nan"),
        }
    return result


def _direct_surprise(prior: torch.Tensor, key: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = F.linear(key.reshape(1, -1), prior).reshape(-1) - target.reshape(-1)
    return torch.outer(residual, key.reshape(-1))


def _prepare_candidate(
    prior: torch.Tensor,
    pending: Sequence[dict[str, torch.Tensor]],
    *,
    name: str,
    mapping: RetentionMapping,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay one canonical Eq.93 chunk using transformed controls."""
    values = torch.cat([token["v"] for token in pending], dim=0).detach()
    keys = torch.cat([token["k"] for token in pending], dim=0).detach()
    raw_eta = torch.cat([token["raw_eta"] for token in pending], dim=0).detach()
    raw_alpha = torch.cat([token["raw_alpha"] for token in pending], dim=0).detach()
    eta, alpha = mapping.controls(raw_eta, raw_alpha)
    target = F.linear(values, prior).detach()
    candidate = prior.detach().clone()
    total_surprise = torch.zeros_like(prior)
    for index in range(len(pending)):
        key = keys[index]
        gradient = _direct_surprise(prior, key, target[index])
        candidate = (
            alpha[index, 0] * candidate
            - eta[index, 0] * torch.outer(candidate @ key, key)
            - eta[index, 0] * gradient
        )
        total_surprise.add_(gradient)
    return candidate.detach(), total_surprise.detach()


def _trace_points(length: int) -> tuple[int, ...]:
    points = {0, 1, 2, 4, 8, 16, length}
    points.update(range(32, length + 1, 16))
    return tuple(sorted(point for point in points if 0 <= point <= length))


def run_experimental_smt(
    module: SelfModifyingTitans,
    x: torch.Tensor,
    mapping: RetentionMapping,
    *,
    capture_trace: bool = True,
) -> tuple[torch.Tensor, tuple[dict[str, Any], ...]]:
    """Run the experiment-side recurrence and return causal output plus trace."""
    module._validate_input(x)
    if x.shape[0] != 1:
        raise ValueError("Gate-1 experimental runner expects B=1")
    with torch.no_grad():
        h = module.preprocess(x)
        initial = {name: memory.weight.detach().clone() for name, memory in module.memories.items()}
        memory_pending: list[dict[str, torch.Tensor]] = []
        auxiliary_pending: list[dict[str, torch.Tensor]] = []
        memory_prior: torch.Tensor | None = None
        auxiliary_prior: dict[str, torch.Tensor] = {}
        outputs: list[torch.Tensor] = []
        raw_eta_values: list[torch.Tensor] = []
        raw_alpha_values: list[torch.Tensor] = []
        eta_values: list[torch.Tensor] = []
        alpha_values: list[torch.Tensor] = []
        alpha_product = 1.0
        alpha_log_product = 0.0
        alpha_product_16: float | None = None
        trace_rows: list[dict[str, Any]] = []
        points = set(_trace_points(x.shape[1]))

        def add_trace(point: int, kind: str) -> None:
            if kind != "non-finite-failure" and point not in points:
                return
            if not capture_trace and point != h.shape[1] and kind != "non-finite-failure":
                return
            controls_eta = torch.cat(raw_eta_values) if raw_eta_values else torch.empty(0)
            controls_alpha = torch.cat(raw_alpha_values) if raw_alpha_values else torch.empty(0)
            transformed_eta = torch.cat(eta_values) if eta_values else torch.empty(0)
            transformed_alpha = torch.cat(alpha_values) if alpha_values else torch.empty(0)
            row: dict[str, Any] = {
                "point": point,
                "kind": kind,
                "alpha_product_float64": float(alpha_product) if point else None,
                "alpha_log_product_float64": float(alpha_log_product) if point else None,
                "alpha_product_16_float64": alpha_product_16,
                "mapping": mapping.name,
            }
            for prefix, values in (
                ("raw_eta", controls_eta), ("eta", transformed_eta),
                ("raw_alpha", controls_alpha), ("alpha", transformed_alpha),
            ):
                row.update({f"{prefix}_{key}": value for key, value in six_stats(values).items()} if values.numel() else {})
            if transformed_alpha.numel():
                row["alpha_gt_one_count"] = int((transformed_alpha > 1.0).sum().item())
                row["alpha_sample_count"] = int(transformed_alpha.numel())
                row["alpha_gt_one_fraction"] = float((transformed_alpha > 1.0).float().mean().item())
            else:
                row["alpha_gt_one_count"] = 0
                row["alpha_sample_count"] = 0
                row["alpha_gt_one_fraction"] = 0.0
            for name, values in _state_metrics(module, initial).items():
                for key, value in values.items():
                    row[f"{name}_{key}"] = value
            prefix_output = torch.cat(outputs, dim=1) if outputs else torch.empty(1, 0, module.dim)
            if prefix_output.numel():
                geometry = tensor_geometry(prefix_output)
                row.update({f"smt_{key}": value for key, value in geometry.items()})
            else:
                row.update({"smt_norm": None, "smt_centered_patch_variance": None, "smt_effective_rank": None})
            trace_rows.append(row)

        add_trace(0, "initial")
        for position in range(x.shape[1]):
            if not memory_pending:
                memory_prior = module.memories["memory"].weight.detach().clone()
            if not auxiliary_pending:
                auxiliary_prior = {
                    name: module.memories[name].weight.detach().clone()
                    for name in module.auxiliary_names
                }
            token_h = h[:, position:position + 1]
            raw_eta = F.linear(token_h, auxiliary_prior["eta"])
            raw_alpha = F.linear(token_h, auxiliary_prior["alpha"])
            projected = module._project(token_h, auxiliary_prior)
            eta, alpha = mapping.controls(raw_eta, raw_alpha)
            output = F.linear(projected.q, memory_prior)
            outputs.append(output.detach())
            raw_eta_values.append(raw_eta.detach().reshape(-1))
            raw_alpha_values.append(raw_alpha.detach().reshape(-1))
            eta_values.append(eta.detach().reshape(-1))
            alpha_values.append(alpha.detach().reshape(-1))
            alpha_value = float(alpha.detach().reshape(-1)[0].item())
            alpha_product *= alpha_value
            alpha_log_product += math.log(alpha_value) if alpha_value > 0.0 else float("-inf")
            if position + 1 == 16:
                alpha_product_16 = alpha_product
            token = {
                "k": projected.k.detach().reshape(1, -1),
                "v": projected.v.detach().reshape(1, -1),
                "raw_eta": raw_eta.detach().reshape(1, -1),
                "raw_alpha": raw_alpha.detach().reshape(1, -1),
            }
            memory_pending.append(token)
            auxiliary_pending.append(token)
            end = position + 1
            memory_boundary = end % module.memory_chunk_size == 0 or end == h.shape[1]
            auxiliary_boundary = end % module.auxiliary_memory_chunk_size == 0 or end == h.shape[1]
            prepared: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            if memory_boundary:
                prepared["memory"] = _prepare_candidate(
                    memory_prior, memory_pending, name="memory", mapping=mapping,
                )
            if auxiliary_boundary:
                for name in module.auxiliary_names:
                    prepared[name] = _prepare_candidate(
                        auxiliary_prior[name], auxiliary_pending, name=name, mapping=mapping,
                    )
            for name, (candidate, _surprise) in prepared.items():
                module.memories[name].weight.copy_(candidate)
            module.memory_update_count.add_(int(memory_boundary))
            module.auxiliary_update_count.add_(int(auxiliary_boundary))
            module.online_update_count.add_(int(memory_boundary) + int(auxiliary_boundary))
            if memory_boundary:
                memory_pending.clear()
            if auxiliary_boundary:
                auxiliary_pending.clear()
            if end in points:
                add_trace(end, "post-token" if not (memory_boundary or auxiliary_boundary) else "post-boundary")
            if any(not torch.isfinite(memory.weight).all().item() for memory in module.memories.values()):
                add_trace(end, "non-finite-failure")
                return torch.cat(outputs, dim=1), tuple(trace_rows)
        return torch.cat(outputs, dim=1), tuple(trace_rows)


def probe_objective(level_index: int, level_input: torch.Tensor, level_output: torch.Tensor, metadata: Mapping[str, Any] | None) -> torch.Tensor:
    """Candidate-B identity-preserving objective (probe-only project mapping)."""
    del level_index, metadata
    return 0.5 * (level_output - level_input.detach()).square().mean()


def run_cms(representation: torch.Tensor, cms: ContinuumMemorySystem) -> torch.Tensor:
    """Evaluate one candidate representation through fresh canonical CMS."""
    result = cms.commit_image(representation, [probe_objective] * cms.K)
    return result.output.detach()


def load_bottle_event(path: Path) -> tuple[torch.Tensor, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    patches = payload["patches"]
    if patches.shape != (40, 784, 768) or patches.dtype != torch.float32:
        raise ValueError(f"unexpected bottle cache shape/dtype: {patches.shape} {patches.dtype}")
    return patches[0:1].contiguous(), payload["relative_paths"][0]


def load_cached_stream(feature_root: Path, count: int = 50) -> list[dict[str, Any]]:
    """Load the first cached normal events in the established class order."""
    if count < 1 or count > 80:
        raise ValueError("cached stream count must be between 1 and 80")
    rows: list[dict[str, Any]] = []
    for class_name in ("bottle", "carpet"):
        payload = torch.load(feature_root / f"class_{class_name}.pt", map_location="cpu", weights_only=False)
        patches = payload["patches"]
        if patches.shape != (40, 784, 768) or patches.dtype != torch.float32:
            raise ValueError(f"unexpected {class_name} cache shape/dtype: {patches.shape} {patches.dtype}")
        paths = payload["relative_paths"]
        if payload.get("class_name") != class_name or len(paths) != 40 or paths != sorted(paths):
            raise ValueError(f"invalid {class_name} cache identity/order")
        if not all(path.startswith(f"{class_name}/train/good/") for path in paths):
            raise ValueError("cached stream must contain only normal train/good images")
        if not torch.isfinite(patches).all().item():
            raise ValueError(f"non-finite {class_name} cached patches")
        for index, relative_path in enumerate(paths):
            rows.append({
                "patches": patches[index:index + 1].contiguous(),
                "class_name": class_name,
                "relative_path": relative_path,
            })
            if len(rows) == count:
                return rows
    return rows


def write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """Write a small Parquet table with pyarrow and return its format."""
    import pyarrow as pa
    import pyarrow.parquet as parquet

    keys = sorted({key for row in rows for key in row})
    normalized = [{key: row.get(key) for key in keys} for row in rows]
    table = pa.Table.from_pylist(normalized)
    parquet.write_table(table, path)
    return "parquet"


def read_table(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    return parquet.read_table(path).to_pylist()


def canonical_initial_states(seed: int, dim: int) -> tuple[dict[str, Any], dict[str, Any]]:
    torch.manual_seed(seed)
    smt = SelfModifyingTitans(dim, adaptive_q=False, memory_chunk_size=16, auxiliary_memory_chunk_size=16)
    cms = ContinuumMemorySystem(dim, (1, 8), (1e-3, 1e-3), hidden_dim=dim)
    return clone_state_dict(smt.state_dict()), clone_state_dict(cms.state_dict())


def run_candidate(
    x: torch.Tensor,
    mapping: RetentionMapping,
    smt_state: Mapping[str, Any],
    cms_state: Mapping[str, Any],
) -> dict[str, Any]:
    smt = clone_smt_from_state(smt_state, x.shape[-1])
    cms = clone_cms_from_state(cms_state, x.shape[-1])
    initial_memories = {name: value.detach().clone() for name, value in smt.memory_state().items()}
    representation, trace = run_experimental_smt(smt, x, mapping)
    complete = representation.shape == x.shape and bool(torch.isfinite(representation).all().item())
    output = run_cms(representation, cms) if complete else torch.empty(0, dtype=x.dtype)
    memory_metrics = _state_metrics(smt, initial_memories)
    final_smt = tensor_geometry(representation)
    final_hope = tensor_geometry(output) if complete else {
        "norm": float("nan"), "mean": float("nan"), "std": float("nan"),
        "average_patch_norm": float("nan"), "centered_patch_variance": float("nan"),
        "effective_rank": float("nan"), "finite": False,
    }
    raw_alpha = torch.cat([row["raw_alpha_mean"] for row in []]) if False else None
    trace_final = trace[-1] if trace else {}
    exact_zero = all(memory_metrics[name]["nonzero_count"] == 0 for name in memory_metrics)
    max_state_norm = max(float(values["norm"]) for values in memory_metrics.values())
    exploding = (not complete) or (not final_smt["finite"]) or (not final_hope["finite"]) or max_state_norm > 1e6
    collapsed = (not exploding) and (exact_zero or float(final_smt["effective_rank"]) <= 1.5 or float(final_smt["centered_patch_variance"]) <= DTYPE_EPS * 10.0)
    if exploding:
        classification = "EXPLODING"
    elif collapsed:
        classification = "COLLAPSED"
    else:
        classification = "NUMERICALLY_VIABLE"
    return {
        "candidate": mapping.name,
        "alpha_mapping": mapping.alpha_kind,
        "alpha_scale": mapping.alpha_scale,
        "eta_mapping": mapping.eta_kind,
        "eta_scale": mapping.eta_scale,
        "representation": representation,
        "hope_output": output,
        "trace": trace,
        "smt_state": smt,
        "cms_state": cms,
        "smt_geometry": final_smt,
        "hope_geometry": final_hope,
        "memory_metrics": memory_metrics,
        "final_memory_norm": memory_metrics["memory"]["norm"],
        "final_alpha_norm": memory_metrics["alpha"]["norm"],
        "final_smt_centered_variance": final_smt["centered_patch_variance"],
        "final_smt_effective_rank": final_smt["effective_rank"],
        "final_hope_centered_variance": final_hope["centered_patch_variance"],
        "final_hope_effective_rank": final_hope["effective_rank"],
        "explosion": bool(exploding),
        "collapse": bool(collapsed),
        "mechanically_valid": bool(not exploding and final_smt["finite"] and final_hope["finite"]),
        "classification": classification,
        "trace_final": trace_final,
        "complete": complete,
    }


def _stream_state_summary(smt: SelfModifyingTitans, cms: ContinuumMemorySystem) -> dict[str, Any]:
    smt_state = smt.state_dict()
    cms_state = cms.state_dict()
    fast_nonzero = sum(int(torch.count_nonzero(memory.weight).item()) for memory in smt.memories.values())
    fast_elements = sum(int(memory.weight.numel()) for memory in smt.memories.values())
    try:
        finite_state = bool(validate_finite_state(smt_state) and validate_finite_state(cms_state))
    except ValueError:
        finite_state = False
    schema = [
        (f"{prefix}.{name}", list(value.shape), str(value.dtype))
        for prefix, state in (("smt", smt_state), ("cms", cms_state))
        for name, value in sorted(state.items()) if isinstance(value, torch.Tensor)
    ]
    persistent_grad_fn = any(
        value.grad_fn is not None
        for module in (smt, cms)
        for value in (*module.parameters(), *module.buffers())
    )
    online_requires_grad = any(value.requires_grad for module in (smt, cms) for value in module.buffers())
    return {
        "state_bytes": persistent_tensor_bytes(smt_state) + persistent_tensor_bytes(cms_state),
        "smt_state_bytes": persistent_tensor_bytes(smt_state),
        "cms_state_bytes": persistent_tensor_bytes(cms_state),
        "state_key_count": len(smt_state) + len(cms_state),
        "state_tensor_count": len(schema),
        "state_schema_signature": hashlib.sha256(json.dumps(schema).encode()).hexdigest(),
        "persistent_grad_fn": persistent_grad_fn,
        "online_requires_grad": online_requires_grad,
        "finite_state": finite_state,
        "fast_nonzero_count": fast_nonzero,
        "fast_element_count": fast_elements,
        "fast_all_nonzero": fast_nonzero == fast_elements,
        "smt_memory_updates": int(smt.memory_update_count.item()),
        "smt_auxiliary_updates": int(smt.auxiliary_update_count.item()),
        "smt_online_updates": int(smt.online_update_count.item()),
        "cms_completed_events": int(cms.completed_events.item()),
        "cms_update_counts": tuple(int(level.update_count.item()) for level in cms.levels),
        "cms_pending_counts": tuple(int(level.pending_count.item()) for level in cms.levels),
    }


def _invalid_geometry() -> dict[str, float | bool]:
    return {
        "norm": float("nan"), "rms": float("nan"), "mean": float("nan"), "std": float("nan"),
        "average_patch_norm": float("nan"), "centered_patch_variance": float("nan"),
        "effective_rank": float("nan"), "finite": False,
    }


def build_read_only_rms_reference(
    records: Sequence[Mapping[str, Any]],
    smt_state: Mapping[str, Any],
    cms_state: Mapping[str, Any],
    device: str | torch.device = "cpu",
) -> list[dict[str, float]]:
    """Compute fresh-state read-only SMT/HOPE RMS references per image."""
    if not records:
        raise ValueError("records must be non-empty")
    dim = int(records[0]["patches"].shape[-1])
    smt = clone_smt_from_state(smt_state, dim, device)
    cms = clone_cms_from_state(cms_state, dim, device)
    references: list[dict[str, float]] = []
    with torch.no_grad():
        for record in records:
            image = record["patches"].to(device)
            smt_output = smt.forward(image, update=False).memory_prediction
            hope_output = cms.forward(smt_output)
            references.append({
                "smt_rms": float(smt_output.detach().double().square().mean().sqrt().item()),
                "hope_rms": float(hope_output.detach().double().square().mean().sqrt().item()),
            })
    return references


def run_stream_candidate(
    records: Sequence[Mapping[str, Any]],
    mapping: RetentionMapping,
    smt_state: Mapping[str, Any],
    cms_state: Mapping[str, Any],
    device: str | torch.device = "cpu",
    read_only_references: Sequence[Mapping[str, float]] | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Run one candidate over ordered normal images without resetting state."""
    if not records:
        raise ValueError("records must be non-empty")
    if read_only_references is not None and len(read_only_references) != len(records):
        raise ValueError("read-only references must align one-to-one with records")
    dim = int(records[0]["patches"].shape[-1])
    smt = clone_smt_from_state(smt_state, dim, device)
    cms = clone_cms_from_state(cms_state, dim, device)
    initial_memories = {name: memory.weight.detach().clone() for name, memory in smt.memories.items()}
    rows: list[dict[str, Any]] = []
    failure: str | None = None
    previous_class: str | None = None
    cumulative_alpha_product = 1.0
    cumulative_alpha_log_product = 0.0
    for event_index, record in enumerate(records, start=1):
        image = record["patches"].to(device)
        representation, trace = run_experimental_smt(smt, image, mapping, capture_trace=False)
        trace_final = trace[-1] if trace else {}
        failure_token = trace_final.get("point") if trace_final.get("kind") == "non-finite-failure" else None
        complete = (
            representation.shape == image.shape
            and bool(torch.isfinite(representation).all().item())
            and failure_token is None
        )
        output = torch.empty(0, dtype=image.dtype)
        cms_result = None
        if complete:
            try:
                cms_result = cms.commit_image(representation, [probe_objective] * cms.K)
                output = cms_result.output.detach()
                complete = output.shape == image.shape and bool(torch.isfinite(output).all().item())
            except Exception as exc:
                failure = f"CMS event {event_index}: {type(exc).__name__}: {exc}"
                complete = False
        else:
            failure = f"SMT event {event_index} became non-finite or incomplete"
        memory_metrics = _state_metrics(smt, initial_memories)
        try:
            smt_geometry = tensor_geometry(representation) if representation.numel() else _invalid_geometry()
        except (RuntimeError, ValueError):
            smt_geometry = _invalid_geometry()
        try:
            hope_geometry = tensor_geometry(output) if complete else _invalid_geometry()
        except (RuntimeError, ValueError):
            hope_geometry = _invalid_geometry()
        state_summary = _stream_state_summary(smt, cms)
        observed_alpha_product = trace_final.get("alpha_product_float64")
        if observed_alpha_product is not None:
            cumulative_alpha_product *= float(observed_alpha_product)
        observed_alpha_log_product = trace_final.get("alpha_log_product_float64")
        if observed_alpha_log_product is not None:
            cumulative_alpha_log_product += float(observed_alpha_log_product)
        reference = read_only_references[event_index - 1] if read_only_references is not None else {}
        row: dict[str, Any] = {
            "candidate": mapping.name,
            "event_id": event_index,
            "class_name": record["class_name"],
            "relative_path": record["relative_path"],
            "previous_class": previous_class,
            "is_class_boundary": previous_class is not None and previous_class != record["class_name"],
            "finite": bool(complete and state_summary["finite_state"]),
            "complete_event": complete,
            "smt_tokens_processed": int(representation.shape[1]),
            "smt_tokens_requested": int(image.shape[1]),
            "smt_failure_token": failure_token,
            "error": failure,
            "due_cms_levels": tuple(cms_result.due_levels) if cms_result is not None else tuple(),
            "alpha_product_observed": observed_alpha_product,
            "alpha_product_cumulative": cumulative_alpha_product,
            "alpha_log_product_observed": observed_alpha_log_product,
            "alpha_log_product_cumulative": cumulative_alpha_log_product,
            "alpha_product_16": trace_final.get("alpha_product_16_float64"),
            "alpha_cumulative_float64_underflow": cumulative_alpha_product == 0.0 and math.isfinite(cumulative_alpha_log_product),
            "alpha_product_is_partial": int(representation.shape[1]) != int(image.shape[1]),
            "raw_eta_min": trace_final.get("raw_eta_min"),
            "raw_eta_p01": trace_final.get("raw_eta_p01"),
            "raw_eta_median": trace_final.get("raw_eta_median"),
            "raw_eta_mean": trace_final.get("raw_eta_mean"),
            "raw_eta_p99": trace_final.get("raw_eta_p99"),
            "raw_eta_max": trace_final.get("raw_eta_max"),
            "eta_min": trace_final.get("eta_min"),
            "eta_p01": trace_final.get("eta_p01"),
            "eta_median": trace_final.get("eta_median"),
            "eta_mean": trace_final.get("eta_mean"),
            "eta_p99": trace_final.get("eta_p99"),
            "eta_max": trace_final.get("eta_max"),
            "raw_alpha_min": trace_final.get("raw_alpha_min"),
            "raw_alpha_p01": trace_final.get("raw_alpha_p01"),
            "raw_alpha_median": trace_final.get("raw_alpha_median"),
            "raw_alpha_mean": trace_final.get("raw_alpha_mean"),
            "raw_alpha_p99": trace_final.get("raw_alpha_p99"),
            "raw_alpha_max": trace_final.get("raw_alpha_max"),
            "alpha_min": trace_final.get("alpha_min"),
            "alpha_p01": trace_final.get("alpha_p01"),
            "alpha_median": trace_final.get("alpha_median"),
            "alpha_mean": trace_final.get("alpha_mean"),
            "alpha_p99": trace_final.get("alpha_p99"),
            "alpha_max": trace_final.get("alpha_max"),
            "alpha_gt_one_fraction": trace_final.get("alpha_gt_one_fraction"),
            "smt_norm": smt_geometry["norm"],
            "smt_rms": smt_geometry["rms"],
            "smt_centered_variance": smt_geometry["centered_patch_variance"],
            "smt_effective_rank": smt_geometry["effective_rank"],
            "smt_finite": smt_geometry["finite"],
            "smt_pairwise_cosine_mean": smt_geometry.get("cosine_mean", float("nan")),
            "smt_pairwise_cosine_std": smt_geometry.get("cosine_std", float("nan")),
            "hope_norm": hope_geometry["norm"],
            "hope_rms": hope_geometry["rms"],
            "hope_centered_variance": hope_geometry["centered_patch_variance"],
            "hope_effective_rank": hope_geometry["effective_rank"],
            "hope_finite": hope_geometry["finite"],
            "hope_pairwise_cosine_mean": hope_geometry.get("cosine_mean", float("nan")),
            "hope_pairwise_cosine_std": hope_geometry.get("cosine_std", float("nan")),
            "smt_rms_fresh_read_only": reference.get("smt_rms"),
            "hope_rms_fresh_read_only": reference.get("hope_rms"),
            "smt_rms_vs_fresh_read_only": (
                smt_geometry["rms"] / (reference["smt_rms"] + 1e-12)
                if reference.get("smt_rms") is not None else None
            ),
            "hope_rms_vs_fresh_read_only": (
                hope_geometry["rms"] / (reference["hope_rms"] + 1e-12)
                if reference.get("hope_rms") is not None else None
            ),
            **{f"{name}_{key}": value for name, metrics in memory_metrics.items() for key, value in metrics.items()},
            **state_summary,
        }
        rows.append(row)
        if progress and (event_index in STREAM_CHECKPOINTS or not complete):
            print(
                f"{mapping.name}: event={event_index} tokens={representation.shape[1]}/{image.shape[1]} "
                f"finite={row['finite']} SMT_RMS={row['smt_rms']:.6g} "
                f"rank={row['smt_effective_rank']:.6g} CMS_events={row['cms_completed_events']}",
                flush=True,
            )
        previous_class = record["class_name"]
        if not complete:
            break
    if rows:
        first = rows[0]
        for row in rows:
            for name in ("memory", "k", "v", "eta", "alpha"):
                baseline = first[f"{name}_norm"]
                row[f"{name}_norm_rel_event1"] = row[f"{name}_norm"] / baseline if baseline else float("nan")
            row["smt_centered_variance_rel_event1"] = row["smt_centered_variance"] / (first["smt_centered_variance"] + 1e-12)
            row["smt_rms_rel_event1"] = row["smt_rms"] / (first["smt_rms"] + 1e-12)
            row["smt_effective_rank_rel_event1"] = row["smt_effective_rank"] / (first["smt_effective_rank"] + 1e-12)
            row["hope_centered_variance_rel_event1"] = row["hope_centered_variance"] / (first["hope_centered_variance"] + 1e-12)
            row["hope_rms_rel_event1"] = row["hope_rms"] / (first["hope_rms"] + 1e-12)
            row["hope_effective_rank_rel_event1"] = row["hope_effective_rank"] / (first["hope_effective_rank"] + 1e-12)
    final = rows[-1]
    nontrivial = all(bool(row["smt_finite"] and row["hope_finite"]) for row in rows)
    if failure or not nontrivial or not all(row["finite_state"] for row in rows):
        classification = "EXPLODING"
    elif any(row["fast_nonzero_count"] == 0 for row in rows):
        classification = "COLLAPSED"
    elif (
        len({row["state_schema_signature"] for row in rows}) != 1
        or len({row["state_bytes"] for row in rows}) != 1
        or any(row["persistent_grad_fn"] or row["online_requires_grad"] for row in rows)
    ):
        classification = "INCONCLUSIVE"
    elif (
        float(final["smt_effective_rank_rel_event1"]) < 0.5
        or float(final["hope_effective_rank_rel_event1"]) < 0.5
        or float(final["smt_centered_variance_rel_event1"]) < 0.25
        or float(final["hope_centered_variance_rel_event1"]) < 0.25
        or float(final["smt_rms_rel_event1"]) < 0.5
        or float(final["hope_rms_rel_event1"]) < 0.5
    ):
        classification = "DECAYING"
    elif any(
        row[f"{prefix}_rms_rel_event1"] > 10.0 or row[f"{prefix}_pairwise_cosine_mean"] > 0.999
        for row in rows for prefix in ("smt", "hope")
    ):
        classification = "INCONCLUSIVE"
    else:
        classification = "STABLE"
    return {
        "candidate": mapping.name,
        "mapping": mapping,
        "rows": rows,
        "classification": classification,
        "failure": failure,
        "event_count": len(rows),
        "state_bytes_constant": len({row["state_bytes"] for row in rows}) == 1,
        "state_schema_constant": len({row["state_schema_signature"] for row in rows}) == 1,
        "all_finite": all(bool(row["finite_state"]) for row in rows),
    }


def flatten_candidate_row(result: Mapping[str, Any], x: torch.Tensor) -> dict[str, Any]:
    trace = result["trace"]
    final = trace[-1] if trace else {}
    row: dict[str, Any] = {
        "candidate": result["candidate"],
        "alpha_mapping": result["alpha_mapping"],
        "alpha_scale": result["alpha_scale"],
        "eta_mapping": result["eta_mapping"],
        "eta_scale": result["eta_scale"],
        "alpha_mean_final": final.get("alpha_mean"),
        "alpha_product_16": next((r.get("alpha_product_float64") for r in trace if r["point"] == min(16, x.shape[1])), None),
        "alpha_product_784": final.get("alpha_product_float64") if result["complete"] else None,
        "alpha_product_at_failure": final.get("alpha_product_float64") if not result["complete"] else None,
        "completed_tokens": int(result["representation"].shape[1]),
        "final_memory_norm": result["final_memory_norm"],
        "final_alpha_norm": result["final_alpha_norm"],
        "smt_centered_variance": result["final_smt_centered_variance"],
        "smt_effective_rank": result["final_smt_effective_rank"],
        "hope_centered_variance": result["final_hope_centered_variance"],
        "hope_effective_rank": result["final_hope_effective_rank"],
        "explosion": result["explosion"],
        "collapse": result["collapse"],
        "mechanically_valid": result["mechanically_valid"],
        "classification": result["classification"],
    }
    for name, metrics in result["memory_metrics"].items():
        row[f"{name}_norm"] = metrics["norm"]
        row[f"{name}_relative_norm"] = metrics["relative_norm"]
    return row


def canonical_oracle(
    x: torch.Tensor,
    smt_state: Mapping[str, Any],
    experimental: Mapping[str, Any],
) -> dict[str, Any]:
    oracle = clone_smt_from_state(smt_state, x.shape[-1], x.device)
    oracle_inspection = oracle.inspect(x, update=True)
    oracle_result = oracle_inspection.result
    candidate_output = experimental["representation"]
    output_delta = float((oracle_result.memory_prediction - candidate_output).abs().max().item())
    state_deltas = {
        name: float((oracle.memories[name].weight - experimental["smt_state"].memories[name].weight).abs().max().item())
        for name in oracle.memories
    }
    counter_deltas = {
        name: int(getattr(oracle, name).item()) - int(getattr(experimental["smt_state"], name).item())
        for name in ("memory_update_count", "auxiliary_update_count", "online_update_count")
    }
    tolerance = 2e-5
    expected_memory = tuple(range(16, x.shape[1] + 1, 16))
    if not expected_memory or expected_memory[-1] != x.shape[1]:
        expected_memory = expected_memory + (x.shape[1],)
    expected_auxiliary = expected_memory
    actual_memory = tuple(trace.positions for trace in oracle_inspection.updates if trace.name == "memory")
    actual_auxiliary = tuple(trace.positions for trace in oracle_inspection.updates if trace.name == "k")
    expected_memory_spans = tuple(tuple(range(start, min(start + 16, x.shape[1]))) for start in range(0, x.shape[1], 16))
    chunk_equal = actual_memory == expected_memory_spans and actual_auxiliary == expected_memory_spans
    return {
        "output_max_abs": output_delta,
        "state_max_abs": state_deltas,
        "counter_deltas": counter_deltas,
        "output_equal": output_delta <= tolerance,
        "state_equal": all(value <= tolerance for value in state_deltas.values()),
        "counters_equal": all(value == 0 for value in counter_deltas.values()),
        "chunk_boundaries_equal": chunk_equal,
        "tolerance": tolerance,
    }
