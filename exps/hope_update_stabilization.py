# exps/hope_update_stabilization.py
"""Experimental update-term and patch-horizon diagnostics.

This module is deliberately isolated from the canonical HOPE implementation.
It replays the locked linear recurrence while allowing diagnostic control
mappings and term isolation for the retention/plasticity experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from exps.hope_retention_stabilization import (
    HORIZON_TOKENS,
    _state_metrics,
    _stream_state_summary,
    _direct_surprise,
    build_read_only_rms_reference,
    canonical_initial_states,
    canonical_oracle,
    clone_cms_from_state,
    clone_smt_from_state,
    clone_state_dict,
    effective_rank,
    load_cached_stream,
    six_stats,
    tensor_geometry,
    RetentionMapping,
    run_experimental_smt,
    probe_objective,
)


EPS = 1e-12


@dataclass(frozen=True)
class UpdateMapping:
    """An experiment-only control mapping and term-isolation policy."""

    name: str
    alpha_kind: str = "one"
    lambda_h: float = 0.0
    eta_kind: str = "scaled_sigmoid"
    eta_scale: float = 0.02
    eta_h: float = 0.0
    horizon: int = HORIZON_TOKENS
    disable_rank: bool = False
    disable_surprise: bool = False
    no_update: bool = False
    freeze_eta: bool = False
    freeze_alpha: bool = False
    rank_eta_multiplier: float = 1.0
    surprise_eta_multiplier: float = 1.0

    def controls(
        self,
        raw_eta: torch.Tensor,
        raw_alpha: torch.Tensor,
        *,
        fixed_eta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.eta_kind == "sigmoid":
            eta = torch.sigmoid(raw_eta)
        elif self.eta_kind == "scaled_sigmoid":
            eta = self.eta_scale * torch.sigmoid(raw_eta)
        elif self.eta_kind == "horizon_sigmoid":
            eta = (self.eta_h / float(self.horizon)) * torch.sigmoid(raw_eta)
        elif self.eta_kind == "fixed_matched":
            if fixed_eta is None:
                fixed_eta = torch.full_like(raw_eta, self.eta_h / float(self.horizon) * 0.5)
            eta = fixed_eta.expand_as(raw_eta)
        else:
            raise ValueError(f"unknown eta mapping: {self.eta_kind}")

        if self.alpha_kind == "one":
            alpha = torch.ones_like(raw_alpha)
        elif self.alpha_kind == "pm0":
            alpha = torch.sigmoid(raw_alpha)
        elif self.alpha_kind == "horizon_near_one":
            alpha = 1.0 - (self.lambda_h / float(self.horizon)) * torch.sigmoid(raw_alpha)
        else:
            raise ValueError(f"unknown alpha mapping: {self.alpha_kind}")
        return eta, alpha


def pm0() -> UpdateMapping:
    return UpdateMapping("PM0", alpha_kind="pm0", eta_kind="sigmoid", eta_scale=1.0)


def phase_a_mappings() -> tuple[UpdateMapping, ...]:
    return (
        UpdateMapping("FULL_alpha1_eta0.02", disable_rank=False, disable_surprise=False),
        UpdateMapping("RANK_ONLY_alpha1_eta0.02", disable_surprise=True),
        UpdateMapping("SURPRISE_ONLY_alpha1_eta0.02", disable_rank=True),
        UpdateMapping("NO_UPDATE_alpha1_eta0.02", no_update=True),
    )


def phase_b_mappings() -> tuple[UpdateMapping, ...]:
    return tuple(
        UpdateMapping(f"HNP_eta_h_{value:g}_alpha1", eta_kind="horizon_sigmoid", eta_h=value)
        for value in (0.02, 0.10, 0.50, 1.00)
    )


def phase_c_mapping(eta_h: float, kind: str) -> UpdateMapping:
    suffix = f"{eta_h:g}"
    if kind == "learned":
        return UpdateMapping(
            f"HNP_learned_eta_h_{suffix}_alpha1", eta_kind="horizon_sigmoid", eta_h=eta_h,
        )
    if kind == "fixed":
        return UpdateMapping(
            f"HNP_FIXED_ETA_eta_h_{suffix}_alpha1", eta_kind="fixed_matched", eta_h=eta_h,
        )
    if kind == "frozen":
        return UpdateMapping(
            f"HNP_FROZEN_M_ETA_eta_h_{suffix}_alpha1", eta_kind="horizon_sigmoid",
            eta_h=eta_h, freeze_eta=True,
        )
    raise ValueError(f"unknown Phase C kind: {kind}")


def phase_d_mapping(eta_h: float, lambda_h: float) -> UpdateMapping:
    return UpdateMapping(
        f"HNR_lambda_h_{lambda_h:g}_HNP_eta_h_{eta_h:g}",
        alpha_kind="horizon_near_one", lambda_h=lambda_h,
        eta_kind="horizon_sigmoid", eta_h=eta_h,
    )


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    l = left.detach().reshape(-1).double()
    r = right.detach().reshape(-1).double()
    denom = float(l.norm().item() * r.norm().item())
    if denom <= EPS:
        return float("nan")
    return float(torch.dot(l, r).item() / denom)


def _term_metrics(
    before: torch.Tensor,
    delta_ret: torch.Tensor,
    delta_rank: torch.Tensor,
    delta_surprise: torch.Tensor,
    after: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Any]:
    before_norm = float(before.double().norm().item())
    denominator = before_norm + EPS
    residual = F.linear(key.reshape(1, -1), before).reshape(-1) - target.reshape(-1)
    return {
        "before_norm": before_norm,
        "delta_ret_norm": float(delta_ret.double().norm().item()),
        "delta_rank_norm": float(delta_rank.double().norm().item()),
        "delta_surprise_norm": float(delta_surprise.double().norm().item()),
        "after_norm": float(after.double().norm().item()),
        "ret_ratio": float(delta_ret.double().norm().item() / denominator),
        "rank_ratio": float(delta_rank.double().norm().item() / denominator),
        "surprise_ratio": float(delta_surprise.double().norm().item() / denominator),
        "cos_ret_before": _cosine(delta_ret, before),
        "cos_rank_before": _cosine(delta_rank, before),
        "cos_surprise_before": _cosine(delta_surprise, before),
        "cos_rank_surprise": _cosine(delta_rank, delta_surprise),
        "key_norm": float(key.double().norm().item()),
        "value_norm": float(value.double().norm().item()),
        "target_norm": float(target.double().norm().item()),
        "residual_norm": float(residual.double().norm().item()),
        "finite": bool(torch.isfinite(after).all().item()),
    }


def _prepare_chunk(
    prior: torch.Tensor,
    pending: Sequence[dict[str, torch.Tensor]],
    *,
    name: str,
    mapping: UpdateMapping,
    event_id: int,
    capture_terms: bool,
    fixed_eta: torch.Tensor | None,
    capture_term_summary: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]], bool, dict[str, Any] | None]:
    """Prepare a chunk with the same target ownership as production Eq.93."""
    values = torch.cat([token["v"] for token in pending], dim=0).detach()
    keys = torch.cat([token["k"] for token in pending], dim=0).detach()
    raw_eta = torch.cat([token["raw_eta"] for token in pending], dim=0).detach()
    raw_alpha = torch.cat([token["raw_alpha"] for token in pending], dim=0).detach()
    eta, alpha = mapping.controls(raw_eta, raw_alpha, fixed_eta=fixed_eta)
    target = F.linear(values, prior).detach()
    candidate = prior.detach().clone()
    rows: list[dict[str, Any]] = []
    summary_values: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for index, token in enumerate(pending):
        key = keys[index]
        before = candidate.detach().clone()
        gradient = _direct_surprise(prior, key, target[index])
        delta_ret = (alpha[index, 0] - 1.0) * before
        delta_rank = -(self_eta := eta[index, 0]) * mapping.rank_eta_multiplier * torch.outer(before @ key, key)
        delta_surprise = -self_eta * mapping.surprise_eta_multiplier * gradient
        if mapping.disable_rank or mapping.no_update:
            delta_rank = torch.zeros_like(delta_rank)
        if mapping.disable_surprise or mapping.no_update:
            delta_surprise = torch.zeros_like(delta_surprise)
        if mapping.no_update:
            delta_ret = torch.zeros_like(delta_ret)
        candidate = (before + delta_ret + delta_rank + delta_surprise).detach()
        if capture_term_summary:
            summary_values.append((delta_rank.detach().double().norm(), delta_surprise.detach().double().norm(), before.detach().double().norm()))
        if capture_terms:
            rows.append({
                "event_id": event_id,
                "memory": name,
                "token_index": int(token["position"]),
                "chunk_start": int(pending[0]["position"]),
                "chunk_end": int(pending[-1]["position"] + 1),
                "raw_eta": float(raw_eta[index].reshape(-1)[0].item()),
                "eta": float(eta[index].reshape(-1)[0].item()),
                "raw_alpha": float(raw_alpha[index].reshape(-1)[0].item()),
                "alpha": float(alpha[index].reshape(-1)[0].item()),
                **_term_metrics(before, delta_ret, delta_rank, delta_surprise, candidate, key, values[index], target[index]),
            })
    summary: dict[str, Any] | None = None
    if capture_term_summary and summary_values:
        rank_norms = torch.stack([value[0] for value in summary_values])
        surprise_norms = torch.stack([value[1] for value in summary_values])
        before_norms = torch.stack([value[2] for value in summary_values])
        rank_ratios = rank_norms / (before_norms + EPS)
        surprise_ratios = surprise_norms / (before_norms + EPS)
        summary = {
            "memory": name,
            "rank_update_norm": float(rank_norms.square().sum().sqrt().item()),
            "surprise_update_norm": float(surprise_norms.square().sum().sqrt().item()),
            "state_norm": float(before_norms.square().sum().sqrt().item()),
            "max_rank_state_ratio": float(rank_ratios.max().item()),
            "max_surprise_state_ratio": float(surprise_ratios.max().item()),
        }
        summary["rank_update_state_ratio"] = summary["rank_update_norm"] / (summary["state_norm"] + EPS)
        summary["surprise_update_state_ratio"] = summary["surprise_update_norm"] / (summary["state_norm"] + EPS)
        summary["surprise_rank_norm_ratio"] = summary["surprise_update_norm"] / (summary["rank_update_norm"] + EPS)
    return candidate.detach(), rows, bool(torch.isfinite(candidate).all().item()), summary


def _trace_row(
    point: int,
    kind: str,
    mapping: UpdateMapping,
    module: Any,
    initial: Mapping[str, torch.Tensor],
    outputs: Sequence[torch.Tensor],
    raw_eta: Sequence[torch.Tensor],
    raw_alpha: Sequence[torch.Tensor],
    eta_values: Sequence[torch.Tensor],
    alpha_values: Sequence[torch.Tensor],
    alpha_product: float,
    alpha_log_product: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "mapping": mapping.name,
        "point": point,
        "kind": kind,
        "alpha_product_float64": alpha_product,
        "alpha_log_product_float64": alpha_log_product,
    }
    for prefix, values in (("raw_eta", raw_eta), ("eta", eta_values), ("raw_alpha", raw_alpha), ("alpha", alpha_values)):
        if values:
            flat = torch.cat(values).detach().double().reshape(-1)
            quantiles = torch.quantile(flat, flat.new_tensor([0.01, 0.25, 0.50, 0.75, 0.99]))
            row.update({
                f"{prefix}_min": float(flat.min().item()),
                f"{prefix}_p01": float(quantiles[0].item()),
                f"{prefix}_q25": float(quantiles[1].item()),
                f"{prefix}_median": float(quantiles[2].item()),
                f"{prefix}_mean": float(flat.mean().item()),
                f"{prefix}_q75": float(quantiles[3].item()),
                f"{prefix}_p99": float(quantiles[4].item()),
                f"{prefix}_max": float(flat.max().item()),
            })
    for name, values in _state_metrics(module, initial).items():
        for key, value in values.items():
            row[f"{name}_{key}"] = value
    # Representation geometry is computed once per completed image by the
    # stream runner.  Keeping it out of token trace rows avoids repeated full
    # 784-row SVDs while preserving the recurrence trace itself.
    return row


def run_update_smt(
    module: Any,
    x: torch.Tensor,
    mapping: UpdateMapping,
    *,
    event_id: int = 1,
    capture_trace: bool = True,
    capture_terms: bool = False,
    capture_term_names: frozenset[str] | None = None,
    trace_points: frozenset[int] | None = None,
    term_summary_sink: list[dict[str, Any]] | None = None,
) -> tuple[torch.Tensor, tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Run one image with optional term traces, without touching production code."""
    module._validate_input(x)
    if x.shape[0] != 1:
        raise ValueError("experimental update runner expects B=1")
    with torch.no_grad():
        h = module.preprocess(x)
        initial = {name: memory.weight.detach().clone() for name, memory in module.memories.items()}
        frozen_eta = initial["eta"].detach().clone() if mapping.freeze_eta else None
        frozen_alpha = initial["alpha"].detach().clone() if mapping.freeze_alpha else None
        memory_pending: list[dict[str, torch.Tensor]] = []
        auxiliary_pending: list[dict[str, torch.Tensor]] = []
        memory_prior: torch.Tensor | None = None
        auxiliary_prior: dict[str, torch.Tensor] = {}
        outputs: list[torch.Tensor] = []
        raw_eta_values: list[torch.Tensor] = []
        raw_alpha_values: list[torch.Tensor] = []
        eta_values: list[torch.Tensor] = []
        alpha_values: list[torch.Tensor] = []
        trace_rows: list[dict[str, Any]] = []
        term_rows: list[dict[str, Any]] = []
        points = set(trace_points) if trace_points is not None else {0, 1, 2, 4, 8, 16, h.shape[1]}
        if trace_points is None:
            points.update(range(32, h.shape[1] + 1, 16))
        alpha_product = 1.0
        alpha_log_product = 0.0
        fixed_eta = None
        if mapping.eta_kind == "fixed_matched":
            fixed_eta = torch.full((1, 1), mapping.eta_h / float(mapping.horizon) * 0.5, device=h.device, dtype=h.dtype)

        def add_trace(point: int, kind: str) -> None:
            if not capture_trace or point not in points:
                return
            trace_rows.append(_trace_row(
                point, kind, mapping, module, initial, outputs, raw_eta_values, raw_alpha_values,
                eta_values, alpha_values, alpha_product, alpha_log_product,
            ))

        add_trace(0, "initial")
        for position in range(h.shape[1]):
            if not memory_pending:
                memory_prior = module.memories["memory"].weight.detach().clone()
            if not auxiliary_pending:
                auxiliary_prior = {
                    name: module.memories[name].weight.detach().clone()
                    for name in module.auxiliary_names
                }
                if frozen_eta is not None:
                    auxiliary_prior["eta"] = frozen_eta
                if frozen_alpha is not None:
                    auxiliary_prior["alpha"] = frozen_alpha
            token_h = h[:, position:position + 1]
            raw_eta_tensor = F.linear(token_h, auxiliary_prior["eta"])
            raw_alpha_tensor = F.linear(token_h, auxiliary_prior["alpha"])
            projected = module._project(token_h, auxiliary_prior)
            eta_tensor, alpha_tensor = mapping.controls(raw_eta_tensor, raw_alpha_tensor, fixed_eta=fixed_eta)
            output = F.linear(projected.q, memory_prior)
            outputs.append(output.detach())
            raw_eta_values.append(raw_eta_tensor.detach().reshape(-1))
            raw_alpha_values.append(raw_alpha_tensor.detach().reshape(-1))
            eta_values.append(eta_tensor.detach().reshape(-1))
            alpha_values.append(alpha_tensor.detach().reshape(-1))
            alpha_value = float(alpha_tensor.reshape(-1)[0].item())
            alpha_product *= alpha_value
            alpha_log_product += math.log(alpha_value) if alpha_value > 0.0 else float("-inf")
            token = {
                "position": position,
                "k": projected.k.detach().reshape(1, -1),
                "v": projected.v.detach().reshape(1, -1),
                "raw_eta": raw_eta_tensor.detach().reshape(1, -1),
                "raw_alpha": raw_alpha_tensor.detach().reshape(1, -1),
            }
            memory_pending.append(token)
            auxiliary_pending.append(token)
            end = position + 1
            memory_boundary = end % module.memory_chunk_size == 0 or end == h.shape[1]
            auxiliary_boundary = end % module.auxiliary_memory_chunk_size == 0 or end == h.shape[1]
            prepared: dict[str, torch.Tensor] = {}
            prepared_rows: list[dict[str, Any]] = []
            prepared_ok = True
            if memory_boundary:
                candidate, rows, ok, summary = _prepare_chunk(
                    memory_prior, memory_pending, name="memory", mapping=mapping,
                    event_id=event_id,
                    capture_terms=capture_terms and (capture_term_names is None or "memory" in capture_term_names),
                    fixed_eta=fixed_eta,
                    capture_term_summary=term_summary_sink is not None and (capture_term_names is None or "memory" in capture_term_names),
                )
                prepared["memory"] = candidate
                prepared_rows.extend(rows)
                if summary is not None and term_summary_sink is not None:
                    term_summary_sink.append({"event_id": event_id, "token_count": len(memory_pending), **summary})
                prepared_ok = prepared_ok and ok
            if auxiliary_boundary:
                for name in module.auxiliary_names:
                    candidate, rows, ok, summary = _prepare_chunk(
                        auxiliary_prior[name], auxiliary_pending, name=name, mapping=mapping,
                        event_id=event_id,
                        capture_terms=capture_terms and (capture_term_names is None or name in capture_term_names),
                        fixed_eta=fixed_eta,
                        capture_term_summary=term_summary_sink is not None and (capture_term_names is None or name in capture_term_names),
                    )
                    prepared[name] = candidate
                    prepared_rows.extend(rows)
                    if summary is not None and term_summary_sink is not None:
                        term_summary_sink.append({"event_id": event_id, "token_count": len(auxiliary_pending), **summary})
                    prepared_ok = prepared_ok and ok
            if prepared_ok and not mapping.no_update:
                for name, candidate in prepared.items():
                    if name == "eta" and mapping.freeze_eta:
                        continue
                    if name == "alpha" and mapping.freeze_alpha:
                        continue
                    module.memories[name].weight.copy_(candidate)
                module.memory_update_count.add_(int(memory_boundary))
                module.auxiliary_update_count.add_(int(auxiliary_boundary))
                module.online_update_count.add_(int(memory_boundary) + int(auxiliary_boundary))
            if capture_terms:
                term_rows.extend(prepared_rows)
            if memory_boundary:
                memory_pending.clear()
            if auxiliary_boundary:
                auxiliary_pending.clear()
            if end in points:
                add_trace(end, "post-boundary" if memory_boundary or auxiliary_boundary else "post-token")
            if not prepared_ok or any(not torch.isfinite(memory.weight).all().item() for memory in module.memories.values()):
                if capture_trace and (not trace_rows or trace_rows[-1]["point"] != end):
                    add_trace(end, "non-finite-failure")
                return torch.cat(outputs, dim=1), tuple(trace_rows), tuple(term_rows)
        return torch.cat(outputs, dim=1), tuple(trace_rows), tuple(term_rows)


def run_stream_update(
    records: Sequence[Mapping[str, Any]],
    mapping: UpdateMapping,
    smt_state: Mapping[str, Any],
    cms_state: Mapping[str, Any],
    device: str | torch.device,
    *,
    max_events: int,
    read_only_references: Sequence[Mapping[str, float]] | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Run a candidate over an ordered normal stream, stopping at max_events."""
    dim = int(records[0]["patches"].shape[-1])
    smt = clone_smt_from_state(smt_state, dim, device)
    cms = clone_cms_from_state(cms_state, dim, device)
    initial_memories = {name: memory.weight.detach().clone() for name, memory in smt.memories.items()}
    rows: list[dict[str, Any]] = []
    failure: str | None = None
    previous_class: str | None = None
    cumulative_alpha_product = 1.0
    cumulative_alpha_log = 0.0
    for event_id, record in enumerate(records[:max_events], start=1):
        image = record["patches"].to(device)
        representation, trace, _ = run_update_smt(smt, image, mapping, event_id=event_id, capture_trace=True)
        trace_final = trace[-1] if trace else {}
        failed_token = trace_final.get("point") if trace_final.get("kind") == "non-finite-failure" else None
        complete = representation.shape == image.shape and bool(torch.isfinite(representation).all().item()) and failed_token is None
        output = torch.empty(0, device=image.device, dtype=image.dtype)
        cms_result = None
        if complete:
            try:
                cms_result = cms.commit_image(representation, [probe_objective] * cms.K)
                output = cms_result.output.detach()
                complete = output.shape == image.shape and bool(torch.isfinite(output).all().item())
            except Exception as exc:
                failure = f"CMS event {event_id}: {type(exc).__name__}: {exc}"
                complete = False
        else:
            failure = f"SMT event {event_id} became non-finite or incomplete"
        state_summary = _stream_state_summary(smt, cms)
        metrics = _state_metrics(smt, initial_memories)
        smt_geom = tensor_geometry(representation) if representation.numel() else {
            "norm": float("nan"), "rms": float("nan"), "centered_patch_variance": float("nan"),
            "effective_rank": float("nan"), "finite": False, "cosine_mean": float("nan"), "cosine_std": float("nan"),
        }
        hope_geom = tensor_geometry(output) if complete else {
            "norm": float("nan"), "rms": float("nan"), "centered_patch_variance": float("nan"),
            "effective_rank": float("nan"), "finite": False, "cosine_mean": float("nan"), "cosine_std": float("nan"),
        }
        observed_alpha = trace_final.get("alpha_product_float64")
        observed_alpha_log = trace_final.get("alpha_log_product_float64")
        if observed_alpha is not None:
            cumulative_alpha_product *= float(observed_alpha)
        if observed_alpha_log is not None:
            cumulative_alpha_log += float(observed_alpha_log)
        reference = read_only_references[event_id - 1] if read_only_references is not None else {}
        row: dict[str, Any] = {
            "candidate": mapping.name,
            "event_id": event_id,
            "class_name": record["class_name"],
            "relative_path": record["relative_path"],
            "previous_class": previous_class,
            "is_class_boundary": previous_class is not None and previous_class != record["class_name"],
            "complete_event": bool(complete),
            "finite": bool(complete and state_summary["finite_state"]),
            "smt_failure_token": failed_token,
            "error": failure,
            "due_cms_levels": tuple(cms_result.due_levels) if cms_result is not None else tuple(),
            "alpha_product_observed": observed_alpha,
            "alpha_log_product_observed": observed_alpha_log,
            "alpha_product_cumulative": cumulative_alpha_product,
            "alpha_log_product_cumulative": cumulative_alpha_log,
            "raw_eta_min": trace_final.get("raw_eta_min"), "raw_eta_p01": trace_final.get("raw_eta_p01"),
            "raw_eta_median": trace_final.get("raw_eta_median"), "raw_eta_mean": trace_final.get("raw_eta_mean"),
            "raw_eta_p99": trace_final.get("raw_eta_p99"), "raw_eta_max": trace_final.get("raw_eta_max"),
            "eta_min": trace_final.get("eta_min"), "eta_p01": trace_final.get("eta_p01"),
            "eta_median": trace_final.get("eta_median"), "eta_mean": trace_final.get("eta_mean"),
            "eta_p99": trace_final.get("eta_p99"), "eta_max": trace_final.get("eta_max"),
            "raw_alpha_min": trace_final.get("raw_alpha_min"), "raw_alpha_p01": trace_final.get("raw_alpha_p01"),
            "raw_alpha_median": trace_final.get("raw_alpha_median"), "raw_alpha_mean": trace_final.get("raw_alpha_mean"),
            "raw_alpha_p99": trace_final.get("raw_alpha_p99"), "raw_alpha_max": trace_final.get("raw_alpha_max"),
            "alpha_min": trace_final.get("alpha_min"), "alpha_p01": trace_final.get("alpha_p01"),
            "alpha_median": trace_final.get("alpha_median"), "alpha_mean": trace_final.get("alpha_mean"),
            "alpha_p99": trace_final.get("alpha_p99"), "alpha_max": trace_final.get("alpha_max"),
            "smt_norm": smt_geom["norm"], "smt_rms": smt_geom["rms"],
            "smt_centered_variance": smt_geom["centered_patch_variance"], "smt_effective_rank": smt_geom["effective_rank"],
            "smt_pairwise_cosine_mean": smt_geom.get("cosine_mean"), "smt_pairwise_cosine_std": smt_geom.get("cosine_std"),
            "hope_norm": hope_geom["norm"], "hope_rms": hope_geom["rms"],
            "hope_centered_variance": hope_geom["centered_patch_variance"], "hope_effective_rank": hope_geom["effective_rank"],
            "hope_pairwise_cosine_mean": hope_geom.get("cosine_mean"), "hope_pairwise_cosine_std": hope_geom.get("cosine_std"),
            "smt_rms_fresh_read_only": reference.get("smt_rms"), "hope_rms_fresh_read_only": reference.get("hope_rms"),
            **{f"{name}_{key}": value for name, vals in metrics.items() for key, value in vals.items()},
            **state_summary,
        }
        rows.append(row)
        if progress and (event_id in (1, 2, 4, 8, 14, 16, 32, 40, 41, 50) or not complete):
            print(
                f"{mapping.name}: event={event_id} complete={complete} SMT_RMS={row['smt_rms']:.6g} "
                f"rank={row['smt_effective_rank']:.6g} M={row['memory_norm']:.6g}", flush=True,
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
            row["smt_rms_rel_event1"] = row["smt_rms"] / (first["smt_rms"] + EPS)
            row["hope_rms_rel_event1"] = row["hope_rms"] / (first["hope_rms"] + EPS)
            row["smt_centered_variance_rel_event1"] = row["smt_centered_variance"] / (first["smt_centered_variance"] + EPS)
            row["hope_centered_variance_rel_event1"] = row["hope_centered_variance"] / (first["hope_centered_variance"] + EPS)
            row["smt_effective_rank_rel_event1"] = row["smt_effective_rank"] / (first["smt_effective_rank"] + EPS)
            row["hope_effective_rank_rel_event1"] = row["hope_effective_rank"] / (first["hope_effective_rank"] + EPS)
    complete_count = sum(int(row["complete_event"]) for row in rows)
    final = rows[-1] if rows else {}
    all_finite = bool(rows) and all(bool(row["finite_state"]) for row in rows)
    state_constant = bool(rows) and len({row["state_bytes"] for row in rows}) == 1 and len({row["state_schema_signature"] for row in rows}) == 1
    exact_zero = any(row["fast_nonzero_count"] == 0 for row in rows)
    exploding = failure is not None or not all_finite
    if exploding:
        classification = "EXPLODING"
    elif exact_zero:
        classification = "COLLAPSED"
    elif not state_constant or any(row["persistent_grad_fn"] or row["online_requires_grad"] for row in rows):
        classification = "INCONCLUSIVE"
    elif complete_count < max_events:
        classification = "INCONCLUSIVE"
    elif max(float(row["smt_rms_rel_event1"]) for row in rows) > 10.0 or max(float(row["hope_rms_rel_event1"]) for row in rows) > 10.0:
        classification = "INCONCLUSIVE"
    elif float(final["smt_rms_rel_event1"]) < 0.5 or float(final["hope_rms_rel_event1"]) < 0.5 or float(final["smt_centered_variance_rel_event1"]) < 0.25 or float(final["hope_centered_variance_rel_event1"]) < 0.25:
        classification = "DECAYING"
    else:
        classification = "STABLE"
    return {
        "candidate": mapping.name,
        "mapping": mapping,
        "rows": rows,
        "smt": smt,
        "cms": cms,
        "failure": failure,
        "classification": classification,
        "event_count": len(rows),
        "complete_event_count": complete_count,
        "all_finite": all_finite,
        "state_bytes_constant": state_constant,
        "trace": trace if rows else tuple(),
    }


def flatten_term_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"term_rows": 0}
    summary: dict[str, Any] = {"term_rows": len(rows)}
    for key in ("before_norm", "delta_ret_norm", "delta_rank_norm", "delta_surprise_norm", "ret_ratio", "rank_ratio", "surprise_ratio", "residual_norm"):
        values = torch.tensor([float(row[key]) for row in rows], dtype=torch.float64)
        summary[f"{key}_mean"] = float(values.mean().item())
        summary[f"{key}_max"] = float(values.max().item())
    return summary


def pm0_oracle_check(x: torch.Tensor, smt_state: Mapping[str, Any]) -> dict[str, Any]:
    """Compare this recurrence and the existing validated PM0 recurrence."""
    baseline = clone_smt_from_state(smt_state, x.shape[-1], x.device)
    expected, _ = run_experimental_smt(baseline, x, RetentionMapping("PM0"), capture_trace=False)
    candidate_module = clone_smt_from_state(smt_state, x.shape[-1], x.device)
    actual, _, _ = run_update_smt(candidate_module, x, pm0(), capture_trace=False)
    production = clone_smt_from_state(smt_state, x.shape[-1], x.device)
    production_result = production(x, update=True).memory_prediction
    state_delta_helper = {
        name: float((baseline.memories[name].weight - candidate_module.memories[name].weight).abs().max().item())
        for name in baseline.memories
    }
    state_delta_production = {
        name: float((production.memories[name].weight - candidate_module.memories[name].weight).abs().max().item())
        for name in production.memories
    }
    result = {
        "helper_output_max_abs": float((expected - actual).abs().max().item()),
        "production_output_max_abs": float((production_result - actual).abs().max().item()),
        "helper_state_max_abs": state_delta_helper,
        "production_state_max_abs": state_delta_production,
        "counters_equal_production": all(
            int(getattr(production, name).item()) == int(getattr(candidate_module, name).item())
            for name in ("memory_update_count", "auxiliary_update_count", "online_update_count")
        ),
        "tolerance": 3e-5,
    }
    result["passed"] = bool(
        result["helper_output_max_abs"] <= result["tolerance"]
        and result["production_output_max_abs"] <= result["tolerance"]
        and max(result["helper_state_max_abs"].values()) <= result["tolerance"]
        and max(result["production_state_max_abs"].values()) <= result["tolerance"]
        and result["counters_equal_production"]
    )
    return result
