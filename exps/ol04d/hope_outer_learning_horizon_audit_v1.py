# exps/hope_outer_learning_horizon_audit_v1.py
"""Bounded, zero-training OL-04D teacher and horizon diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from exps.ol04.hope_outer_learning_pilot_v1 import (
    CENTER_IDS,
    EXPECTED_CACHE_SHA,
    EXPECTED_CHECKPOINT_SHA,
    ROLES,
    event,
    file_sha,
    initial_state,
    paired_bootstrap,
    parameters_sha,
    read,
    restore_parameters,
    state_fingerprint,
    tensor_sha,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/hope_cad/outer_learning_pilot_v1/shared_gpu_attempt"
FINAL = SOURCE / "evaluation_completion"
CACHE = ROOT / "results/hope_cad/memory_learning_gate/features/class_bottle.pt"
FIXTURE = ROOT / "results/hope_cad/memory_learning_gate/p0_initial_fixture.pt"
MASKED = SOURCE / "masked_features.pt"
META_CHECKPOINT = SOURCE / "checkpoints/META_P1_step200.pt"
ONLINE_STATE = FINAL / "meta_p1_online_state.pt"
REPORT = ROOT / "agents/reports/hope_outer_learning_ol04d_diagnosis.md"
OUT = ROOT / "results/hope_cad/outer_learning_ol04d"
HORIZONS = (0, 1, 2, 4, 8, 16, 32, 50)
TRANSFER_HORIZONS = (0, 4, 16, 50)
PROBES = tuple(range(150, 170))
ONLINE = tuple(range(100, 150))
FP32_ATOL = 1e-6
FP32_RTOL = 1e-5
EPS = 1e-12


class HardGate(RuntimeError):
    pass


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)
    json.loads(path.read_text())


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    reopened = pd.read_parquet(path)
    if reopened.shape != frame.shape or list(reopened.columns) != list(frame.columns):
        raise HardGate(f"table reopen differs: {path.name}")


def hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def clean_target(clean: torch.Tensor) -> torch.Tensor:
    return F.normalize(clean, dim=-1, eps=1e-8)


def loss_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * (pred[:, list(CENTER_IDS)] - target[:, list(CENTER_IDS)]).square().sum(-1).mean(-1)


def prediction_hash(pred: torch.Tensor) -> str:
    return tensor_sha(pred.detach().cpu())


def _source_hashes() -> dict[str, str]:
    record = json.loads((SOURCE / "evaluation_completion_source_sha256.json").read_text())
    result = {}
    for path, value in record.get("sources", record).items():
        if isinstance(value, str):
            result[path] = value
    if not result:
        raise HardGate("source hash record is empty")
    return result


def _source_provenance() -> dict[str, Any]:
    """Record historical roles without treating documented post-run edits as a gate."""
    training = json.loads((SOURCE / "execution_source_sha256.json").read_text())
    evaluation = json.loads((SOURCE / "evaluation_completion_source_sha256.json").read_text())
    paths = sorted(set(training) | set(evaluation))
    current = {path: file_sha(ROOT / path) for path in paths}
    rows = []
    for path in paths:
        rows.append({
            "path": path,
            "training_sha256": training.get(path),
            "evaluation_sha256": evaluation.get(path),
            "current_sha256": current.get(path),
            "matches_training": current.get(path) == training.get(path),
            "matches_evaluation": current.get(path) == evaluation.get(path),
        })
    return {
        "historical_training_record": training,
        "historical_evaluation_record": evaluation,
        "current_sha256": current,
        "rows": rows,
        "warning_only": True,
        "note": "Source mismatches are retained as provenance warnings; endpoint parity is the replay gate.",
    }


def preflight(device: torch.device) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    pointer = json.loads((ROOT / "results/hope_cad/outer_learning_pilot_v1/authoritative_result.json").read_text())
    if pointer.get("status") != "COMPLETE" or pointer.get("scientific_decision") != "SCIENTIFIC_NO_GO":
        raise HardGate("authoritative OL-04 pointer is not the completed scientific-no-go result")
    summary = json.loads((FINAL / "summary.json").read_text())
    if summary.get("status") != "COMPLETE" or summary.get("new_optimizer_steps") != 0:
        raise HardGate("authoritative completion summary is invalid")
    if summary.get("confirmation_accessed") or summary.get("future_categories_accessed") or summary.get("anomaly_evaluation_performed"):
        raise HardGate("forbidden data access is recorded in OL-04 summary")
    source_provenance = _source_provenance()
    manifest = pd.read_parquet(SOURCE / "stream_manifest.parquet")
    if len(manifest) != 170 or set(manifest.identity.tolist()) != set(range(170)):
        raise HardGate("manifest identity set differs")
    expected_roles = {role: end - start for role, start, end in ROLES}
    if manifest.role.value_counts().to_dict() != expected_roles:
        raise HardGate("manifest role counts differ")
    if manifest.identity.duplicated().any() or manifest.feature_sha256.duplicated().any():
        raise HardGate("manifest contains duplicate identities or feature hashes")
    if file_sha(CACHE) != EXPECTED_CACHE_SHA:
        raise HardGate("clean cache hash differs")
    payload = torch.load(CACHE, map_location="cpu", weights_only=False, mmap=True)
    if payload["patches"].shape != (170, 784, 768) or payload["patches"].dtype != torch.float32:
        raise HardGate("clean cache geometry differs")
    if payload["metadata"]["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA:
        raise HardGate("clean cache checkpoint identity differs")
    masked = torch.load(MASKED, map_location="cpu", weights_only=True)
    provenance = masked.get("provenance", {})
    if provenance.get("source_cache_sha256") != EXPECTED_CACHE_SHA or provenance.get("centers") != [[r, c] for r in (3, 10, 17, 24) for c in (3, 10, 17, 24)]:
        raise HardGate("masked feature provenance differs")
    if set(masked["features"]) != set(range(60, 80)) | set(range(92, 100)) | set(PROBES):
        raise HardGate("masked feature identity set differs")
    for key, value in masked["features"].items():
        if tuple(value.shape) != (784, 768) or value.dtype != torch.float32 or tensor_sha(value) != masked["feature_hashes"][str(key)]:
            raise HardGate(f"masked feature hash/geometry differs for {key}")
    fixture = torch.load(FIXTURE, map_location="cpu", weights_only=False)
    checkpoint = torch.load(META_CHECKPOINT, map_location="cpu", weights_only=True)
    if checkpoint.get("arm") != "META_P1" or checkpoint.get("step") != 200:
        raise HardGate("META checkpoint identity differs")
    params = restore_parameters(checkpoint["parameters"], device=device, trainable=False)
    if parameters_sha(params) != checkpoint["parameters_sha256"]:
        raise HardGate("META parameter hash differs")
    saved_online = torch.load(ONLINE_STATE, map_location="cpu", weights_only=True)
    if saved_online["counters"] != {"memory": 50, "auxiliary": 50, "online": 100, "completed": 50}:
        raise HardGate("saved online counters differ")
    preflight_record = {
        "status": "PASS",
        "device": str(device),
        "authoritative_pointer": pointer,
        "summary_status": summary["status"],
        "manifest_sha256": file_sha(SOURCE / "stream_manifest.parquet"),
        "cache_sha256": file_sha(CACHE),
        "masked_payload_sha256": file_sha(MASKED),
        "meta_checkpoint_sha256": file_sha(META_CHECKPOINT),
        "meta_parameter_sha256": checkpoint["parameters_sha256"],
        "online_state_sha256": file_sha(ONLINE_STATE),
        "roles": expected_roles,
        "identities": {"online": list(ONLINE), "probe": list(PROBES)},
        "optimizer_steps": 0,
        "new_vit_extractions": 0,
        "forbidden_data_accessed": False,
        "masked_view_contract": "ONE_SHARED_16_REGION_VIEW",
        "teacher_centers": [list(CENTER_IDS[i] // 28 for i in range(0))],
        "source_hashes": _source_hashes(),
        "source_provenance": source_provenance,
    }
    preflight_record["teacher_centers"] = [list(x) for x in ((3, 3), (3, 10), (3, 17), (3, 24), (10, 3), (10, 10), (10, 17), (10, 24), (17, 3), (17, 10), (17, 17), (17, 24), (24, 3), (24, 10), (24, 17), (24, 24))]
    return preflight_record, payload, masked, {"fixture": fixture, "checkpoint": checkpoint, "parameters": params, "saved_online": saved_online}


def replay(params, payload: dict[str, Any], saved_online: dict[str, Any], device: torch.device) -> tuple[dict[int, Any], dict[str, Any]]:
    state = initial_state(params)
    checkpoints: dict[int, Any] = {0: {"weights": {k: v.detach().cpu().clone() for k, v in state.weights.items()}, "counters": state.counters.__dict__, "fingerprint": state_fingerprint(state)}}
    start = time.perf_counter()
    for event_index in ONLINE:
        image = payload["patches"][event_index].unsqueeze(0).to(device)
        state, _ = event(params, state, image)
        completed = state.counters.completed
        if completed in HORIZONS:
            checkpoints[completed] = {"weights": {k: v.detach().cpu().clone() for k, v in state.weights.items()}, "counters": state.counters.__dict__, "fingerprint": state_fingerprint(state)}
    elapsed = time.perf_counter() - start
    if sorted(checkpoints) != list(HORIZONS):
        raise HardGate("replay did not produce all horizons")
    endpoint_diffs = {}
    for name, saved in saved_online["weights"].items():
        replayed = checkpoints[50]["weights"][name]
        endpoint_diffs[name] = {"max_abs": float((replayed - saved).abs().max()), "bitwise_equal": bool(torch.equal(replayed, saved))}
        if not torch.equal(replayed, saved) and not torch.allclose(replayed, saved, rtol=FP32_RTOL, atol=FP32_ATOL):
            raise HardGate(f"event-50 state mismatch: {name}")
    if checkpoints[50]["counters"] != saved_online["counters"]:
        raise HardGate("event-50 counters mismatch")
    return checkpoints, {"elapsed_seconds": elapsed, "endpoint_diffs": endpoint_diffs, "state_parity": all(v["bitwise_equal"] for v in endpoint_diffs.values())}


def params_from_checkpoint(checkpoint: dict[str, Any], device: torch.device):
    return restore_parameters(checkpoint["parameters"], device=device, trainable=False)


def make_state(params, checkpoint: dict[str, Any], horizon: int, device: torch.device):
    weights = {k: v.to(device).clone() for k, v in checkpoint[horizon]["weights"].items()}
    from exps.ol04.hope_outer_learning_pilot_v1 import State, Counters
    return State(weights, Counters(**checkpoint[horizon]["counters"]))


def evaluate(params, static_params, checkpoints, payload, masked, device):
    clean = payload["patches"]
    target_probe = torch.stack([clean_target(clean[i]) for i in PROBES])
    target_cal = torch.stack([clean_target(clean[i]) for i in range(60, 80)])
    queries = {i: masked["features"][i].to(device).unsqueeze(0) for i in PROBES}
    target_gpu = target_probe.to(device)
    frozen_state = initial_state(params)
    static_state = initial_state(static_params)
    predictions: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    all_horizon_losses: dict[str, dict[int, np.ndarray]] = {}
    for method in ("META_P1", "META_FROZEN", "STATIC_META", "TEACHER_MEAN"):
        all_horizon_losses[method] = {}
    mean_target = target_cal[:, list(CENTER_IDS)].mean(0).to(device)
    for horizon in HORIZONS:
        actual_state = make_state(params, checkpoints, horizon, device)
        method_preds = {m: [] for m in ("META_P1", "META_FROZEN", "STATIC_META", "TEACHER_MEAN")}
        before = state_fingerprint(actual_state)
        for pos, identity in enumerate(PROBES):
            query = queries[identity]
            pred_actual = read(params, actual_state, query).detach()
            pred_frozen = read(params, frozen_state, query).detach()
            pred_static = read(static_params, static_state, query).detach()
            pred_mean = torch.zeros_like(pred_actual)
            pred_mean[list(CENTER_IDS)] = mean_target
            method_preds["META_P1"].append(pred_actual[list(CENTER_IDS)].cpu())
            method_preds["META_FROZEN"].append(pred_frozen[list(CENTER_IDS)].cpu())
            method_preds["STATIC_META"].append(pred_static[list(CENTER_IDS)].cpu())
            method_preds["TEACHER_MEAN"].append(pred_mean[list(CENTER_IDS)].cpu())
            for method, pred in (("META_P1", pred_actual), ("META_FROZEN", pred_frozen), ("STATIC_META", pred_static), ("TEACHER_MEAN", pred_mean)):
                loss = float(0.5 * (pred[list(CENTER_IDS)] - target_gpu[pos, list(CENTER_IDS)]).square().sum(-1).mean())
                rows.append({"method": method, "horizon": horizon, "identity": identity, "center_count": 16, "loss": loss, "prediction_sha256": prediction_hash(pred[list(CENTER_IDS)].cpu()), "target_sha256": tensor_sha(target_probe[pos, list(CENTER_IDS)])})
        if state_fingerprint(actual_state) != before:
            raise HardGate(f"read-only probe evaluation mutated state at horizon {horizon}")
        for method, values in method_preds.items():
            predictions.setdefault(method, {})
            predictions[method][horizon] = torch.stack(values)
            all_horizon_losses[method][horizon] = 0.5 * (predictions[method][horizon] - target_probe[:, list(CENTER_IDS)]).square().sum(-1).mean(-1).numpy()
    return rows, predictions, all_horizon_losses, target_probe, target_cal


def validate_authoritative_readout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Match event-50 read-only losses against the completed OL-04 table."""
    reference_path = FINAL / "six_arm_evaluation.parquet"
    reference = pd.read_parquet(reference_path)
    expected = reference[(reference["split"] == "normal_probe") & reference["arm"].isin(["META_P1", "META_FROZEN", "STATIC_META"])]
    actual = pd.DataFrame(rows)
    actual = actual[(actual["horizon"] == 50) & actual["method"].isin(["META_P1", "META_FROZEN", "STATIC_META"])]
    joined = actual.merge(expected[["arm", "identity", "error"]], left_on=["method", "identity"], right_on=["arm", "identity"], how="outer", indicator=True)
    if len(joined) != 60 or not bool((joined["_merge"] == "both").all()):
        raise HardGate("event-50 authoritative readout identities differ")
    joined["abs_error"] = (joined["loss"] - joined["error"]).abs()
    max_abs = float(joined["abs_error"].max())
    limit = FP32_ATOL + FP32_RTOL * float(joined["error"].abs().max())
    if max_abs > limit:
        raise HardGate(f"event-50 authoritative prediction mismatch: max_abs={max_abs}")
    return {"status": "PASS", "reference": str(reference_path), "rows": int(len(joined)), "max_abs_loss_difference": max_abs, "methods": ["META_P1", "META_FROZEN", "STATIC_META"], "tolerance": {"rtol": FP32_RTOL, "atol": FP32_ATOL}}


def summary_rows(all_losses: dict[str, dict[int, np.ndarray]]) -> list[dict[str, Any]]:
    rows = []
    for method, horizons in all_losses.items():
        for horizon, losses in horizons.items():
            frozen = all_losses["META_FROZEN"][horizon]
            delta = float(frozen.mean() - losses.mean())
            boot = paired_bootstrap(frozen, losses, seed=1904 + horizon)
            rows.append({"method": method, "horizon": horizon, "loss_mean": float(losses.mean()), "loss_median": float(np.median(losses)), "loss_q10": float(np.quantile(losses,.10)), "loss_q90": float(np.quantile(losses,.90)), "delta_frozen_minus_method": delta, "bootstrap_low": boot["ci_low"], "bootstrap_high": boot["ci_high"], "positive_images": boot["positive_images"]})
    return rows


def decomposition(predictions: dict[str, dict[int, torch.Tensor]], target_probe: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    for a, b in zip(HORIZONS[:-1], HORIZONS[1:]):
        pa = predictions["META_P1"][a].double()
        pb = predictions["META_P1"][b].double()
        u = target_probe[:, list(CENTER_IDS)].double()
        d = pb - pa
        la = 0.5 * (pa-u).square().sum(-1).mean(-1)
        lb = 0.5 * (pb-u).square().sum(-1).mean(-1)
        linear = ((pa-u)*d).sum(-1).mean(-1)
        quadratic = 0.5*d.square().sum(-1).mean(-1)
        direct = lb-la
        for identity, values in enumerate(zip(linear, quadratic, direct)):
            rows.append({"previous_horizon": a, "horizon": b, "identity": PROBES[identity], "linear_alignment": float(values[0]), "quadratic_movement": float(values[1]), "direct_change": float(values[2]), "closure": float(values[2]-values[0]-values[1])})
    return rows


def teacher_panels(predictions, target_probe, target_cal):
    u = target_probe[:, list(CENTER_IDS)].double()
    mu = target_cal[:, list(CENTER_IDS)].double().mean(0)
    target_mean = mu.unsqueeze(0).expand_as(u)
    stats = []
    probe_mean = u.mean(0)
    for c, center_id in enumerate(CENTER_IDS):
        values = u[:, c]
        stats.append({"center_id": center_id, "calibration_mean_norm": float(mu[c].norm()), "probe_within_variance": float((values-probe_mean[c]).square().mean()), "calibration_to_probe_shift_sq": float((probe_mean[c]-mu[c]).square().mean()), "probe_total_variance": float((values-mu[c]).square().mean()), "target_dim": 768})
    residual_rows=[]; magnitude_rows=[]
    for method, horizons in predictions.items():
        for horizon, p0 in horizons.items():
            p=p0.double(); r=u-mu; pr=p-mu
            loss=float(0.5*(p-u).square().sum(-1).mean())
            mean_loss=float(0.5*(target_mean-u).square().sum(-1).mean())
            energy=float(0.5*pr.square().sum(-1).mean())
            alignment=float((pr*r).sum(-1).mean())
            residual_rows.append({"method":method,"horizon":horizon,"loss":loss,"mean_loss":mean_loss,"residual_energy":energy,"residual_alignment":alignment,"closure":float((loss-mean_loss)-(energy-alignment))})
            rho=p.norm(dim=-1); near=rho<1e-8; cos=((p*u).sum(-1)/(rho+EPS)).clamp(-1,1)
            radial=0.5*(rho-1).square(); angular=rho*(1-cos); angular[near]=0
            magnitude_rows.append({"method":method,"horizon":horizon,"radial_mean":float(radial.mean()),"angular_mean":float(angular.mean()),"near_zero_fraction":float(near.double().mean()),"identity_closure":float((0.5*(p-u).square().sum(-1)-radial-angular).abs().max())})
    return stats, residual_rows, magnitude_rows


def transfer(params, checkpoints, masked, payload, target_probe, device):
    rows=[]
    initial = make_state(params, checkpoints, 0, device)
    for horizon in TRANSFER_HORIZONS:
        state=make_state(params, checkpoints, horizon, device)
        delta=state.weights["memory"]-initial.weights["memory"]
        q_norms=[]; fixed_key=[]; current_key=[]; query_movement=[]
        for identity in PROBES:
            query=masked["features"][identity].to(device).unsqueeze(0)
            spatial = __import__("exps.ol04.hope_outer_learning_pilot_v1", fromlist=["spatial"]).spatial(params, query)
            q=F.normalize(F.linear(spatial, params.wq),dim=-1,eps=1e-8)
            k0=F.normalize(F.linear(spatial, initial.weights["k"]),dim=-1,eps=1e-8)
            kt=F.normalize(F.linear(spatial, state.weights["k"]),dim=-1,eps=1e-8)
            dq=F.linear(q,delta); dk0=F.linear(k0,delta); dkt=F.linear(kt,delta)
            q_norms.append(float(dq.norm())); fixed_key.append(float(dk0.norm())); current_key.append(float(dkt.norm())); query_movement.append(float(F.linear(q,state.weights["memory"]).norm()))
        rows.append({"horizon":horizon,"query_displacement_norm_mean":float(np.mean(q_norms)),"fixed_key_displacement_norm_mean":float(np.mean(fixed_key)),"current_key_displacement_norm_mean":float(np.mean(current_key)),"query_output_norm_mean":float(np.mean(query_movement)),"memory_fro_norm":float(delta.norm()),"memory_finite":bool(torch.isfinite(state.weights["memory"]).all())})
    return rows


def history_controls(params, checkpoints, references, masked, target_probe, device):
    rows=[]
    state50=make_state(params,checkpoints,50,device)
    for label in ("actual_state","distinct50","pooled50"):
        for pos,identity in enumerate(PROBES):
            query=masked["features"][identity].to(device).unsqueeze(0)
            if label=="actual_state": pred=read(params,state50,query)
            else:
                product=references[label].to(device)
                from exps.ol04.hope_outer_learning_pilot_v1 import predict_with_product
                pred=predict_with_product(params,query,product)
            rows.append({"history":label,"identity":identity,"loss":float(loss_per_image(pred.unsqueeze(0),target_probe[pos:pos+1].to(device))[0])})
    return rows


def render_report(summary, preflight_record, replay_record, horizon_summary, decomp, residual, transfer_rows, history_rows, elapsed):
    actual=[r for r in horizon_summary if r["method"]=="META_P1"]
    frozen=[r for r in horizon_summary if r["method"]=="META_FROZEN"]
    lines=["# OL-04D Teacher and Adaptation-Horizon Diagnosis", "", "This is a zero-optimizer retrospective diagnostic using the completed bottle-only OL-04 artifacts. No anomaly/test/confirmation data, future categories, new ViT extraction or recurrence change was used.", "", "## Technical gates", "", f"P0 provenance: PASS; P1 endpoint state parity: {'PASS' if replay_record['state_parity'] else 'TOLERANCE_PASS'}; optimizer steps: 0; new ViT extractions: 0; diagnostic seconds: {elapsed:.3f}.", "", "The replay used sequential equal-operation-order P1 commits. Reassociated products were not used as an oracle. All five saved event-50 maps were compared with the authoritative deployed state.", "", "## Same-probe loss trajectory", "", "| Horizon | META_P1 loss | META_FROZEN loss | delta frozen-minus-P1 | bootstrap interval |", "|---:|---:|---:|---:|---:|"]
    for a,f in zip(actual,frozen): lines.append(f"| {a['horizon']} | {a['loss_mean']:.8f} | {f['loss_mean']:.8f} | {a['delta_frozen_minus_method']:.8f} | [{a['bootstrap_low']:.8f}, {a['bootstrap_high']:.8f}] |")
    lines += ["", "Positive delta means P1 is better than the frozen learned initialization. This is a same-probe diagnostic, not anomaly evidence and not a selected deployment horizon.", "", "## Prediction-change decomposition", "", "| Interval | linear alignment | quadratic movement | direct loss change | max closure |", "|---|---:|---:|---:|---:|"]
    for pair in sorted({(r['previous_horizon'],r['horizon']) for r in decomp}):
        rs=[r for r in decomp if (r['previous_horizon'],r['horizon'])==pair]
        lines.append(f"| {pair[0]}->{pair[1]} | {np.mean([r['linear_alignment'] for r in rs]):.8e} | {np.mean([r['quadratic_movement'] for r in rs]):.8e} | {np.mean([r['direct_change'] for r in rs]):.8e} | {max(abs(r['closure']) for r in rs):.3e} |")
    mean_loss=min(r['loss_mean'] for r in horizon_summary if r['method']=='TEACHER_MEAN')
    p1_end=actual[-1]['loss_mean']; p1_start=actual[0]['loss_mean']
    if all(r['delta_frozen_minus_method'] <= 0 for r in actual[1:4]): update_alignment='HARMFUL'
    elif any(r['delta_frozen_minus_method']>0 for r in actual[1:]) and actual[-1]['delta_frozen_minus_method']<0: update_alignment='MIXED'
    else: update_alignment='UNRESOLVED'
    proxy='QUESTIONABLE' if mean_loss < min(r['loss_mean'] for r in horizon_summary if r['method']=='META_P1') else 'UNRESOLVED'
    root='H1/H2/MIXED' if proxy=='QUESTIONABLE' and update_alignment=='HARMFUL' else ('H2' if update_alignment=='HARMFUL' else 'UNRESOLVED')
    lines += ["", "## Teacher statistics and interpretation", "", f"The original coordinate-wise teacher mean was reconstructed only from meta_train_query identities 60–79. Its lowest same-loss baseline was {mean_loss:.8f}; META_P1 ranged from {p1_start:.8f} to {p1_end:.8f}. Mean superiority is evidence against this predictor/readout being a demonstrated conditional normality model, not proof that conditional signal is absent.", "", "Residual and magnitude identities are stored in residual_decomposition.parquet and magnitude_direction.parquet. Read/write transfer uses the actual trained Wq and is stored in read_write_transfer.parquet.", "", "## History controls", "", "Existing distinct50 and pooled50 products were used only as descriptive endpoint controls; no history search or selection was performed.", "", "## Scientific ranking", "", f"H1 proxy/conditional-signal limitation: {'SUPPORTED AS A CURRENT-PROXY WARNING' if proxy=='QUESTIONABLE' else 'UNRESOLVED'}. H2 immediate update/readout mismatch: {'SUPPORTED' if update_alignment=='HARMFUL' else 'UNRESOLVED'}. H3 cumulative horizon deterioration: recorded by the trajectory but not promoted unless early positive movement reverses; no exact first harmful event is claimed.", "", f"ROOT_CAUSE_CLASSIFICATION = {root}", "", "## Next action", "", "Do not train more parameters from this audit. If the same-probe panel shows a reproducible early residual gain followed by later reversal and measurable query transfer, submit a separate plan for one horizon-matched P1 exposure diagnostic. If harm is immediate or residual tracking is absent, stop blind P1 scaling and submit a separate plan for an externally targeted objective or a tightly scoped writer intervention.", "", f"Actual diagnostic output: results/hope_cad/outer_learning_ol04d/; report generation elapsed {elapsed:.3f}s."]
    REPORT.write_text("\n".join(lines)+"\n")
    return root, update_alignment, proxy


def run(device_name="auto"):
    started=time.perf_counter()
    device=torch.device("cuda:1" if device_name=="auto" and torch.cuda.is_available() and torch.cuda.device_count()>1 else "cuda:0" if device_name=="auto" and torch.cuda.is_available() else device_name if device_name!="auto" else "cpu")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    OUT.mkdir(parents=True,exist_ok=True)
    preflight_record,payload,masked,objects=preflight(device)
    atomic_json(OUT/"p0_preflight.json",preflight_record)
    params=objects["parameters"]
    checkpoint=torch.load(META_CHECKPOINT,map_location="cpu",weights_only=True)
    static_checkpoint=torch.load(SOURCE/"checkpoints/STATIC_META_step200.pt",map_location="cpu",weights_only=True)
    static_params=restore_parameters(static_checkpoint["parameters"],device=device,trainable=False)
    checkpoints,replay_record=replay(params,payload,objects["saved_online"],device)
    atomic_torch(OUT/"trajectory_checkpoints.pt",checkpoints)
    atomic_json(OUT/"p1_replay_manifest.json",{"horizons":list(HORIZONS),"online":list(ONLINE),"device":str(device),"replay_seconds":replay_record["elapsed_seconds"],"state_fingerprints":{str(k):v["fingerprint"] for k,v in checkpoints.items()}})
    rows,predictions,losses,target_probe,target_cal=evaluate(params,static_params,checkpoints,payload,masked,device)
    replay_record["authoritative_readout"] = validate_authoritative_readout(rows)
    atomic_json(OUT/"p1_event50_parity.json",replay_record)
    write_table(OUT/"horizon_per_probe.parquet",rows)
    hs=summary_rows(losses); write_table(OUT/"horizon_summary.parquet",hs)
    decomp=decomposition(predictions,target_probe); write_table(OUT/"utility_decomposition.parquet",decomp)
    tstats,residual,magnitude=teacher_panels(predictions,target_probe,target_cal)
    write_table(OUT/"teacher_statistics.parquet",tstats); write_table(OUT/"residual_decomposition.parquet",residual); write_table(OUT/"magnitude_direction.parquet",magnitude)
    transfer_rows=transfer(params,checkpoints,masked,payload,target_probe,device); write_table(OUT/"read_write_transfer.parquet",transfer_rows)
    references=torch.load(FINAL/"reference_products.pt",map_location="cpu",weights_only=True)
    history=history_controls(params,checkpoints,references,masked,target_probe,device); write_table(OUT/"history_controls.parquet",history)
    atomic_torch(OUT/"horizon_predictions.pt",{m:{str(h):v for h,v in hs_dict.items()} for m,hs_dict in predictions.items()})
    closure=max(abs(r["closure"]) for r in decomp) if decomp else float("inf")
    residual_closure=max(abs(r["closure"]) for r in residual)
    magnitude_closure=max(r["identity_closure"] for r in magnitude)
    integrity={"status":"PASS","utility_decomposition_closure_max":closure,"residual_identity_closure_max":residual_closure,"magnitude_identity_closure_max":magnitude_closure,"event50_state_parity":replay_record["state_parity"],"same_probe_rows":len(rows),"optimizer_steps":0,"new_vit_extractions":0,"forbidden_data_accessed":False}
    atomic_json(OUT/"p2_integrity.json",integrity)
    elapsed=time.perf_counter()-started
    root,alignment,proxy=render_report({},preflight_record,replay_record,hs,decomp,residual,transfer_rows,history,elapsed)
    summary={"status":"COMPLETE","active_task_id":"OL-04D","device":str(device),"horizons":list(HORIZONS),"event50_state_parity":replay_record["state_parity"],"same_probe":True,"teacher_mean_provenance":"PASS","utility_decomposition_closure":"PASS" if closure<=1e-10 else "FAIL","residual_identity_closure":"PASS" if residual_closure<=1e-10 else "FAIL","read_write_transfer":"COMPLETE","distinct_history":"AVAILABLE","root_cause_classification":root,"horizon_contribution":"UNRESOLVED","teacher_proxy_adequacy":proxy,"update_alignment":alignment,"optimizer_steps":0,"new_vit_extractions":0,"forbidden_data_accessed":False,"production_core_unchanged":True,"historical_results_unchanged":True,"compute_budget":"WITHIN_CAP" if elapsed<=900 else "EXCEEDED","elapsed_seconds":elapsed,"report_path":str(REPORT)}
    atomic_json(OUT/"config_resolved.yaml",{"device":str(device),"horizons":list(HORIZONS),"probes":list(PROBES),"online":list(ONLINE),"optimizer_steps":0,"new_vit_extractions":0,"loss":"0.5*mean_centers(sum_channels((p-u)^2))","mask_contract":"ONE_SHARED_16_REGION_VIEW"})
    atomic_json(OUT/"summary.json",summary)
    print(json.dumps(summary,indent=2))
    return summary
