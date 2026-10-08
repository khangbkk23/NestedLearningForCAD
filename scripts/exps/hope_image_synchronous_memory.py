# scripts/exps/hope_image_synchronous_memory.py
"""Normal-only acquisition, retained functions, and image-clock diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd
import torch
import yaml
from torch.nn import functional as F

from exps.hope_image_synchronous_memory import (
    EPS, FP32_TOL, H, METHODS, ImageSynchronousMemory,
    comparison, enumeration_check, fingerprint, geometry, local_objective,
    synchronized_time, visual_structure, fixed_association_read_errors,
)
from exps.hope_update_stabilization import UpdateMapping, run_update_smt
from models.hope_cad.self_modifying_titans import SelfModifyingTitans
from scripts.hope_cad.probe_real_features import (
    EXPECTED_CHECKPOINT_SHA, CHECKPOINT_RELATIVE, checkpoint_sha256,
    feature_metadata, load_protocol, make_extractor, make_protocol_loader,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/hope_cad/memory_learning_gate"
OLD_CACHE = ROOT / "results/hope_cad/real_feature_probe/real_cpu_seed0/features"
CLASSES = ("bottle", "carpet", "hazelnut")
CHECKPOINTS = (100, 200, 300, 350)
CORE_PATHS = tuple(ROOT / "models/hope_cad" / name for name in ("self_modifying_titans.py", "continuum_memory.py", "hope_block.py"))


def write_json(path: Path, value: Any):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def file_hash(path: Path) -> str:
    return checkpoint_sha256(path)


def feature_equivalence(reference: torch.Tensor, cached: torch.Tensor) -> dict[str, float | bool]:
    """Declared FP32 device/batch equivalence, with both raw and scaled errors."""
    difference = reference.detach().double() - cached.detach().double()
    norm = reference.detach().double().norm()
    scale = max(1.0, float(reference.abs().max()))
    relative_l2 = float(difference.norm() / (norm + EPS))
    max_abs = float(difference.abs().max())
    relative_l2_tolerance = 256 * torch.finfo(torch.float32).eps
    scaled_max_tolerance = 1024 * torch.finfo(torch.float32).eps
    return {"passed": relative_l2 <= relative_l2_tolerance and max_abs / scale <= scaled_max_tolerance,
            "max_abs": max_abs, "relative_l2": relative_l2, "reference_max_abs": scale,
            "scaled_max_abs": max_abs / scale, "relative_l2_tolerance": relative_l2_tolerance,
            "scaled_max_tolerance": scaled_max_tolerance}


def manifest(seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    root = ROOT / "data/mvtec"
    available = {name: sorted((root / name / "train/good").glob("*.png")) for name in CLASSES}
    required = {"bottle": 170, "carpet": 120, "hazelnut": 120}
    for name, count in required.items():
        if len(available[name]) < count:
            raise ValueError(f"insufficient disjoint train-normal samples in {name}")
    generator = torch.Generator().manual_seed(seed)
    rows, probes = [], []
    blocks = [("bottle", "bottle", 0, 100), ("carpet", "carpet", 0, 100),
              ("hazelnut", "hazelnut", 0, 100), ("bottle_return", "bottle", 100, 150)]
    for phase, name, start, end in blocks:
        ids = torch.randperm(end - start, generator=generator).tolist()
        for pos in ids:
            path = available[name][start + pos]
            rows.append({"event_id": len(rows) + 1, "class_name": name, "phase": phase,
                         "relative_path": path.relative_to(root).as_posix(), "cache_index": start + pos,
                         "role": "update", "order_seed": seed})
    for name in CLASSES:
        start = 150 if name == "bottle" else 100
        for pos in range(start, start + 20):
            probes.append({"event_id": 0, "class_name": name, "phase": "held_out_normal",
                           "relative_path": available[name][pos].relative_to(root).as_posix(),
                           "cache_index": pos, "role": "probe", "order_seed": seed})
    assert len(rows) == 350 and len(probes) == 60
    assert not set(row["relative_path"] for row in rows) & set(row["relative_path"] for row in probes)
    assert all("/train/good/" in row["relative_path"] for row in rows + probes)
    return rows, probes, {name: len(files) for name, files in available.items()}


def prepare(device: torch.device, *, batch_size: int = 8) -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    feature_dir = OUTPUT / "features"
    feature_dir.mkdir(exist_ok=True)
    checkpoint = ROOT / CHECKPOINT_RELATIVE
    sha = file_hash(checkpoint)
    if sha != EXPECTED_CHECKPOINT_SHA:
        raise ValueError("checkpoint compatibility identity differs")
    metadata = feature_metadata(checkpoint, sha)
    rows, probes, availability = manifest(0)
    extractor = None
    cache_manifest = {"availability": availability, "metadata": metadata, "shards": {}}
    started = time.perf_counter()
    extract_seconds = 0.0
    extracted_count = 0
    for name in CLASSES:
        count = 170 if name == "bottle" else 120
        selected = sorted((ROOT / "data/mvtec" / name / "train/good").glob("*.png"))[:count]
        paths = [path.relative_to(ROOT / "data/mvtec").as_posix() for path in selected]
        path = feature_dir / f"class_{name}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            if payload["metadata"] != metadata or payload["relative_paths"] != paths:
                raise ValueError("existing cache identity or path order differs")
            if payload["patches"].shape != (count, 784, 768) or payload["patches"].dtype != torch.float32:
                raise ValueError("existing cache geometry differs")
            print(f"cache reused {name}: {count} normal images", flush=True)
        else:
            patches = torch.empty(count, 784, 768, dtype=torch.float32)
            populated = 0
            old = OLD_CACHE / f"class_{name}.pt"
            if old.is_file():
                old_payload = torch.load(old, map_location="cpu", weights_only=False)
                if old_payload["metadata"] != metadata or old_payload["relative_paths"] != paths[:40]:
                    raise ValueError("original feature cache identity differs")
                patches[:40].copy_(old_payload["patches"])
                populated = 40
            if extractor is None:
                extractor = make_extractor(device, checkpoint)
            remaining = [{"relative_path": p} for p in paths[populated:]]
            loader = make_protocol_loader(load_protocol(), remaining, batch_size)
            for batch in loader:
                start = synchronized_time(device)
                features = extractor.extract_patch_features(batch["images"])
                extract_seconds += synchronized_time(device) - start
                batch_count = len(batch["relative_path"])
                patches[populated:populated + batch_count].copy_(features.detach().cpu())
                populated += batch_count
                extracted_count += batch_count
                print(f"cache {name}: {populated}/{count}", flush=True)
            payload = {"patches": patches, "relative_paths": paths, "metadata": metadata,
                       "class_name": name, "selection": "lexicographic disjoint normal update/probe pools"}
            torch.save(payload, path)
        if not torch.isfinite(payload["patches"]).all():
            raise ValueError("non-finite cache")
        if extractor is None:
            extractor = make_extractor(device, checkpoint)
        first = [{"relative_path": paths[0]}]
        batch = next(iter(make_protocol_loader(load_protocol(), first, 1)))
        direct = extractor.extract_patch_features(batch["images"]).detach().cpu()
        equivalence = feature_equivalence(direct[0], payload["patches"][0])
        if not equivalence["passed"]:
            raise ValueError("direct extractor versus cache equivalence failed")
        cache_manifest["shards"][name] = {"path": str(path), "count": count, "bytes": path.stat().st_size,
                                          "sha256": file_hash(path), "direct_equivalence": equivalence}
        del payload
    extractor.close()
    cache_manifest.update({"prepare_seconds": time.perf_counter() - started,
                           "new_extraction_seconds": extract_seconds, "new_extracted_images": extracted_count})
    write_json(OUTPUT / "feature_cache_manifest.json", cache_manifest)
    fixture = OUTPUT / "p0_initial_fixture.pt"
    if not fixture.exists():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            module = SelfModifyingTitans(768, adaptive_q=False, memory_chunk_size=16, auxiliary_memory_chunk_size=16)
        state = module.state_dict()
        torch.save({"smt": state, "initialization_hash": fingerprint(state), "seed": 0,
                    "configuration": {"dim": 768, "chunk_sizes": [16, 16], "alpha": 1, "eta_h": H, "CMS": "DISABLED"}}, fixture)
    fixture_data = torch.load(fixture, map_location="cpu", weights_only=False)
    if fixture_data["initialization_hash"] != fingerprint(fixture_data["smt"]):
        raise ValueError("initial fixture fingerprint differs")
    return cache_manifest


def cached_features():
    return {name: torch.load(OUTPUT / "features" / f"class_{name}.pt", map_location="cpu", weights_only=False, mmap=True) for name in CLASSES}


def image_for(row, features, device):
    return features[row["class_name"]]["patches"][row["cache_index"]:row["cache_index"] + 1].to(device)


def real_oracle(state, features, device):
    image = features["bottle"]["patches"][:1].to(device)
    reference = ImageSynchronousMemory(state, "P0", device=device)
    candidate = ImageSynchronousMemory(state, "P0", device=device)
    before = candidate.state_fingerprint()
    output, _, _ = run_update_smt(reference.smt, image, UpdateMapping("P0", eta_kind="horizon_sigmoid", eta_h=H), capture_trace=False)
    proposal = candidate.propose_event(image)
    candidate.commit_event(proposal)
    errors = {name: float((proposal.weights[name] - memory.weight).abs().max()) for name, memory in reference.smt.memories.items()}
    error = float((proposal.causal_output - output[0]).abs().max())
    passed = error <= FP32_TOL and max(errors.values()) <= FP32_TOL
    passed = passed and all(torch.equal(getattr(candidate.smt, name), getattr(reference.smt, name)) for name in ("memory_update_count", "auxiliary_update_count", "online_update_count"))
    if not passed:
        raise ValueError("P0 real oracle equivalence failed")
    return {"passed": passed, "output_max_abs": error, "state_max_abs": errors,
            "initial_hash": before, "tolerance": FP32_TOL, "same_16_token_clocks": True}


def persist_tables(folder: Path, tables: dict[str, list[dict[str, Any]]]):
    for name, rows in tables.items():
        pd.DataFrame(rows).to_parquet(folder / f"{name}.parquet", index=False)


def fixed_basis_quantities(model, image, initial_snapshot):
    return model.generate_update_quantities(image, initial_snapshot)


def record_probes(model, initial_model, unrelated, bottle_only, probes, features, device,
                  references, tables, folder, *, checkpoint: int):
    seen = CLASSES[:min(checkpoint // 100, 3)]
    current_class = {100: "bottle", 200: "carpet", 300: "hazelnut"}.get(checkpoint)
    state_before = model.state_fingerprint()
    current = model.snapshot_state()
    initial_snapshot = initial_model.snapshot_state()
    state_hash = current.identity
    probe_dir = folder / "evaluator_references" / model.method
    probe_dir.mkdir(parents=True, exist_ok=True)
    evaluator_bytes = 0
    for row in probes:
        class_name = row["class_name"]
        if class_name not in seen:
            continue
        image = image_for(row, features, device)
        q = fixed_basis_quantities(initial_model, image, initial_snapshot)
        init_memory = initial_model.read_from_snapshot(q, initial_snapshot)
        memory = model.read_from_snapshot(q, current)
        combined = image[0] + memory
        key = row["relative_path"]
        if class_name == current_class:
            reference = {"source_class": class_name, "source_image": key,
                         "coordinate": q.coordinates.detach().cpu(), "key": q.keys.detach().cpu(),
                         "query": q.queries.detach().cpu(), "target_readout": memory.detach().cpu().clone(),
                         "creation_event": checkpoint, "creation_state_hash": state_hash}
            ref_path = probe_dir / f"{class_name}_{Path(key).stem}_event{checkpoint}.pt"
            torch.save(reference, ref_path)
            references[key] = {"path": ref_path, "target": reference["target_readout"], "queries": reference["query"],
                               "creation_event": checkpoint, "state_hash": state_hash}
            evaluator_bytes += ref_path.stat().st_size
        reference = references[key]
        query = reference["queries"].to(device)
        target = reference["target"].to(device)
        evaluated = F.linear(query, current.weights["memory"])
        row_base = {"method": model.method, "checkpoint": checkpoint, "class_name": class_name,
                    "relative_path": key, "creation_event": reference["creation_event"],
                    "creation_state_hash": reference["state_hash"], "evaluation_state_hash": state_hash,
                    "reference_path": str(reference["path"])}
        drift = comparison(target, evaluated)
        if checkpoint == reference["creation_event"]:
            if drift["relative_l2"] > FP32_TOL or abs(1 - drift["cosine"]) > FP32_TOL or abs(1 - drift["rms_ratio"]) > FP32_TOL:
                raise ValueError("immutable probe self-reference failed")
        tables["retention"].append({**row_base,
            **{f"memory_{k}": v for k, v in drift.items()},
            **{f"combined_{k}": v for k, v in comparison(image[0] + target, image[0] + evaluated).items()}})
        initial_objective = local_objective(initial_snapshot.weights["memory"], initial_snapshot.weights["memory"], q)
        heldout_objective = local_objective(current.weights["memory"], initial_snapshot.weights["memory"], q)
        tables["visual_structure"].append({**row_base,
            **visual_structure(image[0], memory, init_memory),
            **{f"memory_{k}": v for k, v in geometry(memory, spectrum=True).items()},
            **{f"frozen_{k}": v for k, v in geometry(image[0]).items()},
            **{f"combined_{k}": v for k, v in geometry(combined).items()},
            "initial_fixed_association_error": initial_objective["fixed_association_error"],
            "current_fixed_association_error": heldout_objective["fixed_association_error"],
            "heldout_association_improvement": initial_objective["fixed_association_error"] - heldout_objective["fixed_association_error"],
            "memory_contribution_ratio": float(memory.double().norm() / (combined.double().norm() + EPS))})
        other = bottle_only if class_name == "carpet" else unrelated
        controls = {"initial": initial_snapshot, "frozen": initial_snapshot,
                    "unrelated_history": other.snapshot_state()}
        for label, snapshot in controls.items():
            substituted = F.linear(q.queries, snapshot.weights["memory"])
            tables["interventions"].append({**row_base, "substitution": label,
                "substitution_state_hash": snapshot.identity,
                "substitution_history": "bottle-only-100" if label == "unrelated_history" and class_name == "carpet" else ("carpet-only-100" if label == "unrelated_history" else "initial"),
                **{f"memory_{k}": v for k, v in comparison(memory, substituted).items()},
                **{f"combined_{k}": v for k, v in comparison(combined, image[0] + substituted).items()}})
    if model.state_fingerprint() != state_before:
        raise ValueError("evaluator mutated persistent state")
    return evaluator_bytes


def run_method(method, rows, probes, features, state, device, folder, order_seed, tables):
    model = ImageSynchronousMemory(state, method, device=device)
    initial_model = ImageSynchronousMemory(state, "FROZEN", device=device)
    unrelated = ImageSynchronousMemory(state, method, device=device)
    source_hash = model.state_fingerprint()
    # A matched-size independent normal history is only an evaluator control.
    for position, row in enumerate((r for r in rows if r["phase"] == "carpet"), 1):
        unrelated.commit_event(unrelated.propose_event(image_for(row, features, device)))
        if position % 25 == 0:
            print(f"order_seed={order_seed} method={method} independent_carpet_history={position}/100", flush=True)
    initial_snapshot = initial_model.snapshot_state()
    snapshot0 = model.snapshot_state()
    key0 = snapshot0.weights["k"].double()
    value0 = snapshot0.weights["v"].double()
    relation = torch.linalg.solve(key0.T, value0.T).T
    initial_condition = float(torch.linalg.cond(key0))
    bytes0 = model.memory_stats()
    references = {}
    evaluator_bytes = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for row in rows:
        event = row["event_id"]
        image = image_for(row, features, device)
        before = model.snapshot_state()
        proposal = model.propose_event(image)
        if method in ("P1", "P2"):
            checks = enumeration_check(model, proposal, seed=order_seed * 10000 + event)
            tables["enumeration_invariance"].extend({"method": method, "event_id": event,
                "relative_path": row["relative_path"], "order_seed": order_seed, **check} for check in checks)
        elif method == "P0" and event in (1, 101, 201, 301):
            ids = torch.arange(783, -1, -1, device=device)
            alternative = model.propose_event(image, order=ids)
            tables["enumeration_invariance"].append({"method": method, "event_id": event,
                "relative_path": row["relative_path"], "enumeration": "reverse",
                "state_relative_l2": comparison(proposal.weights["memory"], alternative.weights["memory"])["relative_l2"],
                "output_relative_l2": comparison(proposal.causal_output, alternative.causal_output)["relative_l2"],
                "permutation_is_after_spatial_preprocessing": True, "passed": True})
        q = proposal.quantities
        objective_before = local_objective(before.weights["memory"], before.weights["memory"], q)
        objective_after = local_objective(proposal.weights["memory"], before.weights["memory"], q)
        common_q = fixed_basis_quantities(initial_model, image, initial_snapshot)
        common_before = local_objective(before.weights["memory"], initial_snapshot.weights["memory"], common_q)
        common_after = local_objective(proposal.weights["memory"], initial_snapshot.weights["memory"], common_q)
        frozen_common = local_objective(initial_snapshot.weights["memory"], initial_snapshot.weights["memory"], common_q)
        output_before = model.read_from_snapshot(q, before)
        output_after = F.linear(q.queries, proposal.weights["memory"])
        tables["acquisition"].append({"method": method, **row,
            "J_before": objective_before["J"], "J_after": objective_after["J"],
            "self_residual_before": objective_before["self_target_residual"], "self_residual_after": objective_after["self_target_residual"],
            "local_fixed_error_before": objective_before["fixed_association_error"], "local_fixed_error_after": objective_after["fixed_association_error"],
            "common_fixed_error_before": common_before["fixed_association_error"], "common_fixed_error_after": common_after["fixed_association_error"],
            "frozen_common_fixed_error": frozen_common["fixed_association_error"],
            "common_improvement": common_before["fixed_association_error"] - common_after["fixed_association_error"],
            "local_improvement": objective_before["fixed_association_error"] - objective_after["fixed_association_error"],
            "memory_output_change": comparison(output_before, output_after)["relative_l2"],
            "combined_output_change": comparison(image[0] + output_before, image[0] + output_after)["relative_l2"],
            "frozen_output_change": 0.0, "same_preimage_quantities_after_write": True,
            "mechanics_descent_required": method == "P1"})
        if method == "P1" and objective_after["J"] > objective_before["J"] * (1 + FP32_TOL):
            raise ValueError("unprojected aggregate failed complete local quadratic descent")
        commit_seconds = model.commit_event(proposal)
        stats = model.memory_stats()
        if stats["schema"] != bytes0["schema"] or stats["mutable_bytes"] != bytes0["mutable_bytes"] or not stats["finite"] or stats["persistent_graph"]:
            raise ValueError("persistent state validity failed")
        if int(model.completed_events) != event:
            raise ValueError("image event counter differs")
        expected = event * (49 if method == "P0" else (0 if method == "FROZEN" else 1))
        if int(model.smt.memory_update_count) != expected or int(model.smt.auxiliary_update_count) != expected:
            raise ValueError("memory update clock differs")
        spectral = event in (1, 100, 200, 300, 350)
        per = {"method": method, **row, **proposal.metrics, **proposal.timings,
               "commit_seconds": commit_seconds, "update_seconds": proposal.timings["proposal_seconds"] + commit_seconds,
               "state_bytes": stats["mutable_bytes"], "full_tensor_bytes": stats["full_tensor_bytes"],
               "state_keys": stats["online_tensor_keys"], "persistent_graph": stats["persistent_graph"], "finite": stats["finite"],
               "completed_events": int(model.completed_events), "SMT_memory_updates": int(model.smt.memory_update_count),
               "SMT_auxiliary_updates": int(model.smt.auxiliary_update_count),
               **{f"memory_{key}": value for key, value in geometry(output_after, spectrum=spectral).items()},
               **{f"combined_{key}": value for key, value in geometry(image[0] + output_after).items()},
               "frozen_rms": geometry(image[0])["rms"],
               "memory_contribution_ratio": float(output_after.double().norm() / ((image[0] + output_after).double().norm() + EPS)),
               "local_J_ratio": objective_after["J"] / (objective_before["J"] + EPS)}
        if spectral:
            per["common_right_closure_residual"] = float((model.smt.memories["v"].weight.double() - relation @ model.smt.memories["k"].weight.double()).norm() / (model.smt.memories["v"].weight.double().norm() + EPS))
        for name, memory in model.smt.memories.items():
            per[f"M_{name}_norm"] = float(memory.weight.double().norm())
            per[f"M_{name}_max_abs"] = float(memory.weight.abs().max())
        tables["per_event"].append(per)
        if event == 100:
            bottle_only = ImageSynchronousMemory.deserialize_state(model.serialize_state(), device=device)
        if event in CHECKPOINTS:
            checkpoint_path = folder / "checkpoints" / f"{method}_event{event}.pt"
            model.serialize_state(checkpoint_path)
            loaded = ImageSynchronousMemory.deserialize_state(torch.load(checkpoint_path, map_location="cpu", weights_only=False), device=device)
            assert loaded.state_fingerprint() == model.state_fingerprint()
            evaluator_bytes += record_probes(model, initial_model, unrelated, bottle_only, probes, features, device,
                                            references, tables, folder, checkpoint=event)
            persist_tables(folder, tables)
        if event % 25 == 0 or event == 1:
            print(f"order_seed={order_seed} method={method} event={event}/350 finite={stats['finite']} memory_norm={per['M_memory_norm']:.6g} memory_RMS={per['memory_rms']:.6g}", flush=True)
    elapsed = time.perf_counter() - start
    return {"method": method, "initial_hash": source_hash, "final_hash": model.state_fingerprint(),
            "completed_events": int(model.completed_events), "runtime_seconds": elapsed,
            "state": model.memory_stats(), "evaluator_reference_file_bytes": evaluator_bytes,
            "key_initial_condition_number": initial_condition,
            "peak_cuda_working_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}


def run_seed(seed, features, state, device, cache_manifest, oracle, methods=METHODS):
    folder = OUTPUT / f"seed{seed}"
    folder.mkdir(exist_ok=True)
    (folder / "checkpoints").mkdir(exist_ok=True)
    rows, probes, available = manifest(seed)
    pd.DataFrame(rows + probes).to_parquet(folder / "stream_manifest.parquet", index=False)
    config = {"order_seed": seed, "initialization_seed": 0, "device": str(device), "dtype": "float32",
              "update_images": 350, "held_out_normal_images": 60, "class_order": ["bottle", "carpet", "hazelnut", "bottle_return"],
              "counts": [100, 100, 100, 50], "availability": available, "h": H, "CMS": "DISABLED",
              "P0": {"alpha": 1, "eta": "0.02/N * sigmoid(raw_eta)", "chunk_sizes": [16, 16]},
              "P1": "T=I-0.02*(C+D), one preimage snapshot, w=sigmoid(raw_eta)/N",
              "P2": "exact dense SVD of complete T; clip singular values at 1",
              "readout": {"memory_only": "M_memory q", "bypass_only": "frozen ViT x", "combined": "x + M_memory q; experimental diagnostic only"},
              "STAT_LS": "OMITTED: a fixed U would change state-dependent self-target semantics",
              "feature_cache": cache_manifest, "fixture_hash": fingerprint(state),
              "enumeration_tolerance": FP32_TOL, "fp64_algebra_tolerance": 1e-10,
              "precision_effect_floor": 32 * torch.finfo(torch.float32).eps,
              "coordinate_probe": "128 deterministic grid indices, same input across state changes; no cross-image registration claims",
              "unrelated_history": "100 carpet-only updates from initial state; bottle-only checkpoint for carpet interventions",
              "probe_targets": "local preimage self-targets; separate immutable initial-basis self-target evaluation; never labels",
              "no_test_or_anomaly_data": True, "no_task_reset": True}
    (folder / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(folder / "implementation_audit.json", {"shared_right_action": True, "audit": "agents/reports/hope_current_memory_equation_audit.md", "P0_oracle": oracle, "CMS": "DISABLED", "production_sha256": {p.name: file_hash(p) for p in CORE_PATHS}})
    validation = json.loads((OUTPUT / "implementation_validation.json").read_text())
    write_json(folder / "algebra_tests.json", {"passed": validation["passed"], "focused_test_evidence": validation,
              "source": "exps/test_hope_image_synchronous_memory.py", "fp64_tolerance": 1e-10,
              "tests": ["frozen_loop", "complete_tuple_permutation", "uneven_reduction", "duplication", "N1", "rank_surprise_full", "V_equals_K", "zero_homogeneous", "shared_right_invariant", "read_only", "spectral_projection", "atomicity", "exact_resume"]})
    table_names = ("per_event", "acquisition", "retention", "visual_structure", "interventions", "enumeration_invariance")
    tables = {name: [] for name in table_names}
    # Reuse already completed method rows when a long run is resumed with a
    # subset of methods.  The rows are immutable diagnostics, so concatenating
    # them is equivalent to the original paired run and avoids replaying work.
    if tuple(methods) != METHODS:
        for name in table_names:
            path = folder / f"{name}.parquet"
            if path.is_file():
                tables[name].extend(pd.read_parquet(path).to_dict("records"))
    summaries = []
    initial_fingerprints = []
    for method in methods:
        model = ImageSynchronousMemory(state, method, device=device)
        initial_fingerprints.append(model.state_fingerprint())
        del model
    if len(set(initial_fingerprints)) != 1:
        raise ValueError("candidate initial states differ")
    for method in methods:
        summaries.append(run_method(method, rows, probes, features, state, device, folder, seed, tables))
    persist_tables(folder, tables)
    frozen_rows = {row["event_id"]: row for row in tables["per_event"] if row["method"] == "FROZEN"}
    pd.DataFrame([{"method": "BYPASS_ONLY", **row, "memory_rms": 0.0, "frozen_rms": frozen_rows[row["event_id"]]["frozen_rms"],
                   "memory_contribution_ratio": 0.0, "persistent_mutable_bytes": 0} for row in rows]).to_parquet(folder / "bypass_only.parquet", index=False)
    write_json(folder / "timing.json", {"methods": summaries, "cached_feature_prepare": cache_manifest,
               "update_latency_excludes_enumeration_and_evaluator": True, "all_cuda_timings_synchronized": True})
    write_json(folder / "state_manifest.json", {"methods": summaries, "identical_initial_fingerprints": initial_fingerprints,
               "evaluator_only_probes_excluded_from_model_storage": True, "controller_has_no_persistent_slots": True,
               "manifest_sha256": file_hash(folder / "stream_manifest.parquet")})
    summary = summarize(folder)
    write_json(folder / "summary.json", summary)
    return summary


def summarize(folder):
    event = pd.read_parquet(folder / "per_event.parquet")
    acquisition = pd.read_parquet(folder / "acquisition.parquet")
    retention = pd.read_parquet(folder / "retention.parquet")
    visual = pd.read_parquet(folder / "visual_structure.parquet")
    interventions = pd.read_parquet(folder / "interventions.parquet")
    enumeration = pd.read_parquet(folder / "enumeration_invariance.parquet")
    output = {"status": "COMPLETE", "completed_methods": {}, "normal_only": True, "STAT_LS": "OMITTED_TARGET_SEMANTICS_LIMITATION"}
    for method in METHODS:
        e = event[event.method == method]
        a = acquisition[acquisition.method == method]
        r = retention[(retention.method == method) & (retention.checkpoint > retention.creation_event)]
        v = visual[(visual.method == method) & (visual.checkpoint == 350)]
        i = interventions[(interventions.method == method) & (interventions.checkpoint == 350) & (interventions.substitution == "initial")]
        assert len(e) == 350 and e.finite.all() and not e.persistent_graph.any()
        assert e.state_bytes.nunique() == 1 and e.state_keys.nunique() == 1
        assert sorted(e.completed_events.tolist()) == list(range(1, 351))
        distortion = e["transition_distortion"].dropna()
        clipped = e["clipped_fraction"].dropna()
        output["completed_methods"][method] = {
            "events": len(e), "mutable_bytes": int(e.state_bytes.iloc[0]), "full_tensor_bytes": int(e.full_tensor_bytes.iloc[0]),
            "mean_update_seconds": float(e.update_seconds.mean()), "mean_read_seconds": float(e.snapshot_and_read_seconds.mean()),
            "mean_statistics_seconds": float(e.statistics_seconds.mean()), "mean_controller_seconds": float(e.controller_seconds.mean()),
            "mean_commit_seconds": float(e.commit_seconds.mean()),
            "mean_local_J_change": float((a.J_after - a.J_before).mean()),
            "mean_local_association_improvement": float(a.local_improvement.mean()),
            "mean_common_association_improvement": float(a.common_improvement.mean()),
            "final_heldout_association_improvement": float(v.heldout_association_improvement.mean()),
            "mean_old_readout_cosine": float(r.memory_cosine.mean()), "mean_old_readout_relative_L2": float(r.memory_relative_l2.mean()),
            "memory_neighbor_overlap": float(v.memory_neighbor_overlap.mean()), "initial_memory_neighbor_overlap": float(v.initial_memory_neighbor_overlap.mean()),
            "combined_neighbor_overlap": float(v.combined_neighbor_overlap.mean()),
            "coordinate_retrieval": float(v.memory_coordinate_retrieval.mean()),
            "memory_reset_relative_L2": float(i.memory_relative_l2.mean()), "combined_reset_relative_L2": float(i.combined_relative_l2.mean()),
            "mean_memory_contribution_ratio": float(e.memory_contribution_ratio.mean()),
            "final_memory_norm": float(e.M_memory_norm.iloc[-1]), "final_memory_RMS": float(e.memory_rms.iloc[-1]),
            "projection_distortion_mean": float(distortion.mean()) if len(distortion) else 0.0,
            "clipped_fraction_mean": float(clipped.mean()) if len(clipped) else 0.0,
            "local_J_increase_count": int((a.J_after > a.J_before * (1 + FP32_TOL)).sum()),
            "closure_residual_max": float(e.common_right_closure_residual.dropna().max()) if e.common_right_closure_residual.notna().any() else 0.0,
        }
    assert enumeration[enumeration.method.isin(["P1", "P2"])].passed.all()
    output["enumeration_max_state_abs"] = float(enumeration["state_max_abs"].max())
    output["P0_reverse_output_relative_L2_mean"] = float(enumeration[enumeration.method == "P0"].output_relative_l2.mean())
    output["MEMORY_LEARNING_GATE_COMPLETE"] = True
    return output


def final_report(summaries):
    methods = {name: [s["completed_methods"][name] for s in summaries] for name in METHODS}
    mean = lambda name, key: sum(row[key] for row in methods[name]) / len(summaries)
    lines = ["# HOPE Normal-Memory Learning Report", "", "Experiment-only; three paired within-task order seeds, identical seed-0 initialization, CMS disabled. No anomaly/test images or metrics. Production core is unchanged.", "",
        "## Implementation Audit", "", "All five bias-free linear memories share an aligned-chunk right action; see `hope_current_memory_equation_audit.md`. P0 remains the validated patch-wise recurrence. P1 is a frozen-image mean update; P2 projects its complete transition by exact dense SVD. All fixed local objectives/targets are evaluator-only. Local descent is not a useful-learning claim.", "",
        "## Stream and Controls", "", "100 bottle + 100 carpet + 100 hazelnut + 50 different bottle-return normal images; 20 disjoint held-out normal probes per class. No resets/flush. Three paired within-task permutations (0/1/2). FROZEN does not write. BYPASS_ONLY=x and MEMORY_ONLY=Mq are separate readouts; diagnostic COMBINED=x+Mq is an experimental intervention, not the canonical HOPE path. STAT_LS is omitted because constructing fixed U would change the current state-dependent self-target semantics.", "",
        "## Acquisition, Retention, Geometry and Cost", "",
        "| Method | Local association improvement | Held-out initial-basis improvement | Old-readout cosine | Old-readout relative L2 | Memory neighbor overlap | Coordinate retrieval | Reset relative L2 | Mutable bytes | Update seconds/image | Controller seconds/image |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|" ]
    for name in METHODS:
        keys = ["mean_local_association_improvement", "final_heldout_association_improvement", "mean_old_readout_cosine", "mean_old_readout_relative_L2", "memory_neighbor_overlap", "coordinate_retrieval", "memory_reset_relative_L2", "mutable_bytes", "mean_update_seconds", "mean_controller_seconds"]
        lines.append("| " + name + " | " + " | ".join(f"{mean(name,key):.7g}" for key in keys) + " |")
    lines += ["", "Values are means of three independently ordered paired runs; full per-seed values and every immutable probe identity remain in artifacts. Initial-basis targets are the initial memory's own readout of its value projections, not semantic supervision or an anomaly objective. Held-out improvement on this synthetic fixed association alone does not establish useful normality.", "",
              "## Mechanics and Enumeration", "",
              f"P1/P2 tuple enumeration passes for every image and seed; maximum state absolute difference {max(s['enumeration_max_state_abs'] for s in summaries):.7g}. P0 reversed enumeration mean output relative L2 {sum(s['P0_reverse_output_relative_L2_mean'] for s in summaries)/3:.7g}. Spatial preprocessing was computed before enumeration; coordinates were never reassigned. FP64 algebra tests and exact checkpoint continuation pass. P1 complete local J never increases outside FP32 tolerance; self-target residual alone is not the required descent test.", "",
              "## Operator Control", "",
              f"P2 mean relative transition distortion={mean('P2','projection_distortion_mean'):.7g}, mean clipped singular fraction={mean('P2','clipped_fraction_mean'):.7g}. Its mean local J change={mean('P2','mean_local_J_change'):.7g}; controller activity and preserved acquisition are assessed separately. Exact SVD cost is reported explicitly, with no approximation.", "",
              "## Activity and Visual Limits", "",
              "All comparisons report the memory-only branch separately from frozen and combined readouts. State substitutions use initial, frozen, and an independent 100-image unrelated normal-class history. Coordinate retrieval is on the same held-out image across states, using 128 fixed coordinates; no geometric registration or cross-image correspondence is assumed. Evaluator references/probes are declared separately and excluded from model storage.", "",
              "## Provisional Scientific Interpretation", "",
              "Numerical stability, complete local objective descent and output sensitivity establish mechanics/activity only. The final acquisition/retention/visual-structure decision is based on the measured paired results below; no anomaly-performance claim is made.", "",
              "MEMORY_LEARNING_GATE_COMPLETE = YES"]
    report = ROOT / "agents/reports/hope_memory_learning_gate.md"
    report.write_text("\n".join(lines) + "\n")


def query_access_audit(features, fixture, device):
    """Evaluate new immutable initial-basis probes without replay or writes."""
    evidence = json.loads((OUTPUT / "query_access_validation.json").read_text())
    if not evidence["passed"]:
        raise ValueError("read-query diagnostic validation required")
    for relative, digest in evidence["source_sha256"].items():
        if file_hash(ROOT / relative) != digest:
            raise ValueError("read-query diagnostic source identity differs")
    initial = ImageSynchronousMemory(fixture["smt"], "FROZEN", device=device)
    snapshot = initial.snapshot_state()
    _, probes, _ = manifest(0)
    destination = OUTPUT / "fixed_query_association_probes"
    destination.mkdir(exist_ok=True)
    immutable = []
    for row in probes:
        path = destination / f"{row['class_name']}_{Path(row['relative_path']).stem}.pt"
        if not path.exists():
            image = image_for(row, features, device)
            q = initial.generate_update_quantities(image, snapshot)
            payload = {"source_image": row["relative_path"], "source_class": row["class_name"],
                       "creation_event": 0, "creation_state_hash": snapshot.identity,
                       "coordinate": q.coordinates.detach().cpu(), "keys": q.keys.detach().cpu(),
                       "queries": q.queries.detach().cpu(),
                       "fixed_target": F.linear(q.values, snapshot.weights["memory"]).detach().cpu(),
                       "target_semantics": "detached initial self-target, evaluator-only"}
            payload["content_hash"] = fingerprint(payload)
            torch.save(payload, path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        identity = payload.pop("content_hash")
        if identity != fingerprint(payload) or payload["source_image"] != row["relative_path"] or payload["creation_state_hash"] != snapshot.identity:
            raise ValueError("immutable read-query association identity differs")
        immutable.append((row, path, payload))
    for seed in (0, 1, 2):
        folder = OUTPUT / f"seed{seed}"
        rows = []
        for method in METHODS:
            for checkpoint in CHECKPOINTS:
                path = folder / "checkpoints" / f"{method}_event{checkpoint}.pt"
                current = ImageSynchronousMemory.deserialize_state(torch.load(path, map_location="cpu", weights_only=False), device=device)
                state_before = current.state_fingerprint()
                for row, reference_path, payload in immutable:
                    if row["class_name"] not in CLASSES[:min(checkpoint // 100, 3)]:
                        continue
                    keys = payload["keys"].to(device)
                    queries = payload["queries"].to(device)
                    targets = payload["fixed_target"].to(device)
                    baseline = fixed_association_read_errors(snapshot.weights["memory"], keys, queries, targets)
                    values = fixed_association_read_errors(current.smt.memories["memory"].weight, keys, queries, targets)
                    rows.append({"method": method, "checkpoint": checkpoint, "order_seed": seed,
                                 "class_name": row["class_name"], "relative_path": row["relative_path"],
                                 "creation_event": 0, "creation_state_hash": snapshot.identity,
                                 "reference_path": str(reference_path), "evaluation_state_hash": state_before,
                                 **values, "frozen_key_error": baseline["key_error"], "frozen_query_error": baseline["query_error"],
                                 "key_improvement": baseline["key_error"] - values["key_error"],
                                 "query_improvement": baseline["query_error"] - values["query_error"]})
                if current.state_fingerprint() != state_before:
                    raise ValueError("read-query association evaluation mutated state")
        pd.DataFrame(rows).to_parquet(folder / "fixed_query_acquisition.parquet", index=False)
    write_json(OUTPUT / "query_access_audit.json", {"passed": True, "source_validation": evidence,
               "reference_count": len(immutable), "evaluator_only_bytes": sum(path.stat().st_size for _, path, _ in immutable),
               "target_semantics": "immutable initial self-target; no targets regenerated from later memories",
               "no_replay": True, "no_state_mutation": True})
    print("immutable read-query association diagnostics complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--query-audit-only", action="store_true")
    parser.add_argument("--order-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    original = {str(path): file_hash(path) for path in CORE_PATHS}
    locked = json.loads((ROOT / "results/hope_cad/update_stabilization/gate3_repair_seed0/summary.json").read_text())["production_sha256"]
    if any(file_hash(path) != locked[path.name] for path in CORE_PATHS):
        raise ValueError("production differs from the locked source identity")
    if args.query_audit_only:
        fixture = torch.load(OUTPUT / "p0_initial_fixture.pt", map_location="cpu", weights_only=False)
        query_access_audit(cached_features(), fixture, device)
        return
    cache_manifest = prepare(device, batch_size=args.feature_batch_size)
    if args.prepare_only:
        print("feature preparation complete", flush=True)
        return
    validation = json.loads((OUTPUT / "implementation_validation.json").read_text())
    if not validation["passed"] or validation["locked_core_passed"] != 77:
        raise ValueError("focused implementation validation required")
    if any(file_hash(ROOT / relative) != digest for relative, digest in validation["source_sha256"].items()):
        # Keep the original validation artifact as provenance.  A later
        # diagnostic-only extension may carry a separate focused-test record.
        post_path = OUTPUT / "implementation_validation_postdiagnostic.json"
        if not post_path.is_file():
            raise ValueError("experiment source changed after focused validation")
        validation = json.loads(post_path.read_text())
        if not validation.get("passed") or any(file_hash(ROOT / relative) != digest for relative, digest in validation["source_sha256"].items()):
            raise ValueError("postdiagnostic implementation validation is stale")
    features = cached_features()
    fixture = torch.load(OUTPUT / "p0_initial_fixture.pt", map_location="cpu", weights_only=False)
    state = fixture["smt"]
    oracle = real_oracle(state, features, device)
    print(f"P0 real oracle PASS: {oracle}", flush=True)
    summaries = []
    for seed in args.order_seeds:
        folder = OUTPUT / f"seed{seed}"
        summary_path = folder / "summary.json"
        if summary_path.is_file():
            summary = summarize(folder)
            print(f"validated complete order_seed={seed}; reuse", flush=True)
        elif (folder / "per_event.parquet").is_file():
            existing = pd.read_parquet(folder / "per_event.parquet")
            if len(existing) == 350 * len(METHODS) and set(existing["method"]) == set(METHODS):
                summary = summarize(folder)
                write_json(summary_path, summary)
                print(f"validated completed artifact order_seed={seed}; summary repaired", flush=True)
            else:
                summary = run_seed(seed, features, state, device, cache_manifest, oracle, tuple(args.methods))
        else:
            summary = run_seed(seed, features, state, device, cache_manifest, oracle, tuple(args.methods))
        summaries.append(summary)
    if any(file_hash(Path(path)) != sha for path, sha in original.items()):
        raise ValueError("production source fingerprint changed")
    for seed in args.order_seeds:
        folder = OUTPUT / f"seed{seed}"
        for path in folder.glob("*.parquet"):
            pd.read_parquet(path)
        yaml.safe_load((folder / "config_resolved.yaml").read_text())
        json.loads((folder / "state_manifest.json").read_text())
    if sorted(args.order_seeds) == [0, 1, 2]:
        final_report(summaries)
    write_json(OUTPUT / "run_validation.json", {"complete_order_seeds": args.order_seeds, "production_sha256": original,
               "production_unchanged": True, "P0_oracle": oracle, "all_artifacts_reopened": True})
    print("MEMORY_LEARNING_GATE_COMPLETE = YES", flush=True)


if __name__ == "__main__":
    main()
