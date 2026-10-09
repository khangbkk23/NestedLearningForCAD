# scripts/exps/hope_normality_audit.py
"""Bounded normal-cache inspection without training or anomaly evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import pandas as pd
import torch
from torch.nn import functional as F
import yaml

from exps.hope_image_synchronous_memory import (
    ImageSynchronousMemory, aggregate_image_statistics, fingerprint, geometry,
    local_objective, propose_transition, fixed_association_read_errors,
)
from exps.hope_normality_audit import (
    FrozenTeacherMap, cosine_summary, direction_overlap, fit_teacher_map, fixed_teacher,
    interimage_cosine_correlation,
    objective_gradient, output_change, relational_alignment,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/hope_cad/memory_learning_gate"
OUTPUT = ROOT / "results/hope_cad/normality_objective_audit"
CLASSES = ("bottle", "carpet", "hazelnut")
LOCKED = {"self_modifying_titans.py": "ffcd8ca7810ade758954effcb708903eb36b182835dac58e0e577cb497e1932c",
          "continuum_memory.py": "920135550843a33d2cc064030bd5155af349dbab0d99cddae213ddc589055581",
          "hope_block.py": "16e9b00cf9bba7e66ceaebe94d2c7f6cf23f95c907bbe59d47d879b16b179437"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@torch.no_grad()
def run(device):
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "summary.json").exists():
        raise FileExistsError("completed diagnostics already exist; inspect them before rerunning")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protected = {str(path.relative_to(ROOT)): sha(path) for path in
                 [ROOT / "models/hope_cad" / name for name in LOCKED]}
    assert list(protected.values()) == list(LOCKED.values())
    for path in (SOURCE / "p0_initial_fixture.pt", SOURCE / "seed0/stream_manifest.parquet",
                 ROOT / "exps/hope_image_synchronous_memory.py", ROOT / "exps/hope_anomaly_signal.py",
                 ROOT / "exps/hope_anomaly_score_ablation.py"):
        protected[str(path.relative_to(ROOT))] = sha(path)
    fixture = torch.load(SOURCE / "p0_initial_fixture.pt", map_location="cpu", weights_only=False)
    initial = ImageSynchronousMemory(fixture["smt"], "FROZEN", device=device)
    initial_snapshot = initial.snapshot_state()
    base = pd.read_parquet(SOURCE / "seed0/stream_manifest.parquet")
    selected = []
    for name in CLASSES:
        for role, subset in (("fit", base[(base.class_name == name) & (base.phase == name)].head(10)),
                             ("probe", base[(base.class_name == name) & (base.role == "probe")].head(10))):
            selected.extend({**row, "audit_role": role} for row in subset.to_dict("records"))
    assert len(selected) == 60
    assert len({row["relative_path"] for row in selected}) == 60
    assert all("/train/good/" in row["relative_path"] for row in selected)
    pd.DataFrame(selected).to_parquet(OUTPUT / "normal_manifest.parquet", index=False)
    cache_manifest = json.loads((SOURCE / "feature_cache_manifest.json").read_text())
    features = {}
    for name in CLASSES:
        path = SOURCE / "features" / f"class_{name}.pt"
        assert sha(path) == cache_manifest["shards"][name]["sha256"]
        protected[str(path.relative_to(ROOT))] = cache_manifest["shards"][name]["sha256"]
        features[name] = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    prepared = {}
    for row in selected:
        shard = features[row["class_name"]]
        assert shard["relative_paths"][row["cache_index"]] == row["relative_path"]
        image = shard["patches"][row["cache_index"]:row["cache_index"] + 1].to(device)
        assert image.shape == (1, 784, 768) and torch.isfinite(image).all()
        quantities = initial.generate_update_quantities(image, initial_snapshot)
        prepared[row["relative_path"]] = (image, quantities, fixed_teacher(image[0]))
    fit = [prepared[r["relative_path"]] for r in selected if r["audit_role"] == "fit"]
    teachers = torch.cat([x[2] for x in fit])
    maps = {}
    for space in ("q", "k"):
        inputs = torch.cat([F.linear(getattr(q, "queries" if space == "q" else "keys"), initial_snapshot.weights["memory"])
                            for _, q, _ in fit])
        maps[space] = fit_teacher_map(inputs, teachers)
    torch.save({space: {"coefficient": m.coefficient.cpu(), "input_mean": m.input_mean.cpu(),
                       "target_mean": m.target_mean.cpu(), "ridge": m.ridge} for space, m in maps.items()},
               OUTPUT / "evaluator_only_teacher_maps.pt")
    config = {"device": str(device), "dtype": "float32 model / float64 diagnostics", "seed": 0,
              "normal_images_per_class": {"fit": 10, "probe": 10}, "new_image_loading": 0,
              "no_committed_training_events": True, "checkpoint_events": [0, 100, 200, 300],
              "order_seeds": [0, 1, 2], "teacher": "unit-L2 frozen ViT x; independent of mutable state",
              "teacher_decoder": "global centered ridge from INITIAL Mq/Mk only; 30 fit train-good images; ridge_fraction=.001",
              "evaluator_future_class_access": "fit pool covers all categories; diagnostics only, never model supervision",
              "frozen_teacher_map_hash": sha(OUTPUT / "evaluator_only_teacher_maps.pt"),
              "effective_rank": "all 784 centered rows, FP64 svdvals; entropy of singular values with eps=1e-12",
              "spectrum_schedule": "first 2 probe images per class and checkpoint/seed",
              "relations": "128 linspace patch indices per image; centered linear CKA and off-diagonal cosine-kernel Pearson",
              "source_initialization_hash": fixture["initialization_hash"], "source_cache": cache_manifest,
              "production_sha256": protected.copy()}
    (OUTPUT / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    geometry_rows, history_rows, objective_rows, overlap_rows, invariants = [], [], [], [], []
    ids = torch.linspace(0, 783, 128, device=device).long()
    initial_key = initial_snapshot.weights["k"].double()
    relation = torch.linalg.solve(initial_key.T, initial_snapshot.weights["v"].double().T).T
    qrelation = torch.linalg.solve(initial_key.T, initial_snapshot.weights["memory"].double().T).T
    checkpoint_models = {}
    for seed in (0, 1, 2):
        for event in (0, 100, 200, 300):
            if event == 0:
                current = ImageSynchronousMemory(fixture["smt"], "P1", device=device)
            else:
                path = SOURCE / f"seed{seed}/checkpoints/P1_event{event}.pt"
                protected[str(path.relative_to(ROOT))] = sha(path)
                current = ImageSynchronousMemory.deserialize_state(torch.load(path, map_location="cpu", weights_only=False), device=device)
                assert int(current.completed_events) == event
                assert int(current.smt.memory_update_count) == event
            checkpoint_models[(seed, event)] = current
            before = current.state_fingerprint()
            snapshot = current.snapshot_state()
            invariants.append({"order_seed": seed, "checkpoint": event,
                               "value_key_closure": float((snapshot.weights["v"].double() - relation @ snapshot.weights["k"].double()).norm() / snapshot.weights["v"].double().norm()),
                               "content_key_closure": float((snapshot.weights["memory"].double() - qrelation @ snapshot.weights["k"].double()).norm() / snapshot.weights["memory"].double().norm()),
                               "initial_key_condition": float(torch.linalg.cond(initial_key)),
                               "state_hash": before, **current.memory_stats()})
            for name in CLASSES:
                probes = [r for r in selected if r["class_name"] == name and r["audit_role"] == "probe"]
                for position, row in enumerate(probes):
                    image, q0, target = prepared[row["relative_path"]]
                    q = current.generate_update_quantities(image, snapshot)
                    mem, mem0 = snapshot.weights["memory"], initial_snapshot.weights["memory"]
                    mk, mq = F.linear(q.keys, mem), F.linear(q.queries, mem)
                    fixed_target = F.linear(q0.values, mem0)
                    errors = fixed_association_read_errors(mem, q0.keys, q0.queries, fixed_target)
                    key0, query0 = F.linear(q0.keys, mem0), F.linear(q0.queries, mem0)
                    item = {"order_seed": seed, "checkpoint": event, "class_name": name, "relative_path": row["relative_path"],
                            "evaluation_state_hash": before, "teacher_q_error": maps["q"].error(mq, target),
                            "teacher_k_current_error": maps["k"].error(mk, target),
                            "teacher_k_fixed_error": maps["k"].error(F.linear(q0.keys, mem), target), **errors}
                    for label, reference, evaluated in (("q_read", query0, mq), ("k_read_fixed", key0, F.linear(q0.keys, mem)),
                                                         ("k_read_current", key0, mk), ("key", q0.keys, q.keys), ("value", q0.values, q.values)):
                        item.update({f"{label}_{metric}": value for metric, value in output_change(reference, evaluated).items()})
                    item.update({f"qk_cos_{metric}": value for metric, value in cosine_summary(q.queries, q.keys).items()})
                    for label, a, b in (("qk", q.queries, q.keys), ("qv", q.queries, q.values), ("kv", q.keys, q.values), ("qx", q.queries, image[0]),
                                        ("kx", q.keys, image[0]), ("mx", mq, image[0])):
                        item.update({f"{label}_{metric}": value for metric, value in relational_alignment(a[ids], b[ids]).items()})
                    history_rows.append(item)
                    if position < 2:
                        for space, tensor in (("x", image[0]), ("q", q.queries), ("k", q.keys), ("v", q.values), ("Mq", mq), ("Mk", mk)):
                            geometry_rows.append({"order_seed": seed, "checkpoint": event, "class_name": name,
                                                  "relative_path": row["relative_path"], "space": space, **geometry(tensor, spectrum=True),
                                                  "average_patch_norm": float(tensor.double().norm(dim=-1).mean())})
            assert current.state_fingerprint() == before
            print(f"normal read-only diagnostics seed={seed} checkpoint={event} complete", flush=True)
        for class_index, name in enumerate(CLASSES):
            current = checkpoint_models[(seed, class_index * 100)]
            before = current.state_fingerprint()
            row = next(r for r in selected if r["class_name"] == name and r["audit_role"] == "fit")
            image, _, target = prepared[row["relative_path"]]
            snapshot = current.snapshot_state()
            proposal = current.propose_event(image)
            q = proposal.quantities
            a, b = snapshot.weights["memory"], proposal.weights["memory"]
            jb, ja = local_objective(a, a, q), local_objective(b, a, q)
            stat = aggregate_image_statistics(q)
            gradient = objective_gradient(a, q)
            analytic = a.double() - .02 * gradient
            item = {"order_seed": seed, "pre_checkpoint": class_index * 100, "class_name": name,
                    "relative_path": row["relative_path"], "proposal_not_committed": True,
                    "J_before": jb["J"], "J_after": ja["J"], "J_fractional_change": ja["J"] / jb["J"] - 1,
                    "gradient_update_relative_error": float((analytic - b.double()).norm() / analytic.norm()),
                    "lipschitz": float(2 * torch.linalg.eigvalsh(stat.C.double()).max())}
            for space, directions in (("q", q.queries), ("k", q.keys)):
                eb, ea = maps[space].error(F.linear(directions, a), target), maps[space].error(F.linear(directions, b), target)
                item.update({f"teacher_{space}_before": eb, f"teacher_{space}_after": ea,
                             f"teacher_{space}_fractional_change": ea / eb - 1})
            objective_rows.append(item)
            overlap_rows.append({"order_seed": seed, "class_name": name, "pre_checkpoint": class_index * 100,
                                 **direction_overlap(q.queries, q.keys, b - a),
                                 "delta_q_over_delta_k_rms": float(F.linear(q.queries, b - a).double().norm() / F.linear(q.keys, b - a).double().norm())})
            assert current.state_fingerprint() == before
    # Reuse prior matched-size carpet-only intervention evidence without replay.
    unrelated_rows = []
    for seed in (0, 1, 2):
        existing = pd.read_parquet(SOURCE / f"seed{seed}/interventions.parquet")
        unrelated_rows.extend(existing[(existing.method == "P1") & (existing.substitution == "unrelated_history")].to_dict("records"))
    tables = {"qkv_geometry": geometry_rows, "history_alignment": history_rows,
              "fixed_teacher_local_objective": objective_rows, "write_read_overlap": overlap_rows,
              "shared_right_invariants": invariants, "existing_unrelated_interventions": unrelated_rows}
    for name, records in tables.items():
        if name == "shared_right_invariants":
            save_json(name + ".json", records)
        else:
            pd.DataFrame(records).to_parquet(OUTPUT / f"{name}.parquet", index=False)
            reopened = pd.read_parquet(OUTPUT / f"{name}.parquet")
            assert len(reopened) == len(records)
            assert reopened.select_dtypes("number").dropna(axis=1).apply(lambda c: c.map(lambda v: bool(torch.isfinite(torch.tensor(v)).item())).all()).all()
    for relative, digest in protected.items():
        assert sha(ROOT / relative) == digest
    save_json("source_integrity.json", {"unchanged": True, "source_sha256": protected,
                                        "confirmation_not_opened": True, "normal_only": True})
    save_json("summary.json", {"status": "COMPLETE", "device": str(device), "normal_identities": 60,
                              "fit_images": 30, "probe_images": 30, "committed_events": 0,
                              "local_proposals": len(objective_rows), "history_rows": len(history_rows),
                              "geometry_rows": len(geometry_rows), "source_unchanged": True,
                              "no_anomaly_data": True, "elapsed_seconds": time.perf_counter() - started,
                              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None})
    print(json.dumps(json.loads((OUTPUT / "summary.json").read_text())), flush=True)


@torch.no_grad()
def supplement(device):
    """Reuse completed diagnostics; add pooled relations and an existing return checkpoint."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    summary = json.loads((OUTPUT / "summary.json").read_text())
    assert summary["status"] == "COMPLETE"
    fixture = torch.load(SOURCE / "p0_initial_fixture.pt", map_location="cpu", weights_only=False)
    initial = ImageSynchronousMemory(fixture["smt"], "FROZEN", device=device)
    snap0 = initial.snapshot_state()
    manifest = pd.read_parquet(OUTPUT / "normal_manifest.parquet")
    probes = manifest[manifest.audit_role == "probe"].to_dict("records")
    caches = {name: torch.load(SOURCE / "features" / f"class_{name}.pt", map_location="cpu", weights_only=False, mmap=True)
              for name in CLASSES}
    prepared = {}
    for row in probes:
        shard = caches[row["class_name"]]
        assert shard["relative_paths"][row["cache_index"]] == row["relative_path"]
        image = shard["patches"][row["cache_index"]:row["cache_index"]+1].to(device)
        prepared[row["relative_path"]] = (image, initial.generate_update_quantities(image, snap0))
    maps = {space: FrozenTeacherMap(value["coefficient"].to(device), value["input_mean"].to(device),
                                    value["target_mean"].to(device), value["ridge"])
            for space, value in torch.load(OUTPUT / "evaluator_only_teacher_maps.pt", map_location="cpu", weights_only=False).items()}
    ids = torch.linspace(0, 783, 32, device=device).long()
    image_ids = torch.arange(10, device=device).repeat_interleave(32)
    rows, returned, source_hashes = [], [], {}
    for seed in (0, 1, 2):
        for event in (0, 100, 200, 300, 350):
            if event == 0:
                model = initial
            else:
                path = SOURCE / f"seed{seed}/checkpoints/P1_event{event}.pt"
                digest = sha(path)
                model = ImageSynchronousMemory.deserialize_state(torch.load(path, map_location="cpu", weights_only=False), device=device)
                assert sha(path) == digest and int(model.completed_events) == event
                source_hashes[str(path.relative_to(ROOT))] = digest
            before = fingerprint(model.smt.state_dict()), model.state_fingerprint()
            snap = model.snapshot_state()
            for name in CLASSES:
                qs, ks, vs = [], [], []
                for row in (r for r in probes if r["class_name"] == name):
                    image, q0 = prepared[row["relative_path"]]
                    q = model.generate_update_quantities(image, snap)
                    qs.append(q.queries[ids]); ks.append(q.keys[ids]); vs.append(q.values[ids])
                    if event == 350:
                        m, m0 = snap.weights["memory"], snap0.weights["memory"]
                        read = F.linear(q.queries, m)
                        returned.append({"order_seed": seed, "checkpoint": event, "class_name": name,
                                         "relative_path": row["relative_path"],
                                         "teacher_q_error": maps["q"].error(read, fixed_teacher(image[0])),
                                         "q_read_relative_l2": output_change(F.linear(q0.queries, m0), read)["relative_l2"],
                                         **fixed_association_read_errors(m, q0.keys, q0.queries, F.linear(q0.values, m0))})
                q, k, v = torch.cat(qs), torch.cat(ks), torch.cat(vs)
                for label, a, b in (("qk", q, k), ("qv", q, v), ("kv", k, v)):
                    rows.append({"order_seed": seed, "checkpoint": event, "class_name": name, "pair": label,
                                 "patch_samples": 320, "images": 10, **relational_alignment(a, b),
                                 "cross_image_cosine_correlation": interimage_cosine_correlation(a, b, image_ids)})
            assert before == (fingerprint(model.smt.state_dict()), model.state_fingerprint())
    for name, records in (("cross_image_relations", rows), ("history_after_return", returned)):
        pd.DataFrame(records).to_parquet(OUTPUT / (name + ".parquet"), index=False)
        assert len(pd.read_parquet(OUTPUT / (name + ".parquet"))) == len(records)
    save_json("supplement_validation.json", {"passed": True, "normal_only": True, "committed_events": 0,
                                            "source_hashes": source_hashes, "relation_rows": len(rows),
                                            "return_probe_rows": len(returned), "no_SVD_replay": True})
    print("pooled cross-image relations and existing return checkpoint inspection complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--supplement", action="store_true")
    args = parser.parse_args()
    (supplement if args.supplement else run)(torch.device(args.device))
