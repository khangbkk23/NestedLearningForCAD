"""P2d exclusive protocol lock, causal CUDA training, then read-only development scores."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import resource
import time
from datetime import datetime, timezone

import torch

from exps.task_p.p1_data import sha, write_json
from exps.task_p.p2d.data import (ARMS, ORDERS, OUT, P2, P2_PROTOCOL_SHA, P1, P1_SHA,
                                  ROOT, SEEDS, NormalReader, Replay, evaluation_features,
                                  p1_protocol, require, stream, tensor_bytes, tree_digest)
from exps.task_p.p2d.model import Learner2
from exps.task_p.p2d.preflight import configure


def now():
    return datetime.now(timezone.utc).isoformat()


def load_protocol():
    path = OUT / "protocol_resolved.json"
    require(sha(path) == (OUT / "protocol.sha256").read_text().strip(), "P2d protocol digest mismatch")
    protocol = json.loads(path.read_text())
    for name, expected in protocol["code_sha256"].items():
        require(sha(ROOT / name) == expected, f"frozen source changed: {name}")
    return protocol


def lock():
    inherited = json.loads((OUT / "inheritance.json").read_text())
    feasible = json.loads((OUT / "normal_feasibility.json").read_text())
    probe = json.loads((OUT / "probe.json").read_text())
    require(inherited["passed"] and inherited["p2_reference_pins"]["passed"], "inheritance gate failed")
    require(feasible["device_tensor_sum"] == 4, "feasibility gate failed")
    require(probe["passed"], "Stage A probe gate failed; refuse to lock")
    tests = (OUT / "mechanics_tests_lock.txt").read_text()
    require("passed" in tests and "failed" not in tests and "skipped" not in tests,
            "mechanical tests did not pass in full")
    original = p1_protocol()
    sources = {str(path.relative_to(ROOT)): sha(path)
               for path in sorted((ROOT / "exps/task_p/p2d").glob("*.py")) if path.is_file()}
    sources.update({name: sha(ROOT / name) for name in
        ("exps/task_p/p1_data.py", "exps/task_p/p1_readout.py", "exps/task_p/p1_bootstrap_bounds.py",
         "exps/task_p/p2/data.py", "exps/task_p/p2/model.py", "exps/task_p/p2/metrics.py",
         "agents/tasks/current_task_p2d.md", "AGENTS.md")})
    dataset = ROOT / "data/mvtec"
    for row in original["evaluations"]:
        require(sha(dataset / row["relative_path"]) == row["sha256"], "development image changed")
        if row["mask_path"]:
            require(sha(dataset / row["mask_path"]) == row["mask_sha256"], "development mask changed")
    protocol = {"schema": "p2d_replay_anchored_development_v1", "locked_utc": now(),
        "authorization": "user P2d instruction (replay-anchored nested continual learning, execute Stage A fully and Stage B only behind gates); CUDA explicitly authorized",
        "exposure": "DEVELOPMENT EVIDENCE ONLY; the P1 MVTec AD v1 fixture is consumed, never blind validation",
        "p1_protocol_sha256": P1_SHA,
        "p2_reference": {"protocol_sha256": P2_PROTOCOL_SHA,
            "note": "arm B/C/D results are inherited untouched from P2 as external references",
            "pins_file": str((OUT / "inheritance.json").relative_to(ROOT))},
        "dataset_root": str(dataset),
        "evaluations": original["evaluations"], "sealed_identity_paths": original["sealed_identity_paths"],
        "updates_by_order": {order: stream(original, order) for order in ORDERS},
        "seeds": list(SEEDS), "orders": list(ORDERS), "arms": list(ARMS), "boundaries": [100, 200, 300, 350],
        "architecture": {"encoder": "immutable ViT-B/8 block index 8 raw FP32 784x768 P1 features; no encoder updates",
            "predictor": "unchanged P2 predictor (768->64 GELU; four-neighbor latent mean replicate padding; 130->128 GELU->768); 165,056 parameters",
            "controller": "unchanged P2 Controller: shared 4->16 tanh->2 tanh per parameter tensor; zero-initialized output",
            "arms": "C2 fixed gates (controller present but unused), D2 learned gates; both identical two-step update otherwise"},
        "optimization": {"cycles_per_image": 2,
            "direct_steps_per_cycle": 2,
            "support": "even/odd positions alternate by event+cycle; 392 supervised positions (step 1)",
            "query": "remaining 392 positions, equally weighted current image and one replay image when available (step 2)",
            "step1": "s1=stats(phi0,theta,m,v); gates G(s1)=(alpha1=.001*exp(a1), mix1=.02*exp(c1)) or fixed .001/.02; w1=(1-mix1)*phi0+mix1*theta; L1=feature MSE at support positions of the arriving image; gradient at w1, global clip norm<=1; Adam moments .9/.999 updated; bias-corrected direction at step 2k-1 with second-moment sqrt epsilon 1e-12 and denominator epsilon 1e-8; delta1 rms-clipped to .02; phi1=w1-delta1",
            "step2": "s2=stats(phi1,theta,m1,v1); w2=(1-mix2)*phi1+mix2*theta; L2=feature MSE at query positions of the stacked (current,replay) pair when replay exists else current only; same clip/Adam/bias-correction at step 2k; delta2 rms-clipped to .02; phi2=w2-delta2 committed as the deployed state",
            "state": "phi committed detached; moments committed detached; no graph or moment derivative crosses cycles",
            "outer": "L2 evaluated at phi2 trains theta (Adam lr=1e-4) and, in D2 only, the controller (Adam lr=1e-3); exact unroll of both inner steps including Hessian terms; global gradient clip <=1 both groups",
            "controller_ablation": "C2 is the fixed-rule twin (alpha=.001, c=.02) with an unused frozen controller; D2 first cycle is bitwise identical to C2 by zero-initialized output gates",
            "boundedness": "alpha in [.001/e,.001*e], c in [.02/e,.02*e]; per-tensor inner update RMS<=.02; precision FP32, TF32 disabled, deterministic algorithms",
            "replay": "unchanged P2 Algorithm R over arrived full feature grids, 24 images FP32=57,802,752 bytes, same draw and contents for C2 and D2, inserted only after current-image training",
            "request_contract": "step 1 uses only the support half, step 2 only the query half; disjoint sets covering 784 positions; no duplicated gradients and equal 392/392 weighting for current/replay in step 2",
            "design_note": "minimal intervention on P2 C/D: the original one-step update applied twice per cycle with replay entering the deployed state directly; the consolidation pull is retained bounded and Stage A verifies it does not overwrite acquisition"},
        "scoring": "L2 norm of raw 768-channel residual per patch; image=max of 784; FP32 28x28 to native bilinear align_corners=False antialias=False, no smoothing/fusion/calibration/task ID (identical P2 geometry and native masks)",
        "metrics": {"primary": "exact native per-category pixel average precision, equal-category macro at event350; best of {C2,D2} canonical vs decision threshold 0.45",
            "secondary": "Image AUROC, per-category AP, acquisition trajectory 100/200/300/350, forgetting, old-category harm vs controls, separation, compute/storage ledger; arm-B/D/C P2 references",
            "aggregation": "compute metrics per seed/order first; arithmetic mean metrics over seeds per order; canonical order primary, both-order mean secondary; no score ensembling",
            "bootstrap": "1000 paired whole-image resamples, class-stratified within category, seed102701+category index, identical multiplicities for every arm including inherited P2 B/C/D records (pinned deterministic procedure); exact point AP; corrected P1 65536-bin conservative AP bounds; 95% percentile",
            "decision_H1": "H1_SUPPORTED iff max(canonical macro pixel AP of C2,D2) >= 0.45; otherwise H1_FALSIFIED (exploratory threshold from the independent audit)",
            "decision_NL": "LEARNED_NL_SUPPORTED iff D2-C2 canonical point >= +0.02 AND conservative paired 95% lower bound > 0 AND alternate-order point difference >= 0; otherwise LEARNED_NL_NOT_SUPPORTED",
            "old_harm": "D2 minus the best eligible control (C2 or inherited B) per category per boundary after acquisition reported with paired intervals; material harm if point < -0.02; no gating of the primary verdict beyond reporting",
            "verdicts": "P2D_H1_SUPPORTED_NL_SUPPORTED, P2D_H1_SUPPORTED_NL_NOT, P2D_H1_FALSIFIED; validity failure -> P2D_BLOCKED",
            "decision_scope": "exploratory development screen on a consumed fixture; no blind/generalization or novelty claim; H1 recovery is not NL superiority"},
        "stage_a": {"inheritance_sha256": sha(OUT / "inheritance.json"),
            "feasibility_sha256": sha(OUT / "normal_feasibility.json"),
            "probe_sha256": sha(OUT / "probe.json"), "probe_trace_sha256": probe["probe_trace_sha256"],
            "gates": probe["gates"], "last_window_query_loss": probe["last_window_query_loss"],
            "acquisition_gate": probe["acquisition_gate"], "retention": probe["retention_first_ten_mean_residual"],
            "retention_gate": probe["retention_gate"], "travel": probe["travel"],
            "travel_reference_t100": probe["travel_anchor_t100"],
            "conflict_cosine_mean": probe["conflict_cosine"]["mean"]},
        "code_sha256": sources,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "gpu": feasible["gpu"], "threads": 4},
        "stop": "fail closed on provenance/future-source/nonfinite/state chain; no outcome-driven schedule/architecture/seed/order changes; never open sealed payloads"}
    write_json(OUT / "protocol_resolved.json", protocol)
    with (OUT / "protocol.sha256").open("x") as handle:
        handle.write(sha(OUT / "protocol_resolved.json") + "\n")
    print("LOCKED", sha(OUT / "protocol_resolved.json"), flush=True)


def save_exclusive(path, value):
    with path.open("xb") as handle:
        torch.save(value, handle)


def train():
    configure()
    require(torch.cuda.is_available(), "CUDA required for scientific training")
    protocol = load_protocol()
    digest = sha(OUT / "protocol_resolved.json")
    write_json(OUT / "training_started.json", {"started_utc": now(), "protocol_sha256": digest})
    overall_start = time.perf_counter()
    for order in protocol["orders"]:
        for seed in protocol["seeds"]:
            directory = OUT / f"{order}_seed{seed}"
            directory.mkdir(exist_ok=False)
            write_json(directory / "started.json", {"protocol_sha256": digest, "utc": now()})
            rows = protocol["updates_by_order"][order]
            reader, replay = NormalReader(rows), Replay(seed)
            learners = {arm: Learner2(arm, seed, "cuda") for arm in ARMS}
            chain = {arm: tree_digest(learner.state()) for arm, learner in learners.items()}
            torch.cuda.reset_peak_memory_stats()
            started, trace, replay_trace, budgets, timings = (time.perf_counter(), [], [], {},
                                                              {arm: 0.0 for arm in ARMS})
            for row in rows:
                features = reader.get(row)
                current = features.cuda()
                for cycle in range(protocol["optimization"]["cycles_per_image"]):
                    historical, identity = replay.sample()
                    historical_gpu = historical.cuda() if historical is not None else None
                    require(identity is None or identity in [previous["relative_path"]
                                                             for previous in rows[:row["event"] - 1]],
                            "replay accessed unarrived source")
                    replay_trace.append({"event": row["event"], "cycle": cycle, "historical_source": identity})
                    parity = (row["event"] + cycle) % 2
                    support = torch.arange(parity, 784, 2, device="cuda")
                    query = torch.arange(1 - parity, 784, 2, device="cuda")
                    for arm, learner in learners.items():
                        before = tree_digest(learner.state())
                        require(before == chain[arm], "state reset outside causal update")
                        torch.cuda.synchronize()
                        update_start = time.perf_counter()
                        record = learner.train_cycle(current, support, query, historical_gpu)
                        torch.cuda.synchronize()
                        duration = time.perf_counter() - update_start
                        timings[arm] += duration
                        after = tree_digest(learner.state())
                        require(before != after and all(torch.isfinite(torch.tensor(list(record.values())))),
                                "invalid update or nonfinite diagnostics")
                        trace.append(dict(record, event=row["event"], cycle=cycle, arm=arm,
                                          before=before, after=after, seconds=duration))
                        chain[arm] = after
                replay.insert(features, row["relative_path"])
                if row["event"] in protocol["boundaries"]:
                    boundary = row["event"]
                    states = {arm: copy.deepcopy(learner.state()) for arm, learner in learners.items()}
                    replay_state = replay.state()
                    save_exclusive(directory / f"state_t{boundary}.pt",
                                   {"protocol_sha256": digest, "boundary": boundary, "states": states,
                                    "replay": replay_state, "state_sha256": dict(chain)})
                    budgets[str(boundary)] = {arm: {"learner_tensor_bytes": tensor_bytes(state),
                        "replay_tensor_bytes": tensor_bytes(replay_state),
                        "inference_tensor_bytes": tensor_bytes(state["model"]),
                        "online_tensor_bytes": tensor_bytes(state) + tensor_bytes(replay_state)}
                        for arm, state in states.items()}
                    print(order, seed, "boundary", boundary, "elapsed",
                          round(time.perf_counter() - started, 1), flush=True)
            resources = {"seconds": time.perf_counter() - started, "update_seconds_by_arm": timings,
                "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "cycles_per_arm": {arm: learner.cycles for arm, learner in learners.items()},
                "direct_steps_per_arm": {arm: 2 * learner.cycles for arm, learner in learners.items()},
                "diagnostic_overhead": "per-cycle GPU synchronization, full state hashing and boundary serialization included in wall time, excluded from update_seconds"}
            write_json(directory / "trace.json", trace)
            write_json(directory / "replay_access.json", replay_trace)
            write_json(directory / "normal_access.json", reader.access)
            write_json(directory / "budgets.json", budgets)
            write_json(directory / "resources.json", resources)
            write_json(directory / "completed.json", {"protocol_sha256": digest, "utc": now()})
            del learners, states, replay, replay_state, current, historical_gpu
            torch.cuda.empty_cache()
    write_json(OUT / "training_completed.json", {"protocol_sha256": digest, "utc": now(),
                                                 "seconds": time.perf_counter() - overall_start})


def evaluate():
    configure()
    protocol = load_protocol()
    require((OUT / "training_completed.json").exists(), "complete all fixed normal-only runs before evaluation")
    digest = sha(OUT / "protocol_resolved.json")
    write_json(OUT / "evaluation_started.json", {"protocol_sha256": digest, "utc": now()})
    for order in protocol["orders"]:
        for seed in protocol["seeds"]:
            directory = OUT / f"{order}_seed{seed}"
            records, timing, maximum_error = [], {arm: [] for arm in ARMS}, 0.0
            for boundary in protocol["boundaries"]:
                checkpoint = torch.load(directory / f"state_t{boundary}.pt", map_location="cpu",
                                        weights_only=True)
                require(checkpoint["protocol_sha256"] == digest, "wrong checkpoint protocol")
                admitted = {row["category"] for row in protocol["updates_by_order"][order][:boundary]}
                learners = {arm: Learner2(arm, seed, "cuda") for arm in ARMS}
                for arm, learner in learners.items():
                    learner.restore(checkpoint["states"][arm])
                    require(tree_digest(learner.state()) == checkpoint["state_sha256"][arm],
                            "checkpoint state drift")
                first_admitted = next(item for item in protocol["evaluations"]
                                      if item["category"] in admitted)
                for row in protocol["evaluations"]:
                    if row["category"] not in admitted:
                        continue
                    features = evaluation_features(row, protocol)
                    gpu_features = features.cuda()
                    for arm, learner in learners.items():
                        before = tree_digest(learner.state())
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        scores = learner.scores(gpu_features)
                        torch.cuda.synchronize()
                        duration = time.perf_counter() - started
                        scores = scores.cpu()
                        require(tree_digest(learner.state()) == before, "evaluation mutated learner")
                        timing[arm].append(duration)
                        if row == first_admitted:
                            cpu = Learner2(arm, seed)
                            cpu.restore(checkpoint["states"][arm])
                            error = float((cpu.scores(features) - scores).abs().max())
                            require(error < 5e-4, "CPU/GPU boundary scoring parity failure")
                            maximum_error = max(maximum_error, error)
                        records.append({"boundary": boundary, "arm": arm, "source": row["relative_path"],
                                        "patch_scores": scores.tolist(), "image_score": float(scores.max()),
                                        "state_sha256": before})
            write_json(directory / "predictions.json", records)
            write_json(directory / "inference_resources.json", {"seconds_by_arm": timing,
                       "cpu_gpu_max_patch_score_difference": maximum_error,
                       "encoder_extraction": "none; immutable P1 features"})
            print("SCORED", order, seed, len(records), flush=True)
    write_json(OUT / "evaluation_completed.json", {"protocol_sha256": digest, "utc": now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("lock", "train", "evaluate"))
    arguments = parser.parse_args()
    try:
        {"lock": lock, "train": train, "evaluate": evaluate}[arguments.mode]()
    except Exception as error:
        path = OUT / f"{arguments.mode}_failure.json"
        if not path.exists():
            write_json(path, {"utc": now(), "type": type(error).__name__, "message": str(error)})
        raise
