"""Exclusive protocol lock, causal CUDA training, then read-only development scores."""
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
from exps.task_p.p2.data import (ARMS, ORDERS, OUT, P1, P1_SHA, ROOT, SEEDS, NormalReader,
                                Replay, evaluation_features, p1_protocol, require, stream,
                                tensor_bytes, tree_digest)
from exps.task_p.p2.model import Learner
from exps.task_p.p2.preflight import configure


def now():
    return datetime.now(timezone.utc).isoformat()


def load_protocol():
    path = OUT / "protocol_resolved.json"
    require(sha(path) == (OUT / "protocol.sha256").read_text().strip(), "P2 protocol digest mismatch")
    protocol = json.loads(path.read_text())
    for name, expected in protocol["code_sha256"].items():
        require(sha(ROOT / name) == expected, f"frozen source changed: {name}")
    return protocol


def lock():
    inherited = json.loads((OUT / "inheritance.json").read_text())
    feasible = json.loads((OUT / "normal_feasibility.json").read_text())
    require(inherited["passed"] and feasible["device_tensor_sum"] == 4, "preflight gate failed")
    tests = (OUT / "mechanics_tests_lock.txt").read_text()
    require("passed" in tests and "failed" not in tests and "skipped" not in tests, "mechanical tests did not pass")
    original = p1_protocol()
    sources = {str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT / "exps/task_p/p2").glob("*"))
               if path.is_file()}
    sources.update({name: sha(ROOT / name) for name in
        ("exps/task_p/p1_data.py", "exps/task_p/p1_readout.py", "exps/task_p/p1_bootstrap_bounds.py",
         "agents/tasks/current_task_p2.md", "AGENTS.md")})
    dataset = ROOT / "data/mvtec"
    for row in original["evaluations"]:
        require(sha(dataset / row["relative_path"]) == row["sha256"], "development image changed")
        if row["mask_path"]:
            require(sha(dataset / row["mask_path"]) == row["mask_sha256"], "development mask changed")
    protocol = {"schema": "p2_neural_nested_development_v1", "locked_utc": now(),
        "authorization": "user P2 specification and explicit bootstrap instruction; CUDA explicitly authorized",
        "exposure": "DEVELOPMENT EVIDENCE ONLY; consumed P1 MVTec AD v1 fixture, never blind validation",
        "p1_protocol_sha256": P1_SHA, "dataset_root": str(dataset),
        "reference_repository": {"path": "externals/VisionHOPE", "commit": "98cfb5d6004472a0ba3d212d3cf884393e39c134",
            "source_sha256": {name: sha(ROOT / "externals/VisionHOPE" / name)
                for name in ("models/srnl.py", "models/operator.py", "models/srnl_core/csrc/memory.cuh")}},
        "evaluations": original["evaluations"], "sealed_identity_paths": original["sealed_identity_paths"],
        "updates_by_order": {order: stream(original, order) for order in ORDERS},
        "seeds": SEEDS, "orders": ORDERS, "arms": ARMS, "boundaries": [100, 200, 300, 350],
        "architecture": {"encoder": "immutable ViT-B/8 block index 8 raw FP32 784x768; no encoder updates",
            "predictor": "shared 768->64 GELU; four-neighbor mean of latent with replicate padding; concatenate center latent, context latent, fixed xy; 130->128 GELU->768",
            "identity_bottleneck": "64-dimensional projection, no skip; encoding nullspace perturbations cannot be reconstructed",
            "leakage": "center latent and neighbors have contextualized teacher content; not hidden-pixel prediction, no claim of pixel independence",
            "predictor_parameters": feasible["arms"]["D"]["predictor_parameters"],
            "controller_parameters": feasible["arms"]["D"]["controller_parameters"]},
        "optimization": {"cycles_per_image": 2, "support": "even/odd positions alternate by event+cycle; 392 supervised positions",
            "query": "remaining 392 positions, equally weighted current image and one replay image when available; A current only",
            "loss": "mean squared raw feature error, uniform across positions/channels/images; no anomaly labels",
            "adam": "betas .9/.999, epsilon 1e-8, weight decay zero; A/B two standard steps/cycle support then query lr=.001",
            "fast": "persistent detached phi; w=(1-c)*phi+c*theta; exact support gradient through w; differentiable bias-corrected Adam direction with second-moment sqrt epsilon 1e-12 and denominator epsilon 1e-8; phi_next=w-alpha*direction",
            "outer": "normal query/replay loss evaluated at phi_next; exact one-step unroll including Hessian terms; theta Adam lr=.0001; psi Adam lr=.001; truncate graph and moment derivatives between cycles",
            "controller": "shared 4->16 tanh->2 tanh network per parameter tensor; detached RMS(phi), RMS(phi-theta), RMS(momentum), RMS(second moment)^.5; alpha=.001*exp(output0), c=.02*exp(output1); output layer initially zero",
            "C": "same full differentiable one-step outer theta update as D, alpha=.001/c=.02 fixed; controller is frozen and unused; strong fixed-rule bilevel two-timescale control, not a HOPE reproduction",
            "D_noinner": "same support computation/moments and learned controller but phi_next=w; removes inner gradient injection, outer still updates through mixing",
            "clipping": "support and outer global gradient norm<=1; per-tensor inner update RMS<=.02; alpha in [.001/e,.001*e], c in [.02/e,.02*e]",
            "precision": "FP32 for all primary arms, TF32 disabled, deterministic algorithms, no AMP/compile; measured normal-only AMP profile separately",
            "slow_timescale": "slow theta lr10x smaller than base fast lr; same update frequency, different effective step rates; not CMS discrete clocks",
            "replay": "Algorithm R over arrived full feature grids, 24 images FP32=57,802,752 bytes, same draw and contents for B/C/D/D_noinner, inserted only after current-image training; A intentionally zero replay",
            "state": "no 8-MiB cap; charge actual parameters, fast copy, moments, optimizers, controller, replay and RNG; unused A/B buffers disclosed",
            "matching": "identical base predictor, current data, supervision and opportunity; C matches fast/slow capacity and Hessian compute, D adds 114 controller parameters and controller differentiation; A/B lack fast/slow extra state and Hessian compute; report actual counts/timing"},
        "scoring": "L2 norm of raw 768-channel residual per patch; image=max of 784; FP32 28x28 to native bilinear align_corners=False antialias=False, no smoothing/fusion/calibration/task ID",
        "metrics": {"primary": "exact native per-category pixel average precision, equal-category macro at event350",
            "secondary": "Image AUROC, per-category AP, normal/defect separation, acquisition boundary and subsequent old-task losses; bottle 100->300 pre-return and 100->350 reported separately",
            "aggregation": "compute metrics per seed/order first; arithmetic mean metrics over seeds per order; canonical order primary, both-order mean secondary; no score ensembling",
            "bootstrap": "1000 paired whole-image resamples, class-stratified within category, seed102701+category index; identical multiplicities for every arm/run/boundary; exact point AP; corrected P1 65536-bin conservative AP bounds; 95% percentile, mean run metrics inside each replicate; seeds/orders never independent images",
            "strongest": "max over A/B/C macro AP averaged over seeds for canonical primary, recompute max within each paired bootstrap draw; also every pairwise contrast and alternate/pooled result",
            "old_harm": "D minus each category's best eligible control AP at every boundary after acquisition (including bottle pre-return): material harm if point<-0.02; support requires conservative lower interval >-0.02; report forgetting loss trajectories regardless",
            "signal": "D or a conventional arm has >=2 final categories with seed-mean Image AUROC>=.70 and AP>=max(3*prevalence,prevalence+.05), macro image lower CI>.5 and AP-minus-prevalence lower CI>0",
            "supported": "canonical D gain>=.02 and conservative paired lower>0, useful D signal, old-harm condition passed, no negative point gain under alternate order; all validity gates pass",
            "not_supported": "signal exists but D primary point<=0 or upper gain interval<.02; otherwise positive but uncertain/submargin outcome is inconclusive; if no neural signal joint failure is NL_NOT_SUPPORTED; NEURAL_SIGNAL_ONLY when signal exists without demonstrated NL gate and no clear inferiority",
            "decision_scope": "exploratory development screen, no blind/generalization or novelty claim"},
        "external_references": "saved P1 Full784 Greedy/Reservoir only, unmatched historical external numbers, never rerun",
        "code_sha256": sources, "input_verification_sha256": sha(OUT / "inheritance.json"),
        "normal_feasibility_sha256": sha(OUT / "normal_feasibility.json"),
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
            learners = {arm: Learner(arm, seed, "cuda") for arm in ARMS}
            chain = {arm: tree_digest(learner.state()) for arm, learner in learners.items()}
            torch.cuda.reset_peak_memory_stats()
            started, trace, replay_trace, budgets, timings = time.perf_counter(), [], [], {}, {arm: 0.0 for arm in ARMS}
            for row in rows:
                features = reader.get(row)
                current = features.cuda()
                for cycle in range(protocol["optimization"]["cycles_per_image"]):
                    historical, identity = replay.sample()
                    historical_gpu = historical.cuda() if historical is not None else None
                    require(identity is None or identity in [previous["relative_path"] for previous in rows[:row["event"]-1]],
                            "replay accessed unarrived source")
                    replay_trace.append({"event": row["event"], "cycle": cycle, "historical_source": identity})
                    parity = (row["event"] + cycle) % 2
                    support = torch.arange(parity, 784, 2, device="cuda")
                    query = torch.arange(1-parity, 784, 2, device="cuda")
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
                        "replay_tensor_bytes": tensor_bytes(replay_state) if arm != "A" else 0,
                        "inference_tensor_bytes": tensor_bytes(state["model"]),
                        "online_tensor_bytes": tensor_bytes(state) + (tensor_bytes(replay_state) if arm != "A" else 0)}
                        for arm, state in states.items()}
                    print(order, seed, "boundary", boundary, "elapsed", round(time.perf_counter()-started, 1), flush=True)
            resources = {"seconds": time.perf_counter()-started, "update_seconds_by_arm": timings,
                "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "cycles_per_arm": {arm: learner.steps for arm, learner in learners.items()},
                "diagnostic_overhead": "per-cycle GPU synchronization, full parameter/optimizer state hashing and boundary serialization included in wall time, excluded from update_seconds"}
            write_json(directory / "trace.json", trace)
            write_json(directory / "replay_access.json", replay_trace)
            write_json(directory / "normal_access.json", reader.access)
            write_json(directory / "budgets.json", budgets)
            write_json(directory / "resources.json", resources)
            write_json(directory / "completed.json", {"protocol_sha256": digest, "utc": now()})
            del learners, states, replay, replay_state, current, historical_gpu
            torch.cuda.empty_cache()
    write_json(OUT / "training_completed.json", {"protocol_sha256": digest, "utc": now(),
                                                "seconds": time.perf_counter()-overall_start})


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
                checkpoint = torch.load(directory / f"state_t{boundary}.pt", map_location="cpu", weights_only=True)
                require(checkpoint["protocol_sha256"] == digest, "wrong checkpoint protocol")
                admitted = {row["category"] for row in protocol["updates_by_order"][order][:boundary]}
                learners = {arm: Learner(arm, seed, "cuda") for arm in ARMS}
                for arm, learner in learners.items():
                    learner.restore(checkpoint["states"][arm])
                    require(tree_digest(learner.state()) == checkpoint["state_sha256"][arm], "checkpoint state drift")
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
                        duration = time.perf_counter()-started
                        scores = scores.cpu()
                        require(tree_digest(learner.state()) == before, "evaluation mutated learner")
                        timing[arm].append(duration)
                        if row == next(item for item in protocol["evaluations"] if item["category"] in admitted):
                            cpu = Learner(arm, seed)
                            cpu.restore(checkpoint["states"][arm])
                            error = float((cpu.scores(features)-scores).abs().max())
                            require(error < 5e-4, "CPU/GPU boundary scoring parity failure")
                            maximum_error = max(maximum_error, error)
                        records.append({"boundary": boundary, "arm": arm, "source": row["relative_path"],
                                        "patch_scores": scores.tolist(), "image_score": float(scores.max()),
                                        "state_sha256": before})
            write_json(directory / "predictions.json", records)
            write_json(directory / "inference_resources.json", {"seconds_by_arm": timing,
                       "cpu_gpu_max_patch_score_difference": maximum_error, "encoder_extraction": "none; immutable P1 features"})
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
