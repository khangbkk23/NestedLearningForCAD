"""P2d non-evaluative integrity, normal-only CUDA feasibility and the acquisition probe.

Stage A reads only normal training caches and hash-pinned P2 reference artifacts.
No anomaly feature tensor and no native mask is touched here.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import OrderedDict

import torch

from exps.task_p.p1_data import sha, write_json
from exps.task_p.p2.model import Learner as P2Learner
from exps.task_p.p2.model import loss, parameters
from exps.task_p.p2d.data import (ARMS, FEATURES, OUT, P2, ROOT, NormalReader, Replay,
                                  p1_protocol, p2_reference_pins, require, stream,
                                  tensor_bytes, tree_digest, verify_inheritance)
from exps.task_p.p2d.model import Learner2, statistics

ACQUISITION_FACTOR = 1.5
RETENTION_FACTOR = 1.5
CYCLES_PER_IMAGE = 2


def configure():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def audit():
    result = verify_inheritance()
    pins = p2_reference_pins()
    result["p2_reference_pins"] = {"passed": pins["passed"],
                                   "p2_protocol_sha256": pins["p2_protocol_sha256"],
                                   "verified": len(pins["checks"])}
    p0c = ROOT / "results/hope_cad/task_p/p0c"
    p1 = ROOT / "results/hope_cad/task_p/p1"
    matches = {}
    for boundary in (100, 200, 300, 350):
        current = torch.load(p1 / f"states_t{boundary}.pt", weights_only=True, mmap=True)
        original = torch.load(p0c / f"states_t{boundary}.pt", weights_only=True, mmap=True)["models"]
        for old_name, new_name in (("GRID16_GREEDY_REFERENCE", "GRID16_GREEDY"),
                                   ("FULL784_GREEDY", "FULL784_GREEDY"),
                                   ("FULL784_RESERVOIR", "FULL784_RESERVOIR")):
            require(tree_digest(original[old_name]) == tree_digest(current[new_name]),
                    f"P1/P0c boundary state differs: {boundary}/{new_name}")
            matches[f"{boundary}/{new_name}"] = True
    result["p1_p0c_bitwise_state_matches"] = matches
    write_json(OUT / "inheritance.json", result)
    print(json.dumps({"passed": True, "checks": len(result["checks"]),
                      "preserved": len(result["preserved"]), "state_matches": len(matches),
                      "p2_references": len(pins["checks"])}), flush=True)


def feasibility():
    configure()
    require(torch.cuda.is_available(), "CUDA unavailable; do not substitute a CPU scientific run")
    torch.cuda.reset_peak_memory_stats()
    protocol = p1_protocol()
    reader = NormalReader(protocol["updates"][:8])
    first = reader.get(protocol["updates"][0])
    validation = torch.stack([reader.get(row) for row in protocol["updates"][1:8]]).cuda()
    masked = torch.load(FEATURES / "masked_rgb16/bottle_train.pt", weights_only=True, mmap=True)
    index = protocol["updates"][0]["cache_index"]
    require(masked["relative_paths"][index] == protocol["updates"][0]["relative_path"],
            "masked diagnostic identity")
    difference = (first - masked["patches"][index]).norm(dim=-1)
    centers = torch.tensor([vertical * 28 + horizontal for vertical in (3, 10, 17, 24)
                            for horizontal in (3, 10, 17, 24)])
    outside = torch.ones(784, dtype=torch.bool)
    outside[centers] = False
    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "device_tensor_sum": float(torch.ones(4, device="cuda").sum()),
              "leakage_diagnostic": {"source": protocol["updates"][0]["relative_path"],
                  "intervention": "inherited pre-backbone RGB16 simultaneous masks; normal training image only",
                  "center_feature_change_mean": float(difference[centers].mean()),
                  "outside_centers_feature_change_mean": float(difference[outside].mean()),
                  "outside_centers_changed_fraction": float((difference[outside] > 1e-5).float().mean()),
                  "interpretation": "globally contextualized tokens respond to removed pixels; post-token masking cannot establish hidden-pixel independence"},
              "arms": {}}
    support = torch.arange(0, 784, 2, device="cuda")
    query = torch.arange(1, 784, 2, device="cuda")
    first_gpu = first.cuda()
    for arm in ARMS:
        learner = Learner2(arm, 2711, "cuda")
        before = float(learner.scores(first_gpu).mean())
        validation_before = float(learner.scores(validation).mean())
        torch.cuda.synchronize()
        started = time.perf_counter()
        records = [learner.train_cycle(first_gpu, support, query, first_gpu) for _ in range(16)]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        cpu = Learner2(arm, 2711)
        cpu.restore(copy.deepcopy(learner.state()))
        gpu_scores = learner.scores(first_gpu).cpu()
        cpu_scores = cpu.scores(first)
        error = float((gpu_scores - cpu_scores).abs().max())
        require(error < 5e-4, "CPU/GPU residual parity failed")
        require(all(bool(torch.isfinite(torch.tensor(list(record.values()))).all())
                    for record in records), "nonfinite feasibility")
        if arm == "D2":
            require(max(record["controller_gradient_norm"] for record in records) > 1e-7,
                    "disconnected controller")
        result["arms"][arm] = {"cycles": 16, "cycle_seconds": elapsed / 16,
            "initial_mean_train_residual": before, "final_mean_train_residual": float(gpu_scores.mean()),
            "initial_mean_normal_validation_residual": validation_before,
            "final_mean_normal_validation_residual": float(learner.scores(validation).mean()),
            "final_record": records[-1], "cpu_gpu_max_score_error": error,
            "predictor_parameters": sum(value.numel() for value in learner.model.parameters()),
            "controller_parameters": sum(value.numel() for value in learner.controller.parameters()),
            "state_tensor_bytes": tensor_bytes(learner.state())}
    result["precision_decision"] = ("FP32 both arms: two-step higher-order gradient fidelity, "
                                    "bitwise first-cycle C2/D2 identity and common numerical objective")
    result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    result["source_sha256"] = {str(path.relative_to(ROOT)): sha(path)
                               for path in sorted((ROOT / "exps/task_p/p2d").glob("*.py"))}
    write_json(OUT / "normal_feasibility.json", result)
    print(json.dumps({"gpu": result["gpu"],
                      "arms": {arm: {"initial": value["initial_mean_train_residual"],
                                     "final": value["final_mean_train_residual"],
                                     "validation_final": value["final_mean_normal_validation_residual"],
                                     "parity": value["cpu_gpu_max_score_error"],
                                     "seconds_per_cycle": value["cycle_seconds"]}
                               for arm, value in result["arms"].items()}}, indent=2), flush=True)


def reference_anchors():
    """Recompute the pinned replay-Adam B anchors from hash-verified P2 artifacts."""
    integrity = json.loads((P2 / "artifact_integrity.json").read_text())["files"]
    trace_path = P2 / "canonical_seed2711/trace.json"
    require(sha(trace_path) == integrity["canonical_seed2711/trace.json"]["sha256"], "P2 B trace changed")
    trace = json.loads(trace_path.read_text())
    window = [row for row in trace if row["arm"] == "B" and 91 <= row["event"] <= 100]
    require(len(window) == 20, "P2 B trace window incomplete")
    query_loss = sum(row["query_loss"] for row in window) / len(window)
    state_path = P2 / "canonical_seed2711/state_t100.pt"
    require(sha(state_path) == integrity["canonical_seed2711/state_t100.pt"]["sha256"],
            "P2 B t100 state changed")
    checkpoint = torch.load(state_path, map_location="cpu", weights_only=True)
    rows = stream(p1_protocol(), "canonical")[:100]
    reader = NormalReader(rows)
    first_ten = [reader.get(row) for row in rows[:10]]
    reference = P2Learner("B", 2711)
    reference.restore(checkpoint["states"]["B"])
    retention = float(reference.scores(torch.stack(first_ten)).mean())
    fresh = P2Learner("B", 2711)
    travel = float(torch.sqrt(sum((checkpoint["states"]["B"]["model"][name].detach() - value).square().sum()
                                  for name, value in parameters(fresh.model).items())))
    return {"query_loss_events_91_100": query_loss, "first_ten_mean_residual_t100": retention,
            "travel_at_t100": travel, "trace_sha256": sha(trace_path), "state_t100_sha256": sha(state_path)}


def conflict_cosine(learner, current, replay, support, query):
    """Exact cosine between the support and stacked query gradients at the arm's own eval point."""
    with torch.enable_grad():
        slow = parameters(learner.model)
        names = list(slow)
        sample = statistics(learner.fast, slow, learner.moments, names)
        if learner.arm == "D2":
            rate, mix = learner.controller(sample)
        else:
            rate = sample.new_full((len(names),), 0.001)
            mix = sample.new_full((len(names),), 0.02)
        weights = OrderedDict((name, (1 - mix[i]) * learner.fast[name] + mix[i] * slow[name])
                              for i, name in enumerate(names))
        g_support = torch.autograd.grad(loss(learner.model, weights, current, support),
                                        tuple(weights.values()), retain_graph=True)
        stacked = torch.stack((current, replay))
        g_query = torch.autograd.grad(loss(learner.model, weights, stacked, query),
                                      tuple(weights.values()))
        flat_s = torch.cat([g.flatten() for g in g_support])
        flat_q = torch.cat([g.flatten() for g in g_query])
        return float(torch.dot(flat_s, flat_q) / (flat_s.norm() * flat_q.norm() + 1e-12))


def probe():
    configure()
    require(torch.cuda.is_available(), "CUDA unavailable; do not substitute a CPU scientific run")
    anchors = reference_anchors()
    acquisition_gate = ACQUISITION_FACTOR * anchors["query_loss_events_91_100"]
    retention_gate = RETENTION_FACTOR * anchors["first_ten_mean_residual_t100"]
    rows = stream(p1_protocol(), "canonical")[:100]
    reader, replay = NormalReader(rows), Replay(2711)
    learners = {arm: Learner2(arm, 2711, "cuda") for arm in ARMS}
    initial = {arm: {name: value.clone() for name, value in learner.fast.items()}
               for arm, learner in learners.items()}
    initial_norm = {arm: float(torch.sqrt(sum(value.square().sum()
                                              for value in learner.fast.values())))
                    for arm, learner in learners.items()}
    chain = {arm: tree_digest(learner.state()) for arm, learner in learners.items()}
    first_ten, last_ten, trace, timings = [], [], [], {arm: 0.0 for arm in ARMS}
    for row in rows:
        features = reader.get(row)
        current = features.cuda()
        if row["event"] <= 10:
            first_ten.append(features.clone())
        if row["event"] > 90:
            last_ten.append(features.clone())
        for cycle in range(CYCLES_PER_IMAGE):
            historical, identity = replay.sample()
            historical_gpu = historical.cuda() if historical is not None else None
            require(identity is None or identity in [previous["relative_path"]
                                                     for previous in rows[:row["event"] - 1]],
                    "replay accessed unarrived source")
            parity = (row["event"] + cycle) % 2
            support = torch.arange(parity, 784, 2, device="cuda")
            query = torch.arange(1 - parity, 784, 2, device="cuda")
            for arm, learner in learners.items():
                before = tree_digest(learner.state())
                require(before == chain[arm], "state reset outside causal update")
                torch.cuda.synchronize()
                started = time.perf_counter()
                record = learner.train_cycle(current, support, query, historical_gpu)
                torch.cuda.synchronize()
                duration = time.perf_counter() - started
                timings[arm] += duration
                after = tree_digest(learner.state())
                require(before != after and all(torch.isfinite(torch.tensor(list(record.values())))),
                        "invalid update or nonfinite diagnostics")
                chain[arm] = after
                with torch.no_grad():
                    current_query = float(loss(learner.model, learner.fast, current, query))
                    replay_query = (float(loss(learner.model, learner.fast, historical_gpu, query))
                                    if historical_gpu is not None else None)
                entry = dict(record, event=row["event"], cycle=cycle, arm=arm, seconds=duration,
                             before=before, after=after, current_query_at_deployed=current_query,
                             replay_query_at_deployed=replay_query)
                if historical_gpu is not None and row["event"] % 10 == 0 and cycle == 0:
                    entry["support_query_conflict_cosine"] = conflict_cosine(
                        learner, current, historical_gpu, support, query)
                trace.append(entry)
        replay.insert(features, row["relative_path"])

    def window(arm, metric, low, high):
        values = [entry[metric] for entry in trace if entry["arm"] == arm and low <= entry["event"] <= high]
        return sum(values) / len(values)

    first_window = {arm: window(arm, "query_loss", 1, 10) for arm in ARMS}
    last_window = {arm: window(arm, "query_loss", 91, 100) for arm in ARMS}
    retention = {arm: float(learners[arm].scores(torch.stack(first_ten).cuda()).mean()) for arm in ARMS}
    acquisition_residual = {arm: float(learners[arm].scores(torch.stack(last_ten).cuda()).mean())
                            for arm in ARMS}
    replay_channel = {arm: {"first": window(arm, "replay_query_at_deployed", 11, 20),
                            "last": window(arm, "replay_query_at_deployed", 91, 100)} for arm in ARMS}
    current_channel = {arm: {"first": window(arm, "current_query_at_deployed", 1, 10),
                             "last": window(arm, "current_query_at_deployed", 91, 100)} for arm in ARMS}
    travel = {arm: float(torch.sqrt(sum((learners[arm].fast[name] - initial[arm][name]).square().sum()
                                        for name in learners[arm].fast))) for arm in ARMS}
    final_norm = {arm: float(torch.sqrt(sum(value.square().sum()
                                            for value in learners[arm].fast.values()))) for arm in ARMS}
    distance = {arm: max(entry["fast_slow_distance"] for entry in trace if entry["arm"] == arm)
                for arm in ARMS}
    controller = {arm: {"min_gradient_norm": min(entry["controller_gradient_norm"]
                                                 for entry in trace if entry["arm"] == arm),
                        "moved": tree_digest(learners[arm].controller.state_dict()) != tree_digest(
                            Learner2(arm, 2711, "cuda").controller.state_dict()),
                        "rate_min": min(entry["rate_min"] for entry in trace if entry["arm"] == arm),
                        "rate_max": max(entry["rate_max"] for entry in trace if entry["arm"] == arm),
                        "mix_min": min(entry["mix_min"] for entry in trace if entry["arm"] == arm),
                        "mix_max": max(entry["mix_max"] for entry in trace if entry["arm"] == arm)}
                  for arm in ARMS}
    conflicts = [entry["support_query_conflict_cosine"] for entry in trace
                 if "support_query_conflict_cosine" in entry]
    gates = {
        "acquisition": (any(last_window[arm] <= acquisition_gate for arm in ARMS)
                        and all(last_window[arm] <= 0.5 * first_window[arm] for arm in ARMS)),
        "retention": all(retention[arm] <= retention_gate for arm in ARMS),
        "stability": (all(final_norm[arm] < 3 * initial_norm[arm] for arm in ARMS)
                      and all(distance[arm] < 20 for arm in ARMS)),
        "controller": (controller["D2"]["moved"] and controller["D2"]["min_gradient_norm"] > 1e-9
                       and controller["C2"]["min_gradient_norm"] == 0.0)}
    result = {"passed": all(gates.values()), "gates": gates, "anchors": anchors,
              "acquisition_gate": acquisition_gate, "retention_gate": retention_gate,
              "first_window_query_loss": first_window, "last_window_query_loss": last_window,
              "retention_first_ten_mean_residual": retention,
              "acquisition_last_ten_mean_residual": acquisition_residual,
              "replay_channel": replay_channel, "current_channel": current_channel,
              "travel": travel, "travel_anchor_t100": anchors["travel_at_t100"],
              "initial_parameter_norm": initial_norm, "final_parameter_norm": final_norm,
              "max_fast_slow_distance": distance, "controller": controller,
              "conflict_cosine": {"values": conflicts, "mean": sum(conflicts) / len(conflicts),
                                  "min": min(conflicts), "max": max(conflicts)},
              "seconds_per_cycle": {arm: timings[arm] / (len(rows) * CYCLES_PER_IMAGE) for arm in ARMS},
              "cycles": len(rows) * CYCLES_PER_IMAGE, "events": len(rows)}
    write_json(OUT / "probe_trace.json", trace)
    result["probe_trace_sha256"] = sha(OUT / "probe_trace.json")
    write_json(OUT / "probe.json", result)
    print(json.dumps({"passed": result["passed"], "gates": gates,
                      "last_window_query_loss": last_window, "gate": acquisition_gate,
                      "retention": retention, "retention_gate": retention_gate,
                      "travel": travel, "travel_reference": anchors["travel_at_t100"],
                      "conflict_mean": result["conflict_cosine"]["mean"],
                      "seconds_per_cycle": result["seconds_per_cycle"]}, indent=2), flush=True)
    require(result["passed"], "Stage A probe gates failed; stop before anomaly evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "feasibility", "probe"))
    arguments = parser.parse_args()
    {"audit": audit, "feasibility": feasibility, "probe": probe}[arguments.mode]()
