"""Non-evaluative integrity, normal-only CUDA feasibility and leakage checks."""
from __future__ import annotations

import argparse
import copy
import json
import time

import torch

from exps.task_p.p1_data import sha, write_json
from exps.task_p.p2.data import (ARMS, FEATURES, OUT, ROOT, NormalReader, p1_protocol,
                                 require, tensor_bytes, tree_digest, verify_inheritance)
from exps.task_p.p2.model import Learner, loss, parameters


def configure():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def audit():
    result = verify_inheritance()
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
                      "preserved": len(result["preserved"]), "state_matches": len(matches)}), flush=True)


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
    require(masked["relative_paths"][index] == protocol["updates"][0]["relative_path"], "masked diagnostic identity")
    difference = (first - masked["patches"][index]).norm(dim=-1)
    centers = torch.tensor([vertical * 28 + horizontal for vertical in (3, 10, 17, 24)
                            for horizontal in (3, 10, 17, 24)])
    outside = torch.ones(784, dtype=torch.bool)
    outside[centers] = False
    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "bf16_supported": torch.cuda.is_bf16_supported(), "device_tensor_sum": float(torch.ones(4, device="cuda").sum()),
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
        learner = Learner(arm, 2711, "cuda")
        before = learner.scores(first_gpu).mean().item()
        validation_before = learner.scores(validation).mean().item()
        torch.cuda.synchronize()
        started = time.perf_counter()
        records = [learner.train_cycle(first_gpu, support, query, first_gpu) for _ in range(16)]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        cpu = Learner(arm, 2711)
        cpu.restore(copy.deepcopy(learner.state()))
        gpu_scores = learner.scores(first_gpu).cpu()
        cpu_scores = cpu.scores(first)
        error = float((gpu_scores - cpu_scores).abs().max())
        require(error < 5e-4, "CPU/GPU residual parity failed")
        require(all(torch.isfinite(torch.tensor(list(record.values()))).all() for record in records), "nonfinite feasibility")
        if arm == "D":
            require(max(record["controller_gradient_norm"] for record in records) > 1e-7, "disconnected controller")
        result["arms"][arm] = {"cycles": 16, "cycle_seconds": elapsed / 16,
            "initial_mean_train_residual": before, "final_mean_train_residual": float(gpu_scores.mean()),
            "initial_mean_normal_validation_residual": validation_before,
            "final_mean_normal_validation_residual": float(learner.scores(validation).mean()),
            "final_record": records[-1], "cpu_gpu_max_score_error": error,
            "predictor_parameters": sum(value.numel() for value in learner.model.parameters()),
            "controller_parameters": sum(value.numel() for value in learner.controller.parameters()),
            "state_tensor_bytes": tensor_bytes(learner.state())}
    profile = {}
    for dtype in (torch.float32, torch.bfloat16):
        learner = Learner("B", 2711, "cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(20):
            learner.optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                objective = loss(learner.model, parameters(learner.model), first_gpu, support)
            objective.backward()
            learner.optimizer.step()
        torch.cuda.synchronize()
        profile[str(dtype)] = {"seconds_per_forward_backward_step": (time.perf_counter() - started) / 20,
                             "final_normal_train_loss": float(objective.detach())}
    result["amp_profile"] = profile
    result["precision_decision"] = "FP32 all arms: preserve one-step higher-order gradient fidelity and common numerical objective; AMP profile disclosed, no memory pressure"
    result["compile_decision"] = "eager, small dynamic functional parameter dictionaries and higher-order graphs; compile not introduced without demonstrated benefit"
    result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    result["source_sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT / "exps/task_p/p2").glob("*.py"))}
    write_json(OUT / "normal_feasibility.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "feasibility"))
    arguments = parser.parse_args()
    audit() if arguments.mode == "audit" else feasibility()
