# scripts/exps/hope_outer_learning_stage0_v1.py
"""Bounded synthetic CPU parity, derivative, lifecycle and geometry checks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import tracemalloc
import weakref

import torch
from torch.nn import functional as F

from exps.hope_image_synchronous_memory import ImageSynchronousMemory, aggregate_image_statistics
from exps.ol03.hope_outer_learning_p1_v1 import (
    CENTERS, MEMORIES, checkpoint_payload, detached_parameters, functional_image_event,
    functional_state_to_detached_snapshot, functional_support_sequence,
    initial_functional_state, load_checkpoint, masked_synthetic_rgb, mock_patch_backbone,
    parameter_fingerprint, read_query_without_update, read_with_transition,
    save_checkpoint, state_fingerprint, synthetic_parameters, tensor_inventory,
)
from models.hope_cad.self_modifying_titans import SelfModifyingTitans


ROOT = Path(__file__).resolve().parents[3]
LOCKED_SOURCES = (
    "models/hope_cad/self_modifying_titans.py", "models/hope_cad/continuum_memory.py",
    "models/hope_cad/hope_block.py", "exps/hope_image_synchronous_memory.py",
    "exps/hope_update_stabilization.py", "exps/test_hope_image_synchronous_memory.py",
)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in LOCKED_SOURCES}


def install_synthetic_io_guard(root=ROOT):
    """Fail before opening dataset, backbone checkpoint or prior-result files."""
    root = Path(root).resolve()
    excluded = (root / "data", root / "checkpoints")
    result_root = root / "results"
    allowed_result = result_root / "hope_cad/outer_learning_stage0"
    observations = {"forbidden_access_attempts": 0, "policy": "synthetic_cpu_only"}

    def audit(event, args):
        if event not in ("open", "os.listdir", "os.scandir") or not args:
            return
        value = args[0]
        if not isinstance(value, (str, bytes, Path)):
            return
        path = Path(value.decode() if isinstance(value, bytes) else value).resolve()
        forbidden = any(path.is_relative_to(base) for base in excluded)
        forbidden |= path.is_relative_to(result_root) and not path.is_relative_to(allowed_result)
        if forbidden:
            observations["forbidden_access_attempts"] += 1
            raise RuntimeError("real-data or prior-result access is excluded")

    sys.addaudithook(audit)
    return observations


def synthetic_inputs(parameters, lengths=(1, 7, 19), *, seed=11):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    support = tuple(torch.randn(1, n, parameters.config.dim,
                                dtype=parameters.a0.dtype, generator=generator) for n in lengths)
    query = torch.randn(1, 5, parameters.config.dim, dtype=parameters.a0.dtype, generator=generator)
    target = torch.randn(5, parameters.config.dim, dtype=parameters.a0.dtype, generator=generator)
    return support, query, target


def detached_oracle(parameters):
    # The isolated module is a synthetic forward oracle, never a live model.
    with torch.random.fork_rng(devices=[]):
        module = SelfModifyingTitans(parameters.config.dim).to(dtype=parameters.a0.dtype)
    payload = module.state_dict()
    values = {"memory": parameters.a0, **parameters.auxiliary}
    for name, value in values.items():
        payload[f"memories.{name}.initial_weight"] = value.detach().clone()
        payload[f"memories.{name}.weight"] = value.detach().clone()
    payload["base_q.weight"] = parameters.wq.detach().clone()
    payload["local_conv.weight"] = parameters.conv_weight.clone()
    payload["local_conv.bias"] = parameters.conv_bias.clone()
    return ImageSynchronousMemory(payload, "P1", device="cpu")


def check_forward_parity(*, dtype=torch.float64, seed=0, lengths=(1, 7, 19, 784)):
    parameters = synthetic_parameters(seed=seed, dtype=dtype)
    support, query, _ = synthetic_inputs(parameters, lengths)
    oracle = detached_oracle(parameters)
    source_before = oracle.state_fingerprint()
    state = initial_functional_state(parameters)
    comparisons = []
    rtol, atol = (1e-11, 1e-12) if dtype == torch.float64 else (1e-5, 1e-6)

    def compare(name, actual, expected, event):
        torch.testing.assert_close(actual.detach(), expected.detach(), rtol=rtol, atol=atol)
        difference = (actual.detach() - expected.detach()).abs()
        comparisons.append({"field": name, "event": event, "max_abs": float(difference.max()),
                            "bitwise_equal": bool(torch.equal(actual.detach(), expected.detach()))})

    for index, image in enumerate(support, 1):
        oracle_before = oracle.state_fingerprint()
        snapshot = oracle.snapshot_state()
        quantities = oracle.generate_update_quantities(image, snapshot)
        stats = aggregate_image_statistics(quantities)
        proposal = oracle.propose_event(image)
        state, event = functional_image_event(parameters, state, image)
        for field in ("spatial", "queries", "keys", "values", "gates"):
            compare(field, getattr(event.projection, field), getattr(quantities, field), index)
        compare("C", event.C, stats.C, index)
        compare("D", event.D, stats.D, index)
        compare("transition", event.transition, proposal.transition, index)
        compare("pre_image_read", event.pre_image_read, proposal.causal_output, index)
        assert oracle.state_fingerprint() == oracle_before
        oracle.commit_event(proposal)
        for name in MEMORIES:
            compare(name, state.weights[name], oracle.smt.memories[name].weight, index)
        assert asdict(state.counters) == {"memory": index, "auxiliary": index,
                                          "online": 2 * index, "completed": index}
        assert int(oracle.smt.memory_update_count) == index
        assert int(oracle.smt.auxiliary_update_count) == index
        assert int(oracle.smt.online_update_count) == 2 * index
        assert int(oracle.completed_events) == index
    before = oracle.state_fingerprint()
    compare("post_support_read", read_query_without_update(parameters, state, query),
            oracle.evaluate_read_only(query)["memory"], len(support))
    assert oracle.state_fingerprint() == before and source_before != before
    return {"dtype": str(dtype), "rtol": rtol, "atol": atol, "passed": True,
            "comparisons": comparisons, "max_abs": max(row["max_abs"] for row in comparisons),
            "all_bitwise_equal": all(row["bitwise_equal"] for row in comparisons),
            "counters": asdict(state.counters)}


def synthetic_loss(parameters, support, query, target):
    result = functional_support_sequence(parameters, support)
    prediction = read_query_without_update(parameters, result.state, query)
    return 0.5 * (prediction - target).square().sum(dim=-1).mean(), result


def check_gradients(*, seed=0):
    parameters = synthetic_parameters(dim=3, seed=seed, dtype=torch.float64)
    support, query, target = synthetic_inputs(parameters, (3, 5, 2), seed=17)
    target_before = target.clone()
    loss, result = synthetic_loss(parameters, support, query, target)
    a_gradient, q_gradient, final_gradient = torch.autograd.grad(
        loss, (parameters.a0, parameters.wq, result.state.weights["memory"])
    )
    expected = final_gradient @ result.product.T
    torch.testing.assert_close(a_gradient, expected, rtol=1e-11, atol=1e-12)
    assert not result.product.requires_grad and result.product.grad_fn is None
    assert torch.equal(target, target_before) and not target.requires_grad
    assert all(not event.transition.requires_grad for event in result.events)
    records = {}
    epsilon = 1e-6
    for name, gradient in (("A0", a_gradient), ("Wq", q_gradient)):
        original = parameters.a0 if name == "A0" else parameters.wq
        numerical = torch.empty_like(original)
        for index in range(original.numel()):
            basis = F.one_hot(torch.tensor(index), original.numel()).to(original.dtype).reshape_as(original)
            values = []
            for sign in (1, -1):
                value = original.detach() + sign * epsilon * basis
                changed = replace(parameters, **{("a0" if name == "A0" else "wq"): value})
                with torch.no_grad():
                    value_loss, _ = synthetic_loss(changed, support, query, target)
                values.append(float(value_loss))
            numerical.reshape(-1)[index] = (values[0] - values[1]) / (2 * epsilon)
        absolute = (numerical - gradient).abs()
        stable = torch.maximum(numerical.abs(), gradient.abs()) >= 1e-6
        relative = absolute[stable] / torch.maximum(numerical.abs(), gradient.abs())[stable]
        assert bool(stable.any()) and float(relative.max()) <= 1e-4
        assert float(absolute.max()) <= 1e-7
        assert float(gradient.norm()) > 1e-8
        records[name] = {"max_relative_error": float(relative.max()),
                         "median_relative_error": float(relative.median()),
                         "max_absolute_error": float(absolute.max()),
                         "zero_gradient_fraction": float((gradient == 0).double().mean()),
                         "stable_components": int(stable.sum()), "components": original.numel(),
                         "gradient_norm": float(gradient.norm()), "passed": True}
    assert all(value.grad is None and not value.requires_grad for value in parameters.auxiliary.values())
    assert parameters.conv_weight.grad is None and parameters.conv_bias.grad is None
    return {"passed": True, "finite_difference_step": epsilon, "relative_limit": 1e-4,
            "absolute_limit": 1e-7, "analytic_a0_max_abs": float((a_gradient - expected).abs().max()),
            "parameters": records, "support_images": len(support),
            "transition_dependency_on_a0_wq": False, "teacher_requires_grad": False}


def _disposable_episode(seed):
    parameters = synthetic_parameters(seed=seed)
    support, query, target = synthetic_inputs(parameters)
    loss, result = synthetic_loss(parameters, support, query, target)
    refs = [weakref.ref(loss), weakref.ref(result.state.weights["memory"]),
            weakref.ref(parameters.a0), weakref.ref(parameters.wq)]
    refs.extend(weakref.ref(event.pre_image_read) for event in result.events)
    loss.backward()
    return refs


def check_graph_lifecycle(repeats=30):
    gc.collect()
    tracemalloc.start()
    rss = []
    for index in range(repeats):
        refs = _disposable_episode(index)
        gc.collect()
        assert all(ref() is None for ref in refs)
        rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"passed": True, "episodes": repeats, "all_weak_references_released": True,
            "python_traced_current_bytes": current, "python_traced_peak_bytes": peak,
            "max_rss_bytes_first": rss[0], "max_rss_bytes_last": rss[-1],
            "rss_limitations": "Linux ru_maxrss is a cumulative high-water mark; allocator caching is not proof of live graphs."}


def check_restart(output: Path):
    parameters = synthetic_parameters()
    support, query, _ = synthetic_inputs(parameters)
    identities = tuple(hashlib.sha256(image.numpy().tobytes()).hexdigest() for image in support)
    order = tuple(range(len(support)))
    prefix = functional_support_sequence(parameters, support[:2])
    payload = checkpoint_payload(parameters, prefix.state, input_identities=identities, order=order)
    path = output / "synthetic_checkpoint.pt"
    save_checkpoint(path, payload)
    loaded, state = load_checkpoint(path, expected_identities=identities, expected_order=order)
    resumed = functional_support_sequence(loaded, support[2:], state=state)
    uninterrupted = functional_support_sequence(parameters, support)
    assert state_fingerprint(resumed.state) == state_fingerprint(uninterrupted.state)
    torch.testing.assert_close(read_query_without_update(loaded, resumed.state, query),
                               read_query_without_update(parameters, uninterrupted.state, query),
                               rtol=0, atol=0)
    assert all(not value.requires_grad and value.grad_fn is None for value in state.weights.values())
    return {"passed": True, "input_identities": identities, "order": order,
            "checkpoint_event": prefix.state.counters.completed,
            "final_counters": asdict(resumed.state.counters), "forward_bitwise_equal": True,
            "serialized_graph": False, "optimizer_state": False}


def check_mask_and_confound():
    generator = torch.Generator().manual_seed(31)
    rgb = torch.randn(1, 3, 224, 224, generator=generator)
    original = rgb.clone()
    view, mask = masked_synthetic_rgb(rgb)
    assert int(mask.sum()) == 144 * 64
    assert torch.equal(rgb, original)
    assert bool((view[:, :, mask] == 0).all())
    teacher, student = mock_patch_backbone(rgb), mock_patch_backbone(view)
    assert teacher.shape == student.shape == (1, 784, 3)
    assert bool((student[:, [r * 28 + c for r, c in CENTERS]] == 0).all())
    parameters = synthetic_parameters()
    _, query, _ = synthetic_inputs(parameters)
    transition = 0.7 * torch.eye(parameters.config.dim, dtype=parameters.a0.dtype)
    prediction = read_with_transition(parameters, query, transition)
    static = replace(parameters, a0=parameters.a0 @ transition)
    static_prediction = read_query_without_update(static, initial_functional_state(static), query)
    torch.testing.assert_close(prediction, static_prediction, rtol=0, atol=0)
    return {"mask_passed": True, "view_contract": "ONE_SHARED_SIXTEEN_HOLE_VIEW",
            "centers": CENTERS, "masked_pixels": int(mask.sum()), "mock_only": True,
            "vit_contextual_leakage_tested": False, "confound_passed": True,
            "constant_transition_absorption_max_abs": float((prediction - static_prediction).detach().abs().max()),
            "useful_history_claim": False}


def run_validation(output: Path, *, seed=0):
    started = time.perf_counter()
    if output.exists() and any(output.iterdir()):
        raise ValueError("result directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    checks = {
        "forward_fp64": check_forward_parity(seed=seed),
        "forward_fp32": check_forward_parity(dtype=torch.float32, seed=seed),
        "gradients": check_gradients(seed=seed),
        "checkpoint_restart": check_restart(output),
        "mask_and_confound": check_mask_and_confound(),
        "graph_lifecycle": check_graph_lifecycle(),
    }
    assert source_hashes() == hashes
    summary = {"status": "COMPLETED", "scope": "synthetic_cpu_correctness", "device": "cpu",
               "seed": seed, "python": sys.executable, "torch": torch.__version__,
               "cpu_threads": torch.get_num_threads(), "wall_seconds": time.perf_counter() - started,
               "max_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               "source_hashes_before": hashes, "source_hashes_after": source_hashes(),
               "production_core_unchanged": True, "real_data_accessed": False,
               "gpu_used": False, "outer_optimizer_steps": 0, "stage1_authorized": False,
               "config": {"dim": 4, "h": 0.02, "normalization_eps": 1e-8,
                          "trainable_subset": ["A0", "Wq"], "mask_view": "shared_sixteen_holes"},
               "checks": {name: (value.get("passed", value.get("mask_passed", False)))
                          for name, value in checks.items()}}
    for name, value in checks.items():
        (output / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for path in output.glob("*.json"):
        json.loads(path.read_text())
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run synthetic CPU memory correctness checks.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/hope_cad/outer_learning_stage0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("CPU thread count must be positive")
    observations = install_synthetic_io_guard()
    torch.set_num_threads(args.threads)
    print("Synthetic CPU validation started; no dataset, GPU or optimizer.", flush=True)
    summary = run_validation(args.output_dir, seed=args.seed)
    summary["data_access_guard"] = observations
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "wall_seconds": summary["wall_seconds"],
                      "checks": summary["checks"], "output": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
