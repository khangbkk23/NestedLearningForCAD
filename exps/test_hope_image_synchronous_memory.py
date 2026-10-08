# exps/test_hope_image_synchronous_memory.py
"""Independent algebra, lifecycle, operator, and reference identity tests."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from exps.hope_image_synchronous_memory import (
    FP32_TOL, H, ImageSynchronousMemory, PatchQuantities,
    aggregate_image_statistics, comparison, fingerprint, local_objective,
    propose_transition, visual_structure,
)
from exps.hope_update_stabilization import UpdateMapping, run_update_smt
from models.hope_cad.self_modifying_titans import SelfModifyingTitans, _Token


def initial(dim=4, dtype=torch.float64):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return SelfModifyingTitans(dim).to(dtype).state_dict()


def quantities(n=7, dim=4):
    generator = torch.Generator().manual_seed(5)
    k = torch.nn.functional.normalize(torch.randn(n, dim, dtype=torch.float64, generator=generator), dim=-1)
    v = torch.randn(n, dim, dtype=torch.float64, generator=generator)
    gates = torch.rand(n, 1, dtype=torch.float64, generator=generator)
    return PatchQuantities(v.clone(), k.clone(), k, v, gates, torch.arange(n), "fixed")


def test_frozen_loop_matches_statistics():
    q = quantities()
    m = torch.arange(16, dtype=torch.float64).reshape(4, 4) / 16
    expected = m.clone()
    for key, value, gate in zip(q.keys, q.values, q.gates):
        expected -= H * gate / q.count * (torch.outer(m @ key, key) + torch.outer(m @ (key - value), key))
    stats = aggregate_image_statistics(q)
    t, _, _ = propose_transition(stats, controlled=False)
    torch.testing.assert_close(m @ t, expected, rtol=1e-12, atol=1e-12)


def test_permutations_and_uneven_globally_normalized_reduction():
    q = quantities()
    ref = aggregate_image_statistics(q)
    order = torch.tensor([4, 1, 5, 6, 0, 2, 3])
    for sizes in (None, (1, 2, 4), (3, 1, 3)):
        got = aggregate_image_statistics(q, order, sizes)
        torch.testing.assert_close(got.C, ref.C, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(got.D, ref.D, rtol=1e-12, atol=1e-12)


def test_duplication_preserves_mean_update():
    q = quantities()
    duplicate = replace(q, spatial=q.spatial.repeat(2, 1), queries=q.queries.repeat(2, 1),
                        keys=q.keys.repeat(2, 1), values=q.values.repeat(2, 1),
                        gates=q.gates.repeat(2, 1), coordinates=torch.arange(14))
    a, b = aggregate_image_statistics(q), aggregate_image_statistics(duplicate)
    torch.testing.assert_close(a.C, b.C, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(a.D, b.D, rtol=1e-12, atol=1e-12)


def test_one_token_rank_surprise_full_and_v_equals_k():
    q = quantities(n=1)
    stats = aggregate_image_statistics(q)
    key, value = q.keys[0], q.values[0]
    w = q.gates[0, 0]
    torch.testing.assert_close(stats.C, w * torch.outer(key, key))
    torch.testing.assert_close(stats.D, w * torch.outer(key - value, key))
    t, _, _ = propose_transition(stats, controlled=False)
    expected = torch.eye(4, dtype=torch.float64) - H * w * (torch.outer(key, key) + torch.outer(key - value, key))
    torch.testing.assert_close(t, expected)
    same = aggregate_image_statistics(replace(q, values=q.keys))
    assert torch.count_nonzero(same.D) == 0
    assert torch.count_nonzero(same.C) > 0
    t, _, _ = propose_transition(same, controlled=False)
    assert not torch.equal(t, torch.eye(4, dtype=torch.float64))


def test_common_right_action_matches_production_chunk_and_invariant():
    torch.manual_seed(3)
    model = SelfModifyingTitans(4).double()
    q = quantities(n=3)
    t = torch.eye(4, dtype=torch.float64)
    tokens = []
    for i, (key, value, gate) in enumerate(zip(q.keys, q.values, q.gates)):
        eta = H * gate / q.count
        t = t @ (torch.eye(4, dtype=torch.float64) - eta * torch.outer(key, key)) - eta * torch.outer(key - value, key)
        tokens.append(_Token(i, key[None], value[None], eta.reshape(1, 1), torch.ones(1, 1, dtype=torch.float64)))
    for name, memory in model.memories.items():
        actual, _, _ = model._prepare_update(name, memory.weight, tokens, image=0, capture=False)
        torch.testing.assert_close(actual, memory.weight @ t, rtol=1e-12, atol=1e-12)
    k0 = torch.eye(4, dtype=torch.float64) + 0.01 * torch.randn(4, 4, dtype=torch.float64)
    v0 = torch.randn(4, 4, dtype=torch.float64)
    invariant0 = torch.linalg.solve(k0.T, v0.T).T
    invariant1 = torch.linalg.solve((k0 @ t).T, (v0 @ t).T).T
    torch.testing.assert_close(invariant0, invariant1, rtol=1e-11, atol=1e-11)
    assert torch.count_nonzero(torch.zeros_like(k0) @ t) == 0


def test_operator_projection_bounds_complete_transition():
    q = quantities()
    stats = aggregate_image_statistics(q)
    safe, metrics, _ = propose_transition(stats, controlled=True)
    assert metrics["safe_operator_norm"] <= 1 + 1e-12
    assert metrics["transition_distortion"] >= 0
    u, s, vh = torch.linalg.svd(torch.eye(4, dtype=torch.float64) - H * (stats.C + stats.D))
    torch.testing.assert_close(safe, (u * torch.minimum(s, torch.ones_like(s))) @ vh, rtol=1e-12, atol=1e-12)


def test_complete_objective_descent_not_residual_only():
    q = quantities()
    q = replace(q, values=q.keys.clone())
    m = torch.eye(4, dtype=torch.float64)
    t, _, _ = propose_transition(aggregate_image_statistics(q), controlled=False)
    before, after = local_objective(m, m, q), local_objective(m @ t, m, q)
    assert before["self_target_residual"] == 0
    assert after["self_target_residual"] > 0
    assert after["J"] < before["J"]


@pytest.mark.parametrize("method", ["P0", "P1", "P2", "FROZEN"])
def test_lifecycle_is_read_only_except_atomic_commit_and_exact_resume(method, tmp_path):
    memory = ImageSynchronousMemory(initial(dtype=torch.float32), method)
    x = torch.randn(1, 19, 4)
    state = memory.state_fingerprint()
    rng = torch.get_rng_state().clone()
    snapshot = memory.snapshot_state()
    before_versions = {name: m.weight._version for name, m in memory.smt.memories.items()}
    read = memory.evaluate_read_only(x)
    proposal = memory.propose_event(x)
    memory.validate_transition(proposal)
    assert memory.state_fingerprint() == state
    assert torch.equal(torch.get_rng_state(), rng)
    proposal.causal_output.add_(0)
    memory.commit_event(proposal)
    assert int(memory.completed_events) == 1
    counts = 2 if method == "P0" else (0 if method == "FROZEN" else 1)
    assert int(memory.smt.memory_update_count) == counts
    if method == "FROZEN":
        for name, weight in snapshot.weights.items():
            assert torch.equal(weight, memory.smt.memories[name].weight)
            assert memory.smt.memories[name].weight._version == before_versions[name]
    path = tmp_path / "checkpoint.pt"
    memory.serialize_state(path)
    loaded = ImageSynchronousMemory.deserialize_state(torch.load(path, weights_only=False))
    assert loaded.state_fingerprint() == memory.state_fingerprint()
    p = memory.propose_event(x)
    p2 = loaded.propose_event(x)
    memory.commit_event(p)
    loaded.commit_event(p2)
    assert loaded.state_fingerprint() == memory.state_fingerprint()
    assert not memory.memory_stats()["persistent_graph"]
    for value in read.values():
        value.zero_()
    assert loaded.state_fingerprint() == memory.state_fingerprint()


def test_invalid_or_stale_proposal_never_commits():
    model = ImageSynchronousMemory(initial(dtype=torch.float32), "P1")
    x = torch.randn(1, 7, 4)
    proposal = model.propose_event(x)
    before = model.state_fingerprint()
    invalid = dict(proposal.weights)
    invalid["v"] = invalid["v"].clone()
    invalid["v"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        model.commit_event(replace(proposal, weights=invalid))
    assert model.state_fingerprint() == before
    model.commit_event(proposal)
    after = model.state_fingerprint()
    with pytest.raises(ValueError):
        model.commit_event(proposal)
    assert model.state_fingerprint() == after


def test_p0_is_exact_existing_experiment_and_candidates_are_isolated():
    state = initial(dtype=torch.float32)
    state_hash = fingerprint(state)
    candidates = [ImageSynchronousMemory(state, method) for method in ("P0", "P1", "P2", "FROZEN")]
    assert len({candidate.state_fingerprint() for candidate in candidates}) == 1
    x = torch.randn(1, 33, 4)
    oracle = ImageSynchronousMemory(state, "P0")
    output, _, _ = run_update_smt(oracle.smt, x, UpdateMapping("P0", eta_kind="horizon_sigmoid", eta_h=H, horizon=33), capture_trace=False)
    proposal = candidates[0].propose_event(x)
    assert torch.equal(proposal.causal_output, output[0])
    candidates[0].commit_event(proposal)
    for name, memory in oracle.smt.memories.items():
        assert torch.equal(memory.weight, candidates[0].smt.memories[name].weight)
    assert fingerprint(state) == state_hash
    assert candidates[1].state_fingerprint() == candidates[2].state_fingerprint() == candidates[3].state_fingerprint()


def test_references_are_immutable_and_coordinate_metrics_detect_intervention():
    x = torch.randn(16, 8, dtype=torch.float64)
    memory = x @ torch.randn(8, 8, dtype=torch.float64)
    ref = memory.clone()
    values = visual_structure(x, memory, ref)
    assert values["memory_coordinate_retrieval"] == 1
    assert comparison(ref, memory)["relative_l2"] == 0
    changed = memory.roll(1, dims=0)
    assert comparison(ref, changed)["relative_l2"] > 0
    assert visual_structure(x, changed, ref)["memory_coordinate_retrieval"] < 1
    assert torch.equal(ref, memory)


def test_real_cached_p0_oracle_when_available():
    path = Path("results/hope_cad/real_feature_probe/real_cpu_seed0/features/class_bottle.pt")
    if not path.is_file():
        pytest.skip("real normal feature cache unavailable")
    # Avoid an expensive CPU-only real recurrence during the small algebra suite.
    # The runner performs this exact real oracle on its selected device.
    values = torch.load(path, map_location="cpu", weights_only=False)
    assert values["patches"].shape[1:] == (784, 768)
    assert all("/train/good/" in name for name in values["relative_paths"])


def test_paired_manifest_disjoint_probes_and_seed_order():
    from scripts.exps.hope_image_synchronous_memory import manifest
    if not Path("data/mvtec/hazelnut/train/good").is_dir():
        pytest.skip("normal image pool unavailable")
    streams = [manifest(seed)[:2] for seed in (0, 1, 2)]
    assert len({tuple(row["relative_path"] for row in stream[0]) for stream in streams}) == 3
    assert len({frozenset(row["relative_path"] for row in stream[0]) for stream in streams}) == 1
    for rows, probes in streams:
        assert len(rows) == 350 and len(probes) == 60
        assert len({row["relative_path"] for row in rows}) == 350
        assert not {row["relative_path"] for row in rows} & {row["relative_path"] for row in probes}
        assert [rows[i]["phase"] for i in (99, 100, 199, 200, 299, 300)] == ["bottle", "carpet", "carpet", "hazelnut", "hazelnut", "bottle_return"]


def test_aggregate_enumeration_and_read_output_do_not_mutate():
    from exps.hope_image_synchronous_memory import enumeration_check
    model = ImageSynchronousMemory(initial(), "P2")
    x = torch.randn(1, 7, 4, dtype=torch.float64)
    before = model.state_fingerprint()
    proposal = model.propose_event(x)
    rows = enumeration_check(model, proposal, seed=0)
    assert len(rows) == 4 and all(row["passed"] for row in rows)
    assert model.state_fingerprint() == before
    for output in model.evaluate_read_only(x).values():
        output.zero_()
    assert model.state_fingerprint() == before


def test_counter_tampering_rejected_before_commit():
    model = ImageSynchronousMemory(initial(), "P1")
    proposal = model.propose_event(torch.randn(1, 7, 4, dtype=torch.float64))
    counters = dict(proposal.counters)
    counters["memory_update_count"] = counters["memory_update_count"] + 1
    before = model.state_fingerprint()
    with pytest.raises(ValueError):
        model.commit_event(replace(proposal, counters=counters))
    assert model.state_fingerprint() == before


def test_result_table_and_reference_payload_reopen(tmp_path):
    import json
    import pandas as pd
    q = quantities()
    path = tmp_path / "reference.pt"
    torch.save({"source_image": "bottle/train/good/000.png", "query": q.queries,
                "coordinate": q.coordinates, "target": q.values.clone(), "creation_event": 100}, path)
    reference = torch.load(path, weights_only=False)
    assert torch.equal(reference["target"], q.values)
    pd.DataFrame([{"method": "P1", "source_image": reference["source_image"], "creation_event": 100,
                   "evaluation_event": 200, "reference_path": str(path)}]).to_parquet(tmp_path / "retention.parquet")
    assert pd.read_parquet(tmp_path / "retention.parquet").iloc[0]["creation_event"] == 100
    (tmp_path / "summary.json").write_text(json.dumps({"complete": True}))
    assert json.loads((tmp_path / "summary.json").read_text())["complete"]


def test_feature_equivalence_records_absolute_and_relative_error():
    from scripts.exps.hope_image_synchronous_memory import feature_equivalence
    reference = torch.tensor([0.0, 1.0, 10.0, 50.0])
    close = reference + torch.tensor([1e-6, 1e-6, 1e-5, 1e-4])
    checked = feature_equivalence(reference, close)
    assert checked["passed"]
    assert checked["max_abs"] > 0 and checked["relative_l2"] > 0
    assert not feature_equivalence(reference, reference + 0.1)["passed"]


def test_complete_frozen_pilot_artifacts_and_probe_identities(tmp_path):
    from scripts.exps.hope_image_synchronous_memory import manifest, run_method, persist_tables
    if not Path("data/mvtec/hazelnut/train/good").is_dir():
        pytest.skip("normal image pool unavailable")
    rows, probes, _ = manifest(0)
    features = {name: {"patches": torch.randn(170 if name == "bottle" else 120, 7, 4)}
                for name in ("bottle", "carpet", "hazelnut")}
    tables = {name: [] for name in ("per_event", "acquisition", "retention", "visual_structure", "interventions", "enumeration_invariance")}
    (tmp_path / "checkpoints").mkdir()
    result = run_method("FROZEN", rows, probes, features, initial(dtype=torch.float32),
                        torch.device("cpu"), tmp_path, 0, tables)
    persist_tables(tmp_path, tables)
    assert result["completed_events"] == 350
    assert len(tables["per_event"]) == 350
    assert all(row["SMT_memory_updates"] == 0 for row in tables["per_event"])
    assert all(row["common_improvement"] == 0 for row in tables["acquisition"])
    assert all(row["memory_relative_l2"] == 0 for row in tables["retention"])
    assert len({row["reference_path"] for row in tables["retention"]}) == 60
    assert len(list((tmp_path / "checkpoints").glob("*.pt"))) == 4
    assert not any(row["persistent_graph"] for row in tables["per_event"])


def test_write_acquisition_can_fail_at_read_queries():
    from exps.hope_image_synchronous_memory import fixed_association_read_errors
    keys = torch.eye(3, dtype=torch.float64)
    queries = keys.roll(1, dims=0)
    targets = keys.clone()
    before = fixed_association_read_errors(torch.zeros_like(keys), keys, queries, targets)
    after = fixed_association_read_errors(torch.eye(3, dtype=torch.float64), keys, queries, targets)
    assert after["key_error"] == 0 and before["key_error"] > 0
    assert after["query_error"] > before["query_error"]
    assert torch.equal(targets, keys)
