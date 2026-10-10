# exps/test_hope_normality_audit.py
"""Independent alignment algebra and immutable diagnostic checks."""

from dataclasses import replace

import torch

from exps.hope_image_synchronous_memory import (
    ImageSynchronousMemory, aggregate_image_statistics, fingerprint, local_objective,
    propose_transition,
)
from exps.anomaly.hope_normality_audit import (
    cosine_summary, direction_overlap, fit_teacher_map, fixed_teacher, interimage_cosine_correlation,
    objective_gradient, relational_alignment,
)
from exps.test_hope_image_synchronous_memory import initial, quantities


def test_frozen_snapshot_gradient_matches_independent_autograd():
    q = quantities()
    a = torch.randn(4, 4, dtype=torch.float64)
    z = a.clone().requires_grad_()
    target = (a @ q.values.T).detach()
    pred = z @ q.keys.T
    w = q.gates.T / q.count
    loss = .5 * (((pred - target).square() + pred.square()) * w).sum()
    expected, = torch.autograd.grad(loss, z)
    torch.testing.assert_close(objective_gradient(a, q), expected, atol=1e-12, rtol=1e-12)
    stats = aggregate_image_statistics(q)
    transition, _, _ = propose_transition(stats, controlled=False)
    torch.testing.assert_close(a @ transition, a - .02 * expected, atol=1e-12, rtol=1e-12)


def test_complete_local_objective_does_not_equal_deployed_residual():
    q = quantities()
    a = torch.eye(4, dtype=torch.float64) * 2
    self_target = a @ q.values.T
    assert not torch.equal(self_target, q.values.T)
    assert local_objective(a, a, q)["J"] > 0


def test_cka_invariant_to_orthogonal_channel_change_and_scale():
    torch.manual_seed(2)
    x = torch.randn(20, 4, dtype=torch.float64)
    u, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))
    result = relational_alignment(x, 3 * x @ u)
    assert abs(result["linear_cka"] - 1) < 1e-12
    assert abs(result["cosine_kernel_correlation"] - 1) < 1e-12
    groups = torch.arange(4).repeat_interleave(5)
    assert abs(interimage_cosine_correlation(x, 3 * x @ u, groups) - 1) < 1e-12


def test_teacher_target_is_external_and_detached():
    image = torch.randn(7, 4, requires_grad=True)
    target = fixed_teacher(image)
    assert not target.requires_grad and target.grad_fn is None
    image.data.zero_()
    torch.testing.assert_close(target.norm(dim=-1), torch.ones(7, dtype=torch.float64))


def test_fixed_teacher_decoder_deterministic_no_refitting():
    torch.manual_seed(1)
    x = torch.randn(50, 4, dtype=torch.float64)
    y = x @ torch.randn(4, 4, dtype=torch.float64) + 1
    decoder = fit_teacher_map(x, y)
    other = fit_teacher_map(x, y)
    torch.testing.assert_close(decoder.coefficient, other.coefficient, atol=0, rtol=0)
    before = fingerprint({"coefficient": decoder.coefficient, "mean": decoder.input_mean})
    assert decoder.error(x, y) < 1e-5
    decoder.error(x + .1, y)
    assert fingerprint({"coefficient": decoder.coefficient, "mean": decoder.input_mean}) == before


def test_zero_cosine_and_relations_finite():
    zero = torch.zeros(8, 4)
    assert all(torch.isfinite(torch.tensor(v)) for v in cosine_summary(zero, zero).values())
    assert all(torch.isfinite(torch.tensor(v)) for v in relational_alignment(zero, zero).values())


def test_disjoint_read_write_subspaces_detected():
    q = torch.tensor([[1., 0], [2, 0]])
    k = torch.tensor([[0., 1], [0, 2]])
    delta = torch.tensor([[0., 1], [0, 0]])
    stats = direction_overlap(q, k, delta, dimensions=1)
    assert stats["dominant_subspace_overlap"] == 0
    assert stats["write_energy_in_query_subspace"] == 0
    assert stats["write_energy_in_key_subspace"] == 1


def test_proposal_and_teacher_inspection_leave_full_state_rng_unchanged():
    model = ImageSynchronousMemory(initial(), "P1")
    image = torch.randn(1, 7, 4, dtype=torch.float64)
    before = fingerprint(model.smt.state_dict())
    rng = torch.get_rng_state().clone()
    proposal = model.propose_event(image)
    snapshot = model.snapshot_state()
    objective_gradient(snapshot.weights["memory"], proposal.quantities)
    local_objective(proposal.weights["memory"], snapshot.weights["memory"], proposal.quantities)
    model.evaluate_read_only(image)
    assert fingerprint(model.smt.state_dict()) == before
    assert torch.equal(rng, torch.get_rng_state())
    assert int(model.completed_events) == 0


def test_shared_right_coupling_and_zero_fixed_point():
    model = ImageSynchronousMemory(initial(), "P1")
    image = torch.randn(1, 8, 4, dtype=torch.float64)
    snap = model.snapshot_state()
    proposal = model.propose_event(image)
    for name, weight in snap.weights.items():
        torch.testing.assert_close(proposal.weights[name], weight @ proposal.transition)
    assert torch.count_nonzero(torch.zeros_like(snap.weights["memory"]) @ proposal.transition) == 0
    relation = torch.linalg.solve(snap.weights["k"].T, snap.weights["v"].T).T
    torch.testing.assert_close(relation @ proposal.weights["k"], proposal.weights["v"])


def test_independent_teacher_error_can_worsen_while_local_J_descends():
    q = replace(quantities(n=1, dim=1), keys=torch.ones(1, 1, dtype=torch.float64),
                values=torch.zeros(1, 1, dtype=torch.float64), gates=torch.ones(1, 1, dtype=torch.float64))
    a = torch.ones(1, 1, dtype=torch.float64)
    t, _, _ = propose_transition(aggregate_image_statistics(q), controlled=False)
    b = a @ t
    assert local_objective(b, a, q)["J"] < local_objective(a, a, q)["J"]
    assert ((b - 1) ** 2).sum() > ((a - 1) ** 2).sum()


def test_diagnostic_artifact_roundtrip_preserves_identity(tmp_path):
    import pandas as pd
    rows = [{"class_name": "bottle", "relative_path": "bottle/train/good/150.png",
             "checkpoint": 100, "order_seed": 0, "teacher_q_error": .2}]
    path = tmp_path / "normal.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    assert pd.read_parquet(path).to_dict("records") == rows
