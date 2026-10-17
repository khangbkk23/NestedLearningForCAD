"""Normal-free numerical contract tests, before any new anomaly scoring."""
import ast
import copy
import inspect
from collections import OrderedDict

import numpy as np
import pytest
import torch

from exps.task_p.p2.model import loss, parameters
from exps.task_p.p2d import model as original
from exps.task_p.p2e import model as intervention
from exps.task_p.p2e.analysis import classify
from exps.task_p.p2e.data import ROOT, Replay, configure, tree_digest


def tiny(device='cpu'):
    return intervention.Learner2('E', 91, device, channels=8, latent=2, hidden=5, side=4)


def independent_update(learner, features, replay, support, query):
    fast = OrderedDict((name, value.clone()) for name, value in learner.fast.items())
    moments = {name: (first.clone(), second.clone()) for name, (first, second) in learner.moments.items()}
    slow = parameters(learner.model)
    for step_index, (data, positions) in enumerate(((features, support), (torch.stack((features, replay)), query)), start=1):
        point = OrderedDict((name, value+0.0*slow[name]) for name, value in fast.items())
        objective = loss(learner.model, point, data, positions)
        gradients = torch.autograd.grad(objective, tuple(point.values()), create_graph=True)
        norm = sum(gradient.square().sum() for gradient in gradients).sqrt()
        factor = (1.0/(norm+1e-12)).clamp(max=1.0)
        next_fast = OrderedDict()
        for (name, value), gradient in zip(point.items(), gradients):
            gradient = gradient*factor
            first, second = moments[name]
            first = first*.9 + gradient*.1
            second = second*.999 + gradient.square()*.001
            step = 2*learner.cycles+step_index
            delta = .001*(first/(1-.9**step))/((second/(1-.999**step)+1e-12).sqrt()+1e-8)
            delta = delta*(.02/(delta.square().mean().sqrt()+1e-12)).clamp(max=1.0)
            next_fast[name] = value-delta
            moments[name] = first, second
        fast = next_fast
    return fast, moments


def test_executable_source_diff_is_one_mix_literal():
    old = ast.parse((ROOT / 'exps/task_p/p2d/model.py').read_text())
    new = ast.parse((ROOT / 'exps/task_p/p2e/model.py').read_text())
    changed = 0
    for node in ast.walk(new):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'mix' for target in node.targets):
            assert isinstance(node.value, ast.Call) and node.value.args[-1].value == 0.0
            node.value.args[-1].value = .02
            changed += 1
    assert changed == 1
    for tree in (old, new):
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
    assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)


def test_two_steps_match_independent_reference():
    configure()
    learner = tiny()
    learner.model.double()
    learner.controller.double()
    learner.fast = OrderedDict((name, value.detach().clone()) for name, value in parameters(learner.model).items())
    learner.moments = {name: (torch.zeros_like(value), torch.zeros_like(value)) for name, value in learner.fast.items()}
    features, replay = torch.randn(16, 8, dtype=torch.float64), torch.randn(16, 8, dtype=torch.float64)
    support, query = torch.arange(0, 16, 2), torch.arange(1, 16, 2)
    expected, expected_moments = independent_update(learner, features, replay, support, query)
    actual, moments, diagnostics = intervention.inner_update2(learner.model, learner.fast, parameters(learner.model),
        learner.controller, features, support, query, replay, learner.moments, 1, learned=False)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=1e-14)
        for first, second in zip(moments[name], expected_moments[name]):
            torch.testing.assert_close(first, second, rtol=0, atol=1e-14)
    assert diagnostics['mix_min'] == diagnostics['mix_max'] == 0.0


def test_restored_literal_reproduces_c2_and_only_it():
    function = ast.parse(inspect.getsource(intervention.inner_update2))
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'mix' for target in node.targets):
            node.value.args[-1].value = .02
    namespace = dict(vars(intervention))
    exec(compile(function, '<single-literal-C2-restoration>', 'exec'), namespace)
    learner = tiny()
    features, replay = torch.randn(16, 8), torch.randn(16, 8)
    arguments = (learner.model, learner.fast, parameters(learner.model), learner.controller,
                 features, torch.arange(8), torch.arange(8, 16), replay, learner.moments, 1)
    expected = original.inner_update2(*arguments, learned=False)
    restored = namespace['inner_update2'](*arguments, learned=False)
    assert tree_digest(expected) == tree_digest(restored)


def test_slow_and_controller_interventions_have_zero_effect():
    learner = tiny()
    features, replay = torch.randn(16, 8), torch.randn(16, 8)
    support, query = torch.arange(8), torch.arange(8, 16)
    slow = parameters(learner.model)
    arguments = (learner.model, learner.fast, slow, learner.controller, features, support, query, replay, learner.moments, 1)
    following, moments, record = intervention.inner_update2(*arguments, learned=False)
    objective = loss(learner.model, following, torch.stack((features, replay)), query)
    gradients = torch.autograd.grad(objective, tuple(slow.values()))
    assert all(torch.isfinite(gradient).all() and not bool(gradient.any()) for gradient in gradients)
    altered = OrderedDict((name, value.detach().clone().add_(torch.randn_like(value)*10).requires_grad_()) for name, value in slow.items())
    with torch.no_grad():
        for value in learner.controller.parameters():
            value.add_(torch.randn_like(value)*10)
    changed, changed_moments, changed_record = intervention.inner_update2(learner.model, learner.fast, altered,
        learner.controller, features, support, query, replay, learner.moments, 1, learned=False)
    assert tree_digest(following) == tree_digest(changed)
    assert tree_digest(moments) == tree_digest(changed_moments)
    before = torch.func.functional_call(learner.model, following, (features,))
    after = torch.func.functional_call(learner.model, changed, (features,))
    assert torch.equal(before, after)


def test_serialization_resume_controller_inactive_and_read_only(tmp_path):
    learner = tiny()
    slow = tree_digest(learner.model.state_dict())
    controller = tree_digest(learner.controller.state_dict())
    features, replay = torch.randn(16, 8), torch.randn(16, 8)
    support, query = torch.arange(8), torch.arange(8, 16)
    learner.train_cycle(features, support, query, replay)
    torch.save(learner.state(), tmp_path / 'state.pt')
    restored = tiny()
    restored.restore(torch.load(tmp_path / 'state.pt', weights_only=True))
    assert tree_digest(restored.state()) == tree_digest(learner.state())
    for instance in (learner, restored):
        instance.train_cycle(features, support, query, replay)
    assert tree_digest(restored.state()) == tree_digest(learner.state())
    assert slow == tree_digest(learner.model.state_dict())
    assert controller == tree_digest(learner.controller.state_dict())
    assert all(not value.requires_grad for value in learner.fast.values())
    assert all(not value.requires_grad for pair in learner.moments.values() for value in pair)
    digest = tree_digest(learner.state())
    learner.scores(features)
    assert tree_digest(learner.state()) == digest
    assert all(value.grad is None for value in learner.controller.parameters())


def test_replay_and_objective_weights():
    replay = Replay(2711, capacity=3)
    generator = torch.Generator().manual_seed(11711)
    identities = []
    for event in range(1, 13):
        for cycle in range(2):
            actual = replay.sample()[1]
            expected = None if not identities else identities[int(torch.randint(len(identities), (1,), generator=generator))]
            assert actual == expected and actual != str(event)
        replay.insert(torch.ones(16, 8)*event, str(event))
        if len(identities) < 3:
            identities.append(str(event))
        else:
            slot = int(torch.randint(event, (1,), generator=generator))
            if slot < 3:
                identities[slot] = str(event)
        assert replay.identities == identities
    learner = tiny()
    current, old = torch.randn(16, 8), torch.randn(16, 8)
    positions = torch.arange(1, 16, 2)
    assert set(positions.tolist()).isdisjoint(torch.arange(0, 16, 2).tolist())
    pooled = loss(learner.model, parameters(learner.model), torch.stack((current, old)), positions)
    separate = .5*(loss(learner.model, parameters(learner.model), current, positions)+loss(learner.model, parameters(learner.model), old, positions))
    torch.testing.assert_close(pooled, separate)


def test_cuda_update_and_score_parity():
    assert torch.cuda.is_available(), 'Full CUDA test suite is mandatory'
    configure()
    cpu, gpu = tiny(), tiny('cuda')
    gpu.restore(copy.deepcopy(cpu.state()))
    current, replay = torch.randn(16, 8), torch.randn(16, 8)
    for instance, device in ((cpu, 'cpu'), (gpu, 'cuda')):
        record = instance.train_cycle(current.to(device), torch.arange(8, device=device),
                                      torch.arange(8, 16, device=device), replay.to(device))
        assert np.isfinite(list(record.values())).all()
        assert record['outer_gradient_norm'] == record['controller_gradient_norm'] == 0
    for name in cpu.fast:
        torch.testing.assert_close(cpu.fast[name], gpu.fast[name].cpu(), rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(cpu.scores(current), gpu.scores(current.cuda()).cpu(), rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize('macro,bottle,band,verdict', [
    (.48, False, 'LOW_RECOVERY', 'P2E_SLOW_PULL_NOT_MAJOR'),
    (.55, False, 'PARTIAL_RECOVERY', 'P2E_PARTIAL_EFFECT'),
    (.61, True, 'NEAR_B_RECOVERY', 'P2E_SLOW_PULL_MAJOR_COST'),
    (.61, False, 'NEAR_B_RECOVERY', 'P2E_PARTIAL_EFFECT'),
    (.48, True, 'LOW_RECOVERY', 'P2E_PARTIAL_EFFECT')])
def test_decision_bands(macro, bottle, band, verdict):
    result = classify(macro, {'difference': .1, 'ci': [.01, .2]},
        {'difference': .06 if bottle else .01, 'ci': [.01, .1]},
        {'low_threshold': .4823408991976236, 'near_B_threshold': .5967970317027004})
    assert result['macro_band'] == band and result['verdict'] == verdict
