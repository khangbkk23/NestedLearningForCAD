"""S3 mechanics tests (plan §5.1). Tiny step counts on real normal train tokens; no anomaly data."""
import math

import pytest
import torch

from exps.cmsb import bc as BC, cms as C, step as ST
from exps.nar_pilot_00 import core as K, full as R

TASKS = ST.ORDER[:3]


@pytest.fixture(scope='module')
def env():
    R.configure()
    tl = ST.train_lock()
    load = ST.loader(tl)
    data = {c: load(c) for c in TASKS}
    return data, ST.encoder_fn()


def eq(a, b):
    return all(torch.equal(a[n][0], b[n][0]) and torch.equal(a[n][1], b[n][1]) for n in a)


def run(env, lam, n, kf=3, kc=4, seed=2711, hook=None, learner=None, start=0):
    data, enc = env
    lr = learner or BC.BCLearner(seed, lam, kf=kf, kc=kc, encode=enc)
    traces = []
    for c in TASKS[start:n]:
        f, roles, rows = data[c]
        roles = dict(roles, FIT=roles['FIT'][:12])          # few reservoir writes keep the test fast
        traces.append(lr.task(c, f, roles, rows, diag_every=2, teacher_hook=hook))
    return lr, traces


def test_copy_fast_learns_slow_frozen_teacher_isolated(env):
    lr, tr = run(env, 1., 2)
    assert eq(lr.last_fast_init, lr.last_slow_before)                 # slow -> fast copy
    assert eq(lr.last_slow_after_fast, lr.last_slow_before)           # slow frozen during fast phase
    assert not eq(lr.last_teacher, lr.last_fast_init)                 # fast learned
    assert eq(lr.teacher_after, lr.last_teacher) and lr.teacher_grads_none   # stop-gradient teacher
    assert not eq(BC.K.get_state(lr.loras_s), lr.last_slow_before)    # slow consolidated at the boundary


def test_transfer_gradient_nonzero_and_lambda_zero_ignores_teacher(env):
    lr, tr = run(env, 1., 1)
    assert all(d['grad_norm_t'] > 0 for d in tr[0]['diag'])

    def perturb(loras):
        with torch.no_grad():
            for w in loras.values():
                w.B.add_(0.01)
    a0, _ = run(env, 0., 2)
    b0, _ = run(env, 0., 2, hook=perturb)
    assert eq(K.get_state(a0.loras_s), K.get_state(b0.loras_s))       # lam=0: teacher has no influence
    a1, _ = run(env, 1., 2)
    b1, _ = run(env, 1., 2, hook=perturb)
    assert not eq(K.get_state(a1.loras_s), K.get_state(b1.loras_s))   # lam=1: teacher changes the slow level


def test_causal_replay_reset_counters_and_access(env):
    lr, tr = run(env, 1., 3, kf=3, kc=4)
    assert tr[0]['reservoir_tags_at_consolidation'] == []
    assert tr[1]['reservoir_tags_at_consolidation'] == [TASKS[0]]
    assert tr[2]['reservoir_tags_at_consolidation'] == sorted(TASKS[:2])
    assert lr.accessed == list(TASKS)
    init = K.init_state(lr.loras_f, 2711)
    assert eq(K.get_state(lr.loras_f), init)                           # fast reset after the boundary
    c1, c2 = lr.counters[0], lr.counters[1]
    assert c1['dec_fb_current'] == 3 + 4 and c1['dec_fb_replay'] == 0 and c1['teacher_fwd'] == 4
    assert c2['dec_fb_current'] == 3 + 4 and c2['dec_fb_replay'] == 4 and c2['encoder_fwd'] == 4
    assert c2['slow_steps'] == 4 and c2['fast_steps'] == 3


def test_save_reload_equivalence(env):
    full, _ = run(env, 1., 2)
    part, _ = run(env, 1., 1)
    st = part.state()
    fresh = BC.BCLearner(2711, 1., kf=3, kc=4, encode=env[1])
    fresh.load(st)
    fresh.accessed = list(part.accessed)
    resumed, _ = run(env, 1., 2, learner=fresh, start=1)
    assert eq(K.get_state(full.loras_s), K.get_state(resumed.loras_s))


def test_memory_invariant_and_ledger(env):
    lr, _ = run(env, 1., 1)
    led = lr.ledger()
    assert led['slow_optimizer_state'] == 3 * led['slow_adapter_fp32']
    sizes = []
    for tasks in (12, 24):
        r = C.Reservoir(BC.REPLAY, 1)
        for t in range(tasks):
            for _ in range(300):
                r.offer(b'z' * 110_000, t)
        sizes.append(r.bytes())
        total = led['slow_adapter_fp32'] + led['slow_optimizer_state'] + r.bytes() + 65_536
        assert r.bytes() <= BC.REPLAY and total <= 48 * 2**20
    assert led['deploy_adapter_bytes_bf16'] <= 6 * 2**20


def test_one_decoder_pass_per_image(env):
    lr, _ = run(env, 1., 1)
    feats = env[0][TASKS[0]][0][:10].clone()
    calls = []
    from exps.vitb8_q2.common import SingleStreamDecoder
    orig = SingleStreamDecoder.forward

    def counted(self, f):
        calls.append(len(f))
        return orig(self, f)
    SingleStreamDecoder.forward = counted
    try:
        g = BC.stage_grids([K.get_state(lr.loras_s)], feats)
    finally:
        SingleStreamDecoder.forward = orig
    assert g.shape == (1, 10, 28, 28) and sum(calls) == 10
