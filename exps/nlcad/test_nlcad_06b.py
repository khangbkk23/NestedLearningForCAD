# exps/nlcad/test_nlcad_06b.py
"""CPU checks for NLCAD-06B-MECH (tiny decoder and synthetic tensors only: no image, no checkpoint, no GPU)."""
from __future__ import annotations

import copy

import numpy as np
import torch

from exps.dinomaly import official as O
from exps.nlcad import closed_loop_06b as M
from exps.nlcad.fast import nl_train_stage
from exps.nlcad.test_nlcad import feats, state, tiny

DIM = 32


def rand_U(r, seed, dim=DIM):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(dim, r, generator=g, dtype=torch.float64))[0]


def layer_bases(r, seed):
    return {l: rand_U(r, seed + j) for j, l in enumerate(M.LAYERS)}


def fake_step(dec, seed):
    """Stand-in for opt.step(): add a random update to every attn.proj weight (returns the updates)."""
    g = torch.Generator().manual_seed(seed)
    mods = dict(dec.named_modules())
    out = {}
    with torch.no_grad():
        for l in M.LAYERS:
            d = torch.randn(DIM, DIM, generator=g) * 1e-3
            mods[l].weight.add_(d)
            out[l] = d
    return out


def weights(dec):
    mods = dict(dec.named_modules())
    return {l: mods[l].weight.detach().clone() for l in M.LAYERS}


def test_tiny_decoder_has_the_eight_layers():
    mods = dict(tiny().named_modules())
    assert all(l in mods and tuple(mods[l].weight.shape) == (DIM, DIM) for l in M.LAYERS)


def test_projection_acts_on_the_input_side_with_exact_doses():
    U = layer_bases(6, 10)
    for alpha in (0.5, 1.0):
        dec = tiny()
        w0 = weights(dec)
        op = M.Protector(dec, "rep", alpha, U, U, "cpu")
        op.before()
        d = fake_step(dec, 3)
        op.after(1e-3)
        w1 = weights(dec)
        for l in M.LAYERS:
            dp = (w1[l] - w0[l]).double()
            Ul = U[l]
            x_par = Ul @ torch.randn(6, dtype=torch.float64)                  # input inside span(U)
            x_perp = torch.randn(DIM, dtype=torch.float64)
            x_perp -= Ul @ (Ul.T @ x_perp)                                    # input orthogonal to U
            assert torch.allclose(dp @ x_perp, d[l].double() @ x_perp, atol=1e-8)
            assert torch.allclose(dp @ x_par, (1 - alpha) * d[l].double() @ x_par, atol=1e-8)
            ratio = float((dp @ Ul).norm() / (d[l].double() @ Ul).norm())
            assert abs(ratio - (1 - alpha)) < 1e-4
        chk = M.operator_checks(op.table(), "rep", alpha)
        assert M.operator_ok(chk, "rep", alpha), chk
        k = {c: i for i, c in enumerate(M.COLS)}
        tab = op.table()[0]
        assert torch.allclose(tab[:, k["removed"]], alpha ** 2 * tab[:, k["e_act"]], rtol=1e-4)


def test_norm_matches_the_rep05_retained_norm_and_keeps_direction():
    U = layer_bases(9, 20)
    dec = tiny()
    w0 = weights(dec)
    op = M.Protector(dec, "norm", 0.5, U, U, "cpu")
    op.before()
    d = fake_step(dec, 4)
    op.after(1e-3)
    w1 = weights(dec)
    for l in M.LAYERS:
        dd, dp = d[l].double(), (w1[l] - w0[l]).double()
        target = (dd - 0.5 * dd @ U[l] @ U[l].T).norm()
        assert abs(float(dp.norm() / target) - 1) < 1e-4
        assert float(torch.nn.functional.cosine_similarity(dp.flatten(), dd.flatten(), dim=0)) > 1 - 1e-6
    assert M.operator_ok(M.operator_checks(op.table(), "norm", 0.5), "norm", 0.5)


def test_identity_operator_never_writes():
    U = layer_bases(5, 30)
    dec = tiny()
    op = M.Protector(dec, "id", 0.0, U, U, "cpu")
    op.before()
    fake_step(dec, 5)
    before = weights(dec)
    op.after(1e-3)
    after = weights(dec)
    assert all(torch.equal(before[l], after[l]) for l in M.LAYERS)
    chk = M.operator_checks(op.table(), "id", 0.0)
    assert chk["removed_max"] == 0.0 and M.operator_ok(chk, "id", 0.0)


def test_optimizer_state_untouched_and_weight_decay_accounting():
    torch.manual_seed(0)
    dec = tiny()
    x = feats(4, 1)
    en, de = dec(x[:, 0], x[:, 1])
    O.global_cosine(en, de).backward()
    opt = O.make_optimizer(dec)
    lr = 2e-3
    for g in opt.param_groups:
        g["lr"] = lr
    U = layer_bases(7, 40)
    op = M.Protector(dec, "rep", 1.0, U, U, "cpu")
    w0 = weights(dec)
    op.before()
    opt.step()
    w_opt = weights(dec)
    snap = copy.deepcopy({id(p): {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s.items()}
                          for p, s in opt.state.items()})
    op.after(lr)
    for p, s in opt.state.items():
        for k, v in s.items():
            ref = snap[id(p)][k]
            assert (torch.equal(v, ref) if torch.is_tensor(v) else v == ref)
    mods = dict(dec.named_modules())
    wd = O.RECIPE["weight_decay"]
    b1, b2 = O.RECIPE["betas"]
    for l in M.LAYERS:
        p = mods[l].weight
        s = opt.state[p]
        denom = (s["max_exp_avg_sq"].sqrt() / (1 - b2) ** 0.5) + O.RECIPE["eps"]
        scale = max(1.0, float((p.grad / denom).norm() / p.grad.numel() ** 0.5))
        adam = -(lr / (1 - b1) / scale) * s["exp_avg"] / denom
        delta = (w_opt[l] - w0[l]).double()
        pred = (-lr * wd * w0[l] + adam).double()
        floor = 2 * torch.finfo(torch.float32).eps * float(w0[l].abs().max())     # FP32 storage of W, not of Delta
        assert float((delta - pred).abs().max()) <= floor
    k = {c: i for i, c in enumerate(M.COLS)}
    tab = op.table()[0]
    expect = torch.tensor([float((w0[l].double() * lr * wd).pow(2).sum()) for l in M.LAYERS], dtype=torch.float64)
    assert torch.allclose(tab[:, k["e_decay"]], expect, rtol=1e-5)


def _run(loop, seed=7, iters=12):
    torch.manual_seed(seed)
    dec = tiny(seed=1)
    cur, rep = feats(20, 2), feats(10, 3)
    torch.manual_seed(seed + 1)
    gen = torch.Generator().manual_seed(seed + 2)
    loop(dec, cur, rep, gen, iters)
    return state(dec), gen.get_state(), torch.get_rng_state()


def test_identity_loop_with_operator_and_probes_matches_the_locked_loop():
    ref = _run(lambda dec, cur, rep, gen, it: nl_train_stage(dec, cur, it, gen, rep, device="cpu", autocast=False))
    U = layer_bases(4, 50)
    probe_x = feats(5, 9)
    seen = []

    def mine(dec, cur, rep, gen, it):
        op = M.Protector(dec, "id", 0.0, U, U, "cpu")

        def probe(i):
            M.measure(dec, probe_x, "cpu", pool=2)
            M.covs(dec, probe_x, "cpu")
            seen.append(i)
        M.train_stage_protected(dec, cur, it, gen, rep, op=op, probe=probe, every=4, device="cpu", autocast=False)

    out = _run(mine)
    assert seen == [4, 8, 12]
    assert all(torch.equal(out[0][k], ref[0][k]) for k in ref[0])
    assert torch.equal(out[1], ref[1]) and torch.equal(out[2], ref[2])


def test_projection_arm_changes_only_attn_proj_weights_after_the_first_update():
    """State after step 2 (step 1 has lr = 0 under the WarmCosine warm-up; captured by the probe hook; the ID loop
    equals the locked loop, test above)."""
    U = layer_bases(4, 60)
    snaps = {}

    def loop(kind, alpha):
        def f(dec, cur, rep, gen, it):
            def probe(i):
                if i == 2:
                    snaps[kind] = state(dec)
            M.train_stage_protected(dec, cur, it, gen, rep, op=M.Protector(dec, kind, alpha, U, U, "cpu"),
                                    probe=probe, every=1, device="cpu", autocast=False)
        return f
    _run(loop("id", 0.0), iters=10)
    _run(loop("rep", 1.0), iters=10)
    changed = {k for k in snaps["id"] if not torch.equal(snaps["id"][k], snaps["rep"][k])}
    assert changed == {l + ".weight" for l in M.LAYERS}


def test_memory_and_probe_consume_no_rng_and_restore_mode():
    dec = tiny()
    x = feats(6, 5)
    for mode in (True, False):
        dec.train(mode)
        s = torch.get_rng_state()
        cv = M.covs(dec, x, "cpu")
        M.measure(dec, x, "cpu", pool=2)
        assert torch.equal(torch.get_rng_state(), s) and dec.training == mode
    assert all(torch.allclose(cv[l], cv[l].T) and cv[l].dtype == torch.float64 for l in M.LAYERS)


def test_random_bases_fixed_nested_orthonormal():
    a = M.rand_bases({l: 5 for l in M.LAYERS}, dim=DIM)
    b = M.rand_bases({l: 9 for l in M.LAYERS}, dim=DIM)
    for l in M.LAYERS:
        assert torch.equal(a[l], b[l][:, :5])
        assert torch.allclose(b[l].T @ b[l], torch.eye(9, dtype=torch.float64), atol=1e-10)
    assert not torch.allclose(a[M.LAYERS[0]], a[M.LAYERS[1]])
    assert all(torch.equal(x, y) for x, y in zip(a.values(), M.rand_bases({l: 5 for l in M.LAYERS}, dim=DIM).values()))


def test_short_warmcosine_schedule_is_valid():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = O.make_optimizer(torch.nn.ParameterList([p]))
    sch = O.make_scheduler(opt, 24)
    lrs = []
    for _ in range(24):
        lrs.append(sch.get_last_lr()[0])
        opt.step()
        sch.step()
    assert all(np.isfinite(lrs)) and lrs[-1] > lrs[0]


def test_stratified_bootstrap_macro_mean():
    g = ["a"] * 20 + ["b"] * 6
    diff = np.r_[np.full(20, -1.0), np.full(6, 1.0)]
    r = M.strat_boot(diff, g, B=200)
    assert r["point"] == 0.0 and r["ci95"] == [0.0, 0.0]
    rng = np.random.default_rng(1)
    diff = rng.normal(-0.5, 1, 26)
    r1, r2 = M.strat_boot(diff, g, B=500), M.strat_boot(diff, g, B=500)
    assert r1 == r2 and r1["ci95"][0] <= r1["point"] <= r1["ci95"][1]


def _summary(D_rep=0.010, D_rand=0.019, D_norm=0.018, ci_rep=(-0.012, -0.008), R=0.95, cats=5, D_id_lo=0.015,
             T_rep=0.010, A_ID=0.15, excess=0.0):
    arms = ["ID", "REP-05", "REP-10", "RAND-05", "NORM-05", "ACQ-05", "ORACLE-05"]
    D_ = {a: 0.02 for a in arms} | {"REP-05": D_rep, "RAND-05": D_rand, "NORM-05": D_norm}
    T_ = {a: 0.03 for a in arms} | {"REP-05": T_rep}
    rec = lambda p, lo, hi: {"point": p, "ci95": [lo, hi], "ci90": [lo, hi]}
    diff = {f"REP-05~{c}": rec(D_rep - D_[c], D_rep - D_[c] - 0.002, D_rep - D_[c] + 0.002)
            for c in ("RAND-05", "NORM-05")}
    diff["REP-05~ID"] = rec(D_rep - 0.02, *ci_rep)
    ret = {s: {"A_ID": A_ID, "R": R if A_ID >= 0.01 else None, "excess": excess} for s in (7, 8, 9)}
    return {"integrity": True, "D": D_, "T": T_, "D_ci": {"ID": [D_id_lo, 0.025]}, "T_ci": {"ID": [0.02, 0.04]},
            "diff": diff, "tdiff": {"REP-05~ID": rec(T_rep - 0.03, T_rep - 0.035, T_rep - 0.025)},
            "retention": {"REP-05": ret}, "cat_better": {"REP-05": cats}}


def test_decision_branches():
    assert M.decide(_summary())[0] == "CLOSED_LOOP_PROTECTION_SUPPORTED"
    assert M.decide(_summary(D_rep=0.0195, ci_rep=(-0.003, 0.002)))[0] == "CLOSED_LOOP_PROTECTION_REJECTED"   # R1
    assert M.decide(_summary(D_rep=0.012, D_norm=0.011))[0] == "CLOSED_LOOP_PROTECTION_REJECTED"            # R2
    assert M.decide(_summary(R=0.70))[0] == "CLOSED_LOOP_PROTECTION_REJECTED"                               # R3
    assert M.decide(_summary(ci_rep=(-0.015, 0.001)))[0] == "CLOSED_LOOP_INCONCLUSIVE"                      # S1 CI
    assert M.decide(_summary(R=0.85))[0] == "CLOSED_LOOP_INCONCLUSIVE"                                      # S4
    assert M.decide(_summary(cats=3))[0] == "CLOSED_LOOP_INCONCLUSIVE"                                      # S5
    assert M.decide(_summary(D_id_lo=-0.001))[0] == "CLOSED_LOOP_INCONCLUSIVE"                              # P0
    assert M.decide(_summary(A_ID=0.005, excess=0.0005))[0] == "CLOSED_LOOP_PROTECTION_SUPPORTED"           # degenerate
    assert M.decide(_summary(A_ID=0.005, excess=0.006))[0] == "CLOSED_LOOP_PROTECTION_REJECTED"
