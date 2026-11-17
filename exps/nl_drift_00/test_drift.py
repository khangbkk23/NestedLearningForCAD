"""NL-DRIFT-00 CPU tests (no data access except the custody decision function and the stage records).

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q exps/nl_drift_00/test_drift.py -p no:cacheprovider
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from exps.nl_drift_00 import analyze as AN
from exps.nl_drift_00 import common as C
from exps.nl_drift_00 import estimators as E


def synthetic(seed=0, nc=15, beta=0.6, meas_sd=0.5, proc_sd=0.05, persist=0.0):
    """Drift driven by a covariate: s_t = s_{t-1} + beta*u_t + process noise; anchored measurement noise with
    sd meas_sd (fraction ``persist`` of it persistent across boundaries)."""
    rng = np.random.default_rng(seed)
    a = np.arange(nc)
    u = np.full((nc, 15, 3), np.nan)
    s = np.full((nc, 15, 2), np.nan)
    ya = np.full((nc, 15, 2), np.nan)
    for i in range(nc):
        s[i, a[i]] = 0.0
        ya[i, a[i]] = 0.0
        b = rng.normal(0, meas_sd, 2)
        for t in range(a[i] + 1, 15):
            u[i, t] = (1.0, rng.normal(0, 1), rng.uniform(0, 0.5))
            s[i, t] = s[i, t - 1] + beta * u[i, t, 1] + rng.normal(0, proc_sd, 2)
            e = persist * b + np.sqrt(1 - persist ** 2) * rng.normal(0, meas_sd, 2)
            ya[i, t] = s[i, t] + e
    rs = np.full((nc, 2), meas_sd ** 2)
    return a, u, s, ya, rs


def mse(est, s):
    m = np.isfinite(est[..., 0])
    return float(np.mean((est[m] - s[m]) ** 2))


def test_custody_decisions():
    R, H = C.ROOT, C.HOME_CACHE
    deny = [R / "data/mvtec/bottle/test/broken_large/000.png", R / "data/mvtec/bottle/ground_truth/x.png",
            R / "data/visa/split_csv/1cls.csv", R / "results/hope_cad/calm02/features/mvtec/bottle_test.pt",
            R / "results/hope_cad/calm02/features/mvtec/MANIFEST.json", H / "vitb8_q2/test_bottle.pt",
            H / "vitb8_q2/train_bottle.pt", R / "results/hope_cad/sand_final/A3_s2711/grids/stage01_bottle.pt",
            R / "results/hope_cad/vitb8_nl00/EVAL_STAGE15.json", R / "results/hope_cad/nl_sgm_01a_am03/DATASET_MANIFEST.json",
            R / "results/hope_cad/rh01b/metrics/x.json"]
    allow = [R / "results/hope_cad/calm02/features/mvtec/bottle_train.pt", C.RUN / "STAGE01.json",
             C.RUN / "stage01_bottle.pt", C.RH01B, C.CACHE_JSON, C.C2DOM_CKPT, C.OUT / "STATS.pt",
             R / "exps/nl_drift_00/common.py", R / "agents/plans/nl_drift_00_prereg.md"]
    assert not any(C.decide(p) for p in deny)
    assert all(C.decide(p) for p in allow)


def test_witness_sets_nested_disjoint_deterministic():
    pool = [f"bottle/train/good/{i:03d}.png" for i in range(60)]
    ws = C.witness_sets("T", "bottle", pool)
    assert len(ws) == C.DRAWS
    for w, ref in ws:
        assert len(w) == 4 and not set(w) & set(ref) and len(ref) == 56
    rev = C.witness_sets("T", "bottle", list(reversed(pool)))
    assert [w for w, _ in ws] == [w for w, _ in rev]                        # independent of pool order
    assert [set(r) for _, r in ws] == [set(r) for _, r in rev]
    assert len({tuple(w) for w, _ in ws}) > 1
    assert C.witness_sets("U", "bottle", pool)[0][0] != ws[0][0] or C.witness_sets("U", "bottle", pool)[1][0] != ws[1][0]


def test_pooled_statistics_match_direct():
    rng = np.random.default_rng(1)
    g = rng.normal(0.02, 0.01, size=(5, C.TOKENS))
    r = np.maximum(g, 0)
    S = np.stack([r.sum(1), (r ** 2).sum(1), g.sum(1)], 1)
    mu, sd = AN.pooled_from_rows(S)
    assert np.isclose(mu, r.mean(), rtol=0, atol=1e-15) and np.isclose(sd, r.std(), rtol=1e-12)


def test_fixed_rules():
    a, u, s, ya, rs = synthetic()
    assert np.all(E.e0(a)[np.isfinite(E.e0(a))] == 0)
    e2a = E.e2a(ya, a)
    m = np.isfinite(e2a)
    assert np.array_equal(e2a[m], ya[m]) and not np.isfinite(e2a[np.arange(15), a]).any()
    e3 = E.e3(ya, a)
    i = 2
    sx = np.zeros(2)
    for t in range(a[i] + 1, 15):
        sx = sx + 0.5 * (ya[i, t] - sx)
        assert np.allclose(e3[i, t], sx)


def test_default_filter_first_step_equals_e3():
    a, u, s, ya, rs = synthetic()
    est, phis, diag = E.learned(ya, u, rs, a, 3)
    e3 = E.e3(ya, a)
    assert np.allclose(est[0, 1], e3[0, 1])                    # t=1: default phi, K = 0.5
    assert diag["finite"] and diag["min_P"] >= 0 and 0 <= diag["min_K"] <= diag["max_K"] <= 1


def test_prefix_only():
    a, u, s, ya, rs = synthetic(seed=3)
    e_ref, ph_ref, _ = E.learned(ya, u, rs, a, 3)
    for t0 in (3, 7, 11):
        y2, u2 = ya.copy(), u.copy()
        y2[:, t0 + 1:] += 5.0
        u2[:, t0 + 1:, 1:] -= 3.0
        e2_, ph2, _ = E.learned(y2, u2, rs, a, 3)
        assert np.array_equal(e_ref[:, :t0 + 1], e2_[:, :t0 + 1], equal_nan=True)
        assert all(np.array_equal(ph_ref[t], ph2[t]) for t in range(t0 + 1))
        assert not np.array_equal(e_ref[:, t0 + 1:], e2_[:, t0 + 1:], equal_nan=True)


def test_shuffle_is_cyclic_derangement_among_old():
    a = np.arange(15)
    u = np.zeros((15, 15, 3))
    for i in range(15):
        u[i, :, 1] = i
        u[i, :, 2] = 100 + i
        u[i, :, 0] = 1
    us = E.shuffle_covariates(u, a)
    assert np.array_equal(us[:, :, 0], u[:, :, 0])
    for t in range(15):
        old = [i for i in range(15) if a[i] < t]
        if len(old) >= 2:
            assert all(us[i, t, 1] != u[i, t, 1] for i in old)
            assert sorted(us[old, t, 1]) == sorted(u[old, t, 1])
    assert us[0, 1, 1] == u[0, 1, 1]                           # single old category: unchanged


def test_positive_control_learning_signal():
    """Mechanics positive control: when a covariate truly drives the drift and measurements are noisy, E4 must beat
    the fixed rules and E5. (This is the only place the pilot shows E4 *can* learn; real data decide whether it does.)"""
    res = {k: [] for k in ("E2A", "E3", "E4", "E4_0", "E5")}
    t0 = time.time()
    n_fit = 0
    for seed in range(3):
        a, u, s, ya, rs = synthetic(seed=seed + 10, beta=0.6, meas_sd=0.6)
        res["E2A"].append(mse(E.e2a(ya, a), s))
        res["E3"].append(mse(E.e3(ya, a), s))
        for arm, k in (("E4", 3), ("E4_0", 1)):
            est, _, dg = E.learned(ya, u, rs, a, k)
            res[arm].append(mse(est, s))
            n_fit += len(dg["fits"])
        est, _, dg = E.learned(ya, E.shuffle_covariates(u, a), rs, a, 3)
        res["E5"].append(mse(est, s))
        n_fit += len(dg["fits"])
    m = {k: float(np.mean(v)) for k, v in res.items()}
    print("positive control MSE", m, f"{(time.time() - t0) / n_fit:.3f} s/fit")
    assert m["E4"] < 0.8 * min(m["E2A"], m["E3"]) and m["E4"] < 0.8 * m["E5"] and m["E4"] < 0.8 * m["E4_0"]


def test_negative_control_no_signal():
    """With pure random-walk drift and no informative covariate, E4 must not beat E4_0 by much (no spurious credit)."""
    a, u, s, ya, rs = synthetic(seed=21, beta=0.0, meas_sd=0.3, proc_sd=0.3)
    e4 = mse(E.learned(ya, u, rs, a, 3)[0], s)
    e40 = mse(E.learned(ya, u, rs, a, 1)[0], s)
    assert e4 > 0.8 * e40


def test_errors_zero_at_truth():
    sr = {"sa": np.array([[0.02, 0.01]]), "star": np.full((1, 15, 2), np.nan)}
    sr["star"][0, 3:] = (0.03, 0.012)
    est = np.full((1, 15, 2), np.nan)
    est[0, 4:] = ((0.03 - 0.02) / 0.01, np.log(0.012 / 0.01))
    W, emu, el = AN.errors(est, sr)
    assert np.allclose(W[0, 4:], 0, atol=1e-12) and np.isnan(W[0, 3])


def test_stage_records_and_replay_lists():
    rep = C.replay_lists()
    assert rep[1] == [] and all(len(rep[s]) == 24 for s in range(2, 16))
    ever = {p for v in rep.values() for p in v}
    assert len(ever) == 78 and not any(p.startswith("zipper/") for p in ever)


def test_analysis_end_to_end_on_synthetic_stats():
    """Runs the full analysis (all arms, draws, n, variants, bootstrap, decision, PQ, ledger) on a STATS object with
    the real identities and synthetic sums, so a defect cannot first appear after the real statistics exist."""
    import torch
    rows = {}
    for c in C.ORDER:
        obj = torch.load(C.FEAT / f"{c}_train.pt", map_location="cpu", weights_only=False, mmap=True)
        rows[c] = [r["relative_path"] for r in obj["rows"]]
    T, U, R, train, held, rep = C.pools(rows)
    from exps.nl_drift_00.stats import witness_candidates
    rng = np.random.default_rng(5)
    st = {"meta": {"checkpoints": {t: "x" for t in range(15)}, "calm02_train_sha256": {c: "x" for c in C.ORDER},
                   "s_per_grid": 0.02, "gpu_forward_s": 1.0, "wall_s": 2.0, "n_grids": 100},
          "paths": {}, "pq_paths": {}, "raw": {}, "pq": {}, "centroid": {}}
    for c in C.ORDER:
        paths = list(T[c]) + list(U.get(c, [])) + list(R[c])
        cand = witness_candidates(c, T, U)
        st["paths"][c], st["pq_paths"][c] = paths, cand
        st["centroid"][c] = rng.normal(size=C.DIM)
        for key, n in (("raw", len(paths)), ("pq", len(cand))):
            arr = np.full((n, 15, 3), np.nan)
            base = rng.uniform(0.01, 0.03, size=n)
            for t in range(C.A[c], 15):
                mu = base * (1 + 0.05 * (t - C.A[c])) * (1.02 if key == "pq" else 1.0)
                sd = 0.5 * mu
                arr[:, t, 0] = C.TOKENS * mu
                arr[:, t, 1] = C.TOKENS * (sd ** 2 + mu ** 2)
                arr[:, t, 2] = C.TOKENS * mu
            st[key][c] = arr
    t0 = time.time()
    res = AN.compute(st, t0, pv3_required=False)
    print("synthetic analysis wall", round(time.time() - t0, 1), "s; verdict", res["verdict"],
          {k: round(v, 3) for k, v in res["summary"]["T/raw/n1"]["pooled_W"].items()})
    assert res["verdict"] in ("DRIFT_TRACKING_FEASIBLE", "FIXED_RULE_SUFFICIENT", "DRIFT_SIGNAL_NOT_IDENTIFIABLE")
    assert res["provenance"]["PV6_prefix_only"] is True
    assert res["summary"]["T/raw/n1"]["n_transitions"] == 105 and res["summary"]["T/raw/n1"]["n_clusters"] == 14
    assert res["summary"]["U/raw/n1"]["n_transitions"] == 14 + 11 + 9
    json_ok = __import__("json").dumps(res, default=C._default)
    assert len(json_ok) > 1000
