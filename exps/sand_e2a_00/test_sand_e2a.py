"""SAND-E2A-00 tests: synthetic data only (no SAND artifact, label or mask is read).

    PYTHONPATH=. python -m pytest -q exps/sand_e2a_00/test_sand_e2a.py
"""
from __future__ import annotations

import numpy as np
import pytest

from exps.sand_e2a_00 import common as E
from exps.sand_e2a_00 import run as R


def _rng(s=0):
    return np.random.default_rng(s)


def test_custody_phases():
    g, d, k = E.GRIDS / "stage03_bottle.pt", E.DIST / "bottle_abc.npz", E.CALDIR / "bottle.json"
    mask = E.MVTEC / "bottle/ground_truth/broken_large/000_mask.png"
    deny_always = [E.MVTEC / "bottle/test/good/000.png", E.MVTEC / "bottle/train/good/000.png",
                   E.ROOT / "data/visa/candle/Data/Images/Normal/0000.JPG", E.HOME_CACHE / "x.pt",
                   E.RUN / "stage05_bottle.pt", E.ROOT / "results/hope_cad/calm02/features/mvtec/bottle_test.pt",
                   E.ROOT / "results/hope_cad/rh01b/PROTOCOL.json", E.A1_OUT / "FINAL.json"]
    for p in (g, d, k, E.MEM / "boundary_t03.json", E.RUN / "STAGE05.json", E.OUT / "x.json", E.ROOT / "exps/x.py"):
        assert E.decide(p, "C") and E.decide(p, "L0") and E.decide(p, "L"), p
    for p in (E.CELLS / "stage15_bottle.json", E.MANIFEST, mask, E.SAND / "PHASE0_RESULTS.json",
              E.A1_OUT / "EVAL_STAGE15.json", E.SAND / "LOCK.json"):
        assert not E.decide(p, "C") and not E.decide(p, "L0") and E.decide(p, "L"), p
    for p in deny_always:
        assert not E.decide(p, "C") and not E.decide(p, "L"), p
    assert not E.decide(E.MVTEC / "screwdriver/ground_truth/a/0.png", "L")


def test_channel_stats_match_sand_calibrate():
    from exps.sand import common as S
    r = _rng(1)
    a = r.normal(0.02, 0.01, (6, 784)).astype(np.float32)
    d = np.abs(r.normal(4.0, 2.0, (6, 784)))
    st = E.channel_stats(a, d)
    cal = S.calibrate(a, d)
    assert st["theta"] == (cal["mu_theta"], cal["sd_theta"]) and st["m"] == (cal["mu_m"], cal["sd_m"])


def test_e2a_identities():
    from exps.sand import common as S
    r = _rng(2)
    a0 = r.normal(0.02, 0.01, (6, 784)).astype(np.float32)
    d0 = np.abs(r.normal(4.0, 2.0, (6, 784)))
    a1 = (a0 * 1.7 + 0.01).astype(np.float32)
    d1 = d0 * 1.1 + 0.3
    cal_a = {k: v for k, v in S.calibrate(a0, d0).items() if k in E.CAL_KEYS}
    cal_t = {k: v for k, v in S.calibrate(a1, d1).items() if k in E.CAL_KEYS}
    wa = [E.channel_stats(a0[j], d0[j]) for j in range(6)]
    wt = [E.channel_stats(a1[j], d1[j]) for j in range(6)]
    for j in range(6):                                          # at acquisition: exactly the frozen calibrators
        assert E.e2a(cal_a, wa[j], wa[j]) == cal_a
    pa = {ch: (cal_a[f"mu_{ch}"], cal_a[f"sd_{ch}"]) for ch in E.CH}
    pt = {ch: (cal_t[f"mu_{ch}"], cal_t[f"sd_{ch}"]) for ch in E.CH}
    e6 = E.e2a(cal_a, pa, pt)                                   # all six probes: the refreshed (D1) calibrators
    assert max(abs(e6[k] - cal_t[k]) / abs(cal_t[k]) for k in E.CAL_KEYS) <= 1e-12
    th = E.e2a(cal_a, wa[0], wt[0], channels=("theta",))
    assert th["mu_m"] == cal_a["mu_m"] and th["sd_m"] == cal_a["sd_m"] and th["mu_theta"] != cal_a["mu_theta"]
    # a pure shift of an (all-positive) witness moves mu by the shift and leaves sd unchanged
    w = np.abs(a0[0]).astype(np.float64) + 0.001                 # float64: the shift itself must be exact
    s0, s1 = E.channel_stats(w, d0[0]), E.channel_stats(w + 0.05, d0[0])
    e = E.e2a(cal_a, s0, s1, channels=("theta",))
    assert abs(e["mu_theta"] - (cal_a["mu_theta"] + 0.05)) < 1e-12 and abs(e["sd_theta"] - cal_a["sd_theta"]) < 1e-12


def test_fuse_nonnegative_with_witness_calibrators():
    from exps.sand import common as S
    r = _rng(3)
    a = np.abs(r.normal(0.02, 0.01, (4, 784))).astype(np.float32)
    d = np.abs(r.normal(4.0, 2.0, (4, 784)))
    cal = {"mu_theta": -0.01, "sd_theta": 0.004, "mu_m": 3.0, "sd_m": 1.5}   # even a negative mu stays non-negative
    g, frac = S.fuse(a, d, cal)
    assert g.min() >= 0 and 0 <= frac <= 1


def test_tracking_error_zero_at_reference():
    cal = {"mu_theta": 0.02, "sd_theta": 0.01, "mu_m": 4.0, "sd_m": 2.0}
    te = E.tracking_error(cal, cal)
    assert te == {"theta": (0.0, 0.0), "m": (0.0, 0.0)}


def _boot(x, y):
    from exps.nl_sgm_01a import am03_eval as A3
    return A3.bootstrap_delta(x, y, seed=E.GATE["boot_seed"], B=2000)


def test_gate_rules():
    r = _rng(4)
    base = r.uniform(0.3, 0.8, 15)
    singles = {"A1": list(base - 0.004), "A2": list(base - 0.003), "TH_P": list(base)}
    au = {"A1": 0.92, "A2": 0.91, "TH_P": 0.91}
    g = E.gate(list(base + 0.02 + r.normal(0, 0.003, 15)), singles, 0.915, au, _boot)
    assert g["best_single_pixel_ap_arm"] == "TH_P" and g["pass"] and g["verdict"] == "E2A_FUSION_SUPPORTED"
    g = E.gate(list(base + 0.008), singles, 0.915, au, _boot)                       # below +0.010
    assert not g["G1_gain"] and not g["pass"]
    g = E.gate(list(base + 0.02), singles, 0.905, au, _boot)                        # AUROC loss > 0.010
    assert g["G1_gain"] and not g["G3_auroc"] and not g["pass"]
    wild = base.copy()
    wild[0] += 0.6
    wild[1:] -= 0.02                                                                # one category carries the mean
    g = E.gate(list(wild), singles, 0.93, au, _boot)
    assert g["G1_gain"] and not g["G2_lower_bound"] and not g["pass"]


def test_forgetting_matches_sand_summarize():
    from exps.sand import analyze as SA
    r = _rng(5)
    cells = {(t, c): {"pixel_ap": float(r.uniform(0.2, 0.9)), "image_auroc": float(r.uniform(0.7, 1.0))}
             for t, c in E.KEYS}
    s = SA.summarize(cells)
    assert E.forgetting({k: v["pixel_ap"] for k, v in cells.items()})[0] == pytest.approx(s["pixel_fm"], abs=1e-15)
    assert E.forgetting({k: v["image_auroc"] for k, v in cells.items()})[0] == pytest.approx(s["image_fm"], abs=1e-15)


def _synthetic_world(seed=6):
    r = _rng(seed)
    m = lambda: {"pixel_ap": float(r.uniform(0.2, 0.9)), "image_auroc": float(r.uniform(0.7, 1.0)),  # noqa: E731
                 "theta_is_min_fraction": float(r.uniform(0, 1))}
    fin = {c: {n: m() for n in ["frozen", "E2A6", *R.W, *R.WT]} for c in E.ORDER}
    sand_acq = {c: {"fusion_P": m()} for c in E.ORDER}
    inter = {(t, c): {w: m() for w in R.W} for k, c in enumerate(E.ORDER) for t in range(k + 1, E.NT - 1)}
    sand_final = {c: {"support_canonical": m(), "theta_P": m(), "n_images": 50} for c in E.ORDER}
    a1_final = {c: m() for c in E.ORDER}
    per_cat = {c: {"pixel_fm": float(r.uniform(0, 0.05))} for c in E.ORDER}
    sand = {"summary": {a: {"pixel_fm": 0.01, "image_fm": 0.01, "acq_pixel_ap": 0.55, "per_category": per_cat}
                        for a in ("A1", "A2", "A3")}}
    return fin, sand_acq, inter, sand_final, a1_final, sand


def test_build_series_and_analyse_end_to_end():
    fin, sand_acq, inter, sand_final, a1_final, sand = _synthetic_world()
    series = R.build_series(fin, sand_acq, inter)
    assert series[(E.NT - 1, "zipper", "E2A1_w0")]["pixel_ap"] == sand_acq["zipper"]["fusion_P"]["pixel_ap"]
    assert series[(E.NT - 1, "bottle", "E2A1_w3")]["pixel_ap"] == fin["bottle"]["E2A1_w3"]["pixel_ap"]
    assert series[(4, "bottle", "E2A1_w2")]["pixel_ap"] == inter[(4, "bottle")]["E2A1_w2"]["pixel_ap"]
    e2a_ap = [float(np.mean([fin[c][w]["pixel_ap"] for w in R.W])) for c in E.ORDER]
    e2a_au = [float(np.mean([fin[c][w]["image_auroc"] for w in R.W])) for c in E.ORDER]
    singles_ap = {"A1": [a1_final[c]["pixel_ap"] for c in E.ORDER],
                  "A2": [sand_final[c]["support_canonical"]["pixel_ap"] for c in E.ORDER],
                  "TH_P": [sand_final[c]["theta_P"]["pixel_ap"] for c in E.ORDER]}
    singles_au = {a: [0.9] * 15 for a in E.SINGLES}
    g = E.gate(e2a_ap, singles_ap, float(np.mean(e2a_au)), {a: 0.9 for a in E.SINGLES}, _boot)
    for s in (series, None):
        out = R.analyse(E, g, fin, sand, sand_final, sand_acq, a1_final, singles_ap, singles_au, e2a_ap, e2a_au, s, _boot)
        assert out["verdict"] == g["verdict"] and len(out["macro_final"]["E2A1_per_choice"]) == 6
        assert out["macro_final"]["E2A1_mean"] == pytest.approx(np.mean(e2a_ap))
        assert out["bytes"]["E2A1_total"] == 318864848 + 15 * (784 * 768 * 4 + 32)
        if s is None:
            assert out["forgetting"] is None
        else:
            f = out["forgetting"]
            want = np.mean([E.forgetting({(t, c): series[(t, c, w)]["pixel_ap"] for t, c in E.KEYS})[0] for w in R.W])
            assert f["E2A1_pixel_fm"] == pytest.approx(want) and len(f["stage_macro_pixel_ap"]) == 15


def test_custody_summary_ordering(monkeypatch):
    log = [{"path": "/x/results/hope_cad/sand_e2a_00/LOCK.json"}, {"mark": "calib_lock_verified"},
           {"mark": "phase_L_entered"}, {"path": "/x/results/hope_cad/nl_sgm_01a_am03/DATASET_MANIFEST.json"},
           {"path": "/x/data/mvtec/bottle/ground_truth/a/000_mask.png"}]
    monkeypatch.setattr(E, "_LOG", log)
    s = R.custody_summary()
    assert s["labels_after_calib_lock"] and s["first_label_access_index"] == 3 and s["n_label_file_opens"] == 2
    monkeypatch.setattr(E, "_LOG", [log[3], log[1]])
    assert not R.custody_summary()["labels_after_calib_lock"]


def test_sources_exist():
    for s in (*E.SOURCES, *E.CANONICAL):
        assert (E.ROOT / s).exists(), s
