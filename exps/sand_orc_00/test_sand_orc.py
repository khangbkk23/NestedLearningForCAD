"""SAND-ORC-00 tests (synthetic only).

    PYTHONPATH=. python -m pytest -q -p no:cacheprovider exps/sand_orc_00/test_sand_orc.py
"""
from __future__ import annotations

import numpy as np

from exps.sand_e2a_00 import common as E
from exps.sand_orc_00 import run as O


def _boot(x, y):
    from exps.nl_sgm_01a import am03_eval as A3
    return A3.bootstrap_delta(x, y, seed=O.BOOT["seed"], B=2000)


def test_custody():
    assert O.decide(O.OUT / "RESULT.json") and O.decide(E.OUT / "RESULT.json") and O.decide(E.GRIDS / "stage01_bottle.pt")
    assert O.decide(E.MVTEC / "bottle/ground_truth/broken_large/000_mask.png") and O.decide(E.MANIFEST)
    for p in (E.MVTEC / "bottle/test/good/000.png", E.ROOT / "data/visa/x.png", E.HOME_CACHE / "x.pt",
              E.ROOT / "results/hope_cad/calm02/features/mvtec/bottle_test.pt", E.RUN / "stage05_bottle.pt"):
        assert not O.decide(p), p


def test_oracle_cal_uses_only_normal_images():
    from exps.sand import common as S
    r = np.random.default_rng(0)
    a = np.abs(r.normal(0.02, 0.01, (10, 784))).astype(np.float32)
    d = np.abs(r.normal(4.0, 2.0, (10, 784)))
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    a[labels == 1] += 5.0                                        # anomalous images must not move the calibrators
    c = O.oracle_cal(S, a, d, labels)
    ref = S.calibrate(a[:4], d[:4])
    assert all(c[k] == ref[k] for k in E.CAL_KEYS)


def test_present_and_verdict():
    yes, weak, wide = {"point": 0.02, "lo": 0.005}, {"point": 0.008, "lo": 0.004}, {"point": 0.03, "lo": -0.001}
    assert O.present(yes) and not O.present(weak) and not O.present(wide)
    assert O.verdict(yes, yes) == "SF_HEADROOM_AND_COMPLEMENTARY"
    assert O.verdict(yes, wide) == "SF_HEADROOM_FUSION_REDUNDANT"
    assert O.verdict(weak, yes) == "NO_SF_HEADROOM"


def test_analyse_end_to_end():
    r = np.random.default_rng(1)
    names = ("ORC_FIN", "ORC_RET", "INS_RET", "PAR", "TH_P_acq", "TH_P_fin", "A2_fin", "E2A1_fin")
    base = r.uniform(0.3, 0.8, 15)
    shift = {"ORC_RET": 0.03, "ORC_FIN": 0.0, "TH_P_acq": 0.0}
    per = {c: {a: {"pixel_ap": float(base[i] + shift.get(a, 0.0) + r.normal(0, 0.002)),
                   "image_auroc": float(r.uniform(0.8, 1.0))} for a in names} for i, c in enumerate(E.ORDER)}
    out = O.analyse(per, _boot)
    assert out["verdict"] == "SF_HEADROOM_AND_COMPLEMENTARY"
    assert out["comparisons"]["RET"]["present"] and out["comparisons"]["COMP"]["present"]
    assert abs(out["comparisons"]["RET"]["point"] - 0.03) < 0.003 and not out["comparisons"]["CAL_FIN"]["present"]
    assert set(out["macro"]) == set(names) and len(out["per_category"]) == 15


def test_sources_exist():
    for s in O.SOURCES:
        assert (E.ROOT / s).exists(), s
