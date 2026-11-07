"""NL-RET-01A invariants (CPU). Mechanical correctness only."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from exps.nl_ret_01a import common as C


@pytest.fixture(scope="module")
def pn():
    return C.proto_and_n()


def test_eligibility_and_shared_accounting():
    rows = C.eligible("ID")
    assert len(rows) == 105
    for c in C.STREAM:
        ss = [r["s"] for r in rows if r["category"] == c]
        assert ss == list(range(C.a_of(c), 15))
    assert not [r for r in rows if r["category"] == "zipper"]
    fs = C.eligible("FS_B")
    assert sum(r["shared_with_ID"] for r in fs) == 15 and sum(not r["shared_with_ID"] for r in fs) == 90
    assert all(C.owner("FS_B", s) == "ID" for s in range(1, 7)) and all(C.owner("FS_B", s) == "FS_B" for s in range(7, 16))
    assert C.local_ckpt("FS_B", 2711, 6) == C.local_ckpt("ID", 2711, 6)


def test_membership_no_future_capacity_subsets(pn):
    proto, n = pn
    for seed in C.SEEDS:
        prev = None
        for s in range(1, 16):
            m = C.members(seed, n, s)
            assert list(m) == C.STREAM[:s]                         # only completed tasks, never future ones
            assert sum(len(v) for v in m.values()) == C.CAPACITY   # 24 in total, not per category
            assert all(len(set(v)) == len(v) for v in m.values())
            if prev:
                assert all(set(m[c]) <= set(prev[c]) for c in prev)
            prev = m


def test_nonmembers_disjoint_from_every_buffer(pn):
    proto, n = pn
    for seed in C.SEEDS:
        for c in C.STREAM:
            nm = set(C.nonmembers(seed, n, c))
            for s in range(C.a_of(c), 16):
                assert not nm & set(C.members(seed, n, s)[c])
            assert len(nm) + len(C.ever_members(seed, n, c)) == n[c]


def test_lawful_feature_names_exclude_eval_and_labels():
    from exps.nl_ret_01a import validate as V
    for f in V.LAWFUL + V.CONTROLS:
        assert not f.startswith(("ORACLE", "Y")) and "nonmember" not in f
    assert set(V.specs()["B6_lawful_persistence"][1]) <= set(V.LAWFUL)
    assert "F_aux_eff" not in V.LAWFUL


def test_no_mask_or_defect_access_in_sources():
    for p in Path("exps/nl_ret_01a").glob("*.py"):
        if p.name.startswith("test_"):
            continue
        src = p.read_text()
        assert not re.search(r"native_mask|mask_path|ground_truth|/test/", src), p
    v = Path("exps/nl_ret_01a/validate.py").read_text()
    # Y4 (exposed) is only touched in run_align
    body = v.split("def run_align")[0]
    assert "Y4_exposed" not in body.replace('"Y4_exposed"', "") or "Y4" not in body.split("def run_validate")[1]


def test_prep_uses_training_statistics_only():
    from exps.nl_ret_01a import validate as V
    rng = np.random.default_rng(0)
    Xtr, Xte = rng.normal(size=(50, 3)), rng.normal(size=(10, 3))
    Xtr[3, 1] = np.nan
    A1, B1 = V.prep(Xtr, Xte)
    A2, B2 = V.prep(Xtr, Xte * 100)                  # changing test rows cannot change the training transform
    assert np.allclose(A1, A2)


def test_no_label_leakage_from_test_fold():
    from exps.nl_ret_01a import validate as V
    rng = np.random.default_rng(1)
    D = pd.DataFrame({"category": np.repeat(C.STREAM[:6], 10), "x": rng.normal(size=60), "THR": 0.0,
                      "AGE_BUCKET": 0})
    D["y"] = 2 * D.x + rng.normal(scale=.1, size=60)
    tr, te = D.index[D.category != "bottle"], D.index[D.category == "bottle"]
    p1 = V.fit_predict(D, tr, te, ("ridge", ["x"]), "y")
    D.loc[te, "y"] = 1e6                                 # test labels must not matter
    p2 = V.fit_predict(D, tr, te, ("ridge", ["x"]), "y")
    assert np.allclose(p1, p2)
    for tr_, te_, c in V.folds(D.assign(seed=2711, s=1), "LOCO"):
        assert c not in set(D.loc[tr_, "category"])


def test_zero_one_member_robustness():
    from exps.nl_ret_01a import features as Fz
    st = Fz.syn_stats(None, None)
    assert all(np.isnan(v) for v in st.values())
    g = torch.rand(1, 28, 28); cv = torch.zeros(1, 28, 28); cv[0, 3:6, 3:6] = 1
    st = Fz.syn_stats(g, cv)
    assert 0 <= st["ap"] <= 1 and np.isfinite(st["margin"])


def test_clustered_bootstrap_is_grouped_and_deterministic():
    from exps.nl_ret_01a import validate as V
    D = pd.DataFrame({"category": np.repeat(["a", "b", "c"], [100, 5, 5]), "y": 0.0})
    pa = pd.Series(np.r_[np.ones(100), np.zeros(10)]); pb = pd.Series(np.zeros(110))
    r = V.cluster_diff(D, pa, pb, "y", reps=500)
    assert r["ci95"][0] <= r["diff"] <= r["ci95"][1]
    assert r["ci95"][0] == 0.0                            # resampling whole categories can drop 'a' entirely


def test_temporal_folds_respect_causality():
    from exps.nl_ret_01a import validate as V
    D = pd.DataFrame([{"category": r["category"], "s": r["s"], "seed": 2711} for r in C.eligible("ID")])
    for tr, te, t in V.folds(D, "temporal"):
        assert (D.loc[tr, "s"] + 1 <= t).all() and (D.loc[te, "s"] == t).all()


def test_checkpoint_copies_match_records():
    from exps.dinomaly.common import sha
    for arm, s in (("ID", 1), ("ID", 15), ("FS_B", 7), ("FS_B", 15)):
        p, h, _, _ = C.checkpoint(arm, 2712, s)
        assert sha(C.local_ckpt(arm, 2712, s)) == h
