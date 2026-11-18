"""NAR-ACQ-01 tests (synthetic, CPU; no dataset, feature, checkpoint or label file is read)."""
from __future__ import annotations

import torch

from exps.nar_acq_01 import run as R


def test_tangent_split_exact_cases():
    torch.manual_seed(0)
    A, B = torch.randn(4, 30), torch.randn(20, 4)
    inside = torch.randn(20, 4) @ A + B @ torch.randn(4, 30)        # in the tangent space
    tot, tan, _ = R.tangent_split(inside, A, B)
    assert abs(tan / tot - 1) < 1e-10
    Qb, _ = torch.linalg.qr(B.double()); Qa, _ = torch.linalg.qr(A.double().T)
    H = torch.randn(20, 30).double()
    H = H - Qb @ (Qb.T @ H); H = H - (H @ Qa) @ Qa.T                 # orthogonal complement
    tot, tan, rnd = R.tangent_split(H, A, B)
    assert tan / tot < 1e-10 and abs(rnd / tot - (1 - (1 - 4 / 20) * (1 - 4 / 30))) < 1e-12


def test_tangent_random_expectation():
    torch.manual_seed(1)
    A, B = torch.randn(8, 400), torch.randn(300, 8)
    shares = [R.tangent_split(torch.randn(300, 400), A, B) for _ in range(20)]
    s = sum(t for _, t, _ in shares) / sum(a for a, _, _ in shares)
    r = shares[0][2] / shares[0][0]
    assert abs(s / r - 1) < 0.1


def _cell(agg=False, obj=False, cap=False):
    c = {"win_own_fin": .3 if agg else .9, "foreign_over_own_median": 1.1 if agg else 3.0,
         "norm_low_high_ratio": 2.0 if obj else 1.0}
    g = {"cos_global_patchmean": .2 if obj else .9, "cos_global_top1": .9, "gnorm_rel_base": .5 if cap else .01,
         "tangent_over_random": 10.0}
    return {"holds": R.holds(c, g)}


def test_decision_rule():
    cats = [f"c{k}" for k in range(12)] + ["screw", "transistor", "zipper"]
    one = {a: {c: _cell(obj=True) for c in cats} for a in ("nested", "warm")}
    assert R.decide(one)[0] == "DECODER_OBJECTIVE_MISMATCH_CANDIDATE"
    none = {a: {c: _cell() for c in cats} for a in ("nested", "warm")}
    assert R.decide(none)[0] == "ROOT_CAUSE_UNRESOLVED"
    tie = {a: {c: _cell(obj=True, agg=True) for c in cats} for a in ("nested", "warm")}
    assert R.decide(tie)[0] == "ROOT_CAUSE_UNRESOLVED"
    pref = {a: {c: _cell(obj=True, agg=c not in R.RULE["WEAK"]) for c in cats} for a in ("nested", "warm")}
    assert R.decide(pref)[0] == "DECODER_OBJECTIVE_MISMATCH_CANDIDATE"
    onearm = {"nested": {c: _cell(cap=True) for c in cats}, "warm": {c: _cell() for c in cats}}
    assert R.decide(onearm)[0] == "ROOT_CAUSE_UNRESOLVED"


def test_custody():
    ok = [R.FULL / "features/screw_train.pt", R.FULL / "BASE.pt", R.WARM / "stage15.pt", R.OUT / "RESULT.json"]
    no = [R.FULL / "features/screw_test.pt", R.ROOT / "data/mvtec/screw/test/good/000.png",
          R.ROOT / "data/mvtec/screw/ground_truth/x/000_mask.png", R.WARM / "scores_screw.pt",
          R.WARM / "metric_15_screw.json", R.ROOT / "data/visa/candle/x.png"]
    assert all(R.allowed(p) for p in ok) and not any(R.allowed(p) for p in no)
