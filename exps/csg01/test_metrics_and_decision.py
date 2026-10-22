"""G1: rank/top-k statistics, bootstrap structure and decision-band logic."""

from __future__ import annotations

import numpy as np

from exps.csg01 import contract as C1
from exps.csg01 import diagnostics
from exps.csg01.run import decide


def _scores(n=10, p=20, mode="good"):
    rng = np.random.default_rng(61)
    A = rng.uniform(0, 10, size=(n, p))
    B = A + 0.01 * rng.normal(size=(n, p))
    C = B + 0.005 * rng.normal(size=(n, p))
    D = rng.uniform(0, 10, size=(n, p))
    E = A + 0.5 * rng.normal(size=(n, p))
    if mode == "bad_quant":
        C = rng.uniform(0, 10, size=(n, p))
    if mode == "bad_selection":
        B = rng.uniform(0, 10, size=(n, p))
        C = B + 0.005 * rng.normal(size=(n, p))
    return {"A": A, "B": B, "C": C, "D": D, "E": E}


def test_spearman_kendall_topk_known():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert diagnostics.spearman(x, x) == 1.0
    assert diagnostics.spearman(x, x[::-1]) == -1.0
    assert diagnostics.kendall_tau_b(x, x) == 1.0
    assert diagnostics.kendall_tau_b(x, np.ones_like(x)) is None
    assert diagnostics.topk_indices(np.array([1.0, 1.0, 1.0, 0.0]), 2).tolist() == [0, 1]
    assert diagnostics.topk_counts(1000, 0.05) == 50


def test_point_and_bootstrap_structure():
    scores = _scores()
    groups = {"bottle": np.arange(5), "carpet": np.arange(5, 10)}
    point = diagnostics.point_estimates(scores, groups)
    assert set(point["macro"]) == {"B_A", "C_A", "C_B", "D_A", "E_A", "C_D", "C_E"}
    assert "spearman" in point["macro"]["C_B"]
    boot = diagnostics.bootstrap_metrics(scores, groups, reps=40, boundary=200)
    assert boot["replicates"] == 40
    assert "C_A:spearman" in boot["macro"]
    assert "adv:C_vs_D_A:spearman" in boot["macro"]
    assert boot["macro"]["adv:C_vs_D_A:spearman"]["n_total"] == 40


def test_tail_error_and_gate_go():
    scores = _scores()
    groups = {"bottle": np.arange(10)}
    point = diagnostics.point_estimates(scores, groups)
    boot = diagnostics.bootstrap_metrics(scores, groups, reps=40, boundary=350)
    tail = diagnostics.quantization_tail_error(scores)
    gates = diagnostics.gate_evaluation(
        point, boot, {"all_hold": True}, {"bound_violations": 0}, tail,
        {"C_D": 1.0, "C_E": 1.0}, ["bottle"])
    assert gates["quantization"]["passed"] is True
    assert gates["selection"]["passed"] is True
    assert gates["memory_efficiency"]["passed"] is True
    assert gates["certificates_hold"] is True


def test_gate_negative_quantization():
    scores = _scores(mode="bad_quant")
    groups = {"bottle": np.arange(10)}
    point = diagnostics.point_estimates(scores, groups)
    boot = diagnostics.bootstrap_metrics(scores, groups, reps=40, boundary=350)
    tail = diagnostics.quantization_tail_error(scores)
    gates = diagnostics.gate_evaluation(
        point, boot, {"all_hold": True}, {"bound_violations": 0}, tail,
        {"C_D": 1.0, "C_E": 1.0}, ["bottle"])
    assert gates["quantization"]["passed"] is False


def test_gate_negative_selection():
    scores = _scores(mode="bad_selection")
    groups = {"bottle": np.arange(10)}
    point = diagnostics.point_estimates(scores, groups)
    boot = diagnostics.bootstrap_metrics(scores, groups, reps=40, boundary=350)
    tail = diagnostics.quantization_tail_error(scores)
    gates = diagnostics.gate_evaluation(
        point, boot, {"all_hold": True}, {"bound_violations": 0}, tail,
        {"C_D": 1.0, "C_E": 1.0}, ["bottle"])
    assert gates["selection"]["passed"] is False


def _metrics(quant_pass=True, sel_pass=True, mem_pass=True, quant_undef=None,
             sel_undef=None, integrity=True):
    def gates():
        return {
            "quantization": {"passed": quant_pass, "undefined": quant_undef or []},
            "selection": {"passed": sel_pass, "undefined": sel_undef or []},
            "memory_efficiency": {"passed": mem_pass, "bytes_within_5pct": True,
                                  "advantage_hits": ["C_vs_D_A:spearman"] if mem_pass else [],
                                  "advantage": {}, "category_loss": {},
                                  "severe_category_loss": False},
            "certificates_hold": True,
        }
    return {"boundaries": [{
        "boundary": t,
        "gates": gates(),
        "state": {"reload_match": {a: True for a in "BCDE"},
                  "C_D_within_5pct": True, "C_E_within_5pct": True},
        "certificates": {"all_hold": True},
        "selection": {a: {"n_unique_global_ids": 1, "n_selected": 1,
                          "all_events_causal": True} for a in "BCDE"},
    } for t in C1.BOUNDARIES]}


def test_decide_enums():
    parity_ok = {"passed": True, "worst_arm_max_abs_diff": 0.0}
    assert decide(_metrics(), parity_ok)["decision"] == \
        "CSG01_GO_REQUEST_FRESH_REAL_ANOMALY_GATE"
    assert decide(_metrics(quant_pass=False), parity_ok)["decision"] == \
        "CSG01_NEGATIVE_QUANTIZATION"
    assert decide(_metrics(sel_pass=False), parity_ok)["decision"] == \
        "CSG01_NEGATIVE_SELECTION_OR_BUDGET"
    assert decide(_metrics(mem_pass=False), parity_ok)["decision"] == \
        "CSG01_NEGATIVE_SELECTION_OR_BUDGET"
    assert decide(_metrics(quant_pass=True, quant_undef=["bottle"]), parity_ok)["decision"] == \
        "CSG01_INCONCLUSIVE"
    assert decide(_metrics(sel_pass=True, sel_undef=["bottle"]), parity_ok)["decision"] == \
        "CSG01_INCONCLUSIVE"
    assert decide(_metrics(), {"passed": False})["decision"] == "CSG01_BLOCKED"
