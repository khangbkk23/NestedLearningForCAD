"""G1: ranking, top-k, coverage, bootstrap and JSON-safety statistics."""

from __future__ import annotations

import numpy as np

from exps.csg00 import contract as C
from exps.csg00 import geometry_diagnostics as D
from exps.csg00.run import clean_json


def test_spearman_and_kendall_known_values_and_ties():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert D.spearman(x, x) == 1.0
    assert D.spearman(x, x[::-1]) == -1.0
    assert D.spearman(x, np.ones_like(x)) is None
    assert D.spearman(x[:2], x[:2]) is None  # too few

    assert D.kendall_tau_b(x, x) == 1.0
    assert D.kendall_tau_b(x, x[::-1]) == -1.0
    # tie-corrected tau-b is strictly below 1 when y has ties
    y = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
    tau = D.kendall_tau_b(x, y)
    assert tau is not None and 0.0 < tau < 1.0
    assert D.kendall_tau_b(x, np.ones_like(x)) is None


def test_topk_total_order_with_ties():
    scores = np.array([1.0, 1.0, 1.0, 0.0])
    assert D.topk_indices(scores, 2).tolist() == [0, 1]
    scores = np.array([0.0, 1.0, 1.0, 1.0])
    assert D.topk_indices(scores, 2).tolist() == [1, 2]
    scores = np.array([3.0, 1.0, 2.0, 0.0])
    assert D.topk_indices(scores, 2).tolist() == [0, 2]
    assert D.topk_indices(scores, 10).tolist() == [0, 1, 2, 3]
    assert D.topk_counts(1000, 0.01) == 10
    assert D.topk_counts(1000, 0.05) == 50


def _scores(n=8, p=10):
    rng = np.random.default_rng(51)
    base = rng.uniform(0, 5, size=(n, p))
    return {"A": base, "B": base.copy(), "C": np.flip(base, axis=1) + 10.0}


def test_selection_metrics_perfect_and_degraded_control():
    scores = _scores()
    metrics = D._metrics_on_selection(scores, np.arange(scores["A"].shape[0]))
    assert metrics["rho_B"] == 1.0
    assert metrics["coverage_B"] == 1.0
    assert metrics["coverage_C"] == 0.0
    for frac in C.TOP_FRACTIONS:
        key = f"{frac:.2f}"
        assert metrics["topk"][key]["precision_B"] == 1.0
        assert metrics["topk"][key]["iou_B"] == 1.0
        assert metrics["topk"][key]["precision_C"] < 1.0


def test_constant_arm_is_undefined_for_spearman():
    scores = _scores()
    scores["B"] = np.zeros_like(scores["B"])
    metrics = D._metrics_on_selection(scores, np.arange(8))
    assert metrics["rho_B"] is None
    assert metrics["delta_rho"] is None
    assert metrics["zero_fraction_B"] == 1.0


def test_patch_summary_histogram_and_zero_fraction():
    values = np.array([0.0, 0.0, 1.0, 2.0, 4.0, 100.0])
    summary = D.patch_summary(values)
    assert summary["n"] == 6
    assert summary["zero_fraction"] == np.float64(2 / 6)
    total = sum(summary["histogram_fraction"].values())
    assert abs(total - 1.0) < 1e-12


def test_bootstrap_single_category_structure():
    scores = _scores()
    result = D.bootstrap_single_category(scores, np.arange(8), reps=48, boundary=200,
                                         category="bottle")
    assert result["replicates"] == 48
    assert set(result["bootstrap"]) >= {"rho_B", "delta_cov", "coverage_B"}
    for entry in result["bootstrap"].values():
        assert len(entry["ci"]) == 2
        assert entry["n_total"] == 48


def test_clean_json_replaces_nonfinite():
    cleaned = clean_json({"a": float("nan"), "b": float("inf"), "c": [1, np.float32(2.5)],
                          "d": np.zeros((2,))})
    assert cleaned["a"] is None and cleaned["b"] is None
    assert cleaned["c"] == [1, 2.5]
    assert cleaned["d"] == [0.0, 0.0]


def test_mode_occupancy_summary_schema():
    bins = [{"bin": b, "n_total": 1600, "n_modes_ready": 2, "n_modes_dropped": 0,
             "counts": [800, 800], "inertia": 10.0, "iterations": 4, "restart": 0,
             "center_distance": 5.0, "orthogonality_max_error": 1e-7,
             "rho": [2.0, 2.0], "R_max": [8.0, 8.0], "rep_index": [0, 1],
             "ready_mode_indices": [0, 1]} for b in range(C.BINS)]
    summary = D.mode_occupancy_summary(bins)
    assert summary["bins"] == C.BINS
    assert summary["ready_mode_count_histogram"] == {"2": C.BINS}
    assert summary["mode_counts"] == [800, 800] * C.BINS


def test_bootstrap_selection_macro_structure():
    scores = _scores(n=12, p=8)
    groups = {"bottle": np.arange(6), "carpet": np.arange(6, 12)}
    result = D.bootstrap_selection(scores, groups, reps=24, boundary=350)
    assert result["endpoint"].startswith("rank/coverage")
    assert "macro_point_estimate" in result
    assert "coverage_B" in result["macro_point_estimate"]
    assert result["bootstrap"]["delta_cov"]["n_total"] == 24
    assert result["topk_bootstrap"]["0.01"]["precision_B"]["n_total"] == 24


def _synthetic_decision_metrics(gfa: float) -> dict:
    boundaries = {}
    for t in C.BOUNDARIES:
        boundaries[str(t)] = {
            "bootstrap_pooled": {
                "bootstrap": {
                    "delta_cov": {"ci": [0.90, 0.95]},
                    "delta_rho": {"ci": [0.05, 0.30]},
                },
                "macro_point_estimate": {"zero_fraction_B": 0.55},
            },
            "bootstrap_by_category": {
                "bottle": {"point_estimate": {"image_score_B": {"max": 5.0, "min": 1.0}}},
                "carpet": {"point_estimate": {"image_score_B": {"max": 4.0, "min": 0.5}}},
                "hazelnut": {"point_estimate": {"image_score_B": {"max": 3.0, "min": 0.2}}},
            },
            "gap": {"gfa_B": gfa, "n_B_accepted": int(gfa * 100), "n_high_A": 100},
            "state": {"byte_ratio_within_5pct": True,
                      "reload_match_B": True, "reload_match_C": True},
        }
    return {"boundaries": boundaries}


def test_decision_bands_go_and_negative():
    from exps.csg00.run import decide

    parity = {"passed": True, "worst_arm_max_abs_diff": 1e-6}
    go = decide(_synthetic_decision_metrics(0.10), parity)
    # the executed artifact stores boundaries as a list; the parser must accept both
    listed = _synthetic_decision_metrics(0.10)
    listed["boundaries"] = [dict(v, boundary=int(k)) for k, v in listed["boundaries"].items()]
    assert decide(listed, parity)["decision"] == "CSG00_GO_FRESH_ANOMALY_GATE_REQUESTED"
    assert go["decision"] == "CSG00_GO_FRESH_ANOMALY_GATE_REQUESTED"
    negative = decide(_synthetic_decision_metrics(0.55), parity)
    assert negative["decision"] == "CSG00_NEGATIVE_GEOMETRY"
    broken = decide(_synthetic_decision_metrics(0.10), {"passed": False})
    assert broken["decision"] == "CSG00_BLOCKED_INTEGRITY"
