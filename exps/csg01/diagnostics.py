"""Ranking, quantization/selection certificates, bootstrap and drift diagnostics."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from exps.csg00.geometry_diagnostics import (
    _ci, _summarize, kendall_tau_b, patch_summary, rankdata_average, spearman,
    topk_counts, topk_indices,
)

from . import contract as C1

COMPARISONS = (("B", "A"), ("C", "A"), ("C", "B"), ("D", "A"), ("E", "A"),
               ("C", "D"), ("C", "E"))


def _comparison_metrics(imgs: Dict[str, np.ndarray],
                        flats: Dict[str, np.ndarray],
                        comparisons: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    tops: Dict[Tuple[str, float], np.ndarray] = {}
    ks: Dict[float, int] = {}
    for arm, flat in flats.items():
        for frac in C1.TOP_FRACTIONS:
            k = topk_counts(flat.size, frac)
            ks[frac] = k
            tops[(arm, frac)] = topk_indices(flat, k)
    out: Dict[str, Any] = {}
    for x, y in comparisons:
        key = f"{x}_{y}"
        row: Dict[str, Any] = {
            "spearman": spearman(imgs[x], imgs[y]),
            "kendall": kendall_tau_b(imgs[x], imgs[y]),
        }
        for frac in C1.TOP_FRACTIONS:
            k = ks[frac]
            mask_x = np.zeros(flats[x].size, dtype=bool)
            mask_y = np.zeros(flats[y].size, dtype=bool)
            mask_x[tops[(x, frac)]] = True
            mask_y[tops[(y, frac)]] = True
            inter = int(np.count_nonzero(mask_x & mask_y))
            union = int(np.count_nonzero(mask_x | mask_y))
            tag = f"top{int(round(frac * 100))}"
            row[f"{tag}_precision"] = inter / float(k)
            row[f"{tag}_iou"] = inter / float(union) if union else 0.0
        out[key] = row
    return out


def selection_metrics(scores: Dict[str, np.ndarray], idx: np.ndarray,
                      comparisons: Sequence[Tuple[str, str]] = COMPARISONS) -> Dict[str, Any]:
    idx = np.asarray(idx, dtype=np.int64)
    imgs = {arm: scores[arm][idx].max(axis=1) for arm in scores}
    flats = {arm: scores[arm][idx].reshape(-1) for arm in scores}
    return _comparison_metrics(imgs, flats, comparisons)


def _macro(per_cat: Dict[str, Dict[str, Any]], key: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for comp in next(iter(per_cat.values())):
        row = {}
        for metric in per_cat[next(iter(per_cat))][comp]:
            values = [per_cat[cat][comp][metric] for cat in per_cat
                      if per_cat[cat][comp][metric] is not None]
            row[metric] = float(np.mean(values)) if values else None
        out[comp] = row
    return out


def point_estimates(scores: Dict[str, np.ndarray],
                    groups: Dict[str, np.ndarray]) -> Dict[str, Any]:
    per_cat = {cat: selection_metrics(scores, idx) for cat, idx in groups.items()}
    macro = _macro(per_cat, "macro")
    pooled = selection_metrics(scores, np.concatenate([groups[c] for c in groups]))
    return {
        "per_category": per_cat,
        "macro": macro,
        "pooled": pooled,
        "image_max": {arm: {cat: _summarize(scores[arm][idx].max(axis=1))
                            for cat, idx in groups.items()} for arm in scores},
        "patch": {arm: {cat: patch_summary(scores[arm][idx]) for cat, idx in groups.items()}
                  for arm in scores},
    }


def bootstrap_metrics(scores: Dict[str, np.ndarray], groups: Dict[str, np.ndarray],
                      reps: int, boundary: int,
                      comparisons: Sequence[Tuple[str, str]] = COMPARISONS) -> Dict[str, Any]:
    """Whole-image, category-stratified, paired bootstrap for every comparison."""
    categories = list(groups.keys())
    from exps.csg00 import contract as C0
    rng = C0.rng_from(C1.BOOTSTRAP_SEED, boundary)
    metric_keys = []
    for comp in comparisons:
        metric_keys.extend([f"{comp[0]}_{comp[1]}:spearman",
                            f"{comp[0]}_{comp[1]}:top1_precision",
                            f"{comp[0]}_{comp[1]}:top5_precision",
                            f"{comp[0]}_{comp[1]}:top5_iou"])
    advantage_keys = ["adv:C_vs_D_A:spearman", "adv:C_vs_E_A:spearman",
                      "adv:C_vs_D_A:top5_precision", "adv:C_vs_E_A:top5_precision"]
    metric_keys.extend(advantage_keys)
    macro_samples: Dict[str, List[float]] = {k: [] for k in metric_keys}
    cat_samples: Dict[str, Dict[str, List[float]]] = {
        cat: {k: [] for k in metric_keys} for cat in categories}
    for _ in range(reps):
        selected = {cat: groups[cat][rng.integers(0, groups[cat].size, groups[cat].size)]
                    for cat in categories}
        cat_metrics = {cat: selection_metrics(scores, selected[cat], comparisons)
                       for cat in categories}
        for comp in comparisons:
            key = f"{comp[0]}_{comp[1]}"
            for metric in ("spearman", "top1_precision", "top5_precision", "top5_iou"):
                sample_key = f"{key}:{metric}"
                values = [cat_metrics[cat][key][metric] for cat in categories
                          if cat_metrics[cat][key][metric] is not None]
                macro_samples[sample_key].append(float(np.mean(values)) if values else np.nan)
                for cat in categories:
                    v = cat_metrics[cat][key][metric]
                    cat_samples[cat][sample_key].append(np.nan if v is None else float(v))
        for comp in ("D_A", "E_A"):
            for metric in ("spearman", "top5_precision"):
                c_vals = [cat_metrics[cat]["C_A"][metric] for cat in categories
                          if cat_metrics[cat]["C_A"][metric] is not None]
                x_vals = [cat_metrics[cat][comp][metric] for cat in categories
                          if cat_metrics[cat][comp][metric] is not None]
                if c_vals and x_vals:
                    macro_samples[f"adv:C_vs_{comp}:{metric}"].append(
                        float(np.mean(c_vals) - np.mean(x_vals)))
                else:
                    macro_samples[f"adv:C_vs_{comp}:{metric}"].append(np.nan)
    return {
        "replicates": int(reps),
        "seed": C1.BOOTSTRAP_SEED,
        "boundary": int(boundary),
        "resampling": "whole images, category-stratified, paired across arms",
        "macro": {k: _ci(v) for k, v in macro_samples.items()},
        "per_category": {cat: {k: _ci(v) for k, v in samples.items()}
                         for cat, samples in cat_samples.items()},
    }


# ---------------------------------------------------------------------------
# Numerical certificates
# ---------------------------------------------------------------------------


def _check(name: str, violations: np.ndarray, excess: np.ndarray) -> Dict[str, Any]:
    return {
        "name": name,
        "n_checks": int(violations.size),
        "n_violations": int(np.count_nonzero(violations)),
        "max_excess": float(np.max(excess)) if excess.size else 0.0,
    }


def certificate_checks(scores: Dict[str, np.ndarray], epsilon: Dict[str, np.ndarray],
                       delta: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Check Propositions 1-4 on every eligible witness patch."""
    from exps.csg00 import contract as C0
    bins = C0.patch_bin_indices()
    tol = C1.QUANT_TOL
    epsB_p = np.asarray(epsilon["B"], dtype=np.float64)[bins]
    epsD_p = np.asarray(epsilon["D"], dtype=np.float64)[bins]
    epsE_p = np.asarray(epsilon["E"], dtype=np.float64)[bins]
    delta_p = np.asarray(delta["C"], dtype=np.float64)[bins]
    deltaD_p = np.asarray(delta["D"], dtype=np.float64)[bins]
    sA, sB, sC, sD, sE = (np.asarray(scores[a], dtype=np.float64) for a in "ABCDE")

    checks = []
    checks.append(_check("B_subset_monotone", (sB < sA - tol), (sA - sB - tol)))
    checks.append(_check("B_selection_bound", (sB - sA > epsB_p + tol),
                         (sB - sA - epsB_p - tol)))
    # D is quantized, so its lower bound is s_A - delta_D, not s_A.
    checks.append(_check("D_quantized_lower", (sD < sA - deltaD_p - tol),
                         (sA - deltaD_p - sD - tol)))
    checks.append(_check("D_combined_upper", (sD - sA > epsD_p + deltaD_p + tol),
                         (sD - sA - epsD_p - deltaD_p - tol)))
    checks.append(_check("E_subset_monotone", (sE < sA - tol), (sA - sE - tol)))
    checks.append(_check("E_selection_bound", (sE - sA > epsE_p + tol),
                         (sE - sA - epsE_p - tol)))
    checks.append(_check("C_quantization_bound", (np.abs(sC - sB) > delta_p + tol),
                         (np.abs(sC - sB) - delta_p - tol)))
    checks.append(_check("C_combined_lower", (sC < sA - delta_p - tol),
                         (sA - delta_p - sC - tol)))
    checks.append(_check("C_combined_upper", (sC > sA + epsB_p + delta_p + tol),
                         (sC - sA - epsB_p - delta_p - tol)))
    return {
        "checks": checks,
        "all_hold": all(c["n_violations"] == 0 for c in checks),
        "tolerance": tol,
        "epsilon_max": {arm: float(np.max(v)) for arm, v in epsilon.items()},
        "delta_max": {arm: float(np.max(v)) for arm, v in delta.items()},
    }


def image_max_certificates(scores: Dict[str, np.ndarray], epsilon: Dict[str, np.ndarray],
                           delta: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Image-max two-sided bounds implied by the per-patch propositions.

    B and E are FP32 subsets: S_X in [S_A, S_A + eps_X].
    C and D are quantized: S_X in [S_A - delta_X, S_A + eps_X + delta_X].
    """
    sA = np.asarray(scores["A"], dtype=np.float64).max(axis=1)
    out: Dict[str, Any] = {}
    for arm in ("B", "E"):
        s = np.asarray(scores[arm], dtype=np.float64).max(axis=1)
        eps = float(np.max(epsilon[arm]))
        out[f"{arm}_vs_A"] = {
            "kind": "fp32_subset",
            "epsilon_max": eps,
            "all_ge_A": bool(np.all(s >= sA - C1.QUANT_TOL)),
            "all_le_A_plus_epsilon": bool(np.all(s <= sA + eps + C1.QUANT_TOL)),
            "max_positive_gap": float(np.max(s - sA)),
            "min_gap": float(np.min(s - sA)),
        }
    sD = np.asarray(scores["D"], dtype=np.float64).max(axis=1)
    epsD = float(np.max(epsilon["D"]))
    deltaD = float(np.max(delta["D"]))
    out["D_vs_A"] = {
        "kind": "quantized",
        "epsilon_max": epsD,
        "delta_max": deltaD,
        "all_ge_A_minus_delta": bool(np.all(sD >= sA - deltaD - C1.QUANT_TOL)),
        "all_le_A_plus_eps_plus_delta": bool(np.all(sD <= sA + epsD + deltaD + C1.QUANT_TOL)),
        "max_positive_gap": float(np.max(sD - sA)),
        "min_gap": float(np.min(sD - sA)),
    }
    sC = np.asarray(scores["C"], dtype=np.float64).max(axis=1)
    epsB = float(np.max(epsilon["B"]))
    deltaC = float(np.max(delta["C"]))
    out["C_vs_A"] = {
        "kind": "quantized",
        "epsilon_B_max": epsB,
        "delta_max": deltaC,
        "all_ge_A_minus_delta": bool(np.all(sC >= sA - deltaC - C1.QUANT_TOL)),
        "all_le_A_plus_eps_plus_delta": bool(np.all(sC <= sA + epsB + deltaC + C1.QUANT_TOL)),
        "max_positive_gap": float(np.max(sC - sA)),
        "min_gap": float(np.min(sC - sA)),
    }
    return out


def quantization_tail_error(scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
    sB = np.asarray(scores["B"], dtype=np.float64)
    sC = np.asarray(scores["C"], dtype=np.float64)
    err = np.abs(sC - sB)
    flatB = sB.reshape(-1)
    k = topk_counts(flatB.size, 0.05)
    tail = topk_indices(flatB, k)
    all_mean = float(err.mean())
    tail_mean = float(err.reshape(-1)[tail].mean())
    return {
        "all_mean_abs_error": all_mean,
        "tail5_mean_abs_error": tail_mean,
        "tail5_max_abs_error": float(err.reshape(-1)[tail].max()),
        "ratio_tail_over_all": tail_mean / all_mean if all_mean > 0 else None,
        "tail_factor_limit": C1.GATES["quantization_C_vs_B"]["tail5_mean_abs_error_max_factor"],
        "tail_within_factor": bool(tail_mean <= max(
            C1.GATES["quantization_C_vs_B"]["tail5_mean_abs_error_max_factor"] * all_mean,
            1e-6)),
    }


def drift_summary(imgmax: Dict[str, Dict[int, np.ndarray]]) -> Dict[str, Any]:
    """Image-max drift for the same witness prefix across boundaries."""
    out: Dict[str, Any] = {}
    for arm in imgmax:
        per_arm = {}
        boundaries = sorted(imgmax[arm])
        for left_t, right_t in zip(boundaries[:-1], boundaries[1:]):
            left = imgmax[arm][left_t]
            right = imgmax[arm][right_t]
            n = min(left.size, right.size)
            per_arm[f"{left_t}->{right_t}"] = {
                "n_images": int(n),
                "spearman": spearman(left[:n], right[:n]),
                "mean_abs_change": float(np.mean(np.abs(left[:n] - right[:n]))),
                "relative_mean_abs_change": float(
                    np.mean(np.abs(left[:n] - right[:n])) / max(np.mean(right[:n]), 1e-12)),
            }
        out[arm] = per_arm
    return out


def gate_evaluation(point: Dict[str, Any], bootstrap: Dict[str, Any],
                    cert: Dict[str, Any], quant: Dict[str, Any], tail: Dict[str, Any],
                    bytes_ratio: Dict[str, float], categories: Sequence[str]) -> Dict[str, Any]:
    """Evaluate the preregistered feasibility gates from point estimates and CIs."""
    q_gate = C1.GATES["quantization_C_vs_B"]
    s_gate = C1.GATES["selection_B_vs_A"]
    m_gate = C1.GATES["memory_efficiency"]

    quant_fail = []
    quant_undefined = []
    for cat in categories:
        row = point["per_category"][cat]["C_B"]
        if row["spearman"] is None:
            quant_undefined.append(f"{cat}:C_B_spearman_undefined")
        elif row["spearman"] < q_gate["image_max_spearman_min_each_category"]:
            quant_fail.append(f"{cat}:C_B_spearman={row['spearman']:.4f}")
        if row["top5_precision"] < q_gate["top5_precision_min"]:
            quant_fail.append(f"{cat}:C_B_top5={row['top5_precision']:.4f}")
    if quant["bound_violations"] > 0:
        quant_fail.append(f"quantization_bound_violations={quant['bound_violations']}")
    if not tail["tail_within_factor"]:
        quant_fail.append(f"C_B_tail_error_ratio={tail['ratio_tail_over_all']}")

    selection_fail = []
    selection_undefined = []
    for cat in categories:
        row = point["per_category"][cat]["B_A"]
        if row["spearman"] is None:
            selection_undefined.append(f"{cat}:B_A_spearman_undefined")
        elif row["spearman"] < s_gate["image_max_spearman_min_each_category"]:
            selection_fail.append(f"{cat}:B_A_spearman={row['spearman']:.4f}")
        if row["top5_precision"] < s_gate["top5_precision_min"]:
            selection_fail.append(f"{cat}:B_A_top5={row['top5_precision']:.4f}")

    bytes_ok = all(m_gate["C_vs_D_byte_ratio"][0] <= bytes_ratio[k] <= m_gate["C_vs_D_byte_ratio"][1]
                   for k in ("C_D", "C_E"))
    if not bytes_ok:
        selection_fail.append(f"byte_ratio={bytes_ratio}")

    # C must beat D or E on the primary or co-primary endpoint vs A.
    # The paired whole-image bootstrap stores C-A minus competitor-A directly.
    def delta(comp: str, metric: str) -> Tuple[float | None, List[float] | None]:
        c = point["macro"]["C_A"][metric]
        x = point["macro"][comp][metric]
        point_delta = None if (c is None or x is None) else c - x
        key = f"adv:C_vs_{comp}:{metric}"
        entry = bootstrap["macro"].get(key)
        ci = entry["ci"] if entry is not None else None
        return point_delta, ci

    advantage = {}
    for comp in ("D_A", "E_A"):
        for metric in ("spearman", "top5_precision"):
            point_delta, ci = delta(comp, metric)
            advantage[f"C_vs_{comp}:{metric}"] = {"delta": point_delta, "ci": ci}
    adv_hits = []
    for key, val in advantage.items():
        if val["delta"] is not None and val["delta"] >= m_gate["meaningful_advantage_min"] \
                and val["ci"] is not None and val["ci"][0] > 0.0:
            adv_hits.append(key)
    # category-level severe loss on the chosen endpoints
    category_loss = {}
    for comp in ("D_A", "E_A"):
        for metric in ("spearman", "top5_precision"):
            losses = []
            for cat in categories:
                c = point["per_category"][cat]["C_A"][metric]
                x = point["per_category"][cat][comp][metric]
                if c is not None and x is not None:
                    losses.append((cat, x - c))  # positive means C worse
            category_loss[f"C_vs_{comp}:{metric}"] = losses
    severe_loss = any(v > m_gate["category_loss_max"] for losses in category_loss.values()
                      for _, v in losses)
    memory_ok = bool(bytes_ok and adv_hits and not severe_loss)

    return {
        "quantization": {
            "passed": len(quant_fail) == 0 and len(quant_undefined) == 0,
            "failures": quant_fail,
            "undefined": quant_undefined,
        },
        "selection": {
            "passed": len(selection_fail) == 0 and len(selection_undefined) == 0,
            "failures": selection_fail,
            "undefined": selection_undefined,
        },
        "memory_efficiency": {
            "passed": memory_ok,
            "bytes_within_5pct": bytes_ok,
            "advantage_hits": adv_hits,
            "advantage": advantage,
            "category_loss": category_loss,
            "severe_category_loss": severe_loss,
        },
        "certificates_hold": cert["all_hold"],
    }
