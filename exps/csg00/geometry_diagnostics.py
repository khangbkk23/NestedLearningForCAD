"""Normal-only geometry diagnostics for CSG-00.

All endpoints are normal-support descriptors.  Nothing in this module knows a
category at scoring time, and no anomaly label, mask or identity is read.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from . import contract as C
from . import scoring


# ---------------------------------------------------------------------------
# Rank statistics (deterministic, tie-aware)
# ---------------------------------------------------------------------------


def rankdata_average(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    if x.size < 3 or y.size != x.size:
        return None
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return None
    return float((rx * ry).sum() / denom)


def kendall_tau_b(x: Iterable[float], y: Iterable[float]) -> float | None:
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    n = x.size
    if n < 3 or y.size != n:
        return None
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    sx = np.sign(x[:, None] - x[None, :])
    sy = np.sign(y[:, None] - y[None, :])
    upper = np.triu_indices(n, 1)
    sx = sx[upper]
    sy = sy[upper]
    concordant = int(np.sum(sx * sy > 0))
    discordant = int(np.sum(sx * sy < 0))
    n0 = n * (n - 1) / 2.0
    n1 = int(np.sum(sx == 0))
    n2 = int(np.sum(sy == 0))
    denom = float(np.sqrt((n0 - n1) * (n0 - n2)))
    if denom == 0.0:
        return None
    return float((concordant - discordant) / denom)


# ---------------------------------------------------------------------------
# Top-k patch sets
# ---------------------------------------------------------------------------


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """First k indices in the total order (-score, flat index)."""
    flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = flat.size
    if k >= n:
        return np.arange(n)
    kth = n - k
    threshold = np.partition(flat, kth)[kth]
    above = np.nonzero(flat > threshold)[0]
    if above.size >= k:
        return above[:k]
    need = k - above.size
    equal = np.nonzero(flat == threshold)[0][:need]
    out = np.concatenate([above, equal])
    out.sort()
    return out


def topk_counts(n_patches: int, fraction: float) -> int:
    return max(1, int(round(fraction * n_patches)))


def _set_metrics(mask_a: np.ndarray, mask_m: np.ndarray, k: int) -> Tuple[float, float]:
    inter = int(np.count_nonzero(mask_a & mask_m))
    precision = inter / float(k)
    union = int(np.count_nonzero(mask_a | mask_m))
    iou = inter / float(union) if union else 0.0
    return precision, iou


# ---------------------------------------------------------------------------
# Selection-level metrics (used for point estimates and bootstrap replicates)
# ---------------------------------------------------------------------------


def _metrics_on_selection(scores: Dict[str, np.ndarray], idx: np.ndarray,
                          with_topk: bool = True) -> Dict[str, Any]:
    """Compute all paired metrics for one image selection (may contain duplicates)."""
    sA = scores["A"][idx]
    sB = scores["B"][idx]
    sC = scores["C"][idx]

    imgA = sA.max(axis=1)
    imgB = sB.max(axis=1)
    imgC = sC.max(axis=1)

    rho_b = spearman(imgA, imgB)
    rho_c = spearman(imgA, imgC)
    tau_b = kendall_tau_b(imgA, imgB)
    tau_c = kendall_tau_b(imgA, imgC)

    covB = float(np.mean(sB <= sA))
    covC = float(np.mean(sC <= sA))
    zeroB = float(np.mean(sB == 0.0))
    zeroC = float(np.mean(sC == 0.0))

    flatA = sA.reshape(-1)
    flatB = sB.reshape(-1)
    flatC = sC.reshape(-1)
    topk: Dict[str, Any] = {}
    for frac in (C.TOP_FRACTIONS if with_topk else ()):
        n = flatA.size
        k = topk_counts(n, frac)
        iA = topk_indices(flatA, k)
        iB = topk_indices(flatB, k)
        iC = topk_indices(flatC, k)
        maskA = np.zeros(n, dtype=bool)
        maskA[iA] = True
        maskB = np.zeros(n, dtype=bool)
        maskB[iB] = True
        maskC = np.zeros(n, dtype=bool)
        maskC[iC] = True
        pb, ib = _set_metrics(maskA, maskB, k)
        pc, ic = _set_metrics(maskA, maskC, k)
        topk[f"{frac:.2f}"] = {
            "k": k,
            "precision_B": pb,
            "iou_B": ib,
            "precision_C": pc,
            "iou_C": ic,
            "delta_precision": pb - pc,
            "delta_iou": ib - ic,
        }

    return {
        "n_images": int(idx.size),
        "rho_B": rho_b,
        "rho_C": rho_c,
        "delta_rho": None if (rho_b is None or rho_c is None) else rho_b - rho_c,
        "kendall_B": tau_b,
        "kendall_C": tau_c,
        "delta_kendall": None if (tau_b is None or tau_c is None) else tau_b - tau_c,
        "coverage_B": covB,
        "coverage_C": covC,
        "delta_cov": covB - covC,
        "zero_fraction_B": zeroB,
        "zero_fraction_C": zeroC,
        "image_score_B": _summarize(imgB),
        "image_score_C": _summarize(imgC),
        "image_score_A": _summarize(imgA),
        "topk": topk,
    }


def _summarize(values: np.ndarray) -> Dict[str, Any]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    q = np.quantile(v, [0.5, 0.9, 0.99]).tolist()
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(q[0]),
        "q90": float(q[1]),
        "q99": float(q[2]),
        "max": float(v.max()),
        "min": float(v.min()),
    }


def patch_summary(values: np.ndarray) -> Dict[str, Any]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    edges = C.SCORE_HIST_EDGES
    counts, _ = np.histogram(v, bins=np.asarray(edges, dtype=np.float64))
    hist = {f"[{edges[i]},{edges[i+1]})": float(counts[i] / v.size) for i in range(len(counts))}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "q90": float(np.quantile(v, 0.90)),
        "q99": float(np.quantile(v, 0.99)),
        "max": float(v.max()),
        "zero_fraction": float(np.mean(v == 0.0)),
        "histogram_fraction": hist,
    }


def _ci(samples: List[float]) -> Dict[str, Any]:
    arr = np.asarray(samples, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"ci": [None, None], "n_valid": 0, "n_total": int(arr.size),
                "bootstrap_mean": None, "bootstrap_sd": None}
    lo, hi = np.percentile(finite, C.BOOTSTRAP_CI).tolist()
    return {
        "ci": [float(lo), float(hi)],
        "n_valid": int(finite.size),
        "n_total": int(arr.size),
        "bootstrap_mean": float(finite.mean()),
        "bootstrap_sd": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
    }


def _macro_of(metrics_by_category: Dict[str, Dict[str, Any]],
              keys: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in keys:
        values = [m[key] for m in metrics_by_category.values() if m[key] is not None]
        out[key] = float(np.mean(values)) if values else None
    return out


_MACRO_KEYS = ("rho_B", "rho_C", "delta_rho", "kendall_B", "kendall_C", "delta_kendall",
               "coverage_B", "coverage_C", "delta_cov", "zero_fraction_B", "zero_fraction_C")


def bootstrap_selection(scores: Dict[str, np.ndarray], groups: Dict[str, np.ndarray],
                        reps: int, boundary: int) -> Dict[str, Any]:
    """Whole-image, category-stratified, paired bootstrap.

    Rank and coverage endpoints use the equal-category macro (mean over categories),
    matching the preregistered decision bands.  Top-k patch statistics are pooled
    across the resampled images because patch sets do not have a category mean.
    """
    categories = list(groups.keys())
    cat_point = {cat: _metrics_on_selection(scores, groups[cat], with_topk=False)
                 for cat in categories}
    pooled_point = _metrics_on_selection(
        scores, np.concatenate([groups[c] for c in categories]), with_topk=True)
    macro_point = _macro_of(cat_point, _MACRO_KEYS)

    rng = C.rng_from(C.BOOTSTRAP_SEED, boundary)
    keys = ["delta_rho", "rho_B", "rho_C", "delta_cov", "coverage_B", "coverage_C",
            "delta_kendall", "zero_fraction_B", "zero_fraction_C"]
    samples: Dict[str, List[float]] = {k: [] for k in keys}
    topk_samples: Dict[str, Dict[str, List[float]]] = {
        f"{frac:.2f}": {k: [] for k in ("precision_B", "iou_B", "precision_C", "iou_C",
                                        "delta_precision", "delta_iou")}
        for frac in C.TOP_FRACTIONS
    }
    for _ in range(reps):
        selected = {}
        for cat in categories:
            idx = groups[cat]
            selected[cat] = idx[rng.integers(0, idx.size, idx.size)]
        cat_metrics = {cat: _metrics_on_selection(scores, selected[cat], with_topk=False)
                       for cat in categories}
        macro = _macro_of(cat_metrics, _MACRO_KEYS)
        for key in keys:
            value = macro[key]
            samples[key].append(np.nan if value is None else float(value))
        pooled = _metrics_on_selection(
            scores, np.concatenate([selected[c] for c in categories]), with_topk=True)
        for frac_key, bucket in topk_samples.items():
            for key in bucket:
                bucket[key].append(float(pooled["topk"][frac_key][key]))

    return {
        "replicates": int(reps),
        "seed": C.BOOTSTRAP_SEED,
        "boundary": int(boundary),
        "resampling": "whole images, category-stratified, paired across arms",
        "endpoint": "rank/coverage: equal-category macro; top-k: pooled images",
        "point_estimate": pooled_point,
        "macro_point_estimate": macro_point,
        "per_category_point_estimate": cat_point,
        "bootstrap": {key: _ci(samples[key]) for key in keys},
        "topk_bootstrap": {
            frac_key: {key: _ci(values) for key, values in bucket.items()}
            for frac_key, bucket in topk_samples.items()
        },
    }


def bootstrap_single_category(scores: Dict[str, np.ndarray], idx: np.ndarray,
                              reps: int, boundary: int, category: str) -> Dict[str, Any]:
    rng = C.rng_from(C.BOOTSTRAP_SEED, boundary, sum(ord(ch) for ch in category))
    point = _metrics_on_selection(scores, idx, with_topk=False)
    keys = ["rho_B", "rho_C", "delta_rho", "coverage_B", "coverage_C", "delta_cov",
            "zero_fraction_B", "zero_fraction_C"]
    samples: Dict[str, List[float]] = {k: [] for k in keys}
    for _ in range(reps):
        selected = idx[rng.integers(0, idx.size, idx.size)]
        metrics = _metrics_on_selection(scores, selected, with_topk=False)
        for key in keys:
            value = metrics[key]
            samples[key].append(np.nan if value is None else float(value))
    return {
        "category": category,
        "replicates": int(reps),
        "point_estimate": point,
        "bootstrap": {key: _ci(samples[key]) for key in keys},
    }


# ---------------------------------------------------------------------------
# Feature-space gap probes (normal training features only)
# ---------------------------------------------------------------------------


def _bin_mode_distances(bin_index: int, Z: torch.Tensor, state: Dict[str, np.ndarray],
                        device: str = "cpu") -> torch.Tensor:
    ready = state["ready"]
    best = torch.full((Z.shape[0],), float("inf"), dtype=torch.float64, device=device)
    n_ready = 0
    for k in range(C.K_MODES):
        if int(ready[bin_index, k]) == 0:
            continue
        n_ready += 1
        dist = scoring.mode_distance(
            Z, torch.as_tensor(state["mu"][bin_index, k]),
            torch.as_tensor(state["U"][bin_index, k]),
            torch.as_tensor(state["R"][bin_index, k]),
            float(state["rho"][bin_index, k]), device=device)
        best = torch.minimum(best, dist)
    if n_ready == 0:
        raise RuntimeError(f"bin {bin_index}: no ready mode")
    return best


def gap_probe_summary(writer_bins: Sequence[torch.Tensor], probe_index: Dict[str, Any],
                      mode_state: Dict[str, np.ndarray], reservoir_state: Dict[str, np.ndarray],
                      witness_binned_A: np.ndarray, positions: Sequence[np.ndarray],
                      boundary: int, max_examples: int = 12) -> Dict[str, Any]:
    """Interpolate/extrapolate between normal training descriptors inside a bin.

    A probe is in the high-A region when its arm-A distance is at least the
    median arm-A distance of eligible witness patches in the same bin.  Arm B
    false-accepts it when s_B <= GAP_ACCEPT_RATIO * s_A.  These are proxy
    probes, not synthetic anomalies and not Pixel AP.
    """
    records: List[Dict[str, Any]] = []
    for b in range(C.BINS):
        D = writer_bins[b].to(torch.float64)
        reps = probe_index[str(b)]["reps"]
        cat_reps = probe_index[str(b)]["cat_reps"]
        pos = positions[b]
        tau = float(np.median(witness_binned_A[:, pos]))
        probes: List[torch.Tensor] = []
        meta: List[Dict[str, Any]] = []
        mode_keys = sorted(reps.keys())
        if len(mode_keys) == 2:
            x0 = D[int(reps[mode_keys[0]])]
            x1 = D[int(reps[mode_keys[1]])]
            for lam in C.GAP_LAMBDAS:
                probes.append((1.0 - lam) * x0 + lam * x1)
                meta.append({"kind": "mode_interp", "lambda": float(lam),
                             "label": "mode_pair"})
            for lam in C.GAP_EXTRAPOLATIONS:
                probes.append((1.0 - lam) * x0 + lam * x1)
                meta.append({"kind": "mode_extrap", "lambda": float(lam),
                             "label": "mode_pair"})
        cats = sorted(cat_reps.keys())
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                c0, c1 = cats[i], cats[j]
                x0 = D[int(cat_reps[c0])]
                x1 = D[int(cat_reps[c1])]
                for lam in C.GAP_CROSS_LAMBDAS:
                    probes.append((1.0 - lam) * x0 + lam * x1)
                    meta.append({"kind": "cross_category", "lambda": float(lam),
                                 "label": f"{c0}|{c1}"})
        if not probes:
            continue
        Z = torch.stack(probes, dim=0)
        sA = scoring.nn_distance(Z, D).cpu()
        sB = _bin_mode_distances(b, Z, mode_state).cpu()
        n_res = int(reservoir_state["count"][b])
        sC = scoring.nn_distance(
            Z, torch.as_tensor(reservoir_state["res"][b, :n_res])).cpu()
        for row in range(Z.shape[0]):
            a = float(sA[row]); bb = float(sB[row]); cc = float(sC[row])
            high = a >= tau
            records.append({
                "bin": int(b),
                **meta[row],
                "A": a,
                "B": bb,
                "C": cc,
                "tau_bin": tau,
                "high_A": bool(high),
                "B_accepts": bool(high and a > 0.0 and bb <= C.GAP_ACCEPT_RATIO * a),
                "C_accepts": bool(high and a > 0.0 and cc <= C.GAP_ACCEPT_RATIO * a),
            })

    high = [r for r in records if r["high_A"]]
    accepted_B = [r for r in high if r["B_accepts"]]
    accepted_C = [r for r in high if r["C_accepts"]]
    by_kind: Dict[str, Any] = {}
    for kind in sorted({r["kind"] for r in records}):
        sub = [r for r in records if r["kind"] == kind]
        sub_high = [r for r in sub if r["high_A"]]
        by_kind[kind] = {
            "n": len(sub),
            "n_high_A": len(sub_high),
            "gfa_B": (sum(1 for r in sub_high if r["B_accepts"]) / len(sub_high)) if sub_high else None,
            "gfa_C": (sum(1 for r in sub_high if r["C_accepts"]) / len(sub_high)) if sub_high else None,
            "mean_A": float(np.mean([r["A"] for r in sub])),
            "mean_B": float(np.mean([r["B"] for r in sub])),
        }
    examples = sorted(accepted_B, key=lambda r: (-r["A"], r["bin"], r["lambda"]))[:max_examples]
    return {
        "boundary": int(boundary),
        "n_probes": len(records),
        "n_high_A": len(high),
        "n_B_accepted": len(accepted_B),
        "n_C_accepted": len(accepted_C),
        "gfa_B": (len(accepted_B) / len(high)) if high else None,
        "gfa_C": (len(accepted_C) / len(high)) if high else None,
        "mean_A_high": float(np.mean([r["A"] for r in high])) if high else None,
        "mean_B_high": float(np.mean([r["B"] for r in high])) if high else None,
        "mean_C_high": float(np.mean([r["C"] for r in high])) if high else None,
        "by_kind": by_kind,
        "examples_B_accepted": examples,
        "definition": {
            "high_A": "A-score >= median eligible-witness A-score in the same bin",
            "accept": f"s_B <= {C.GAP_ACCEPT_RATIO} * s_A",
            "status": "feature-space proxy; NOT a real anomaly and NOT Pixel AP",
        },
    }


# ---------------------------------------------------------------------------
# One-sided support overreach sampling (descriptive lower bound)
# ---------------------------------------------------------------------------


def support_sample_summary(writer_bins: Sequence[torch.Tensor], mode_state: Dict[str, np.ndarray],
                           witness_binned_A: np.ndarray, positions: Sequence[np.ndarray],
                           boundary: int) -> Dict[str, Any]:
    """Sample points inside each fitted mode and measure distance to normal data.

    Finite sampling gives a descriptive LOWER bound on
    ``sup_{u in S_hat} dist(u, D)``; it is not a Hausdorff certificate.
    """
    rng = C.rng_from(C.SUPPORT_SAMPLE_SEED, boundary)
    A_values: List[float] = []
    B_values: List[float] = []
    ge_total = 0
    ge_count = 0
    per_bin: Dict[str, Any] = {}
    for b in range(C.BINS):
        D = writer_bins[b].to(torch.float64)
        pos_t = positions[b]
        tau = float(np.median(witness_binned_A[:, pos_t]))
        bin_A: List[float] = []
        bin_B: List[float] = []
        for k in range(C.K_MODES):
            if int(mode_state["ready"][b, k]) == 0:
                continue
            mu = torch.as_tensor(mode_state["mu"][b, k], dtype=torch.float64)
            U = torch.as_tensor(mode_state["U"][b, k], dtype=torch.float64)
            R = torch.as_tensor(mode_state["R"][b, k], dtype=torch.float64)
            rho = float(mode_state["rho"][b, k])
            samples = []
            for _ in range(C.SUPPORT_SAMPLES_PER_MODE):
                a = torch.as_tensor((rng.random(C.Q_TANGENT) * 2.0 - 1.0), dtype=torch.float64) * R
                g = torch.as_tensor(rng.standard_normal(C.DIM), dtype=torch.float64)
                g = g - U @ (U.transpose(0, 1) @ g)
                g_norm = float(g.norm().item())
                if g_norm <= 0.0:
                    continue
                radius = rho * (rng.random() ** (1.0 / max(C.DIM - C.Q_TANGENT, 1)))
                z = mu + U @ a + (radius / g_norm) * g
                samples.append(z)
            if not samples:
                continue
            Z = torch.stack(samples, dim=0)
            sA = scoring.nn_distance(Z, D).cpu().numpy()
            sB = _bin_mode_distances(b, Z, mode_state).cpu().numpy()
            bin_A.extend(sA.tolist())
            bin_B.extend(sB.tolist())
            ge_count += int(np.sum(sA >= tau))
            ge_total += int(sA.size)
        if bin_A:
            arr = np.asarray(bin_A)
            per_bin[str(b)] = {
                "tau_bin": tau,
                "n": int(arr.size),
                "A_median": float(np.median(arr)),
                "A_q90": float(np.quantile(arr, 0.90)),
                "A_max": float(arr.max()),
                "fraction_A_ge_tau": float(np.mean(arr >= tau)),
            }
            A_values.extend(arr.tolist())
            B_values.extend(bin_B)
    arr = np.asarray(A_values) if A_values else np.zeros((0,))
    return {
        "boundary": int(boundary),
        "n_samples": int(arr.size),
        "A_quantiles": {
            "median": float(np.median(arr)) if arr.size else None,
            "q90": float(np.quantile(arr, 0.90)) if arr.size else None,
            "q99": float(np.quantile(arr, 0.99)) if arr.size else None,
            "max": float(arr.max()) if arr.size else None,
        },
        "B_max_abs": float(np.max(np.abs(B_values))) if B_values else None,
        "fraction_A_ge_bin_tau_pooled": (ge_count / ge_total) if ge_total else None,
        "mean_fraction_A_ge_tau_across_bins": (
            float(np.mean([v["fraction_A_ge_tau"] for v in per_bin.values()])) if per_bin else None),
        "per_bin": per_bin,
        "status": "descriptive lower bound on one-sided overreach; not a certificate",
    }


# ---------------------------------------------------------------------------
# Mode occupancy / aggregation
# ---------------------------------------------------------------------------


def mode_occupancy_summary(bins_diag: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ready_counts = [int(d["n_modes_ready"]) for d in bins_diag]
    dropped = [int(d["n_modes_dropped"]) for d in bins_diag]
    all_counts = [c for d in bins_diag for c in d["counts"]]
    centres = [d["center_distance"] for d in bins_diag if d["center_distance"] == d["center_distance"]]
    rho = [x for d in bins_diag for x in d["rho"]]
    rmax = [x for d in bins_diag for x in d["R_max"]]
    ortho = [d["orthogonality_max_error"] for d in bins_diag]
    return {
        "bins": int(len(bins_diag)),
        "ready_mode_count_histogram": {
            str(k): int(ready_counts.count(k)) for k in sorted(set(ready_counts))
        },
        "bins_with_dropped_mode": int(sum(1 for x in dropped if x > 0)),
        "mode_count_min": int(min(all_counts)),
        "mode_count_median": float(np.median(all_counts)),
        "mode_count_max": int(max(all_counts)),
        "mode_counts": [int(x) for x in all_counts],
        "center_distance_median": float(np.median(centres)) if centres else None,
        "center_distance_min": float(np.min(centres)) if centres else None,
        "duplicate_centres_lt_1e-6": int(sum(1 for x in centres if x < 1e-6)),
        "rho_median": float(np.median(rho)) if rho else None,
        "rho_max": float(np.max(rho)) if rho else None,
        "R_max_median": float(np.median(rmax)) if rmax else None,
        "R_max_max": float(np.max(rmax)) if rmax else None,
        "orthogonality_max_error": float(np.max(ortho)) if ortho else None,
    }


def mode_utilization(queries_by_bin: Sequence[torch.Tensor],
                     state: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Count witness patches whose closest ready mode is each mode slot."""
    counts = np.zeros((C.BINS, C.K_MODES), dtype=np.int64)
    for b in range(C.BINS):
        Q = queries_by_bin[b]
        best = torch.full((Q.shape[0],), float("inf"), dtype=torch.float64)
        arg = torch.full((Q.shape[0],), -1, dtype=torch.long)
        for k in range(C.K_MODES):
            if int(state["ready"][b, k]) == 0:
                continue
            dist = scoring.mode_distance(
                Q, torch.as_tensor(state["mu"][b, k]),
                torch.as_tensor(state["U"][b, k]),
                torch.as_tensor(state["R"][b, k]),
                float(state["rho"][b, k]))
            upd = dist < best
            best = torch.where(upd, dist, best)
            arg = torch.where(upd, torch.full_like(arg, k), arg)
        for k in range(C.K_MODES):
            counts[b, k] = int((arg == k).sum().item())
    total = int(counts.sum())
    return {
        "counts_by_bin_mode": counts.tolist(),
        "fraction_by_mode_slot": [
            float(counts[:, k].sum() / total) if total else None for k in range(C.K_MODES)
        ],
        "bins_with_unused_ready_mode": int(sum(
            1 for b in range(C.BINS) for k in range(C.K_MODES)
            if int(state["ready"][b, k]) == 1 and counts[b, k] == 0
        )),
    }
