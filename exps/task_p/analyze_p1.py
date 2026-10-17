"""Read-only P1 metric analysis from saved patch scores, without learner access."""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from exps.task_p.p0_normal_support import require
from exps.task_p.p1_data import CATEGORIES, BOUNDARIES, native_mask, write_json
from exps.task_p.p1_protocol import OUT, configure, load_protocol
from exps.task_p.p1_readout import (ARMS, PixelRanking, BoundedPixelBootstrap,
    bootstrap_weights, image_metrics, interval, native_map, score_histogram, classify)


def summarize_group(records, root, weights):
    labels = [r["source"]["label"] for r in records]
    scores = [r["image_score"] for r in records]
    histograms, normal_pixel_means, defect_pixel_means, normal_map_means = [], [], [], []
    for row in records:
        mask = native_mask(root, row["source"])
        values = native_map(torch.tensor(row["patch_distances"], dtype=torch.float64), row["source"]["hw"])
        require(int(mask.sum()) == row["positive_pixels"] and mask.size == row["pixels"], "saved mask counts differ")
        histograms.append(score_histogram(values, mask))
        if (~mask).any():
            normal_pixel_means.append(float(values[~mask].mean(dtype=np.float64)))
        if mask.any():
            defect_pixel_means.append(float(values[mask].mean(dtype=np.float64)))
        if row["source"]["label"] == 0:
            normal_map_means.append(float(values.mean(dtype=np.float64)))
    ranking = PixelRanking(histograms)
    pixel = ranking.metrics()
    require(pixel["degenerate"] is None, "BLOCKED_TECHNICAL: degenerate native pixels")
    image = image_metrics(labels, scores)
    require(image["degenerate"] is None, "BLOCKED_TECHNICAL: category lacks both image classes")
    del ranking
    gc.collect()
    bounded = BoundedPixelBootstrap(histograms)
    lower, upper = [], []
    for start in range(0, len(weights), 16):
        lo, hi = bounded.bounds(weights[start:start + 16])
        lower.extend(lo.tolist())
        upper.extend(hi.tolist())
    lower, upper = np.asarray(lower), np.asarray(upper)
    pos_by_image = np.array([h["positive"].sum() for h in histograms])
    n_by_image = np.array([h["total"].sum() for h in histograms])
    prevalence = (weights @ pos_by_image) / (weights @ n_by_image)
    image_boot = np.array([image_metrics(labels, scores, w)["auroc"] for w in weights])
    normal_scores = np.array(scores)[np.asarray(labels) == 0]
    anomalous_scores = np.array(scores)[np.asarray(labels) == 1]
    point = {"image_auroc": image["auroc"], "pixel_ap": pixel["ap"], "pixel_auroc": pixel["auroc"],
             "prevalence": pixel["prevalence"], "images": len(labels),
             "normal_images": int(sum(np.asarray(labels) == 0)), "anomalous_images": int(sum(labels)),
             "native_pixels": int(n_by_image.sum()), "anomaly_pixels": int(pos_by_image.sum()),
             "normal_image_score_median_p95_max": np.quantile(normal_scores, [.5, .95, 1]).tolist(),
             "normal_image_mean": float(normal_scores.mean()), "anomaly_image_mean": float(anomalous_scores.mean()),
             "normal_pixel_mean_equal_image": float(np.mean(normal_pixel_means)),
             "defect_pixel_mean_equal_defective_image": float(np.mean(defect_pixel_means)),
             "normal_image_map_mean": float(np.mean(normal_map_means)),
             "image_auroc_ci": interval(image_boot),
             "pixel_ap_ci_conservative": [interval(lower)[0], interval(upper)[1]],
             "max_bootstrap_within_bin_AP_width": float((upper - lower).max())}
    return point, {"image_auroc": image_boot, "ap_lower": lower, "ap_upper": upper,
                   "prevalence": prevalence}, histograms


def contrast(point_delta, lower, upper):
    ci = [interval(lower)[0], interval(upper)[1]]
    if point_delta >= .02 and ci[0] > 0:
        label = "OUTPERFORMS"
    elif point_delta <= -.02 and ci[1] < 0:
        label = "UNDERPERFORMS"
    elif ci[0] >= -.02 and ci[1] <= .02:
        label = "MATCHES_WITHIN_2PP"
    else:
        label = "INCONCLUSIVE"
    return {"delta_macro_pixel_ap": point_delta, "paired_95pct_CI_conservative": ci, "decision": label}


def heatmaps(records, root):
    """Frozen failure-case selection, visualization only; scores stay in raw units."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/hope_cad_p1_mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    final = [r for r in records if r["boundary"] == 350]
    selected = set()
    for c in CATEGORIES:
        for a in ARMS:
            group = [r for r in final if r["arm"] == a and r["source"]["category"] == c]
            normal = max((r for r in group if r["source"]["label"] == 0), key=lambda r: (r["image_score"], r["source"]["relative_path"]))
            defective = min((r for r in group if r["source"]["label"] == 1), key=lambda r: (r["image_score"], r["source"]["relative_path"]))
            selected.update((normal["source"]["relative_path"], defective["source"]["relative_path"]))
    directory = OUT / "heatmaps"
    directory.mkdir(exist_ok=False)
    output = []
    for path in sorted(selected):
        group = [r for r in final if r["source"]["relative_path"] == path]
        source = group[0]["source"]
        maps = {r["arm"]: native_map(torch.tensor(r["patch_distances"]), source["hw"]) for r in group}
        vmax = max(float(v.max()) for v in maps.values())
        fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
        with Image.open(Path(root) / path) as image:
            axes[0].imshow(image.convert("RGB"))
        axes[0].set_title(path)
        axes[1].imshow(native_mask(root, source), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("native annotation")
        for i, a in enumerate(ARMS, 2):
            axes[i].imshow(maps[a], cmap="inferno", vmin=0, vmax=vmax)
            axes[i].set_title(a)
        for ax in axes:
            ax.axis("off")
        target = directory / (path.replace("/", "_").removesuffix(".png") + ".png")
        fig.savefig(target, dpi=130)
        plt.close(fig)
        output.append({"source": path, "file": str(target.relative_to(OUT)), "raw_vmax": vmax})
    write_json(OUT / "heatmap_manifest.json", output)


def analyze():
    p, digest = load_protocol()
    require((OUT / "run_completed.json").exists(), "experiment incomplete; do not finalize")
    require(not (OUT / "summary.json").exists(), "analysis already finalized")
    started = time.perf_counter()
    all_records = [r for b in BOUNDARIES for r in json.loads((OUT / f"predictions_t{b}.json").read_text())]
    repetitions = p["statistics"]["repetitions"]
    draws = {}
    for index, c in enumerate(CATEGORIES):
        rows = [r for r in p["evaluations"] if r["category"] == c]
        gen = np.random.default_rng(p["statistics"]["seed"] + index)
        draws[c] = np.stack([bootstrap_weights([r["label"] for r in rows], gen) for _ in range(repetitions)])
    metrics, boot = {}, {}
    for b in BOUNDARIES:
        for c in CATEGORIES:
            for a in ARMS:
                rows = [r for r in all_records if r["boundary"] == b and r["arm"] == a and r["source"]["category"] == c]
                if not rows:
                    continue
                require([r["source"] for r in rows] == [r for r in p["evaluations"] if r["category"] == c], "metric identities misaligned")
                key = f"{b}/{c}/{a}"
                metrics[key], boot[key], _ = summarize_group(rows, p["dataset_root"], draws[c])
                print(f"metrics complete: {key}", flush=True)
                gc.collect()
    final, acquisition, macro, macro_ci, worst = {}, {}, {}, {}, {}
    retention = []
    macro_boot = {}
    for a in ARMS:
        final[a] = {c: metrics[f"350/{c}/{a}"] for c in CATEGORIES}
        acquisition[a] = {c: metrics[f"{(i + 1) * 100}/{c}/{a}"] for i, c in enumerate(CATEGORIES)}
        macro[a] = {name: float(np.mean([final[a][c][name] for c in CATEGORIES]))
                    for name in ("image_auroc", "pixel_ap", "pixel_auroc", "prevalence")}
        mb = {name: np.mean([boot[f"350/{c}/{a}"][name] for c in CATEGORIES], axis=0)
              for name in ("image_auroc", "ap_lower", "ap_upper", "prevalence")}
        macro_boot[a] = mb
        macro_ci[a] = {"image_auroc": interval(mb["image_auroc"]),
                       "pixel_ap": [interval(mb["ap_lower"])[0], interval(mb["ap_upper"])[1]],
                       "ap_minus_prevalence": [interval(mb["ap_lower"] - mb["prevalence"])[0],
                                               interval(mb["ap_upper"] - mb["prevalence"])[1]]}
        worst[a] = 0.
        for i, c in enumerate(CATEGORIES):
            acquired = (i + 1) * 100
            initial = acquisition[a][c]
            useful = initial["image_auroc"] >= .70 and initial["pixel_ap"] >= max(3 * initial["prevalence"], initial["prevalence"] + .05)
            # Include pre-return boundaries; reacquisition cannot hide previous loss.
            for b in BOUNDARIES:
                if b <= acquired:
                    continue
                first, later = boot[f"{acquired}/{c}/{a}"], boot[f"{b}/{c}/{a}"]
                ap_loss = initial["pixel_ap"] - metrics[f"{b}/{c}/{a}"]["pixel_ap"]
                auc_loss = initial["image_auroc"] - metrics[f"{b}/{c}/{a}"]["image_auroc"]
                ap_ci = [interval(first["ap_lower"] - later["ap_upper"])[0],
                         interval(first["ap_upper"] - later["ap_lower"])[1]]
                auc_ci = interval(first["image_auroc"] - later["image_auroc"])
                distinct_task_update = not (b == 350 and c == CATEGORIES[0])
                confirmed = useful and distinct_task_update and ((ap_loss > .10 and ap_ci[0] > 0) or (auc_loss > .10 and auc_ci[0] > 0))
                if confirmed:
                    worst[a] = max(worst[a], ap_loss, auc_loss)
                retention.append({"arm": a, "category": c, "acquisition_boundary": acquired, "later_boundary": b,
                                  "pixel_ap_loss": ap_loss, "image_auroc_loss": auc_loss,
                                  "pixel_ap_loss_CI_conservative": ap_ci, "image_auroc_loss_CI": auc_ci,
                                  "initially_useful": useful, "distinct_task_update": distinct_task_update,
                                  "retention_failure": bool(confirmed)})
    contrasts = {}
    for left, right in ((ARMS[2], ARMS[1]), (ARMS[1], ARMS[0])):
        l, r = macro_boot[left], macro_boot[right]
        contrasts[f"{left}_minus_{right}"] = contrast(macro[left]["pixel_ap"] - macro[right]["pixel_ap"],
                                                     l["ap_lower"] - r["ap_upper"], l["ap_upper"] - r["ap_lower"])
    # Global AP is separately pooled, never substituted for equal-category macro.
    global_pixel = {}
    for a in ARMS:
        histograms = []
        for r in all_records:
            if r["boundary"] == 350 and r["arm"] == a:
                values = native_map(torch.tensor(r["patch_distances"]), r["source"]["hw"])
                histograms.append(score_histogram(values, native_mask(p["dataset_root"], r["source"])))
        global_pixel[a] = PixelRanking(histograms).metrics()
        del histograms
        gc.collect()
    summary = {"scientific_decision": classify(final, acquisition, worst, macro_ci),
               "evidence_status": "DEVELOPMENT EVIDENCE — previously exposed MVTec AD v1; no independent validation",
               "protocol_sha256": digest, "per_boundary_category": metrics, "final": final,
               "macro": macro, "macro_ci": macro_ci, "contrasts": contrasts,
               "retention": retention, "globally_pooled_pixel": global_pixel,
               "uncertainty": "whole-image paired bootstrap; exact image AUROC; native AP intervals enclose all within-bin rankings",
               "analysis_seconds": time.perf_counter() - started}
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "bootstrap.json", {k: {n: v.tolist() for n, v in value.items()} for k, value in boot.items()})
    heatmaps(all_records, p["dataset_root"])
    return summary


if __name__ == "__main__":
    configure()
    result = analyze()
    print(result["scientific_decision"], flush=True)
