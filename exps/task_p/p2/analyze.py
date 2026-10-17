"""Native development metrics and preregistered paired image-level inference."""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from exps.task_p.p1_data import native_mask, write_json
from exps.task_p.p1_readout import bootstrap_weights, native_map
from exps.task_p.p2.data import ARMS, OUT, P1, require
from exps.task_p.p2.experiment import load_protocol, now
from exps.task_p.p2.metrics import bootstrap_bounds, exact_ranking, image_auroc
from exps.task_p.p2.preflight import configure


def percentile(lower, upper=None):
    return [float(np.quantile(lower, .025)), float(np.quantile(lower if upper is None else upper, .975))]


def analyze():
    configure()
    protocol = load_protocol()
    require((OUT / "evaluation_completed.json").exists(), "scoring incomplete")
    write_json(OUT / "analysis_started.json", {"utc": now()})
    started = time.perf_counter()
    categories = ("bottle", "carpet", "hazelnut")
    sources = {row["relative_path"]: row for row in protocol["evaluations"]}
    masks = {name: native_mask(protocol["dataset_root"], row) for name, row in sources.items()}
    draws = {}
    for index, category in enumerate(categories):
        group = [row for row in protocol["evaluations"] if row["category"] == category]
        generator = np.random.default_rng(102701 + index)
        draws[category] = np.stack([bootstrap_weights([row["label"] for row in group], generator) for _ in range(1000)])
    points, samples, separation = {}, {}, {}
    torch.cuda.reset_peak_memory_stats()
    for order in protocol["orders"]:
        for seed in protocol["seeds"]:
            directory = OUT / f"{order}_seed{seed}"
            records = json.loads((directory / "predictions.json").read_text())
            for boundary in protocol["boundaries"]:
                for category in categories:
                    rows = [record for record in records if record["boundary"] == boundary
                            and sources[record["source"]]["category"] == category]
                    if not rows:
                        continue
                    for arm in ARMS:
                        group = [record for record in rows if record["arm"] == arm]
                        require(len(group) == 20, "incomplete category evaluation")
                        image_maps, targets, labels, scores = [], [], [], []
                        for record in group:
                            source = sources[record["source"]]
                            distances = torch.tensor(record["patch_scores"], dtype=torch.float32)
                            require(float(distances.max()) == record["image_score"], "image max readout differs")
                            image_maps.append(torch.from_numpy(native_map(distances, source["hw"])).cuda())
                            targets.append(torch.from_numpy(masks[record["source"]]).cuda())
                            labels.append(source["label"])
                            scores.append(record["image_score"])
                        key = f"{order}/{seed}/{boundary}/{category}/{arm}"
                        metric = exact_ranking(torch.cat([score.flatten() for score in image_maps]),
                                               torch.cat([target.flatten() for target in targets]))
                        metric["image_auroc"] = float(image_auroc(labels, scores, np.ones(20))[0])
                        points[key] = metric
                        lower, upper = bootstrap_bounds(image_maps, targets, draws[category])
                        auc = image_auroc(labels, scores, draws[category])
                        samples[key] = {"ap_low": lower.tolist(), "ap_high": upper.tolist(), "image_auroc": auc.tolist()}
                        normal_scores = np.array(scores)[np.array(labels) == 0]
                        anomaly_scores = np.array(scores)[np.array(labels) == 1]
                        positive_sum = sum(float(score[mask].double().sum()) for score, mask in zip(image_maps, targets))
                        negative_sum = sum(float(score[~mask].double().sum()) for score, mask in zip(image_maps, targets))
                        separation[key] = {"normal_image_mean": float(normal_scores.mean()),
                            "normal_image_median": float(np.median(normal_scores)), "normal_image_p95": float(np.quantile(normal_scores, .95)),
                            "anomaly_image_mean": float(anomaly_scores.mean()),
                            "positive_pixel_mean": positive_sum / metric["positive_pixels"],
                            "negative_pixel_mean": negative_sum / (metric["native_pixels"]-metric["positive_pixels"])}
                        del image_maps, targets
            print("METRICS", order, seed, "seconds", round(time.perf_counter()-started, 1), flush=True)
    write_json(OUT / "point_metrics.json", points)
    write_json(OUT / "bootstrap_samples.json", samples)
    write_json(OUT / "score_separation.json", separation)

    def category_point(order, boundary, category, arm, metric):
        return float(np.mean([points[f"{order}/{seed}/{boundary}/{category}/{arm}"][metric] for seed in protocol["seeds"]]))

    def category_samples(order, boundary, category, arm, metric):
        return np.mean([samples[f"{order}/{seed}/{boundary}/{category}/{arm}"][metric] for seed in protocol["seeds"]], axis=0)

    def macro_point(order, arm, metric):
        return float(np.mean([category_point(order, 350, category, arm, metric) for category in categories]))

    def macro_samples(order, arm, metric):
        return np.mean([category_samples(order, 350, category, arm, metric) for category in categories], axis=0)

    final, contrasts, trajectories, forgetting, harms = {}, {}, {}, {}, {}
    signal_by_order = {}
    for order in protocol["orders"]:
        final[order] = {}
        for arm in ARMS:
            final[order][arm] = {"pixel_ap": macro_point(order, arm, "pixel_ap"),
                "pixel_ap_ci": percentile(macro_samples(order, arm, "ap_low"), macro_samples(order, arm, "ap_high")),
                "image_auroc": macro_point(order, arm, "image_auroc"),
                "image_auroc_ci": percentile(macro_samples(order, arm, "image_auroc")),
                "categories": {category: {"pixel_ap": category_point(order, 350, category, arm, "pixel_ap"),
                    "pixel_ap_ci": percentile(category_samples(order, 350, category, arm, "ap_low"),
                                               category_samples(order, 350, category, arm, "ap_high")),
                    "image_auroc": category_point(order, 350, category, arm, "image_auroc"),
                    "prevalence": category_point(order, 350, category, arm, "prevalence")} for category in categories}}
            for boundary in protocol["boundaries"]:
                admitted = {row["category"] for row in protocol["updates_by_order"][order][:boundary]}
                for category in admitted:
                    key = f"{order}/{boundary}/{category}/{arm}"
                    trajectories[key] = {metric: category_point(order, boundary, category, arm, metric)
                                         for metric in ("pixel_ap", "image_auroc")}
                    acquired = 100 * (list(protocol["orders"][order]).index(category) + 1)
                    if boundary <= acquired:
                        continue
                    forgetting[key] = {"acquired_boundary": acquired, "is_bottle_reacquisition": category == "bottle" and boundary == 350,
                        "pixel_ap_loss": category_point(order, acquired, category, arm, "pixel_ap")
                                         - category_point(order, boundary, category, arm, "pixel_ap"),
                        "pixel_ap_loss_ci": percentile(category_samples(order, acquired, category, arm, "ap_low")
                            - category_samples(order, boundary, category, arm, "ap_high"),
                            category_samples(order, acquired, category, arm, "ap_high")
                            - category_samples(order, boundary, category, arm, "ap_low")),
                        "image_auroc_loss": category_point(order, acquired, category, arm, "image_auroc")
                                             - category_point(order, boundary, category, arm, "image_auroc")}
        control_points = {arm: final[order][arm]["pixel_ap"] for arm in ("A", "B", "C")}
        strongest = max(control_points, key=control_points.get)
        lower = macro_samples(order, "D", "ap_low") - np.max([macro_samples(order, arm, "ap_high") for arm in control_points], axis=0)
        upper = macro_samples(order, "D", "ap_high") - np.max([macro_samples(order, arm, "ap_low") for arm in control_points], axis=0)
        contrasts[order] = {"strongest_control": strongest, "D_minus_strongest": final[order]["D"]["pixel_ap"]-control_points[strongest],
            "conservative_paired_ci": percentile(lower, upper), "pairwise": {}}
        for control in ("A", "B", "C", "D_noinner"):
            contrasts[order]["pairwise"][control] = {"difference": final[order]["D"]["pixel_ap"]-final[order][control]["pixel_ap"],
                "ci": percentile(macro_samples(order, "D", "ap_low")-macro_samples(order, control, "ap_high"),
                                 macro_samples(order, "D", "ap_high")-macro_samples(order, control, "ap_low"))}
        signal_by_order[order] = {}
        for arm in ARMS:
            useful = sum(metric["image_auroc"] >= .70 and metric["pixel_ap"] >= max(3*metric["prevalence"], metric["prevalence"]+.05)
                         for metric in final[order][arm]["categories"].values())
            prevalence = np.mean([metric["prevalence"] for metric in final[order][arm]["categories"].values()])
            signal_by_order[order][arm] = bool(useful >= 2 and final[order][arm]["image_auroc_ci"][0] > .5
                                              and final[order][arm]["pixel_ap_ci"][0] > prevalence)
        for boundary in (200, 300, 350):
            for category in categories:
                acquired = 100 * (list(protocol["orders"][order]).index(category) + 1)
                if boundary <= acquired:
                    continue
                difference = category_point(order, boundary, category, "D", "pixel_ap") - max(
                    category_point(order, boundary, category, arm, "pixel_ap") for arm in ("A", "B", "C"))
                lower = category_samples(order, boundary, category, "D", "ap_low") - np.max([
                    category_samples(order, boundary, category, arm, "ap_high") for arm in ("A", "B", "C")], axis=0)
                upper = category_samples(order, boundary, category, "D", "ap_high") - np.max([
                    category_samples(order, boundary, category, arm, "ap_low") for arm in ("A", "B", "C")], axis=0)
                harms[f"{order}/{boundary}/{category}"] = {"difference": difference, "ci": percentile(lower, upper),
                    "no_material_harm_supported": bool(difference >= -.02 and np.quantile(lower, .025) > -.02)}
    primary = contrasts["canonical"]
    supported = primary["D_minus_strongest"] >= .02 and primary["conservative_paired_ci"][0] > 0
    supported = supported and signal_by_order["canonical"]["D"] and all(value["no_material_harm_supported"]
                for key, value in harms.items() if key.startswith("canonical/"))
    supported = supported and contrasts["alternate"]["D_minus_strongest"] >= 0
    any_signal = any(signal_by_order["canonical"].values())
    if supported:
        verdict = "P2_NL_SUPPORTED_DEVELOPMENT"
    elif primary["D_minus_strongest"] <= 0 or primary["conservative_paired_ci"][1] < .02 or not any_signal:
        verdict = "P2_NL_NOT_SUPPORTED"
    elif any_signal:
        verdict = "P2_NEURAL_SIGNAL_ONLY"
    else:
        verdict = "P2_INCONCLUSIVE"
    widths = [max(np.array(value["ap_high"])-np.array(value["ap_low"])) for value in samples.values()]
    historical = json.loads((P1 / "summary.json").read_text())
    summary = {"verdict": verdict, "exposure": protocol["exposure"], "final": final, "contrasts": contrasts,
        "signal": signal_by_order, "trajectories": trajectories, "forgetting": forgetting, "old_category_harm": harms,
        "historical_p1_external_macro": {arm: historical["macro"][arm] for arm in ("FULL784_GREEDY", "FULL784_RESERVOIR")},
        "both_order_mean_pixel_ap": {arm: float(np.mean([final[order][arm]["pixel_ap"] for order in protocol["orders"]])) for arm in ARMS},
        "max_bootstrap_ap_bound_width": float(max(widths)), "metric_groups": len(points),
        "analysis_seconds": time.perf_counter()-started, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}
    write_json(OUT / "summary.json", summary)
    plot(summary, protocol)
    write_json(OUT / "analysis_completed.json", {"utc": now(), "verdict": verdict})
    print(verdict, json.dumps(contrasts), flush=True)


def plot(summary, protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plotting
    figure, axes = plotting.subplots(1, 3, figsize=(13, 3.6))
    for axis, category in zip(axes, ("bottle", "carpet", "hazelnut")):
        for arm in ARMS:
            points = [(boundary, summary["trajectories"].get(f"canonical/{boundary}/{category}/{arm}"))
                      for boundary in protocol["boundaries"]]
            points = [(boundary, value) for boundary, value in points if value is not None]
            axis.plot([boundary for boundary, _ in points], [value["pixel_ap"] for _, value in points], marker="o", label=arm)
        axis.set(title=category, xlabel="normal arrivals", ylabel="native Pixel AP", ylim=(0, 1))
        axis.grid(alpha=.25)
    axes[-1].legend()
    figure.suptitle("P2 DEVELOPMENT EVIDENCE: canonical order, mean of three fixed seeds")
    figure.tight_layout()
    figure.savefig(OUT / "acquisition_forgetting.png", dpi=160)
    figure.savefig(OUT / "acquisition_forgetting.pdf")
    plotting.close(figure)


if __name__ == "__main__":
    analyze()
