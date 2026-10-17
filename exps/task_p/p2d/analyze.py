"""P2d native development metrics, one-path recomputation of inherited references and
preregistered paired image-level inference.

All five arms (inherited B/C/D from their hash-pinned P2 predictions, new C2/D2) pass
through the identical metric/bound machinery with the identical paired bootstrap draws
(seed 102701+category index), so every contrast is exactly paired at the image level.
The recomputed B/C/D records are reconciled against the recorded P2 metrics before any
contrast is computed.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from exps.task_p.p1_data import native_mask, write_json
from exps.task_p.p1_readout import bootstrap_weights, native_map
from exps.task_p.p2.metrics import bootstrap_bounds, exact_ranking, image_auroc
from exps.task_p.p2d.data import ARMS as NEW_ARMS
from exps.task_p.p2d.data import OUT, P2, p2_reference_pins, require
from exps.task_p.p2d.experiment import load_protocol, now
from exps.task_p.p2d.preflight import configure

REFERENCE_ARMS = ("B", "C", "D")
ALL_ARMS = ("B", "C", "D") + NEW_ARMS


def percentile(lower, upper=None):
    return [float(np.quantile(lower, .025)),
            float(np.quantile(lower if upper is None else upper, .975))]


def analyze():
    configure()
    protocol = load_protocol()
    require((OUT / "evaluation_completed.json").exists(), "scoring incomplete")
    pins = p2_reference_pins()
    require(pins["passed"], "P2 references changed")
    write_json(OUT / "analysis_started.json", {"utc": now()})
    started = time.perf_counter()
    categories = ("bottle", "carpet", "hazelnut")
    sources = {row["relative_path"]: row for row in protocol["evaluations"]}
    masks = {name: native_mask(protocol["dataset_root"], row) for name, row in sources.items()}
    draws = {}
    for index, category in enumerate(categories):
        group = [row for row in protocol["evaluations"] if row["category"] == category]
        generator = np.random.default_rng(102701 + index)
        draws[category] = np.stack([bootstrap_weights([row["label"] for row in group], generator)
                                    for _ in range(1000)])
    points, samples, separation = {}, {}, {}
    torch.cuda.reset_peak_memory_stats()
    for order in protocol["orders"]:
        for seed in protocol["seeds"]:
            directory = OUT / f"{order}_seed{seed}"
            records = json.loads((directory / "predictions.json").read_text())
            reference_dir = P2 / f"{order}_seed{seed}"
            records += json.loads((reference_dir / "predictions.json").read_text())
            for boundary in protocol["boundaries"]:
                for category in categories:
                    rows = [record for record in records if record["boundary"] == boundary
                            and sources[record["source"]]["category"] == category]
                    if not rows:
                        continue
                    for arm in ALL_ARMS:
                        group = [record for record in rows if record["arm"] == arm]
                        require(len(group) == 20, "incomplete category evaluation")
                        image_maps, targets, labels, scores = [], [], [], []
                        for record in group:
                            source = sources[record["source"]]
                            distances = torch.tensor(record["patch_scores"], dtype=torch.float32)
                            require(float(distances.max()) == record["image_score"],
                                    "image max readout differs")
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
                        samples[key] = {"ap_low": lower.tolist(), "ap_high": upper.tolist(),
                                        "image_auroc": auc.tolist()}
                        normal_scores = np.array(scores)[np.array(labels) == 0]
                        anomaly_scores = np.array(scores)[np.array(labels) == 1]
                        positive_sum = sum(float(score[mask].double().sum())
                                           for score, mask in zip(image_maps, targets))
                        negative_sum = sum(float(score[~mask].double().sum())
                                           for score, mask in zip(image_maps, targets))
                        separation[key] = {"normal_image_mean": float(normal_scores.mean()),
                            "normal_image_median": float(np.median(normal_scores)),
                            "normal_image_p95": float(np.quantile(normal_scores, .95)),
                            "anomaly_image_mean": float(anomaly_scores.mean()),
                            "positive_pixel_mean": positive_sum / metric["positive_pixels"],
                            "negative_pixel_mean": negative_sum / (metric["native_pixels"] - metric["positive_pixels"])}
                        del image_maps, targets
            print("METRICS", order, seed, "seconds", round(time.perf_counter() - started, 1), flush=True)

    recorded_points = json.loads((P2 / "point_metrics.json").read_text())
    recorded_samples = json.loads((P2 / "bootstrap_samples.json").read_text())
    reconciliation = {"point_metric_max_abs_difference": 0.0, "sample_max_abs_difference": 0.0}
    for key, metric in recorded_points.items():
        order, seed, boundary, category, arm = key.split("/")
        if arm not in REFERENCE_ARMS:
            continue
        recomputed = points[key]
        for name in ("pixel_ap", "image_auroc", "prevalence"):
            difference = abs(recomputed[name] - metric[name])
            reconciliation["point_metric_max_abs_difference"] = max(
                reconciliation["point_metric_max_abs_difference"], difference)
        for name in ("ap_low", "ap_high", "image_auroc"):
            difference = float(np.max(np.abs(np.array(samples[key][name])
                                             - np.array(recorded_samples[key][name]))))
            reconciliation["sample_max_abs_difference"] = max(
                reconciliation["sample_max_abs_difference"], difference)
    require(reconciliation["point_metric_max_abs_difference"] < 1e-9,
            "recomputed P2 point metrics diverge")
    require(reconciliation["sample_max_abs_difference"] < 1e-9,
            "recomputed P2 bootstrap samples diverge")
    write_json(OUT / "point_metrics.json", points)
    write_json(OUT / "bootstrap_samples.json", samples)
    write_json(OUT / "score_separation.json", separation)
    write_json(OUT / "reference_reconciliation.json", reconciliation)

    def category_point(order, boundary, category, arm, metric):
        return float(np.mean([points[f"{order}/{seed}/{boundary}/{category}/{arm}"][metric]
                              for seed in protocol["seeds"]]))

    def category_samples(order, boundary, category, arm, metric):
        return np.mean([samples[f"{order}/{seed}/{boundary}/{category}/{arm}"][metric]
                        for seed in protocol["seeds"]], axis=0)

    def macro_point(order, arm, metric):
        return float(np.mean([category_point(order, 350, category, arm, metric)
                              for category in categories]))

    def macro_samples(order, arm, metric):
        return np.mean([category_samples(order, 350, category, arm, metric)
                        for category in categories], axis=0)

    final, contrasts, trajectories, forgetting, harms = {}, {}, {}, {}, {}
    signal_by_order = {}
    for order in protocol["orders"]:
        final[order] = {}
        for arm in ALL_ARMS:
            final[order][arm] = {"pixel_ap": macro_point(order, arm, "pixel_ap"),
                "pixel_ap_ci": percentile(macro_samples(order, arm, "ap_low"),
                                          macro_samples(order, arm, "ap_high")),
                "image_auroc": macro_point(order, arm, "image_auroc"),
                "image_auroc_ci": percentile(macro_samples(order, arm, "image_auroc")),
                "categories": {category: {"pixel_ap": category_point(order, 350, category, arm, "pixel_ap"),
                    "pixel_ap_ci": percentile(category_samples(order, 350, category, arm, "ap_low"),
                                              category_samples(order, 350, category, arm, "ap_high")),
                    "image_auroc": category_point(order, 350, category, arm, "image_auroc"),
                    "prevalence": category_point(order, 350, category, arm, "prevalence")}
                    for category in categories}}
            for boundary in protocol["boundaries"]:
                admitted = {row["category"] for row in protocol["updates_by_order"][order][:boundary]}
                for category in admitted:
                    key = f"{order}/{boundary}/{category}/{arm}"
                    trajectories[key] = {metric: category_point(order, boundary, category, arm, metric)
                                         for metric in ("pixel_ap", "image_auroc")}
                    acquired = 100 * (list(protocol["orders"][order]).index(category) + 1)
                    if boundary <= acquired:
                        continue
                    forgetting[key] = {"acquired_boundary": acquired,
                        "is_bottle_reacquisition": category == "bottle" and boundary == 350,
                        "pixel_ap_loss": category_point(order, acquired, category, arm, "pixel_ap")
                                         - category_point(order, boundary, category, arm, "pixel_ap"),
                        "pixel_ap_loss_ci": percentile(
                            category_samples(order, acquired, category, arm, "ap_low")
                            - category_samples(order, boundary, category, arm, "ap_high"),
                            category_samples(order, acquired, category, arm, "ap_high")
                            - category_samples(order, boundary, category, arm, "ap_low")),
                        "image_auroc_loss": category_point(order, acquired, category, arm, "image_auroc")
                                            - category_point(order, boundary, category, arm, "image_auroc")}

        def paired(first, second):
            return {"difference": final[order][first]["pixel_ap"] - final[order][second]["pixel_ap"],
                    "ci": percentile(macro_samples(order, first, "ap_low")
                                     - macro_samples(order, second, "ap_high"),
                                     macro_samples(order, first, "ap_high")
                                     - macro_samples(order, second, "ap_low"))}

        best = max(NEW_ARMS, key=lambda arm: final[order][arm]["pixel_ap"])
        contrasts[order] = {"best_new_arm": best, "paired": {
            "D2_minus_C2": paired("D2", "C2"), "best_minus_B": paired(best, "B"),
            "best_minus_original_D": paired(best, "D"),
            "C2_minus_B": paired("C2", "B"), "D2_minus_B": paired("D2", "B"),
            "C2_minus_original_D": paired("C2", "D"), "D2_minus_original_D": paired("D2", "D"),
            "C2_minus_original_C": paired("C2", "C"), "D2_minus_original_C": paired("D2", "C")}}
        signal_by_order[order] = {}
        for arm in ALL_ARMS:
            useful = sum(metric["image_auroc"] >= .70 and metric["pixel_ap"] >= max(
                3 * metric["prevalence"], metric["prevalence"] + .05)
                for metric in final[order][arm]["categories"].values())
            prevalence = np.mean([metric["prevalence"]
                                  for metric in final[order][arm]["categories"].values()])
            signal_by_order[order][arm] = bool(useful >= 2
                                               and final[order][arm]["image_auroc_ci"][0] > .5
                                               and final[order][arm]["pixel_ap_ci"][0] > prevalence)
        for arm in NEW_ARMS:
            controls = ("B",) if arm == "C2" else ("B", "C2")
            for boundary in (200, 300, 350):
                for category in categories:
                    acquired = 100 * (list(protocol["orders"][order]).index(category) + 1)
                    if boundary <= acquired:
                        continue
                    difference = category_point(order, boundary, category, arm, "pixel_ap") - max(
                        category_point(order, boundary, category, control, "pixel_ap")
                        for control in controls)
                    lower = category_samples(order, boundary, category, arm, "ap_low") - np.max([
                        category_samples(order, boundary, category, control, "ap_high")
                        for control in controls], axis=0)
                    upper = category_samples(order, boundary, category, arm, "ap_high") - np.max([
                        category_samples(order, boundary, category, control, "ap_low")
                        for control in controls], axis=0)
                    harms[f"{arm}/{order}/{boundary}/{category}"] = {
                        "difference": difference, "ci": percentile(lower, upper),
                        "no_material_harm_supported": bool(difference >= -.02
                                                           and np.quantile(lower, .025) > -.02)}

    structural = json.loads((OUT / "reference_reconciliation.json").read_text())
    canonical_best = final["canonical"][contrasts["canonical"]["best_new_arm"]]["pixel_ap"]
    alternate_best = final["alternate"][contrasts["alternate"]["best_new_arm"]]["pixel_ap"]
    h1_supported = bool(max(final["canonical"][arm]["pixel_ap"] for arm in NEW_ARMS) >= 0.45)
    nl_primary = contrasts["canonical"]["paired"]["D2_minus_C2"]
    nl_alternate = contrasts["alternate"]["paired"]["D2_minus_C2"]
    nl_supported = bool(nl_primary["difference"] >= .02 and nl_primary["ci"][0] > 0
                        and nl_alternate["difference"] >= 0)
    if h1_supported and nl_supported:
        verdict = "P2D_H1_SUPPORTED_NL_SUPPORTED"
    elif h1_supported:
        verdict = "P2D_H1_SUPPORTED_NL_NOT"
    else:
        verdict = "P2D_H1_FALSIFIED"

    storage = {}
    throughput = {}
    for arm in NEW_ARMS:
        budget_rows, update_seconds, inference_seconds = [], [], []
        for order in protocol["orders"]:
            for seed in protocol["seeds"]:
                directory = OUT / f"{order}_seed{seed}"
                budget_rows.append(json.loads((directory / "budgets.json").read_text())["350"][arm])
                resources = json.loads((directory / "resources.json").read_text())
                update_seconds.append(resources["update_seconds_by_arm"][arm] / resources["cycles_per_arm"][arm])
                inference = json.loads((directory / "inference_resources.json").read_text())
                inference_seconds.append(float(np.mean(inference["seconds_by_arm"][arm])))
        storage[arm] = {"learner_tensor_bytes": float(np.mean([row["learner_tensor_bytes"] for row in budget_rows])),
                        "replay_tensor_bytes": float(np.mean([row["replay_tensor_bytes"] for row in budget_rows])),
                        "inference_tensor_bytes": float(np.mean([row["inference_tensor_bytes"] for row in budget_rows])),
                        "online_tensor_bytes": float(np.mean([row["online_tensor_bytes"] for row in budget_rows]))}
        throughput[arm] = {"seconds_per_cycle": float(np.mean(update_seconds)),
                           "seconds_per_image_evaluation": float(np.mean(inference_seconds))}
    reference_throughput = {}
    for arm in REFERENCE_ARMS:
        update_seconds, inference_seconds = [], []
        for order in protocol["orders"]:
            for seed in protocol["seeds"]:
                resources = json.loads((P2 / f"{order}_seed{seed}/resources.json").read_text())
                update_seconds.append(resources["update_seconds_by_arm"][arm] / 700)
                inference = json.loads((P2 / f"{order}_seed{seed}/inference_resources.json").read_text())
                inference_seconds.append(float(np.mean(inference["seconds_by_arm"][arm])))
        reference_throughput[arm] = {"seconds_per_cycle": float(np.mean(update_seconds)),
                                     "seconds_per_image_evaluation": float(np.mean(inference_seconds))}

    widths = [max(np.array(value["ap_high"]) - np.array(value["ap_low"])) for value in samples.values()]
    historical = json.loads((P2 / "summary.json").read_text())
    summary = {"verdict": verdict, "exposure": protocol["exposure"],
        "decision": {"h1_supported": h1_supported,
                     "h1_threshold": 0.45,
                     "canonical_best_new_arm": contrasts["canonical"]["best_new_arm"],
                     "canonical_best_pixel_ap": canonical_best,
                     "alternate_best_pixel_ap": alternate_best,
                     "learned_nl_supported": nl_supported,
                     "d2_minus_c2_canonical": nl_primary,
                     "d2_minus_c2_alternate": nl_alternate},
        "final": final, "contrasts": contrasts, "signal": signal_by_order,
        "trajectories": trajectories, "forgetting": forgetting, "old_category_harm": harms,
        "reference_reconciliation": structural,
        "inherited_p2_verdict": historical["verdict"],
        "storage_bytes": storage, "throughput": throughput, "reference_throughput": reference_throughput,
        "both_order_mean_pixel_ap": {arm: float(np.mean([final[order][arm]["pixel_ap"]
                                                         for order in protocol["orders"]]))
                                     for arm in ALL_ARMS},
        "max_bootstrap_ap_bound_width": float(max(widths)), "metric_groups": len(points),
        "analysis_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}
    write_json(OUT / "summary.json", summary)
    plot(summary, protocol)
    write_json(OUT / "analysis_completed.json", {"utc": now(), "verdict": verdict})
    print(verdict, json.dumps({"final_canonical": {arm: final["canonical"][arm]["pixel_ap"]
                                                   for arm in ALL_ARMS},
                               "contrasts_canonical": contrasts["canonical"]["paired"],
                               "decision": summary["decision"]}, indent=2), flush=True)


def plot(summary, protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plotting
    figure, axes = plotting.subplots(1, 3, figsize=(13, 3.6))
    for axis, category in zip(axes, ("bottle", "carpet", "hazelnut")):
        for arm in ("B", "D", "C2", "D2"):
            points = [(boundary, summary["trajectories"].get(f"canonical/{boundary}/{category}/{arm}"))
                      for boundary in protocol["boundaries"]]
            points = [(boundary, value) for boundary, value in points if value is not None]
            axis.plot([boundary for boundary, _ in points],
                      [value["pixel_ap"] for _, value in points], marker="o", label=arm)
        axis.set(title=category, xlabel="normal arrivals", ylabel="native Pixel AP", ylim=(0, 1))
        axis.grid(alpha=.25)
    axes[-1].legend()
    figure.suptitle("P2d DEVELOPMENT EVIDENCE: canonical order, mean of three fixed seeds")
    figure.tight_layout()
    figure.savefig(OUT / "acquisition_forgetting.png", dpi=160)
    figure.savefig(OUT / "acquisition_forgetting.pdf")
    plotting.close(figure)


if __name__ == "__main__":
    analyze()
