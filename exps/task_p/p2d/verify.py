"""Independent read-only verification of the frozen P2d artifacts.

Checks (never writes inside run directories):
- protocol/code hashes and inherited P1/P0c/P2 preservation
- state-chain continuity of every locked training trace, including the initial state
- checkpoint digests, budget byte accounting, evaluation budget and read-only scoring
- replay legality against arrived sources
- sklearn cross-checks of the exact native AP and image AUROC for every final boundary group
- sealed-identity exclusion
"""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from exps.task_p.p1_data import native_mask, sha, write_json
from exps.task_p.p1_readout import native_map
from exps.task_p.p2d.data import ARMS, OUT, ROOT, p2_reference_pins, require, tensor_bytes, tree_digest
from exps.task_p.p2d.experiment import load_protocol
from exps.task_p.p2d.model import Learner2
from exps.task_p.p2d.preflight import configure


def verify():
    configure()
    protocol = load_protocol()
    require(p2_reference_pins()["passed"], "P2 references changed")
    categories = ("bottle", "carpet", "hazelnut")
    points = json.loads((OUT / "point_metrics.json").read_text())
    result = {"checks": {}, "max_sklearn_ap_difference": 0.0, "max_sklearn_auroc_difference": 0.0,
              "max_cpu_gpu_difference": 0.0, "runs": {}}
    for order in protocol["orders"]:
        for seed in protocol["seeds"]:
            directory = OUT / f"{order}_seed{seed}"
            rows = protocol["updates_by_order"][order]
            expected_sources = [row["relative_path"] for row in rows]
            normal_access = json.loads((directory / "normal_access.json").read_text())
            require([entry["source"] for entry in normal_access] == expected_sources,
                    f"normal access order differs: {order}/{seed}")
            replay_access = json.loads((directory / "replay_access.json").read_text())
            arrived = set()
            for entry in replay_access:
                source = entry["historical_source"]
                require(source is None or source in arrived, f"replay saw unarrived source: {order}/{seed}")
                arrived.add(expected_sources[entry["event"] - 1])
            trace = json.loads((directory / "trace.json").read_text())
            counts, chains_ok, controllers_ok = {}, {}, {}
            for arm in ARMS:
                subset = [record for record in trace if record["arm"] == arm]
                require(len(subset) == 700, f"cycle count differs: {order}/{seed}/{arm}")
                counts[arm] = len(subset)
                initial = Learner2(arm, seed)
                require(tree_digest(initial.state()) == subset[0]["before"],
                        f"initial state digest differs: {order}/{seed}/{arm}")
                chains_ok[arm] = all(first["after"] == second["before"]
                                     for first, second in zip(subset, subset[1:]))
                require(chains_ok[arm], f"state chain break: {order}/{seed}/{arm}")
                finite = all(np.isfinite(list(record[field] for field in
                               ("support_loss", "query_loss", "outer_loss", "outer_gradient_norm",
                                "controller_gradient_norm", "fast_parameter_displacement")))
                             for record in subset)
                require(finite, f"nonfinite trace record: {order}/{seed}/{arm}")
                if arm == "D2":
                    controllers_ok[arm] = all(record["controller_gradient_norm"] > 0 for record in subset)
                else:
                    controllers_ok[arm] = all(record["controller_gradient_norm"] == 0.0 for record in subset)
                require(controllers_ok[arm], f"controller gradient contract violated: {order}/{seed}/{arm}")
            checkpoint_digests = {}
            for boundary in protocol["boundaries"]:
                checkpoint = torch.load(directory / f"state_t{boundary}.pt", map_location="cpu",
                                        weights_only=True)
                for arm in ARMS:
                    digest = tree_digest(checkpoint["states"][arm])
                    require(digest == checkpoint["state_sha256"][arm],
                            f"checkpoint digest differs: {order}/{seed}/{boundary}/{arm}")
                    checkpoint_digests[boundary, arm] = digest
                budgets = json.loads((directory / "budgets.json").read_text())[str(boundary)]
                replay_bytes = tensor_bytes(checkpoint["replay"])
                for arm in ARMS:
                    require(budgets[arm]["replay_tensor_bytes"] == replay_bytes, "replay byte accounting")
                    require(budgets[arm]["learner_tensor_bytes"] == tensor_bytes(checkpoint["states"][arm]),
                            "learner byte accounting")
                last_cycle = [record for record in trace if record["event"] == boundary
                              and record["cycle"] == protocol["optimization"]["cycles_per_image"] - 1]
                for record in last_cycle:
                    require(record["after"] == checkpoint_digests[boundary, record["arm"]],
                            "trace/checkpoint digest link broken")
            predictions = json.loads((directory / "predictions.json").read_text())
            expected = sum(20 * len({row["category"] for row in rows[:boundary]})
                           for boundary in protocol["boundaries"])
            require(len(predictions) == 2 * expected, f"evaluation budget differs: {order}/{seed}")
            sources = {row["relative_path"]: row for row in protocol["evaluations"]}
            for record in predictions:
                row = sources[record["source"]]
                require(record["source"] not in protocol["sealed_identity_paths"], "sealed identity scored")
                admitted = {item["category"] for item in rows[:record["boundary"]]}
                require(row["category"] in admitted, "future category scored")
                require(len(record["patch_scores"]) == 784, "patch count differs")
                require(float(max(record["patch_scores"])) == record["image_score"], "image score readout")
                require(record["state_sha256"] == checkpoint_digests[record["boundary"], record["arm"]],
                        "prediction state link broken")
            inference = json.loads((directory / "inference_resources.json").read_text())
            result["max_cpu_gpu_difference"] = max(result["max_cpu_gpu_difference"],
                                                   inference["cpu_gpu_max_patch_score_difference"])
            require(inference["cpu_gpu_max_patch_score_difference"] < 5e-4, "CPU/GPU parity exceeded")
            masks = {name: native_mask(protocol["dataset_root"], row) for name, row in sources.items()}
            for boundary in (350,):
                for category in categories:
                    group = [record for record in predictions if record["boundary"] == boundary
                             and record["arm"] in ARMS and sources[record["source"]]["category"] == category]
                    for arm in ARMS:
                        records = [record for record in group if record["arm"] == arm]
                        scores = np.concatenate([np.array(record["patch_scores"]) for record in records])
                        maps = np.concatenate([native_map(torch.tensor(record["patch_scores"]),
                                                          sources[record["source"]]["hw"])
                                               for record in records])
                        targets = np.concatenate([masks[record["source"]].flatten() for record in records])
                        ap = average_precision_score(targets, maps)
                        labels = [sources[record["source"]]["label"] for record in records]
                        image_scores = [record["image_score"] for record in records]
                        auroc = roc_auc_score(labels, image_scores)
                        key = f"{order}/{seed}/{boundary}/{category}/{arm}"
                        result["max_sklearn_ap_difference"] = max(result["max_sklearn_ap_difference"],
                                                                  abs(ap - points[key]["pixel_ap"]))
                        result["max_sklearn_auroc_difference"] = max(result["max_sklearn_auroc_difference"],
                                                                     abs(auroc - points[key]["image_auroc"]))
            result["runs"][f"{order}/{seed}"] = {"cycles": counts, "chains_ok": chains_ok,
                                                 "controller_contract": controllers_ok}
            print("VERIFIED", order, seed, flush=True)
    require(result["max_sklearn_ap_difference"] < 1e-6, "sklearn AP mismatch")
    require(result["max_sklearn_auroc_difference"] < 1e-9, "sklearn AUROC mismatch")
    probe = json.loads((OUT / "probe.json").read_text())
    require(probe["probe_trace_sha256"] == sha(OUT / "probe_trace.json"), "probe trace changed")
    inherited = json.loads((OUT / "inheritance.json").read_text())
    preserved_failures = [name for name, expected in inherited["preserved"].items()
                          if sha(ROOT / name) != expected]
    require(not preserved_failures, f"preserved P0c/P1/frozen files changed: {preserved_failures[:5]}")
    result["preserved_files_checked"] = len(inherited["preserved"])
    result["p1_protocol_sha256"] = json.loads((OUT / "inheritance.json").read_text())["p1_protocol_sha256"]
    result["passed"] = True
    write_json(OUT / "verification.json", result)
    print(json.dumps({"passed": True, "runs": len(result["runs"]),
                      "preserved": result["preserved_files_checked"],
                      "max_sklearn_ap_difference": result["max_sklearn_ap_difference"],
                      "max_sklearn_auroc_difference": result["max_sklearn_auroc_difference"],
                      "max_cpu_gpu_difference": result["max_cpu_gpu_difference"]}, indent=2), flush=True)


if __name__ == "__main__":
    verify()
