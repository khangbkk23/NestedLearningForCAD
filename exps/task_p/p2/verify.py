"""Separate read-only sklearn, chronology, state-chain and preservation audit."""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.functional import interpolate

from exps.task_p.p1_data import sha, write_json
from exps.task_p.p2.data import ARMS, OUT, ROOT, tree_digest


def verify():
    torch.set_num_threads(4)
    started = time.perf_counter()
    protocol = json.loads((OUT / "protocol_resolved.json").read_text())
    digest = sha(OUT / "protocol_resolved.json")
    assert digest == (OUT / "protocol.sha256").read_text().strip()
    assert all(sha(ROOT / name) == expected for name, expected in protocol["code_sha256"].items())
    assert (OUT / "analysis_completed.json").is_file()
    points = json.loads((OUT / "point_metrics.json").read_text())
    source = {row["relative_path"]: row for row in protocol["evaluations"]}
    sealed = set(protocol["sealed_identity_paths"])
    maximum_image_error, numerical, state_checks, updates = 0.0, {}, 0, 0
    resources = {}
    for order in protocol["orders"]:
        for seed in protocol["seeds"]:
            directory = OUT / f"{order}_seed{seed}"
            rows = protocol["updates_by_order"][order]
            access = json.loads((directory / "normal_access.json").read_text())
            assert [row["source"] for row in access] == [row["relative_path"] for row in rows]
            replay = json.loads((directory / "replay_access.json").read_text())
            assert len(replay) == 700
            for record in replay:
                assert record["historical_source"] is None or record["historical_source"] in {
                    row["relative_path"] for row in rows[:record["event"]-1]}
            trace = json.loads((directory / "trace.json").read_text())
            assert len(trace) == 3500
            chain, boundary_chain = {}, {}
            for record in trace:
                arm = record["arm"]
                if arm in chain:
                    assert chain[arm] == record["before"]
                chain[arm] = record["after"]
                assert np.isfinite(record["query_loss"]) and record["support_gradient_norm"] >= 0
                if arm == "D":
                    assert record["controller_gradient_norm"] > 0
                    assert .001/np.e - 1e-9 <= record["rate_min"] <= record["rate_max"] <= .001*np.e + 1e-9
                    assert .02/np.e - 1e-9 <= record["mix_min"] <= record["mix_max"] <= .02*np.e + 1e-9
                if record["event"] in protocol["boundaries"] and record["cycle"] == 1:
                    boundary_chain.setdefault(record["event"], {})[arm] = record["after"]
            for boundary in protocol["boundaries"]:
                checkpoint = torch.load(directory / f"state_t{boundary}.pt", weights_only=True, map_location="cpu", mmap=True)
                assert checkpoint["protocol_sha256"] == digest and checkpoint["boundary"] == boundary
                assert len(checkpoint["replay"]["images"]) == 24 and checkpoint["replay"]["seen"] == boundary
                for arm in ARMS:
                    assert checkpoint["states"][arm]["steps"] == 2*boundary
                    assert tree_digest(checkpoint["states"][arm]) == boundary_chain[boundary][arm]
                    state_checks += 1
            predictions = json.loads((directory / "predictions.json").read_text())
            assert len(predictions) == 900
            for record in predictions:
                assert record["source"] in source and record["source"] not in sealed
                category = source[record["source"]]["category"]
                assert category in {row["category"] for row in rows[:record["boundary"]]}
                assert record["state_sha256"] == boundary_chain[record["boundary"]][record["arm"]]
            for boundary in protocol["boundaries"]:
                for category in ("bottle", "carpet", "hazelnut"):
                    for arm in ARMS:
                        group = [record for record in predictions if record["boundary"] == boundary and record["arm"] == arm
                                 and source[record["source"]]["category"] == category]
                        if not group:
                            continue
                        key = f"{order}/{seed}/{boundary}/{category}/{arm}"
                        labels = [source[record["source"]]["label"] for record in group]
                        scores = [record["image_score"] for record in group]
                        direct_image = roc_auc_score(labels, scores)
                        maximum_image_error = max(maximum_image_error, abs(direct_image-points[key]["image_auroc"]))
                        assert abs(direct_image-points[key]["image_auroc"]) < 1e-12
                        if order != "canonical" or seed != protocol["seeds"][0] or boundary != 350:
                            continue
                        pixels, targets = [], []
                        for record in group:
                            row = source[record["source"]]
                            grid = torch.tensor(record["patch_scores"], dtype=torch.float32).reshape(1, 1, 28, 28)
                            pixels.append(interpolate(grid, size=tuple(row["hw"]), mode="bilinear",
                                                      align_corners=False, antialias=False).numpy().ravel())
                            if row["label"]:
                                path = ROOT / "data/mvtec" / row["mask_path"]
                                assert sha(path) == row["mask_sha256"]
                                with Image.open(path) as image:
                                    mask = np.asarray(image) > 0
                            else:
                                mask = np.zeros(tuple(row["hw"]), dtype=bool)
                            targets.append(mask.ravel())
                        pixels, targets = np.concatenate(pixels), np.concatenate(targets)
                        direct = {"pixel_ap": average_precision_score(targets, pixels),
                                  "pixel_auroc": roc_auc_score(targets, pixels)}
                        delta = {name: abs(value-points[key][name]) for name, value in direct.items()}
                        assert max(delta.values()) < 1e-12, (key, delta)
                        numerical[key] = delta
                        print("SKLEARN_NATIVE", key, delta, flush=True)
            resources[f"{order}/{seed}"] = json.loads((directory / "resources.json").read_text())
            updates += len(trace)
    inherited = json.loads((OUT / "inheritance.json").read_text())
    preservation = {name: sha(ROOT / name) == expected for name, expected in inherited["preserved"].items()}
    assert all(preservation.values()), [name for name, passed in preservation.items() if not passed]
    result = {"passed": True, "agent": "Codex", "independence": "separate numerical implementation; secondary-agent audit still pending",
        "native_sklearn_groups": numerical, "image_auc_groups": len(points), "maximum_image_auc_error": maximum_image_error,
        "state_checks": state_checks, "continuous_updates": updates, "preservation": preservation,
        "sealed_payloads_accessed": 0, "seconds": time.perf_counter()-started,
        "verifier_sha256": sha(__file__), "resources": resources}
    write_json(OUT / "numerical_verification.json", result)
    print("VERIFIED", len(numerical), "native groups;", len(points), "image groups;", state_checks, "states", flush=True)


if __name__ == "__main__":
    verify()
