"""Independent post-run point-metric and causal audit, using sklearn directly.

This analysis-only module is separate from the frozen experimental implementation.
It reads saved patch distances and the locked development masks, never features or
learner inputs. It does not fit, select, update or rerun the experiment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.nn.functional import interpolate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/hope_cad/task_p/p1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def audit():
    p = json.loads((OUT / "protocol_resolved.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    assert digest(OUT / "protocol_resolved.json") == (OUT / "protocol.sha256").read_text().strip()
    for name, value in p["source_sha256"].items():
        assert digest(ROOT / name) == value, name
    assert (OUT / "run_completed.json").is_file()
    assert not (OUT / "execution_error.json").exists()
    access = json.loads((OUT / "access_log.json").read_text())
    updates = [r for r in access if r["role"] == "update"]
    assert [r["path"] for r in updates] == [r["relative_path"] for r in p["updates"]]
    assert [r["event"] for r in updates] == list(range(1, 351))
    allowed = {r["relative_path"]: r for r in p["evaluations"]}
    sealed = set(p["sealed_identity_paths"])
    admitted_at = {}
    for row in p["updates"]:
        admitted_at.setdefault(row["category"], row["event"])
    eval_access = [r for r in access if r["role"] == "evaluation"]
    assert len(eval_access) == 180
    for row in eval_access:
        assert row["path"] in allowed and row["path"] not in sealed
        assert row["event"] in p["boundaries"]
        assert row["event"] >= admitted_at[allowed[row["path"]]["category"]]
    trace = json.loads((OUT / "update_trace.json").read_text())
    assert len(trace) == 1050
    chains = {}
    for row in trace:
        arm = row["arm"]
        if arm in chains:
            assert chains[arm] == row["before"]
        chains[arm] = row["after"]
        assert row["source"] == p["updates"][row["event"] - 1]["relative_path"]
        assert row["ledger"]["charged_persistent_bytes"] <= 8 * 1024**2
    deltas, counts = {}, {}
    for boundary in p["boundaries"]:
        rows = json.loads((OUT / f"predictions_t{boundary}.json").read_text())
        for category in p["categories"]:
            for arm in p["arms"]:
                group = [r for r in rows if r["arm"] == arm and r["source"]["category"] == category]
                if not group:
                    continue
                scores, labels, pixels, targets = [], [], [], []
                for record in group:
                    row = record["source"]
                    assert row["relative_path"] in allowed and row["relative_path"] not in sealed
                    h, w = row["hw"]
                    d = np.array(record["patch_distances"], dtype=np.float64)
                    assert d.shape == (784,) and np.isfinite(d).all()
                    assert d.max() == record["image_score"]
                    image_map = interpolate(torch.from_numpy(d).float().reshape(1, 1, 28, 28),
                                            size=(h, w), mode="bilinear", align_corners=False,
                                            antialias=False)[0, 0].numpy()
                    if row["label"]:
                        file = Path(p["dataset_root"]) / row["mask_path"]
                        assert digest(file) == row["mask_sha256"]
                        with Image.open(file) as image:
                            mask = np.asarray(image) > 0
                    else:
                        mask = np.zeros((h, w), dtype=bool)
                    assert mask.shape == (h, w) and int(mask.sum()) == record["positive_pixels"]
                    pixels.append(image_map.ravel())
                    targets.append(mask.ravel())
                    scores.append(record["image_score"])
                    labels.append(row["label"])
                y, s = np.concatenate(targets), np.concatenate(pixels)
                direct = {"image_auroc": float(roc_auc_score(labels, scores)),
                          "pixel_ap": float(average_precision_score(y, s)),
                          "pixel_auroc": float(roc_auc_score(y, s)), "prevalence": float(y.mean())}
                key = f"{boundary}/{category}/{arm}"
                reported = summary["per_boundary_category"][key]
                delta = {metric: abs(value - reported[metric]) for metric, value in direct.items()}
                assert max(delta.values()) < 1e-12, (key, delta)
                deltas[key], counts[key] = delta, int(y.size)
                print("independent sklearn check passed:", key, flush=True)
    for arm in p["arms"]:
        for name in ("image_auroc", "pixel_ap", "pixel_auroc", "prevalence"):
            mean = np.mean([summary["per_boundary_category"][f"350/{c}/{arm}"][name] for c in p["categories"]])
            assert abs(mean - summary["macro"][arm][name]) < 1e-14
    result = {"passed": True, "metric_deltas": deltas, "native_pixel_counts": counts,
              "updates": len(updates), "evaluation_accesses": len(eval_access),
              "continuous_arm_updates": len(trace), "sealed_payloads_in_access_log": 0,
              "independent_code_sha256": digest(__file__)}
    with (OUT / "independent_verification.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    torch.set_num_threads(4)
    audit()
