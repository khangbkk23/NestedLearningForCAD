# exps/cms_nm01/g3.py
"""One-shot, locked development anomaly evaluation for CMS-NM-01.

This module is intentionally label-facing and is only invoked after the
forensic normal-only gates pass. It never updates a checkpoint or chooses a
score/checkpoint from anomaly outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .cms_nd import CMSNormalDensityMemory
from .run import CATEGORIES, FEATURE_ROOT, PROBE_RANGES, RESULT_ROOT, make_model

DEV_MANIFEST = Path("results/hope_cad/anomaly_signal_gate/manifests/anomaly_dev_manifest.parquet")
CHECKPOINT_ROOT = RESULT_ROOT / "checkpoints"
OUT = RESULT_ROOT / "forensic"
TOP_K = math.ceil(0.01 * 784)  # locked 1% rule: eight patches


def load_development_payloads() -> dict[str, dict]:
    import pandas as pd

    manifest = pd.read_parquet(DEV_MANIFEST)
    if set(manifest["split"].unique()) != {"DEVELOPMENT"} or len(manifest) != 60:
        raise ValueError("unexpected development manifest")
    payloads = {}
    for category in CATEGORIES:
        payload = torch.load(FEATURE_ROOT / f"{category}_dev.pt", map_location="cpu", weights_only=False)
        rows = manifest[manifest.category == category]
        expected = {str(row.relative_path): (int(row.label), str(row.defect_type)) for row in rows.itertuples()}
        if set(payload["paths"]) != set(expected) or payload["patches"].shape != (20, 784, 768):
            raise ValueError(f"development identity/shape mismatch for {category}")
        labels = torch.tensor([expected[str(path)][0] for path in payload["paths"]], dtype=torch.int64)
        if not torch.equal(labels, payload["labels"].to(torch.int64)):
            raise ValueError(f"development labels do not match manifest for {category}")
        if int(labels.sum()) != 10:
            raise ValueError(f"expected ten defects in {category}")
        payloads[category] = {"patches": payload["patches"].float(), "masks": payload["masks"].bool(),
                              "labels": labels, "paths": list(payload["paths"]),
                              "defect_types": [expected[str(path)][1] for path in payload["paths"]]}
    return payloads


@torch.no_grad()
def score_payload(model, payload: dict) -> dict:
    started = time.perf_counter()
    result = model.score_patches(payload["patches"])
    elapsed = time.perf_counter() - started
    patch = result["mixture_nll"].float().cpu()
    if patch.ndim != 2 or patch.shape != (20, 784):
        raise ValueError(f"unexpected score shape {tuple(patch.shape)}")
    image = torch.topk(patch, TOP_K, dim=1).values.mean(1).numpy().astype(np.float64)
    patch_map = patch.reshape(20, 1, 28, 28)
    native = torch.nn.functional.interpolate(patch_map, size=tuple(payload["masks"].shape[-2:]), mode="bilinear", align_corners=False).squeeze(1).numpy()
    labels = payload["labels"].numpy().astype(np.int64)
    masks = payload["masks"].numpy().astype(np.int64)
    return {
        "image_auroc": float(roc_auc_score(labels, image)),
        "pixel_aupr_native": float(average_precision_score(masks.reshape(-1), native.reshape(-1))),
        "image_scores": image.tolist(),
        "pixel_score_mean": float(native.mean()),
        "normal_score_q50": float(np.quantile(image[labels == 0], .50)),
        "normal_score_q95": float(np.quantile(image[labels == 0], .95)),
        "normal_score_q99": float(np.quantile(image[labels == 0], .99)),
        "defect_score_mean": float(image[labels == 1].mean()),
        "inference_seconds_per_image": float(elapsed / len(image)),
        "top_k": TOP_K,
        "score_definition": "mean of top eight patch mixture NLLs; 28x28 bilinear to 224x224",
    }


def load_model(arm: str, seed: int, boundary: int):
    model = make_model(arm, seed=7101)
    path = CHECKPOINT_ROOT / f"seed{seed}_{arm}" / f"boundary{boundary}.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    model.eval()
    return model, path


def forgetting(matrix: list[list[float]], order: list[str]) -> dict:
    # Rows are boundaries 1..3; only rows after a category was acquired count.
    result = {}
    for index, category in enumerate(order):
        observed = [matrix[row][CATEGORIES.index(category)] for row in range(index, len(matrix))]
        result[category] = float(max(observed) - observed[-1])
    return {"per_category": result, "mean": float(np.mean(list(result.values())))}


def main() -> None:
    gates = json.loads((OUT / "verified_gates.json").read_text())
    if not gates["g1"]["pass"] or not gates["g2"]["pass"]:
        raise RuntimeError("G3 is locked until corrected original G1 and G2 pass")
    payloads = load_development_payloads()
    records = []
    normal_records = json.loads((RESULT_ROOT / "normal_only_records.json").read_text())
    for seed in (0, 1, 2):
        order = list(__import__("exps.cms_nm01.run", fromlist=["order_for_seed"]).order_for_seed(seed))
        for arm in ("CMS-ND-3", "CMS-ND-1", "CMS-ND-FROZEN", "GAUSS-64"):
            boundaries = []
            for boundary in range(4):
                model, checkpoint = load_model(arm, seed, boundary)
                per_category = {category: score_payload(model, payloads[category]) for category in CATEGORIES}
                boundaries.append({"boundary": boundary, "checkpoint": str(checkpoint), "per_category": per_category})
            image_matrix = [[entry["per_category"][category]["image_auroc"] for category in CATEGORIES] for entry in boundaries[1:]]
            pixel_matrix = [[entry["per_category"][category]["pixel_aupr_native"] for category in CATEGORIES] for entry in boundaries[1:]]
            normal = next(row for row in normal_records if row["arm"] == arm and int(row["order_seed"]) == seed)
            final = boundaries[-1]["per_category"]
            records.append({"seed": seed, "arm": arm, "order": order, "boundaries": boundaries,
                            "final_per_category": final,
                            "forgetting_image_auroc": forgetting(image_matrix, order),
                            "forgetting_pixel_aupr_native": forgetting(pixel_matrix, order),
                            "persistent_bytes": int(normal["final_metrics"]["persistent_bytes"]),
                            "update_seconds_per_image_mean": normal["update_seconds_per_image_mean"],
                            "inference_seconds_per_image_batched": normal["final_metrics"]["batched_inference_seconds_per_image"],
                            "transient_memory": normal["resource_audit"]["extra_tensor_memory"],
                            "checkpoint_sha256": {str(entry["boundary"]): hashlib.sha256(Path(entry["checkpoint"]).read_bytes()).hexdigest() for entry in boundaries}})
    summary = {}
    for arm in ("CMS-ND-3", "CMS-ND-1", "CMS-ND-FROZEN", "GAUSS-64"):
        rows = [row for row in records if row["arm"] == arm]
        final_rows = [row["final_per_category"][category] for row in rows for category in CATEGORIES]
        summary[arm] = {
            "image_auroc_mean": float(np.mean([row["image_auroc"] for row in final_rows])),
            "pixel_aupr_native_mean": float(np.mean([row["pixel_aupr_native"] for row in final_rows])),
            "image_forgetting_mean": float(np.mean([row["forgetting_image_auroc"]["mean"] for row in rows])),
            "pixel_forgetting_mean": float(np.mean([row["forgetting_pixel_aupr_native"]["mean"] for row in rows])),
            "persistent_bytes": rows[0]["persistent_bytes"],
            "update_seconds_per_image_mean": float(np.mean([row["update_seconds_per_image_mean"] for row in rows])),
            "inference_seconds_per_image_batched_mean": float(np.mean([row["inference_seconds_per_image_batched"] for row in rows])),
        }
    result = {"protocol": "corrected CMS-NM-01 G3; no label tuning", "development_manifest": str(DEV_MANIFEST),
              "top_k": TOP_K, "categories": CATEGORIES, "records": records, "summary": summary,
              "archived_context": {"AD-02_S4": {"image_auroc": .9722, "pixel_aupr_native": .7245},
                                   "AD-02_S7": {"image_auroc": .9822, "pixel_aupr_native": .7184}},
              "sealed_confirmation_accessed": False}
    (OUT / "g3_records.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
