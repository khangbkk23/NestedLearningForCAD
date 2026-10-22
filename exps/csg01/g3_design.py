"""G3 DESIGN ONLY — default-deny fresh real-anomaly evaluation for CSG-01.

Not executed on any real cohort.  It reuses the verified CSG-00 default-deny
guard, authorization schema and native-mask metric helpers, and scores the four
frozen compact CSG-01 arms (B/C/D/E).  Predictions are sealed before masks are
unlocked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import torch

from exps.csg00 import g3_design as G3

from . import contract as C1
from . import scoring


def score_images(features: torch.Tensor, state_path: str | Path, arm: str) -> np.ndarray:
    header, state = C1.read_state(state_path)
    if header["arm"] != arm:
        raise ValueError(f"state arm {header['arm']} != {arm}")
    support = scoring.support_bins_from_state(state, arm)
    queries = scoring.S0.per_bin_queries(features, C1.C0.bin_positions())
    per_bin = scoring.score_bins(queries, support, device="cpu")
    return scoring.S0.assemble(per_bin, features.shape[0], C1.C0.bin_positions()).numpy()


def predict_cohort(manifest_path: str | Path, authorization_path: str | Path,
                   states: Dict[str, str | Path], out_dir: str | Path,
                   feature_fn: Callable[[Sequence[str]], torch.Tensor] | None = None,
                   device: str = "cpu") -> Dict[str, Any]:
    auth = G3.load_authorization(authorization_path)
    manifest = G3.load_cohort(manifest_path)
    G3.require_matching_authorization(manifest, auth)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seal_path = out_dir / "PREDICTION_SEAL.json"
    if seal_path.exists():
        raise G3.G3Refused("prediction seal already exists (one-shot)")
    root = Path(manifest["root"])
    rows = list(manifest["images"])
    paths = [str(root / row["relative_path"]) for row in rows]
    feature_fn = feature_fn or (lambda p: G3.default_feature_fn(p, device=device))
    features = feature_fn(paths)
    if tuple(features.shape)[1:] != (C1.PATCHES, C1.DIM):
        raise ValueError(f"unexpected feature shape {tuple(features.shape)}")
    predictions = {}
    state_hashes = {}
    for arm, state_path in states.items():
        predictions[arm] = score_images(features, state_path, arm)
        state_hashes[arm] = C1.C0.sha256_file(state_path)
    prediction_path = out_dir / "predictions.npz"
    if prediction_path.exists():
        raise G3.G3Refused("prediction file already exists (one-shot)")
    np.savez(prediction_path,
             **{arm: predictions[arm] for arm in predictions},
             relative_path=np.asarray([row["relative_path"] for row in rows]),
             category=np.asarray([row["category"] for row in rows]))
    seal = {
        "schema": "csg01_g3_prediction_seal_v1",
        "sealed_utc": C1.C0.utc_now(),
        "cohort_manifest_sha256": manifest["_manifest_sha256"],
        "authorization_sha256": C1.C0.sha256_file(authorization_path),
        "prediction_sha256": C1.C0.sha256_file(prediction_path),
        "prediction_bytes": prediction_path.stat().st_size,
        "states": state_hashes,
        "images": len(rows),
        "mask_bytes_opened": False,
    }
    with open(seal_path, "x", encoding="utf-8") as fh:
        json.dump(seal, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return {"seal": str(seal_path),
            "sha256": C1.C0.sha256_file(seal_path),
            "prediction_sha256": seal["prediction_sha256"],
            "images": len(rows)}


def score_cohort(manifest_path: str | Path, authorization_path: str | Path,
                 prediction_seal_path: str | Path, out_path: str | Path,
                 unlock_masks: bool, mask_loader: Callable[[Path], np.ndarray] | None = None,
                 bootstrap_replicates: int = 0) -> Dict[str, Any]:
    if not unlock_masks:
        raise G3.G3Refused("native mask scoring requires --unlock-masks")
    auth = G3.load_authorization(authorization_path)
    manifest = G3.load_cohort(manifest_path)
    G3.require_matching_authorization(manifest, auth)
    seal_path = Path(prediction_seal_path)
    if not seal_path.is_file():
        raise G3.G3Refused("mask access refused without a prediction seal")
    seal = json.loads(seal_path.read_text())
    if seal.get("schema") != "csg01_g3_prediction_seal_v1":
        raise G3.G3Refused("invalid prediction seal schema")
    if seal["cohort_manifest_sha256"] != manifest["_manifest_sha256"]:
        raise G3.G3Refused("seal does not cover this cohort manifest")
    prediction_path = seal_path.parent / "predictions.npz"
    if not prediction_path.is_file() or C1.C0.sha256_file(prediction_path) != seal["prediction_sha256"]:
        raise G3.G3Refused("prediction file missing or digest mismatch")
    root = Path(manifest["root"])
    rows = list(manifest["images"])
    predictions = np.load(prediction_path)
    if list(predictions["relative_path"]) != [row["relative_path"] for row in rows]:
        raise G3.G3Refused("prediction row order mismatch")
    from PIL import Image
    mask_loader = mask_loader or G3._load_native_mask
    cat_type = {row["name"]: row["type"] for row in auth["target_categories"]}
    records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for arm in predictions.files:
        if arm in {"relative_path", "category"}:
            continue
        records[arm] = {}
    for i, row in enumerate(rows):
        with Image.open(root / row["relative_path"]) as source:
            width, height = source.size
        label = int(row.get("label", 0))
        if label:
            if not row.get("mask_relative_path"):
                raise G3.G3Refused(f"anomalous image {row['relative_path']} has no mask")
            mask = mask_loader(root / row["mask_relative_path"]).astype(bool)
            if mask.shape != (height, width):
                raise G3.G3Refused("mask shape mismatch")
        else:
            mask = np.zeros((height, width), dtype=bool)
        for arm in records:
            patch = predictions[arm][i].reshape(C1.GRID, C1.GRID)
            native = G3._native_scores(patch, height, width)
            records[arm].setdefault(row["category"], []).append({
                "scores": native.reshape(-1), "mask": mask.reshape(-1),
                "label": label, "image_max": float(native.max())})
    per_arm = {}
    for arm, cats in records.items():
        per_arm[arm] = {}
        for cat, items in cats.items():
            scores = np.concatenate([it["scores"] for it in items])
            masks = np.concatenate([it["mask"] for it in items])
            prevalence = float(masks.mean()) if masks.size else 0.0
            ap = G3.average_precision(scores, masks)
            per_arm[arm][cat] = {
                "type": cat_type.get(cat),
                "native_pixel_ap": ap,
                "prevalence": prevalence,
                "pixel_ap_minus_prevalence": None if ap is None else ap - prevalence,
                "image_auroc": G3.image_auroc(
                    np.asarray([it["image_max"] for it in items]),
                    np.asarray([it["label"] for it in items])),
                "n_images": len(items),
            }
    macro = {arm: {
        "native_pixel_ap": float(np.mean([v["native_pixel_ap"] for v in per_arm[arm].values()
                                          if v["native_pixel_ap"] is not None]))
        if any(v["native_pixel_ap"] is not None for v in per_arm[arm].values()) else None,
        "image_auroc": float(np.mean([v["image_auroc"] for v in per_arm[arm].values()
                                      if v["image_auroc"] is not None]))
        if any(v["image_auroc"] is not None for v in per_arm[arm].values()) else None,
    } for arm in per_arm}
    bootstrap = {"replicates": int(bootstrap_replicates)}
    if bootstrap_replicates > 0:
        rng = np.random.default_rng(C1.BOOTSTRAP_SEED)
        for arm, cats in records.items():
            samples = []
            for _ in range(bootstrap_replicates):
                ap_values = []
                for cat, items in cats.items():
                    idx = rng.integers(0, len(items), len(items))
                    chosen = [items[j] for j in idx]
                    ap = G3.average_precision(
                        np.concatenate([it["scores"] for it in chosen]),
                        np.concatenate([it["mask"] for it in chosen]))
                    if ap is not None:
                        ap_values.append(ap)
                samples.append(float(np.mean(ap_values)) if ap_values else np.nan)
            finite = np.asarray(samples)[np.isfinite(samples)]
            bootstrap[arm] = {
                "macro_pixel_ap_ci": ([float(np.percentile(finite, C1.BOOTSTRAP_CI[0])),
                                       float(np.percentile(finite, C1.BOOTSTRAP_CI[1]))]
                                      if finite.size else [None, None]),
                "n_valid": int(finite.size),
                "n_total": int(bootstrap_replicates),
            }
    result = {
        "schema": "csg01_g3_result_v1",
        "cohort_manifest_sha256": manifest["_manifest_sha256"],
        "prediction_seal_sha256": C1.C0.sha256_file(seal_path),
        "per_arm_category": per_arm,
        "macro": macro,
        "mask_access": "unlocked only after verified prediction seal",
        "bootstrap": bootstrap,
        "note": "G3 design only; not executed on real anomalies",
    }
    out_path = Path(out_path)
    if out_path.exists():
        raise G3.G3Refused("G3 result already exists (one-shot)")
    with open(out_path, "x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSG-01 G3 design only")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--cohort", required=True)
    p.add_argument("--authorization", required=True)
    p.add_argument("--states", required=True, help="JSON mapping arm -> state path")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p = sub.add_parser("score")
    p.add_argument("--cohort", required=True)
    p.add_argument("--authorization", required=True)
    p.add_argument("--prediction-seal", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--unlock-masks", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "predict":
            states = {k: v for k, v in json.loads(args.states).items()}
            print(json.dumps(predict_cohort(args.cohort, args.authorization, states,
                                            args.out, device=args.device), indent=2))
            return 0
        if args.command == "score":
            print(json.dumps(score_cohort(args.cohort, args.authorization,
                                          args.prediction_seal, args.out,
                                          unlock_masks=args.unlock_masks), indent=2))
            return 0
    except G3.G3Refused as exc:
        print(f"G3_REFUSED: {exc}", file=sys.stderr)
        return 2
    raise ValueError(args.command)


if __name__ == "__main__":
    sys.exit(main())
