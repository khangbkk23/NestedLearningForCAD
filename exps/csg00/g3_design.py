"""G3 DESIGN ONLY — default-deny fresh real-anomaly evaluation contract.

This module is **not executed** on any real anomaly cohort in CSG-00.  It is
implemented and unit-tested only with synthetic fixtures.  The default CLI
refuses to run without an independently authorised fresh-cohort manifest, and
refuses any path under historically exposed anomaly roots.

Required future authorisation (separate task): a fresh, legally usable anomaly
cohort with at least one object and one texture category, a locked one-shot
protocol, and a prediction-before-mask seal.  MVTec official-test data, VisA,
the consumed P1/NPIM DEV cohort and the sealed confirmation identities are
forbidden by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

from . import contract as C
from . import geometry_diagnostics as D
from . import scoring

FORBIDDEN_ROOTS = (
    C.ROOT / "data" / "mvtec",
    C.ROOT / "data" / "visa",
    C.ROOT / "data" / "dtd",
    C.ROOT / "results" / "hope_cad" / "npim_01",
    C.ROOT / "results" / "hope_cad" / "task_p" / "p1",
)
FORBIDDEN_TOKENS = (
    "ground_truth", "/test/", "sealed", "consumed_dev", "confirmation",
)


class G3Refused(PermissionError):
    """Raised when the G3 contract's default-deny conditions are not met."""


def assert_fresh_cohort_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    for root in FORBIDDEN_ROOTS:
        if resolved.is_relative_to(root.resolve()):
            raise G3Refused(f"G3 refuses historically exposed anomaly root: {resolved}")
    lowered = str(resolved).lower()
    for token in FORBIDDEN_TOKENS:
        if token in lowered:
            raise G3Refused(f"G3 refuses forbidden path token {token!r}: {resolved}")
    return resolved


def load_authorization(path: str | Path) -> Dict[str, Any]:
    if path is None:
        raise G3Refused("G3 requires an explicit fresh-cohort authorisation file")
    path = Path(path)
    if not path.is_file():
        raise G3Refused(f"G3 authorisation missing: {path}")
    auth = json.loads(path.read_text())
    required = ("schema", "authorized", "cohort_manifest_sha256", "fresh",
                "one_shot", "target_categories", "mask_access")
    missing = [k for k in required if k not in auth]
    if missing:
        raise G3Refused(f"G3 authorisation missing fields: {missing}")
    if auth["schema"] != "csg00_g3_authorization_v1" or not auth.get("authorized"):
        raise G3Refused("G3 authorisation is not an authorised csg00_g3_authorization_v1")
    if not auth.get("fresh") or not auth.get("one_shot"):
        raise G3Refused("G3 requires fresh=true and one_shot=true")
    if auth.get("mask_access") != "after_prediction_seal_only":
        raise G3Refused("G3 requires mask_access=after_prediction_seal_only")
    categories = auth.get("target_categories", [])
    kinds = {row.get("type") for row in categories}
    if not ({"object", "texture"} <= kinds):
        raise G3Refused("G3 requires at least one object and one texture category")
    return auth


def load_cohort(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise G3Refused(f"G3 cohort manifest missing: {path}")
    manifest = json.loads(path.read_text())
    root = Path(manifest["root"]).resolve()
    assert_fresh_cohort_path(root)
    for row in manifest.get("images", []):
        image_path = root / row["relative_path"]
        assert_fresh_cohort_path(image_path)
    manifest["_manifest_path"] = str(path)
    manifest["_manifest_sha256"] = C.sha256_file(path)
    return manifest


def require_matching_authorization(manifest: Dict[str, Any], auth: Dict[str, Any]) -> None:
    if manifest["_manifest_sha256"] != auth["cohort_manifest_sha256"]:
        raise G3Refused("cohort manifest digest does not match the authorisation")
    declared = {(row["name"], row["type"]) for row in auth["target_categories"]}
    cat_type = {name: kind for name, kind in declared}
    present = {(row["category"], cat_type.get(row["category"])) for row in manifest["images"]}
    if not present <= declared:
        raise G3Refused(f"cohort categories not covered by authorisation: {present - declared}")


# ---------------------------------------------------------------------------
# Feature / state adapters
# ---------------------------------------------------------------------------


def default_feature_fn(image_paths: Sequence[str], device: str = "cpu") -> torch.Tensor:
    """Pinned ViT-B/8 block-8 patch features for arbitrary cohort images."""
    from PIL import Image
    from models.feature_extractors.cadic_vit_v1 import CADICViTConfig, CADICViTFeatureExtractor

    config = CADICViTConfig(
        checkpoint_path=str(C.CHECKPOINT),
        checkpoint_identity=f"sha256:{C.CHECKPOINT_SHA256}",
        layer_number=9,
        layer_indexing="one_based",
    )
    extractor = CADICViTFeatureExtractor(config, device=device)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
    batches = []
    for start in range(0, len(image_paths), 8):
        images = []
        for path in image_paths[start:start + 8]:
            with Image.open(path) as source:
                image = source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                array = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1)) / 255.0
            images.append((array - mean) / std)
        with torch.no_grad():
            batches.append(extractor.extract_patch_features(
                torch.stack(images).to(device)).detach().cpu().float())
    extractor.close()
    return torch.cat(batches, dim=0)


def score_images(features: torch.Tensor, state_path: str | Path, arm: str) -> np.ndarray:
    """Score [n,784,768] with a frozen G2 state and return [n,784] scores."""
    header, state = C.read_state(state_path)
    if header["arm"] != arm:
        raise ValueError(f"state arm {header['arm']} != requested {arm}")
    positions = C.bin_positions()
    queries = scoring.per_bin_queries(features, positions)
    if arm == "B":
        per_bin = scoring.score_modes(queries, state, device="cpu")
    elif arm == "C":
        from . import reservoir

        per_bin = scoring.score_reservoir(queries, state, device="cpu")
    else:
        raise ValueError("G3 scores only the frozen compact arms B and C")
    return scoring.assemble(per_bin, features.shape[0], positions).numpy()


# ---------------------------------------------------------------------------
# Prediction seal
# ---------------------------------------------------------------------------


def predict_cohort(manifest_path: str | Path, authorization_path: str | Path,
                   state_b: str | Path, state_c: str | Path, out_dir: str | Path,
                   feature_fn: Callable[[Sequence[str]], torch.Tensor] | None = None,
                   device: str = "cpu") -> Dict[str, Any]:
    auth = load_authorization(authorization_path)
    manifest = load_cohort(manifest_path)
    require_matching_authorization(manifest, auth)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seal_path = out_dir / "PREDICTION_SEAL.json"
    if seal_path.exists():
        raise G3Refused("G3 prediction seal already exists (one-shot; refusing to rescore)")
    root = Path(manifest["root"])
    rows = list(manifest["images"])
    image_paths = [str(root / row["relative_path"]) for row in rows]
    feature_fn = feature_fn or (lambda paths: default_feature_fn(paths, device=device))
    features = feature_fn(image_paths)
    if tuple(features.shape)[1:] != (C.PATCHES, C.DIM):
        raise ValueError(f"unexpected feature shape {tuple(features.shape)}")
    scores_b = score_images(features, state_b, "B")
    scores_c = score_images(features, state_c, "C")
    prediction_path = out_dir / "predictions.npz"
    if prediction_path.exists():
        raise G3Refused("G3 prediction file already exists (one-shot)")
    np.savez(prediction_path,
             B=scores_b, C=scores_c,
             relative_path=np.asarray([row["relative_path"] for row in rows]),
             category=np.asarray([row["category"] for row in rows]))
    seal = {
        "schema": "csg00_g3_prediction_seal_v1",
        "sealed_utc": C.utc_now(),
        "cohort_manifest_sha256": manifest["_manifest_sha256"],
        "authorization_sha256": C.sha256_file(authorization_path),
        "prediction_sha256": C.sha256_file(prediction_path),
        "prediction_bytes": prediction_path.stat().st_size,
        "state_B_sha256": C.sha256_file(state_b),
        "state_C_sha256": C.sha256_file(state_c),
        "images": len(rows),
        "mask_bytes_opened": False,
        "note": "predictions are sealed before any native mask is unlocked",
    }
    with open(seal_path, "x", encoding="utf-8") as fh:
        json.dump(seal, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return {"seal": str(seal_path), "seal_sha256": C.sha256_file(seal_path),
            "prediction_sha256": seal["prediction_sha256"], "images": len(rows)}


# ---------------------------------------------------------------------------
# Native mask scoring (only after the seal)
# ---------------------------------------------------------------------------


def _native_scores(patch_scores: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.tensor(patch_scores, dtype=torch.float64)[None, None, :, :]
    upsampled = torch.nn.functional.interpolate(
        tensor, size=(height, width), mode="bilinear", align_corners=False, antialias=False)
    return upsampled[0, 0].numpy()


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    d_recall = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(d_recall * precision))


def image_auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = D.rankdata_average(scores)
    auc = (ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _load_native_mask(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as source:
        return np.array(source.convert("L")) > 0


def score_cohort(manifest_path: str | Path, authorization_path: str | Path,
                 prediction_seal_path: str | Path, out_path: str | Path,
                 unlock_masks: bool,
                 mask_loader: Callable[[Path], np.ndarray] | None = None,
                 bootstrap_replicates: int = 200) -> Dict[str, Any]:
    if not unlock_masks:
        raise G3Refused("G3 native mask scoring requires the separate --unlock-masks authorisation")
    auth = load_authorization(authorization_path)
    manifest = load_cohort(manifest_path)
    require_matching_authorization(manifest, auth)
    seal_path = Path(prediction_seal_path)
    if not seal_path.is_file():
        raise G3Refused("G3 refuses mask access without an existing prediction seal")
    seal = json.loads(seal_path.read_text())
    if seal.get("schema") != "csg00_g3_prediction_seal_v1":
        raise G3Refused("invalid prediction seal schema")
    if seal["cohort_manifest_sha256"] != manifest["_manifest_sha256"]:
        raise G3Refused("prediction seal does not cover this cohort manifest")
    prediction_path = seal_path.parent / "predictions.npz"
    if not prediction_path.is_file() or C.sha256_file(prediction_path) != seal["prediction_sha256"]:
        raise G3Refused("prediction file missing or its digest does not match the seal")

    root = Path(manifest["root"])
    rows = list(manifest["images"])
    predictions = np.load(prediction_path)
    if list(predictions["relative_path"]) != [row["relative_path"] for row in rows]:
        raise G3Refused("prediction row order does not match the cohort manifest")

    from PIL import Image

    mask_loader = mask_loader or _load_native_mask
    cat_type = {row["name"]: row["type"] for row in auth["target_categories"]}
    # per image: (native score map, native mask, image label, image max score)
    records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {"B": {}, "C": {}}
    for i, row in enumerate(rows):
        with Image.open(root / row["relative_path"]) as source:
            width, height = source.size
        label = int(row.get("label", 0))
        if label:
            if not row.get("mask_relative_path"):
                raise G3Refused(f"anomalous image {row['relative_path']} has no mask declared")
            mask = mask_loader(root / row["mask_relative_path"]).astype(bool)
            if mask.shape != (height, width):
                raise G3Refused(f"mask shape {mask.shape} != native {(height, width)}")
        else:
            mask = np.zeros((height, width), dtype=bool)
        for arm in ("B", "C"):
            patch_scores = predictions[arm][i].reshape(C.GRID, C.GRID)
            native = _native_scores(patch_scores, height, width)
            records[arm].setdefault(row["category"], []).append({
                "scores": native.reshape(-1),
                "mask": mask.reshape(-1),
                "label": label,
                "image_max": float(native.max()),
            })

    per_category: Dict[str, Dict[str, Any]] = {}
    for arm in ("B", "C"):
        per_category[arm] = {}
        for cat, items in records[arm].items():
            scores = np.concatenate([it["scores"] for it in items])
            mask = np.concatenate([it["mask"] for it in items])
            prevalence = float(mask.mean()) if mask.size else 0.0
            ap = average_precision(scores, mask)
            per_category[arm][cat] = {
                "type": cat_type.get(cat),
                "pixel_ap_native": ap,
                "prevalence": prevalence,
                "pixel_ap_minus_prevalence": (None if ap is None else ap - prevalence),
                "image_auroc": image_auroc(
                    np.asarray([it["image_max"] for it in items]),
                    np.asarray([it["label"] for it in items])),
                "n_images": len(items),
                "n_anomalous_images": int(sum(it["label"] for it in items)),
            }

    def macro(arm: str, key: str) -> float | None:
        values = [v[key] for v in per_category[arm].values() if v[key] is not None]
        return float(np.mean(values)) if values else None

    bootstrap: Dict[str, Any] = {"replicates": int(bootstrap_replicates)}
    if bootstrap_replicates > 0:
        rng = np.random.default_rng(C.BOOTSTRAP_SEED)
        macro_samples: List[float] = []
        for _ in range(bootstrap_replicates):
            ap_values = []
            for cat, items in records["B"].items():
                idx = rng.integers(0, len(items), len(items))
                chosen = [items[j] for j in idx]
                ap = average_precision(
                    np.concatenate([it["scores"] for it in chosen]),
                    np.concatenate([it["mask"] for it in chosen]))
                if ap is not None:
                    ap_values.append(ap)
            macro_samples.append(float(np.mean(ap_values)) if ap_values else np.nan)
        arr = np.asarray(macro_samples)
        finite = arr[np.isfinite(arr)]
        bootstrap["arm_B_macro_pixel_ap"] = {
            "ci": ([float(np.percentile(finite, C.BOOTSTRAP_CI[0])),
                    float(np.percentile(finite, C.BOOTSTRAP_CI[1]))] if finite.size else [None, None]),
            "n_valid": int(finite.size),
            "n_total": int(arr.size),
            "note": "whole-image bootstrap within category; replicate count must be locked "
                    "by the future G3 task before execution",
        }

    summary = {
        "schema": "csg00_g3_result_v1",
        "cohort_manifest_sha256": manifest["_manifest_sha256"],
        "prediction_seal_sha256": C.sha256_file(seal_path),
        "per_category": per_category,
        "macro": {
            arm: {
                "native_pixel_ap": macro(arm, "pixel_ap_native"),
                "native_pixel_ap_minus_prevalence": macro(arm, "pixel_ap_minus_prevalence"),
                "image_auroc": macro(arm, "image_auroc"),
            } for arm in ("B", "C")
        },
        "negative_gap_safeguards": {
            arm: {
                kind: {
                    cat: per_category[arm][cat]["pixel_ap_minus_prevalence"]
                    for cat in per_category[arm]
                    if per_category[arm][cat]["type"] == kind
                } for kind in ("object", "texture")
            } for arm in ("B", "C")
        },
        "bootstrap": bootstrap,
        "mask_access": "unlocked only after verified prediction seal",
        "note": "G3 is design-only in CSG-00 and was not executed on real anomalies",
    }
    out_path = Path(out_path)
    if out_path.exists():
        raise G3Refused("G3 result already exists (one-shot; refusing to overwrite)")
    with open(out_path, "x", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return summary


# ---------------------------------------------------------------------------
# CLI (default deny)
# ---------------------------------------------------------------------------


def cmd_predict(args: argparse.Namespace) -> int:
    info = predict_cohort(args.cohort, args.authorization, args.state_b, args.state_c,
                          args.out, device=args.device)
    print(json.dumps(info, indent=2))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    summary = score_cohort(args.cohort, args.authorization, args.prediction_seal,
                           args.out, unlock_masks=args.unlock_masks)
    print(json.dumps({"decision": "G3_EXECUTED_ONE_SHOT",
                      "macro": summary["macro"]}, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CSG-00 G3 DESIGN ONLY — default-deny fresh anomaly evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--cohort", required=True)
    p_pred.add_argument("--authorization", required=True)
    p_pred.add_argument("--state-b", required=True)
    p_pred.add_argument("--state-c", required=True)
    p_pred.add_argument("--out", required=True)
    p_pred.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p_score = sub.add_parser("score")
    p_score.add_argument("--cohort", required=True)
    p_score.add_argument("--authorization", required=True)
    p_score.add_argument("--prediction-seal", required=True)
    p_score.add_argument("--out", required=True)
    p_score.add_argument("--unlock-masks", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "predict":
            return cmd_predict(args)
        if args.command == "score":
            return cmd_score(args)
    except G3Refused as exc:
        print(f"G3_REFUSED: {exc}", file=sys.stderr)
        return 2
    raise ValueError(args.command)


if __name__ == "__main__":
    sys.exit(main())
