"""CALM-00A loading and identity utilities. Read-only over frozen P1 artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT / "results/hope_cad/task_p/p1"
DATA = ROOT / "data/mvtec"

CATEGORIES = ("bottle", "carpet", "hazelnut")
CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}
# Stream: events 0-99 bottle, 100-199 carpet, 200-299 hazelnut, 300-349 bottle return.
EVENT_OWNER = np.full(350, 0, dtype=np.int8)
EVENT_OWNER[100:200] = 1
EVENT_OWNER[200:300] = 2
# events >= 300 stay 0 (bottle), which is correct for the cross-category test.

ARMS = ("FULL784_GREEDY", "FULL784_RESERVOIR")
PRIMARY_ARM = "FULL784_GREEDY"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_protocol() -> dict:
    with open(P1 / "protocol_resolved.json") as fh:
        return json.load(fh)


def load_state(boundary: int, arm: str) -> dict:
    state = torch.load(P1 / f"states_t{int(boundary)}.pt", map_location="cpu", weights_only=False)
    rec = state[arm]
    n = int(rec["n"])
    bank = rec["bank"][:n].contiguous()
    ids = rec["ids"][:n].contiguous()
    if bank.shape != (n, 768) or bank.dtype != torch.float32:
        raise RuntimeError(f"bad bank {arm}@{boundary}: {tuple(bank.shape)} {bank.dtype}")
    if not bool(torch.isfinite(bank).all()):
        raise RuntimeError("nonfinite bank")
    owners = torch.from_numpy(EVENT_OWNER[(ids.numpy() // 784)])
    return {"bank": bank, "ids": ids, "owners": owners, "n": n, "arm": arm, "boundary": boundary}


def load_eval() -> list[dict]:
    """60 development evaluation images; features are boundary-independent."""
    out = []
    for side in sorted(P1.glob("evaluation_features/*.json")):
        meta = json.loads(side.read_text())
        src = meta["source"]
        rel = src["relative_path"]
        key = sha256_bytes(rel.encode())
        if side.stem != key:
            raise RuntimeError(f"feature filename does not hash from identity: {side.name}")
        feat = torch.load(side.with_suffix(".pt"), map_location="cpu", weights_only=False)
        if tuple(feat.shape) != (784, 768) or feat.dtype != torch.float32:
            raise RuntimeError(f"bad feature {rel}: {tuple(feat.shape)} {feat.dtype}")
        if sha256_bytes(feat.numpy().tobytes()) != meta["feature_sha256"]:
            raise RuntimeError(f"feature sha mismatch {rel}")
        out.append({
            "key": key, "rel": rel, "category": src["category"], "cat": CAT_INDEX[src["category"]],
            "label": int(src["label"]), "hw": tuple(src["hw"]), "mask_path": src["mask_path"],
            "mask_sha256": src["mask_sha256"], "feature": feat,
            "feature_sha256": meta["feature_sha256"],
            "file_sha256": meta["file_sha256"], "sidecar_sha256": sha256_file(side),
        })
    if len(out) != 60:
        raise RuntimeError(f"expected 60 evaluation features, got {len(out)}")
    out.sort(key=lambda r: r["rel"])
    for cat in CATEGORIES:
        recs = [r for r in out if r["category"] == cat]
        if len(recs) != 20 or sum(r["label"] for r in recs) != 10:
            raise RuntimeError(f"unexpected cohort composition for {cat}")
    return out


def load_predictions(boundary: int) -> dict[tuple[str, str], np.ndarray]:
    """Cell-11 verification oracle: stored P1 patch distances by (arm, identity)."""
    path = P1 / f"predictions_t{int(boundary)}.json"
    if not path.exists():
        return {}
    records = json.loads(path.read_text())
    out = {}
    for rec in records:
        arm = rec["arm"]
        if arm not in ARMS or int(rec["boundary"]) != int(boundary):
            continue
        out[(arm, rec["source"]["relative_path"])] = {
            "patch_distances": np.asarray(rec["patch_distances"], dtype=np.float64),
            "positive_pixels": int(rec["positive_pixels"]),
            "pixels": int(rec["pixels"]),
        }
    return out


def load_mask(rec: dict) -> np.ndarray:
    """Native binary mask >0. Normals are all-zero at native size, as in P1."""
    h, w = rec["hw"]
    if rec["label"] == 0:
        return np.zeros((h, w), dtype=bool)
    path = DATA / rec["mask_path"]
    raw = path.read_bytes()
    if sha256_bytes(raw) != rec["mask_sha256"]:
        raise RuntimeError(f"mask sha mismatch {rec['mask_path']}")
    from PIL import Image
    import io
    arr = np.array(Image.open(io.BytesIO(raw)))
    mask = arr > 0
    if mask.shape != (h, w):
        raise RuntimeError(f"mask shape {mask.shape} != native {(h, w)} for {rec['mask_path']}")
    if not mask.any():
        raise RuntimeError(f"empty defect mask {rec['mask_path']}")
    return mask
