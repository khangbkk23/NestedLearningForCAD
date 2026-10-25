# exps/dinomaly/common.py
"""Paths, identities and helpers for the Dinomaly reconstruction feasibility gates (DRF-01)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/mvtec"
OUT = ROOT / "results/hope_cad/dinomaly"
P1 = ROOT / "results/hope_cad/task_p/p1"
P1_SHA = "11c1cc03b20cb1f79a6e8b7e472bb6f44758b43046b61043368ea11b4ce1e08f"
PTAG_MANIFEST = ROOT / "results/hope_cad/ptag/MANIFEST.json"
CKPT = {"base": (ROOT / "checkpoints/dinov2/dinov2_vitb14_reg4_pretrain.pth",
                 "73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71")}
CATS = ("bottle", "carpet", "hazelnut")
ORDER = ("bottle", "carpet", "hazelnut")


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:                                  # create-exclusive: never overwrite
        json.dump(value, f, indent=1, sort_keys=True)


def p1_protocol():
    require(sha(P1 / "protocol_resolved.json") == P1_SHA, "P1 protocol changed")
    return read(P1 / "protocol_resolved.json")


def identities():
    """Training = exact P1 update identities; held-out normals = PTAG hosts; evaluation = P1 dev cohort."""
    p1 = p1_protocol()
    sealed = set(p1["sealed_identity_paths"])
    train = {c: [r for r in p1["updates"] if r["category"] == c] for c in CATS}
    hosts = read(PTAG_MANIFEST)["hosts"]
    heldout = {c: [{"relative_path": p, "sha256": s, "category": c}
                   for p, s in zip(hosts[c]["hosts"], hosts[c]["sha256"])] for c in CATS}
    evaluation = {c: [r for r in p1["evaluations"] if r["category"] == c] for c in CATS}
    train_paths = {r["relative_path"] for rows in train.values() for r in rows}
    held_paths = {r["relative_path"] for rows in heldout.values() for r in rows}
    eval_paths = {r["relative_path"] for rows in evaluation.values() for r in rows}
    require(not (train_paths & held_paths) and not (train_paths & eval_paths) and not (held_paths & eval_paths),
            "identity overlap")
    require(not ((train_paths | held_paths | eval_paths) & sealed), "sealed identity requested")
    require(all("/train/good/" in p for p in train_paths | held_paths), "normal identity outside train/good")
    require([len(train[c]) for c in CATS] == [150, 100, 100] and all(len(heldout[c]) == 20 for c in CATS)
            and all(len(evaluation[c]) == 20 for c in CATS), "identity counts changed")
    return {"train": train, "heldout": heldout, "evaluation": evaluation, "sealed_count": len(sealed)}


def source(row):
    path = DATA / row["relative_path"]
    require(sha(path) == row["sha256"], f"source changed: {row['relative_path']}")
    return path
