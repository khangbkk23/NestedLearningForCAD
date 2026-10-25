# exps/calm02/data.py
"""CALM-02 identities for MVTec AD (15 tasks) and VisA (12 tasks). Lists paths only; opens no mask."""
from __future__ import annotations

import csv
from pathlib import Path

from exps.task_p.p0_normal_support import require

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth"
CHECKPOINT_SHA = "755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415"
MEAN, STD = (.485, .456, .406), (.229, .224, .225)

MVTEC_ORDER = ("bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
               "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper")
VISA_ORDER = ("candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
              "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum")
BENCHMARKS = {"mvtec": (ROOT / "data/mvtec", MVTEC_ORDER), "visa": (ROOT / "data/visa", VISA_ORDER)}


def _mvtec(root, task):
    train = [{"relative_path": p.relative_to(root).as_posix()}
             for p in sorted((root / task / "train" / "good").glob("*.png"))]
    test = []
    for d in sorted(p for p in (root / task / "test").iterdir() if p.is_dir()):
        for p in sorted(d.glob("*.png")):
            label = int(d.name != "good")
            mask = f"{task}/ground_truth/{d.name}/{p.stem}_mask.png" if label else None
            test.append({"relative_path": p.relative_to(root).as_posix(), "label": label, "mask_path": mask})
    return train, test


def _visa(root, task, rows):
    mine = [r for r in rows if r["object"] == task]
    train = [{"relative_path": r["image"]} for r in sorted(mine, key=lambda r: r["image"])
             if r["split"] == "train"]
    require(all(r["label"] == "normal" for r in mine if r["split"] == "train"), "VisA train must be normal")
    test = [{"relative_path": r["image"], "label": int(r["label"] == "anomaly"),
             "mask_path": r["mask"] or None}
            for r in sorted(mine, key=lambda r: r["image"]) if r["split"] == "test"]
    return train, test


def identities(benchmark):
    root, order = BENCHMARKS[benchmark]
    rows = list(csv.DictReader(open(root / "split_csv/1cls.csv"))) if benchmark == "visa" else None
    tasks = []
    for task in order:
        train, test = _mvtec(root, task) if benchmark == "mvtec" else _visa(root, task, rows)
        require(train and test, f"empty split for {benchmark}/{task}")
        require(all(bool(r["mask_path"]) == bool(r["label"]) for r in test), f"mask/label mismatch {task}")
        require(not {r["relative_path"] for r in train} & {r["relative_path"] for r in test},
                f"train/test overlap {task}")
        tasks.append({"task": task, "train": train, "test": test})
    return {"benchmark": benchmark, "root": str(root), "tasks": tasks}


def p1_development_paths():
    """The 60 MVTec test images consumed by P1/CALM-00A (reported with and without)."""
    import json
    p = json.loads((ROOT / "results/hope_cad/task_p/p1/protocol_resolved.json").read_text())
    return sorted(r["relative_path"] for r in p["evaluations"])
