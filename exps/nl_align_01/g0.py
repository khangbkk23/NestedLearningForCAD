"""NL-ALIGN-01 G0: read-only objective and implementation feasibility audit.

This module reads only the locked PTAG artifacts, normal-host synthetic artifacts,
and the existing R15/Dinomaly source and bank manifests.  It never opens an
anomaly mask, reads a real defect label, changes a checkpoint, or trains a model.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PTAG = ROOT / "results/hope_cad/ptag"
NLCAD = ROOT / "results/hope_cad/nlcad"
OUT = ROOT / "results/hope_cad/nl_align_01"

SOURCES = [
    "exps/ptag/common.py", "exps/ptag/synth.py", "exps/ptag/metrics.py",
    "exps/ptag/analysis.py", "exps/nlcad/objectives_s.py", "exps/nlcad/fast.py",
    "exps/nlcad/controller.py", "exps/nlcad/bank.py", "exps/nlcad/run.py",
    "exps/rh01b/run.py", "exps/dinomaly/official.py", "exps/dinomaly/encoder.py",
]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _stats(x):
    x = np.asarray(x, dtype=np.float64)
    return {"min": float(np.min(x)), "median": float(np.median(x)),
            "mean": float(np.mean(x)), "max": float(np.max(x))}


def _unpack_masks(packed, hw):
    bits = np.unpackbits(np.asarray(packed), axis=1)
    n = int(hw[0]) * int(hw[1])
    return bits[:, :n].reshape(len(bits), int(hw[0]), int(hw[1])).astype(bool)


def audit_ptag():
    man = read(PTAG / "MANIFEST.json")
    align = read(PTAG / "ALIGNMENT.json")
    cells = man["cells"]
    q2 = align["proxies"]["Q2"]
    q3 = align["proxies"]["Q3"]
    assert len(cells) == 635, len(cells)
    assert q2["confirmed"] and q3["confirmed"]
    # These are the registered values, checked against the archived analysis.
    assert abs(q2["full"]["mean_tau"] - 0.521) < 0.02
    assert abs(q3["full"]["mean_tau"] - 0.642) < 0.02
    assert q2["full"]["qualifying"] >= 10 and q3["full"]["qualifying"] >= 10
    assert q2["full"]["agreement"] >= 0.8 and q3["full"]["agreement"] >= 0.8
    fam = {}
    for row in cells:
        fam[row["family"]] = fam.get(row["family"], 0) + 1
    # PTAG's host manifest is explicitly restricted to train/good images.
    for rec in man["hosts"].values():
        assert all("/train/good/" in p for p in rec["hosts"])
    return {
        "cells": len(cells), "families": fam,
        "verdict": align["verdict"],
        "q2": {"tau": q2["full"]["mean_tau"], "tau_ci": q2["full"]["tau_ci95"],
               "agreement": q2["full"]["agreement"], "heldout": q2["heldout"]},
        "q3": {"tau": q3["full"]["mean_tau"], "tau_ci": q3["full"]["tau_ci95"],
               "agreement": q3["full"]["agreement"], "heldout": q3["heldout"]},
        "limitations": [
            "frozen-detector ranking, not adaptive optimization",
            "real AP is the consumed P1 development cohort",
            "within-family small-headroom alignment is weaker than pooled P2 alignment",
        ],
    }


def audit_generators():
    gen_report = read(PTAG / "GENERATE.json")
    out = {}
    for cat in ("bottle", "carpet", "hazelnut"):
        path = PTAG / "synthetic" / f"{cat}.pt"
        # mmap keeps the roughly 0.8 GB feature tensor virtual; only masks and
        # metadata are touched by this audit.
        d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        assert tuple(d["features"].shape[1:]) == (784, 768)
        assert len(d["host_of_image"]) == 320
        masks = _unpack_masks(d["masks"], d["hw"])
        assert masks.shape[0] == 300
        fams = {"Q1": masks[:100], "Q2": masks[100:200], "Q3": masks[200:300]}
        area = {k: _stats(v.reshape(len(v), -1).mean(1)) for k, v in fams.items()}
        q2, q3 = fams["Q2"], fams["Q3"]
        paired_equal = np.mean(np.all(q2 == q3, axis=(1, 2)))
        # A boundary-artifact diagnostic: fraction of positive pixels touching
        # the image boundary. It is descriptive and never used for selection.
        boundary = {}
        for k, v in fams.items():
            edge = np.zeros(v.shape, dtype=bool)
            edge[:, 0, :] = edge[:, -1, :] = True
            edge[:, :, 0] = edge[:, :, -1] = True
            boundary[k] = float(np.mean(v & edge) / (np.mean(v) + 1e-12))
        meta = d["meta"]
        q3_sizes = [i["size"][0] * i["size"][1] for i in meta["Q3"]]
        out[cat] = {
            "file_sha256": gen_report[cat]["file_sha256"],
            "feature_shape": list(d["features"].shape),
            "host_count": len(set(d["host_of_image"])),
            "mask_area": area,
            "q2_q3_paired_mask_equality": float(paired_equal),
            "boundary_positive_fraction_ratio": boundary,
            "q2_attempts_max": int(gen_report[cat]["q2_attempts_max"]),
            "q3_pixel_area": _stats(q3_sizes),
            "declared_normal_scale": float(gen_report[cat]["normal_scale_m"]),
            "host_paths_are_normal_only": True,
        }
        del d, masks
    return {"families": out,
            "source_definitions": {
                "Q2": "repo Perlin mask plus DTD texture, generated from train/good hosts",
                "Q3": "CutPaste-style within-image patch transform from train/good hosts",
                "Q1": "feature-space Gaussian blobs; excluded from G1",
            },
            "interpretation": [
                "Q2 and Q3 are structurally distinct generator families, not merely seed variants",
                "mask-area and boundary statistics differ by category and must be reported",
                "synthetic generator artifacts remain a Goodhart risk",
            ]}


def audit_r15_interface():
    bank_manifest = NLCAD / "BANK_seed2711.json"
    assert bank_manifest.exists()
    b = read(bank_manifest)
    assert b["generators"] == ["Q2", "Q3"]
    assert b["variants"] == 2
    files, missing = {}, []
    for name, rec in b["files"].items():
        path = ROOT / "results/hope_cad/nlcad_bank/seed2711" / f"{name}.pt"
        if not path.exists():
            missing.append(name)
            continue
        assert sha(path) == rec["sha256"], name
        files[name] = {"sha256": rec["sha256"], "n": rec["n"]}
    # The R15 bank is DINOv2/Dinomaly input, while PTAG is ViT-B/8. They cannot
    # be silently substituted for each other.
    shape = None
    if not missing:
        sample = next(ROOT.glob("results/hope_cad/nlcad_bank/seed2711/Q2_*.pt"))
        d = torch.load(sample, map_location="cpu", weights_only=False, mmap=True)
        shape = list(d["features"].shape)
        assert shape[1:] == [2, 789, 768], shape
        del d
    return {
        "learner": "Dinomaly-B + Replay-24 R15",
        "ptag_feature_shape": [784, 768],
        "r15_synthetic_feature_shape": None if shape is None else shape[1:],
        "seed2711_bank_files": files,
        "seed2711_bank_missing": missing,
        "bank_build_required": bool(missing),
        "seed2712_2713_bank_present": [
            (NLCAD / "BANK_seed2712.json").exists(),
            (NLCAD / "BANK_seed2713.json").exists(),
        ],
        "mechanical_path": "existing objective_s/fast interface is causal and finite; G1 must use a new fixed-objective runner",
        "selection": "R15 is required for the decisive test; P2 is an implementation smoke only",
    }


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "task": "NL-ALIGN-01 G0",
        "sources": {p: sha(ROOT / p) for p in SOURCES},
        "ptag": audit_ptag(),
        "generators": audit_generators(),
        "r15_interface": audit_r15_interface(),
        "decision": "G0_PASS_CONDITIONAL_G1",
        "decision_scope": "objective feasibility and mechanical meaning only; no anomaly efficacy claim",
        "g1_prerequisite": [
            "build and hash seed2712 and seed2713 DINOv2 synthetic banks",
            "lock generator roles, disjoint hosts, fixed synthetic gradient weight and controls",
            "seal all predictions before any consumed MVTec mask read",
        ],
    }
    target = OUT / "G0.json"
    if target.exists():
        raise RuntimeError(f"exists: {target}")
    target.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    run()
