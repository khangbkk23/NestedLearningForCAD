"""CALM-00A runner: lock -> label-free cell scoring -> single label/mask pass -> analysis."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import torch

from exps.calm import certify, decompose, load, stats

SLUG = "calm00a"
CONSTANTS = {
    "schema": "calm00a_v1",
    "chunk": 64,
    "bootstrap_reps": 1000,
    "bootstrap_seed": 770011,
    "bootstrap_bins": 16384,
    "radius_quantile": 0.95,
    "cells": list(decompose.CELLS),
    "arms": list(load.ARMS),
    "primary_arm": load.PRIMARY_ARM,
    "groups": [
        {"name": "bottle@200", "cat": "bottle", "acq": 100, "T": 200, "role": "first_transition"},
        {"name": "bottle@300", "cat": "bottle", "acq": 100, "T": 300, "role": "primary"},
        {"name": "carpet@300", "cat": "carpet", "acq": 200, "T": 300, "role": "primary"},
        {"name": "hazelnut@350", "cat": "hazelnut", "acq": 300, "T": 350, "role": "primary"},
        {"name": "bottle@350", "cat": "bottle", "acq": 100, "T": 350, "role": "descriptive_reacquisition"},
        {"name": "carpet@350", "cat": "carpet", "acq": 200, "T": 350, "role": "descriptive"},
    ],
    "bands": {
        "ap_points": 1.0,
        "vir_min": 2.0,
        "vir_drop": 1.0,
        "affected_defect_frac_min": 0.05,
    },
    # Pre-data amendment: the stored P1 predictions were produced under a different BLAS
    # reduction order. A mask-free parity probe measured max |Δ| = 2.03e-12 (relative 9.0e-13,
    # 3 of 15,680 entries above 1e-12, median 8.9e-15). The tolerance below is 500x that
    # residual and still 6 orders of magnitude below the FP32 readout quantisation.
    "verify_atol": 1e-9,
    "verify_rtol": 1e-11,
    "p1_reference_macro_ap_t350": 0.726197,
    "p1_reference_per_category_ap_t350": {"bottle": 0.780498, "carpet": 0.690916, "hazelnut": 0.707176},
}


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def rel(p) -> str:
    return str(Path(p).relative_to(load.ROOT))


def input_hashes() -> tuple[dict, dict, dict]:
    fixed = [
        load.P1 / "protocol_resolved.json",
        *(load.P1 / f"states_t{t}.pt" for t in (100, 200, 300, 350)),
        *(load.P1 / f"predictions_t{t}.json" for t in (100, 200, 300, 350)),
    ]
    inputs = {rel(p): load.sha256_file(p) for p in fixed if p.exists()}
    for p in sorted(load.P1.glob("evaluation_features/*")):
        inputs[rel(p)] = load.sha256_file(p)
    expected_masks = {}
    for p in sorted(load.P1.glob("evaluation_features/*.json")):
        meta = json.loads(p.read_text())["source"]
        if meta["label"] == 1:
            expected_masks[meta["mask_path"]] = meta["mask_sha256"]
    sources = {}
    for p in sorted((load.ROOT / "exps/calm").glob("*.py")):
        sources[rel(p)] = load.sha256_file(p)
    for p in ("exps/task_p/p0_normal_support.py", "exps/task_p/p1_readout.py"):
        sources[p] = load.sha256_file(load.ROOT / p)
    return inputs, expected_masks, sources


def cmd_lock(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    inputs, masks, sources = input_hashes()
    lock = {
        "schema": CONSTANTS["schema"], "created_utc": utcnow(),
        "branch_head": "794d7e498ea093f99ae861e9f43f4ab59d887772",
        "constants": CONSTANTS, "inputs": inputs,
        "expected_mask_sha256": masks, "sources": sources,
        "run_order": ["lock", "a1_label_free_scoring", "a1_verification_against_predictions",
                      "a2_single_label_mask_pass", "audit"],
        "one_shot_rule": "analysis.json is create-exclusive; A2 never re-reads masks after it exists",
        "notes": [
            "dataset_root in protocol_resolved.json is stale; masks resolve under the local data/mvtec and are SHA-256 verified",
            "the CUDA device is authorized but unused: FP64 scorer, consumer Ada",
        ],
    }
    (out / "LOCK.json").write_text(json.dumps(lock, indent=1, sort_keys=True))
    print(f"LOCK written: {out/'LOCK.json'}")


def verify_lock(out: Path) -> None:
    """Inputs and expected mask hashes are immutable. Source corrections are permitted only
    through AMENDMENTS.json, and the actually executed hashes are written into the analysis."""
    lock = json.loads((out / "LOCK.json").read_text())
    inputs, masks, sources = input_hashes()
    bad = [k for k, v in lock["inputs"].items() if inputs.get(k) != v]
    if bad:
        raise SystemExit(f"STOP: LOCK input hash mismatch: {bad[:3]}")
    if masks != lock["expected_mask_sha256"]:
        raise SystemExit("STOP: expected mask hash set changed")
    amended = set()
    amend_path = out / "AMENDMENTS.json"
    if amend_path.exists():
        amended = {e["path"] for e in json.loads(amend_path.read_text())["source_amendments"]}
    bad = [k for k, v in lock["sources"].items() if k not in amended and sources.get(k) != v]
    if bad:
        raise SystemExit(f"STOP: LOCK source hash mismatch: {bad[:3]}")
    print(f"LOCK verified (inputs {len(inputs)}, sources {len(sources)}, amended {sorted(amended)})")


def group_images(eval_recs, cat):
    return sorted((r for r in eval_recs if r["category"] == cat), key=lambda r: r["rel"])


def cmd_a1(out: Path) -> None:
    verify_lock(out)
    ev = load.load_eval()
    scores_dir = out / "scores"
    scores_dir.mkdir(exist_ok=True)
    states = {t: {a: load.load_state(t, a) for a in load.ARMS} for t in (100, 200, 300, 350)}
    summary = {"created_utc": utcnow(), "groups": {}}
    for arm in load.ARMS:
        for g in CONSTANTS["groups"]:
            imgs = group_images(ev, g["cat"])
            feats = [r["feature"] for r in imgs]
            cells = decompose.build_cells(states[g["acq"]][arm], states[g["T"]][arm],
                                          load.CAT_INDEX[g["cat"]])
            res = decompose.score_group(cells, feats)
            payload = {f"{name}_{k}": v for name, rec in res.items() for k, v in rec.items()}
            np.savez_compressed(scores_dir / f"{arm}__{g['name']}.npz", **payload)
            key = arm + "__" + g["name"]
            n_rows = {name: int(cells[name]["bank"].shape[0]) for name in cells}
            summary["groups"][key] = {
                "n_rows": n_rows, "cells": sorted(cells), "images": [r["rel"] for r in imgs],
            }
            print("A1 %-18s %-12s rows %s" % (arm, g["name"], " ".join(f"{k}={v}" for k, v in n_rows.items())))
    (out / "A1_progress.json").write_text(json.dumps(summary, indent=1))


def cmd_verify(out: Path) -> None:
    """Cell 11 must reproduce the stored P1 predictions exactly (same frozen scorer)."""
    ev = {r["rel"]: r for r in load.load_eval()}
    result = {}
    worst = 0.0
    for arm in load.ARMS:
        for g in CONSTANTS["groups"]:
            pred = load.load_predictions(g["T"])
            z = np.load(out / "scores" / f"{arm}__{g['name']}.npz")
            imgs = [r for r in ev.values() if r["category"] == g["cat"]]
            imgs.sort(key=lambda r: r["rel"])
            d1 = z["11_d1"]
            if len(imgs) != d1.shape[0]:
                raise SystemExit("STOP: image order mismatch")
            adiff = rdiff = 0.0
            for i, rec in enumerate(imgs):
                ref = pred.get((arm, rec["rel"]))
                if ref is None:
                    raise SystemExit(f"STOP: no prediction for {arm} {rec['rel']} @{g['T']}")
                a = np.abs(d1[i] - ref["patch_distances"])
                adiff = max(adiff, float(a.max()))
                rdiff = max(rdiff, float((a / np.maximum(np.abs(ref["patch_distances"]), 1e-30)).max()))
            ok = adiff <= CONSTANTS["verify_atol"] and rdiff <= CONSTANTS["verify_rtol"]
            worst = max(worst, rdiff)
            result[f"{arm}__{g['name']}"] = {"abs": adiff, "rel": rdiff, "ok": ok}
            print(f"verify {arm:18s} {g['name']:12s} max|Δ|={adiff:.3e} max_rel={rdiff:.3e}")
    payload = {"created_utc": utcnow(), "per_group": result, "worst_rel": worst,
               "atol": CONSTANTS["verify_atol"], "rtol": CONSTANTS["verify_rtol"],
               "passed": all(v["ok"] for v in result.values())}
    (out / "A1_COMPLETE.json").write_text(json.dumps(payload, indent=1))
    if not payload["passed"]:
        raise SystemExit(f"STOP: cell-11 verification failed, worst_rel={worst:.3e}; no mask read")
    print(f"A1 verification PASS (worst {worst:.3e})")


def cmd_a2(out: Path) -> None:
    verify_lock(out)
    if (out / "analysis.json").exists():
        raise SystemExit("STOP: analysis.json exists; the one-shot masked pass is closed")
    a1 = json.loads((out / "A1_COMPLETE.json").read_text())
    if not a1["passed"]:
        raise SystemExit("STOP: A1 verification did not pass")
    ev = load.load_eval()
    by_cat = {c: [r for r in ev if r["category"] == c] for c in load.CATEGORIES}
    for c in by_cat:
        by_cat[c].sort(key=lambda r: r["rel"])

    # One mask/label read for the whole phase.
    masks = {r["rel"]: load.load_mask(r) for r in ev}
    labels = {r["rel"]: r["label"] for r in ev}
    for r in ev:
        m = masks[r["rel"]]
        if m.shape != r["hw"]:
            raise SystemExit("STOP: mask shape mismatch")

    # Category certificate radii from acquisition-time normals (A2 labels, one pass).
    radii, radius_detail = {}, {}
    for cat in load.CATEGORIES:
        acq = {"bottle": 100, "carpet": 200, "hazelnut": 300}[cat]
        bank = load.load_state(acq, load.PRIMARY_ARM)["bank"]
        normals = [r["feature"] for r in by_cat[cat] if r["label"] == 0]
        radii[load.CAT_INDEX[cat]] = certify.category_radius(bank, normals)
        radius_detail[cat] = radii[load.CAT_INDEX[cat]]
    print("radii", radius_detail)

    _, _, executed_sources = input_hashes()
    analysis = {"created_utc": utcnow(), "constants": CONSTANTS, "radii": radius_detail,
                "executed_sources": executed_sources,
                "arms": {}, "p1_validation": {}, "pooled_t350": {}}
    boot_arrays = {}

    for arm in load.ARMS:
        arm_out = {"groups": {}}
        for g in CONSTANTS["groups"]:
            imgs = by_cat[g["cat"]]
            hws = [r["hw"] for r in imgs]
            mks = [masks[r["rel"]] for r in imgs]
            labs = np.array([labels[r["rel"]] for r in imgs])
            z = np.load(out / "scores" / f"{arm}__{g['name']}.npz")
            ap, iauc, hists, maps = {}, {}, {}, {}
            for name in decompose.CELLS:
                maps[name] = stats.maps_from_patches(z[f"{name}_d1"], hws)
                met = stats.pooled_ap(maps[name], mks)
                ap[name] = met["ap"]
                iauc[name] = stats.image_auroc(labs, z[f"{name}_d1"].max(1))["auroc"]
                hists[name] = stats.binned_hist(maps[name], mks)
            sh = stats.shapley(ap["00"], ap["01"], ap["10"], ap["11"])
            # paired bootstrap: identical image weights across the four cells
            reps = stats.bootstrap_weights(labs, CONSTANTS["bootstrap_reps"], CONSTANTS["bootstrap_seed"])
            boot = {"intrusion": [], "eviction": [], "interaction": []}
            for w in reps:
                a = {n: stats.weighted_ap(hists[n], w) for n in ("00", "01", "10", "11")}
                s = stats.shapley(a["00"], a["01"], a["10"], a["11"])
                for k in boot:
                    boot[k].append(s[k])
            ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}
            viol = certify.violation(z["11_d1"], z["11_owner1"], load.CAT_INDEX[g["cat"]], radii)
            vs = certify.violation_summary(viol, z["11_d1"], z["11_d2"], labs)
            arm_out["groups"][g["name"]] = {
                "role": g["role"], "cat": g["cat"], "acq": g["acq"], "T": g["T"],
                "ap": ap, "image_auroc": iauc,
                "prevalence": stats.pooled_ap(maps["11"], mks)["prevalence"],
                "shapley": sh, "shapley_ci95": ci, "vir": vs,
            }
            boot_arrays[f"{arm}__{g['name']}__shapley"] = np.array([boot["intrusion"], boot["eviction"], boot["interaction"]])
            print(f"A2 {arm:18s} {g['name']:12s} AP00={ap['00']:.6f} AP01={ap['01']:.6f} "
                  f"AP10={ap['10']:.6f} AP11={ap['11']:.6f} intr={sh['intrusion']:+.4f} evict={sh['eviction']:+.4f}")
        prim = [g for g in CONSTANTS["groups"] if g["role"] == "primary"]
        arm_out["primary_macro"] = {
            k: float(np.mean([arm_out["groups"][g["name"]]["shapley"][k] for g in prim]))
            for k in ("intrusion", "eviction", "interaction", "total")}
        for k in ("intrusion", "eviction", "interaction"):
            arm_out["primary_macro"][f"{k}_ci95"] = [
                float(np.mean([arm_out["groups"][g["name"]]["shapley_ci95"][k][i] for g in prim])) for i in (0, 1)]
        analysis["arms"][arm] = arm_out

    # End-to-end validation against P1's published per-category AP at t=350 (cell 11).
    for cat in load.CATEGORIES:
        name = f"{cat}@350"
        mine = analysis["arms"][load.PRIMARY_ARM]["groups"][name]["ap"]["11"]
        ref = CONSTANTS["p1_reference_per_category_ap_t350"][cat]
        analysis["p1_validation"][name] = {"mine": mine, "p1": ref, "abs_diff": abs(mine - ref)}
    macro350 = float(np.mean([analysis["arms"][load.PRIMARY_ARM]["groups"][f"{c}@350"]["ap"]["11"]
                              for c in load.CATEGORIES]))
    analysis["p1_validation"]["macro_t350"] = {
        "mine": macro350, "p1": CONSTANTS["p1_reference_macro_ap_t350"],
        "abs_diff": abs(macro350 - CONSTANTS["p1_reference_macro_ap_t350"])}

    # Pooled final-checkpoint AP over the three category groups at t=350 (cell 11), greedy arm.
    pooled_maps, pooled_masks = [], []
    for cat in load.CATEGORIES:
        imgs = by_cat[cat]
        z = np.load(out / "scores" / f"{load.PRIMARY_ARM}__{cat}@350.npz")
        pooled_maps += stats.maps_from_patches(z["11_d1"], [r["hw"] for r in imgs])
        pooled_masks += [masks[r["rel"]] for r in imgs]
    analysis["pooled_t350"] = stats.pooled_ap(pooled_maps, pooled_masks)

    (out / "analysis.json").write_text(json.dumps(analysis, indent=1, sort_keys=True))
    np.savez_compressed(out / "bootstrap.npz", **boot_arrays)
    print("analysis written")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stage", required=True, choices=("lock", "a1", "verify", "a2"))
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    {"lock": cmd_lock, "a1": cmd_a1, "verify": cmd_verify, "a2": cmd_a2}[args.stage](args.out)


if __name__ == "__main__":
    main()
