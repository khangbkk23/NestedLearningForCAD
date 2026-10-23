"""CALM-00A artifact-only audit: recompute every reported number from stored arrays.

Reads the stored per-patch distances and the already-authorized development masks; never
re-scores a feature against a bank, so the frozen scoring pass remains one-shot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from exps.calm import certify, load, stats
from exps.calm.run import CONSTANTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run = args.run
    analysis = json.loads((run / "analysis.json").read_text())
    ev = load.load_eval()
    by_cat = {c: sorted((r for r in ev if r["category"] == c), key=lambda r: r["rel"]) for c in load.CATEGORIES}
    masks = {r["rel"]: load.load_mask(r) for r in ev}
    labels = {r["rel"]: r["label"] for r in ev}

    # radii must reproduce exactly
    radii, radius_diff = {}, {}
    for cat in load.CATEGORIES:
        acq = {"bottle": 100, "carpet": 200, "hazelnut": 300}[cat]
        bank = load.load_state(acq, load.PRIMARY_ARM)["bank"]
        normals = [r["feature"] for r in by_cat[cat] if r["label"] == 0]
        radii[load.CAT_INDEX[cat]] = certify.category_radius(bank, normals)
        radius_diff[cat] = abs(radii[load.CAT_INDEX[cat]] - analysis["radii"][cat])

    report = {"schema": "calm00a_audit_v1", "run": str(run), "groups": {}, "max_ap_diff": 0.0,
              "radius_diff": radius_diff, "vir_max_diff": 0.0, "shapley": {}}
    for arm in CONSTANTS["arms"]:
        for g in CONSTANTS["groups"]:
            imgs = by_cat[g["cat"]]
            hws = [r["hw"] for r in imgs]
            mks = [masks[r["rel"]] for r in imgs]
            labs = np.array([labels[r["rel"]] for r in imgs])
            z = np.load(run / "scores" / f"{arm}__{g['name']}.npz")
            apv = {}
            for name in ("00", "01", "10", "11", "oracle"):
                maps = stats.maps_from_patches(z[f"{name}_d1"], hws)
                apv[name] = stats.pooled_ap(maps, mks)["ap"]
                d = abs(apv[name] - analysis["arms"][arm]["groups"][g["name"]]["ap"][name])
                report["max_ap_diff"] = max(report["max_ap_diff"], d)
                if d > 1e-9:
                    raise SystemExit(f"STOP: AP mismatch {arm} {g['name']} {name}: {d:.3e}")
            sh = stats.shapley(apv["00"], apv["01"], apv["10"], apv["11"])
            viol = certify.violation(z["11_d1"], z["11_owner1"], load.CAT_INDEX[g["cat"]], radii)
            vs = certify.violation_summary(viol, z["11_d1"], z["11_d2"], labs)
            stored = analysis["arms"][arm]["groups"][g["name"]]["vir"]
            d = abs(vs["vir"] - stored["vir"]) if np.isfinite(vs["vir"]) and np.isfinite(stored["vir"]) else 0.0
            report["vir_max_diff"] = max(report["vir_max_diff"], d)
            if d > 1e-12:
                raise SystemExit(f"STOP: VIR mismatch {arm} {g['name']}: {d:.3e}")
            report["groups"][f"{arm}__{g['name']}"] = {
                "ap": apv, "shapley": sh, "vir": vs, "role": g["role"]}
            print(f"audit {arm:18s} {g['name']:12s} " +
                  " ".join(f"AP{k}={apv[k]:.6f}" for k in ("00", "01", "10", "11", "oracle")) +
                  f" intr={sh['intrusion']:+.4f} evict={sh['eviction']:+.4f} inter={sh['interaction']:+.4f}")

    for arm in CONSTANTS["arms"]:
        prim = [g for g in CONSTANTS["groups"] if g["role"] == "primary"]
        report["shapley"][arm] = {
            k: float(np.mean([report["groups"][f"{arm}__{g['name']}"]["shapley"][k] for g in prim]))
            for k in ("intrusion", "eviction", "interaction", "total", "intrusion_first",
                      "intrusion_second", "eviction_first", "eviction_second")}
        report["shapley"][arm]["groups"] = [g["name"] for g in prim]
    report["checks"] = {
        "ap_tolerance": 1e-9, "vir_tolerance": 1e-12,
        "passed": report["max_ap_diff"] <= 1e-9 and report["vir_max_diff"] <= 1e-12,
    }
    out = args.out or (run / "audit.json")
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print("AUDIT", "PASS" if report["checks"]["passed"] else "FAIL",
          f"max_ap_diff={report['max_ap_diff']:.2e} vir={report['vir_max_diff']:.2e}")
    if not report["checks"]["passed"]:
        raise SystemExit("STOP: audit failed")


if __name__ == "__main__":
    main()
