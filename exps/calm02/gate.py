# exps/calm02/gate.py
"""CALM-02A early falsification gate: MVTec B1, CALM-L2 vs CALM-L1, on the consumed P1 development
cohort only (60 images: 10 good + 10 defective x bottle/carpet/hazelnut). Exploratory evidence.

    score    label-free: every checkpoint x learned dev category; patch distances + latency
    replay   label-free: retrain a CALM_L2_B1 prefix in memory and compare states bitwise
    lock     create-exclusive GATE_LOCK.json
    evaluate the single mask read (30 masks, SHA-256 verified): GATE_EVAL.json
    audit    independent sklearn recomputation + reproduction of P1's published t=350 numbers
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from exps.calm.stats import binned_hist, maps_from_patches, pooled_ap, weighted_ap
from exps.calm01 import arms as A
from exps.calm02 import run as R
from exps.calm02.data import ROOT
from exps.calm02.score import Scorer, decode
from exps.task_p.p0_normal_support import require
from exps.task_p.p1_data import native_mask, sha
from exps.task_p.p1_readout import image_metrics

GATE = R.OUT / "gate_mvtec_b1"
ARMS = ("CALM_L1", "CALM_L2")
BUDGET, BENCH, FINAL = "B1", "mvtec", 14
CATS = ("bottle", "carpet", "hazelnut")
ACQ = {"bottle": 0, "carpet": 3, "hazelnut": 5}          # checkpoint after the category's task
P1_PROTOCOL = ROOT / "results/hope_cad/task_p/p1/protocol_resolved.json"
SEED, REPS, M_AP, M_AUC = 770011, 1000, 0.005, 0.01
TASKS = ["agents/tasks/task_calm_02a_early_gate.md"]


def cohort():
    rows = json.loads(P1_PROTOCOL.read_text())["evaluations"]
    require(len(rows) == 60, "P1 cohort must have 60 identities")
    return {c: [r for r in rows if r["category"] == c] for c in CATS}


def score(device):
    R.verify_freeze(BENCH)
    GATE.mkdir(parents=True, exist_ok=True)
    co = cohort()
    queries = {}
    for c in CATS:
        feats, rows = R.task_features(BENCH, c, "test")
        idx = {r["relative_path"]: i for i, r in enumerate(rows)}
        for r in co[c]:
            require(rows[idx[r["relative_path"]]]["source_sha256"] == r["sha256"], "identity mismatch")
        queries[c] = torch.stack([feats[idx[r["relative_path"]]] for r in co[c]])
    for arm in ARMS:
        target = GATE / f"{arm}_scores.npz"
        require(not target.exists(), "gate scores exist")
        d = R.run_dir(BENCH, arm, BUDGET)
        arrays, latency = {}, []
        for k in range(FINAL + 1):
            ck = torch.load(d / f"task_{k:02d}.pt", map_location="cpu", weights_only=False)
            sc = Scorer(decode(ck["state"])[ck["occupied"]], device)
            for c in CATS:
                if k < ACQ[c]:
                    continue
                out = np.empty((len(queries[c]), 784))
                for i, q in enumerate(queries[c]):
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    t = time.perf_counter()
                    dist, _ = sc.patch(q)
                    latency.append(time.perf_counter() - t)
                    out[i] = dist.numpy()
                arrays[f"{c}_{k:02d}"] = out
        np.savez(target, latency=np.array(latency), **arrays)
        print(arm, "scored", len(arrays), "cells", flush=True)


def replay(prefix, threads, device):
    """Retrain the first ``prefix`` tasks of CALM_L2_B1 in memory; compare with the saved states."""
    torch.set_num_threads(threads)
    ids = R.identities(BENCH)
    m = A.build(R.ARMS["CALM_L2"], R.BUDGETS[BUDGET], len(ids["tasks"]), cache_device=device)
    serial, out = 0, {}
    for k, t in enumerate(ids["tasks"][:prefix]):
        feats, _ = R.task_features(BENCH, t["task"], "train")
        m.begin_category(k)
        for i in range(len(feats)):
            m.update(feats[i].clone(), list(range(serial, serial + 784))); serial += 784
        m.end_category()
        saved = torch.load(R.run_dir(BENCH, "CALM_L2", BUDGET) / f"task_{k:02d}.pt", map_location="cpu", weights_only=False)
        now = A.persistent_state(m, R.ARMS["CALM_L2"], len(ids["tasks"]))
        out[t["task"]] = all(torch.equal(now[key], saved["state"][key]) for key in now) and \
            torch.equal(m.bank.ids, saved["ids"])
        print("replay", t["task"], "bitwise equal:", out[t["task"]], flush=True)
    R.write_json(GATE / "REPLAY.json", {"arm": "CALM_L2_B1", "prefix_tasks": prefix, "threads": threads,
                                        "cache_device": device, "bitwise_equal": out})


def lock():
    target = GATE / "GATE_LOCK.json"
    require(not target.exists(), "gate LOCK exists")
    R.verify_freeze(BENCH)
    files = {s: sha(ROOT / s) for s in R.SOURCES + TASKS + ["agents/tasks/task_calm_02_cadic_benchmark.md"]}
    files[str(P1_PROTOCOL.relative_to(ROOT))] = sha(P1_PROTOCOL)
    files[str((R.OUT / "FREEZE_mvtec.json").relative_to(ROOT))] = sha(R.OUT / "FREEZE_mvtec.json")
    for arm in ARMS:
        d = R.run_dir(BENCH, arm, BUDGET)
        for p in sorted(d.glob("*")) + [GATE / f"{arm}_scores.npz"]:
            files[str(p.relative_to(ROOT))] = sha(p)
    if (GATE / "REPLAY.json").exists():
        files[str((GATE / "REPLAY.json").relative_to(ROOT))] = sha(GATE / "REPLAY.json")
    R.write_json(target, {"sha256": files, "seed": SEED, "reps": REPS, "margins": {"ap": M_AP, "auroc": M_AUC},
                          "locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def verify_gate_lock():
    lk = json.loads((GATE / "GATE_LOCK.json").read_text())
    for name, digest in lk["sha256"].items():
        require(sha(ROOT / name) == digest, f"post-LOCK change: {name}")


def weights(co, reps, seed):
    """Paired image weights, stratified within category x label; one draw per rep for all arms/cells."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        w = {}
        for c in CATS:
            y = np.array([r["label"] for r in co[c]]); wc = np.zeros(len(y))
            for v in (0, 1):
                idx = np.flatnonzero(y == v)
                wc[idx] = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx))
            w[c] = wc
        out.append(w)
    return out


def classify(d, ci, margin, lower_is_better=False):
    s = -1 if lower_is_better else 1
    if s * d >= margin and (ci[0] > 0 if s > 0 else ci[1] < 0):
        return "IMPROVEMENT"
    if s * d <= -margin and (ci[1] < 0 if s > 0 else ci[0] > 0):
        return "DETERIORATION"
    return "NO_DIFFERENCE_ESTABLISHED"


def gate_verdict(cls):
    primary, secondary = (cls["macro_final_ap"], cls["macro_forgetting"]), (cls["pooled_final_ap"], cls["macro_image_auroc"])
    if "DETERIORATION" in primary:
        return "NEGATIVE_SIGNAL"
    if "IMPROVEMENT" in primary and "DETERIORATION" not in secondary:
        return "POSITIVE_SIGNAL"
    return "INCONCLUSIVE"


def evaluate():
    verify_gate_lock()
    target = GATE / "GATE_EVAL.json"
    require(not target.exists(), "gate evaluation exists: one-shot")
    co = cohort()
    root = ROOT / "data/mvtec"
    masks = {c: [native_mask(root, r) for r in co[c]] for c in CATS}       # the single mask read
    labels = {c: np.array([r["label"] for r in co[c]]) for c in CATS}
    hws = {c: [r["hw"] for r in co[c]] for c in CATS}
    W = weights(co, REPS, SEED)
    res, boot = {"cohort": "P1 development (consumed; exploratory)", "arms": {}}, {}
    for arm in ARMS:
        z = np.load(GATE / f"{arm}_scores.npz")
        traj, auc = {c: {} for c in CATS}, {}
        hist, fin_maps = {}, {}
        for c in CATS:
            for k in range(ACQ[c], FINAL + 1):
                maps = maps_from_patches(z[f"{c}_{k:02d}"], hws[c])
                traj[c][k] = pooled_ap(maps, masks[c])["ap"]
                if k in (ACQ[c], FINAL):
                    hist[(c, k)] = binned_hist(maps, masks[c])
                if k == FINAL:
                    fin_maps[c] = maps
            auc[c] = image_metrics(labels[c], z[f"{c}_{FINAL:02d}"].max(1))["auroc"]
        all_maps = [mp for c in CATS for mp in fin_maps[c]]
        all_masks = [mk for c in CATS for mk in masks[c]]
        pooled = pooled_ap(all_maps, all_masks)
        hist["pooled"] = binned_hist(all_maps, all_masks)
        forget = {c: traj[c][ACQ[c]] - traj[c][FINAL] for c in CATS}
        tl = json.loads((R.run_dir(BENCH, arm, BUDGET) / "train_log.json").read_text())
        res["arms"][arm] = {
            "final_ap": {c: traj[c][FINAL] for c in CATS}, "acquisition_ap": {c: traj[c][ACQ[c]] for c in CATS},
            "ap_trajectory": {c: {str(k): v for k, v in traj[c].items()} for c in CATS},
            "macro_final_ap": float(np.mean([traj[c][FINAL] for c in CATS])),
            "pooled_final_ap": pooled["ap"], "pooled_prevalence": pooled["prevalence"],
            "forgetting": forget, "macro_forgetting": float(np.mean(list(forget.values()))),
            "image_auroc": auc, "macro_image_auroc": float(np.mean(list(auc.values()))),
            "persistent_bytes": tl["ledger"]["charged_persistent_bytes"], "capacity_rows": tl["capacity"],
            "update_seconds_total": float(sum(t["seconds"] for t in tl["tasks"])),
            "inference_latency_ms_median": float(np.median(z["latency"]) * 1e3),
            "rows_final_by_task": tl["tasks"][-1]["rows_by_task"], "prices_final": tl["tasks"][-1]["price"]}
        # bootstrap replicates per arm
        b = {"macro_final_ap": [], "macro_forgetting": [], "pooled_final_ap": [], "macro_image_auroc": []}
        for w in W:
            fin = [weighted_ap(hist[(c, FINAL)], w[c]) for c in CATS]
            acq = [weighted_ap(hist[(c, ACQ[c])], w[c]) for c in CATS]
            b["macro_final_ap"].append(np.mean(fin))
            b["macro_forgetting"].append(np.mean(np.array(acq) - np.array(fin)))
            b["pooled_final_ap"].append(weighted_ap(hist["pooled"], np.concatenate([w[c] for c in CATS])))
            b["macro_image_auroc"].append(np.mean([image_metrics(labels[c], z[f"{c}_{FINAL:02d}"].max(1), w[c])["auroc"]
                                                   for c in CATS]))
        boot[arm] = {k: np.array(v) for k, v in b.items()}
    L2, L1 = res["arms"]["CALM_L2"], res["arms"]["CALM_L1"]
    effects, cls = {}, {}
    for key, margin, low in (("macro_final_ap", M_AP, False), ("macro_forgetting", M_AP, True),
                             ("pooled_final_ap", M_AP, False), ("macro_image_auroc", M_AUC, False)):
        d = L2[key] - L1[key]
        ci = np.percentile(boot["CALM_L2"][key] - boot["CALM_L1"][key], [2.5, 97.5]).tolist()
        effects[key] = {"L2_minus_L1": d, "ci95": ci}
        cls[key] = classify(d, ci, margin, lower_is_better=low)
    effects["per_category_final_ap"] = {c: L2["final_ap"][c] - L1["final_ap"][c] for c in CATS}
    effects["per_category_forgetting"] = {c: L2["forgetting"][c] - L1["forgetting"][c] for c in CATS}
    res.update(effects=effects, classification=cls, verdict=gate_verdict(cls))
    R.write_json(target, res)
    print(json.dumps({"verdict": res["verdict"], "classification": cls}, indent=1))


def audit():
    from sklearn.metrics import average_precision_score, roc_auc_score
    from exps.task_p.p1_readout import native_map
    verify_gate_lock()
    ev = json.loads((GATE / "GATE_EVAL.json").read_text())
    co = cohort()
    root = ROOT / "data/mvtec"
    masks = {c: [native_mask(root, r) for r in co[c]] for c in CATS}
    labels = {c: np.array([r["label"] for r in co[c]]) for c in CATS}
    worst_ap, worst_auc, worst_pooled = 0.0, 0.0, 0.0
    for arm in ARMS:
        z = np.load(GATE / f"{arm}_scores.npz")
        for c in CATS:
            for k in range(ACQ[c], FINAL + 1):
                s = np.concatenate([native_map(torch.from_numpy(p), r["hw"]).ravel() for p, r in zip(z[f"{c}_{k:02d}"], co[c])])
                y = np.concatenate([m.ravel() for m in masks[c]])
                worst_ap = max(worst_ap, abs(average_precision_score(y, s) - ev["arms"][arm]["ap_trajectory"][c][str(k)]))
            worst_auc = max(worst_auc, abs(roc_auc_score(labels[c], z[f"{c}_{FINAL:02d}"].max(1)) - ev["arms"][arm]["image_auroc"][c]))
        s = np.concatenate([native_map(torch.from_numpy(p), r["hw"]).ravel() for c in CATS for p, r in zip(z[f"{c}_{FINAL:02d}"], co[c])])
        y = np.concatenate([m.ravel() for c in CATS for m in masks[c]])
        worst_pooled = max(worst_pooled, abs(average_precision_score(y, s) - ev["arms"][arm]["pooled_final_ap"]))
    # The gate metric path applied to P1's own stored t=350 predictions must reproduce P1's published numbers.
    preds = json.loads((ROOT / "results/hope_cad/task_p/p1/predictions_t350.json").read_text())
    by = {(p["arm"], p["source"]["relative_path"]): np.array(p["patch_distances"]) for p in preds}
    p1_ap = {}
    for c in CATS:
        maps = maps_from_patches(np.stack([by[("FULL784_GREEDY", r["relative_path"])] for r in co[c]]), [r["hw"] for r in co[c]])
        p1_ap[c] = pooled_ap(maps, masks[c])["ap"]
    all_maps = [mp for c in CATS for mp in maps_from_patches(np.stack([by[("FULL784_GREEDY", r["relative_path"])] for r in co[c]]), [r["hw"] for r in co[c]])]
    p1_pooled = pooled_ap(all_maps, [m for c in CATS for m in masks[c]])["ap"]
    published = {"bottle": 0.780498, "carpet": 0.690916, "hazelnut": 0.707176, "macro": 0.726197, "pooled": 0.600111}
    rep = {"max_abs_ap_vs_sklearn": worst_ap, "max_abs_image_auroc_vs_sklearn": worst_auc,
           "max_abs_pooled_ap_vs_sklearn": worst_pooled,
           "p1_reproduction": {**p1_ap, "macro": float(np.mean(list(p1_ap.values()))), "pooled": p1_pooled},
           "p1_published": published}
    rep["p1_max_abs_diff_6dp"] = max(abs(round(rep["p1_reproduction"][k], 6) - v) for k, v in published.items())
    rep["pass"] = worst_ap < 1e-9 and worst_auc < 1e-12 and worst_pooled < 1e-9 and rep["p1_max_abs_diff_6dp"] < 1.5e-6
    R.write_json(GATE / "GATE_AUDIT.json", rep)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("score", "replay", "lock", "evaluate", "audit"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--prefix", type=int, default=2)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    {"score": lambda: score(a.device), "replay": lambda: replay(a.prefix, a.threads, a.device),
     "lock": lock, "evaluate": evaluate, "audit": audit}[a.stage]()


if __name__ == "__main__":
    main()
