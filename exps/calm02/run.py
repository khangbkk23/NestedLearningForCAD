# exps/calm02/run.py
"""CALM-02 staged runner. Irreversibility is pushed last.

    freeze   create-exclusive FREEZE_<bench>.json: source/config hashes before any training
    train    label-free: stream one arm x budget; save the charged state after each task
    score    label-free: score every checkpoint on the test images of the tasks learned so far
    lock     create-exclusive LOCK_<bench>.json over sources (= FREEZE), features, states and scores
    evaluate the single mask/label pass (create-exclusive EVAL_<bench>.json):
             primary   final native pixel AP and native forgetting AP[j,j] - AP[T,j], L2 vs L1 with
                       a paired image bootstrap;
             secondary 224^2 CADIC-definition P-AUPR / I-AUROC matrices, best-previous FM, and the
                       unmatched comparison with CADIC's reported numbers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from exps.calm01 import arms as A
from exps.calm02.data import ROOT, identities, p1_development_paths
from exps.calm02.score import Scorer, decode, pixel_map
from exps.task_p.p0_normal_support import require
from exps.task_p.p1_data import sha

OUT = ROOT / "results/hope_cad/calm02"
FEATURES = OUT / "features"
ARMS = {"CALM_L2": "L2_NESTED_U8", "CALM_L1": "L1_FIXED_U8"}   # task amendment 5: no CADIC arm
# CADIC paper (local PDF, Tables I/II/VII): (I-AUROC, P-AUPR) at 2,500 (B1) and 10,000 (B2) FP32 patches.
CADIC_PAPER = {"mvtec": {"B1": (0.955, 0.559), "B2": (0.972, 0.584)},
               "visa": {"B1": (0.828, 0.415), "B2": (0.891, 0.438)}}
BUDGETS = {"B1": A.BUDGETS["cadic2500"], "B2": A.BUDGETS["cadic10000"]}
FROZEN = {"eta": 2.0, "shortlist": 16, "floor": 1, "b_neighbours": 9, "bootstrap_reps": 1000,
          "bootstrap_seed": 20261023, "margin": 0.005, "bootstrap_bins": 8192}
SOURCES = ["exps/calm01/storage.py", "exps/calm01/closest_pair.py", "exps/calm01/calm.py",
           "exps/calm01/arms.py", "exps/calm02/data.py", "exps/calm02/extract.py", "exps/calm02/score.py",
           "exps/calm02/native.py", "exps/calm02/run.py", "exps/calm02/gate.py", "exps/calm02/test_calm02.py",
           "exps/task_p/p0_normal_support.py", "exps/calm/stats.py", "exps/task_p/p1_readout.py",
           "training/benchmark_metrics_v1.py", "dataset/benchmark_protocol_v1.py",
           "models/feature_extractors/cadic_vit_v1.py", "agents/tasks/task_calm_02_cadic_benchmark.md",
           "agents/tasks/task_calm_02a_early_gate.md"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=True)


def run_dir(bench, arm, budget):
    return OUT / "runs" / bench / f"{arm}_{budget}"


def task_features(bench, task, split):
    obj = torch.load(FEATURES / bench / f"{task}_{split}.pt", map_location="cpu", weights_only=False, mmap=True)
    return obj["features"], obj["rows"]


def freeze(bench):
    write_json(OUT / f"FREEZE_{bench}.json",
               {"benchmark": bench, "frozen": FROZEN, "arms": ARMS, "budgets": BUDGETS,
                "sha256": {s: sha(ROOT / s) for s in SOURCES},
                "features_manifest_sha256": sha(FEATURES / bench / "MANIFEST.json"),
                "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def verify_freeze(bench):
    fz = json.loads((OUT / f"FREEZE_{bench}.json").read_text())
    for s, digest in fz["sha256"].items():
        require(sha(ROOT / s) == digest, f"source changed after FREEZE: {s}")
    require(fz["frozen"] == FROZEN and fz["arms"] == ARMS and fz["budgets"] == BUDGETS, "config changed after FREEZE")
    require(sha(FEATURES / bench / "MANIFEST.json") == fz["features_manifest_sha256"], "features changed after FREEZE")
    return fz


def train(bench, arm, budget, threads, device):
    verify_freeze(bench)
    torch.set_num_threads(threads)
    ids = identities(bench)
    n_cat = len(ids["tasks"])
    name = ARMS[arm]
    m = A.build(name, BUDGETS[budget], n_cat, cache_device=device)   # FP64 distance cache only
    require(m.eta == (FROZEN["eta"] if name == "L2_NESTED_U8" else 0.0)
            and m.shortlist == FROZEN["shortlist"] and m.floor == FROZEN["floor"], "frozen config drift")
    out = run_dir(bench, arm, budget)
    out.mkdir(parents=True, exist_ok=False)
    log, serial = [], 0
    for k, t in enumerate(ids["tasks"]):
        feats, rows = task_features(bench, t["task"], "train")
        require([r["relative_path"] for r in rows] == [r["relative_path"] for r in t["train"]], "row order drift")
        m.begin_category(k)
        start, stats = time.perf_counter(), {"inserted": 0, "replaced": 0, "rejected": 0}
        for i in range(len(rows)):
            s = m.update(feats[i].clone(), list(range(serial, serial + 784))); serial += 784
            for key in stats:
                stats[key] += s.get(key, 0)
        m.end_category()
        torch.save({"state": A.persistent_state(m, name, n_cat), "occupied": m.bank.occupied.clone(),
                    "owner": m.bank.owner.clone(), "ids": m.bank.ids.clone(),
                    "ledger": A.ledger(name, m.capacity, n_cat)}, out / f"task_{k:02d}.pt")
        rec = {"task": t["task"], "seconds": time.perf_counter() - start, **stats,
               "rows_by_task": {ids["tasks"][j]["task"]: int((m.bank.owner == j).sum()) for j in range(k + 1)},
               "price": {ids["tasks"][j]["task"]: m.price[j] for j in m.price},
               "deficit": {ids["tasks"][j]["task"]: d for j, d in m.deficits().items()}}
        log.append(rec)
        print(bench, arm, budget, t["task"], f"{rec['seconds']:.0f}s", stats, flush=True)
    write_json(out / "train_log.json", {"arm": name, "budget_bytes": BUDGETS[budget], "capacity": m.capacity,
                                        "cache_device": device, "torch": torch.__version__,
                                        "ledger": A.ledger(name, m.capacity, n_cat), "frozen": FROZEN, "tasks": log})


def score(bench, arm, budget, device):
    verify_freeze(bench)
    ids = identities(bench)
    out = run_dir(bench, arm, budget)
    test = [task_features(bench, t["task"], "test") for t in ids["tasks"]]
    for k in range(len(ids["tasks"])):
        target = out / f"scores_{k:02d}.npz"
        require(not target.exists(), "scores exist")
        ck = torch.load(out / f"task_{k:02d}.pt", map_location="cpu", weights_only=False)
        sc = Scorer(decode(ck["state"])[ck["occupied"]], device)
        arrays = {}
        for j in range(k + 1):
            feats, rows = test[j]
            pd, ic = np.empty((len(rows), 784)), np.empty(len(rows))
            for i in range(len(rows)):
                d, idx = sc.patch(feats[i])
                pd[i], ic[i] = d.numpy(), sc.image_cadic(feats[i], d, idx)
            arrays[f"patch_{j:02d}"], arrays[f"image_cadic_{j:02d}"] = pd, ic
        np.savez(target, **arrays)
        print(bench, arm, budget, "scored checkpoint", k, flush=True)


def lock(bench):
    target = OUT / f"LOCK_{bench}.json"
    require(not target.exists(), "LOCK exists")
    verify_freeze(bench)
    files = {s: sha(ROOT / s) for s in SOURCES}
    files[f"features/{bench}/MANIFEST.json"] = sha(FEATURES / bench / "MANIFEST.json")
    files[f"FREEZE_{bench}.json"] = sha(OUT / f"FREEZE_{bench}.json")
    for arm in ARMS:
        for budget in BUDGETS:
            d = run_dir(bench, arm, budget)
            T = len(identities(bench)["tasks"])
            require(all((d / f"{p}_{k:02d}.{e}").exists() for k in range(T) for p, e in (("task", "pt"), ("scores", "npz"))),
                    f"incomplete run {d}")
            for p in sorted(d.glob("*")):
                files[str(p.relative_to(OUT))] = sha(p)
    write_json(target, {"benchmark": bench, "frozen": FROZEN, "arms": ARMS, "budgets": BUDGETS,
                        "sha256": files, "locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def verify_lock(bench):
    lk = json.loads((OUT / f"LOCK_{bench}.json").read_text())
    for name, digest in lk["sha256"].items():
        path = OUT / name if name.startswith(("runs/", "features/", "FREEZE_")) else ROOT / name
        require(sha(path) == digest, f"post-LOCK change: {name}")
    return lk


def masks_224(root, task_rows):
    from PIL import Image
    out = np.zeros((len(task_rows), 224, 224), dtype=bool)
    for i, r in enumerate(task_rows):
        if r["label"]:
            with Image.open(root / r["mask_path"]) as m:
                out[i] = np.array(m.convert("L").resize((224, 224), Image.Resampling.NEAREST)) > 0
    return out


def _band(point, ci, m, up="BETTER", down="WORSE"):
    if point >= m and ci[0] > 0:
        return up
    if point <= -m and ci[1] < 0:
        return down
    return "NO_DIFFERENCE_ESTABLISHED"


def evaluate(bench):
    from sklearn.metrics import roc_auc_score
    from exps.calm.stats import ap_from_hist, bootstrap_weights, pooled_ap
    from exps.calm02.native import native_ap, native_hists, native_hw, native_mask
    from training.benchmark_metrics_v1 import forgetting_matrix
    lk = verify_lock(bench)
    target = OUT / f"EVAL_{bench}.json"
    require(not target.exists(), "evaluation exists: one-shot")
    ids = identities(bench)
    root = Path(ids["root"])
    names = [t["task"] for t in ids["tasks"]]
    T, reps, m = len(names), FROZEN["bootstrap_reps"], FROZEN["margin"]
    rows = [task_features(bench, n, "test")[1] for n in names]
    labels = [np.array([r["label"] for r in rs]) for rs in rows]
    hws = [[native_hw(root / r["relative_path"]) for r in rs] for rs in rows]
    dev = set(p1_development_paths()) if bench == "mvtec" else set()
    keep = [[i for i, r in enumerate(rs) if r["relative_path"] not in dev] for rs in rows]
    W = [bootstrap_weights(labels[j], reps, FROZEN["bootstrap_seed"] + j) for j in range(T)]
    combos = [(a, b) for a in ARMS for b in BUDGETS]
    result = {"benchmark": bench, "lock_sha256": sha(OUT / f"LOCK_{bench}.json"), "tasks": names, "arms": {}}

    # Secondary: 224^2 CADIC-definition matrices (all checkpoints x learned tasks).
    m224 = [masks_224(root, rs) for rs in rows]                       # mask read (224^2)
    cells = {}
    for arm, budget in combos:
        d = run_dir(bench, arm, budget)
        P = np.full((T, T), np.nan); I = np.full((T, T), np.nan); Imax = np.full((T, T), np.nan)
        for k in range(T):
            z = np.load(d / f"scores_{k:02d}.npz")
            for j in range(k + 1):
                pd = torch.from_numpy(z[f"patch_{j:02d}"])
                P[k, j] = pooled_ap([pixel_map(pd[i]).numpy() for i in range(len(pd))], list(m224[j]))["ap"]
                I[k, j] = roc_auc_score(labels[j], z[f"image_cadic_{j:02d}"])
                Imax[k, j] = roc_auc_score(labels[j], pd.max(1).values.numpy())
                if k == j or k == T - 1:
                    cells[(arm, budget, k, j)] = (z[f"patch_{j:02d}"], z[f"image_cadic_{j:02d}"])
        fm_p, fm_i = forgetting_matrix(P), forgetting_matrix(I)
        result["arms"][f"{arm}_{budget}"] = {
            "secondary_224": {"p_aupr_matrix": P.tolist(), "i_auroc_matrix": I.tolist(),
                              "i_auroc_max_matrix": Imax.tolist(), "macro_p_aupr": float(P[-1].mean()),
                              "macro_i_auroc": float(I[-1].mean()), "macro_i_auroc_max": float(Imax[-1].mean()),
                              "fm_best_previous_p_aupr": fm_p["fm"], "fm_best_previous_i_auroc": fm_i["fm"]},
            "train_ledger": json.loads((d / "train_log.json").read_text())["ledger"]}
        print(bench, arm, budget, "224: P-AUPR %.4f I-AUROC %.4f" % (P[-1].mean(), I[-1].mean()), flush=True)
    del m224

    # Primary: native pixel AP at acquisition (j,j) and final (T-1,j), with paired bootstrap per task.
    nat = {c: {"acq": np.zeros(T), "fin": np.zeros(T), "fin_no_dev": np.zeros(T)} for c in combos}
    rep = {c: {"acq": np.zeros((reps, T)), "fin": np.zeros((reps, T)), "iauroc": np.zeros((reps, T))} for c in combos}
    for j in range(T):
        masks = [native_mask(root / r["mask_path"] if r["mask_path"] else None, hw)   # native mask read
                 for r, hw in zip(rows[j], hws[j])]
        for c in combos:
            arm, budget = c
            for kind, k in (("acq", j), ("fin", T - 1)):
                patches = cells[(arm, budget, k, j)][0]
                nat[c][kind][j] = native_ap(patches, hws[j], masks)["ap"]
                pos, tot = native_hists(patches, hws[j], masks, FROZEN["bootstrap_bins"])
                for r in range(reps):
                    rep[c][kind][r, j] = ap_from_hist(pos.T @ W[j][r], tot.T @ W[j][r])
            fin_patches, img = cells[(arm, budget, T - 1, j)]
            sel = keep[j]
            nat[c]["fin_no_dev"][j] = (nat[c]["fin"][j] if len(sel) == len(rows[j]) else
                                       native_ap(fin_patches[sel], [hws[j][i] for i in sel], [masks[i] for i in sel])["ap"])
            for r in range(reps):
                rep[c]["iauroc"][r, j] = roc_auc_score(labels[j], img, sample_weight=W[j][r])
        print(bench, names[j], "native", {f"{a}_{b}": round(nat[(a, b)]["fin"][j], 4) for a, b in combos}, flush=True)
        del masks

    old = slice(0, T - 1)                                                # tasks with a later checkpoint
    for c in combos:
        arm, budget = c
        forget = nat[c]["acq"][old] - nat[c]["fin"][old]
        result["arms"][f"{arm}_{budget}"]["primary_native"] = {
            "final_ap": dict(zip(names, nat[c]["fin"].tolist())), "acquisition_ap": dict(zip(names, nat[c]["acq"].tolist())),
            "macro_final_ap": float(nat[c]["fin"].mean()), "macro_forgetting": float(forget.mean()),
            "forgetting_per_task": dict(zip(names[:-1], forget.tolist())),
            "worst_task_forgetting": float(forget.max()),
            "macro_final_ap_without_p1_dev": float(nat[c]["fin_no_dev"].mean())}

    # Pre-registered contrast: CALM_L2 - CALM_L1, same memory/scorer/data, paired image bootstrap.
    contrasts, decisions = {}, {}
    for budget in BUDGETS:
        a, b = ("CALM_L2", budget), ("CALM_L1", budget)
        ra, rb, pa, pb = rep[a], rep[b], result["arms"][f"CALM_L2_{budget}"], result["arms"][f"CALM_L1_{budget}"]
        d_ap = ra["fin"].mean(1) - rb["fin"].mean(1)
        d_f = (ra["acq"][:, old] - ra["fin"][:, old]).mean(1) - (rb["acq"][:, old] - rb["fin"][:, old]).mean(1)
        d_i = ra["iauroc"].mean(1) - rb["iauroc"].mean(1)
        ap = {"point": pa["primary_native"]["macro_final_ap"] - pb["primary_native"]["macro_final_ap"],
              "ci95": np.percentile(d_ap, [2.5, 97.5]).tolist()}
        fg = {"point": pa["primary_native"]["macro_forgetting"] - pb["primary_native"]["macro_forgetting"],
              "ci95": np.percentile(d_f, [2.5, 97.5]).tolist()}
        ia = {"point": pa["secondary_224"]["macro_i_auroc"] - pb["secondary_224"]["macro_i_auroc"],
              "ci95": np.percentile(d_i, [2.5, 97.5]).tolist()}
        contrasts[budget] = {"native_final_ap": ap, "native_forgetting": fg, "i_auroc_cadic": ia}
        ap_band = _band(ap["point"], ap["ci95"], m)
        f_band = _band(fg["point"], fg["ci95"], m, up="HIGHER", down="LOWER")
        decisions[budget] = {"final_ap": ap_band, "forgetting": f_band,
                             "slow_level_supported_on_this_benchmark": ap_band == "BETTER" and f_band != "HIGHER"}
    result["contrast_L2_minus_L1"], result["decisions"] = contrasts, decisions

    if bench in CADIC_PAPER:                    # unmatched, descriptive: no CI exists for these
        vs = {}
        for arm, budget in combos:
            ref_i, ref_p = CADIC_PAPER[bench][budget]
            s2 = result["arms"][f"{arm}_{budget}"]["secondary_224"]
            dp, di = s2["macro_p_aupr"] - ref_p, s2["macro_i_auroc"] - ref_i
            vs[f"{arm}_{budget}"] = {"cadic_paper_i_auroc": ref_i, "cadic_paper_p_aupr": ref_p,
                                     "delta_p_aupr_224": dp, "delta_i_auroc": di, "matched": False,
                                     "label": ("exceeds reported CADIC" if dp >= m else "below reported CADIC"
                                               if dp <= -m else "within +-0.005 of reported CADIC")}
        result["vs_cadic_paper"] = vs
    write_json(target, result)
    print(json.dumps(decisions, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("freeze", "train", "score", "lock", "evaluate"))
    ap.add_argument("--benchmark", required=True, choices=("mvtec", "visa"))
    ap.add_argument("--arm", choices=tuple(ARMS))
    ap.add_argument("--budget", choices=tuple(BUDGETS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    {"freeze": lambda: freeze(a.benchmark), "lock": lambda: lock(a.benchmark), "evaluate": lambda: evaluate(a.benchmark),
     "train": lambda: train(a.benchmark, a.arm, a.budget, a.threads, a.device),
     "score": lambda: score(a.benchmark, a.arm, a.budget, a.device)}[a.stage]()


if __name__ == "__main__":
    main()
