# exps/ntm/stage0.py
"""NTM-00 Stage 0: exploratory screen of the Nested Test-time Memory on the consumed P1 cohort.

Arms (patch scores; image score = max; maps = frozen P1 readout):
    BANK       d_M(z)                                  continual CALM-L1 bank (slow level)
    CTX        d_X(z)                                  in-image context memory (fast level)
    NTM_FLOOR  min(d_M, max(d_X, tau_{owner(NN_M)}))   nested, primary
    NTM_MIN    min(d_M, d_X)                           nested, no floor

    prepare   label-free: tau (training data only), d_M + owners (parity with CALM-02A), d_X
    lock      create-exclusive LOCK.json
    evaluate  the single mask read; GO / NO-GO by the pre-registered rule
    audit     sklearn recomputation; BANK must reproduce CALM-02A's CALM-L1 numbers
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from exps.calm.stats import binned_hist, maps_from_patches, pooled_ap, weighted_ap
from exps.calm02 import gate as G
from exps.calm02 import run as R
from exps.calm02.data import ROOT, identities
from exps.calm02.score import Scorer, decode
from exps.task_p.p0_normal_support import distances, require
from exps.task_p.p1_data import native_mask, sha
from exps.task_p.p1_readout import image_metrics

OUT = ROOT / "results/hope_cad/ntm/stage0"
RUN = R.run_dir("mvtec", "CALM_L1", "B1")
RADIUS, Q = 4, 0.99
ARMS = ("BANK", "CTX", "NTM_FLOOR", "NTM_MIN")
CATS, ACQ, FINAL = G.CATS, G.ACQ, G.FINAL
SOURCES = ["exps/ntm/stage0.py", "exps/ntm/test_stage0.py", "exps/calm02/gate.py", "exps/calm02/score.py",
           "exps/calm02/run.py", "exps/calm/stats.py", "exps/task_p/p1_readout.py", "exps/task_p/p1_data.py",
           "exps/task_p/p0_normal_support.py", "agents/tasks/task_ntm_00_stage0.md"]
GO_MARGIN, F_MARGIN = 0.010, 0.005


def context_mask(radius=RADIUS, side=28):
    """True where patch u may serve as context for patch z (outside z's Chebyshev window)."""
    r, c = torch.arange(side * side) // side, torch.arange(side * side) % side
    return ((r[:, None] - r[None]).abs() > radius) | ((c[:, None] - c[None]).abs() > radius)


def context_distance(x, allowed):
    """Exact FP64 1-NN distance of every patch to the same image's allowed context."""
    D = distances(x, x).masked_fill(~allowed, torch.inf)
    return D.min(1).values


def combine(arm, d_m, d_x, tau_owner):
    if arm == "BANK":
        return d_m
    if arm == "CTX":
        return d_x
    if arm == "NTM_FLOOR":
        return np.minimum(d_m, np.maximum(d_x, tau_owner))
    if arm == "NTM_MIN":
        return np.minimum(d_m, d_x)
    raise ValueError(arm)


def bank(k):
    ck = torch.load(RUN / f"task_{k:02d}.pt", map_location="cpu", weights_only=False)
    occ = ck["occupied"]
    return decode(ck["state"])[occ], ck["owner"][occ], ck["ids"][occ]


@torch.no_grad()
def loo_bank_distances(feats, image_ids, rows, row_ids, device):
    """Per training image: 1-NN distance of its patches to the bank, excluding rows from that image."""
    B = rows.float().to(device); bn = B.square().sum(1); B64 = rows.double()
    src = (row_ids // 784).to(device)
    out = []
    for x, g in zip(feats, image_ids):
        q = x.float().to(device)
        d2 = q.square().sum(1)[:, None] + bn[None] - 2 * (q @ B.T)
        d2[:, src == int(g)] = torch.inf
        idx = d2.argmin(1).cpu()
        out.append((x.double() - B64[idx]).square().sum(1).sqrt())
    return torch.cat(out)


def taus(device):
    """tau_c = Q-quantile of leave-one-image-out bank distances of category c's training patches,
    against the bank at c's acquisition checkpoint. Training data only."""
    ids, off, out = identities("mvtec"), 0, {}
    for k, t in enumerate(ids["tasks"]):
        feats, _ = R.task_features("mvtec", t["task"], "train")
        rows, _, row_ids = bank(k)
        d = loo_bank_distances(feats, range(off, off + len(feats)), rows, row_ids, device)
        out[k] = {"task": t["task"], "tau": float(torch.quantile(d, Q)), "n_patches": int(d.numel())}
        off += len(feats)
        print("tau", t["task"], round(out[k]["tau"], 4), flush=True)
    return out


def prepare(device):
    OUT.mkdir(parents=True, exist_ok=False)
    co = G.cohort()
    gate_scores = np.load(G.GATE / "CALM_L1_scores.npz")
    T = taus(device)
    allowed = context_mask()
    arrays, parity = {}, {}
    for c in CATS:
        feats, rows = R.task_features("mvtec", c, "test")
        idx = {r["relative_path"]: i for i, r in enumerate(rows)}
        q = [feats[idx[r["relative_path"]]] for r in co[c]]
        arrays[f"dX_{c}"] = np.stack([context_distance(x, allowed).numpy() for x in q])
        for k in (ACQ[c], FINAL):
            brows, owners, _ = bank(k)
            sc = Scorer(brows, device)
            dm, own = [], []
            for x in q:
                d, i = sc.patch(x)
                dm.append(d.numpy()); own.append(owners[i].numpy())
            dm = np.stack(dm)
            parity[f"{c}_{k:02d}"] = float(np.abs(dm - gate_scores[f"{c}_{k:02d}"]).max())
            arrays[f"dM_{c}_{k:02d}"], arrays[f"owner_{c}_{k:02d}"] = dm, np.stack(own)
    require(max(parity.values()) <= 1e-12, f"BANK distances do not reproduce CALM-02A: {parity}")
    np.savez(OUT / "prepared.npz", **arrays)
    tau_vec = np.array([T[k]["tau"] for k in range(len(T))])
    active = {}
    for c in CATS:
        for k in (ACQ[c], FINAL):
            dm, own = arrays[f"dM_{c}_{k:02d}"], arrays[f"owner_{c}_{k:02d}"]
            active[f"{c}_{k:02d}"] = float((dm > tau_vec[own]).mean())   # label-free: share of patches where L2 may act
    R.write_json(OUT / "prepared.json", {"tau": T, "parity_max_abs_vs_calm02a": parity, "radius": RADIUS, "q": Q,
                                          "share_patches_bank_above_tau": active})
    print(json.dumps({"parity": parity, "active": active}, indent=1))


def lock():
    target = OUT / "LOCK.json"
    require(not target.exists(), "LOCK exists")
    files = {s: sha(ROOT / s) for s in SOURCES + G.TASKS}
    manifest = json.loads((R.FEATURES / "mvtec" / "MANIFEST.json").read_text())
    for p in [RUN / f"task_{k:02d}.pt" for k in range(FINAL + 1)] + [G.GATE / "CALM_L1_scores.npz",
              G.GATE / "GATE_EVAL.json", G.P1_PROTOCOL, OUT / "prepared.npz", OUT / "prepared.json",
              R.FEATURES / "mvtec" / "MANIFEST.json"]:
        files[str(p.relative_to(ROOT))] = sha(p)
    for t in manifest["tasks"]:
        for split in ("train", "test"):
            p = R.FEATURES / "mvtec" / manifest["tasks"][t][split]["file"]
            require(sha(p) == manifest["tasks"][t][split]["file_sha256"], f"feature cache changed: {p.name}")
    R.write_json(target, {"sha256": files, "radius": RADIUS, "q": Q, "seed": G.SEED, "reps": G.REPS,
                          "go_rule": {"ap_margin": GO_MARGIN, "forgetting_margin": F_MARGIN},
                          "locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def verify_lock():
    lk = json.loads((OUT / "LOCK.json").read_text())
    for name, digest in lk["sha256"].items():
        require(sha(ROOT / name) == digest, f"post-LOCK change: {name}")


def arm_scores(arm, P, tau_vec, c, k):
    dm, own = P[f"dM_{c}_{k:02d}"], P[f"owner_{c}_{k:02d}"]
    return combine(arm, dm, P[f"dX_{c}"], tau_vec[own])


def evaluate():
    verify_lock()
    target = OUT / "EVAL.json"
    require(not target.exists(), "evaluation exists: one-shot")
    co = G.cohort()
    P = np.load(OUT / "prepared.npz")
    prep = json.loads((OUT / "prepared.json").read_text())
    tau_vec = np.array([prep["tau"][str(k)]["tau"] for k in range(len(prep["tau"]))])
    root = ROOT / "data/mvtec"
    masks = {c: [native_mask(root, r) for r in co[c]] for c in CATS}               # the single mask read
    labels = {c: np.array([r["label"] for r in co[c]]) for c in CATS}
    hws = {c: [r["hw"] for r in co[c]] for c in CATS}
    W = G.weights(co, G.REPS, G.SEED)
    res, boot = {"cohort": "P1 development (consumed; exploratory)", "arms": {}}, {}
    for arm in ARMS:
        ap, hist, auc, fin_maps = {}, {}, {}, {}
        for c in CATS:
            for k in (ACQ[c], FINAL):
                maps = maps_from_patches(arm_scores(arm, P, tau_vec, c, k), hws[c])
                ap[(c, k)] = pooled_ap(maps, masks[c])["ap"]
                hist[(c, k)] = binned_hist(maps, masks[c])
                if k == FINAL:
                    fin_maps[c] = maps
            auc[c] = image_metrics(labels[c], arm_scores(arm, P, tau_vec, c, FINAL).max(1))["auroc"]
        all_maps = [m for c in CATS for m in fin_maps[c]]
        all_masks = [m for c in CATS for m in masks[c]]
        pooled = pooled_ap(all_maps, all_masks)["ap"]
        hist["pooled"] = binned_hist(all_maps, all_masks)
        forget = {c: ap[(c, ACQ[c])] - ap[(c, FINAL)] for c in CATS}
        res["arms"][arm] = {"final_ap": {c: ap[(c, FINAL)] for c in CATS}, "acquisition_ap": {c: ap[(c, ACQ[c])] for c in CATS},
                            "macro_final_ap": float(np.mean([ap[(c, FINAL)] for c in CATS])),
                            "forgetting": forget, "macro_forgetting": float(np.mean(list(forget.values()))),
                            "pooled_final_ap": pooled, "image_auroc": auc, "macro_image_auroc": float(np.mean(list(auc.values())))}
        b = {"macro_final_ap": [], "macro_forgetting": [], "pooled_final_ap": [], "macro_image_auroc": []}
        for w in W:
            fin = [weighted_ap(hist[(c, FINAL)], w[c]) for c in CATS]
            acq = [weighted_ap(hist[(c, ACQ[c])], w[c]) for c in CATS]
            b["macro_final_ap"].append(np.mean(fin))
            b["macro_forgetting"].append(np.mean(np.array(acq) - np.array(fin)))
            b["pooled_final_ap"].append(weighted_ap(hist["pooled"], np.concatenate([w[c] for c in CATS])))
            b["macro_image_auroc"].append(np.mean([image_metrics(labels[c], arm_scores(arm, P, tau_vec, c, FINAL).max(1), w[c])["auroc"]
                                                   for c in CATS]))
        boot[arm] = {k: np.array(v) for k, v in b.items()}
        print(arm, "macro final AP %.4f forgetting %.4f pooled %.4f I-AUROC %.4f" % (
            res["arms"][arm]["macro_final_ap"], res["arms"][arm]["macro_forgetting"], pooled, res["arms"][arm]["macro_image_auroc"]), flush=True)
    effects = {}
    for a, b_ in (("NTM_FLOOR", "BANK"), ("NTM_FLOOR", "CTX"), ("NTM_MIN", "BANK"), ("CTX", "BANK"), ("NTM_MIN", "NTM_FLOOR")):
        effects[f"{a}-{b_}"] = {key: {"d": res["arms"][a][key] - res["arms"][b_][key],
                                      "ci95": np.percentile(boot[a][key] - boot[b_][key], [2.5, 97.5]).tolist()}
                                for key in ("macro_final_ap", "macro_forgetting", "pooled_final_ap", "macro_image_auroc")}
    e = effects["NTM_FLOOR-BANK"]
    c1 = e["macro_final_ap"]["d"] >= GO_MARGIN and e["macro_final_ap"]["ci95"][0] > 0
    c2 = not (e["macro_forgetting"]["d"] >= F_MARGIN and e["macro_forgetting"]["ci95"][0] > 0)
    c3 = effects["NTM_FLOOR-CTX"]["macro_final_ap"]["d"] > 0
    res.update(effects=effects, go_conditions={"ap_gain": c1, "forgetting_not_worse": c2, "beats_context_only": c3},
               decision="GO" if (c1 and c2 and c3) else "NO-GO")
    R.write_json(target, res)
    print(json.dumps({"decision": res["decision"], "conditions": res["go_conditions"]}, indent=1))


def audit():
    from sklearn.metrics import average_precision_score, roc_auc_score
    from exps.task_p.p1_readout import native_map
    verify_lock()
    ev = json.loads((OUT / "EVAL.json").read_text())
    gate = json.loads((G.GATE / "GATE_EVAL.json").read_text())["arms"]["CALM_L1"]
    co = G.cohort()
    P = np.load(OUT / "prepared.npz")
    prep = json.loads((OUT / "prepared.json").read_text())
    tau_vec = np.array([prep["tau"][str(k)]["tau"] for k in range(len(prep["tau"]))])
    root = ROOT / "data/mvtec"
    masks = {c: [native_mask(root, r) for r in co[c]] for c in CATS}
    labels = {c: np.array([r["label"] for r in co[c]]) for c in CATS}
    worst_ap = worst_auc = 0.0
    for arm in ARMS:
        for c in CATS:
            for k, key in ((ACQ[c], "acquisition_ap"), (FINAL, "final_ap")):
                s = arm_scores(arm, P, tau_vec, c, k)
                flat = np.concatenate([native_map(torch.from_numpy(np.ascontiguousarray(p)), r["hw"]).ravel() for p, r in zip(s, co[c])])
                y = np.concatenate([m.ravel() for m in masks[c]])
                worst_ap = max(worst_ap, abs(average_precision_score(y, flat) - ev["arms"][arm][key][c]))
            worst_auc = max(worst_auc, abs(roc_auc_score(labels[c], arm_scores(arm, P, tau_vec, c, FINAL).max(1)) - ev["arms"][arm]["image_auroc"][c]))
    b = ev["arms"]["BANK"]
    repro = max([abs(b["final_ap"][c] - gate["final_ap"][c]) for c in CATS] + [abs(b["acquisition_ap"][c] - gate["acquisition_ap"][c]) for c in CATS]
                + [abs(b["pooled_final_ap"] - gate["pooled_final_ap"]), abs(b["macro_image_auroc"] - gate["macro_image_auroc"])])
    rep = {"max_abs_ap_vs_sklearn": worst_ap, "max_abs_auroc_vs_sklearn": worst_auc, "bank_reproduces_calm02a_max_abs": repro}
    rep["pass"] = worst_ap < 1e-9 and worst_auc < 1e-12 and repro < 1e-12
    R.write_json(OUT / "AUDIT.json", rep)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("prepare", "lock", "evaluate", "audit"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    {"prepare": lambda: prepare(a.device), "lock": lock, "evaluate": evaluate, "audit": audit}[a.stage]()


if __name__ == "__main__":
    main()
