# exps/nlcad/forensic_syn.py
"""NLCAD-04 Stage B: label-free synthetic-anomaly forensic on existing checkpoints (FP32, TF32 off, CPU or GPU; forward +
parameter gradients of fixed objectives; no training, no real defect label, no development image).

For every arm (ID, N, N1, F1, RND; seed 2711) and stage transition s-1 -> s (s = 7..15; the start of s = 7 is the shared
ID task-6 state), on the items a stage-s slow event may use (bank.allowed: the 16 current-task hosts and the stage's
Replay-24 members, both variants, both generators):

  - per item: synthetic-positive (coverage >= 0.5) and negative (coverage == 0) score means, item AUROC, the
    deterministic all-negatives ranking loss, and the 28x28 score map (float16);
  - per (generator, source) group: J = mean item ranking loss; its gradient at both ends; per-block first-order credits
    <grad_b J, Delta_b> with Delta = theta_s - theta_{s-1} (trapezoid predicted change vs actual change); per-block
    gradient norms and the Q2-vs-Q3 gradient cosine;
  - the normal reconstruction objective (global cosine) on 8 current + 8 Replay-24 normal training identities, for
    the signal-scale comparison.

  python -m exps.nlcad.forensic_syn run [--threads 11] [--device cuda]   # one JSON per (arm, stage) cell; resumable
  python -m exps.nlcad.forensic_syn summary                # results/hope_cad/nlcad/FORENSIC_SYN_SUMMARY.json
  python -m exps.nlcad.forensic_syn deep                   # results/hope_cad/nlcad/FORENSIC_SYN_DEEP.json (maps)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import readout as R
from exps.dinomaly.common import require, sha
from exps.nlcad import bank as BK
from exps.nlcad.controller import BLOCKS, block_of
from exps.nlcad.objectives_s import ranking_terms
from exps.rh01 import numerics as NUM
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

OUT = D.ROOT / "results/hope_cad/nlcad"
CELLS = OUT / "forensic_syn"
SEED, START, FINAL, TAU = 2711, 6, 15, 0.05
ARMS = {"ID": None, "N": "N_per_task_seed2711_from6", "N1": "N1_per_task_seed2711_from6",
        "F1": "F1_per_task_seed2711_from6", "RND": "RND_per_task_seed2711_from6"}
GENS, SOURCES = ("Q2", "Q3"), ("current", "replay")


def checkpoint(arm, s):
    if arm == "ID" or s == START:
        rec = D.read(C.OUT / f"runs/R15_seed{SEED}/STAGE{s}.json")
    else:
        rec = D.read(OUT / f"runs/{ARMS[arm]}/STAGE{s}.json")
    path = D.ROOT / rec["checkpoint"]
    require(sha(path) == rec["checkpoint_sha256"], f"checkpoint changed: {path}")
    return torch.load(path, weights_only=False, map_location="cpu")["state"], rec["checkpoint_sha256"]


def items_at(bank, hosts, n, s):
    allowed, cur = BK.allowed(hosts, SEED, n, s), C.STREAM[s - 1]
    return {g: {src: [it for it in bank[g] if it["host"] in allowed and (it["host"][0] == cur) == (src == "current")]
                for src in SOURCES} for g in GENS}


def normal_queries(n, s):
    """8 current + 8 Replay-24 normal training identities, fixed per stage (seed-independent of the arms)."""
    cur, done = C.STREAM[s - 1], list(C.STREAM[:s - 1])
    g = torch.Generator().manual_seed(90000 + s)
    ci = torch.randperm(n[cur], generator=g)[:8].tolist()
    idx = C.buffer_index(SEED, n, done)
    keys = [(d, i) for d in done for i in idx[d]]
    rk = [keys[j] for j in torch.randperm(len(keys), generator=g)[:8].tolist()]
    feats = torch.stack([C.load_features(cur)[i] for i in ci] + [C.load_features(d)[i] for d, i in rk])
    return feats, {"current": [[cur, i] for i in ci], "replay": [list(k) for k in rk]}


def item_auroc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def block_grads(dec):
    out = {b: [] for b in BLOCKS}
    for name, p in dec.named_parameters():
        out[block_of(name)].append(p.grad.detach().to("cpu", copy=True) if p.grad is not None else torch.zeros(p.shape))
    return out


def dot(a, b):
    return {k: float(sum((x.double() * y.double()).sum() for x, y in zip(a[k], b[k]))) for k in BLOCKS}


def synthetic_group(dec, items):
    """Item statistics, maps and the gradient of J = mean ranking loss over items with a positive patch."""
    dec.zero_grad(set_to_none=True)
    dev = next(dec.parameters()).device
    keep = [it for it in items if bool((it["coverage"] >= 0.5).any())]
    rows, maps, total = [], [], 0.
    for k in range(0, len(items), 8):
        chunk = items[k:k + 8]
        en, de = dec(*torch.stack([it["features"] for it in chunk]).float().to(dev).unbind(1))
        s = R.grid([e.float() for e in en], [d.float() for d in de])
        losses = []
        for it, m in zip(chunk, s):
            t = ranking_terms(m, it["coverage"].to(dev), TAU)
            mp = m.detach().cpu()
            maps.append(mp.half())
            row = {"host": list(it["host"]), "variant": it["variant"], "n_pos": int((it["coverage"] >= 0.5).sum())}
            if t is not None:
                losses.append(t["loss"])
                pos, neg = mp[it["coverage"] >= 0.5].numpy(), mp[it["coverage"] == 0].numpy()
                row |= {"loss": float(t["loss"]), "pos_mean": float(pos.mean()), "neg_mean": float(neg.mean()),
                        "neg_p99": float(np.quantile(neg, 0.99)), "auroc": item_auroc(pos, neg)}
            rows.append(row)
        if losses:
            loss = torch.stack(losses).sum() / len(keep)
            loss.backward()
            total += float(loss)
    pos = np.concatenate([m.float().numpy()[it["coverage"].numpy() >= 0.5] for it, m in zip(items, maps)])
    neg = np.concatenate([m.float().numpy()[it["coverage"].numpy() == 0] for it, m in zip(items, maps)])
    ap = float(average_precision_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))
    return {"J": total, "used": len(keep), "skipped": len(items) - len(keep), "patch_ap": ap, "items": rows}, \
        block_grads(dec), torch.stack(maps)


def normal_group(dec, x):
    dec.zero_grad(set_to_none=True)
    total = 0.
    dev = next(dec.parameters()).device
    for xb in x.split(8):
        en, de = dec(xb[:, 0].float().to(dev), xb[:, 1].float().to(dev))
        loss = O.global_cosine([e.float() for e in en], [d.float() for d in de]) * len(xb) / len(x)
        loss.backward()
        total += float(loss)
    return total, block_grads(dec)


def endpoint(dec, sd, groups, normal_x):
    dec.load_state_dict(sd)
    res, grads, maps = {}, {}, {}
    for g in GENS:
        for src in SOURCES:
            res[f"{g}/{src}"], grads[f"{g}/{src}"], maps[f"{g}/{src}"] = synthetic_group(dec, groups[g][src])
    res["normal"], grads["normal"] = {}, None
    res["normal"]["J"], grads["normal"] = normal_group(dec, normal_x)
    return res, grads, maps


def norms(g):
    return {b: float(torch.sqrt(sum((x.double() ** 2).sum() for x in g[b]))) for b in BLOCKS}


def run_cells(threads, device):
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    NUM.use(True)
    proto = RB.verify_lock()
    n = R1.n_train(proto)
    bank, hosts = BK.load_bank(SEED)
    CELLS.mkdir(parents=True, exist_ok=True)
    dec = O.Decoder(768, 12).eval().to(device)
    for p in dec.parameters():
        p.requires_grad_(True)
    cache = {}
    for s in range(START + 1, FINAL + 1):
        groups = items_at(bank, hosts, n, s)
        normal_x, normal_ids = normal_queries(n, s)
        for arm in ARMS:
            target = CELLS / f"CELL_{arm}_{s}.json"
            if target.exists():
                continue
            t0 = time.perf_counter()
            sd0, h0 = checkpoint(arm, s - 1)
            sd1, h1 = checkpoint(arm, s)
            if (h0, s) in cache:
                r0, g0, m0 = cache[(h0, s)]
            else:
                r0, g0, m0 = endpoint(dec, sd0, groups, normal_x)
                cache = {(h0, s): (r0, g0, m0)} if s == START + 1 else {}
            r1, g1, m1 = endpoint(dec, sd1, groups, normal_x)
            delta = {b: [] for b in BLOCKS}
            for name, _ in dec.named_parameters():
                delta[block_of(name)].append(sd1[name].float() - sd0[name].float())
            dn = {b: float(torch.sqrt(sum((x.double() ** 2).sum() for x in delta[b]))) for b in BLOCKS}
            cell = {"arm": arm, "stage": s, "checkpoints": [h0, h1], "delta_norm": dn, "normal_ids": normal_ids,
                    "groups": {}}
            for key in list(r1):
                c0, c1 = dot(g0[key], delta), dot(g1[key], delta)
                cell["groups"][key] = {
                    "start": r0[key], "end": r1[key], "actual_dJ": r1[key]["J"] - r0[key]["J"],
                    "credit_start": c0, "credit_end": c1,
                    "predicted_dJ_trapezoid": 0.5 * (sum(c0.values()) + sum(c1.values())),
                    "grad_norm_start": norms(g0[key]), "grad_norm_end": norms(g1[key])}
            for src in SOURCES:
                a, b = g1[f"Q2/{src}"], g1[f"Q3/{src}"]
                ab, aa, bb = dot(a, b), dot(a, a), dot(b, b)
                cell[f"grad_cos_Q2_Q3/{src}"] = {k: ab[k] / np.sqrt(aa[k] * bb[k] + 1e-300) for k in BLOCKS}
                cell[f"grad_cos_Q2_Q3/{src}"]["global"] = sum(ab.values()) / np.sqrt(sum(aa.values()) * sum(bb.values()))
            cell["seconds"], cell["device"] = time.perf_counter() - t0, str(device)
            torch.save({"start": m0, "end": m1}, CELLS / f"MAPS_{arm}_{s}.pt")
            D.write_json(target, cell)
            print("cell", arm, s, f"{cell['seconds']:.0f}s",
                  {k: (round(v["start"]["patch_ap"], 3), round(v["end"]["patch_ap"], 3))
                   for k, v in cell["groups"].items() if k != "normal"}, flush=True)


# ----------------------------------------------------------------------------------------------- summary
def ccos(u, v):
    u, v = np.asarray(u) - np.mean(u), np.asarray(v) - np.mean(v)
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300))


def summary():
    cells = {(a, s): D.read(CELLS / f"CELL_{a}_{s}.json") for a in ARMS for s in range(START + 1, FINAL + 1)}
    keys = [f"{g}/{src}" for g in GENS for src in SOURCES]
    res = {"per_cell": {}, "vs_ID": {}, "agreement": {}, "credit_validity": {}, "scale": {}, "per_category": {}}
    for (a, s), c in cells.items():
        res["per_cell"][f"{a}/{s}"] = {k: {"patch_ap": c["groups"][k]["end"]["patch_ap"],
                                           "J": c["groups"][k]["end"]["J"],
                                           "gap": float(np.mean([r["pos_mean"] - r["neg_mean"]
                                                                 for r in c["groups"][k]["end"]["items"] if "loss" in r])),
                                           "pos": float(np.mean([r["pos_mean"] for r in c["groups"][k]["end"]["items"] if "loss" in r])),
                                           "neg": float(np.mean([r["neg_mean"] for r in c["groups"][k]["end"]["items"] if "loss" in r])),
                                           "auroc": float(np.mean([r["auroc"] for r in c["groups"][k]["end"]["items"] if "loss" in r])),
                                           "used": c["groups"][k]["end"]["used"], "skipped": c["groups"][k]["end"]["skipped"]}
                                       for k in keys}
    for (a, s), c in cells.items():
        if a == "ID":
            continue
        ref = res["per_cell"][f"ID/{s}"]
        cur = res["per_cell"][f"{a}/{s}"]
        res["vs_ID"][f"{a}/{s}"] = {k: {"d_patch_ap": cur[k]["patch_ap"] - ref[k]["patch_ap"],
                                        "gap_ratio": cur[k]["gap"] / ref[k]["gap"],
                                        "d_pos": cur[k]["pos"] - ref[k]["pos"], "d_neg": cur[k]["neg"] - ref[k]["neg"]}
                                    for k in keys}
    for src in SOURCES:
        x = [res["vs_ID"][k][f"Q2/{src}"]["d_patch_ap"] for k in res["vs_ID"]]
        y = [res["vs_ID"][k][f"Q3/{src}"]["d_patch_ap"] for k in res["vs_ID"]]
        tx = [cells[k]["groups"][f"Q2/{src}"]["actual_dJ"] for k in cells]
        ty = [cells[k]["groups"][f"Q3/{src}"]["actual_dJ"] for k in cells]
        cq = [ccos([0.5 * (cells[k]["groups"][f"Q2/{src}"]["credit_start"][b] + cells[k]["groups"][f"Q2/{src}"]["credit_end"][b]) for b in BLOCKS],
                   [0.5 * (cells[k]["groups"][f"Q3/{src}"]["credit_start"][b] + cells[k]["groups"][f"Q3/{src}"]["credit_end"][b]) for b in BLOCKS])
              for k in cells]
        res["agreement"][src] = {
            "spearman_dAP_vs_ID_Q2_Q3": list(map(float, spearmanr(x, y))), "n_cells_vs_ID": len(x),
            "transition_dJ_sign_agreement": float(np.mean(np.sign(tx) == np.sign(ty))),
            "spearman_transition_dJ": list(map(float, spearmanr(tx, ty))),
            "credit_centred_cosine_Q2_Q3": {"median": float(np.median(cq)), "min": float(np.min(cq)),
                                            "frac_positive": float(np.mean(np.array(cq) > 0))},
            "grad_cos_global_end": {"median": float(np.median([cells[k][f"grad_cos_Q2_Q3/{src}"]["global"] for k in cells])),
                                    "min": float(np.min([cells[k][f"grad_cos_Q2_Q3/{src}"]["global"] for k in cells]))}}
    for k in keys + ["normal"]:
        pred = np.array([cells[c]["groups"][k]["predicted_dJ_trapezoid"] for c in cells])
        act = np.array([cells[c]["groups"][k]["actual_dJ"] for c in cells])
        res["credit_validity"][k] = {"sign_agreement": float(np.mean(np.sign(pred) == np.sign(act))),
                                     "spearman": list(map(float, spearmanr(pred, act))),
                                     "median_rel_error": float(np.median(np.abs(pred - act) / (np.abs(act) + 1e-12)))}
    for k in keys:
        ratio = [cells[c]["groups"][k]["grad_norm_end"][b] / cells[c]["groups"]["normal"]["grad_norm_end"][b]
                 for c in cells for b in BLOCKS]
        cred = [abs(0.5 * (cells[c]["groups"][k]["credit_start"][b] + cells[c]["groups"][k]["credit_end"][b])) /
                (abs(0.5 * (cells[c]["groups"]["normal"]["credit_start"][b] + cells[c]["groups"]["normal"]["credit_end"][b])) + 1e-300)
                for c in cells for b in BLOCKS]
        res["scale"][k] = {"grad_norm_ratio_to_normal_median": float(np.median(ratio)),
                           "abs_credit_ratio_to_normal_median": float(np.median(cred))}
    for (a, s), c in cells.items():
        if a == "ID" or s != FINAL:
            continue
        ref = {tuple(r["host"]) + (r["variant"],): r for r in cells[("ID", s)]["groups"]["Q2/replay"]["end"]["items"]}
        for g in GENS:
            ref = {tuple(r["host"]) + (r["variant"],): r for r in cells[("ID", s)]["groups"][f"{g}/replay"]["end"]["items"]}
            per = {}
            for r in c["groups"][f"{g}/replay"]["end"]["items"]:
                q = ref[tuple(r["host"]) + (r["variant"],)]
                if "loss" in r and "loss" in q:
                    per.setdefault(r["host"][0], []).append((r["pos_mean"] - r["neg_mean"]) / (q["pos_mean"] - q["neg_mean"]))
            res["per_category"][f"{a}/{g}"] = {cat: float(np.median(v)) for cat, v in per.items()}
    D.write_json(OUT / "FORENSIC_SYN_SUMMARY.json", res)
    return res


def _med(v):
    v = np.asarray(v, dtype=float)
    return float(np.nanmedian(v)) if np.isfinite(v).any() else None


def _stats(rows):
    rows = [r for r in rows if "loss" in r]
    return {k: float(np.mean([r[k] for r in rows])) for k in ("pos_mean", "neg_mean", "neg_p99", "auroc")} | \
        {"gap": float(np.mean([r["pos_mean"] - r["neg_mean"] for r in rows]))}


def deep():
    """Map-level forensic (no new forward pass): every transition's change of positive / negative / gap / p99 / item
    AUROC per group; spatial rank agreement of each arm's end maps with ID's; Q2 vs Q3 response of the same transition
    on the same host and variant (difference maps on patches negative in both items, and positive-mean changes);
    per-category replay gap of every arm at every stage (no selection)."""
    n = R1.n_train(RB.verify_lock())
    bank, hosts = BK.load_bank(SEED)
    keys = [f"{g}/{src}" for g in GENS for src in SOURCES]
    res = {"transition": {}, "rank_vs_ID": {}, "q2q3_same_host": {}, "per_category_replay_gap": {}}
    for s in range(START + 1, FINAL + 1):
        groups = items_at(bank, hosts, n, s)
        cells = {a: D.read(CELLS / f"CELL_{a}_{s}.json") for a in ARMS}
        maps = {a: torch.load(CELLS / f"MAPS_{a}_{s}.pt") for a in ARMS}
        for a in ARMS:
            c = cells[a]
            res["transition"][f"{a}/{s}"] = {k: {"start": _stats(c["groups"][k]["start"]["items"]),
                                                 "end": _stats(c["groups"][k]["end"]["items"]),
                                                 "patch_ap": [c["groups"][k]["start"]["patch_ap"], c["groups"][k]["end"]["patch_ap"]]}
                                             for k in keys}
            if a != "ID":
                res["rank_vs_ID"][f"{a}/{s}"] = {
                    k: _med([spearmanr(maps[a]["end"][k][j].float().flatten(), maps["ID"]["end"][k][j].float().flatten())[0]
                             for j in range(len(maps[a]["end"][k]))]) for k in keys}
            out = {}
            for src in SOURCES:
                q2 = {(it["host"], it["variant"]): (j, it["coverage"]) for j, it in enumerate(groups["Q2"][src])}
                r2 = {(tuple(r["host"]), r["variant"]): r for r in c["groups"][f"Q2/{src}"]["end"]["items"]}
                r2s = {(tuple(r["host"]), r["variant"]): r for r in c["groups"][f"Q2/{src}"]["start"]["items"]}
                r3 = {(tuple(r["host"]), r["variant"]): r for r in c["groups"][f"Q3/{src}"]["end"]["items"]}
                r3s = {(tuple(r["host"]), r["variant"]): r for r in c["groups"][f"Q3/{src}"]["start"]["items"]}
                rho_delta, rho_end, dpos2, dpos3, dgap2, dgap3 = [], [], [], [], [], []
                for j3, it in enumerate(groups["Q3"][src]):
                    key = (it["host"], it["variant"])
                    j2, c2 = q2[key]
                    m = ((it["coverage"] == 0) & (c2 == 0)).flatten()
                    get = lambda g, j, e: maps[a][e][f"{g}/{src}"][j].float().flatten()[m]
                    rho_delta.append(spearmanr(get("Q2", j2, "end") - get("Q2", j2, "start"),
                                               get("Q3", j3, "end") - get("Q3", j3, "start"))[0])
                    rho_end.append(spearmanr(get("Q2", j2, "end"), get("Q3", j3, "end"))[0])
                    if all("loss" in r[key] for r in (r2, r2s, r3, r3s)):
                        dpos2.append(r2[key]["pos_mean"] - r2s[key]["pos_mean"])
                        dpos3.append(r3[key]["pos_mean"] - r3s[key]["pos_mean"])
                        dgap2.append((r2[key]["pos_mean"] - r2[key]["neg_mean"]) - (r2s[key]["pos_mean"] - r2s[key]["neg_mean"]))
                        dgap3.append((r3[key]["pos_mean"] - r3[key]["neg_mean"]) - (r3s[key]["pos_mean"] - r3s[key]["neg_mean"]))
                out[src] = {"negative_delta_map_spearman_median": _med(rho_delta),
                            "negative_end_map_spearman_median": _med(rho_end),
                            "item_dpos_spearman_Q2_Q3": float(spearmanr(dpos2, dpos3)[0]),
                            "item_dgap_spearman_Q2_Q3": float(spearmanr(dgap2, dgap3)[0]), "n_pairs": len(dgap2)}
            res["q2q3_same_host"][f"{a}/{s}"] = out
            for g in GENS:
                per = {}
                for r in c["groups"][f"{g}/replay"]["end"]["items"]:
                    if "loss" in r:
                        per.setdefault(r["host"][0], []).append(r["pos_mean"] - r["neg_mean"])
                res["per_category_replay_gap"][f"{a}/{g}/{s}"] = {cat: float(np.median(v)) for cat, v in per.items()}
        print("deep", s, flush=True)
    D.write_json(OUT / "FORENSIC_SYN_DEEP.json", res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run", "summary", "deep"))
    ap.add_argument("--threads", type=int, default=11)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    {"run": lambda: run_cells(a.threads, a.device), "summary": summary, "deep": deep}[a.stage]()


if __name__ == "__main__":
    main()
