"""NL-RET-01A phases 0 and 2 (inference only; no training, no defect data).

  audit        G0: checkpoint inventory + SHA, Replay-24 occupancy (verified against the R15 run records), cohort
               custody, coverage; copies decoder states to an ext4 cache (SHA-verified) for fast repeated loads
  parity       label-free: this pipeline's held-out image scores vs the image_score_RD logged by R15 training
  extract --seed S   per category (outer) x every existing state with s >= a(c) (inner): per-image normal statistics
               for ALL train identities of c (members and nonmembers), 64x64 maps of ever-members, Q2 synthetic grids
  grads --seed S     fresh no-update gradient diagnostics + parameter drift (lawful inputs only)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from exps.cadfinal import common as K
from exps.dinomaly import common as D
from exps.dinomaly.common import require, sha
from exps.nl_ret_01a import common as C

EXT = C.OUT / "extract"
STAT_COLS = ["img", "q50", "q90", "q95", "q99", "q999", "grid_mean", "gcos"]
BS = 16


def _setup():
    from exps.dinomaly import run as DR
    from exps.rh01 import numerics as NUM
    DR.CAP_GIB = 4.5
    DR.configure()
    NUM.use(True)


def _decoder(arm, seed, s):
    from exps.cadfinal import dino as DN
    p = C.local_ckpt(arm, seed, s)
    dec = DN.O_decoder()
    dec.load_state_dict(torch.load(p, weights_only=False)["state"])
    return dec.eval()


# ---- G0 -----------------------------------------------------------------------------------------------------

def audit():
    C.OUT.mkdir(parents=True, exist_ok=True)
    proto, n = C.proto_and_n()
    C.CK.mkdir(parents=True, exist_ok=True)
    inv, bad = [], []
    for seed in C.SEEDS:
        for arm in C.ARMS:
            for s in C.states(arm):
                p, h, prov, rec = C.checkpoint(arm, seed, s)
                ok = sha(p) == h
                loc = C.local_ckpt(arm, seed, s)
                if ok and not (loc.exists() and sha(loc) == h):
                    shutil.copyfile(p, loc)
                    ok = sha(loc) == h
                if not ok:
                    bad.append(str(p))
                st = torch.load(loc, weights_only=False, map_location="cpu")
                inv.append({"arm": arm, "seed": seed, "stage": s, "owner": C.owner(arm, s), "category": C.STREAM[s - 1],
                            "checkpoint": str(p.relative_to(C.ROOT)), "sha256": h, "verified": ok, "provenance": prov,
                            "checkpoint_keys": sorted(st.keys()), "optimizer_state": "optimizer" in st,
                            "logged": sorted(k for k in rec if k not in ("heldout_normal", "replay"))})
                del st
    # Replay occupancy, verified against the R15 records (STAGE{s+1}.replay = members at end of s)
    occ, mism = [], []
    for seed in C.SEEDS:
        prev = None
        for s in range(1, 16):
            m = C.members(seed, n, s)
            if s < 15:
                rec = D.read(C.ROOT / f"results/hope_cad/rh01b/runs/R15_seed{seed}/STAGE{s + 1}.json")["replay"]
                want = [proto["train"][c][i]["relative_path"] for c in m for i in m[c]]
                if rec["identities"] != want:
                    mism.append((seed, s))
            for c in C.STREAM[:s]:
                idx = m[c]
                was = prev.get(c, []) if prev else []
                occ.append({"seed": seed, "end_of_stage": s, "category": c, "a_c": C.a_of(c), "n_members": len(idx),
                            "n_unique": len(set(idx)), "duplicates": len(idx) - len(set(idx)),
                            "removed_since_prev": len(set(was) - set(idx)), "added_since_prev": len(set(idx) - set(was)),
                            "subset_of_prev": set(idx) <= set(was) if was else None,
                            "status": "UNOBSERVABLE" if not idx else ("LOW_N" if len(idx) < 3 else "OK"),
                            "n_train": n[c], "n_nonmember_fixed": len(C.nonmembers(seed, n, c))})
            prev = m
    with open(C.OUT / "REPLAY_OCCUPANCY.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(occ[0])); w.writeheader(); w.writerows(occ)
    cust = {"pools": C.__doc__, "capacity_semantics": "24 images in TOTAL over all completed tasks: 24 // k per "
            "completed task plus the remainder to the first completed tasks in a seed-fixed priority order "
            "(exps.rh01.common.allocation); members are prefixes of a seed-fixed per-class permutation, so later "
            "buffers are subsets of earlier ones; FS_B uses the identical buffer_index call (exps/nl_align_01/g1.py "
            "train(), lines ~411-413)",
            "per_seed": {}}
    for seed in C.SEEDS:
        cust["per_seed"][seed] = {c: {"n_train": n[c], "ever_members": C.ever_members(seed, n, c),
                                      "nonmembers_fixed": len(C.nonmembers(seed, n, c)),
                                      "train_paths_sha256": hashlib.sha256(json.dumps(
                                          [r["relative_path"] for r in proto["train"][c]]).encode()).hexdigest()}
                                  for c in C.STREAM}
    cust["exposed_defect_eval"] = "only results/hope_cad/nl_ret_00b/STAGE_TRAJECTORY.json (sealed real AP per stage)"
    cust["eval_only_note"] = "nonmembers were trained on at their acquisition stage (seen-but-not-replayed normals)"
    K.write_json(C.OUT / "COHORT_CUSTODY.json", cust)
    cov = {"rows_per_arm": {a: len(C.eligible(a)) for a in C.ARMS},
           "rows_per_arm_distinct_from_ID": {a: sum(not r["shared_with_ID"] for r in C.eligible(a)) for a in C.ARMS},
           "states": {a: C.states(a) for a in C.ARMS}, "inventory": inv, "bad_checkpoints": bad,
           "replay_record_mismatches": mism, "optimizer_moments": "NOT_OBSERVABLE_FROM_ARCHIVE" if not any(
               x["optimizer_state"] for x in inv) else "present", "utc": K.now()}
    K.write_json(C.OUT / "FULL_STREAM_COVERAGE.json", cov)
    print("audit", len(inv), "states; bad", bad, "replay mismatches", mism)
    require(not bad and not mism, "integrity")


@torch.no_grad()
def parity():
    from exps.dinomaly import readout as R
    from exps.dinomaly import run as DR
    _setup()
    out, worst = [], 0.0
    for s in (1, 4, 6, 10, 15):
        dec = _decoder("ID", 2711, s)
        rec = D.read(C.ROOT / f"results/hope_cad/rh01b/runs/R15_seed2711/STAGE{s}.json")["heldout_normal"]
        for c in rec:
            f = DR.load_split("heldout", c)
            im = []
            for xb in f.split(10):
                xb = xb.cuda()
                en, de = dec(xb[:, 0], xb[:, 1])
                im += R.dinomaly(en, de)[1].tolist()
            d = float(np.max(np.abs(np.array(im) - np.array(rec[c]["image_score_RD"]))))
            worst = max(worst, d)
            out.append({"stage": s, "category": c, "max_abs": d, "n": len(im)})
        del dec
    res = {"cells": out, "worst": worst, "tolerance": 1e-4, "pass": worst <= 1e-4, "utc": K.now()}
    K.write_json(C.OUT / "PARITY.json", res)
    print("parity", worst, res["pass"])
    require(res["pass"], "parity")


def _img_stats(en, de):
    from exps.dinomaly import official as O
    from exps.dinomaly import readout as R
    m, img = R.dinomaly(en, de)                                  # [B,256,256], [B]
    flat = m.flatten(1)
    q = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.95, 0.99, 0.999], device=flat.device), dim=1).T
    g = R.grid(en, de).mean(dim=(1, 2))
    gc = torch.stack([(1 - F.cosine_similarity(e.flatten(1), d.flatten(1))) for e, d in zip(en, de)], 1).mean(1)
    return torch.cat([img[:, None], q, g[:, None], gc[:, None]], 1), m


def _bank(seed):
    from exps.nl_align_02 import m3 as M3
    items, hosts = M3.load_bank_15(seed)
    return items["Q2"], hosts


@torch.no_grad()
def extract(seed):
    from exps.dinomaly import readout as R
    from exps.rh01 import common as RC
    _setup()
    proto, n = C.proto_and_n()
    bank, hosts = _bank(seed)
    out_dir = EXT / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in C.STREAM:
        path = out_dir / f"{c}.pt"
        if path.exists():
            continue
        t0 = time.time()
        a = C.a_of(c)
        feats = RC.load_features(c, mmap=False)                    # [n_c, 2, 789, 768] FP32 (SHA-verified)
        require(len(feats) == n[c], "cache size")
        ever = C.ever_members(seed, n, c)
        keys = sorted(k for k in bank if k[0] == c)
        sf = torch.stack([bank[k][0] for k in keys]).float()
        cov = torch.stack([bank[k][1] for k in keys]).float()
        rec = {"category": c, "a": a, "paths": [r["relative_path"] for r in proto["train"][c]], "ever": ever,
               "syn_keys": keys, "syn_cov": cov.half(), "hosts": hosts[c], "stats": {}, "maps64": {}, "syn_grid": {},
               "cols": STAT_COLS}
        for arm in C.ARMS:
            for s in C.states(arm):
                if s < a or (arm == "FS_B" and C.owner(arm, s) == "ID"):
                    continue
                dec = _decoder(arm, seed, s)
                st, mp = [], []
                for i in range(0, len(feats), BS):
                    xb = feats[i:i + BS].cuda(non_blocking=True)
                    en, de = dec(xb[:, 0], xb[:, 1])
                    v, m = _img_stats(en, de)
                    st.append(v.cpu())
                    sel = [j - i for j in ever if i <= j < i + BS]
                    if sel:
                        mp.append(F.avg_pool2d(m[sel][:, None], 4)[:, 0].half().cpu())
                g = []
                for i in range(0, len(sf), BS):
                    xb = sf[i:i + BS].cuda()
                    en, de = dec(xb[:, 0], xb[:, 1])
                    g.append(R.grid(en, de).half().cpu())
                key = f"{C.owner(arm, s)}|{s}"
                rec["stats"][key] = torch.cat(st)
                rec["maps64"][key] = torch.cat(mp) if mp else None
                rec["syn_grid"][key] = torch.cat(g)
                del dec
                torch.cuda.empty_cache()
        torch.save(rec, path)
        print(f"extract seed {seed} {c} n={len(feats)} states={len(rec['stats'])} syn={len(keys)} "
              f"{time.time() - t0:.0f}s", flush=True)
        del feats, sf, cov, rec


def grads(seed):
    """Fresh, no-update, deterministic gradient diagnostics at the stage-s checkpoint (NOT the historical gradients):
    g_cur = d global_cosine / d theta on 16 fixed task-s train normals; g_c on the Replay-24 members of each completed
    category c held at the end of s; g_rep on all members. Parameter drift vs the previous state (lawful: s-1 <= s)."""
    from exps.dinomaly import official as O
    from exps.rh01 import common as RC
    _setup()
    proto, n = C.proto_and_n()
    out_dir = EXT / f"seed{seed}"
    path = out_dir / "grads.json"
    require(not path.exists(), "exists")
    res = {}
    cache = {}

    def feats(c, idx):
        if c not in cache:
            cache.clear()
            cache[c] = RC.load_features(c, mmap=True)
        return torch.stack([cache[c][i] for i in idx])

    for arm in C.ARMS:
        for s in C.states(arm):
            if arm == "FS_B" and C.owner(arm, s) == "ID":
                continue
            dec = _decoder(arm, seed, s)
            params = [p for p in dec.parameters() if p.requires_grad]

            def grad(x):
                """Gradient of the batch-mean global_cosine, accumulated over micro-batches of 2 (exact)."""
                dec.zero_grad(set_to_none=True)
                for xb in x.split(2):
                    en, de = dec(xb[:, 0].cuda(), xb[:, 1].cuda())
                    (O.global_cosine(en, de) * (len(xb) / len(x))).backward()
                return torch.cat([p.grad.flatten() for p in params]).detach().clone()

            c_now = C.STREAM[s - 1]
            g_cur = grad(feats(c_now, torch.randperm(n[c_now], generator=torch.Generator().manual_seed(seed * 100 + s))
                               [:16].tolist()))
            mem = C.members(seed, n, s)
            # global_cosine is a per-image mean and the decoder (eval, LayerNorm) is per-sample independent, so the
            # gradient on all members is exactly sum_c (n_c / N) g_c; accumulate it instead of holding every g_c on GPU
            N = sum(len(v) for v in mem.values())
            g_rep = torch.zeros_like(g_cur)
            row = {"per_cat": {}}
            for c, idx in mem.items():
                g = grad(feats(c, idx))
                g_rep += g * (len(idx) / N)
                row["per_cat"][c] = {"norm": float(g.norm()), "cos_cur": float(F.cosine_similarity(g_cur, g, 0))}
                del g
            for c, idx in mem.items():                                  # second pass: cosine with the full replay gradient
                g = grad(feats(c, idx))
                row["per_cat"][c]["cos_rep"] = float(F.cosine_similarity(g_rep, g, 0))
                del g
            row.update({"g_cur_norm": float(g_cur.norm()), "g_rep_norm": float(g_rep.norm()),
                        "cos_cur_rep": float(F.cosine_similarity(g_cur, g_rep, 0))})
            # parameter drift vs the previous state of the same lineage
            if s > 1:
                cur = torch.load(C.local_ckpt(arm, seed, s), weights_only=False, map_location="cpu")["state"]
                prv = torch.load(C.local_ckpt(arm, seed, s - 1), weights_only=False, map_location="cpu")["state"]
                blocks = {}
                tot_d, tot_n = 0.0, 0.0
                for k, v in cur.items():
                    if not v.is_floating_point():
                        continue
                    d2 = float((v - prv[k]).double().pow(2).sum()); n2 = float(prv[k].double().pow(2).sum())
                    b = ".".join(k.split(".")[:2])
                    bd = blocks.setdefault(b, [0.0, 0.0]); bd[0] += d2; bd[1] += n2
                    tot_d += d2; tot_n += n2
                row["param_rel_change"] = (tot_d / tot_n) ** 0.5
                rel = {b: (d / max(nn, 1e-30)) ** 0.5 for b, (d, nn) in blocks.items()}
                row["param_block_max_rel"] = max(rel.values())
                row["param_block_rel"] = rel
                del cur, prv
            res[f"{C.owner(arm, s)}|{s}"] = row
            print("grads", seed, arm, s, round(row["cos_cur_rep"], 3), flush=True)
            del dec, g_cur, g_rep
            torch.cuda.empty_cache()
    path.write_text(json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit", "parity", "extract", "grads"])
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    if a.cmd in ("extract", "grads"):
        globals()[a.cmd](a.seed)
    else:
        globals()[a.cmd]()
