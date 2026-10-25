# exps/rh01/diagnose_gap.py
"""Post-hoc audit of the Replay-15 gap from existing states and logs (CPU only; normal features only; no labels,
no training). Separates three hypotheses with the evidence each diagnostic can and cannot provide:

  H-A buffer coverage   : allocation trajectory, held-out loss (logged), buffer member vs non-member loss
  H-B warm-restart drift: per-stage parameter drift, displacement from acquisition, restart loss signature
  H-C gradient conflict : cos(g_new, g_old) at the start of each stage and cos(realized update, -g_old)

  python -m exps.rh01.diagnose_gap drift|coverage|conflict --seeds 2711 2712
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.rh01 import common as C
from exps.rh01 import run as RR

OUT = C.OUT / "diag"
MODULES = ["bottleneck.0"] + [f"decoder.{i}" for i in range(8)]
ACQ = {c: C.STREAM.index(c) + 1 for c in C.EVAL}


def stage_paths(seed):
    rec = D.read(C.OUT / f"runs/R15_seed{seed}/TRAIN.json")
    out = {}
    for st in rec["stages"]:
        path = D.ROOT / st["checkpoint"]
        require(sha(path) == st["checkpoint_sha256"], f"checkpoint changed: {path}")
        out[st["index"]] = (path, st)
    return out


def load(path):
    return torch.load(path, weights_only=True, map_location="cpu")["state"]


def flat(sd, prefix=None):
    return torch.cat([v.flatten().double() for k, v in sd.items() if prefix is None or k.startswith(prefix + ".")])


# --- H-B ---------------------------------------------------------------------------------------------


def drift(seeds):
    res = {}
    for seed in seeds:
        paths = stage_paths(seed)
        prev, acq, rows = None, {}, {}
        for s in range(1, len(C.STREAM) + 1):
            sd = load(paths[s][0])
            v = flat(sd)
            row = {}
            if prev is not None:
                pv = flat(prev)
                row["rel_drift"] = float((v - pv).norm() / pv.norm())
                row["module_rel_drift"] = {m: float((flat(sd, m) - flat(prev, m)).norm() / flat(prev, m).norm())
                                           for m in MODULES}
                st = paths[s][1]
                curve = st.get("loss_curve_every50", [])
                row["loss_first50_last50"] = [curve[0], st["loss_last50"]] if curve else None
            for c in C.EVAL:
                if s == ACQ[c]:
                    acq[c] = v
                if c in acq:
                    row[f"displacement_from_{c}_acq"] = float((v - acq[c]).norm() / acq[c].norm())
            rows[s] = row
            prev = sd
        res[seed] = rows
        print("drift", seed, [round(rows[s]["rel_drift"], 3) for s in range(2, 16)], flush=True)
    D.write_json(OUT / "HB_drift.json", res)


# --- H-A ---------------------------------------------------------------------------------------------


@torch.no_grad()
def per_image_loss(dec, f):
    out = []
    for xb in f.split(4):
        en, de = dec(xb[:, 0].float(), xb[:, 1].float())
        out += torch.stack([1 - F.cosine_similarity(e.flatten(1), d.flatten(1)) for e, d in zip(en, de)], 1)\
            .mean(1).tolist()
    return np.array(out)


def coverage(seeds, stages=(7, 15)):
    proto = RR.verify_lock()
    n = RR.n_train(proto)
    held = {c: DR.load_split("heldout", c) for c in C.EVAL}
    res = {}
    for seed in seeds:
        paths = stage_paths(seed)
        res[seed] = {}
        for s in stages:
            dec = O.Decoder(768, 12).eval()
            dec.load_state_dict(load(paths[s][0]))
            idx = C.buffer_index(seed, n, list(C.STREAM[:s - 1]))
            ids = [proto["train"][d][i]["relative_path"] for d in C.STREAM[:s - 1] for i in idx[d]]
            require(ids == paths[s][1]["replay"]["identities"], "buffer re-derivation mismatch")
            row = {}
            for c in C.EVAL:
                f = C.load_features(c, mmap=True)
                loss = per_image_loss(dec, f)
                m = np.zeros(len(f), bool)
                m[idx.get(c, [])] = True
                row[c] = {"members_n": int(m.sum()), "members": float(loss[m].mean()) if m.any() else None,
                          "nonmembers": float(loss[~m].mean()), "heldout": float(per_image_loss(dec, held[c]).mean())}
            res[seed][s] = row
            print("coverage", seed, s, {c: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                                        for c, r in row.items()}, flush=True)
    D.write_json(OUT / "HA_coverage.json", res)


# --- H-C ---------------------------------------------------------------------------------------------


def grad(dec, f):
    dec.zero_grad(set_to_none=True)
    for xb in f.split(4):
        en, de = dec(xb[:, 0].float(), xb[:, 1].float())
        (O.global_cosine(en, de) * len(xb) / len(f)).backward()
    return torch.cat([p.grad.flatten().double() for p in dec.parameters()])


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm()))


def conflict(seeds):
    proto = RR.verify_lock()
    n = RR.n_train(proto)
    res = {}
    for seed in seeds:
        paths = stage_paths(seed)
        res[seed] = {}
        for s in range(2, len(C.STREAM) + 1):
            prev, cur = load(paths[s - 1][0]), load(paths[s][0])
            dec = O.Decoder(768, 12).eval()                           # FP32, no dropout: a proxy of training grads
            dec.load_state_dict(prev)
            c_new, done = C.STREAM[s - 1], list(C.STREAM[:s - 1])
            idx = C.buffer_index(seed, n, done)
            pick = torch.randperm(n[c_new], generator=torch.Generator().manual_seed(seed * 1000 + s))[:8].tolist()
            g_new = grad(dec, C.load_features(c_new)[pick])
            buf = torch.stack([C.load_features(d)[i] for d in done for i in idx[d]])
            g_buf = grad(dec, buf)
            update = flat(cur) - flat(prev)
            row = {"cos_new_buffer": cos(g_new, g_buf), "cos_update_minus_gbuf": cos(update, -g_buf),
                   "cos_update_minus_gnew": cos(update, -g_new)}
            for c in C.EVAL:
                if c in done:
                    g_c = grad(dec, C.load_features(c)[idx[c]])
                    row[c] = {"members": len(idx[c]), "cos_new_old": cos(g_new, g_c),
                              "cos_update_minus_gold": cos(update, -g_c)}
            res[seed][s] = row
            print("conflict", seed, s, c_new, {k: (round(v, 3) if isinstance(v, float) else
                                                   {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in row.items()},
                  flush=True)
    D.write_json(OUT / "HC_conflict.json", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("drift", "coverage", "conflict"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[2711, 2712])
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    {"drift": drift, "coverage": coverage, "conflict": conflict}[a.stage](a.seeds)


if __name__ == "__main__":
    main()
