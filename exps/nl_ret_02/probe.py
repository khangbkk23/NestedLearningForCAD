"""NL-RET-02 G1: causally timed, no-update probes at every task boundary s -> s+1 (FRESH_PREUPDATE_PROBES).

For each (seed, arm, s): restore theta_s (SHA-verified copy), M_s (Replay-24 members held at the end of s), the incoming
task's probe images (first K sampler batches of stage s+1, see common.py) and compute, with NO optimizer and NO
parameter update:
  gradient   g_new (probe images), g_b1 (exact first training batch: 8 incoming + 8 replay), g_c (members of each old
             category c), g_rep = sum_c (n_c / N) g_c; normal loss = global_cosine (== the iteration-0 training loss,
             hm p = 0), bf16 autocast + FP32 linear-attention core as in training, micro-batches of 8 weighted
             len(xb)/len(x) as in training, dropout disabled (eval) for determinism
  geometry   frozen-encoder features of the probe images vs lawful replay members (no nonmembers, no new memory)
  function   stage-s scores / losses of probe images and members (canonical R_D scorer, FP32)
  FS_B only  Q2 auxiliary quantities of stage s+1's first K host draws (Aux, unchanged): aux loss, ||g_aux||, cos with
             g_b1, EPS-attenuated norm-matched scale and dose, saturation, synthetic margins
Parameters are hashed before and after every state: any change aborts.

  python -m exps.nl_ret_02.probe bench            (compute budget + label-free probe stability; no outcomes)
  python -m exps.nl_ret_02.probe run --seed S     (needs PROBE_MANIFEST.json + SIGNAL_DICTIONARY lock)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from exps.cadfinal import common as K
from exps.dinomaly import official as O
from exps.dinomaly.common import require, sha
from exps.nl_ret_02 import common as C

MICRO = 8
PREFIX = 5
SAT = 1e-5


def setup():
    from exps.dinomaly import run as DR
    from exps.rh01 import numerics as NUM
    DR.CAP_GIB = 4.5
    DR.configure()
    NUM.use(True)                                  # D4': FP32 linear-attention core, as in R15/FS_B training


def decoder(arm, seed, s):
    from exps.cadfinal import dino as DN
    dec = DN.O_decoder()
    dec.load_state_dict(torch.load(C.local_ckpt(arm, seed, s), weights_only=False)["state"])
    return dec.eval()


def param_hash(dec):
    h = hashlib.sha256()
    for k, v in dec.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Feats:
    """Encoder-feature access with one category resident at a time (SHA-verified cache)."""

    def __init__(self):
        from exps.rh01 import common as RC
        self.RC, self.cache = RC, {}

    def get(self, c, idx):
        if c not in self.cache:
            if len(self.cache) > 1:
                self.cache.pop(next(iter(self.cache)))
            self.cache[c] = self.RC.load_features(c, mmap=True)
        return torch.stack([self.cache[c][i] for i in idx])


def grad(dec, params, x):
    """Training-identical reduction: sum over micro-batches of global_cosine(xb) * len(xb)/len(x) (bf16 autocast).
    Returns (flat FP32 gradient, per-image loss [B])."""
    dec.zero_grad(set_to_none=True)
    lv = []
    for xb in x.split(MICRO):
        xb = xb.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            en, de = dec(xb[:, 0], xb[:, 1])
        en, de = [e.float() for e in en], [d.float() for d in de]
        (O.global_cosine(en, de) * (len(xb) / len(x))).backward()
        with torch.no_grad():
            lv.append(torch.stack([1 - F.cosine_similarity(e.flatten(1), d.flatten(1)) for e, d in zip(en, de)],
                                  1).mean(1).cpu())
    g = torch.cat([p.grad.flatten() for p in params]).detach().clone()
    dec.zero_grad(set_to_none=True)
    return g, torch.cat(lv)


@torch.no_grad()
def scores(dec, x):
    from exps.dinomaly import readout as R
    out = []
    for xb in x.split(MICRO):
        en, de = dec(xb[:, 0].cuda(), xb[:, 1].cuda())
        out.append(R.dinomaly(en, de)[1].cpu())
    return torch.cat(out)


def block_slices(dec):
    from exps.nlcad.controller import BLOCKS, block_of
    sl, off = {b: [] for b in BLOCKS}, 0
    for n_, p in dec.named_parameters():
        if p.requires_grad:
            sl[block_of(n_)].append((off, off + p.numel()))
            off += p.numel()
    merged = {}
    for b, v in sl.items():                          # merge adjacent ranges (blocks are contiguous in parameter order)
        m = []
        for a, z in v:
            if m and m[-1][1] == a:
                m[-1] = (m[-1][0], z)
            else:
                m.append((a, z))
        merged[b] = m
    return merged


def cos(a, b):
    return float(F.cosine_similarity(a, b, 0))


def grad_feats(g_new, g_b1, g_c, g_mix, blocks):
    """Per-(boundary, old category) gradient-interaction features (definitions in SIGNAL_DICTIONARY.json)."""
    dot = float(g_new @ g_c)
    nn_, nc = float(g_new.norm()), float(g_c.norm())
    cs = dot / (nn_ * nc)
    orth = math.sqrt(max(nn_ ** 2 - dot ** 2 / nc ** 2, 0.0))       # == ||g_new - proj_{g_c} g_new||
    bc, bd = [], []
    for rng in blocks.values():
        d_ = sum(float(g_new[a:z] @ g_c[a:z]) for a, z in rng)
        na = math.sqrt(sum(float(g_new[a:z] @ g_new[a:z]) for a, z in rng))
        nb = math.sqrt(sum(float(g_c[a:z] @ g_c[a:z]) for a, z in rng))
        bc.append(d_ / (na * nb)); bd.append(d_)
    bd = np.array(bd)
    l1 = float(g_c.abs().sum())
    return {"GN_old_norm": nc, "GN_cos": cs, "GN_dot": dot, "GN_negalign": max(0.0, -cs), "GN_ratio": nn_ / nc,
            "GN_orth_rel": orth / nc, "GN_blk_cos_min": min(bc), "GN_blk_cos_mean": float(np.mean(bc)),
            "GN_blk_negfrac": float(np.mean(np.array(bc) < 0)),
            "GN_blk_conc": float((np.abs(bd) / max(np.abs(bd).sum(), 1e-30)).__pow__(2).sum()),
            "GN_fo_sgd": -dot, "GN_fo_sign": -float(g_c @ torch.sign(g_new)) / l1,
            "GN_b1_cos": cos(g_b1, g_c), "GN_b1_fo_sign": -float(g_c @ torch.sign(g_b1)) / l1,
            "GN_cos_mix": cos(g_mix, g_c)}


# ---- geometry (frozen encoder features; lawful pools only) ----------------------------------------------------

def img_vec(x):
    """[B, 2, 789, 768] -> L2-normalised [B, 1536] patch-token means of both layer groups."""
    v = x[:, :, PREFIX:, :].float().mean(2).flatten(1)
    return F.normalize(v, dim=1)


def patch_vec(x):
    """[B, 2, 789, 768] -> L2-normalised [B*784, 1536] patch descriptors (both groups concatenated)."""
    p = x[:, :, PREFIX:, :].float().permute(0, 2, 1, 3).flatten(2)
    return F.normalize(p.flatten(0, 1), dim=1)


def nn_dist(a, b, chunk=4096):
    """For each row of a: min cosine distance to rows of b (both normalised, on GPU)."""
    out = []
    for i in range(0, len(a), chunk):
        out.append((1 - (a[i:i + chunk] @ b.T).max(1).values).cpu())
    return torch.cat(out)


def principal_overlap(A, B, r):
    """Mean squared cosine of principal angles between top-r centred subspaces (r < min(n_A, n_B))."""
    ua = torch.linalg.svd(A - A.mean(0), full_matrices=False)[2][:r]
    ub = torch.linalg.svd(B - B.mean(0), full_matrices=False)[2][:r]
    return float((torch.linalg.svdvals(ua @ ub.T) ** 2).mean())


def geometry(fe, seed, n, s, pr):
    c_in = pr["incoming"]
    xi = fe.get(c_in, pr["incoming_idx"]).cuda()
    vi, pi = img_vec(xi), patch_vec(xi)
    del xi
    mem = C.members(seed, n, s)
    vm, pm = {}, {}
    for c, idx in mem.items():
        xc = fe.get(c, idx).cuda()
        vm[c], pm[c] = img_vec(xc), patch_vec(xc)
        del xc
    allv = torch.cat(list(vm.values()))
    nov = nn_dist(vi, allv)
    nn_i2c = {c: float(nn_dist(vi, v).mean()) for c, v in vm.items()}
    cent = {c: 1 - cos(vi.mean(0), v.mean(0)) for c, v in vm.items()}
    order = sorted(cent, key=cent.get)
    b = {"GEB_novelty": float(nov.mean()), "GEB_novelty_q90": float(nov.quantile(0.9)),
         "GEB_inc_dispersion": float((1 - vi @ vi.T)[~torch.eye(len(vi), dtype=bool, device=vi.device)].mean())}
    rows = {}
    for c, v in vm.items():
        others = torch.cat([vm[d] for d in vm if d != c]) if len(vm) > 1 else None
        pc = nn_dist(pm[c], pi)
        r = {"GE_cent_dist": cent[c], "GE_nn_c2i": float(nn_dist(v, vi).mean()), "GE_nn_i2c": nn_i2c[c],
             "GE_rel_nn": nn_i2c[c] / min(nn_i2c.values()),
             "GE_rank": order.index(c) / max(len(order) - 1, 1),
             "GE_patch_cov": float(pc.mean()), "GE_patch_q90": float(pc.quantile(0.9)),
             "GE_redund": float((v @ others.T).max(1).values.mean()) if others is not None else float("nan"),
             "GE_cov_overlap": principal_overlap(vi, v, min(len(v) - 1, 5)) if len(v) >= 3 else float("nan")}
        rows[c] = r
    del vi, pi, vm, pm, allv
    return b, rows


# ---- FS_B: Q2 auxiliary quantities at stage s+1's first host draws (Aux unchanged) -----------------------------

def q2_feats(dec, params, seed, n, s, K, g_b1, g_new, bank):
    from exps.dinomaly import readout as R
    from exps.nl_align_01 import g1 as G
    items, hosts, role = bank
    aux = G.Aux("FS_B", items, hosts, role, seed, n, s + 1, None, None, None)
    per, g1 = [], None
    for k in range(K):
        keys = aux.draw()
        dec.zero_grad(set_to_none=True)
        al = aux.backward(dec, keys, 0.0)               # bf16 autocast + ranking loss, as in training (eval: no dropout)
        g = None if al is None else torch.cat([p.grad.flatten() for p in params]).detach().clone()
        dec.zero_grad(set_to_none=True)
        with torch.no_grad():
            x = torch.stack([aux.syn[q][0] for q in keys]).cuda()
            cv = torch.stack([aux.syn[q][1] for q in keys]).cuda()
            en, de = dec(x[:, 0], x[:, 1])
            gr = R.grid(en, de)
            mg = [float(sc[cc >= 0.5].mean() - sc[cc == 0].quantile(0.99)) for sc, cc in zip(gr, cv)
                  if (cc >= 0.5).any() and (cc == 0).any()]
            bg = [float(sc[cc == 0].quantile(0.99)) for sc, cc in zip(gr, cv) if (cc == 0).any()]
        nA = float(g.norm()) if g is not None else 0.0
        rec = {"loss": float("nan") if al is None else al, "nA": nA, "sat": float(al is not None and al < SAT),
               "margin": float(np.mean(mg)) if mg else float("nan"), "bg_q99": float(np.mean(bg)) if bg else float("nan"),
               "cos_new": cos(g, g_new) if g is not None and nA > 0 else float("nan")}
        if k == 0:
            nN = float(g_b1.norm())
            ok = g is not None and math.isfinite(nA) and nA > 0
            scale = G.RHO * nN / (nA + G.EPS) if ok else 0.0
            rec.update({"nN_b1": nN, "cos_b1": cos(g, g_b1) if ok else float("nan"), "scale": scale,
                        "dose": scale * nA, "dose_rel": scale * nA / nN})
            g1 = g
        else:
            del g
        per.append(rec)
    b = {"FSQ_loss": float(np.nanmean([r["loss"] for r in per])), "FSQ_nA": float(np.mean([r["nA"] for r in per])),
         "FSQ_sat": float(np.mean([r["sat"] for r in per])), "FSQ_margin": float(np.nanmean([r["margin"] for r in per])),
         "FSQ_bg_q99": float(np.nanmean([r["bg_q99"] for r in per])),
         "FSQ_cos_new": float(np.nanmean([r["cos_new"] for r in per])),
         "FSQ_step1_nN": per[0]["nN_b1"], "FSQ_step1_cos": per[0]["cos_b1"], "FSQ_step1_scale": per[0]["scale"],
         "FSQ_step1_dose_rel": per[0]["dose_rel"]}
    return b, g1, per


# ---- one boundary --------------------------------------------------------------------------------------------

def boundary(arm, seed, s, K, fe, n, bank=None, timing=None):
    t0 = time.time()
    pr = C.probe(seed, n, s, K)
    dec = decoder(arm, seed, s)
    h0 = param_hash(dec)
    params = [p for p in dec.parameters() if p.requires_grad]
    blocks = block_slices(dec)
    xi = fe.get(pr["incoming"], pr["incoming_idx"])
    half = len(xi) // 2
    g_a, l_a = grad(dec, params, xi[:half])
    g_b, l_b = grad(dec, params, xi[half:])
    g_new = 0.5 * (g_a + g_b) if len(xi) == 2 * half else None
    if g_new is None:
        g_new, _ = grad(dec, params, xi)
    split_cos = cos(g_a, g_b)
    del g_a, g_b
    loss_inc = torch.cat([l_a, l_b])
    sc_inc = scores(dec, xi)
    xb1 = torch.stack([fe.get(c, [i])[0] for c, i in pr["first_batch"]])
    g_b1, _ = grad(dec, params, xb1)
    del xb1
    mem = C.members(seed, n, s)
    N = sum(len(v) for v in mem.values())
    g_rep = torch.zeros_like(g_new)
    old = {}
    for c, idx in mem.items():
        x = fe.get(c, idx)
        g, lv = grad(dec, params, x)
        g_rep += g * (len(idx) / N)
        old[c] = {"loss": float(lv.mean()), "img": float(scores(dec, x).mean()), "n": len(idx)}
        del g
    g_rep_norm, cos_new_rep = float(g_rep.norm()), cos(g_new, g_rep)
    g_mix = g_rep.mul_(0.5).add_(g_new, alpha=0.5)           # in place: g_rep is not needed after this point
    del g_rep
    rows = {}
    for c, idx in mem.items():
        g, _ = grad(dec, params, fe.get(c, idx))
        rows[c] = grad_feats(g_new, g_b1, g, g_mix, blocks)
        del g
    tg = time.time() - t0
    gb, gr = geometry(fe, seed, n, s, pr)
    allimg = float(np.average([v["img"] for v in old.values()], weights=[v["n"] for v in old.values()]))
    inc_img = sc_inc.numpy()
    b = {"GNB_new_norm": float(g_new.norm()), "GNB_b1_norm": float(g_b1.norm()), "GNB_rep_norm": g_rep_norm,
         "GNB_cos_new_rep": cos_new_rep, "GNB_split_half_cos": split_cos, **gb,
         "FNB_inc_img_mean": float(inc_img.mean()), "FNB_inc_img_q90": float(np.quantile(inc_img, 0.9)),
         "FNB_inc_loss": float(loss_inc.mean()), "FNB_inc_vs_buffer": math.log(float(inc_img.mean()) / allimg)}
    fs = None
    if arm == "FS_B":
        fs, g_aux, per = q2_feats(dec, params, seed, n, s, K, g_b1, g_new, bank)
        b.update(fs)
        for c, idx in mem.items():
            g, _ = grad(dec, params, fe.get(c, idx))
            rows[c]["FSQ_cos_aux_c"] = cos(g_aux, g) if g_aux is not None else float("nan")
            del g
        b["FSQ_per_draw"] = per
    for c in rows:
        rows[c].update(gr[c])
        rows[c].update({"FN_old_loss": old[c]["loss"], "FN_old_img": old[c]["img"],
                        "FN_loss_ratio": float(loss_inc.mean()) / old[c]["loss"],
                        "FN_shift": math.log(float(inc_img.mean()) / old[c]["img"]), "n_members": old[c]["n"]})
    h1 = param_hash(dec)
    require(h0 == h1, "parameters changed during probing")
    require(all(p.grad is None for p in params), "gradient buffers left on parameters")
    del dec, g_new, g_b1, g_mix
    torch.cuda.empty_cache()
    if timing is not None:
        timing.update({"grad_s": tg, "total_s": time.time() - t0})
    return {"arm": arm, "seed": seed, "s": s, "owner": C.owner(arm, s), "probe": pr, "param_sha_before": h0,
            "param_sha_after": h1, "boundary": b, "rows": rows}


def bench():
    """Compute budget and label-free probe stability on one representative boundary (ID seed 2711, s = 7, 7 old
    categories) for K in {2, 4, 8}. Reads no outcome."""
    setup()
    proto, n = C.proto_and_n()
    fe = Feats()
    res = {}
    torch.cuda.reset_peak_memory_stats()
    for k in (2, 4, 8):
        tm = {}
        r = boundary("ID", 2711, 7, k, fe, n, timing=tm)
        res[f"K{k}"] = {**tm, "split_half_cos": r["boundary"]["GNB_split_half_cos"], "n_probe": len(r["probe"]["incoming_idx"])}
        print(k, res[f"K{k}"], flush=True)
    res["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    K.write_json(C.OUT / "BENCH.json", res)


def run(seed):
    setup()
    man = K.read(C.OUT / "PROBE_MANIFEST.json")
    lock = K.read(C.OUT / "AUDIT_LOCK.json")
    allowed = dict(lock["sources"])
    am = C.OUT / "AMENDMENT_01.json"                      # pre-output engineering amendment (memory only)
    if am.exists():
        allowed.update(K.read(am)["sources_after"])
    for f, h in allowed.items():
        require(sha(C.ROOT / f) == h, f"locked source changed: {f}")
    Kp = man["K"]
    proto, n = C.proto_and_n()
    fe = Feats()
    from exps.nl_align_01 import g1 as G
    from exps.nl_align_02 import m3 as M3
    items, hosts = M3.load_bank_15(seed)
    bank = (items, hosts, G.roles(hosts))
    d = C.EXT / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    for arm in ("ID", "FS_B"):
        for s in C.boundaries(arm):
            p = d / f"{arm}_s{s}.json"
            if p.exists():
                continue
            t0 = time.time()
            r = boundary(arm, seed, s, Kp, fe, n, bank=bank)
            want = man["probes"][str(seed)][str(s)]
            require(r["probe"]["incoming_idx"] == want["incoming_idx"], "probe identities differ from manifest")
            r["seconds"] = time.time() - t0
            r["utc"] = K.now()
            p.write_text(json.dumps(r))
            print(f"{K.now()} seed {seed} {arm} s={s} {r['seconds']:.0f}s split_cos={r['boundary']['GNB_split_half_cos']:.3f}",
                  flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "run"])
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    bench() if a.cmd == "bench" else run(a.seed)
