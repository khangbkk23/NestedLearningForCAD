# exps/nlcad/closed_loop_06b.py
"""NLCAD-06B-MECH: closed-loop, per-step effective-update protection of the eight decoder attn.proj weights with FIXED
policies (agents/tasks/task_nlcad_06b_closed_loop_mechanism.md). No learned controller, no slow objective, no labels.

Training is exps.rh01.run.train_stage_guarded op for op (RH-01b R15 seed 2711; stages 7-9 from the SHA-verified stage-6
checkpoint; D4' numerics; same sampler and dropout seeds per stage, hard mining, global clipping, skip guard, StableAdamW
and WarmCosine rebuilt at every task, 667 steps), plus one operator applied right after every opt.step():

    Delta  = W_after - W_before                     effective update (Adam step + decoupled weight decay), per layer
    W_next = W_before + Delta - alpha (Delta U) U^T  projection arms (U: 768 x r orthonormal input basis, right side)
    W_next = W_before + beta_t Delta                 NORM-05, beta_t = ||Delta (I - 0.5 U_REP U_REP^T)|| / ||Delta||

Disclosed intervention: optimizer moments (exp_avg, exp_avg_sq, max_exp_avg_sq, step) are NOT touched; they keep the
history of the unprojected gradients. Biases and all other parameters follow the unmodified optimizer.

Memories (token-level uncentred second moments of the attn.proj inputs, trace-normalised per category, mean over old
categories, top-r eigenvectors at tau = 0.97 energy; FP32 eval-mode forward, FP64 accumulation, RNG forked):
  REP     causal: at every boundary, from the arm's OWN decoder state and the lawful Replay-24 members of that stage.
  ACQ     causal: task t's summary is written at the end of task t (24 of its training normals, at the arm's own
          theta_t; for t <= 6 the shared R15 trajectory); stage s uses the mean over t < s.  Stale by construction.
  ORACLE  NON-CAUSAL diagnostic: arm's own theta_{s-1} on 8 non-retained training normals per old task.
  RAND    fixed random orthogonal matrix per layer (seed RAND_SEED + layer index), first r columns, r = rank of the arm's
          own REP basis at that boundary (rank-matched; nested across stages).
Every arm also computes its own REP basis as a common yardstick for update statistics (training unaffected).

Measurements (normal-only, fixed sets, FP32, eval mode, RNG forked; stage start, every 128 steps, stage end): per-image
global-cosine loss, Dinomaly image score, p99 of the smoothed 256x256 normal map, pooled-output drift; on non-retained
old normals (DRF-01 held-out for bottle/carpet/hazelnut, 6 non-retained training normals otherwise; FORENSIC_06A
subsets), the current task's 16 training normals not used for any memory, and the stage's Replay-24 members.

  python -m exps.nlcad.closed_loop_06b prereg          # CPU: lock config, rules, sources, checkpoints, sets
  python -m exps.nlcad.closed_loop_06b preflight       # GPU (~5 min): loop parity, operator checks, timing, budget
  python -m exps.nlcad.closed_loop_06b train --arm ID  # GPU: one arm, stages 7-9 (only after a PASS preflight)
  python -m exps.nlcad.closed_loop_06b analyze         # CPU: paired comparisons and the predeclared decision
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as Fn

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import readout as RO
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nlcad import forensic_06a as F
from exps.nlcad import run as R
from exps.nlcad.controller import BLOCKS, block_of
from exps.rh01 import numerics as NUM
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

OUT = R.OUT
TASK = "agents/tasks/task_nlcad_06b_closed_loop_mechanism.md"
SRC = "exps/nlcad/closed_loop_06b.py"
SOURCES = [SRC, "exps/nlcad/test_nlcad_06b.py", "exps/nlcad/chain_06b.sh", "exps/nlcad/forensic_06a.py",
           "exps/nlcad/run.py", "exps/nlcad/controller.py", "exps/nlcad/fast.py", "exps/dinomaly/official.py",
           "exps/dinomaly/readout.py", "exps/dinomaly/run.py", "exps/dinomaly/common.py", "exps/rh01/run.py",
           "exps/rh01/common.py", "exps/rh01/numerics.py", "exps/rh01b/common.py", "exps/rh01b/run.py", TASK]
PREREG, PREFLIGHT = OUT / "PREREG_06B.json", OUT / "PREFLIGHT_06B.json"
AMEND = OUT / "PREREG_06B_AMENDMENT_1.json"
ANALYSIS, DECISION = OUT / "ANALYSIS_06B.json", OUT / "DECISION_06B.json"
SEED, START, STAGES = 2711, 6, (7, 8, 9)
ITERS, EVERY, TAU = R1.ITERS, 128, F.TAU
LAYERS = F.LAYERS
RAND_SEED = 606200
ARMS = {"ID": ("id", 0.0), "REP-05": ("rep", 0.5), "REP-10": ("rep", 1.0), "RAND-05": ("rand", 0.5),
        "NORM-05": ("norm", 0.5), "ACQ-05": ("acq", 0.5), "ORACLE-05": ("oracle", 0.5)}
ORDER = ["ID", "REP-05", "NORM-05", "RAND-05", "REP-10", "ACQ-05", "ORACLE-05"]
O6 = C.STREAM[:START]                                   # old tasks shared by every arm and every stage: primary set
PF_ITERS = 110                                          # preflight parity / timing segment (> 100 warm-up steps)
BUDGET_H, OVERRUN_STOP_H = 2.0, 2.4
TOL = {"comp_res_a1": 1e-5, "comp_ratio_a05": 1e-5, "real_res_a1": 1e-3, "real_ratio_a05": 1e-3, "norm_ratio": 1e-3}
# per-step statistics columns (per layer)
COLS = ["e_tot", "e_act", "e_rep", "removed", "comp_res", "real_res", "e_decay", "e_decay_act", "beta", "e_real", "lr"]
BOOT_B, BOOT_SEED = 10000, 6062
RULES = {
    "notation": "O6 = old-normal eval images of tasks 1-6 (78; DRF-01 held-out x20 for bottle/carpet/hazelnut, 6 "
                "non-retained training normals for cable/capsule/grid). M[.] = category-macro mean (mean over the six "
                "categories of the per-category image mean). L_a(x), p99_a(x): loss and map p99 at arm a's stage-9 end; "
                "L_6, p99_6 at the shared theta_6.",
    "D_a": "M[L_a - L_6]  (cumulative old-normal loss inflation over stages 7-9)",
    "T_a": "M[p99_a - p99_6]  (cumulative upper-tail inflation)",
    "protection": "1 - D_a / D_ID; tail protection 1 - T_a / T_ID",
    "uncertainty": f"paired, category-stratified image bootstrap of M[L_a - L_b] (B={BOOT_B}, seed {BOOT_SEED}); 95% "
                   "percentile CI for superiority, 90% for equivalence. Covers image sampling only, NOT trajectory or "
                   "seed variability: one seed, three stages, mechanism feasibility only.",
    "acquisition": "stage s current set C_s (16 training normals of task s, never used by any memory): A_ID(s) = "
                   "mean L(theta_{s-1}^ID) - mean L(theta_s^ID); retention R_a(s) = (mean L(theta_{s-1}^ID) - mean "
                   "L(theta_s^a)) / A_ID(s), i.e. the arm's end-of-stage fit measured against ID's improvement from "
                   "ID's own stage start (common reference). Degenerate case A_ID(s) < 0.01: R undefined; 'retained' "
                   "iff mean L(theta_s^a) - mean L(theta_s^ID) <= 0.001; 'prohibitive' iff that excess > 0.005.",
    "P0_damage_present": "D_ID > 0 with 95% CI lower bound > 0, else CLOSED_LOOP_INCONCLUSIVE (nothing to protect)",
    "S1_protection": "protection(REP-05) >= 0.25 AND 95% CI of D_REP05 - D_ID entirely < 0",
    "S2_tail": "if T_ID > 0 with 95% CI lower bound > 0: tail protection(REP-05) >= 0.25 AND 95% CI of "
               "T_REP05 - T_ID entirely < 0; if ID shows no measurable tail inflation, S2 fails (tail not exercised)",
    "S3_selectivity": "for c in (RAND-05, NORM-05): 95% CI of D_REP05 - D_c entirely < 0 AND D_REP05 - D_c <= -0.10 D_ID",
    "S4_acquisition": "R_REP05(s) >= 0.90 for s = 7, 8, 9 (degenerate rule where applicable)",
    "S5_categories": "D_REP05,c < D_ID,c in >= 4 of the 6 O6 categories",
    "SUPPORTED": "integrity AND P0 AND S1 AND S2 AND S3 AND S4 AND S5",
    "R1_no_protection": "protection(REP-05) < 0.10 (includes closed-loop harm)",
    "R2_not_selective": "D_REP05 >= min(D_RAND05, D_NORM05) (point estimates)",
    "R3_prohibitive": "R_REP05(s) < 0.75 at any stage (degenerate rule where applicable)",
    "REJECTED": "integrity AND P0 AND (R1 OR R2 OR R3)",
    "INCONCLUSIVE": "otherwise",
    "equivalence_margin": "delta_eq = 0.002 global-cosine loss units on M[.] (about 13% of the 06A stage-7 identity "
                          "inflation 0.0151 on the same O6 images); equivalence iff the 90% CI lies inside "
                          "(-delta_eq, +delta_eq). Failure to reject a difference is never read as equivalence.",
    "Q1_reversal": "REP-10 and REP-05 vs ID on O6 at the stage-7 end (06A open-loop REP a=1 protection was -0.79): "
                   "'reversed' iff 95% CI of M[L7_a - L7_ID] < 0; 'still harmful' iff CI > 0; else 'undetermined'. "
                   "Also reported cumulatively at stage 9.",
    "Q2_soft_vs_hard": "'soft better' iff 95% CI of D_REP05 - D_REP10 < 0, or (equivalent within delta_eq and "
                       "min_s R_REP05 - min_s R_REP10 >= 0.05); 'hard better' iff 95% CI of D_REP10 - D_REP05 < 0 and "
                       "min_s R_REP10 >= 0.90; else 'undetermined'. REP-10 is a stress dose, never compared with "
                       "strength-mismatched controls.",
    "Q3_direction_vs_scaling": "REP-05 vs NORM-05: 'direction adds value' iff 95% CI of D_REP05 - D_NORM05 < 0; "
                               "'scaling as good or better' iff CI > 0 or equivalent within delta_eq; else 'undetermined'",
    "Q4_rep_vs_acq": "'REP better' iff 95% CI of D_REP05 - D_ACQ05 < 0; 'ACQ better' iff CI > 0; 'equivalent' within "
                     "delta_eq; else 'undetermined'",
    "Q5_oracle_vs_rep": "'ORACLE better' iff 95% CI of D_ORACLE05 - D_REP05 < 0 AND point <= -delta_eq; 'equivalent' "
                        "within delta_eq; 'REP better' iff CI > 0 AND point >= delta_eq; else 'undetermined'. ORACLE is "
                        "non-causal: a gap motivates a memory investigation, never a deployable claim or a Level-3 claim.",
    "integrity_stops": "ID bitwise parity with R15 at stages 7, 8, 9; finite parameters at every probe; skip guard "
                       f"limits; operator tolerances {TOL}; source/plan hashes. Efficacy never stops a run.",
    "budget": f"launch only if the preflight projection <= {BUDGET_H} GPU-h (device wall time, sequential arms); "
              f"resource stop before starting a new arm-stage once cumulative GPU time exceeds {OVERRUN_STOP_H} h",
    "not_claimed": "native Pixel AP, Image AUROC, anomaly sensitivity, multi-seed generality, any Level-3 necessity",
}
DELTA_EQ = 0.002


# --- memories --------------------------------------------------------------------------------------------------------

def rng_fork(device):
    return torch.random.fork_rng(devices=[torch.device(device).index or 0] if torch.device(device).type == "cuda" else [])


@torch.no_grad()
def covs(dec, x, device, micro=4):
    """Token-level uncentred second moment of each attn.proj input: FP32 eval-mode forward, FP64 accumulation, RNG
    forked, hooks removed and the module's train/eval mode restored."""
    acc, cnt, hooks = {}, {}, []
    mods = dict(dec.named_modules())
    was = dec.training

    def hook(name):
        def fn(_, inp):
            a = inp[0].reshape(-1, inp[0].shape[-1]).double()
            acc[name] = acc.get(name, 0) + a.T @ a
            cnt[name] = cnt.get(name, 0) + len(a)
        return fn

    with rng_fork(device):
        dec.eval()
        try:
            for l in LAYERS:
                hooks.append(mods[l].register_forward_pre_hook(hook(l)))
            for xb in x.split(micro):
                xb = xb.to(device).float()
                dec(xb[:, 0], xb[:, 1])
        finally:
            for h in hooks:
                h.remove()
            dec.train(was)
    return {l: (acc[l] / cnt[l]).cpu() for l in LAYERS}


def bases(cov, tau=TAU):
    """Per layer top-r eigenvectors (FP64, CPU) at tau energy."""
    return {l: F.basis(cov[l].double().cpu(), tau)[0] for l in LAYERS}


def rand_orthogonal(j, dim=768):
    g = torch.Generator().manual_seed(RAND_SEED + j)
    return torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=torch.float64))[0]


def rand_bases(ranks, dim=768):
    """Fixed random orthonormal bases, nested across stages: first r columns of a seeded orthogonal matrix per layer."""
    return {l: rand_orthogonal(j, dim)[:, :ranks[l]] for j, l in enumerate(LAYERS)}


def capture(cov, U):
    return {l: F.frac_in(cov[l].double(), U[l]) for l in LAYERS}


# --- post-step operator -------------------------------------------------------------------------------------------

class Protector:
    """Fixed post-step operator on the attn.proj weights and per-step update statistics (kept on the device, one row of
    len(COLS) per layer per step). kind: id (no write), rep/rand/acq/oracle (projection with alpha), norm (uniform
    scaling matched to the REP-05 retained norm). U_act: the acting basis (REP basis for id/norm); U_rep: the arm's own
    REP basis (common yardstick)."""

    def __init__(self, dec, kind, alpha, U_act, U_rep, device):
        require(kind in ("id", "rep", "rand", "acq", "oracle", "norm"), f"bad kind {kind}")
        mods = dict(dec.named_modules())
        self.W = [mods[l].weight for l in LAYERS]
        self.kind, self.alpha, self.device = kind, float(alpha), device
        self.Ua = [U_act[l].to(device, torch.float32) for l in LAYERS]
        self.Ur = [U_rep[l].to(device, torch.float32) for l in LAYERS]
        self.same = all(torch.equal(a, b) for a, b in zip(self.Ua, self.Ur))
        self.wd = O.RECIPE["weight_decay"]
        self.rows, self.prev = [], None

    def before(self):
        self.prev = [w.detach().clone() for w in self.W]

    @torch.no_grad()
    def after(self, lr):
        out = torch.zeros(len(LAYERS), len(COLS), dtype=torch.float64, device=self.device)
        a = self.alpha
        for j, (w, w0) in enumerate(zip(self.W, self.prev)):
            Ua, Ur = self.Ua[j], self.Ur[j]
            d = w.detach() - w0
            da = d @ Ua
            dr = da if self.same else d @ Ur
            e_tot, e_act, e_rep = d.double().pow(2).sum(), da.double().pow(2).sum(), dr.double().pow(2).sum()
            beta = torch.ones((), dtype=torch.float64, device=self.device)
            if self.kind in ("rep", "rand", "acq", "oracle") and a != 0.0:
                new = d - a * (da @ Ua.T)
                w.copy_(w0 + new)
            elif self.kind == "norm":
                beta = torch.sqrt((e_tot - (2 * a - a * a) * e_rep).clamp(min=0) / e_tot.clamp(min=1e-300))
                new = d * beta.float()
                w.copy_(w0 + new)
            else:
                new = d
            real = w.detach() - w0
            decay = w0 * (lr * self.wd)                           # p.mul_(1 - lr*wd): the decay part of Delta is -decay
            out[j] = torch.stack([e_tot, e_act, e_rep, (d - real).double().pow(2).sum(),
                                  (new @ Ua).double().pow(2).sum(), (real @ Ua).double().pow(2).sum(),
                                  decay.double().pow(2).sum(), (decay @ Ua).double().pow(2).sum(), beta,
                                  real.double().pow(2).sum(), torch.tensor(lr, dtype=torch.float64, device=self.device)])
        self.rows.append(out)
        self.prev = None

    def table(self):
        return torch.stack(self.rows).cpu() if self.rows else torch.zeros(0, len(LAYERS), len(COLS), dtype=torch.float64)


def operator_checks(tab, kind, alpha):
    """Max per-step deviations of the operator from its specification (steps with a nonzero acting component)."""
    k = {c: i for i, c in enumerate(COLS)}
    e_act, e_tot = tab[..., k["e_act"]], tab[..., k["e_tot"]]
    m = e_act > 0
    comp = torch.sqrt(tab[..., k["comp_res"]][m] / e_act[m])
    real = torch.sqrt(tab[..., k["real_res"]][m] / e_act[m])
    out = {"steps": int(tab.shape[0]), "comp_ratio_max_dev": None, "real_ratio_max_dev": None, "norm_ratio_max_dev": None}
    if kind in ("rep", "rand", "acq", "oracle", "id"):
        target = 1.0 - (alpha if kind != "id" else 0.0)
        out["comp_ratio_max_dev"] = float((comp - target).abs().max()) if comp.numel() else 0.0
        out["real_ratio_max_dev"] = float((real - target).abs().max()) if real.numel() else 0.0
    if kind == "norm":
        ratio = torch.sqrt(tab[..., k["e_real"]] / e_tot.clamp(min=1e-300))
        out["norm_ratio_max_dev"] = float((ratio - tab[..., k["beta"]]).abs().max())
    if kind == "id":
        out["removed_max"] = float(tab[..., k["removed"]].max()) if tab.numel() else 0.0
    return out


def operator_ok(chk, kind, alpha):
    if kind == "id":
        return chk.get("removed_max", 0.0) == 0.0
    if kind == "norm":
        return chk["norm_ratio_max_dev"] <= TOL["norm_ratio"]
    comp_tol = TOL["comp_res_a1"] if alpha == 1.0 else TOL["comp_ratio_a05"]
    real_tol = TOL["real_res_a1"] if alpha == 1.0 else TOL["real_ratio_a05"]
    return chk["comp_ratio_max_dev"] <= comp_tol and chk["real_ratio_max_dev"] <= real_tol


# --- training (exps.rh01.run.train_stage_guarded op for op, plus the operator and probes) ----------------------------

def train_stage_protected(dec, current, iters, gen, replay=None, op=None, probe=None, every=EVERY, device="cuda",
                          autocast=True):
    opt = O.make_optimizer(dec)
    sch = O.make_scheduler(opt, iters)
    ramp = round(0.1 * iters)
    batch = O.RECIPE["batch_size"]
    cur = DR.Batches(len(current), batch // 2 if replay is not None else batch, gen)
    rep = DR.Batches(len(replay), batch // 2, gen) if replay is not None else None
    dec.train()
    dtype = torch.device(device).type
    losses, skips, run_ = [], [], 0
    for i in range(iters):
        p = min(O.RECIPE["hm_p_final"] * i / ramp, O.RECIPE["hm_p_final"])
        x = current[cur.next()]
        if rep is not None:
            x = torch.cat([x, replay[rep.next()]])
            x = x[torch.randperm(len(x), generator=gen)]
        x = x.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        total = 0.
        for xb in x.split(DR.MICRO):
            with torch.autocast(dtype, dtype=torch.bfloat16, enabled=autocast):
                en, de = dec(xb[:, 0], xb[:, 1])
            loss = O.global_cosine_hm_percent([e.float() for e in en], [d.float() for d in de],
                                              p=p, factor=O.RECIPE["hm_factor"]) * len(xb) / len(x)
            loss.backward()
            total += float(loss)
        norm = torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=O.RECIPE["clip_grad_norm"])
        if not (np.isfinite(total) and bool(torch.isfinite(norm))):
            opt.zero_grad(set_to_none=True)
            skips.append({"iteration": i, "loss": total, "grad_norm": float(norm)})
            run_ += 1
            require(len(skips) <= R1.MAX_SKIPS and run_ < R1.MAX_CONSECUTIVE, f"non-finite steps exceed limits: {skips}")
            sch.step()
            losses.append(float("nan"))
        else:
            run_ = 0
            lr = float(opt.param_groups[0]["lr"])
            if op is not None:
                op.before()
            opt.step()
            if op is not None:
                op.after(lr)
            sch.step()
            losses.append(total)
        if probe is not None and ((i + 1) % every == 0 or i + 1 == iters):
            probe(i + 1)
    return losses, skips                          # the optimizer (3 x 235 MB of moments) is freed with this frame


# --- probes ------------------------------------------------------------------------------------------------------

@torch.no_grad()
def measure(dec, x, device, pool=14, micro=8):
    """Per image: global-cosine loss, Dinomaly image score, p99 of the smoothed normal map; pooled outputs (CPU) for
    drift. FP32 eval-mode forward, RNG forked, train/eval mode restored."""
    loss, score, p99, pooled = [], [], [], []
    was = dec.training
    with rng_fork(device):
        dec.eval()
        try:
            for xb in x.split(micro):
                xb = xb.to(device).float()
                en, de = dec(xb[:, 0], xb[:, 1])
                loss += torch.stack([1 - Fn.cosine_similarity(e.flatten(1), d.flatten(1)) for e, d in zip(en, de)],
                                    1).mean(1).tolist()
                smap, img = RO.dinomaly(en, de)
                score += img.tolist()
                p99 += torch.quantile(smap.flatten(1).float(), 0.99, dim=1).tolist()
                pooled.append(torch.stack([Fn.adaptive_avg_pool2d(d, pool).flatten(1) for d in de], 1).cpu())
        finally:
            dec.train(was)
    return {"loss": loss, "score": score, "p99": p99}, torch.cat(pooled)


def drift(pooled, ref):
    return (1 - Fn.cosine_similarity(pooled, ref, dim=2)).mean(1).tolist()


def params_finite(dec):
    return all(bool(torch.isfinite(p).all()) for p in dec.parameters())


# --- data -------------------------------------------------------------------------------------------------------

def stage_sets(s, n, sub, held):
    """Fixed normal-only sets of stage s (identical construction to FORENSIC_06A)."""
    cur, old = C.STREAM[s - 1], list(C.STREAM[:s - 1])
    idx = C.buffer_index(SEED, n, old)
    old_x = torch.cat([held[c] if c in C.EVAL else F.rows(c, sub[c]["eval"]) for c in old])
    groups = [c for c in old for _ in range(len(held[c]) if c in C.EVAL else len(sub[c]["eval"]))]
    replay_keys = [(d, i) for d in old for i in idx[d]]
    return {"current": cur, "old": old, "idx": idx, "replay_keys": replay_keys,
            "replay": torch.stack([C.load_features(d)[i] for d, i in replay_keys]),
            "old_x": old_x, "groups": groups, "cur_x": F.rows(cur, sub[cur]["cur"])}


def probe_images(s, sub, held):
    old = C.STREAM[:s - 1]
    return sum(len(held[c]) if c in C.EVAL else len(sub[c]["eval"]) for c in old) + F.N_CUR + C.CAPACITY


def rep_cov(dec, st, device):
    return F.mean_cov([F.tnorm(covs(dec, C.load_features(d)[st["idx"][d]], device)) for d in st["old"]])


def oracle_cov(dec, st, sub, device):
    return F.mean_cov([F.tnorm(covs(dec, F.rows(c, sub[c]["oracle"]), device)) for c in st["old"]])


def run_dir(arm):
    return OUT / f"runs/CL06B_{arm}_seed{SEED}"


# --- prereg ------------------------------------------------------------------------------------------------------

def source_hashes():
    return {s: sha(D.ROOT / s) for s in SOURCES}


def prereg():
    require(not PREREG.exists(), f"already preregistered: {PREREG}")
    proto = RB.verify_lock()
    n = R1.n_train(proto)
    sub = F.subsets(n)
    ck = {s: D.read(C.OUT / f"runs/R15_seed{SEED}/STAGE{s}.json") for s in range(1, 10)}
    for s, rec in ck.items():
        require(sha(D.ROOT / rec["checkpoint"]) == rec["checkpoint_sha256"], f"R15 checkpoint {s} changed")
    D.write_json(PREREG, {
        "task": TASK, "utc": DR.now(), "sources": source_hashes(),
        "design": {"seed": SEED, "start_stage": START, "stages": list(STAGES), "iters_per_stage": ITERS,
                   "stage_seeds": {s: RB.stage_seeds("R15", SEED, s) for s in STAGES}, "numerics": "D4prime_fp32core",
                   "layers": LAYERS, "tau": TAU, "arms": {a: {"kind": k, "alpha": al} for a, (k, al) in ARMS.items()},
                   "order": ORDER, "primary_arm": "REP-05", "primary_controls": ["ID", "RAND-05", "NORM-05"],
                   "stress_dose": "REP-10", "rand_seed": RAND_SEED,
                   "rand_construction": "per layer j: Q_j = QR(randn(768,768; torch.Generator CPU seed RAND_SEED+j, "
                                        "float64)); U = Q_j[:, :r_j,s], r_j,s = rank of the RAND arm's own REP basis",
                   "norm_construction": "per layer per step beta = sqrt((||D||^2 - 0.75 ||D U_REP||^2) / ||D||^2), "
                                        "U_REP from the NORM arm's own state: equals ||D (I - 0.5 U U^T)|| / ||D||",
                   "acq_construction": "task t summary at the end of task t on FORENSIC_06A 'mem' normals (24) at the "
                                       "arm's theta_t (R15 checkpoints for t <= 6); stage s: mean over t < s",
                   "oracle_construction": "NON-CAUSAL: arm's theta_{s-1} on FORENSIC_06A 'oracle' normals (8/old task)",
                   "rep_construction": "arm's theta_{s-1} on the stage's Replay-24 members (C.buffer_index)",
                   "optimizer_moments": "untouched by the post-step operator (disclosed)",
                   "probe_every": EVERY, "probe_points": [0] + list(range(EVERY, ITERS, EVERY)) + [ITERS],
                   "subset_seed": F.SUBSET_SEED, "subsets": {c: {k: list(map(int, v)) for k, v in d.items()}
                                                             for c, d in sub.items()},
                   "labels_used": "none: normal images only; no anomaly mask, no Stage D output, no development AP"},
        "rules": RULES, "delta_eq": DELTA_EQ, "tolerances": TOL, "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED},
        "checkpoints": {s: {"path": rec["checkpoint"], "sha256": rec["checkpoint_sha256"]} for s, rec in ck.items()},
        "budget": {"cap_gpu_h": BUDGET_H, "overrun_stop_h": OVERRUN_STOP_H, "preflight_iters": PF_ITERS}})
    print("preregistered", sha(PREREG))


def amendments():
    return [D.read(p) for p in sorted(OUT.glob("PREREG_06B_AMENDMENT_*.json"))]


def expected_sources():
    """Preregistered source hashes, updated in order by every amendment for the files it lists."""
    expected = dict(D.read(PREREG)["sources"])
    for am in amendments():
        require(am["prereg_sha256"] == sha(PREREG), "amendment belongs to another preregistration")
        for s, rec in am["sources"].items():
            require(expected[s] == rec["locked_sha256"], f"amendment chain broken: {s}")
            expected[s] = rec["amended_sha256"]
    return expected


def verify_prereg():
    pr = D.read(PREREG)
    bad = {s: v for s, v in expected_sources().items() if sha(D.ROOT / s) != v}
    require(not bad, f"source changed after preregistration: {sorted(bad)}")
    return pr


def budget():
    b = [am["budget"] for am in amendments() if "budget" in am]
    return (b[-1]["cap_gpu_h"], b[-1]["overrun_stop_h"]) if b else (BUDGET_H, OVERRUN_STOP_H)


def amend_engineering(reason, seen):
    """Engineering-only amendment (code that does not change any computation, set, rule or arm)."""
    path = OUT / f"PREREG_06B_AMENDMENT_{len(amendments()) + 1}.json"
    require(not path.exists(), f"amendment exists: {path}")
    changed = {s: {"locked_sha256": v, "amended_sha256": sha(D.ROOT / s)}
               for s, v in expected_sources().items() if sha(D.ROOT / s) != v}
    require(changed and set(changed) <= {SRC}, f"unexpected source changes: {sorted(changed)}")
    D.write_json(path, {"prereg_sha256": sha(PREREG), "sources": changed, "reason": reason,
                        "unchanged": "arms, alphas, bases, seeds, stages, steps, measurements, sets, rules, decision "
                                     "code, tolerances, budget",
                        "results_seen_before_amendment": seen, "utc": DR.now()})
    print("amendment written", path.name, sha(path))


def gate():
    """Launch gate: preregistered (or amended) sources, integrity-passing preflight, projection within the cap."""
    verify_prereg()
    pf = D.read(PREFLIGHT)
    cap, _ = budget()
    ok = pf["prereg_sha256"] == sha(PREREG) and pf["integrity_pass"] and pf["projected_gpu_h"] <= cap
    require(ok, f"gate closed: integrity {pf['integrity_pass']}, projected {pf['projected_gpu_h']:.2f} h, cap {cap} h")
    return pf


def amend_budget(cap, stop, authorization):
    """Amendment 1 (budget only): written after an integrity-passing HOLD_BUDGET preflight and before any training."""
    require(not AMEND.exists(), f"amendment exists: {AMEND}")
    pr, pf = D.read(PREREG), D.read(PREFLIGHT)
    require(pf["integrity_pass"] and pf["status"] == "HOLD_BUDGET", "only for an integrity-passing budget HOLD")
    require(not list(OUT.glob(f"runs/CL06B_*_seed{SEED}")), "an arm has already started")
    changed = {s: {"locked_sha256": v, "amended_sha256": sha(D.ROOT / s)}
               for s, v in pr["sources"].items() if sha(D.ROOT / s) != v}
    require(set(changed) <= {SRC, "exps/nlcad/chain_06b.sh"}, f"unexpected source changes: {sorted(changed)}")
    D.write_json(AMEND, {
        "prereg_sha256": sha(PREREG), "preflight_sha256": sha(PREFLIGHT), "sources": changed,
        "budget": {"cap_gpu_h": cap, "overrun_stop_h": stop, "previous": [BUDGET_H, OVERRUN_STOP_H],
                   "preflight_projection_gpu_h": pf["projected_gpu_h"]},
        "user_authorization": authorization,
        "code_change": "launch gate accepts an integrity-passing preflight whose projection is within the amended cap; "
                       "resource stop uses the amended overrun limit; source check honours this record",
        "unchanged": "arms, alphas, bases, seeds, stages, steps, measurements, sets, rules, decision code, tolerances",
        "results_seen_before_amendment": "preflight timing and integrity only; no arm trained, no efficacy outcome",
        "utc": DR.now()})
    print("amendment written", sha(AMEND))


# --- GPU setup shared by preflight and train -----------------------------------------------------------------------

def setup():
    require(torch.cuda.is_available(), "CUDA required")
    pr = verify_prereg()
    DR.configure()
    R.cap()
    proto = RB.verify_lock()
    NUM.use(True)
    n = R1.n_train(proto)
    sub = F.subsets(n)
    require({c: {k: list(map(int, v)) for k, v in d.items()} for c, d in sub.items()} == pr["design"]["subsets"],
            "subsets differ from the preregistration")
    held = {c: DR.load_split("heldout", c) for c in C.EVAL}
    return pr, n, sub, held


def r15_state(s):
    rec = D.read(C.OUT / f"runs/R15_seed{SEED}/STAGE{s}.json")
    require(sha(D.ROOT / rec["checkpoint"]) == rec["checkpoint_sha256"], f"R15 checkpoint {s} changed")
    return torch.load(D.ROOT / rec["checkpoint"], weights_only=True)["state"], rec["checkpoint_sha256"]


def acq_summaries_shared(sub, device):
    """ACQ summaries of tasks 1..START at the shared R15 theta_t (written at acquisition; causal for every arm)."""
    out = {}
    with rng_fork(device):
        aux = O.Decoder(768, 12).to(device)
    for t in range(1, START + 1):
        c = C.STREAM[t - 1]
        aux.load_state_dict(r15_state(t)[0])
        out[c] = F.tnorm(covs(aux, F.rows(c, sub[c]["mem"]), device))
    del aux
    torch.cuda.empty_cache()
    return out


# --- preflight ---------------------------------------------------------------------------------------------------

def preflight():
    require(not PREFLIGHT.exists(), f"preflight already written: {PREFLIGHT}")
    pr, n, sub, held = setup()
    dev = "cuda"
    sd6, _ = r15_state(START)
    dec = DR.new_decoder(SEED)
    st = stage_sets(7, n, sub, held)
    current = C.load_features(st["current"])
    dseed, gseed = RB.stage_seeds("R15", SEED, 7)
    res = {"prereg_sha256": sha(PREREG), "iters": PF_ITERS, "utc": DR.now()}

    def fresh():
        dec.load_state_dict(sd6)
        torch.manual_seed(dseed)
        return torch.Generator().manual_seed(gseed)

    # REP basis (timed) at theta_6
    dec.load_state_dict(sd6)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    U_rep = bases(rep_cov(dec, st, dev))
    torch.cuda.synchronize()
    res["rep_basis_s"] = time.perf_counter() - t0
    res["rep_ranks"] = {l: int(U_rep[l].shape[1]) for l in LAYERS}
    # 1. reference: the locked loop
    gen = fresh()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    R1.train_stage_guarded(dec, current, PF_ITERS, gen, st["replay"])
    torch.cuda.synchronize()
    res["ref_s_per_step"] = (time.perf_counter() - t0) / PF_ITERS
    ref = {k: v.detach().clone() for k, v in dec.state_dict().items()}
    ref_gen = gen.get_state()
    # 2. this loop, ID operator + one probe inside the segment
    tp = []

    def probe(i):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        measure(dec, st["old_x"], dev)
        measure(dec, st["cur_x"], dev)
        measure(dec, st["replay"], dev)
        torch.cuda.synchronize()
        tp.append(time.perf_counter() - t1)

    gen = fresh()
    op = Protector(dec, "id", 0.0, U_rep, U_rep, dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_stage_protected(dec, current, PF_ITERS, gen, st["replay"], op=op, probe=probe, every=64)
    torch.cuda.synchronize()
    res["probe_s_stage7"] = float(np.mean(tp))
    res["probe_images_stage7"] = len(st["old_x"]) + len(st["cur_x"]) + len(st["replay"])
    res["id_s_per_step"] = (time.perf_counter() - t0 - sum(tp)) / PF_ITERS
    state = dec.state_dict()
    res["loop_parity_bitwise"] = all(torch.equal(state[k], ref[k]) for k in ref)
    res["sampler_generator_parity"] = bool(torch.equal(gen.get_state(), ref_gen))
    res["id_operator"] = operator_checks(op.table(), "id", 0.0)
    # 3. REP-05 operator: overhead and specification checks
    for name in ("REP-05", "REP-10"):
        gen = fresh()
        kind, alpha = ARMS[name]
        op = Protector(dec, kind, alpha, U_rep, U_rep, dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_stage_protected(dec, current, PF_ITERS if name == "REP-05" else 24, gen, st["replay"], op=op)
        torch.cuda.synchronize()
        chk = operator_checks(op.table(), kind, alpha)
        res[name] = {"s_per_step": (time.perf_counter() - t0) / (PF_ITERS if name == "REP-05" else 24),
                     "operator": chk, "operator_ok": operator_ok(chk, kind, alpha), "finite": params_finite(dec)}
    # 4. NORM operator spec check (short)
    gen = fresh()
    op = Protector(dec, "norm", 0.5, U_rep, U_rep, dev)
    train_stage_protected(dec, current, 24, gen, st["replay"], op=op)
    chk = operator_checks(op.table(), "norm", 0.5)
    res["NORM-05"] = {"operator": chk, "operator_ok": operator_ok(chk, "norm", 0.5)}
    # 5. ORACLE / ACQ memory cost
    dec.load_state_dict(sd6)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    bases(oracle_cov(dec, st, sub, dev))
    torch.cuda.synchronize()
    res["oracle_basis_s_stage7"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    acq_summaries_shared(sub, dev)
    torch.cuda.synchronize()
    res["acq_shared_summaries_s"] = time.perf_counter() - t0
    res["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    # 6. projection (sequential arms, device wall time)
    proj = {}
    n_img = {s: probe_images(s, sub, held) for s in STAGES}
    require(n_img[7] == res["probe_images_stage7"], "probe image count mismatch")
    n_probe = len(pr["design"]["probe_points"])
    for arm in ORDER:
        kind, _ = ARMS[arm]
        step = res["id_s_per_step"] if kind == "id" else res["REP-05"]["s_per_step"]
        sec = 60.0 + (res["acq_shared_summaries_s"] if kind == "acq" else 0.0)       # setup, theta_6 baseline
        for s in STAGES:
            sec += ITERS * step + n_probe * res["probe_s_stage7"] * n_img[s] / n_img[7] + res["rep_basis_s"] + 10.0
            if kind == "oracle":
                sec += res["oracle_basis_s_stage7"] * (s - 1) / 6
            if kind == "acq":
                sec += 2 * res["rep_basis_s"]
        proj[arm] = sec
    res["projection_s"] = proj
    res["projected_gpu_h"] = sum(proj.values()) / 3600
    integrity = (res["loop_parity_bitwise"] and res["sampler_generator_parity"] and res["REP-05"]["operator_ok"]
                 and res["REP-10"]["operator_ok"] and res["NORM-05"]["operator_ok"] and res["REP-05"]["finite"])
    res["integrity_pass"] = bool(integrity)
    res["budget_pass"] = bool(res["projected_gpu_h"] <= BUDGET_H)
    res["status"] = "PASS" if integrity and res["budget_pass"] else ("HOLD_BUDGET" if integrity else "FAIL_INTEGRITY")
    D.write_json(PREFLIGHT, res)
    print(res["status"], "projected GPU-h", round(res["projected_gpu_h"], 3), "parity", res["loop_parity_bitwise"],
          flush=True)


# --- train one arm -----------------------------------------------------------------------------------------------

def train(arm):
    require(arm in ARMS, f"unknown arm {arm}")
    gate()
    _, stop_h = budget()
    pr, n, sub, held = setup()
    dev = "cuda"
    kind, alpha = ARMS[arm]
    run = run_dir(arm)
    run.mkdir(parents=True, exist_ok=True)
    spent = sum(D.read(p)["gpu_seconds"] for p in OUT.glob(f"runs/CL06B_*_seed{SEED}/STAGE*.json")) / 3600
    torch.cuda.reset_peak_memory_stats()
    sd6, sha6 = r15_state(START)
    dec = DR.new_decoder(SEED)
    dec.load_state_dict(sd6)
    st7 = stage_sets(7, n, sub, held)
    o6 = [i for i, g in enumerate(st7["groups"]) if g in O6]
    require(o6 == list(range(len(o6))), "O6 must be the prefix of every stage's old set")
    base_m, base_pool = measure(dec, st7["old_x"], dev)
    if (run / "BASE.json").exists():
        require(D.read(run / "BASE.json")["theta6_old"] == base_m, "theta_6 baseline not reproducible")
    else:
        D.write_json(run / "BASE.json", {"prereg_sha256": sha(PREREG), "start_sha256": sha6, "groups": st7["groups"],
                                         "theta6_old": base_m})
    acq = acq_summaries_shared(sub, dev) if kind == "acq" else {}
    for s in STAGES:
        js = run / f"STAGE{s}.json"
        c = C.STREAM[s - 1]
        if js.exists():                                          # resume at stage granularity
            rec = D.read(js)
            require(rec["prereg_sha256"] == sha(PREREG) and rec["integrity_pass"], f"stage {s} record invalid")
            if kind == "id":
                dec.load_state_dict(r15_state(s)[0])
            else:
                path = D.ROOT / rec["checkpoint"]
                require(sha(path) == rec["checkpoint_sha256"], f"stage {s} checkpoint changed")
                dec.load_state_dict(torch.load(path, weights_only=True)["state"])
            if kind == "acq":
                acq[c] = F.tnorm(covs(dec, F.rows(c, sub[c]["mem"]), dev))
            print(arm, s, "resumed", flush=True)
            continue
        require(spent <= stop_h, f"resource stop: cumulative GPU time {spent:.2f} h > {stop_h} h")
        st = st7 if s == 7 else stage_sets(s, n, sub, held)
        torch.cuda.synchronize()
        t_stage = time.perf_counter()
        # boundary memory (own state, eval mode, forked RNG)
        t0 = time.perf_counter()
        cov_rep = rep_cov(dec, st, dev)
        U_rep = bases(cov_rep)
        U_act = {"id": U_rep, "rep": U_rep, "norm": U_rep,
                 "rand": rand_bases({l: U_rep[l].shape[1] for l in LAYERS}),
                 "acq": bases(F.mean_cov([acq[d] for d in st["old"]])) if kind == "acq" else None,
                 "oracle": bases(oracle_cov(dec, st, sub, dev)) if kind == "oracle" else None}[kind]
        torch.cuda.synchronize()
        basis_s = time.perf_counter() - t0
        mem = {"rep_rank": {l: int(U_rep[l].shape[1]) for l in LAYERS},
               "act_rank": {l: int(U_act[l].shape[1]) for l in LAYERS},
               "act_capture_of_replay_cov": capture(cov_rep, U_act),
               "act_basis_bytes_fp32": int(sum(768 * U_act[l].shape[1] * 4 for l in LAYERS)),
               "rep_basis_bytes_fp32": int(sum(768 * U_rep[l].shape[1] * 4 for l in LAYERS))}
        # probes
        probes, refs, t_probe, blocks = {}, {}, [], []
        snap = {"state": {k: v.detach().cpu().clone() for k, v in dec.named_parameters()}}

        def probe(i):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            require(params_finite(dec), f"non-finite parameters at stage {s} step {i}")
            rec = {}
            for name, x in (("old", st["old_x"]), ("cur", st["cur_x"]), ("rep", st["replay"])):
                m, pooled = measure(dec, x, dev)
                if i == 0:
                    refs[name] = pooled
                m["drift_stage"] = drift(pooled, refs[name])
                if name == "old":
                    m["drift_theta6_O6"] = drift(pooled[:len(o6)], base_pool[:len(o6)])
                rec[name] = m
            now = {k: v.detach().cpu() for k, v in dec.named_parameters()}
            blk = {b: 0.0 for b in BLOCKS}
            for k, v in now.items():
                blk[block_of(k)] += float((v.double() - snap["state"][k].double()).pow(2).sum())
            snap["state"] = now
            blocks.append({"step": i, "block_displacement_norm": {b: math.sqrt(v) for b, v in blk.items()}})
            probes[str(i)] = rec
            torch.cuda.synchronize()
            t_probe.append(time.perf_counter() - t1)

        probe(0)
        dseed, gseed = RB.stage_seeds("R15", SEED, s)
        torch.manual_seed(dseed)
        gen = torch.Generator().manual_seed(gseed)
        op = Protector(dec, kind, alpha, U_act, U_rep, dev)
        current = C.load_features(c)
        t0 = time.perf_counter()
        losses, skips = train_stage_protected(dec, current, ITERS, gen, st["replay"], op=op, probe=probe)
        torch.cuda.synchronize()
        loop_s = time.perf_counter() - t0
        tab = op.table()
        chk = operator_checks(tab, kind, alpha)
        ok = operator_ok(chk, kind, alpha) and params_finite(dec)
        state = {k: v.detach().cpu() for k, v in dec.state_dict().items()}
        parity = None
        if kind == "id":
            ref_state, ref_sha = r15_state(s)
            parity = all(torch.equal(state[k], ref_state[k]) for k in ref_state) and state.keys() == ref_state.keys()
            ok = ok and parity
            ck = {"checkpoint": None, "checkpoint_sha256": ref_sha, "note": "bitwise equal to the R15 checkpoint"}
        else:
            path = run / f"stage{s}_{c}.pt"
            ck = {"checkpoint": str(path.relative_to(D.ROOT)),
                  "checkpoint_sha256": DR.save_ckpt(dec, path, {"arm": arm, "seed": SEED, "stage": s, "name": c})}
        if kind == "acq":                                         # task s summary written at the end of task s
            acq[c] = F.tnorm(covs(dec, F.rows(c, sub[c]["mem"]), dev))
        stats_path = run / f"STEPSTATS{s}.pt"
        require(not stats_path.exists(), f"exists: {stats_path}")
        torch.save({"cols": COLS, "layers": LAYERS, "table": tab}, stats_path)
        torch.cuda.synchronize()
        gpu_s = time.perf_counter() - t_stage
        k = {cc: i for i, cc in enumerate(COLS)}
        win = []
        for a0 in range(0, len(tab), EVERY):
            w = tab[a0:a0 + EVERY]
            tot = float(w[..., k["e_tot"]].sum())
            win.append({"steps": [a0, a0 + len(w)], "removed_energy_fraction": float(w[..., k["removed"]].sum()) / tot,
                        "act_energy_fraction": float(w[..., k["e_act"]].sum()) / tot,
                        "rep_energy_fraction": float(w[..., k["e_rep"]].sum()) / tot,
                        "decay_energy_fraction": float(w[..., k["e_decay"]].sum()) / tot,
                        "mean_beta": float(w[..., k["beta"]].mean()),
                        "attn_proj_update_norm_per_layer": {l: float(w[:, j, k["e_real"]].sqrt().sum())
                                                            for j, l in enumerate(LAYERS)}})
        rec = {"arm": arm, "kind": kind, "alpha": alpha, "stage": s, "current": c, "prereg_sha256": sha(PREREG),
               "seeds": [dseed, gseed], "groups_old": st["groups"], "replay_keys": st["replay_keys"],
               "losses_every50": [float(np.nanmean(losses[i:i + 50])) for i in range(0, len(losses), 50)],
               "skips": skips, "probes": probes, "block_displacement": blocks, "memory": mem, "windows": win,
               "operator": chk, "identity_state_bitwise_parity": parity, "integrity_pass": bool(ok),
               "stepstats": str(stats_path.relative_to(D.ROOT)), "stepstats_sha256": sha(stats_path),
               "loop_seconds_incl_probes": loop_s, "probe_seconds": sum(t_probe), "basis_seconds": basis_s,
               "gpu_seconds": gpu_s, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
               "temporary_bytes": {"W_before_fp32": int(8 * 768 * 768 * 4), "bases_on_device_fp32":
                                   mem["act_basis_bytes_fp32"] + (0 if op.same else mem["rep_basis_bytes_fp32"])},
               **ck, "utc": DR.now()}
        D.write_json(js, rec)
        spent += gpu_s / 3600
        print(arm, s, c, f"{gpu_s:.0f}s", "integrity", ok, "parity", parity,
              "old", round(float(np.mean(probes[str(ITERS)]["old"]["loss"])), 5),
              "cur", round(float(np.mean(probes[str(ITERS)]["cur"]["loss"])), 5), flush=True)
        require(ok, f"integrity failure at {arm} stage {s}: {chk} parity={parity}")
    D.write_json(run / "TRAIN.json", {"arm": arm, "finished_utc": DR.now(), "prereg_sha256": sha(PREREG),
                                      "persistent_decoder_bytes": DR.state_bytes(dec),
                                      "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})


# --- analysis -----------------------------------------------------------------------------------------------------

def strat_boot(diff, groups, B=BOOT_B, seed=BOOT_SEED):
    """Category-stratified paired bootstrap of the category-macro mean of per-image differences."""
    rng = np.random.default_rng(seed)
    diff, groups = np.asarray(diff, dtype=np.float64), np.asarray(groups)
    cats = list(dict.fromkeys(groups.tolist()))
    draws = np.zeros(B)
    for c in cats:
        v = diff[groups == c]
        draws += v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    draws /= len(cats)
    point = float(np.mean([diff[groups == c].mean() for c in cats]))
    return {"point": point, "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "ci90": [float(np.quantile(draws, .05)), float(np.quantile(draws, .95))]}


def per_category(diff, groups):
    diff, groups = np.asarray(diff), np.asarray(groups)
    return {c: float(diff[groups == c].mean()) for c in dict.fromkeys(groups.tolist())}


def acq_status(r):
    """(retained >= 0.90, prohibitive < 0.75) for one stage's retention record, with the degenerate rule."""
    if r["A_ID"] >= 0.01:
        return r["R"] >= 0.90, r["R"] < 0.75
    return r["excess"] <= 0.001, r["excess"] > 0.005


def equivalent(d, delta=DELTA_EQ):
    return -delta < d["ci90"][0] and d["ci90"][1] < delta


def decide(sm):
    """Predeclared decision from the summary (see RULES). sm keys: integrity, D, T, D_ci, T_ci, diff, tdiff, retention,
    cat_better. diff/tdiff keys 'A~B' hold strat_boot(A - B) records."""
    require(sm["integrity"], "integrity failed: no decision")
    D_, T_ = sm["D"], sm["T"]
    reasons = {}
    reasons["P0"] = D_["ID"] > 0 and sm["D_ci"]["ID"][0] > 0
    if not reasons["P0"]:
        return "CLOSED_LOOP_INCONCLUSIVE", reasons
    prot = 1 - D_["REP-05"] / D_["ID"]
    reasons["protection_REP05"] = prot
    reasons["S1"] = prot >= 0.25 and sm["diff"]["REP-05~ID"]["ci95"][1] < 0
    tail_on = T_["ID"] > 0 and sm["T_ci"]["ID"][0] > 0
    reasons["tail_exercised"] = tail_on
    reasons["S2"] = bool(tail_on and 1 - T_["REP-05"] / T_["ID"] >= 0.25 and sm["tdiff"]["REP-05~ID"]["ci95"][1] < 0)
    reasons["S3"] = all(sm["diff"][f"REP-05~{c}"]["ci95"][1] < 0 and sm["diff"][f"REP-05~{c}"]["point"] <= -0.10 * D_["ID"]
                        for c in ("RAND-05", "NORM-05"))
    st = [acq_status(sm["retention"]["REP-05"][s]) for s in sorted(sm["retention"]["REP-05"])]
    reasons["S4"] = all(a for a, _ in st)
    reasons["S5"] = sm["cat_better"]["REP-05"] >= 4
    reasons["R1"] = prot < 0.10
    reasons["R2"] = D_["REP-05"] >= min(D_["RAND-05"], D_["NORM-05"])
    reasons["R3"] = any(p for _, p in st)
    if all(reasons[k] for k in ("S1", "S2", "S3", "S4", "S5")):
        return "CLOSED_LOOP_PROTECTION_SUPPORTED", reasons
    if reasons["R1"] or reasons["R2"] or reasons["R3"]:
        return "CLOSED_LOOP_PROTECTION_REJECTED", reasons
    return "CLOSED_LOOP_INCONCLUSIVE", reasons


def secondary(sm, s7):
    """Predeclared readings Q1-Q5 (RULES)."""
    d = sm["diff"]
    out = {"Q1": {}}
    for a in ("REP-10", "REP-05"):
        x = s7[f"{a}~ID"]
        out["Q1"][a] = {"stage7": x, "reading": "reversed" if x["ci95"][1] < 0 else
                        ("still harmful" if x["ci95"][0] > 0 else "undetermined"),
                        "cumulative_stage9": d[f"{a}~ID"]}
    minR = {a: min((r["R"] for r in sm["retention"][a].values() if r["A_ID"] >= 0.01), default=float("nan"))
            for a in ("REP-05", "REP-10")}
    x = d["REP-05~REP-10"]
    if x["ci95"][1] < 0 or (equivalent(x) and minR["REP-05"] - minR["REP-10"] >= 0.05):
        q2 = "soft better"
    elif x["ci95"][0] > 0 and minR["REP-10"] >= 0.90:
        q2 = "hard better"
    else:
        q2 = "undetermined"
    out["Q2"] = {"diff": x, "min_retention": minR, "reading": q2}
    x = d["REP-05~NORM-05"]
    out["Q3"] = {"diff": x, "reading": "direction adds value" if x["ci95"][1] < 0 else
                 ("scaling as good or better" if x["ci95"][0] > 0 or equivalent(x) else "undetermined")}
    x = d["REP-05~ACQ-05"]
    out["Q4"] = {"diff": x, "reading": "REP better" if x["ci95"][1] < 0 else "ACQ better" if x["ci95"][0] > 0 else
                 "equivalent" if equivalent(x) else "undetermined"}
    x = d["ORACLE-05~REP-05"]
    out["Q5"] = {"diff": x, "reading": "ORACLE better" if x["ci95"][1] < 0 and x["point"] <= -DELTA_EQ else
                 "equivalent" if equivalent(x) else
                 "REP better" if x["ci95"][0] > 0 and x["point"] >= DELTA_EQ else "undetermined"}
    return out


def analyze():
    require(not DECISION.exists(), "decision already written")
    pr = verify_prereg()
    recs = {a: {s: D.read(run_dir(a) / f"STAGE{s}.json") for s in STAGES} for a in ORDER}
    base = {a: D.read(run_dir(a) / "BASE.json") for a in ORDER}
    integrity = all(r["integrity_pass"] and r["prereg_sha256"] == sha(PREREG) for a in ORDER for r in recs[a].values())
    integrity &= all(recs["ID"][s]["identity_state_bitwise_parity"] for s in STAGES)
    integrity &= all(base[a]["theta6_old"] == base["ID"]["theta6_old"] for a in ORDER)
    groups = base["ID"]["groups"]
    n6 = len(groups)
    require(set(groups) == set(O6) and all(recs[a][9]["groups_old"][:n6] == groups for a in ORDER), "O6 prefix")
    L6, P6 = np.array(base["ID"]["theta6_old"]["loss"]), np.array(base["ID"]["theta6_old"]["p99"])
    end = lambda a, s, key, part="old": np.array(recs[a][s]["probes"][str(ITERS)][part][key])
    L9 = {a: end(a, 9, "loss")[:n6] for a in ORDER}
    P9 = {a: end(a, 9, "p99")[:n6] for a in ORDER}
    sm = {"integrity": bool(integrity), "D": {}, "T": {}, "D_ci": {}, "T_ci": {}, "diff": {}, "tdiff": {},
          "retention": {}, "cat_better": {}, "per_category_D": {}}
    for a in ORDER:
        bd, bt = strat_boot(L9[a] - L6, groups), strat_boot(P9[a] - P6, groups)
        sm["D"][a], sm["D_ci"][a], sm["T"][a], sm["T_ci"][a] = bd["point"], bd["ci95"], bt["point"], bt["ci95"]
        sm["per_category_D"][a] = per_category(L9[a] - L6, groups)
    pairs = [(a, "ID") for a in ORDER[1:]] + [("REP-05", "RAND-05"), ("REP-05", "NORM-05"), ("REP-05", "REP-10"),
                                              ("REP-05", "ACQ-05"), ("ORACLE-05", "REP-05")]
    for a, b in pairs:
        sm["diff"][f"{a}~{b}"] = strat_boot(L9[a] - L9[b], groups)
        sm["tdiff"][f"{a}~{b}"] = strat_boot(P9[a] - P9[b], groups)
    for a in ORDER:
        sm["cat_better"][a] = sum(sm["per_category_D"][a][c] < sm["per_category_D"]["ID"][c] for c in O6)
        sm["retention"][a] = {}
        for s in STAGES:
            l0 = float(np.mean(recs["ID"][s]["probes"]["0"]["cur"]["loss"]))
            lid = float(np.mean(end("ID", s, "loss", "cur")))
            la = float(np.mean(end(a, s, "loss", "cur")))
            A = l0 - lid
            sm["retention"][a][s] = {"A_ID": A, "R": (l0 - la) / A if A >= 0.01 else None, "excess": la - lid,
                                     "own_start_R": (float(np.mean(recs[a][s]["probes"]["0"]["cur"]["loss"])) - la) / A
                                     if A >= 0.01 else None}
    s7 = {f"{a}~ID": strat_boot(end(a, 7, "loss")[:n6] - end("ID", 7, "loss")[:n6], groups) for a in ORDER[1:]}
    label, reasons = decide(sm)
    sec = secondary(sm, s7)
    traj = {a: {s: {i: {part: {"loss_mean": float(np.mean(p[part]["loss"])), "p99_mean": float(np.mean(p[part]["p99"])),
                               "drift_stage_mean": float(np.mean(p[part]["drift_stage"]))}
                        for part in ("old", "cur", "rep")} for i, p in recs[a][s]["probes"].items()}
                for s in STAGES} for a in ORDER}
    ledger = {a: {"gpu_seconds": sum(recs[a][s]["gpu_seconds"] for s in STAGES),
                  "probe_seconds": sum(recs[a][s]["probe_seconds"] for s in STAGES),
                  "basis_seconds": sum(recs[a][s]["basis_seconds"] for s in STAGES),
                  "peak_allocated_mib": max(recs[a][s]["peak_allocated_mib"] for s in STAGES),
                  "memory": {s: recs[a][s]["memory"] for s in STAGES},
                  "temporary_bytes": recs[a][9]["temporary_bytes"],
                  "removed_energy_fraction": {s: [w["removed_energy_fraction"] for w in recs[a][s]["windows"]]
                                              for s in STAGES}} for a in ORDER}
    D.write_json(ANALYSIS, {"prereg_sha256": sha(PREREG), "summary": sm, "stage7": s7, "secondary": sec,
                            "trajectories": traj, "ledger": ledger, "utc": DR.now()})
    D.write_json(DECISION, {"prereg_sha256": sha(PREREG), "analysis_sha256": sha(ANALYSIS), "label": label,
                            "reasons": reasons, "rules": pr["rules"], "utc": DR.now()})
    print(label, {k: v for k, v in reasons.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prereg", "preflight", "amend", "amend-eng", "gate", "train", "analyze"])
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--cap", type=float)
    ap.add_argument("--stop", type=float)
    ap.add_argument("--authorization")
    ap.add_argument("--reason")
    ap.add_argument("--seen")
    a = ap.parse_args()
    {"prereg": prereg, "preflight": preflight, "amend": lambda: amend_budget(a.cap, a.stop, a.authorization),
     "amend-eng": lambda: amend_engineering(a.reason, a.seen),
     "gate": lambda: print("gate open", gate()["projected_gpu_h"]), "train": lambda: train(a.arm),
     "analyze": analyze}[a.cmd]()


if __name__ == "__main__":
    main()
