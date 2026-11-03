"""NLCAD full-stream optimization-shock audit (Job B of task_nlcad_06c_execution_and_shock_audit). CPU only.

Uses existing RH-01b R15 artifacts (seeds 2711, 2712, 2713; stages 1-15): the 50-step training-loss curves, the
recorded held-out normal losses, and the stage checkpoints. No training, no anomaly label, no AP.

Commands
  plan      write SHOCK_AUDIT_PLAN.json (rules and metrics; written before any full-stream summary is computed)
  evaluate  evaluate every R15 checkpoint theta_s on the normal eval images of categories 1..s (CPU, FP32, eval mode,
            the 06B `measure`), and record per-component parameter drift ||theta_s - theta_{s-1}||
  analyze   apply the plan -> SHOCK_AUDIT.json

Disclosure (also in the plan): the shock rule (50-step training-loss peak >= 1.5 x the preceding minimum) was chosen in
the NLCAD-06B-WASHOUT after inspecting stages 7-8 of seed 2711, and that phase already printed the peak ratios of all
15 stages of all three seeds and the bottle/carpet/hazelnut held-out changes of stages 7-15. The audit is therefore
exploratory and not blind to those quantities; it reuses the same trajectories and is no independent confirmation.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nlcad import closed_loop_06b as M
from exps.nlcad import forensic_06a as F
from exps.nlcad.washout_06b import group_norms, strkeys
from exps.rh01 import numerics as NUM
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

SEEDS = (2711, 2712, 2713)
STAGES = tuple(range(1, 16))
SHOCK_X = 1.5
SENSITIVITY_X = (1.3, 2.0)
PEAK_LR_FRAC = 0.8                   # "near peak learning rate": lr >= 0.8 x lr_max at the shock's peak window
PERM_B, PERM_SEED = 10000, 6063
OUT = M.OUT
PLAN, EVAL, AUDIT = OUT / "SHOCK_AUDIT_PLAN.json", OUT / "SHOCK_AUDIT_EVAL.json", OUT / "SHOCK_AUDIT.json"
SRC = "exps/nlcad/shock_audit.py"
TASK = "agents/tasks/task_nlcad_06c_execution_and_shock_audit.md"

RULES = {
    "shock": "stage s is a shock stage iff r_s = max_{w >= 2} L_w / min_{1 <= v < w} L_v >= 1.5, L_w = the w-th "
             "50-step mean of the R15 training loss (loss_curve_every50; windows 0 and 1 are warm-up). Sensitivity: "
             "thresholds 1.3 and 2.0. Peak location = start step of the peak window",
    "peak_lr": "lr at the peak window's midpoint from the exact WarmCosine schedule of a 667-step stage; 'near peak LR' "
               "iff lr >= 0.8 x lr_max",
    "eval_set": "per category: DRF-01 held-out normals (bottle, carpet, hazelnut; never trained) or the 6 FORENSIC_06A "
                "'eval' training normals (trained at acquisition, never held in the Replay buffer of any of the three "
                "seeds); exclusions: images held by any seed's buffer (cable#68, transistor#134)",
    "damage": "for transition s (theta_{s-1} -> theta_s) and old category c (learned at t_c < s): d_c(s) = mean_c "
              "L(theta_s) - mean_c L(theta_{s-1}) (global-cosine loss; tail: map p99). Category-balanced = mean over "
              "old categories; worst = max over old categories. Transitions s = 2..15",
    "recovery": "next-transition change d(s+1) after shock vs non-shock transitions; lag-1 correlation of the "
                "category-balanced damage series per seed; per category, fraction of positive d_c(s) followed by a "
                "negative d_c(s+1)",
    "association": "shock vs non-shock transitions: difference of mean category-balanced damage (and tail, worst), per "
                   "seed and pooled; descriptive p-value from a within-seed permutation of shock labels across that "
                   "seed's transitions (B = 1e4). Transitions within a seed share one trajectory: not independent",
    "drift": "||theta_s - theta_{s-1}|| per component group; Spearman with category-balanced damage (per seed, pooled); "
             "cumulative drift from theta_{t_c} versus cumulative damage per category",
    "representativeness": "rank of stages 7 and 8 among stages 2-15 in shock ratio and in damage, per seed",
    "categories": "cumulative damage L_c(theta_15) - L_c(theta_{t_c}) per category and seed; Spearman of the category "
                  "ranking between seed pairs; share of the summed damage carried by the top category",
    "validation": "the CPU global-cosine means of bottle/carpet/hazelnut must reproduce the R15 records' "
                  "heldout_normal global_cosine (max |diff| reported)",
    "limits": "no per-step update/gradient statistics exist for R15 (only for the 06B/06C runs, stages 7-9); shock "
              "detection is limited to 50-step resolution; no anomaly label is used and nothing is said about AP"}


def schedule_lr():
    """The exact per-step learning rate of a 667-step stage, as logged by the bitwise-R15 06B identity run."""
    tab = torch.load(M.run_dir("ID") / "STEPSTATS7.pt", weights_only=True)
    return tab["table"][:, 0, tab["cols"].index("lr")].numpy()


def shock_ratio(c):
    pk = max(range(2, len(c)), key=lambda i: c[i])
    return float(c[pk] / min(c[1:pk])), 50 * pk


def record(seed, s):
    return D.read(C.OUT / f"runs/R15_seed{seed}/STAGE{s}.json")


def eval_sets():
    n = R1.n_train(RB.verify_lock())
    sub = F.subsets(n)
    ever = {c: set() for c in C.STREAM}
    for seed in SEEDS:
        for s in range(2, len(C.STREAM) + 1):
            for c, idx in C.buffer_index(seed, n, list(C.STREAM[:s - 1])).items():
                ever[c] |= set(idx)
    held = {c: DR.load_split("heldout", c) for c in C.EVAL}
    ids, xs = {}, {}
    for c in C.STREAM:
        if c in C.EVAL:
            ids[c], xs[c] = list(range(len(held[c]))), held[c]
        else:
            ids[c] = [int(i) for i in sub[c]["eval"] if i not in ever[c]]
            xs[c] = F.rows(c, ids[c])
    return ids, xs


def plan():
    require(not PLAN.exists(), f"exists: {PLAN}")
    ids, _ = eval_sets()
    D.write_json(PLAN, strkeys({"task": TASK, "task_sha256": sha(D.ROOT / TASK), "source_sha256": sha(D.ROOT / SRC),
                        "utc": DR.now(), "rules": RULES, "shock_x": SHOCK_X, "sensitivity_x": list(SENSITIVITY_X),
                        "peak_lr_frac": PEAK_LR_FRAC, "permutation": {"B": PERM_B, "seed": PERM_SEED},
                        "eval_ids": ids, "n_eval": {c: len(v) for c, v in ids.items()},
                        "disclosure": __doc__.split("Disclosure (also in the plan): ")[1].strip(),
                        "r15_checkpoints": {seed: {s: record(seed, s)["checkpoint_sha256"] for s in STAGES}
                                            for seed in SEEDS}}))
    print("plan", sha(PLAN))


def evaluate():
    require(PLAN.exists() and not EVAL.exists(), "plan first; evaluation once")
    pl = D.read(PLAN)
    require(pl["source_sha256"] == sha(D.ROOT / SRC), "audit source changed after the plan")
    NUM.use(True)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    ids, xs = eval_sets()
    require(ids == pl["eval_ids"], "eval set differs from the plan")
    dec = O.Decoder(768, 12)
    out = {"per_image": {}, "drift": {}, "seconds": {}}
    for seed in SEEDS:
        t0 = time.perf_counter()
        out["per_image"][seed], out["drift"][seed] = {}, {}
        prev = None
        for s in STAGES:
            rec = record(seed, s)
            path = D.ROOT / rec["checkpoint"]
            require(sha(path) == pl["r15_checkpoints"][str(seed)][str(s)], f"{seed} {s}: checkpoint changed")
            sd = torch.load(path, weights_only=True)["state"]
            dec.load_state_dict(sd)
            res = {}
            for c in C.STREAM[:s]:
                m, _ = M.measure(dec, xs[c], "cpu")
                res[c] = {"loss": m["loss"], "p99": m["p99"]}
            out["per_image"][seed][s] = res
            if prev is not None:
                out["drift"][seed][s] = group_norms({k: sd[k] - prev[k] for k in sd})
            prev = {k: v.clone() for k, v in sd.items()}
            print(seed, s, round(time.perf_counter() - t0, 1), flush=True)
        out["seconds"][seed] = time.perf_counter() - t0
    D.write_json(EVAL, strkeys({"plan_sha256": sha(PLAN), "utc": DR.now(), **out}))
    print("evaluated", sha(EVAL))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze():
    require(PLAN.exists() and EVAL.exists() and not AUDIT.exists(), "plan and evaluation first; analysis once")
    pl, ev = D.read(PLAN), D.read(EVAL)
    require(ev["plan_sha256"] == sha(PLAN), "evaluation belongs to another plan")
    lr = schedule_lr()
    lr_max = float(lr.max())
    t = {c: i + 1 for i, c in enumerate(C.STREAM)}               # stage at which c is learned
    out = {"lr_max": lr_max, "warmup_end_step": int(np.argmax(lr)), "seeds": {}}
    rows = []                                                    # one row per (seed, transition s = 2..15)
    for seed in SEEDS:
        recs = {s: record(seed, s) for s in STAGES}
        L = {s: {c: np.mean(v["loss"]) for c, v in ev["per_image"][str(seed)][str(s)].items()} for s in STAGES}
        P = {s: {c: np.mean(v["p99"]) for c, v in ev["per_image"][str(seed)][str(s)].items()} for s in STAGES}
        val = max(abs(L[s][c] - recs[s]["heldout_normal"][c]["global_cosine"]) for s in STAGES for c in C.EVAL
                  if c in L[s])
        sk = {}
        for s in STAGES:
            r, step = shock_ratio(recs[s]["loss_curve_every50"])
            mid = min(step + 25, len(lr) - 1)
            sk[s] = {"ratio": r, "peak_step": step, "lr_at_peak": float(lr[mid]), "lr_frac": float(lr[mid] / lr_max),
                     "shock": r >= SHOCK_X, **{f"shock_{x}": r >= x for x in SENSITIVITY_X}}
        trans = {}
        for s in STAGES[1:]:
            old = [c for c in C.STREAM[:s - 1]]
            d = {c: float(L[s][c] - L[s - 1][c]) for c in old}
            dt = {c: float(P[s][c] - P[s - 1][c]) for c in old}
            trans[s] = {"old": old, "d": d, "d_tail": dt, "macro": float(np.mean(list(d.values()))),
                        "macro_tail": float(np.mean(list(dt.values()))), "worst": max(d.values()),
                        "worst_cat": max(d, key=d.get), "drift": ev["drift"][str(seed)][str(s)]}
            rows.append({"seed": seed, "s": s, "shock": sk[s]["shock"], "ratio": sk[s]["ratio"],
                         "macro": trans[s]["macro"], "tail": trans[s]["macro_tail"], "worst": trans[s]["worst"],
                         "drift_total": float(np.sqrt(sum(v ** 2 for k, v in trans[s]["drift"].items()
                                                          if not k.startswith("block.")))),
                         "drift_proj": trans[s]["drift"]["attn.proj.weight"]})
        cum = {c: float(L[15][c] - L[t[c]][c]) for c in C.STREAM[:14]}
        out["seeds"][seed] = {"validation_max_abs_diff_vs_R15_heldout": float(val), "shocks": sk, "transitions": trans,
                              "cumulative_damage": cum, "acq_level": {c: float(L[t[c]][c]) for c in C.STREAM}}
    # Q1-Q3: frequency, location, consistency
    sh = {seed: [s for s in STAGES if out["seeds"][seed]["shocks"][s]["shock"]] for seed in SEEDS}
    all_sh = [out["seeds"][seed]["shocks"][s] for seed in SEEDS for s in sh[seed]]
    per_stage = {s: sum(out["seeds"][seed]["shocks"][s]["shock"] for seed in SEEDS) for s in STAGES}
    rng = np.random.default_rng(PERM_SEED)
    k = [len(sh[seed]) for seed in SEEDS]
    obs3 = sum(v == 3 for v in per_stage.values())
    null3 = 0.0
    if all(k):
        draws = []
        for _ in range(PERM_B):
            sets = [set((rng.choice(15, kk, replace=False) + 1).tolist()) for kk in k]
            draws.append(len(set.intersection(*sets)))
        null3 = float(np.mean(draws))
    out["Q1_frequency"] = {"shock_stages": sh, "count": dict(zip(map(str, SEEDS), k)),
                           "sensitivity": {str(x): {seed: sum(out["seeds"][seed]["shocks"][s][f"shock_{x}"]
                                                             for s in STAGES) for seed in SEEDS} for x in SENSITIVITY_X}}
    out["Q2_location"] = {"peak_steps": [x["peak_step"] for x in all_sh],
                          "near_peak_lr_fraction": float(np.mean([x["lr_frac"] >= PEAK_LR_FRAC for x in all_sh]))
                          if all_sh else None}
    out["Q3_consistency"] = {"seeds_with_shock_per_stage": per_stage, "stages_with_3_of_3": obs3,
                             "expected_3_of_3_if_random": float(null3)}
    # Q4: association with damage (descriptive; within-seed permutation)
    def assoc(key):
        def diff(lbl):
            return np.mean([r[key] for r, l in zip(rows, lbl) if l]) - np.mean([r[key] for r, l in zip(rows, lbl)
                                                                                 if not l])
        lbl = [r["shock"] for r in rows]
        if all(lbl) or not any(lbl):
            return None
        obs = diff(lbl)
        null = []
        for _ in range(PERM_B):
            perm = []
            for seed in SEEDS:
                sl = [r["shock"] for r in rows if r["seed"] == seed]
                perm += list(rng.permutation(sl))
            null.append(diff(perm))
        return {"shock_mean": float(np.mean([r[key] for r in rows if r["shock"]])),
                "nonshock_mean": float(np.mean([r[key] for r in rows if not r["shock"]])), "diff": float(obs),
                "perm_p_two_sided": float(np.mean(np.abs(null) >= abs(obs))),
                "per_seed": {seed: [float(np.mean([r[key] for r in rows if r["seed"] == seed and r["shock"]] or [np.nan])),
                                    float(np.mean([r[key] for r in rows if r["seed"] == seed and not r["shock"]]))]
                             for seed in SEEDS}}
    out["Q4_association"] = {k_: assoc(k_) for k_ in ("macro", "tail", "worst")}
    # Q5: recovery
    nxt = []
    for seed in SEEDS:
        tr = out["seeds"][seed]["transitions"]
        ser = [tr[s]["macro"] for s in range(2, 16)]
        lag = float(np.corrcoef(ser[:-1], ser[1:])[0, 1])
        after = {"shock": [tr[s + 1]["macro"] for s in range(2, 15) if out["seeds"][seed]["shocks"][s]["shock"]],
                 "nonshock": [tr[s + 1]["macro"] for s in range(2, 15) if not out["seeds"][seed]["shocks"][s]["shock"]]}
        pos = [(tr[s]["d"][c], tr[s + 1]["d"][c]) for s in range(2, 15) for c in tr[s]["old"]]
        nxt.append({"seed": seed, "lag1": lag, "next_after_shock": after["shock"], "next_after_nonshock":
                    after["nonshock"], "frac_positive_then_negative": float(np.mean([b < 0 for a, b in pos if a > 0]))})
    out["Q5_recovery"] = nxt
    # Q6: drift vs damage
    out["Q6_drift"] = {
        "spearman_total_vs_macro": {seed: spearman([r["drift_total"] for r in rows if r["seed"] == seed],
                                                   [r["macro"] for r in rows if r["seed"] == seed]) for seed in SEEDS},
        "spearman_proj_vs_macro": {seed: spearman([r["drift_proj"] for r in rows if r["seed"] == seed],
                                                  [r["macro"] for r in rows if r["seed"] == seed]) for seed in SEEDS},
        "pooled_spearman_total_vs_macro": spearman([r["drift_total"] for r in rows], [r["macro"] for r in rows]),
        "drift_shock_vs_nonshock": [float(np.mean([r["drift_total"] for r in rows if r["shock"]])),
                                    float(np.mean([r["drift_total"] for r in rows if not r["shock"]]))]}
    # Q7: representativeness of stages 7-8
    rep = {}
    for seed in SEEDS:
        ratio = {s: out["seeds"][seed]["shocks"][s]["ratio"] for s in range(2, 16)}
        dmg = {s: out["seeds"][seed]["transitions"][s]["macro"] for s in range(2, 16)}
        rank = lambda d, s: 1 + sorted(d, key=d.get, reverse=True).index(s)
        rep[seed] = {s: {"ratio_rank": rank(ratio, s), "damage_rank": rank(dmg, s)} for s in (7, 8)}
    out["Q7_representativeness"] = rep
    # Q8: categories
    cats = list(C.STREAM[:14])
    cumv = {seed: [out["seeds"][seed]["cumulative_damage"][c] for c in cats] for seed in SEEDS}
    out["Q8_categories"] = {
        "cumulative": {seed: dict(zip(cats, cumv[seed])) for seed in SEEDS},
        "rank_spearman_between_seeds": {f"{a}~{b}": spearman(cumv[a], cumv[b]) for i, a in enumerate(SEEDS)
                                        for b in SEEDS[i + 1:]},
        "top_category": {seed: cats[int(np.argmax(cumv[seed]))] for seed in SEEDS},
        "top_share_of_positive_sum": {seed: float(max(cumv[seed]) / sum(v for v in cumv[seed] if v > 0))
                                      for seed in SEEDS},
        "worst_category_counts": {seed: {c: sum(out["seeds"][seed]["transitions"][s]["worst_cat"] == c
                                                for s in range(2, 16)) for c in cats} for seed in SEEDS}}
    out["rows"] = rows
    D.write_json(AUDIT, strkeys({"plan_sha256": sha(PLAN), "eval_sha256": sha(EVAL), "utc": DR.now(), **out}))
    print("audit", sha(AUDIT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "evaluate", "analyze"])
    a = ap.parse_args()
    {"plan": plan, "evaluate": evaluate, "analyze": analyze}[a.cmd]()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
