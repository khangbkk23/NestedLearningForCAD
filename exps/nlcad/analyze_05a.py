# exps/nlcad/analyze_05a.py
"""NLCAD-05A analysis (CPU only; no anomaly labels, no Stage D, no GPU).

  python -m exps.nlcad.analyze_05a eligibility   # per-slow-event legal replay hosts after inner-batch exclusion, 7-15
  python -m exps.nlcad.analyze_05a credit        # recorded one-step credits: phase/stage summaries, phi-space credit
  python -m exps.nlcad.analyze_05a openloop      # CPU open-loop controller replay of recorded credit, penalty grid
  python -m exps.nlcad.analyze_05a counterfactual

Eligibility needs no decoder: the fast loop's sampler consumes only its stage generator (Batches + one randperm per
iteration), independently of the controller, so the inner batch of every slow event can be reproduced exactly. The
reproduction is checked against the normal-query identities that the GPU recorder stored at the same events.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import run as DR
from exps.dinomaly.common import require
from exps.nlcad import bank as BK
from exps.nlcad.controller import BLOCKS, BlockController
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

OUT = D.ROOT / "results/hope_cad/nlcad"
TAG = "RETENTION_CREDIT_RH01b_R15_seed2711_stage7_9"
RUN = OUT / "runs" / TAG
SEED, START, K = 2711, 6, 32
PHASES = ((0, 99, "warmup"), (100, 299, "mid_high"), (300, 499, "mid_low"), (500, 666, "tail"))
LAMBDA_GRID = (0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
SAT = 0.95


def phase(i):
    return next(name for lo, hi, name in PHASES if lo <= i <= hi)


def inner_batches(stage, n):
    """Yield (iteration, ci, ri) for every slow event of `stage`, consuming the stage generator exactly like
    exps.nlcad.fast.nl_train_stage / exps.rh01.run.train_stage_guarded (no non-finite skip occurs in R15)."""
    c = C.STREAM[stage - 1]
    done = list(C.STREAM[:stage - 1])
    n_replay = sum(len(v) for v in C.buffer_index(SEED, n, done).values())
    _, gseed = RB.stage_seeds("R15", SEED, stage)
    gen = torch.Generator().manual_seed(gseed)
    batch = O.RECIPE["batch_size"]
    cur, rep = DR.Batches(n[c], batch // 2, gen), DR.Batches(n_replay, batch // 2, gen)
    for i in range(R1.ITERS):
        ci, ri = cur.next(), rep.next()
        torch.randperm(batch, generator=gen)
        if i % K == K - 1:
            yield i, ci, ri


def host_positive_counts(bank, gen_name):
    """{host: [(variant, n_positive_patches, n_background_patches)]} from synthetic coverage only."""
    out = {}
    for it in bank[gen_name]:
        cov = it["coverage"]
        out.setdefault(tuple(it["host"]), []).append((it["variant"], int((cov >= 0.5).sum()), int((cov == 0).sum())))
    return out


def eligibility(stages=range(START + 1, len(C.STREAM) + 1)):
    proto = RB.verify_lock()
    n = R1.n_train(proto)
    bank, _ = BK.load_bank(SEED)
    counts = {g: host_positive_counts(bank, g) for g in BK.GENERATORS}
    res = {"stages": {}, "rule": "legal replay hosts = stage Replay-24 members minus the inner-batch replay identities "
                                 "of the fast step that precedes the slow event; a host is one identity regardless of "
                                 "its synthetic variants; a host is usable for a generator if at least one variant "
                                 "has a positive patch (coverage >= 0.5)"}
    for s in stages:
        done = list(C.STREAM[:s - 1])
        idx = C.buffer_index(SEED, n, done)
        keys = [(d, i) for d in done for i in idx[d]]
        rows = []
        for it, ci, ri in inner_batches(s, n):
            inner = {keys[j] for j in ri}
            row = {"iter": it, "phase": phase(it), "inner_replay_hosts": sorted([list(k) for k in inner])}
            for g in BK.GENERATORS:
                per = {}
                for d in done:
                    avail = [k for k in keys if k[0] == d and k not in inner]
                    usable = [k for k in avail if any(p > 0 for _, p, _ in counts[g][k])]
                    pos = {k: sum(p for _, p, _ in counts[g][k]) for k in usable}
                    neg = {k: sum(b for _, p, b in counts[g][k] if p > 0) for k in usable}
                    cross = sum(pos[h] * neg[h2] for h, h2 in itertools.permutations(usable, 2))
                    per[d] = {"buffer_hosts": len(idx[d]), "available_hosts": len(avail), "usable_hosts": len(usable),
                              "cross_host_pairs": cross, "within_host_pairs": sum(pos[h] * neg[h] for h in usable)}
                row[g] = {"per_category": per,
                          "categories_ge2": sum(v["usable_hosts"] >= 2 for v in per.values()),
                          "categories_1": sum(v["usable_hosts"] == 1 for v in per.values()),
                          "categories_0": sum(v["usable_hosts"] == 0 for v in per.values()),
                          "usable_hosts": sum(v["usable_hosts"] for v in per.values()),
                          "cross_host_pairs": sum(v["cross_host_pairs"] for v in per.values())}
            rows.append(row)
        summ = {}
        for g in BK.GENERATORS:
            ge2 = np.array([r[g]["categories_ge2"] for r in rows])
            summ[g] = {"old_categories": len(done), "replay_members": len(keys),
                       "buffer_hosts_per_category": {d: len(idx[d]) for d in done},
                       "events": len(rows),
                       "categories_ge2_min_median_max": [int(ge2.min()), float(np.median(ge2)), int(ge2.max())],
                       "categories_lt2_mean": float(np.mean([r[g]["categories_1"] + r[g]["categories_0"] for r in rows])),
                       "categories_0_mean": float(np.mean([r[g]["categories_0"] for r in rows])),
                       "usable_hosts_mean": float(np.mean([r[g]["usable_hosts"] for r in rows])),
                       "frac_events_with_cross_host_pairs": float(np.mean([r[g]["cross_host_pairs"] > 0 for r in rows])),
                       "frac_events_all_categories_ge2": float(np.mean(ge2 == len(done))),
                       "frac_events_any_usable": float(np.mean([r[g]["usable_hosts"] > 0 for r in rows])),
                       "median_cross_host_pairs": float(np.median([r[g]["cross_host_pairs"] for r in rows]))}
        res["stages"][str(s)] = {"summary": summ, "events": rows}
    check = recorder_check(n) if (RUN / "STAGE7.json").exists() else None
    res["recorder_reproduction_check"] = check
    D.write_json(OUT / "ELIGIBILITY_05A.json", res)
    for s, v in res["stages"].items():
        q = v["summary"]["Q3"]
        print(s, C.STREAM[int(s) - 1], "old", q["old_categories"], "hosts/cat", sorted(set(q["buffer_hosts_per_category"].values())),
              "ge2 min/med/max", q["categories_ge2_min_median_max"], "lt2 mean", round(q["categories_lt2_mean"], 2),
              "zero mean", round(q["categories_0_mean"], 2), "cross-pair events", q["frac_events_with_cross_host_pairs"],
              flush=True)
    print("recorder check", check, flush=True)
    return res


def recorder_check(n):
    """The recorder stored its normal-query identities, drawn from the non-inner indices with a fixed generator;
    recompute them from the reproduced inner batches. Exact equality validates the reproduction."""
    from exps.nlcad.retention_credit import CreditRecorder
    out = {"events": 0, "mismatches": 0}
    for s in range(7, 10):
        p = RUN / f"STAGE{s}.json"
        if not p.exists():
            continue
        rec = D.read(p)
        done = list(C.STREAM[:s - 1]); idx = C.buffer_index(SEED, n, done)
        cur, rep = list(range(n[C.STREAM[s - 1]])), list(range(sum(len(v) for v in idx.values())))
        r = CreditRecorder.__new__(CreditRecorder)
        r.stage, r.seed = s, SEED * 1000 + 500 + s
        events = {e["iter"]: e for e in rec["events"]}
        for it, ci, ri in inner_batches(s, n):
            qc, qr = r.normal_query(cur, rep, {"iter": it, "ci": ci, "ri": ri}, "normal")
            stored = events[it]["queries"]["normal"]
            out["events"] += 1
            out["mismatches"] += int(stored["query_current"] != qc or stored["query_replay"] != qr)
    return out


def eligibility_ret():
    """Structural audit of the actual J_ret selection (exps.nlcad.objectives_ret) on the real bank for every slow event
    of stages 7-15, with the reproduced inner batches: every selected host is legal, outside the inner batch and in
    the category it is pooled under; pair counts come from category_loss itself (map values do not affect counts)."""
    from exps.nlcad.objectives_ret import RetentionRanking, category_loss
    proto = RB.verify_lock()
    n = R1.n_train(proto)
    bank, hosts = BK.load_bank(SEED)
    res = {}
    for s in range(START + 1, len(C.STREAM) + 1):
        c = C.STREAM[s - 1]
        done = list(C.STREAM[:s - 1])
        idx = C.buffer_index(SEED, n, done)
        keys = [(d, i) for d in done for i in idx[d]]
        allowed = BK.allowed(hosts, SEED, n, s)
        obj = RetentionRanking(bank, "Q3", "Q2", allowed, c, keys, seed=SEED * 1000 + 700 + s, device="cpu", autocast=False)
        rows, bad = [], []
        for it, ci, ri in inner_batches(s, n):
            inner = {keys[j] for j in ri}
            hs = obj.hosts({"iter": it, "ci": ci, "ri": ri})
            sel = obj.select(hs, "Q3")
            pairs = cross = 0
            single = multi = 0
            for cat, items in sel.items():
                for x in items:
                    h = tuple(x["host"])
                    if h[0] != cat or h not in allowed or h in inner or h[0] == c or h not in keys:
                        bad.append((it, cat, h))
                if len({tuple(x["host"]) for x in items}) != len(items):
                    bad.append((it, cat, "duplicate host"))
                _, p, x_ = category_loss([torch.zeros(28, 28)] * len(items), [x["coverage"] for x in items], 0.05, 64)
                pairs, cross = pairs + p, cross + x_
                single, multi = single + (len(items) == 1), multi + (len(items) >= 2)
            q2 = obj.select(hs, "Q2")
            rows.append({"iter": it, "categories": len(sel), "old_categories": len(done), "single_host_fallback": single,
                         "multi_host": multi, "hosts": sum(len(v) for v in sel.values()), "pairs": pairs,
                         "cross_host_pair_fraction": cross / pairs if pairs else 0.0,
                         "q2_same_hosts": {k: sorted(tuple(x["host"]) for x in v) for k, v in q2.items()} ==
                                          {k: sorted(tuple(x["host"]) for x in v) for k, v in sel.items()}})
        res[str(s)] = {"events": len(rows), "violations": bad[:10], "n_violations": len(bad),
                       "old_categories": len(done),
                       "categories_mean": float(np.mean([r["categories"] for r in rows])),
                       "multi_host_mean": float(np.mean([r["multi_host"] for r in rows])),
                       "single_host_fallback_mean": float(np.mean([r["single_host_fallback"] for r in rows])),
                       "hosts_mean": float(np.mean([r["hosts"] for r in rows])),
                       "cross_host_pair_fraction_mean": float(np.mean([r["cross_host_pair_fraction"] for r in rows])),
                       "frac_events_usable": float(np.mean([r["categories"] > 0 for r in rows])),
                       "q2_same_hosts_frac": float(np.mean([r["q2_same_hosts"] for r in rows]))}
        print(s, c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[str(s)].items() if k != "violations"}, flush=True)
    D.write_json(OUT / "ELIGIBILITY_RET_05B.json", res)
    return res


# --- recorded credit -----------------------------------------------------------------------------------------

def load_events():
    rows = [D.read(RUN / f"STAGE{s}.json") for s in range(7, 10)]
    return rows, [dict(e, stage=r["stage"]) for r in rows for e in r["events"]]


def vec(d):
    return np.array([d[b] for b in BLOCKS], dtype=float)


def ccos(u, v):
    u, v = u - u.mean(), v - v.mean()
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300))


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300))


def credit():
    rows, ev = load_events()
    keys = ("Q3_current", "Q3_replay", "Q2_replay", "normal")
    res = {"events": len(ev), "parity": {r["stage"]: r["identity_state_bitwise_parity"] for r in rows},
           "checkpoint_sha256": {r["stage"]: r["checkpoint_sha256"] for r in rows},
           "identity_checkpoint_sha256": {r["stage"]: r["identity_checkpoint_sha256"] for r in rows},
           "seconds": {r["stage"]: r["seconds"] for r in rows},
           "peak_allocated_mib": {r["stage"]: r["peak_allocated_mib"] for r in rows}, "groups": {}, "pairs": {}}
    phi = {k: np.array([vec(e["groups"][k]["phi_hypergrad"]) for e in ev if e["groups"].get(k)]) for k in keys}
    s = {k: np.array([vec(e["groups"][k]["dJ_da"]) for e in ev if e["groups"].get(k)]) for k in keys}
    for k in keys:
        g = phi[k]
        cons = np.sign(g) == np.sign(g.mean(0))
        res["groups"][k] = {"events": len(g), "median_abs_phi_credit": float(np.median(np.abs(g))),
                            "median_phi_norm": float(np.median(np.linalg.norm(g, axis=1))),
                            "median_abs_dJ_da": float(np.median(np.abs(s[k]))),
                            "sign_consistency_vs_pooled_mean": float(cons.mean()),
                            "pooled_mean_phi_credit": dict(zip(BLOCKS, g.mean(0).tolist())),
                            "mean_over_std_by_block": dict(zip(BLOCKS, (g.mean(0) / (g.std(0) + 1e-300)).tolist())),
                            "by_stage_phase": {}}
        sel = [e for e in ev if e["groups"].get(k)]
        for st in (7, 8, 9):
            for ph in [p[2] for p in PHASES]:
                m = np.array([vec(e["groups"][k]["phi_hypergrad"]) for e in sel if e["stage"] == st and e["phase"] == ph])
                if len(m):
                    res["groups"][k]["by_stage_phase"][f"{st}/{ph}"] = {
                        "n": len(m), "median_norm": float(np.median(np.linalg.norm(m, axis=1))),
                        "mean_direction": (m.mean(0) / (np.linalg.norm(m.mean(0)) + 1e-300)).round(3).tolist()}
    res["per_stage"] = {}
    for st in (7, 8, 9):
        sev = [e for e in ev if e["stage"] == st]
        row = {}
        for k in keys:
            g = np.array([vec(e["groups"][k]["phi_hypergrad"]) for e in sev if e["groups"].get(k)])
            sc = [np.mean(list(e["groups"][k]["gradient_scale_to_normal"].values())) for e in sev
                  if e["groups"].get(k) and e["groups"][k]["gradient_scale_to_normal"]]
            row[k] = {"n": len(g), "median_phi_norm": float(np.median(np.linalg.norm(g, axis=1))),
                      "sign_consistency_vs_stage_mean": float(np.mean(np.sign(g) == np.sign(g.mean(0)))),
                      "stage_mean_direction": (g.mean(0) / (np.linalg.norm(g.mean(0)) + 1e-300)).round(3).tolist(),
                      "mean_over_std_norm": float(np.linalg.norm(g.mean(0)) / (np.linalg.norm(g.std(0)) + 1e-300)),
                      "median_grad_scale_to_normal": float(np.median(sc)) if sc else None}
            if k != "normal":
                u = [(vec(e["groups"][k]["phi_hypergrad"]), vec(e["groups"]["normal"]["phi_hypergrad"])) for e in sev
                     if e["groups"].get(k)]
                c = [cos(x, y) for x, y in u]
                row[k] |= {"median_cos_to_normal": float(np.median(c)), "frac_pos_cos_to_normal": float(np.mean(np.array(c) > 0)),
                           "median_norm_ratio_to_normal": float(np.median([np.linalg.norm(x) / np.linalg.norm(y) for x, y in u]))}
        rr = [cos(vec(e["groups"]["Q3_replay"]["phi_hypergrad"]), vec(e["groups"]["Q3_current"]["phi_hypergrad"])) for e in sev]
        row["Q3_replay~Q3_current_median_cos"] = float(np.median(rr))
        means = {k: np.array([vec(e["groups"][k]["phi_hypergrad"]) for e in sev if e["groups"].get(k)]).mean(0) for k in keys}
        row["stage_mean_cos"] = {f"{a}~{b}": cos(means[a], means[b]) for a, b in
                                 (("Q3_replay", "normal"), ("Q3_replay", "Q2_replay"), ("Q3_replay", "Q3_current"))}
        res["per_stage"][st] = row
    st_dirs = [np.array(res["per_stage"][st]["Q3_replay"]["stage_mean_direction"]) for st in (7, 8, 9)]
    res["Q3_replay_stage_mean_direction_cosines"] = {f"{a}~{b}": cos(st_dirs[i], st_dirs[j])
                                                     for (i, a), (j, b) in itertools.combinations(enumerate((7, 8, 9)), 2)}
    for a, b in (("Q3_replay", "normal"), ("Q3_current", "normal"), ("Q3_replay", "Q3_current"),
                 ("Q2_replay", "Q3_replay"), ("Q2_replay", "normal")):
        both = [e for e in ev if e["groups"].get(a) and e["groups"].get(b)]
        u = [vec(e["groups"][a]["phi_hypergrad"]) for e in both]
        v = [vec(e["groups"][b]["phi_hypergrad"]) for e in both]
        c = np.array([cos(x, y) for x, y in zip(u, v)])
        sg = np.array([np.mean(np.sign(x) == np.sign(y)) for x, y in zip(u, v)])
        ratio = np.array([np.linalg.norm(x) / (np.linalg.norm(y) + 1e-300) for x, y in zip(u, v)])
        res["pairs"][f"{a}~{b}"] = {"n": len(both), "median_phi_cosine": float(np.median(c)),
                                    "frac_positive_cosine": float(np.mean(c > 0)),
                                    "mean_sign_agreement": float(sg.mean()), "median_norm_ratio": float(np.median(ratio)),
                                    "per_stage_median_cosine": {st: float(np.median([x for x, e in zip(c, both) if e["stage"] == st]))
                                                                for st in (7, 8, 9)},
                                    "cosine_of_pooled_means": cos(np.mean(u, 0), np.mean(v, 0))}
    D.write_json(OUT / "CREDIT_05A.json", res)
    print("parity", res["parity"], "seconds", res["seconds"], "peak MiB", res["peak_allocated_mib"])
    for k, v in res["groups"].items():
        print(k, {x: v[x] for x in ("events", "median_phi_norm", "median_abs_dJ_da", "sign_consistency_vs_pooled_mean")})
    for k, v in res["pairs"].items():
        print(k, {x: v[x] for x in ("n", "median_phi_cosine", "frac_positive_cosine", "mean_sign_agreement",
                                    "median_norm_ratio", "cosine_of_pooled_means")})
    return res


# --- open-loop controller replay --------------------------------------------------------------------------------

def replay(seq, lam, lr=0.05, phi0=None):
    """Feed a recorded dJ/da sequence to a fresh controller (locked controller.py: Adam, bounded mean-1 multipliers,
    full phi Jacobian evaluated at the current, generally non-identity phi, penalty lam*|phi|^2), optionally from a
    non-identity start phi0. Open loop: the multipliers do not feed back into the trajectory. To first order the
    recorded dJ/da_b = <g, Delta_b>/a_b does not depend on a_b (Delta_b scales with a_b), so feeding the identity-
    trajectory credit to a non-identity controller is the first-order closed-loop approximation."""
    ctrl = BlockController(lr=lr, weight_decay=lam)
    if phi0 is not None:
        with torch.no_grad():
            ctrl.phi.copy_(torch.as_tensor(phi0, dtype=torch.float64))
    phis, sig, pen = [], [], []
    for s in seq:
        before = ctrl.phi.detach().clone()
        sig.append(np.abs(ctrl.grad_phi(s, lam=0.0).numpy()))
        pen.append(np.abs(2 * lam * before.numpy()))
        ctrl.slow_step(s)
        phis.append(ctrl.phi.detach().numpy().copy())
    phis = np.array(phis)
    t = np.tanh(phis)
    a = np.exp(BlockController().c * t)
    a = a / a.mean(1, keepdims=True)
    sig, pen = np.array(sig), np.array(pen)
    half = len(seq) // 2
    return {"lam": lam, "max_abs_tanh_phi": float(np.abs(t).max()), "end_abs_tanh_phi_max": float(np.abs(t[-1]).max()),
            "saturated_block_events": int((np.abs(t) > SAT).sum()), "saturated_blocks_end": int((np.abs(t[-1]) > SAT).sum()),
            "max_abs_a_minus_1": float(np.abs(a - 1).max()), "end_a": dict(zip(BLOCKS, a[-1].round(4).tolist())),
            "mean_abs_log_a_second_half": float(np.mean(np.abs(np.log(a[half:])))),
            "median_abs_a_minus_1_second_half": float(np.median(np.abs(a[half:] - 1))),
            "frac_block_events_penalty_dominates_second_half": float(np.mean(pen[half:] > sig[half:])),
            "median_penalty_over_signal_second_half": float(np.median(pen[half:] / (sig[half:] + 1e-300))),
            "phi_end": phis[-1].tolist(), "_a": a}


def sign_null(seq, rng):
    """Same per-event, per-block magnitudes, independent random signs: no consistent direction."""
    return [{b: v * (1 if rng.random() < 0.5 else -1) for b, v in s.items()} for s in seq]


NULL_DRAWS, SEP_MIN = 200, 0.95


def openloop_rule(seq, grid=LAMBDA_GRID, draws=NULL_DRAWS, seed=0):
    """Penalty selection rule (fixed before the 60-event data were read; no label, no Q2 outcome).
    Movement statistic m = mean |log a| over the second half of the replay (the controller's realised deviation).
    For each lam: real m, and the distribution of m over `draws` sign-randomised null sequences.
    Admissible lam: no saturated block event in the real replay, AND the real m exceeds the SEP_MIN quantile of the
    null (the movement comes from a consistent direction, not from noise that Adam amplifies at small lam).
    Chosen lam: the admissible value with the largest separation (real m / null median m); ties -> the larger lam.
    If none is admissible, report SIGNAL_NOT_SEPARABLE (no lam is chosen; the learned arm is not justified as is)."""
    rng = np.random.default_rng(seed)
    nulls = [sign_null(seq, rng) for _ in range(draws)]
    rows = {}
    for lam in grid:
        r = replay(seq, lam)
        nm = np.array([replay(n, lam)["mean_abs_log_a_second_half"] for n in nulls])
        m = r["mean_abs_log_a_second_half"]
        r.pop("_a")
        r |= {"null_median_movement": float(np.median(nm)), "null_q95_movement": float(np.quantile(nm, SEP_MIN)),
              "real_movement_percentile_in_null": float(np.mean(nm < m)),
              "separation_ratio": float(m / (np.median(nm) + 1e-300))}
        r["admissible"] = bool(r["saturated_block_events"] == 0 and m > r["null_q95_movement"])
        rows[str(lam)] = r
    ok = [l for l in grid if rows[str(l)]["admissible"]]
    best = max(ok, key=lambda l: (round(rows[str(l)]["separation_ratio"], 6), l)) if ok else None
    return rows, best


STARTS = {"identity": None, "alternating_0.5": [0.5 * (-1) ** j for j in range(9)],
          "ramp_0.8": np.linspace(-0.8, 0.8, 9).tolist(), "deep_up_0.6": [-0.6] * 4 + [0.0] + [0.6] * 4}


def openloop(scale_ratio=1.0, out_name="OPENLOOP_05A.json"):
    """Source credit: the recorded one-step Q3-replay dJ/da along the identity trajectory, stages 7-9 (60 events),
    times scale_ratio (J_ret / recorder gradient scale; 1 for the recorder's own objective). The 05B arm spans 9 stages;
    7-9 are recorded, so the 60-event sequence is replayed three times as the horizon proxy (180 events)."""
    _, ev = load_events()
    seq = [{b: e["groups"]["Q3_replay"]["dJ_da"][b] * scale_ratio for b in BLOCKS} for e in ev]
    nseq = [e["groups"]["normal"]["dJ_da"] for e in ev]
    rows, chosen = openloop_rule(seq * 3)
    res = {"rule": openloop_rule.__doc__, "source": openloop.__doc__, "scale_ratio": scale_ratio,
           "grid": rows, "chosen_lambda": chosen,
           "status": "CHOSEN" if chosen is not None else "SIGNAL_NOT_SEPARABLE",
           "single_pass_60_events": {str(l): {k: v for k, v in replay(seq, l).items() if k != "_a"} for l in LAMBDA_GRID},
           "non_identity_starts": ({name: {k: v for k, v in replay(seq * 3, chosen, phi0=p).items() if k != "_a"}
                                    for name, p in STARTS.items()} if chosen is not None else None),
           "normal_credit_same_rule": {str(l): {k: v for k, v in r.items()} for l, r in openloop_rule(nseq * 3, draws=50)[0].items()},
           "caveat": "open loop: in a closed loop the multipliers change the trajectory and hence the credit; the "
                     "chosen value calibrates scale and separability, it does not predict the learned run"}
    D.write_json(OUT / out_name, res)
    for l, r in rows.items():
        print(l, {k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in
                  ("max_abs_tanh_phi", "saturated_block_events", "mean_abs_log_a_second_half", "null_median_movement",
                   "real_movement_percentile_in_null", "separation_ratio",
                   "frac_block_events_penalty_dominates_second_half", "admissible")})
    print("chosen", chosen)
    if chosen is not None:
        for name, r in res["non_identity_starts"].items():
            print("start", name, {k: round(r[k], 4) for k in ("max_abs_tanh_phi", "end_abs_tanh_phi_max",
                                                              "mean_abs_log_a_second_half")}, "sat", r["saturated_block_events"])
    return res


# --- counterfactuals --------------------------------------------------------------------------------------------

def counterfactual():
    rows, ev = load_events()
    out = {"events": [], "reading": {}}
    for e in ev:
        cf = e.get("counterfactual")
        if cf is None:
            continue
        v = cf["variants"]
        idv = v["identity"]
        row = {"stage": e["stage"], "iter": e["iter"], "restoration_exact": cf["restoration_exact"],
               "identity_parity_next_step": cf.get("identity_parity_next_step"), "variants": {}}
        for name, m in v.items():
            d = {}
            for q in ("q2_replay", "q2_current", "q3_replay"):
                for f in ("pooled_ap", "per_image_ap", "background_p99", "margin"):
                    d[f"{q}.{f}"] = m[q][f] - idv[q][f]
            d["normal_current"] = m["normal_current"] - idv["normal_current"]
            d["effective_update_norm"] = m["effective_update_norm"]
            d["update_norm_ratio_to_identity"] = m["effective_update_norm"] / (idv["effective_update_norm"] + 1e-300)
            d["a_range"] = [min(m["a"].values()), max(m["a"].values())]
            # first-order prediction of the change in the recorded outer objectives from the credit
            for k in ("Q3_replay", "Q3_current", "normal", "Q2_replay"):
                g = e["groups"].get(k)
                if g is not None:
                    d[f"pred_dJ.{k}"] = float(sum(g["dJ_da"][b] * (m["a"][b] - 1.0) for b in BLOCKS))
            row["variants"][name] = d
        rv, rnd = row["variants"]["q3_replay"], row["variants"]["matched_norm_random"]
        row["retention_minus_random"] = {k: rv[k] - rnd[k] for k in rv if isinstance(rv[k], float) and not k.startswith("update")}
        row["query_identity_shared"] = True     # one fixed query list per event, scored for every variant
        row["q2_replay_hosts"] = cf["queries"]["q2_replay"]
        out["events"].append(row)
    keys = ["q3_replay.pooled_ap", "q3_replay.margin", "q3_replay.background_p99", "q2_replay.pooled_ap",
            "q2_replay.margin", "q2_replay.background_p99", "q2_current.pooled_ap", "normal_current"]
    out["reading"] = {k: {"retention_minus_identity": [r["variants"]["q3_replay"][k] for r in out["events"]],
                          "random_minus_identity": [r["variants"]["matched_norm_random"][k] for r in out["events"]],
                          "retention_better_than_random": [
                              (r["variants"]["q3_replay"][k] < r["variants"]["matched_norm_random"][k]) if
                              ("p99" in k or k == "normal_current") else
                              (r["variants"]["q3_replay"][k] > r["variants"]["matched_norm_random"][k]) for r in out["events"]]}
                      for k in keys}
    D.write_json(OUT / "COUNTERFACTUAL_05A.json", out)
    for r in out["events"]:
        print(r["stage"], r["iter"], "restore", r["restoration_exact"], "parity", r["identity_parity_next_step"])
        for name, d in r["variants"].items():
            print("  ", name, {k: (f"{x:+.2e}" if isinstance(x, float) else x) for k, x in d.items()
                               if k.split(".")[-1] in ("pooled_ap", "margin", "background_p99") or k.startswith("pred")
                               or k in ("normal_current", "update_norm_ratio_to_identity")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("eligibility", "eligibility_ret", "credit", "openloop", "counterfactual"))
    a = ap.parse_args()
    {"eligibility": eligibility, "eligibility_ret": eligibility_ret, "credit": credit, "openloop": openloop,
     "counterfactual": counterfactual}[a.command]()


if __name__ == "__main__":
    main()
