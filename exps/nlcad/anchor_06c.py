"""NLCAD-06C-ANCHOR: does an equilibrium-shifting intervention persist where path-shaping (REP-05) reverts?

NLCAD-06B-WASHOUT §9: REP-05's stage-7 old-normal benefit came back within one stage (arm-level reversion slope ~ -1).
REP-05 changes the path a stage takes; it leaves the stage objective, hence what the stage converges to, unchanged.
This experiment adds one fixed intervention that changes the objective of every stage: a function-space anchor on the
lawful Replay-24 members. No learned controller; normal images only.

Per seed in SEEDS, from that seed's SHA-verified R15 theta_6, stages 7-9 under that seed's exact R15 protocol:
  ID      the R15 trajectory (theta_7..theta_9 evaluated with the 06B probes; seed 2711: the bitwise-verified CL06B_ID run)
  REP-05  the NLCAD-06B post-step projection (alpha .5, the arm's own Replay-24 basis); seed 2711: CL06B_REP-05
  ANCHOR  training loss + LAMBDA * mean over the batch's replay rows of mean_maps(1 - cos(de, de_start)); de_start =
          the stage-start decoder's eval-mode FP32 outputs on that stage's Replay-24 members (refreshed at every stage
          start). The training-side outputs are the noisy train-mode outputs, exactly as the reconstruction loss pairs
          noisy decoder outputs with clean encoder targets. No projection: the ID operator only logs update statistics.

Commands: prereg | preflight | gate | eval-id --seed S | train --arm A --seed S | analyze | amend-eng.
"""
from __future__ import annotations

import argparse
import gc
import math
import time

import numpy as np
import torch
import torch.nn.functional as Fn

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nlcad import closed_loop_06b as M
from exps.nlcad import forensic_06a as F
from exps.nlcad import run as R
from exps.nlcad.controller import BLOCKS, block_of
from exps.rh01 import numerics as NUM
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

SEEDS = (2711, 2712, 2713)
START, STAGES, ITERS, EVERY = 6, (7, 8, 9), 667, 128
LAMBDA = 1.0
ARMS = {"REP-05": ("rep", 0.5), "ANCHOR": ("anchor", LAMBDA)}
REUSE = {("ID", 2711): "CL06B_ID_seed2711", ("REP-05", 2711): "CL06B_REP-05_seed2711"}
NEW_RUNS = [("ANCHOR", 2711), ("REP-05", 2712), ("ANCHOR", 2712), ("REP-05", 2713), ("ANCHOR", 2713)]
EXCLUDE = {"cable": [68]}            # O6 image held in seed 2712's Replay buffer: dropped from O6 for every seed
PF_ITERS, BUDGET_H, STOP_H = 110, 3.0, 3.4
DELTA = 0.002                        # 06B equivalence margin; minimum stage-7 effect for a persistence reading
SHOCK_X = 1.5                        # shock stage: peak 50-step training loss >= 1.5 x the preceding minimum

OUT = M.OUT
TASK = "agents/tasks/task_nlcad_06c_equilibrium_anchor.md"
SRC = "exps/nlcad/anchor_06c.py"
SOURCES = [SRC, "exps/nlcad/test_anchor_06c.py", "exps/nlcad/chain_06c.sh", "exps/nlcad/closed_loop_06b.py",
           "exps/nlcad/forensic_06a.py", "exps/nlcad/run.py", "exps/nlcad/controller.py", "exps/dinomaly/official.py",
           "exps/dinomaly/readout.py", "exps/dinomaly/run.py", "exps/dinomaly/common.py", "exps/rh01/run.py",
           "exps/rh01/common.py", "exps/rh01/numerics.py", "exps/rh01b/common.py", "exps/rh01b/run.py", TASK]
PREREG, PREFLIGHT = OUT / "PREREG_06C.json", OUT / "PREFLIGHT_06C.json"
ANALYSIS, DECISION = OUT / "ANALYSIS_06C.json", OUT / "DECISION_06C.json"

RULES = {
    "endpoint": "O6' = the 06B O6 old normals of tasks 1-6 minus EXCLUDE (77 images; identical images and order for "
                "every seed). g_s(a,k) = M[L_s^{a,k} - L_s^{ID,k}] on O6' at the end of stage s (category macro); "
                "G_s(a) = mean over the three seeds; per-seed category-stratified image bootstrap for description",
    "stage7_effect": "S7(a) iff g_7(a,k) < 0 for every seed k and G_7(a) <= -DELTA (DELTA = 0.002)",
    "persistence": "P(a) = G_9(a) / G_7(a) (defined when S7(a)); P8(a) = G_8(a) / G_7(a). P = 1 + (arm-level "
                   "reversion slope): P 0 = full reversion (slope -1), P 0.5 = slope -0.5",
    "acquisition": "R_a(s,k) as NLCAD-06B: (mean L(theta_{s-1}^ID) - mean L(theta_s^a)) / A_ID(s) on the 16 current-"
                   "task normals of stage s; degenerate A_ID < 0.01: retained iff excess <= 0.001, prohibitive iff > "
                   "0.005. minR(a) = min over stages and seeds",
    "EQUILIBRIUM_SHIFT_SUPPORTED": "S7(ANCHOR) and P(ANCHOR) >= 0.5 and g_9(ANCHOR,k) < 0 for every seed and minR("
                                   "ANCHOR) >= 0.90 (retained) and, if S7(REP-05), P(ANCHOR) - P(REP-05) >= 0.3",
    "EQUILIBRIUM_SHIFT_REJECTED": "acquisition prohibitive for ANCHOR (any R < 0.75), or S7(ANCHOR) and P(ANCHOR) "
                                  "<= 0.2 (reverts like REP-05: slope <= -0.8)",
    "NO_STAGE7_EFFECT": "not S7(ANCHOR) and not rejected: persistence is untestable at this anchor weight",
    "INCONCLUSIVE": "otherwise",
    "secondary_REP": "REP-05 reversion replicated iff S7(REP-05) and P(REP-05) <= 0.2; 'no stage-7 effect' iff not "
                     "S7(REP-05); else 'partial'",
    "shock_dependence": "shock stage iff the 50-step training-loss peak ratio >= 1.5 (R15 loss_curve_every50 for ID; "
                        "reconstruction loss for ANCHOR). The washout falsifier 'benefit only in seeds where ID "
                        "shocks' is evaluable only if some seed's ID has no stage-7 shock; otherwise reported as not "
                        "evaluable, with g_7 against ID's stage-7 shock ratio (descriptive, n = 3)",
    "descriptive": "tail (map p99) analogues, per category, within-stage probes, per-step gradient norms and attn.proj "
                   "update energies, anchor values",
    "stops": "integrity, numerical, resource only (parity, finite parameters at every probe, skip guard, operator "
             "tolerances, hashes, budget); efficacy never stops a run",
    "not_claimed": "native Pixel AP, Image AUROC, anomaly sensitivity, a learned slow level, any Level-3 claim"}


# --- inputs -------------------------------------------------------------------------------------------------------

def r15_state(seed, s):
    rec = D.read(C.OUT / f"runs/R15_seed{seed}/STAGE{s}.json")
    require(sha(D.ROOT / rec["checkpoint"]) == rec["checkpoint_sha256"], f"R15 {seed} checkpoint {s} changed")
    return torch.load(D.ROOT / rec["checkpoint"], weights_only=True)["state"], rec["checkpoint_sha256"]


def r15_loss_curve(seed, s):
    return D.read(C.OUT / f"runs/R15_seed{seed}/STAGE{s}.json")["loss_curve_every50"]


def stage_sets(seed, s, n, sub, held):
    """closed_loop_06b.stage_sets with the seed as a parameter (the Replay-24 members depend on the seed)."""
    cur, old = C.STREAM[s - 1], list(C.STREAM[:s - 1])
    idx = C.buffer_index(seed, n, old)
    old_x = torch.cat([held[c] if c in C.EVAL else F.rows(c, sub[c]["eval"]) for c in old])
    groups = [c for c in old for _ in range(len(held[c]) if c in C.EVAL else len(sub[c]["eval"]))]
    replay_keys = [(d, i) for d in old for i in idx[d]]
    return {"current": cur, "old": old, "idx": idx, "replay_keys": replay_keys,
            "replay": torch.stack([C.load_features(d)[i] for d, i in replay_keys]),
            "old_x": old_x, "groups": groups, "cur_x": F.rows(cur, sub[cur]["cur"])}


def o6_keep(sub, held):
    """Boolean mask over the 06B O6 order (tasks 1-6) that drops EXCLUDE."""
    keep = []
    for c in C.STREAM[:START]:
        ids = list(range(len(held[c]))) if c in C.EVAL else list(sub[c]["eval"])
        keep += [not (c not in C.EVAL and i in EXCLUDE.get(c, [])) for i in ids]
    return keep


def run_dir(arm, seed):
    return OUT / "runs" / (REUSE[(arm, seed)] if (arm, seed) in REUSE else f"CL06C_{arm}_seed{seed}")


# --- anchor --------------------------------------------------------------------------------------------------------

@torch.no_grad()
def anchor_targets(dec, x, device, micro=8):
    """Stage-start decoder outputs on the Replay-24 members: eval mode, FP32, RNG forked, mode restored."""
    outs = None
    was = dec.training
    with M.rng_fork(device):
        dec.eval()
        try:
            for xb in x.split(micro):
                xb = xb.to(device).float()
                _, de = dec(xb[:, 0], xb[:, 1])
                de = [d.float() for d in de]
                outs = [[d] for d in de] if outs is None else [o + [d] for o, d in zip(outs, de)]
        finally:
            dec.train(was)
    return [torch.cat(o) for o in outs]


def anchor_distance(de, tgt):
    """Per image: mean over the fused output maps of the global cosine distance (the probes' loss form)."""
    return torch.stack([1 - Fn.cosine_similarity(d.float().flatten(1), t.flatten(1)) for d, t in zip(de, tgt)],
                       1).mean(1)


def next_batch(current, replay, cur, rep, gen):
    """The 06B batch (8 current + 8 replay rows, one randperm of the joint batch) and each row's replay index (-1 for
    current rows), permuted by the same randperm call: the sampler stream is the 06B stream."""
    ci, ri = cur.next(), rep.next()
    x = torch.cat([current[ci], replay[ri]])
    rid = torch.tensor([-1] * len(ci) + list(ri))
    perm = torch.randperm(len(x), generator=gen)
    return x[perm], rid[perm]


def micro_loss(dec, xb, rb, p, n_batch, n_rep, device, autocast, anchor=None):
    """One micro-batch's loss. anchor=None: exactly the 06B expression. Otherwise the reconstruction branch gets its own
    copy of the outputs (the hard-mining hook acts on that copy only) and lam * sum over the micro-batch's replay rows
    of the anchor distance / n_rep is added (not hard-mined). Returns (loss, anchor part as float)."""
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=autocast):
        en, de = dec(xb[:, 0], xb[:, 1])
    if anchor is None:
        return O.global_cosine_hm_percent([e.float() for e in en], [d.float() for d in de],
                                          p=p, factor=O.RECIPE["hm_factor"]) * len(xb) / n_batch, 0.0
    loss = O.global_cosine_hm_percent([e.float() for e in en], [d.float().clone() for d in de],
                                      p=p, factor=O.RECIPE["hm_factor"]) * len(xb) / n_batch
    m = rb >= 0
    if not bool(m.any()):
        return loss, 0.0
    md, idx = m.to(device), rb[m].to(device)
    a = anchor["lam"] * anchor_distance([d[md] for d in de], [t[idx] for t in anchor["targets"]]).sum() / n_rep
    return loss + a, float(a)


def train_stage_06c(dec, current, iters, gen, replay, op=None, probe=None, every=EVERY, device="cuda", autocast=True,
                    anchor=None):
    """closed_loop_06b.train_stage_protected op for op (with anchor=None it is bitwise the 06B loop: CPU test), plus the
    per-step pre-clip gradient norm and, with anchor = {"lam", "targets"}, the anchor term (micro_loss)."""
    opt = O.make_optimizer(dec)
    sch = O.make_scheduler(opt, iters)
    ramp = round(0.1 * iters)
    batch = O.RECIPE["batch_size"]
    cur = DR.Batches(len(current), batch // 2, gen)
    rep = DR.Batches(len(replay), batch // 2, gen)
    dec.train()
    losses, alosses, gnorms, skips, run_ = [], [], [], [], 0
    for i in range(iters):
        p = min(O.RECIPE["hm_p_final"] * i / ramp, O.RECIPE["hm_p_final"])
        x, rid = next_batch(current, replay, cur, rep, gen)
        x = x.to(device, non_blocking=True)
        n_rep = int((rid >= 0).sum())
        opt.zero_grad(set_to_none=True)
        total, atotal = 0., 0.
        for xb, rb in zip(x.split(DR.MICRO), rid.split(DR.MICRO)):
            loss, a = micro_loss(dec, xb, rb, p, len(x), n_rep, device, autocast, anchor)
            loss.backward()
            total += float(loss)
            atotal += a
        norm = torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=O.RECIPE["clip_grad_norm"])
        gnorms.append(float(norm))
        if not (np.isfinite(total) and bool(torch.isfinite(norm))):
            opt.zero_grad(set_to_none=True)
            skips.append({"iteration": i, "loss": total, "grad_norm": float(norm)})
            run_ += 1
            require(len(skips) <= R1.MAX_SKIPS and run_ < R1.MAX_CONSECUTIVE, f"non-finite steps exceed limits: {skips}")
            sch.step()
            losses.append(float("nan"))
            alosses.append(float("nan"))
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
            alosses.append(atotal)
        if probe is not None and ((i + 1) % every == 0 or i + 1 == iters):
            probe(i + 1)
    return losses, alosses, gnorms, skips


# --- prereg / gate ---------------------------------------------------------------------------------------------------

def source_hashes():
    return {s: sha(D.ROOT / s) for s in SOURCES}


def reuse_hashes():
    out = {}
    for (arm, seed), name in REUSE.items():
        d = OUT / "runs" / name
        out[f"{arm}_{seed}"] = {p.name: sha(p) for p in sorted(d.glob("*.json"))}
    return out


def prereg():
    require(not PREREG.exists(), f"already preregistered: {PREREG}")
    require(sha(OUT / "DECISION_06B.json").startswith("51989b0a"), "06B decision changed")
    proto = RB.verify_lock()
    n = R1.n_train(proto)
    sub = F.subsets(n)
    held = {c: DR.load_split("heldout", c) for c in C.EVAL}
    keep = o6_keep(sub, held)
    ck = {seed: {s: D.read(C.OUT / f"runs/R15_seed{seed}/STAGE{s}.json")["checkpoint_sha256"]
                 for s in range(START, 10)} for seed in SEEDS}
    for seed in SEEDS:
        for s in range(START, 10):
            require(r15_state(seed, s)[1] == ck[seed][s], f"R15 {seed} {s}")
    D.write_json(PREREG, {
        "task": TASK, "utc": DR.now(), "sources": source_hashes(), "reused_runs": reuse_hashes(),
        "design": {"seeds": list(SEEDS), "start_stage": START, "stages": list(STAGES), "iters_per_stage": ITERS,
                   "stage_seeds": {seed: {s: RB.stage_seeds("R15", seed, s) for s in STAGES} for seed in SEEDS},
                   "arms": {"ID": "R15 trajectory (evaluated; 2711 = CL06B_ID)", **{a: {"kind": k, "param": v}
                                                                                    for a, (k, v) in ARMS.items()}},
                   "lambda": LAMBDA,
                   "lambda_rationale": "fixed a priori, no tuning: unit weight in the training loss's own units (global "
                                       "cosine distance per image), so the anchor term carries the same total weight "
                                       "as the reconstruction objective",
                   "anchor_construction": "targets = stage-start decoder outputs (both fused maps), eval mode, FP32, RNG "
                                          "forked, on that stage's Replay-24 members of that seed; training side = the "
                                          "replay rows' train-mode outputs in the same forward; per image mean over "
                                          "maps of 1 - cos(flattened map); summed over the batch's replay rows / their "
                                          "count (8); gradient not hard-mined",
                   "reused_runs": {f"{a}_{s}": v for (a, s), v in REUSE.items()}, "new_runs": NEW_RUNS,
                   "o6_exclude": EXCLUDE, "o6_keep_mask": keep, "o6_n": int(sum(keep)),
                   "numerics": "D4prime_fp32core", "probe_every": EVERY,
                   "labels_used": "none: normal images only"},
        "rules": RULES, "delta": DELTA, "shock_x": SHOCK_X, "r15_checkpoints": ck,
        "budget": {"cap_gpu_h": BUDGET_H, "stop_h": STOP_H, "preflight_iters": PF_ITERS}})
    print("preregistered", sha(PREREG))


def amendments():
    return [D.read(p) for p in sorted(OUT.glob("PREREG_06C_AMENDMENT_*.json"))]


def expected_sources():
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
    require(reuse_hashes() == pr["reused_runs"], "reused 06B run records changed")
    return pr


def amend_engineering(reason, seen):
    path = OUT / f"PREREG_06C_AMENDMENT_{len(amendments()) + 1}.json"
    require(not path.exists(), f"amendment exists: {path}")
    changed = {s: {"locked_sha256": v, "amended_sha256": sha(D.ROOT / s)}
               for s, v in expected_sources().items() if sha(D.ROOT / s) != v}
    require(changed and set(changed) <= {SRC, "exps/nlcad/chain_06c.sh"}, f"unexpected source changes: {sorted(changed)}")
    D.write_json(path, {"prereg_sha256": sha(PREREG), "sources": changed, "reason": reason,
                        "unchanged": "arms, lambda, seeds, stages, steps, measurements, sets, rules, decision code, "
                                     "tolerances, budget", "results_seen_before_amendment": seen, "utc": DR.now()})
    print("amendment", path.name, sha(path))


def gate():
    verify_prereg()
    pf = D.read(PREFLIGHT)
    ok = pf["prereg_sha256"] == sha(PREREG) and pf["integrity_pass"] and pf["projected_gpu_h"] <= BUDGET_H
    require(ok, f"gate closed: integrity {pf['integrity_pass']}, projected {pf['projected_gpu_h']:.2f} h, cap {BUDGET_H}")
    print("gate open", pf["projected_gpu_h"])
    return pf


def setup():
    require(torch.cuda.is_available(), "CUDA required")
    pr = verify_prereg()
    DR.configure()
    R.cap()
    proto = RB.verify_lock()
    NUM.use(True)
    n = R1.n_train(proto)
    sub = F.subsets(n)
    held = {c: DR.load_split("heldout", c) for c in C.EVAL}
    require(o6_keep(sub, held) == pr["design"]["o6_keep_mask"], "O6 mask differs from the preregistration")
    return pr, n, sub, held


def spent_h():
    return sum(D.read(p)["gpu_seconds"] for p in OUT.glob("runs/CL06C_*/STAGE*.json")) / 3600


# --- preflight -----------------------------------------------------------------------------------------------------

def preflight_seed(seed, n, sub, held, dev, res):
    """One seed's preflight segments. Everything it allocates is freed on return (the reference state is kept on the
    CPU), so the next seed starts from an empty device."""
    sd6, _ = r15_state(seed, START)
    dec = DR.new_decoder(seed)
    st = stage_sets(seed, 7, n, sub, held)
    current = C.load_features(st["current"])
    dseed, gseed = RB.stage_seeds("R15", seed, 7)

    def fresh():
        dec.load_state_dict(sd6)
        torch.manual_seed(dseed)
        return torch.Generator().manual_seed(gseed)
    dec.load_state_dict(sd6)
    U = M.bases(M.rep_cov(dec, st, dev))
    gen = fresh()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    R1.train_stage_guarded(dec, current, PF_ITERS, gen, st["replay"])
    torch.cuda.synchronize()
    ref_s = (time.perf_counter() - t0) / PF_ITERS
    ref = {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()}
    ref_gen = gen.get_state()
    gen = fresh()
    op = M.Protector(dec, "id", 0.0, U, U, dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_stage_06c(dec, current, PF_ITERS, gen, st["replay"], op=op)
    torch.cuda.synchronize()
    t_loop = (time.perf_counter() - t0) / PF_ITERS
    state = {k: v.detach().cpu() for k, v in dec.state_dict().items()}
    res["parity"][seed] = {"loop_parity_bitwise": all(torch.equal(state[k], ref[k]) for k in ref),
                           "sampler_generator_parity": bool(torch.equal(gen.get_state(), ref_gen)),
                           "ref_s_per_step": ref_s, "s_per_step": t_loop,
                           "id_operator": M.operator_checks(op.table(), "id", 0.0)}
    if seed != 2712:
        return
    # REP-05 and ANCHOR on this loop: cost, operator spec, finiteness, anchor values
    gen = fresh()
    op = M.Protector(dec, "rep", 0.5, U, U, dev)
    t0 = time.perf_counter()
    train_stage_06c(dec, current, PF_ITERS, gen, st["replay"], op=op)
    torch.cuda.synchronize()
    chk = M.operator_checks(op.table(), "rep", 0.5)
    res["REP-05"] = {"s_per_step": (time.perf_counter() - t0) / PF_ITERS, "operator": chk,
                     "operator_ok": M.operator_ok(chk, "rep", 0.5), "finite": M.params_finite(dec)}
    gen = fresh()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tg = anchor_targets(dec, st["replay"], dev)
    torch.cuda.synchronize()
    t_tg = time.perf_counter() - t0
    op = M.Protector(dec, "id", 0.0, U, U, dev)
    t0 = time.perf_counter()
    losses, al, gn, sk = train_stage_06c(dec, current, PF_ITERS, gen, st["replay"], op=op,
                                         anchor={"lam": LAMBDA, "targets": tg})
    torch.cuda.synchronize()
    res["ANCHOR"] = {"s_per_step": (time.perf_counter() - t0) / PF_ITERS, "targets_s": t_tg,
                     "targets_bytes": int(sum(t.numel() * 4 for t in tg)), "finite": M.params_finite(dec),
                     "anchor_first_last": [al[0], al[-1]], "loss_first_last": [losses[0], losses[-1]],
                     "anchor_finite": bool(np.all(np.isfinite(al))), "skips": len(sk),
                     "peak_mib": torch.cuda.max_memory_allocated() / 2**20}
    # probe cost on this stage's sets
    t0 = time.perf_counter()
    for x in (st["old_x"], st["cur_x"], st["replay"]):
        M.measure(dec, x, dev)
    torch.cuda.synchronize()
    res["probe_s"] = time.perf_counter() - t0


def preflight():
    require(not PREFLIGHT.exists(), f"preflight already written: {PREFLIGHT}")
    pr, n, sub, held = setup()
    dev = "cuda"
    res = {"prereg_sha256": sha(PREREG), "iters": PF_ITERS, "utc": DR.now(), "parity": {}}
    for seed in (2712, 2713):
        preflight_seed(seed, n, sub, held, dev, res)
        gc.collect()
        torch.cuda.empty_cache()
    step = max(res["REP-05"]["s_per_step"], res["ANCHOR"]["s_per_step"])
    per_run = 60.0 + 3 * (ITERS * step + 7 * res["probe_s"] * 1.05 + 15.0)
    res["projection_s"] = {f"{a}_{s}": per_run for a, s in NEW_RUNS}
    res["projected_gpu_h"] = (len(NEW_RUNS) * per_run + 2 * 120.0) / 3600
    integrity = (all(v["loop_parity_bitwise"] and v["sampler_generator_parity"] for v in res["parity"].values())
                 and res["REP-05"]["operator_ok"] and res["REP-05"]["finite"] and res["ANCHOR"]["finite"]
                 and res["ANCHOR"]["anchor_finite"])
    res["integrity_pass"] = bool(integrity)
    res["budget_pass"] = bool(res["projected_gpu_h"] <= BUDGET_H)
    res["status"] = "PASS" if integrity and res["budget_pass"] else ("HOLD_BUDGET" if integrity else "FAIL_INTEGRITY")
    D.write_json(PREFLIGHT, res)
    print(res["status"], "projected GPU-h", round(res["projected_gpu_h"], 3),
          "parity", {s: v["loop_parity_bitwise"] for s, v in res["parity"].items()}, flush=True)


# --- ID evaluation (seeds without a 06B ID run) ------------------------------------------------------------------------

def eval_id(seed):
    require((("ID", seed) not in REUSE), "seed 2711 reuses CL06B_ID")
    gate()
    pr, n, sub, held = setup()
    dev = "cuda"
    run = run_dir("ID", seed)
    run.mkdir(parents=True, exist_ok=True)
    dec = O.Decoder(768, 12).to(dev)
    sd6, sha6 = r15_state(seed, START)
    dec.load_state_dict(sd6)
    st7 = stage_sets(seed, 7, n, sub, held)
    t0 = time.perf_counter()
    base_m, base_pool = M.measure(dec, st7["old_x"], dev)
    n6 = len(pr["design"]["o6_keep_mask"])
    if not (run / "BASE.json").exists():
        D.write_json(run / "BASE.json", {"prereg_sha256": sha(PREREG), "start_sha256": sha6, "groups": st7["groups"],
                                         "theta6_old": base_m, "seed": seed})
    for s in STAGES:
        js = run / f"STAGE{s}.json"
        if js.exists():
            continue
        st = st7 if s == 7 else stage_sets(seed, s, n, sub, held)
        probes, refs = {}, {}
        for i, sd_s in ((0, s - 1), (ITERS, s)):
            state_i, sha_i = r15_state(seed, sd_s)
            dec.load_state_dict(state_i)
            rec = {}
            for name, x in (("old", st["old_x"]), ("cur", st["cur_x"]), ("rep", st["replay"])):
                m, pooled = M.measure(dec, x, dev)
                if i == 0:
                    refs[name] = pooled
                m["drift_stage"] = M.drift(pooled, refs[name])
                if name == "old":
                    m["drift_theta6_O6"] = M.drift(pooled[:n6], base_pool[:n6])
                rec[name] = m
            probes[str(i)] = rec
        D.write_json(js, {"arm": "ID", "seed": seed, "stage": s, "current": st["current"], "prereg_sha256": sha(PREREG),
                          "groups_old": st["groups"], "replay_keys": st["replay_keys"], "probes": probes,
                          "losses_every50": r15_loss_curve(seed, s), "checkpoint_sha256": sha_i,
                          "note": "R15 checkpoints evaluated (no training)", "integrity_pass": True,
                          "gpu_seconds": 0.0, "eval_seconds": time.perf_counter() - t0, "utc": DR.now()})
    D.write_json(run / "TRAIN.json", {"arm": "ID", "seed": seed, "finished_utc": DR.now(), "prereg_sha256": sha(PREREG),
                                      "note": "evaluation only"})
    print("ID", seed, "evaluated", round(time.perf_counter() - t0, 1), "s", flush=True)


# --- train one new run ------------------------------------------------------------------------------------------------

def train(arm, seed):
    require((arm, seed) in NEW_RUNS, f"not a new run: {arm} {seed}")
    gate()
    pr, n, sub, held = setup()
    dev = "cuda"
    kind, param = ARMS[arm]
    run = run_dir(arm, seed)
    run.mkdir(parents=True, exist_ok=True)
    spent = spent_h()
    torch.cuda.reset_peak_memory_stats()
    sd6, sha6 = r15_state(seed, START)
    dec = DR.new_decoder(seed)
    dec.load_state_dict(sd6)
    st7 = stage_sets(seed, 7, n, sub, held)
    n6 = len(pr["design"]["o6_keep_mask"])
    base_m, base_pool = M.measure(dec, st7["old_x"], dev)
    if (run / "BASE.json").exists():
        require(D.read(run / "BASE.json")["theta6_old"] == base_m, "theta_6 baseline not reproducible")
    else:
        D.write_json(run / "BASE.json", {"prereg_sha256": sha(PREREG), "start_sha256": sha6, "groups": st7["groups"],
                                         "theta6_old": base_m, "seed": seed})
    for s in STAGES:
        js = run / f"STAGE{s}.json"
        c = C.STREAM[s - 1]
        if js.exists():
            rec = D.read(js)
            require(rec["prereg_sha256"] == sha(PREREG) and rec["integrity_pass"], f"stage {s} record invalid")
            path = D.ROOT / rec["checkpoint"]
            require(sha(path) == rec["checkpoint_sha256"], f"stage {s} checkpoint changed")
            dec.load_state_dict(torch.load(path, weights_only=True)["state"])
            print(arm, seed, s, "resumed", flush=True)
            continue
        require(spent <= STOP_H, f"resource stop: cumulative GPU time {spent:.2f} h > {STOP_H} h")
        st = st7 if s == 7 else stage_sets(seed, s, n, sub, held)
        torch.cuda.synchronize()
        t_stage = time.perf_counter()
        U = M.bases(M.rep_cov(dec, st, dev))
        anchor = {"lam": param, "targets": anchor_targets(dec, st["replay"], dev)} if kind == "anchor" else None
        op = M.Protector(dec, "rep" if kind == "rep" else "id", param if kind == "rep" else 0.0, U, U, dev)
        mem = {"rep_rank": {l: int(U[l].shape[1]) for l in M.LAYERS},
               "anchor_target_bytes": int(sum(t.numel() * 4 for t in anchor["targets"])) if anchor else 0}
        probes, refs, t_probe, blocks = {}, {}, [], []
        snap = {"state": {k: v.detach().cpu().clone() for k, v in dec.named_parameters()}}

        def probe(i):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            require(M.params_finite(dec), f"non-finite parameters at stage {s} step {i}")
            rec = {}
            for name, x in (("old", st["old_x"]), ("cur", st["cur_x"]), ("rep", st["replay"])):
                m, pooled = M.measure(dec, x, dev)
                if i == 0:
                    refs[name] = pooled
                m["drift_stage"] = M.drift(pooled, refs[name])
                if name == "old":
                    m["drift_theta6_O6"] = M.drift(pooled[:n6], base_pool[:n6])
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
        dseed, gseed = RB.stage_seeds("R15", seed, s)
        torch.manual_seed(dseed)
        gen = torch.Generator().manual_seed(gseed)
        current = C.load_features(c)
        losses, alosses, gnorms, skips = train_stage_06c(dec, current, ITERS, gen, st["replay"], op=op, probe=probe,
                                                        anchor=anchor)
        del anchor
        tab = op.table()
        chk = M.operator_checks(tab, op.kind, op.alpha)
        ok = M.operator_ok(chk, op.kind, op.alpha) and M.params_finite(dec)
        path = run / f"stage{s}_{c}.pt"
        ck_sha = DR.save_ckpt(dec, path, {"arm": arm, "seed": seed, "stage": s, "name": c})
        stats_path = run / f"STEPSTATS{s}.pt"
        require(not stats_path.exists(), f"exists: {stats_path}")
        torch.save({"cols": M.COLS, "layers": M.LAYERS, "table": tab, "loss_total": torch.tensor(losses),
                    "loss_anchor": torch.tensor(alosses), "grad_norm": torch.tensor(gnorms)}, stats_path)
        torch.cuda.synchronize()
        gpu_s = time.perf_counter() - t_stage
        main = np.array(losses) - np.array(alosses)
        every50 = lambda v: [float(np.nanmean(v[i:i + 50])) for i in range(0, len(v), 50)]
        D.write_json(js, {
            "arm": arm, "seed": seed, "kind": kind, "param": param, "stage": s, "current": c,
            "prereg_sha256": sha(PREREG), "seeds": [dseed, gseed], "groups_old": st["groups"],
            "replay_keys": st["replay_keys"], "losses_every50": every50(main), "total_every50": every50(losses),
            "anchor_every50": every50(alosses), "grad_norm_every50": every50(gnorms),
            "grad_norm_max": float(np.nanmax(gnorms)), "skips": skips, "probes": probes,
            "block_displacement": blocks, "memory": mem, "operator": chk, "integrity_pass": bool(ok),
            "stepstats": str(stats_path.relative_to(D.ROOT)), "stepstats_sha256": sha(stats_path),
            "checkpoint": str(path.relative_to(D.ROOT)), "checkpoint_sha256": ck_sha,
            "probe_seconds": sum(t_probe), "gpu_seconds": gpu_s,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "utc": DR.now()})
        spent += gpu_s / 3600
        print(arm, seed, s, c, f"{gpu_s:.0f}s", "integrity", ok, "old",
              round(float(np.mean(probes[str(ITERS)]["old"]["loss"])), 5), flush=True)
        require(ok, f"integrity failure at {arm} {seed} stage {s}: {chk}")
    D.write_json(run / "TRAIN.json", {"arm": arm, "seed": seed, "finished_utc": DR.now(), "prereg_sha256": sha(PREREG),
                                      "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})


# --- analysis ----------------------------------------------------------------------------------------------------------

def shock_ratio(c):
    pk = max(range(2, len(c)), key=lambda i: c[i])
    return float(c[pk] / min(c[1:pk])), 50 * pk


def acq(cur_start_id, cur_end_id, cur_end_a):
    a_id = float(np.mean(cur_start_id) - np.mean(cur_end_id))
    excess = float(np.mean(cur_end_a) - np.mean(cur_end_id))
    return {"A_ID": a_id, "R": float((np.mean(cur_start_id) - np.mean(cur_end_a)) / a_id) if a_id >= 0.01 else None,
            "excess": excess}


def acq_flags(r):
    if r["A_ID"] >= 0.01:
        return r["R"] >= 0.90, r["R"] < 0.75
    return r["excess"] <= 0.001, r["excess"] > 0.005


def decide(sm):
    """Predeclared readings from the per-seed gaps g[arm][seed][stage] and acquisition flags."""
    def s7(a):
        g7 = [sm["g"][a][k]["7"]["point"] for k in sm["seeds"]]
        return all(x < 0 for x in g7) and np.mean(g7) <= -DELTA
    G = {a: {s: float(np.mean([sm["g"][a][k][s]["point"] for k in sm["seeds"]])) for s in ("7", "8", "9")}
         for a in ARMS}
    P = {a: (G[a]["9"] / G[a]["7"] if s7(a) else None) for a in ARMS}
    out = {"G": G, "S7": {a: bool(s7(a)) for a in ARMS}, "P": P,
           "P8": {a: (G[a]["8"] / G[a]["7"] if s7(a) else None) for a in ARMS},
           "anchor_g9_all_negative": all(sm["g"]["ANCHOR"][k]["9"]["point"] < 0 for k in sm["seeds"]),
           "anchor_retained": sm["acq"]["ANCHOR"]["retained_all"], "anchor_prohibitive": sm["acq"]["ANCHOR"]["prohibitive_any"]}
    if out["anchor_prohibitive"] or (out["S7"]["ANCHOR"] and P["ANCHOR"] <= 0.2):
        label = "EQUILIBRIUM_SHIFT_REJECTED"
    elif (out["S7"]["ANCHOR"] and P["ANCHOR"] >= 0.5 and out["anchor_g9_all_negative"] and out["anchor_retained"]
          and (not out["S7"]["REP-05"] or P["ANCHOR"] - P["REP-05"] >= 0.3)):
        label = "EQUILIBRIUM_SHIFT_SUPPORTED"
    elif not out["S7"]["ANCHOR"]:
        label = "NO_STAGE7_EFFECT"
    else:
        label = "INCONCLUSIVE"
    rep = ("REP-05 reversion replicated" if out["S7"]["REP-05"] and P["REP-05"] <= 0.2 else
           "REP-05 no stage-7 effect" if not out["S7"]["REP-05"] else "REP-05 partial")
    return label, rep, out


def analyze():
    require(not DECISION.exists(), "decision already written")
    pr = verify_prereg()
    keep = np.array(pr["design"]["o6_keep_mask"])
    arms = ["ID"] + list(ARMS)
    recs = {a: {k: {s: D.read(run_dir(a, k) / f"STAGE{s}.json") for s in STAGES} for k in SEEDS} for a in arms}
    base = {a: {k: D.read(run_dir(a, k) / "BASE.json") for k in SEEDS} for a in arms}
    integrity = all(recs[a][k][s]["integrity_pass"] for a in arms for k in SEEDS for s in STAGES)
    integrity &= all(base[a][k]["theta6_old"] == base["ID"][k]["theta6_old"] for a in arms for k in SEEDS)
    groups = np.array(base["ID"][2711]["groups"])[keep]
    require(all(np.array(base["ID"][k]["groups"])[keep].tolist() == groups.tolist() for k in SEEDS), "O6 order")
    end = lambda a, k, s, key="loss", part="old": np.array(recs[a][k][s]["probes"][str(ITERS)][part][key])
    o6 = lambda v: v[:len(keep)][keep]
    sm = {"seeds": [str(k) for k in SEEDS], "integrity": bool(integrity), "g": {}, "t": {}, "g_cat": {}, "acq": {},
          "shock": {}, "D": {}}
    for a in arms:
        sm["D"][a] = {str(k): {str(s): M.strat_boot(o6(end(a, k, s)) - o6(np.array(base[a][k]["theta6_old"]["loss"])),
                                                     groups)["point"] for s in STAGES} for k in SEEDS}
        sm["shock"][a] = {str(k): {str(s): shock_ratio(recs[a][k][s]["losses_every50"]) for s in STAGES}
                          for k in SEEDS}
    for a in ARMS:
        sm["g"][a], sm["t"][a], sm["g_cat"][a] = {}, {}, {}
        for k in SEEDS:
            sm["g"][a][str(k)] = {str(s): M.strat_boot(o6(end(a, k, s)) - o6(end("ID", k, s)), groups)
                                  for s in STAGES}
            sm["t"][a][str(k)] = {str(s): M.strat_boot(o6(end(a, k, s, "p99")) - o6(end("ID", k, s, "p99")), groups)
                                  for s in STAGES}
            sm["g_cat"][a][str(k)] = {str(s): M.per_category(o6(end(a, k, s)) - o6(end("ID", k, s)), groups)
                                      for s in STAGES}
        rows = {str(k): {str(s): acq(recs["ID"][k][s]["probes"]["0"]["cur"]["loss"],
                                     recs["ID"][k][s]["probes"][str(ITERS)]["cur"]["loss"],
                                     recs[a][k][s]["probes"][str(ITERS)]["cur"]["loss"]) for s in STAGES}
                for k in SEEDS}
        flags = [acq_flags(r) for v in rows.values() for r in v.values()]
        sm["acq"][a] = {"rows": rows, "retained_all": all(f[0] for f in flags), "prohibitive_any": any(f[1] for f in flags),
                        "minR": min((r["R"] for v in rows.values() for r in v.values() if r["R"] is not None),
                                    default=None)}
    label, rep, detail = decide(sm)
    id7 = [sm["shock"]["ID"][str(k)]["7"][0] for k in SEEDS]
    sm["shock_dependence"] = {
        "id_stage7_shock_ratio": dict(zip(sm["seeds"], id7)),
        "evaluable": any(x < SHOCK_X for x in id7),
        "g7_vs_id_shock": {a: [[x, sm["g"][a][str(k)]["7"]["point"]] for x, k in zip(id7, SEEDS)] for a in ARMS}}
    t_mean = {a: {s: float(np.mean([sm["t"][a][str(k)][s]["point"] for k in SEEDS])) for s in ("7", "8", "9")}
              for a in ARMS}
    sm["tail_mean"] = t_mean
    sm["tail_persistence"] = {a: (t_mean[a]["9"] / t_mean[a]["7"] if t_mean[a]["7"] < 0 else None) for a in ARMS}
    D.write_json(ANALYSIS, {"prereg_sha256": sha(PREREG), "summary": sm, "decision_detail": detail, "utc": DR.now()})
    D.write_json(DECISION, {"label": label, "secondary_REP": rep, "detail": detail, "rules": RULES,
                            "prereg_sha256": sha(PREREG), "analysis_sha256": sha(ANALYSIS), "utc": DR.now()})
    print(label, "|", rep, "|", {a: (round(v, 3) if v is not None else None) for a, v in detail["P"].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prereg", "preflight", "gate", "eval-id", "train", "analyze", "amend-eng"])
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--reason")
    ap.add_argument("--seen")
    a = ap.parse_args()
    if a.cmd == "prereg":
        prereg()
    elif a.cmd == "preflight":
        preflight()
    elif a.cmd == "gate":
        gate()
    elif a.cmd == "eval-id":
        eval_id(a.seed)
    elif a.cmd == "train":
        train(a.arm, a.seed)
    elif a.cmd == "analyze":
        analyze()
    else:
        amend_engineering(a.reason, a.seen)


if __name__ == "__main__":
    main()
