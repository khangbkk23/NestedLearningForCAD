# exps/nlcad/run_05b.py
"""NLCAD-05B: one learned retention-credit arm (RET1) and its evaluation. Nothing here trains automatically: `train`
needs the user's explicit launch approval after `prereg` has locked the configuration.

RET1 = the NLCAD-02 N1 machinery (locked controller.py and fast.py: K = 32, one-step look-ahead, Adam lr_phi 0.05,
bounded mean-1 nine-block multipliers, per-task fast optimizer, Replay-24, D4' numerics, seed 2711, stages 7-15 from
the RH-01b R15 task-6 checkpoint) with two changes fixed by NLCAD-05A, label-free:
  1. slow objective = J_ret (exps/nlcad/objectives_ret.py): old-category, category-balanced, cross-image Q3 ranking on
     legal replay hosts outside the inner batch; Q2 on the same hosts is a no-grad diagnostic; no current-task term;
  2. controller penalty lambda chosen by the null-separated open-loop rule of exps.nlcad.analyze_05a on the recorded
     one-step J_ret credit along the exact identity trajectory (`record` + `calibrate`), with an N1-likeness gate.
The fast objective, decoder, scorer, stream, sampler and evaluation are unchanged.

  python -m exps.nlcad.run_05b scale       # GPU forward/gradient only on frozen identity checkpoints (no training)
  python -m exps.nlcad.run_05b record      # GPU: J_ret credit on the exact identity trajectory 7-9 (no learning)
  python -m exps.nlcad.run_05b calibrate   # CPU: lambda by the null-separated open-loop rule on RETREC credit
  python -m exps.nlcad.run_05b preflight   # CPU: compute and storage ledger
  python -m exps.nlcad.run_05b prereg      # CPU: lock config, sources, task, start checkpoint, bank, evaluation
  python -m exps.nlcad.run_05b train       # GPU (approval required)
  python -m exps.nlcad.run_05b lock        # before any score; then stage_d score/metrics/compare --run RET1_...
  python -m exps.nlcad.run_05b dynamics    # CPU
  python -m exps.nlcad.run_05b compare     # CPU: RET1 vs ID / N1 / F1 / RND (historical, descriptive)
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nlcad import analyze_02 as A2
from exps.nlcad import analyze_05a as A5
from exps.nlcad import analyze_c as AC
from exps.nlcad import bank as BK
from exps.nlcad import openloop as OL
from exps.nlcad import run as R
from exps.nlcad import stage_d as SD
from exps.nlcad.controller import BLOCKS, BlockController, block_of
from exps.nlcad.fast import nl_train_stage
from exps.nlcad.objectives import NormalLookahead, WithDiagnostics
from exps.nlcad.objectives_ret import RetentionRanking
from exps.nlcad.retention_credit import CreditRecorder, phi_hypergrad, ranking_audit
from exps.nlcad.run_04 import synthetic_preservation
from exps.rh01 import numerics as NUM
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

OUT = R.OUT
SEED, START = 2711, 6
TAG = f"RET1_per_task_seed{SEED}_from{START}"
TASK = "agents/tasks/task_nlcad_05b_retention_slow_objective.md"
SOURCES = SD.SOURCES + ["exps/nlcad/objectives_ret.py", "exps/nlcad/run_05b.py", "exps/nlcad/test_nlcad_05b.py",
                        "exps/nlcad/chain_05b.sh", "exps/nlcad/analyze_05a.py", "exps/nlcad/retention_credit.py",
                        "exps/nlcad/run_04.py", "exps/nlcad/objectives_s.py", "exps/nlcad/analyze_02.py",
                        "exps/nlcad/run_02.py", "exps/nlcad/openloop.py"]
PREREG, LOCK = OUT / "PREREG_05B.json", OUT / "LOCK_05B.json"
SCALE, CALIB, PREFLIGHT = OUT / "SCALE_05B.json", OUT / "CALIBRATION_05B.json", OUT / "PREFLIGHT_05B.json"
MAX_HOSTS, TAU, N_NEG = 8, 0.05, 64


def config():
    lam = D.read(CALIB)["lambda"]
    cfg = R.make_config("S", SEED, START, "per_task", wd_phi=lam)
    cfg |= {"arm": "RET1", "outer": "retention_ret05b", "objective_impl": "exps/nlcad/objectives_ret.py:RetentionRanking",
            "max_hosts_per_category": MAX_HOSTS, "tau": TAU, "n_neg": N_NEG, "current_task_term": False,
            "host_rule": "Replay-24 members outside the inner batch, one view per host, >= 1 host per category",
            "lambda_rule": "analyze_05a.openloop_rule on RETREC J_ret credit (stages 7-9 x3); N1-likeness gate 0.9"}
    return cfg


def objective(bank, allowed, c, replay_keys, s, device="cuda"):
    return RetentionRanking(bank, "Q3", "Q2", allowed, c, replay_keys, seed=SEED * 1000 + 700 + s, tau=TAU,
                            n_neg=N_NEG, max_hosts=MAX_HOSTS, device=device)


# --- label-free calibration -----------------------------------------------------------------------------------

def scale(contexts=5):
    """Per-block gradient norms of J_ret and of the 05A recorder's Q3-replay objective at the frozen identity stage-end
    checkpoints 7-9, on the first `contexts` slow-event inner batches of that stage (exact reproduction). Dropout off,
    FP32 TF32-off numerics as the 05A recorder. Forward/gradient only: nothing is optimised or written but the JSON."""
    require(torch.cuda.is_available(), "CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    NUM.use(True)
    DR.configure()
    proto = RB.verify_lock()
    n = R1.n_train(proto)
    bank, hosts = BK.load_bank(SEED)
    dec = DR.new_decoder(SEED).cuda()
    for m in dec.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    rows = []
    for s in (7, 8, 9):
        rec = D.read(C.OUT / f"runs/R15_seed{SEED}/STAGE{s}.json")
        require(sha(D.ROOT / rec["checkpoint"]) == rec["checkpoint_sha256"], "identity checkpoint changed")
        dec.load_state_dict(torch.load(D.ROOT / rec["checkpoint"], weights_only=True)["state"])
        dec.train()
        c = C.STREAM[s - 1]
        done = list(C.STREAM[:s - 1])
        idx = C.buffer_index(SEED, n, done)
        keys = [(d, i) for d in done for i in idx[d]]
        allowed = BK.allowed(hosts, SEED, n, s)
        ret = objective(bank, allowed, c, keys, s)
        recd = CreditRecorder(bank, hosts, allowed, c, keys, s, SEED * 1000 + 500 + s, "cuda")
        for k, (it, ci, ri) in enumerate(A5.inner_batches(s, n)):
            if k >= contexts:
                break
            ctx = {"iter": it, "ci": ci, "ri": ri}
            dec.zero_grad(set_to_none=True)
            out = ret(dec, ctx)
            g_ret = {b: [] for b in BLOCKS}
            for nme, p in dec.named_parameters():
                g_ret[block_of(nme)].append(p.grad.detach().double().cpu().flatten() if p.grad is not None
                                            else torch.zeros(p.numel(), dtype=torch.float64))
            dec.zero_grad(set_to_none=True)
            info, g_rec = recd._loss(dec, recd.sample("Q3", "replay", ctx, "Q3_replay"), "Q3_replay", grad=True)
            dec.zero_grad(set_to_none=True)
            row = {"stage": s, "iter": it, "J_ret": out["loss"], "J_rec": info["loss"], "train": {
                kk: v for kk, v in out["train"].items() if kk != "host_keys"}, "blocks": {}}
            for b in BLOCKS:
                u = torch.cat(g_ret[b])
                v = torch.cat([x.double().flatten() for x in g_rec[b]])
                row["blocks"][b] = {"norm_ret": float(u.norm()), "norm_rec": float(v.norm()),
                                    "ratio": float(u.norm() / (v.norm() + 1e-300)),
                                    "cosine": float(u @ v / (u.norm() * v.norm() + 1e-300))}
            rows.append(row)
            print(s, it, round(out["loss"], 4), round(info["loss"], 4),
                  "ratio", round(float(np.median([x["ratio"] for x in row["blocks"].values()])), 3),
                  "cos", round(float(np.median([x["cosine"] for x in row["blocks"].values()])), 3),
                  "map err", out["train"].get("map_reproduction_max_abs"), flush=True)
    ratios = [x["ratio"] for r in rows for x in r["blocks"].values()]
    D.write_json(SCALE, {"utc": DR.now(), "rows": rows, "median_ratio": float(np.median(ratios)),
                         "median_cosine": float(np.median([x["cosine"] for r in rows for x in r["blocks"].values()])),
                         "max_map_reproduction_abs": max(r["train"]["map_reproduction_max_abs"] for r in rows),
                         "note": "gradient-norm ratio is a scale proxy for the credit ratio <g, Delta>; directions "
                                 "differ (cosine), so it calibrates magnitude only"})
    print("median ratio", float(np.median(ratios)), flush=True)


RECTAG = f"RETREC_identity_seed{SEED}_stage7_9"
N1_LIKE_GATE = 0.9


def record():
    """J_ret one-step credit along the exact identity trajectory, stages 7-9: the locked fast loop with the identity
    controller and learn=False (bitwise identity, tests/test_nlcad_05b.py), outer = J_ret (+ Q2 and normal no-grad
    diagnostics). Each stage-end state is compared bitwise with the RH-01b R15 checkpoint. No learning, no label."""
    run = OUT / "runs" / RECTAG
    require(not (run / "STAGE9.json").exists(), "already recorded")
    DR.configure()
    R.cap()
    proto = RB.verify_lock()
    NUM.use(True)
    run.mkdir(parents=True, exist_ok=True)
    n = R1.n_train(proto)
    sd0, _ = R.start_state(SEED, START)
    dec = DR.new_decoder(SEED)
    dec.load_state_dict(sd0)
    bank, hosts = BK.load_bank(SEED)
    torch.cuda.reset_peak_memory_stats()
    for s in (7, 8, 9):
        c = C.STREAM[s - 1]
        if (run / f"STAGE{s}.json").exists():
            # Exact resume: a recorded stage passed bitwise parity, so its end state IS the R15 stage-s checkpoint.
            require(D.read(run / f"STAGE{s}.json")["identity_state_bitwise_parity"], f"stage {s} parity")
            ref = D.read(C.OUT / f"runs/R15_seed{SEED}/STAGE{s}.json")
            require(sha(D.ROOT / ref["checkpoint"]) == ref["checkpoint_sha256"], "identity checkpoint changed")
            dec.load_state_dict(torch.load(D.ROOT / ref["checkpoint"], weights_only=True)["state"])
            print("RETREC", s, c, "resumed from the parity-verified identity checkpoint", flush=True)
            continue
        require(not (run / f"EVENTS{s}.json").exists(), f"orphan events file for stage {s}")
        done = list(C.STREAM[:s - 1])
        idx = C.buffer_index(SEED, n, done)
        replay_keys = [(d, i) for d in done for i in idx[d]]
        replay = torch.stack([C.load_features(d)[i] for d, i in replay_keys])
        current = C.load_features(c)
        dseed, gseed = RB.stage_seeds("R15", SEED, s)
        torch.manual_seed(dseed)
        gen = torch.Generator().manual_seed(gseed)
        ret = objective(bank, BK.allowed(hosts, SEED, n, s), c, replay_keys, s)
        normal = NormalLookahead(current, replay, seed=SEED * 1000 + s)
        t0 = time.perf_counter()
        _, skips, events = nl_train_stage(dec, current, R1.ITERS, gen, replay, ctrl=BlockController(),
                                          outer=WithDiagnostics(ret, {"normal_lookahead": normal.evaluate}),
                                          K=32, learn=False)
        torch.cuda.synchronize()
        state = {k: v.detach().cpu() for k, v in dec.state_dict().items()}
        ref = D.read(C.OUT / f"runs/R15_seed{SEED}/STAGE{s}.json")
        ref_state = torch.load(D.ROOT / ref["checkpoint"], weights_only=True)["state"]
        parity = all(torch.equal(state[k], ref_state[k]) for k in state)
        D.write_json(run / f"EVENTS{s}.json", events)
        D.write_json(run / f"STAGE{s}.json", {"stage": s, "seconds": time.perf_counter() - t0, "skips": skips,
                                               "identity_state_bitwise_parity": parity,
                                               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})
        print("RETREC", s, c, "parity", parity, "events", len(events), flush=True)
        require(parity, f"identity parity lost at stage {s}")
        del replay, current


def retrec_credit(evs):
    """J_ret one-step credit in phi space at identity (descriptive; not a gate): per stage, its norm, sign consistency,
    mean/std, and event-by-event comparison with the 05A recorder's normal and Q3-replay credit at the same iterations
    (same trajectory and the same fast step Delta, both bitwise identity)."""
    rec = {s: {e["iter"]: e for e in D.read(A5.RUN / f"STAGE{s}.json")["events"]} for s in (7, 8, 9)}
    phi = lambda d: np.array(phi_hypergrad(torch.zeros(len(BLOCKS), dtype=torch.float64), d))
    out = {}
    for s, es in evs.items():
        g = np.array([phi(e["dJ_da"]) for e in es])
        nrm = [phi(rec[s][e["iter"]]["groups"]["normal"]["dJ_da"]) for e in es]
        q3r = [phi(rec[s][e["iter"]]["groups"]["Q3_replay"]["dJ_da"]) for e in es]
        cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300))
        out[s] = {"median_phi_norm": float(np.median(np.linalg.norm(g, axis=1))),
                  "sign_consistency_vs_stage_mean": float(np.mean(np.sign(g) == np.sign(g.mean(0)))),
                  "mean_over_std_norm": float(np.linalg.norm(g.mean(0)) / (np.linalg.norm(g.std(0)) + 1e-300)),
                  "median_cos_to_normal": float(np.median([cos(x, y) for x, y in zip(g, nrm)])),
                  "median_norm_ratio_to_normal": float(np.median([np.linalg.norm(x) / np.linalg.norm(y) for x, y in zip(g, nrm)])),
                  "stage_mean_cos_to_normal": cos(g.mean(0), np.mean(nrm, 0)),
                  "median_cos_to_recorder_Q3_replay": float(np.median([cos(x, y) for x, y in zip(g, q3r)])),
                  "stage_mean_cos_to_recorder_Q3_replay": cos(g.mean(0), np.mean(q3r, 0)),
                  "stage_mean_direction": (g.mean(0) / (np.linalg.norm(g.mean(0)) + 1e-300)).round(3).tolist(),
                  "J_ret_mean": float(np.mean([e["outer"] for e in evs[s]])),
                  "J_ret_Q2_mean": float(np.nanmean([e["diag_loss"] for e in evs[s]])),
                  "categories_mean": float(np.mean([e["train"]["categories"] for e in evs[s]])),
                  "cross_host_pair_fraction_mean": float(np.mean([e["train"]["cross_host_pair_fraction"] for e in evs[s]]))}
    dirs = [np.array(out[s]["stage_mean_direction"]) for s in (7, 8, 9)]
    out["stage_mean_direction_cosines"] = {"7~8": float(dirs[0] @ dirs[1]), "7~9": float(dirs[0] @ dirs[2]),
                                           "8~9": float(dirs[1] @ dirs[2])}
    return out


def calibrate():
    """lambda = the analyze_05a null-separated open-loop rule (grid, null, starts unchanged) applied to the RECORDED
    J_ret one-step dJ/da (RETREC, stages 7-9, x3 horizon proxy). Gate (fixed before reading RETREC): if the open-loop
    J_ret profile at the chosen lambda has cos(log a) > N1_LIKE_GATE with N1's learned stage-15 profile, RET1 is
    N1-like and is not preregistered. No anomaly label, no Q2 outcome is used."""
    require(not CALIB.exists(), "already calibrated")
    run = OUT / "runs" / RECTAG
    evs = {s: D.read(run / f"EVENTS{s}.json") for s in (7, 8, 9)}
    ev = [e for s in (7, 8, 9) for e in evs[s]]
    # integrity gate (declared before RETREC is read): parity, finiteness, exact two-pass gradient, 20 events/stage
    require(all(D.read(run / f"STAGE{s}.json")["identity_state_bitwise_parity"] for s in (7, 8, 9)), "parity")
    require(all(len(evs[s]) == 20 for s in evs) and all(e["finite"] for e in ev), "non-finite or missing events")
    require(max(e["train"]["map_reproduction_max_abs"] for e in ev) == 0.0, "two-pass gradient not exact")
    seq = [e["dJ_da"] for e in ev]
    rows, lam = A5.openloop_rule(seq * 3)
    res = {"utc": DR.now(), "rule": calibrate.__doc__ + "\n" + A5.openloop_rule.__doc__, "grid": rows,
           "chosen_lambda": lam, "status": "CHOSEN" if lam is not None else "SIGNAL_NOT_SEPARABLE",
           "credit": retrec_credit(evs)}
    if lam is not None:
        end = A5.replay(seq * 3, lam)
        la = np.log(end.pop("_a")[-1])
        n1 = D.read(OUT / f"runs/N1_per_task_seed{SEED}_from{START}/STAGE15.json")["controller_a"]
        ln1 = np.log(np.array([n1[b] for b in BLOCKS]))
        cosn1 = float(la @ ln1 / (np.linalg.norm(la) * np.linalg.norm(ln1) + 1e-300))
        res |= {"open_loop_end": end, "cos_log_a_with_N1_stage15": cosn1, "n1_like": cosn1 > N1_LIKE_GATE,
                "non_identity_starts": {k: {x: v for x, v in A5.replay(seq * 3, lam, phi0=p).items() if x != "_a"}
                                        for k, p in A5.STARTS.items()}}
    D.write_json(OUT / "OPENLOOP_RET_05B.json", res)
    require(lam is not None, "SIGNAL_NOT_SEPARABLE: no admissible lambda; do not preregister RET1")
    require(not res["n1_like"], f"J_ret open-loop profile is N1-like (cos {res['cos_log_a_with_N1_stage15']:.3f})")
    D.write_json(CALIB, {"utc": DR.now(), "lambda": lam, "rule": res["rule"],
                         "sources": {"OPENLOOP_RET_05B": sha(OUT / "OPENLOOP_RET_05B.json"),
                                     "RETREC_events": {s: sha(run / f"EVENTS{s}.json") for s in (7, 8, 9)}},
                         "inputs": "recorded one-step J_ret credit only; no anomaly label, no Q2 outcome"})
    print("lambda", lam, "cos with N1", res["cos_log_a_with_N1_stage15"], flush=True)


def preflight():
    """Compute and storage ledger from measured timings (R15 identity stages, NLCAD-02 N1, NLCAD-05A recorder)."""
    n1 = [D.read(OUT / f"runs/N1_per_task_seed{SEED}_from{START}/STAGE{s}.json") for s in range(START + 1, 16)]
    idn = [D.read(C.OUT / f"runs/R15_seed{SEED}/STAGE{s}.json") for s in range(START + 1, 16)]
    rec = A5.load_events()[0]
    n1_s, id_s = sum(r["seconds"] for r in n1), sum(r["seconds"] for r in idn)
    rec_s = float(np.mean([r["seconds"] for r in rec]))
    id79 = float(np.mean([r["seconds"] for r in idn[:3]]))
    # The R15 identity stages were timed under different machine load (slower than N1 itself), so they are not a
    # baseline. Per stage, RET1 is bounded by the 05A recorder measured on this GPU (three outer gradients + Q2 +
    # a counterfactual vs RET1's two-pass J_ret + Q2/normal no-grad diagnostics), and is at least N1 (one outer).
    outer_ratio = 1.5
    train_s = 9 * max(rec_s, n1_s / 9 * 1.1)
    bank_bytes = sum((BK.BANK / f"seed{SEED}" / f"{g}_{c}.pt").stat().st_size for g in BK.GENERATORS for c in C.STREAM)
    res = {"utc": DR.now(),
           "measured": {"identity_stage7_15_seconds": id_s, "N1_stage7_15_seconds": n1_s,
                        "N1_peak_mib": max(r["peak_allocated_mib"] for r in n1),
                        "recorder_05A_mean_stage_seconds": rec_s, "identity_mean_stage7_9_seconds": id79,
                        "recorder_05A_peak_mib": max(r["peak_allocated_mib"] for r in rec)},
           "projection": {"train_gpu_hours": train_s / 3600, "stage_end_audits_gpu_hours": 9 * 120 / 3600,
                          "stage_d_gpu_hours": 10 / 60,
                          "total_gpu_hours": (train_s + 9 * 120) / 3600 + 10 / 60,
                          "hard_cap_gpu_hours": 2.5, "outer_cost_ratio_vs_N1": outer_ratio},
           "storage": {"slow_controller_persistent_bytes": BlockController().persistent_bytes(),
                       "decoder_copies": 0, "replay24_images": 24,
                       "synthetic_bank_bytes_all_categories_seed2711": bank_bytes,
                       "synthetic_views_used": "derived from Replay-24 members only (2 Q2 + 2 Q3 per host); the bank "
                                               "file also holds current-task hosts, unused by J_ret",
                       "transient_cpu_bytes_per_slow_event": "theta_old copy for the look-ahead (61.4 M FP32 params, "
                                                             "about 246 MB, freed after the event; not persistent)",
                       "checkpoints_written": "9 stage-end decoder checkpoints (as every arm)"}}
    require(res["projection"]["total_gpu_hours"] <= res["projection"]["hard_cap_gpu_hours"], "projection above cap")
    D.write_json(PREFLIGHT, res)
    print(res, flush=True)


# --- preregistration, training, lock, analysis ----------------------------------------------------------------

def prereg():
    require(not PREREG.exists(), "already preregistered")
    start = D.read(C.OUT / f"runs/R15_seed{SEED}/STAGE{START}.json")
    bank_man = D.ROOT / f"results/hope_cad/nlcad/BANK_seed{SEED}.json"
    pf = D.read(PREFLIGHT)
    D.write_json(PREREG, {
        "utc": DR.now(), "task": {TASK: sha(D.ROOT / TASK)}, "sources": {s: sha(D.ROOT / s) for s in SOURCES},
        "config": config(), "tag": TAG,
        "calibration": {"path": str(CALIB.relative_to(D.ROOT)), "sha256": sha(CALIB)},
        "eligibility": {"path": "results/hope_cad/nlcad/ELIGIBILITY_05A.json", "sha256": sha(OUT / "ELIGIBILITY_05A.json")},
        "start_checkpoint": {"path": start["checkpoint"], "sha256": start["checkpoint_sha256"]},
        "bank_manifest": {"path": str(bank_man.relative_to(D.ROOT)), "sha256": sha(bank_man)},
        "generator_selection": "Q3 trains, Q2 diagnoses: fixed by NLCAD-04/05 before any learned run; not from real AP",
        "evaluation": {"cohort": "consumed P1 development cohort (bottle/carpet/hazelnut), via exps.nlcad.stage_d",
                       "rh01b_prereg_lock_sha256": sha(C.OUT / "PREREG_LOCK.json"),
                       "nlcad02_lock_sha256": sha(OUT / "LOCK_02.json")},
        "primary": "final (stage 15) macro native Pixel AP (R_D), RET1 - ID",
        "secondary": ["pooled AP", "per-category AP", "acquisition-to-final forgetting (shared acquisition checkpoints)",
                      "stage 7-15 macro trajectory, its mean and minimum", "Image AUROC", "R_P1",
                      "stage-end Q2/Q3 category-level ranking audit (within-image A, cross-image B, background p99)",
                      "held-out normal statistics (acquisition)", "controller dynamics, penalty vs signal",
                      "effective decoder update norm per stage vs identity", "GPU time and persistent storage",
                      "RET1 - N1 / F1 / RND (historical; N1 is compute-matched up to the outer's extra forward)"],
        "reading": "one seed, descriptive. 'Positive exploratory' = RET1-ID final macro > 0 and pooled > 0 and trajectory "
                   "mean > 0 and RET1-RND final > 0 and RET1-N1 final > 0 and every stage 7-15 RET1-ID macro >= -0.025 "
                   "(the NLCAD-01 single-seed noise floor). It justifies a matched confirmation (seeds, fixed RET1 "
                   "profile, compute-matched identity, direct synthetic-regularisation control, matched-compute "
                   "clock ablation), not a claim. Mixed or negative results are reported as such",
        "budget": {"gpu_hours_projected": pf["projection"]["total_gpu_hours"], "gpu_hours_hard_cap": 2.5,
                   "vram_cap_gib": float(__import__("os").environ.get("NLCAD_CAP_GIB", "2.6")), "arms": 1},
        "preflight_sha256": sha(PREFLIGHT),
        "bounds": "exactly one training run (RET1); no automatic follow-up arm; no tuning on development labels"})
    print("preregistered", sha(PREREG), flush=True)


def verify_prereg():
    pr = D.read(PREREG)
    for s, h in pr["sources"].items():
        require(sha(D.ROOT / s) == h, f"source changed after preregistration: {s}")
    require(sha(D.ROOT / TASK) == pr["task"][TASK], "task file changed after preregistration")
    require(sha(CALIB) == pr["calibration"]["sha256"], "calibration changed")
    require(pr["config"] == config(), "config differs from the preregistered one")
    return pr


def update_norms(dec, ref_state):
    out = {b: 0.0 for b in BLOCKS}
    for n, p in dec.named_parameters():
        out[block_of(n)] += float((p.detach().cpu().double() - ref_state[n].double()).pow(2).sum())
    return {b: math.sqrt(v) for b, v in out.items()}


def train():
    verify_prereg()
    cfg = config()
    DR.configure()
    R.cap()
    proto = RB.verify_lock()
    NUM.use(True)
    run = OUT / f"runs/{TAG}"
    run.mkdir(parents=True, exist_ok=True)
    cfg_path = run / "CONFIG.json"
    if cfg_path.exists():
        require(D.read(cfg_path) == cfg, "config differs from the recorded one")
    else:
        D.write_json(cfg_path, cfg)
    n = R1.n_train(proto)
    held = {c: DR.load_split("heldout", c) for c in C.EVAL}
    sd0, _ = R.start_state(SEED, START)
    dec = DR.new_decoder(SEED)
    dec.load_state_dict(sd0)
    ctrl = BlockController(lr=cfg["lr_phi"], weight_decay=cfg["wd_phi"])
    bank, hosts = BK.load_bank(SEED)
    torch.cuda.reset_peak_memory_stats()
    t_all = time.perf_counter()
    for s in range(START + 1, len(C.STREAM) + 1):
        c = C.STREAM[s - 1]
        ck, js = run / f"stage{s}_{c}.pt", run / f"STAGE{s}.json"
        if js.exists():                                                # resume at a stage boundary (exact)
            rec = D.read(js)
            require(sha(ck) == rec["checkpoint_sha256"], f"stage {s} checkpoint changed")
            snap = torch.load(ck, weights_only=False)
            dec.load_state_dict(snap["state"])
            ctrl.restore(snap["controller"])
            continue
        require(not ck.exists(), f"orphan checkpoint: {ck}")
        done = list(C.STREAM[:s - 1])
        idx = C.buffer_index(SEED, n, done)
        replay_keys = [(d, i) for d in done for i in idx[d]]
        replay = torch.stack([C.load_features(d)[i] for d, i in replay_keys])
        current = C.load_features(c)
        dseed, gseed = RB.stage_seeds("R15", SEED, s)
        torch.manual_seed(dseed)
        gen = torch.Generator().manual_seed(gseed)
        allowed = BK.allowed(hosts, SEED, n, s)
        ret = objective(bank, allowed, c, replay_keys, s)
        normal = NormalLookahead(current, replay, seed=SEED * 1000 + s)
        outer = WithDiagnostics(ret, {"normal_lookahead": normal.evaluate})
        start_state = {k: v.detach().cpu().clone() for k, v in dec.named_parameters()}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        losses, skips, events = nl_train_stage(dec, current, R1.ITERS, gen, replay, ctrl=ctrl, outer=outer,
                                               K=cfg["K"], learn=True)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        snap = {"state": {kk: v.detach().cpu() for kk, v in dec.state_dict().items()},
                "controller": ctrl.snapshot(), "config": cfg, "stage": s}
        require(not ck.exists(), f"exists: {ck}")
        torch.save(snap, ck)
        idrec = D.read(C.OUT / f"runs/R15_seed{SEED}/STAGE{s}.json")
        idprev = D.read(C.OUT / f"runs/R15_seed{SEED}/STAGE{s - 1}.json")
        id_end = torch.load(D.ROOT / idrec["checkpoint"], weights_only=True)["state"]
        id_start = torch.load(D.ROOT / idprev["checkpoint"], weights_only=True)["state"]
        id_norm = {b: 0.0 for b in BLOCKS}
        for k_, v in id_end.items():
            if k_ in start_state:
                id_norm[block_of(k_)] += float((v.double() - id_start[k_].double()).pow(2).sum())
        rec = {"index": s, "name": c, "seconds": seconds, "nonfinite_skips": skips,
               "loss_curve_every50": [float(np.nanmean(losses[i:i + 50])) for i in range(0, len(losses), 50)],
               "heldout_normal": DR.normal_stats(dec, held),
               "synthetic_preservation": synthetic_preservation(dec, bank, allowed, c),
               "ranking_audit": ranking_audit(dec, bank, allowed, c, s),
               "effective_update_norm": update_norms(dec, start_state),
               "identity_update_norm": {b: math.sqrt(v) for b, v in id_norm.items()},
               "replay_allocation": {d: len(idx[d]) for d in done}, "controller_a": ctrl.a(),
               "slow_events": len(events), "applied_events": sum(bool(e.get("applied")) for e in events),
               "checkpoint": str(ck.relative_to(D.ROOT)), "checkpoint_sha256": sha(ck),
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
               "elapsed_total_seconds": time.perf_counter() - t_all, "utc": DR.now()}
        D.write_json(run / f"EVENTS{s}.json", events)
        D.write_json(js, rec)
        print("RET1", s, c, f"{seconds:.0f}s", "skips", len(skips),
              "a", {b[-1] if b != "bottleneck.0" else "bn": round(v, 3) for b, v in ctrl.a().items()},
              {k: round(v["patch_ap"], 3) for k, v in rec["synthetic_preservation"].items()},
              "peak", round(rec["peak_allocated_mib"]), flush=True)
        del replay, current


def lock():
    pr = verify_prereg()
    require(not LOCK.exists(), "already locked")
    require((OUT / f"runs/{TAG}/STAGE{len(C.STREAM)}.json").exists(), "RET1 incomplete")
    require(not (SD.d(TAG) / "SCORES_nl.json").exists(), "scored before the lock")
    SD.lock(TAG)
    D.write_json(LOCK, {"utc": DR.now(), "prereg_sha256": sha(PREREG), "sources": pr["sources"], "task": pr["task"],
                        "tag": TAG, "config": D.read(OUT / f"runs/{TAG}/CONFIG.json"),
                        "stage_d_lock_sha256": sha(SD.d(TAG) / "LOCK.json"),
                        "checkpoints": D.read(SD.d(TAG) / "LOCK.json")["nl_checkpoints"]})
    print("locked", sha(LOCK), flush=True)


def dynamics():
    verify_prereg()
    target = OUT / "ANALYSIS_05B.json"
    require(not target.exists(), "one-shot")
    cfg, rows = AC.stage_records(TAG, START)
    require(len(rows) == len(C.STREAM) - START, "RET1 incomplete")
    ev = [e for s in sorted(rows) for e in rows[s]["events"]]
    expect = [{b: 1.0 for b in BLOCKS}] + [dict(zip(BLOCKS, e["a"])) for e in ev[:-1]]
    bad = [i for i, (e, x) in enumerate(zip(ev, expect)) if any(e["a_used"][b] != x[b] for b in BLOCKS)]
    a = np.array([e["a"] for e in ev if "a" in e])
    sat = np.array([e["saturation"] for e in ev if "saturation" in e])
    tr = [e["train"] for e in ev if e.get("train")]
    res = {"config": cfg, "parity": {"events": len(ev), "mismatches": len(bad), "first_mismatch": bad[:5],
                                     "nonfinite_events": sum(not e["finite"] for e in ev),
                                     "applied": sum(bool(e.get("applied")) for e in ev)},
           "max_abs_a_minus_1": float(np.abs(a - 1).max()) if len(a) else None,
           "saturation": {"max": float(sat.max()), "mean": float(sat.mean())} if len(sat) else None,
           "task_dependence": A2.task_dependence({s: r["events"] for s, r in rows.items()}),
           "penalty_vs_signal": OL.decompose(OL.events(TAG, START), cfg["wd_phi"]),
           "a_end": {s: r["stage"]["controller_a"] for s, r in rows.items()},
           "outer": {"J_ret_Q3_mean": float(np.nanmean([e["outer"] for e in ev])),
                     "J_ret_Q2_mean": float(np.nanmean([e["diag_loss"] for e in ev])),
                     "categories_mean": float(np.mean([t["categories"] for t in tr])),
                     "multi_host_categories_mean": float(np.mean([t["multi_host_categories"] for t in tr])),
                     "cross_host_pair_fraction_mean": float(np.mean([t["cross_host_pair_fraction"] for t in tr])),
                     "map_reproduction_max_abs": float(max(t["map_reproduction_max_abs"] for t in tr))},
           "synthetic_preservation": {s: r["stage"]["synthetic_preservation"] for s, r in rows.items()},
           "ranking_audit": {s: r["stage"]["ranking_audit"] for s, r in rows.items()},
           "update_norm_vs_identity": {s: {b: r["stage"]["effective_update_norm"][b] / (r["stage"]["identity_update_norm"][b] + 1e-300)
                                           for b in BLOCKS} for s, r in rows.items()},
           "heldout_vs_identity": AC.heldout_vs_identity(rows, SEED), "integrity": AC.integrity(rows, SEED),
           "gpu_seconds": sum(r["stage"]["seconds"] for r in rows.values()),
           "peak_allocated_mib": max(r["stage"]["peak_allocated_mib"] for r in rows.values()),
           "persistent_bytes": BlockController().persistent_bytes()}
    D.write_json(target, res)
    print("RET1 parity", res["parity"], "max|a-1|", res["max_abs_a_minus_1"], "sat", res["saturation"],
          "penalty", res["penalty_vs_signal"], "outer", res["outer"], flush=True)


def compare():
    lk = D.read(LOCK)
    for s, h in lk["sources"].items():
        require(sha(D.ROOT / s) == h, f"post-lock change: {s}")
    target = OUT / "COMPARE_05B.json"
    require(not target.exists(), "one-shot")
    P, B = A2.load_points()
    require(D.read(SD.d(TAG) / "METRICS_identity.json")["points"] == P["ID"], "identity not reproduced in RET1")
    P["RET1"], B["RET1"] = D.read(SD.d(TAG) / "METRICS_nl.json")["points"], D.read(SD.d(TAG) / "REPLICATES_nl.json")
    res = {f"RET1-{y}": A2.pair(P, B, "RET1", y) for y in ("ID", "N1", "F1", "RND")}
    m, traj = res["RET1-ID"]["macro"], res["RET1-ID"]["macro"]["trajectory_diff"]
    res["reading"] = {"positive_exploratory": bool(m["diff"] > 0 and m["pooled_diff"] > 0 and m["trajectory_mean_diff"] > 0
                                                   and res["RET1-RND"]["macro"]["diff"] > 0
                                                   and res["RET1-N1"]["macro"]["diff"] > 0
                                                   and min(traj.values()) >= -0.025),
                      "min_trajectory_diff_vs_ID": float(min(traj.values())),
                      "rule": D.read(PREREG)["reading"],
                      "matching": {"ID": "no slow events: not compute-matched", "N1": "slow events at the same clock; "
                                   "J_ret costs about 1.5x N1's outer", "F1/RND": "fixed profiles derived from N1"}}
    D.write_json(target, res)
    for k, r in res.items():
        if k != "reading":
            mm = r["macro"]
            print(k, "final", [round(v, 4) for v in mm["final"]], "diff", round(mm["diff"], 4),
                  "ci", [round(v, 4) for v in mm["diff_ci95_images"]], "pooled", round(mm["pooled_diff"], 4),
                  "traj", round(mm["trajectory_mean_diff"], 4), flush=True)
    print("reading", res["reading"]["positive_exploratory"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("scale", "record", "calibrate", "preflight", "prereg", "train", "lock", "dynamics",
                                      "compare"))
    a = ap.parse_args()
    {"scale": scale, "record": record, "calibrate": calibrate, "preflight": preflight, "prereg": prereg, "train": train, "lock": lock,
     "dynamics": dynamics, "compare": compare}[a.stage]()


if __name__ == "__main__":
    main()
