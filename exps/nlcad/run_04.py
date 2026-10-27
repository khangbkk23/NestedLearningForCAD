# exps/nlcad/run_04.py
"""NLCAD-04 Stage C/D: one learned Variant-S arm (S1) and its evaluation. Nothing here runs automatically; `train`
needs the user's explicit launch approval after `prereg` has locked the configuration.

S1 = N1 (NLCAD-02: K = 32, lr_phi 0.05, lambda 1e-5, per-task fast optimizer, Replay-24, D4' numerics, seed 2711, stages
7-15 from the RH-01b R15 task-6 checkpoint) with exactly one change: the slow objective is the synthetic-anomaly ranking
look-ahead of exps/nlcad/objectives_s.py (Q3 trains, Q2 is the cross-generator diagnostic; 8 current + 8 replay hosts,
disjoint from the inner batch). The fast objective, decoder, scorer, stream and sampler are unchanged.

  python -m exps.nlcad.run_04 prereg      # CPU: config, sources, task, start checkpoint, bank, evaluation identities
  python -m exps.nlcad.run_04 train       # GPU (approval required)
  python -m exps.nlcad.run_04 lock        # before any score; then stage_d score/metrics/compare --run S1_...
  python -m exps.nlcad.run_04 dynamics    # CPU: controller dynamics, parity, penalty vs signal, Q2/Q3 records
  python -m exps.nlcad.run_04 compare     # CPU: S1 vs ID / N1 / F1 / RND (historical, descriptive)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from exps.dinomaly import common as D
from exps.dinomaly import readout as RO
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nlcad import analyze_02 as A2
from exps.nlcad import analyze_c as AC
from exps.nlcad import bank as BK
from exps.nlcad import openloop as OL
from exps.nlcad import run as R
from exps.nlcad import stage_d as SD
from exps.nlcad.controller import BLOCKS, BlockController
from exps.nlcad.fast import nl_train_stage
from exps.nlcad.objectives import NormalLookahead, WithDiagnostics, _forward
from exps.nlcad.objectives_s import SyntheticRankingS
from exps.rh01 import numerics as NUM
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

OUT = R.OUT
SEED, START, LAM = 2711, 6, 1e-5
TAG = f"S1_per_task_seed{SEED}_from{START}"
TASK = "agents/tasks/task_nlcad_04_synthetic_anomaly_aware_nl.md"
SOURCES = SD.SOURCES + ["exps/nlcad/objectives_s.py", "exps/nlcad/run_04.py", "exps/nlcad/test_nlcad_04.py",
                        "exps/nlcad/chain_04.sh", "exps/nlcad/analyze_02.py", "exps/nlcad/run_02.py"]
PREREG, LOCK = OUT / "PREREG_04.json", OUT / "LOCK_04.json"
NLCAD02_LOCK = OUT / "LOCK_02.json"


def config():
    cfg = R.make_config("S", SEED, START, "per_task", wd_phi=LAM)
    cfg |= {"arm": "S1", "outer": "synthetic_s04", "objective_impl": "exps/nlcad/objectives_s.py:SyntheticRankingS",
            "n_current": 8, "n_replay": 8, "tau": 0.05, "n_neg": 64, "host_separation": "disjoint from inner batch"}
    return cfg


def prereg():
    require(not PREREG.exists(), "already preregistered")
    start = D.read(C.OUT / f"runs/R15_seed{SEED}/STAGE{START}.json")
    bank_man = D.ROOT / f"results/hope_cad/nlcad/BANK_seed{SEED}.json"
    D.write_json(PREREG, {
        "utc": DR.now(), "task": {TASK: sha(D.ROOT / TASK)}, "sources": {s: sha(D.ROOT / s) for s in SOURCES},
        "config": config(), "tag": TAG,
        "start_checkpoint": {"path": start["checkpoint"], "sha256": start["checkpoint_sha256"]},
        "bank_manifest": {"path": str(bank_man.relative_to(D.ROOT)), "sha256": sha(bank_man)},
        "generator_selection": "Q3 (CutPaste) trains, Q2 (DRAEM-like) cross-generator diagnostic: fixed by the task "
                               "before the Stage B forensic; not chosen from real AP",
        "evaluation": {"cohort": "consumed P1 development cohort (bottle/carpet/hazelnut), via exps.nlcad.stage_d",
                       "rh01b_prereg_lock_sha256": sha(C.OUT / "PREREG_LOCK.json"),
                       "nlcad02_lock_sha256": sha(NLCAD02_LOCK)},
        "primary": "final (stage 15) macro native Pixel AP (R_D), S1 - ID",
        "secondary": ["pooled AP", "per-category AP and forgetting (shared acquisition checkpoints)",
                      "stage 7-15 macro trajectory, its mean and its minimum", "Image AUROC", "R_P1",
                      "S1 - N1, S1 - F1, S1 - RND (historical, not compute/objective matched: see report)",
                      "label-free stage-end Q2/Q3 preservation on the allowed hosts"],
        "reading": "one seed, descriptive. 'Positive exploratory' = S1-ID final macro > 0 and pooled > 0 and trajectory "
                   "mean > 0 and S1-RND final > 0 and S1-N1 final > 0 and no transient collapse (every stage 7-15 "
                   "S1-ID macro >= -0.025, the NLCAD-01 single-seed noise floor). Justifies a matched confirmation "
                   "(fixed S profile, SameClock, seeds 2712/2713), not a claim",
        "budget": {"gpu_hours_expected": 1.6, "gpu_hours_hard_cap": 2.5, "vram_cap_gib": 2.3, "arms": 1},
        "bounds": "exactly one training run (S1); no automatic follow-up arm"})
    print("preregistered", sha(PREREG), flush=True)


def verify_prereg():
    pr = D.read(PREREG)
    for s, h in pr["sources"].items():
        require(sha(D.ROOT / s) == h, f"source changed after preregistration: {s}")
    require(sha(D.ROOT / TASK) == pr["task"][TASK], "task file changed after preregistration")
    require(pr["config"] == config(), "config differs from the preregistered one")
    return pr


@torch.no_grad()
def synthetic_preservation(dec, bank, allowed, current_cat):
    """Label-free stage-end record: pooled patch AP and the mean positive-negative score gap per generator and source,
    on every allowed item (dropout off; FP32 grid as in the slow objective)."""
    dec.eval()
    out = {}
    for g in BK.GENERATORS:
        for src in ("current", "replay"):
            items = [it for it in bank[g] if it["host"] in allowed and (it["host"][0] == current_cat) == (src == "current")]
            pos, neg, gaps = [], [], []
            for k in range(0, len(items), 8):
                chunk = items[k:k + 8]
                en, de = _forward(dec, torch.stack([it["features"] for it in chunk]).cuda(), "cuda", True)
                s = RO.grid(en, de).float().cpu()
                for it, m in zip(chunk, s):
                    p, n = m[it["coverage"] >= 0.5], m[it["coverage"] == 0]
                    if p.numel():
                        pos.append(p.numpy())
                        neg.append(n.numpy())
                        gaps.append(float(p.mean() - n.mean()))
            y = np.r_[np.ones(sum(map(len, pos))), np.zeros(sum(map(len, neg)))]
            out[f"{g}/{src}"] = {"patch_ap": float(average_precision_score(y, np.r_[np.concatenate(pos), np.concatenate(neg)])),
                                 "gap_mean": float(np.mean(gaps)), "used": len(gaps), "items": len(items)}
    dec.train()
    return out


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
        syn = SyntheticRankingS(bank, cfg["syn_train"], cfg["syn_diag"], allowed, c, replay_keys,
                                seed=SEED * 1000 + 500 + s, n_current=cfg["n_current"], n_replay=cfg["n_replay"],
                                tau=cfg["tau"], n_neg=cfg["n_neg"])
        normal = NormalLookahead(current, replay, seed=SEED * 1000 + s)
        outer = WithDiagnostics(syn, {"normal_lookahead": normal.evaluate})
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
        rec = {"index": s, "name": c, "seconds": seconds, "nonfinite_skips": skips,
               "loss_curve_every50": [float(np.nanmean(losses[i:i + 50])) for i in range(0, len(losses), 50)],
               "heldout_normal": DR.normal_stats(dec, held),
               "synthetic_preservation": synthetic_preservation(dec, bank, allowed, c),
               "replay_allocation": {d: len(idx[d]) for d in done}, "controller_a": ctrl.a(),
               "slow_events": len(events), "applied_events": sum(bool(e.get("applied")) for e in events),
               "checkpoint": str(ck.relative_to(D.ROOT)), "checkpoint_sha256": sha(ck),
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "utc": DR.now()}
        D.write_json(run / f"EVENTS{s}.json", events)
        D.write_json(js, rec)
        print("S1", s, c, f"{seconds:.0f}s", "skips", len(skips),
              "a", {b[-1] if b != "bottleneck.0" else "bn": round(v, 3) for b, v in ctrl.a().items()},
              {k: round(v["patch_ap"], 3) for k, v in rec["synthetic_preservation"].items()}, flush=True)
        del replay, current


def lock():
    pr = verify_prereg()
    require(not LOCK.exists(), "already locked")
    require((OUT / f"runs/{TAG}/STAGE{len(C.STREAM)}.json").exists(), "S1 incomplete")
    require(not (SD.d(TAG) / "SCORES_nl.json").exists(), "scored before the lock")
    SD.lock(TAG)
    D.write_json(LOCK, {"utc": DR.now(), "prereg_sha256": sha(PREREG), "sources": pr["sources"], "task": pr["task"],
                        "tag": TAG, "config": D.read(OUT / f"runs/{TAG}/CONFIG.json"),
                        "stage_d_lock_sha256": sha(SD.d(TAG) / "LOCK.json"),
                        "checkpoints": D.read(SD.d(TAG) / "LOCK.json")["nl_checkpoints"]})
    print("locked", sha(LOCK), flush=True)


def dynamics():
    verify_prereg()
    target = OUT / "ANALYSIS_04.json"
    require(not target.exists(), "one-shot")
    cfg, rows = AC.stage_records(TAG, START)
    require(len(rows) == len(C.STREAM) - START, "S1 incomplete")
    ev = [e for s in sorted(rows) for e in rows[s]["events"]]
    expect = [{b: 1.0 for b in BLOCKS}] + [dict(zip(BLOCKS, e["a"])) for e in ev[:-1]]
    bad = [i for i, (e, x) in enumerate(zip(ev, expect)) if any(e["a_used"][b] != x[b] for b in BLOCKS)]
    a = np.array([e["a"] for e in ev if "a" in e])
    sat = np.array([e["saturation"] for e in ev if "saturation" in e])
    part = lambda key, src: [e[key][src]["loss"] for e in ev if e.get(key) and e[key][src]["loss"] is not None]
    res = {"config": cfg, "parity": {"events": len(ev), "mismatches": len(bad), "first_mismatch": bad[:5],
                                     "nonfinite_events": sum(not e["finite"] for e in ev),
                                     "applied": sum(bool(e.get("applied")) for e in ev)},
           "max_abs_a_minus_1": float(np.abs(a - 1).max()) if len(a) else None,
           "saturation": {"max": float(sat.max()), "mean": float(sat.mean())} if len(sat) else None,
           "task_dependence": A2.task_dependence({s: r["events"] for s, r in rows.items()}),
           "penalty_vs_signal": OL.decompose(OL.events(TAG, START), cfg["wd_phi"]),
           "a_end": {s: r["stage"]["controller_a"] for s, r in rows.items()},
           "outer": {"Q3_current": float(np.mean(part("train_parts", "current"))),
                     "Q3_replay": float(np.mean(part("train_parts", "replay"))),
                     "Q2_current": float(np.mean(part("cross_generator_parts", "current"))),
                     "Q2_replay": float(np.mean(part("cross_generator_parts", "replay"))),
                     "skipped_train": sum(e["train_parts"][x]["skipped"] for e in ev if e.get("train_parts") for x in ("current", "replay")),
                     "skipped_diag": sum(e["cross_generator_parts"][x]["skipped"] for e in ev if e.get("cross_generator_parts") for x in ("current", "replay"))},
           "synthetic_preservation": {s: r["stage"]["synthetic_preservation"] for s, r in rows.items()},
           "heldout_vs_identity": AC.heldout_vs_identity(rows, SEED), "integrity": AC.integrity(rows, SEED),
           "persistent_bytes": BlockController().persistent_bytes()}
    D.write_json(target, res)
    print("S1 parity", res["parity"], "max|a-1|", res["max_abs_a_minus_1"], "sat", res["saturation"],
          "cos", res["task_dependence"]["cross_stage_cosine"], "penalty", res["penalty_vs_signal"], flush=True)


def compare():
    lk = D.read(LOCK)
    for s, h in lk["sources"].items():
        require(sha(D.ROOT / s) == h, f"post-lock change: {s}")
    target = OUT / "COMPARE_04.json"
    require(not target.exists(), "one-shot")
    P, B = A2.load_points()
    require(D.read(SD.d(TAG) / "METRICS_identity.json")["points"] == P["ID"], "identity not reproduced in S1")
    P["S1"], B["S1"] = D.read(SD.d(TAG) / "METRICS_nl.json")["points"], D.read(SD.d(TAG) / "REPLICATES_nl.json")
    res = {f"S1-{y}": A2.pair(P, B, "S1", y) for y in ("ID", "N1", "F1", "RND")}
    m, traj = res["S1-ID"]["macro"], res["S1-ID"]["macro"]["trajectory_diff"]
    res["reading"] = {"positive_exploratory": bool(m["diff"] > 0 and m["pooled_diff"] > 0 and m["trajectory_mean_diff"] > 0
                                                   and res["S1-RND"]["macro"]["diff"] > 0
                                                   and res["S1-N1"]["macro"]["diff"] > 0
                                                   and min(traj.values()) >= -0.025),
                      "min_trajectory_diff_vs_ID": float(min(traj.values())),
                      "rule": D.read(PREREG)["reading"],
                      "matching": {"ID": "no slow events: not compute-matched", "N1": "compute-matched, different outer",
                                   "F1/RND": "fixed profiles derived from N1, not from S1: not a matched S control"}}
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
    ap.add_argument("stage", choices=("prereg", "train", "lock", "dynamics", "compare"))
    a = ap.parse_args()
    {"prereg": prereg, "train": train, "lock": lock, "dynamics": dynamics, "compare": compare}[a.stage]()


if __name__ == "__main__":
    main()
