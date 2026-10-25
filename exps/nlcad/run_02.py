# exps/nlcad/run_02.py
"""NLCAD-02 runner (agents/tasks/task_nlcad_02_controller_penalty_revision.md, v2). Three bounded arms over the
nested segment (stages 7-15, seed 2711, RH-01b R15 task-6 start), launched through the unchanged Stage C runner:

  N1   arm N with wd_phi = 1e-5 (the only change from NLCAD-01 Stage C N)
  F1   arm F, phi fixed at the mean of N1's recorded phi over its slow events (retrospective fixed-profile diagnostic)
  RND  arm F, phi = s * centred N(0, I) draw (seed 4242), s matching F1's centred log-multiplier norm (one random draw)

  python -m exps.nlcad.run_02 prereg              # before any NLCAD-02 GPU work; one-shot
  python -m exps.nlcad.run_02 train --arm N1|F1|RND
  python -m exps.nlcad.run_02 profiles            # after N1; one-shot
  python -m exps.nlcad.run_02 lock                # after all three arms; before any NLCAD-02 score
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nlcad import run as R
from exps.nlcad import stage_d as SD
from exps.nlcad.controller import BLOCKS, C_BOUND, BlockController
from exps.rh01b import common as C

OUT = R.OUT
SEED, START, LAM, RND_SEED = 2711, 6, 1e-5, 4242
ARMS2 = ("N1", "F1", "RND")
TASK = "agents/tasks/task_nlcad_02_controller_penalty_revision.md"
TASK_V1 = "agents/tasks/task_nlcad_02_controller_penalty_revision.v1_proposal.md"
NLCAD01_LOCK = "results/hope_cad/nlcad/stage_d/N_per_task_seed2711_from6/LOCK.json"
SOURCES = SD.SOURCES + ["exps/nlcad/run_02.py", "exps/nlcad/analyze_02.py", "exps/nlcad/analyze_c.py",
                        "exps/nlcad/chain_02.sh", "exps/nlcad/test_nlcad_02.py"]
PREREG, LOCK = OUT / "PREREG_02.json", OUT / "LOCK_02.json"
PROFILE = {"F1": OUT / "F1_PROFILE.json", "RND": OUT / "RND_PROFILE.json"}


def tag(arm):
    return f"{arm}_per_task_seed{SEED}_from{START}"


def config(arm, profile=None):
    require(arm in ARMS2, f"unknown arm {arm}")
    if arm == "N1":
        return R.make_config("N", SEED, START, "per_task", wd_phi=LAM)
    require(profile is not None, f"{arm} needs a profile")
    return R.make_config("F", SEED, START, "per_task", wd_phi=LAM, profile=profile)


def centred_log_a(phi, c=C_BOUND):
    h = c * np.tanh(np.asarray(phi, dtype=np.float64))
    return h - h.mean()                                              # = log a - mean log a (mean-1 normalisation)


def multipliers(phi):
    ctrl = BlockController()
    with torch.no_grad():
        ctrl.phi.copy_(torch.as_tensor(np.asarray(phi, dtype=np.float64)))
    return ctrl.multipliers().detach().numpy()


def f1_phi(events):
    """Arithmetic mean of the recorded post-step phi over every slow event of the segment."""
    return np.mean([e["phi"] for e in events], axis=0)


def rnd_phi(target_norm, seed=RND_SEED, tol=1e-12):
    z = torch.randn(len(BLOCKS), generator=torch.Generator().manual_seed(seed), dtype=torch.float64).numpy()
    z = z - z.mean()
    norm = lambda s: float(np.linalg.norm(centred_log_a(s * z)))
    hi = 1.0
    while norm(hi) < target_norm:
        hi *= 2
        require(hi < 1e6, "target norm unreachable inside the bound")
    lo = 0.0
    while hi - lo > tol * max(1.0, hi):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if norm(mid) < target_norm else (lo, mid)
    s = (lo + hi) / 2
    require(abs(norm(s) - target_norm) < 1e-9, "norm matching failed")
    return z, s, s * z


def prereg():
    require(not PREREG.exists(), "already preregistered")
    start = D.read(C.OUT / f"runs/R15_seed{SEED}/STAGE{START}.json")
    D.write_json(PREREG, {
        "utc": DR.now(), "task": {TASK: sha(D.ROOT / TASK), TASK_V1: sha(D.ROOT / TASK_V1)},
        "sources": {s: sha(D.ROOT / s) for s in SOURCES},
        "start_checkpoint": {"path": start["checkpoint"], "sha256": start["checkpoint_sha256"]},
        "identity_reference": {"nlcad01_lock": NLCAD01_LOCK, "sha256": sha(D.ROOT / NLCAD01_LOCK)},
        "N1_config": config("N1"), "lambda": LAM, "stage_c_N_config": D.read(OUT / f"runs/{tag('N')}/CONFIG.json"),
        "F1_rule": "phi_F1 = arithmetic mean of N1's recorded post-step phi over all slow events of stages 7-15; arm F "
                   "(learn=False), same slow-event machinery; retrospective fixed-profile diagnostic",
        "RND_rule": {"seed": RND_SEED, "draw": "torch.randn(9, float64, Generator(seed)), centred",
                     "scale": "bisection so that ||c*tanh(phi) - mean|| equals F1's (|diff| < 1e-9)",
                     "label": "single random control, not a statistical noise estimate"},
        "primary": "final (stage 15) macro native Pixel AP (R_D), N1 - identity",
        "secondary": ["pooled AP", "per-category AP and forgetting (shared acquisition checkpoints)",
                      "stage 7-15 macro trajectory and its mean", "Image AUROC", "R_P1", "N1-F1, N1-RND, F1-RND"],
        "reading": "one seed, descriptive; no fixed effect-size threshold; 'modest consistent positive' = N1-ID final "
                   "macro > 0 and pooled > 0 and stage 7-15 trajectory mean > 0 and N1-RND final macro > 0; justifies "
                   "another validation round, not confirmation",
        "bounds": "exactly three training runs; lambda=0 not launched; no automatic follow-up"})
    print("preregistered", sha(PREREG), flush=True)


def verify_prereg():
    pr = D.read(PREREG)
    for s, h in pr["sources"].items():
        require(sha(D.ROOT / s) == h, f"source changed after preregistration: {s}")
    require(sha(D.ROOT / TASK) == pr["task"][TASK], "task file changed after preregistration")
    return pr


def complete(arm):
    return (OUT / f"runs/{tag(arm)}/STAGE{len(C.STREAM)}.json").exists()


def events(arm):
    return [e for s in range(START + 1, len(C.STREAM) + 1) for e in D.read(OUT / f"runs/{tag(arm)}/EVENTS{s}.json")]


def profiles():
    verify_prereg()
    require(complete("N1"), "N1 incomplete")
    require(not any(p.exists() for p in PROFILE.values()), "profiles are one-shot")
    ev = events("N1")
    require(all(e["finite"] and e.get("applied") for e in ev), "N1 has non-applied slow events")
    files = {f"EVENTS{s}.json": sha(OUT / f"runs/{tag('N1')}/EVENTS{s}.json") for s in range(START + 1, len(C.STREAM) + 1)}
    phi = f1_phi(ev)
    target = float(np.linalg.norm(centred_log_a(phi)))
    D.write_json(PROFILE["F1"], {"rule": D.read(PREREG)["F1_rule"], "events": len(ev), "source_events_sha256": files,
                                 "phi": dict(zip(BLOCKS, phi.tolist())), "a": dict(zip(BLOCKS, multipliers(phi).tolist())),
                                 "centred_log_a_norm": target})
    z, s, rp = rnd_phi(target)
    cos = float(centred_log_a(rp) @ centred_log_a(phi) / (np.linalg.norm(centred_log_a(rp)) * target + 1e-300))
    D.write_json(PROFILE["RND"], {"rule": D.read(PREREG)["RND_rule"], "z": z.tolist(), "scale": s,
                                  "phi": dict(zip(BLOCKS, rp.tolist())), "a": dict(zip(BLOCKS, multipliers(rp).tolist())),
                                  "centred_log_a_norm": float(np.linalg.norm(centred_log_a(rp))),
                                  "cosine_with_F1_centred_log_a": cos})
    print("F1", {b[-1] if b != "bottleneck.0" else "bn": round(v, 3) for b, v in zip(BLOCKS, multipliers(phi))},
          "RND", {b[-1] if b != "bottleneck.0" else "bn": round(v, 3) for b, v in zip(BLOCKS, multipliers(rp))},
          "cos", round(cos, 3), flush=True)


def train(arm):
    verify_prereg()
    profile = None
    if arm != "N1":
        require(PROFILE[arm].exists(), f"{arm} profile missing")
        profile = D.read(PROFILE[arm])["phi"]
    R.train(config(arm, profile), tag(arm))


def lock():
    pr = verify_prereg()
    require(not LOCK.exists(), "already locked")
    require(all(complete(a) for a in ARMS2), "an arm is incomplete")
    for a in ARMS2:
        require(not (SD.d(tag(a)) / "SCORES_nl.json").exists(), f"{a} scored before the lock")
        SD.lock(tag(a))
    D.write_json(LOCK, {
        "utc": DR.now(), "prereg_sha256": sha(PREREG), "sources": pr["sources"], "task": pr["task"],
        "profiles": {a: sha(p) for a, p in PROFILE.items()},
        "arms": {a: {"tag": tag(a), "config": D.read(OUT / f"runs/{tag(a)}/CONFIG.json"),
                     "stage_d_lock_sha256": sha(SD.d(tag(a)) / "LOCK.json"),
                     "checkpoints": D.read(SD.d(tag(a)) / "LOCK.json")["nl_checkpoints"]} for a in ARMS2},
        "identity_reference": pr["identity_reference"]})
    print("locked", sha(LOCK), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prereg", "train", "profiles", "lock"))
    ap.add_argument("--arm", choices=ARMS2)
    a = ap.parse_args()
    if a.stage == "train":
        require(a.arm is not None, "--arm required")
        train(a.arm)
    else:
        {"prereg": prereg, "profiles": profiles, "lock": lock}[a.stage]()


if __name__ == "__main__":
    main()
