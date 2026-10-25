# exps/nlcad/run.py
"""NL-CAD stream runner (Stage B/C; GPU). Runs one arm over a contiguous segment of the RH-01 15-task stream,
starting from an RH-01b R15 checkpoint (shared start) or from scratch, with the RH-01b data, Replay-24 buffer,
numerics (D4' FP32 linear-attention core) and seeds. The slow controller persists across task boundaries; the fast
optimizer is re-created per task exactly as in the Replay baseline.

Arms (one config each; nothing here is tuned on development labels):
  R   Replay-24 through the NL loop with no controller (must reproduce RH-01b R15)
  NR  Replay-24 without LR warm restart (one WarmCosine across the segment; moments reset per task)
  F   fixed blockwise profile: identical slow-event machinery and compute, learn=False
  N   learned controller, Variant N (normal look-ahead), slow clock K=32
  SC  SameClock: as N with K=1
  S   learned controller, Variant S (synthetic ranking look-ahead on one PTAG generator; the other is diagnostic)

  python -m exps.nlcad.run --arm N --seed 2711 --start 6 --base per_task
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nlcad.controller import BLOCKS, BlockController
from exps.nlcad.fast import nl_train_stage
from exps.nlcad import bank as BK
from exps.nlcad.objectives import NormalLookahead, SyntheticRanking, WithDiagnostics
from exps.rh01 import numerics as NUM
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

OUT = D.ROOT / "results/hope_cad/nlcad"
ARMS = ("R", "NR", "F", "N", "SC", "S")


def global_schedule(total_iters):
    """The recipe's WarmCosine values over a whole segment (no warm restart at task boundaries)."""
    r = O.RECIPE
    warm = np.linspace(0, r["lr"], r["warmup_iters"])
    it = np.arange(total_iters - r["warmup_iters"])
    return np.concatenate((warm, r["final_lr"] + 0.5 * (r["lr"] - r["final_lr"]) * (1 + np.cos(np.pi * it / len(it)))))


def start_state(seed, start):
    if start == 0:
        return None, None
    rec = D.read(C.OUT / f"runs/R15_seed{seed}/STAGE{start}.json")
    path = D.ROOT / rec["checkpoint"]
    require(sha(path) == rec["checkpoint_sha256"], "start checkpoint changed")
    return torch.load(path, weights_only=True)["state"], rec


def make_config(arm, seed, start, base, K=32, lr_phi=0.05, wd_phi=1e-3, profile=None):
    require(arm in ARMS and base in ("per_task", "single"), "bad arm/base")
    if arm == "NR":
        base = "single"
    return {"arm": arm, "seed": seed, "start": start, "base": base, "K": 1 if arm == "SC" else K,
            "lr_phi": lr_phi, "wd_phi": wd_phi, "profile": profile if arm == "F" else None,
            "learn": arm in ("N", "SC", "S"), "outer": {"N": "normal", "SC": "normal", "F": "normal",
                                                       "S": "synthetic"}.get(arm), "numerics": "D4prime_fp32core",
            "syn_train": "Q3", "syn_diag": "Q2", "diagnostics": True}


def cap():
    """Per-process VRAM cap (GiB) when sharing the GPU with another run: NLCAD_CAP_GIB (default 2.3)."""
    import os
    g = float(os.environ.get("NLCAD_CAP_GIB", "2.3"))
    torch.cuda.set_per_process_memory_fraction(g * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    return g


def train(cfg, tag):
    DR.configure()
    cfg_cap = cap()
    proto = RB.verify_lock()                                         # RH-01b data, identities and lock
    NUM.use(True)
    run = OUT / f"runs/{tag}"
    run.mkdir(parents=True, exist_ok=True)
    cfg_path = run / "CONFIG.json"
    if cfg_path.exists():
        require(D.read(cfg_path) == cfg, "config differs from the recorded one")
    else:
        D.write_json(cfg_path, cfg)
    seed, start = cfg["seed"], cfg["start"]
    n = R1.n_train(proto)
    held = {c: DR.load_split("heldout", c) for c in C.EVAL}
    sd0, _ = start_state(seed, start)
    dec = DR.new_decoder(seed)
    if sd0 is not None:
        dec.load_state_dict(sd0)
    ctrl = None
    if cfg["arm"] != "R" and cfg["arm"] != "NR":
        ctrl = BlockController(lr=cfg["lr_phi"], weight_decay=cfg["wd_phi"])
        if cfg["profile"] is not None:
            with torch.no_grad():
                ctrl.phi.copy_(torch.tensor([cfg["profile"][b] for b in BLOCKS], dtype=torch.float64))
    bank = hosts = None
    if cfg["outer"] == "synthetic" or cfg.get("diagnostics"):
        bank, hosts = BK.load_bank(seed)
    stages = list(range(start + 1, len(C.STREAM) + 1))
    sched = global_schedule(len(stages) * R1.ITERS) if cfg["base"] == "single" else None
    torch.cuda.reset_peak_memory_stats()
    for k, s in enumerate(stages):
        c = C.STREAM[s - 1]
        ck, js = run / f"stage{s}_{c}.pt", run / f"STAGE{s}.json"
        if js.exists():                                                # resume
            rec = D.read(js)
            require(sha(ck) == rec["checkpoint_sha256"], f"stage {s} checkpoint changed")
            snap = torch.load(ck, weights_only=False)
            dec.load_state_dict(snap["state"])
            if ctrl is not None:
                ctrl.restore(snap["controller"])
            continue
        require(not ck.exists(), f"orphan checkpoint: {ck}")
        done = list(C.STREAM[:s - 1])
        replay, idx = None, {}
        if done:
            idx = C.buffer_index(seed, n, done)
            replay = torch.stack([C.load_features(d)[i] for d in done for i in idx[d]])
        current = C.load_features(c)
        dseed, gseed = RB.stage_seeds("R15", seed, s)
        torch.manual_seed(dseed)
        gen = torch.Generator().manual_seed(gseed)
        outer = None
        if cfg["outer"] is not None:
            keys = BK.allowed(hosts, seed, n, s) if bank is not None else set()
            normal = NormalLookahead(current, replay, seed=seed * 1000 + s)
            syn = SyntheticRanking(bank, cfg["syn_train"], cfg["syn_diag"], allowed=lambda ctx, k=keys: k,
                                   seed=seed * 1000 + 500 + s) if bank is not None else None
            if cfg["outer"] == "normal":
                diags = {} if syn is None else {f"syn_{g}": (lambda d, x, g=g: syn.evaluate(d, x, g))
                                                for g in (cfg["syn_train"], cfg["syn_diag"])}
                outer = WithDiagnostics(normal, diags)
            else:
                outer = WithDiagnostics(syn, {"normal_lookahead": normal.evaluate})
        lr_at = None if sched is None else (lambda i, o=k * R1.ITERS: float(sched[o + i]))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        losses, skips, events = nl_train_stage(dec, current, R1.ITERS, gen, replay, ctrl=ctrl, outer=outer,
                                               K=cfg["K"], learn=cfg["learn"], lr_at=lr_at)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        snap = {"state": {kk: v.detach().cpu() for kk, v in dec.state_dict().items()},
                "controller": ctrl.snapshot() if ctrl is not None else None, "config": cfg, "stage": s}
        require(not ck.exists(), f"exists: {ck}")
        torch.save(snap, ck)
        rec = {"index": s, "name": c, "seconds": seconds, "nonfinite_skips": skips,
               "loss_curve_every50": [float(np.nanmean(losses[i:i + 50])) for i in range(0, len(losses), 50)],
               "heldout_normal": DR.normal_stats(dec, held),
               "replay_allocation": {d: len(idx[d]) for d in done},
               "controller_a": ctrl.a() if ctrl is not None else None,
               "slow_events": len(events), "checkpoint": str(ck.relative_to(D.ROOT)), "checkpoint_sha256": sha(ck),
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "utc": DR.now()}
        D.write_json(run / f"EVENTS{s}.json", events)
        D.write_json(js, rec)
        print(cfg["arm"], seed, s, c, f"{seconds:.0f}s", "skips", len(skips),
              "a", None if ctrl is None else {b[-1] if b != "bottleneck.0" else "bn": round(v, 3)
                                              for b, v in ctrl.a().items()},
              {kk: round(v["global_cosine"], 4) for kk, v in rec["heldout_normal"].items()}, flush=True)
        del replay, current


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, default=2711)
    ap.add_argument("--start", type=int, default=6)
    ap.add_argument("--base", default="per_task", choices=("per_task", "single"))
    ap.add_argument("--tag")
    a = ap.parse_args()
    cfg = make_config(a.arm, a.seed, a.start, a.base)
    train(cfg, a.tag or f"{a.arm}_{a.base}_seed{a.seed}_from{a.start}")


if __name__ == "__main__":
    main()
