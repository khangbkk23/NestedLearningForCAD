# exps/nlcad/preflight.py
"""Stage B CUDA preflight at full size (normal features only; no labels): VRAM, step overhead, finite hypergradients,
BF16-vs-FP32 agreement of the look-ahead hypergradient. Runs beside RH-01b under its own VRAM cap.

  python -m exps.nlcad.preflight --cap 2.3
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nlcad.controller import BLOCKS, BlockController
from exps.nlcad.fast import nl_train_stage
from exps.nlcad.objectives import NormalLookahead
from exps.rh01 import numerics as NUM
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

OUT = D.ROOT / "results/hope_cad/nlcad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=2.3)
    ap.add_argument("--seed", type=int, default=2711)
    ap.add_argument("--stage", type=int, default=5)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(a.cap * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    proto = RB.verify_lock()
    NUM.use(True)
    rec = D.read(C.OUT / f"runs/R15_seed{a.seed}/STAGE{a.stage}.json")
    path = D.ROOT / rec["checkpoint"]
    require(sha(path) == rec["checkpoint_sha256"], "checkpoint changed")
    sd = torch.load(path, weights_only=True)["state"]
    n = R1.n_train(proto)
    s = a.stage + 1
    c, done = C.STREAM[s - 1], list(C.STREAM[:s - 1])
    idx = C.buffer_index(a.seed, n, done)
    replay = torch.stack([C.load_features(d)[i] for d in done for i in idx[d]])
    current = C.load_features(c)
    res = {"start": rec["checkpoint"], "task": c, "cap_gib": a.cap}

    def fresh():
        dec = DR.new_decoder(a.seed)
        dec.load_state_dict(sd)
        return dec

    for name, kw in (("baseline", {}),
                     ("N_K16", {"ctrl": BlockController(), "outer": "N", "K": 16, "learn": True})):
        dec = fresh()
        outer = NormalLookahead(current, replay, seed=1) if kw.get("outer") == "N" else None
        torch.manual_seed(1)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        losses, skips, ev = nl_train_stage(dec, current, 48, torch.Generator().manual_seed(2), replay,
                                           ctrl=kw.get("ctrl"), outer=outer, K=kw.get("K", 32),
                                           learn=kw.get("learn", True))
        torch.cuda.synchronize()
        res[name] = {"s_per_step": (time.perf_counter() - t0) / 48, "skips": len(skips),
                     "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                     "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                     "events": len(ev), "events_finite": all(e["finite"] for e in ev),
                     "phi_after": kw["ctrl"].phi.tolist() if kw.get("ctrl") is not None else None,
                     "a_after": kw["ctrl"].a() if kw.get("ctrl") is not None else None,
                     "dJ_da_first": ev[0]["dJ_da"] if ev else None, "outer_first": ev[0]["outer"] if ev else None}
        print(name, {k: v for k, v in res[name].items() if k not in ("dJ_da_first", "phi_after")}, flush=True)
        del dec
        torch.cuda.empty_cache()

    # BF16+FP32core vs full-FP32 outer at the same look-ahead event (identical fast step, learn=False)
    agree = {}
    for mode in ("bf16", "fp32"):
        dec = fresh()
        ctrl = BlockController()
        with torch.no_grad():
            ctrl.phi.copy_(torch.linspace(-0.5, 0.5, len(BLOCKS), dtype=torch.float64))
        outer = NormalLookahead(current, replay, seed=3, micro=4, autocast=(mode == "bf16"))   # same loss value
        torch.manual_seed(5)
        _, _, ev = nl_train_stage(dec, current, 10, torch.Generator().manual_seed(6), replay, ctrl=ctrl, outer=outer,
                                  K=10, learn=False)                # one look-ahead event at iteration 9
        agree[mode] = np.array([ev[0]["dJ_da"][b] for b in BLOCKS])
        del dec
        torch.cuda.empty_cache()
    x, y = agree["bf16"], agree["fp32"]
    res["hypergrad_bf16_vs_fp32"] = {"cosine": float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))),
                                     "max_rel_err": float(np.max(np.abs(x - y)) / np.max(np.abs(y))),
                                     "sign_agreement": float(np.mean(np.sign(x) == np.sign(y))),
                                     "bf16": x.tolist(), "fp32": y.tolist()}
    res["overhead_ratio"] = res["N_K16"]["s_per_step"] / res["baseline"]["s_per_step"]
    print("agreement", {k: v for k, v in res["hypergrad_bf16_vs_fp32"].items() if k in ("cosine", "max_rel_err",
                                                                                         "sign_agreement")},
          "overhead", round(res["overhead_ratio"], 3), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    D.write_json(OUT / "PREFLIGHT_B.json", res)


if __name__ == "__main__":
    main()
