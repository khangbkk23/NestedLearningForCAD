# exps/rh01b/run.py
"""RH-01b stages (agents/tasks/task_rh_01b_replay_headroom_fp32core.md): RH-01 rerun with the FP32 linear-attention
core in every training forward pass (D4'); R15 stage 1 trained here.

  python -m exps.rh01b.run prerun | lock | cache | probe
  python -m exps.rh01b.run train --arm R15|J15 --seed 2711
  python -m exps.rh01b.run score --seeds 2711
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import readout as R
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.rh01 import numerics as NUM
from exps.rh01 import run as R1
from exps.rh01b import common as C

OUT = C.OUT
SEEDS = DR.SEEDS
ITERS, JOINT_ITERS = R1.ITERS, R1.JOINT_ITERS
TASK = "agents/tasks/task_rh_01b_replay_headroom_fp32core.md"
SOURCES = ["exps/rh01b/common.py", "exps/rh01b/run.py", "exps/rh01b/analysis.py", "exps/rh01b/test_rh01b.py",
           "exps/rh01/common.py", "exps/rh01/run.py", "exps/rh01/numerics.py", "exps/rh01/analysis.py",
           "exps/dinomaly/official.py", "exps/dinomaly/encoder.py", "exps/dinomaly/common.py",
           "exps/dinomaly/readout.py", "exps/dinomaly/run.py", "exps/dinomaly/analysis.py",
           "exps/task_p/p2/metrics.py", "exps/task_p/p1_readout.py", TASK,
           "results/hope_cad/rh01/CACHE.json", "results/hope_cad/dinomaly/FROZEN_BASELINE.json"]


def numerics_d4prime():
    """D4': FP32 linear-attention core in every forward of this process (identical to the original in FP32)."""
    NUM.use(True)


def stage_seeds(arm, seed, stage):
    if arm == "R15":
        return (seed * 100 + 1, seed * 100 + 1) if stage == 1 else (seed * 100 + stage, seed * 100 + stage + 50)
    return R1.stage_seeds("J15", seed, 1)


# --- lock --------------------------------------------------------------------------------------------


def prerun():
    """Re-derive RH-01's ledger (identities, buffer, exposure, scheduler) and require equality with RH-01's."""
    proto, led = R1.prerun()
    ref = D.read(C.RH01 / "PREREG_PRERUN.json")
    same = {k: led[k] == ref[k] for k in ("identities", "n_train")}
    same["boundaries"] = all(led["seeds"][s]["boundaries"] == {int(b): v for b, v in ref["seeds"][str(s)]["boundaries"].items()}
                             for s in SEEDS)
    same["J15_exposure"] = all(led["seeds"][s]["J15_sample_gradients_per_class"] == ref["seeds"][str(s)]["J15_sample_gradients_per_class"]
                               for s in SEEDS)
    require(all(same.values()), f"ledger differs from RH-01: {same}")
    led["equal_to_RH01"] = same
    return proto, led


def lock():
    target = OUT / "PREREG_LOCK.json"
    require(not target.exists(), "already locked")
    proto, led = prerun()
    D.write_json(OUT / "PREREG_PRERUN.json", led)
    D.write_json(OUT / "PROTOCOL.json", {"stream": C.STREAM, "eval": C.EVAL, "capacity": C.CAPACITY, "seeds": SEEDS,
                                         "iters_per_task": ITERS, "joint_iters": JOINT_ITERS, "numerics": "D4prime_fp32core",
                                         **proto})
    D.write_json(target, {"locked_utc": DR.now(), "sources": {s: sha(D.ROOT / s) for s in SOURCES},
                          "protocol_sha256": sha(OUT / "PROTOCOL.json"), "prerun_sha256": sha(OUT / "PREREG_PRERUN.json"),
                          "encoder_sha256": D.CKPT["base"][1]})
    print("locked", sha(target), flush=True)


def verify_lock():
    lk = D.read(OUT / "PREREG_LOCK.json")
    digests = dict(lk["sources"])
    for rec in sorted(OUT.glob("AMENDMENT_*.json")):
        for name, a in D.read(rec)["files"].items():
            require(digests[name] == a["locked_sha256"], f"amendment chain broken: {name}")
            digests[name] = a["amended_sha256"]
    for name, digest in digests.items():
        require(sha(D.ROOT / name) == digest, f"post-lock change: {name}")
    require(sha(OUT / "PROTOCOL.json") == lk["protocol_sha256"], "protocol changed")
    return D.read(OUT / "PROTOCOL.json")


def cache():
    """No extraction: every RH-01 cache file must match RH-01's CACHE.json."""
    verify_lock()
    target = OUT / "CACHE_CHECK.json"
    require(not target.exists(), "one-shot")
    rec = D.read(C.RH01 / "CACHE.json")
    res = {c: {"sha256": sha(C.CACHE / f"train_{c}.pt"), "expected": rec[c]["sha256"]} for c in C.STREAM}
    require(all(v["sha256"] == v["expected"] for v in res.values()), "feature cache differs from RH-01")
    D.write_json(target, res)
    print("cache verified", len(res), flush=True)


def probe():
    DR.configure()
    verify_lock()
    numerics_d4prime()
    target = OUT / "PROBE.json"
    require(not target.exists(), "one-shot")
    pool = C.Pool([C.load_features(c) for c in C.STREAM])
    res = {}
    for name, current, replay in (("J15", pool, None),
                                  ("R15", C.load_features("zipper"),
                                   torch.stack([C.load_features(c)[i] for c in ("bottle", "cable", "capsule")
                                                for i in range(8)]))):
        dec = DR.new_decoder(0)
        gen = torch.Generator().manual_seed(1)
        R1.train_stage_guarded(dec, current, 10, gen, replay)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        R1.train_stage_guarded(dec, current, 40, gen, replay)
        torch.cuda.synchronize()
        res[name] = {"s_per_step": (time.perf_counter() - t1) / 40,
                     "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
        del dec
        torch.cuda.empty_cache()
        print("probe", name, res[name], flush=True)
    res["pass"] = all(v["s_per_step"] <= R1.PROBE_LIMIT for v in res.values())
    D.write_json(target, res)
    require(res["pass"], "resource probe above limit")


# --- training ----------------------------------------------------------------------------------------


def train(arm, seed):
    DR.configure()
    proto = verify_lock()
    numerics_d4prime()
    require(arm in ("R15", "J15") and seed in SEEDS, "bad arm/seed")
    run = OUT / f"runs/{arm}_seed{seed}"
    require(not (run / "TRAIN.json").exists(), "run already completed")
    run.mkdir(parents=True, exist_ok=True)
    held = {c: DR.load_split("heldout", c) for c in C.EVAL}
    n = R1.n_train(proto)
    torch.cuda.reset_peak_memory_stats()
    if arm == "J15":
        require(not any(run.glob("*.pt")), "J15 restarts from scratch: remove the partial run first (recorded)")
        torch.manual_seed(stage_seeds(arm, seed, 1)[0])
        dec = DR.new_decoder(seed)
        gen = torch.Generator().manual_seed(stage_seeds(arm, seed, 1)[1])
        pool = C.Pool([C.load_features(c) for c in C.STREAM])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        losses, skips = R1.train_stage_guarded(dec, pool, JOINT_ITERS, gen)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        path = run / "stage1_joint15.pt"
        stages = [{"index": 1, "name": "joint15", "iters": JOINT_ITERS, "seconds": seconds,
                   "loss_curve_every500": [float(np.nanmean(losses[i:i + 500])) for i in range(0, len(losses), 500)],
                   "loss_last50": float(np.nanmean(losses[-50:])), "nonfinite_skips": skips,
                   "heldout_normal": DR.normal_stats(dec, held), "checkpoint": str(path.relative_to(D.ROOT)),
                   "checkpoint_sha256": DR.save_ckpt(dec, path, {"arm": arm, "seed": seed, "stage": 1})}]
        print(arm, seed, f"{seconds:.0f}s skips {len(skips)}",
              {k: round(v["global_cosine"], 4) for k, v in stages[0]["heldout_normal"].items()}, flush=True)
    else:
        dec, stages = None, []
        for s in range(1, len(C.STREAM) + 1):
            c = C.STREAM[s - 1]
            ck, js = run / f"stage{s}_{c}.pt", run / f"STAGE{s}.json"
            if js.exists():                                            # resume: completed stage
                rec = D.read(js)
                require(sha(ck) == rec["checkpoint_sha256"], f"stage {s} checkpoint changed")
                if dec is None:
                    dec = DR.new_decoder(seed)
                dec.load_state_dict(torch.load(ck, weights_only=True)["state"])
                stages.append({**rec, "resumed": True})
                continue
            require(not ck.exists(), f"orphan checkpoint without record: {ck} (decide manually)")
            if dec is None:                                           # stage 1: fresh decoder, DRF-01 arm-A seeds
                dec = DR.new_decoder(seed)
            done = list(C.STREAM[:s - 1])
            replay, rows, idx = None, [], {}
            if done:
                idx = C.buffer_index(seed, n, done)
                rows = [proto["train"][d][i] for d in done for i in idx[d]]
                replay = torch.stack([C.load_features(d)[i] for d in done for i in idx[d]])
            current = C.load_features(c)
            torch.manual_seed(stage_seeds(arm, seed, s)[0])
            gen = torch.Generator().manual_seed(stage_seeds(arm, seed, s)[1])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            losses, skips = R1.train_stage_guarded(dec, current, ITERS, gen, replay)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            rec = {"index": s, "name": c, "iters": ITERS, "seconds": seconds, "train_images": len(current),
                   "loss_curve_every50": [float(np.nanmean(losses[i:i + 50])) for i in range(0, len(losses), 50)],
                   "loss_last50": float(np.nanmean(losses[-50:])), "nonfinite_skips": skips,
                   "heldout_normal": DR.normal_stats(dec, held),
                   "replay": None if replay is None else {
                       "allocation": {d: len(idx[d]) for d in done},
                       "identities": [r["relative_path"] for r in rows], "png_bytes": sum(r["bytes"] for r in rows),
                       "feature_bytes": replay.numel() * replay.element_size()},
                   "checkpoint": str(ck.relative_to(D.ROOT)),
                   "checkpoint_sha256": DR.save_ckpt(dec, ck, {"arm": arm, "seed": seed, "stage": s, "name": c}),
                   "utc": DR.now()}
            D.write_json(js, rec)
            stages.append(rec)
            print(arm, seed, s, c, f"{seconds:.0f}s skips {len(skips)}", "loss", round(rec["loss_last50"], 4),
                  {k: round(v["global_cosine"], 4) for k, v in rec["heldout_normal"].items()}, flush=True)
            del replay, current
    D.write_json(run / "TRAIN.json", {"arm": arm, "seed": seed, "stages": stages, "finished_utc": DR.now(),
                                      "numerics": "D4prime_fp32core",
                                      "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                                      "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                                      "persistent_decoder_bytes": DR.state_bytes(dec)})


# --- scoring (identical to RH-01; FP32 evaluation, where D4' equals the original forward) -----------------


def checkpoints(seeds):
    out = []
    for seed in seeds:
        for arm in ("R15", "J15"):
            rec = D.read(OUT / f"runs/{arm}_seed{seed}/TRAIN.json")
            for st in rec["stages"]:
                path = D.ROOT / st["checkpoint"]
                require(sha(path) == st["checkpoint_sha256"], f"checkpoint changed: {path}")
                admitted = list(C.EVAL) if arm == "J15" else [c for c in C.EVAL if C.STREAM.index(c) < st["index"]]
                if admitted:
                    out.append((arm, seed, st["index"], admitted, path))
    return out


@torch.no_grad()
def score(seeds):
    DR.configure()
    proto = verify_lock()
    numerics_d4prime()
    tag = "_".join(map(str, seeds))
    target = OUT / f"scores/SCORES_{tag}.json"
    require(not target.exists(), "scoring is one-shot per seed batch")
    key = "results/hope_cad/dinomaly/features/dev.pt"
    require(sha(D.ROOT / key) == R1.frozen()[key]["sha256"], "dev features not frozen")
    dev = torch.load(D.ROOT / key, weights_only=True)
    index = {p: i for i, p in enumerate(dev["paths"])}
    record = {}
    for arm, seed, stage, admitted, path in checkpoints(seeds):
        dec = O.Decoder(768, 12).cuda()
        dec.load_state_dict(torch.load(path, weights_only=True)["state"])
        dec.eval()
        rows = [r for c in admitted for r in proto["evaluation"][c]]
        x = dev["features"][[index[r["relative_path"]] for r in rows]]
        s256, img, grids = [], [], []
        for xb in x.split(10):
            en, de = dec(xb[:, 0].cuda(), xb[:, 1].cuda())
            s, i = R.dinomaly(en, de)
            s256.append(s.cpu()); img.append(i.cpu()); grids.append(R.grid(en, de).cpu())
        out = OUT / f"scores/maps/{arm}_seed{seed}_stage{stage}.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        require(not out.exists(), f"exists: {out}")
        torch.save({"paths": [r["relative_path"] for r in rows], "s256": torch.cat(s256), "image_RD": torch.cat(img),
                    "grid": torch.cat(grids)}, out)
        record[f"{arm}/{seed}/{stage}"] = {"maps": str(out.relative_to(D.ROOT)), "sha256": sha(out),
                                           "checkpoint_sha256": sha(path), "admitted": admitted}
        del dec
    D.write_json(target, record)
    print("scored", tag, len(record), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prerun", "lock", "cache", "probe", "train", "score"))
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--seeds", type=int, nargs="+")
    a = ap.parse_args()
    if a.stage == "prerun":
        _, led = prerun()
        print("equal_to_RH01", led["equal_to_RH01"], "cost", led["cost_projection_gpu_h"])
    elif a.stage == "lock":
        lock()
    elif a.stage == "cache":
        cache()
    elif a.stage == "probe":
        probe()
    elif a.stage == "train":
        train(a.arm, a.seed)
    else:
        score(a.seeds)


if __name__ == "__main__":
    main()
