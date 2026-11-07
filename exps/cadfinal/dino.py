# exps/cadfinal/dino.py
"""Dinomaly-B + Replay-24 (the RH-01b R15 recipe) on the benchmark streams.

MVTec: the existing RH-01b R15 checkpoints (seeds 2711-2713) are scored; nothing is retrained.
VisA : the same recipe on the VisA 12-class stream, unchanged:
  - frozen DINOv2-B reg4 encoder (exps.dinomaly.encoder, 392^2 full field of view), two fused groups;
  - official decoder and loss (exps.dinomaly.official) through exps.rh01.run.train_stage_guarded
    (667 iterations per task, batch 8 current + 8 replay, StableAdamW + WarmCosine rebuilt each task);
  - D4' numerics (FP32 linear-attention core), TF32 off, 2.6 GiB process cap (exps.dinomaly.run.configure);
  - stage seeds (seed*100+1, seed*100+1) at stage 1, else (seed*100+s, seed*100+s+50) (exps.rh01b.run);
  - Replay-24: exps.rh01.common's plan/allocation rule, parameterised by the stream (equality with the MVTec
    rule is a test); stored as encoder features of the buffered training images.
Scores: official R_D chain (exps.dinomaly.readout.dinomaly): 256^2 maps and top-1% image score.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from exps.cadfinal import common as K
from exps.dinomaly import common as D
from exps.dinomaly import encoder as E
from exps.dinomaly import readout as R
from exps.dinomaly import run as DR
from exps.dinomaly.common import require
from exps.rh01 import numerics as NUM
from exps.rh01 import run as R1
from exps.rh01b import common as RC
from exps.rh01b import run as RB

CAPACITY = RC.CAPACITY
ITERS = R1.ITERS


def data_root(bench):
    return K.ROOT / ("data/visa" if bench == "visa" else "data/mvtec")


# --- Replay-24 rule, parameterised by the stream (exps.rh01.common.buffer_plan / allocation / buffer_index) ----

def buffer_plan(seed, n_train, stream):
    gen = torch.Generator().manual_seed(seed + 9000)
    perms = {c: torch.randperm(n_train[c], generator=gen).tolist() for c in stream}
    order = torch.randperm(len(stream), generator=torch.Generator().manual_seed(seed + 9001)).tolist()
    return perms, [stream[i] for i in order]


def buffer_index(seed, n_train, done, stream):
    perms, priority = buffer_plan(seed, n_train, stream)
    base, rem = divmod(CAPACITY, len(done))
    extra = set([c for c in priority if c in done][:rem])
    return {c: perms[c][:base + (c in extra)] for c in done}


# --- VisA encoder-feature cache ----------------------------------------------------------------------------

def cache_path(task):
    return K.CACHE / "visa_dinomaly" / f"train_{task}.pt"


def cache_visa():
    DR.configure()
    target = K.OUT / "CACHE_visa_dinomaly.json"
    require(not target.exists(), "cache exists")
    ids = K.benchmark_identities("visa")
    enc = E.load(D.CKPT["base"][0], "base", "cuda")
    root, rec = data_root("visa"), {}
    for t in ids["tasks"]:
        path = cache_path(t["task"])
        require(not path.exists(), f"exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        t0, out = time.perf_counter(), []
        rows = t["train"]
        with torch.no_grad():
            for i in range(0, len(rows), 8):
                x = torch.stack([E.preprocess(root / r["relative_path"]) for r in rows[i:i + 8]]).cuda()
                out.append(enc.groups(x).float().cpu())
        f = torch.cat(out)
        require(bool(torch.isfinite(f).all()) and f.shape[1:] == (2, 789, 768), "bad features")
        torch.save({"features": f, "paths": [r["relative_path"] for r in rows]}, path)
        rec[t["task"]] = {"n": len(f), "seconds": time.perf_counter() - t0, "sha256": K.sha(path)}
        print("cached", t["task"], rec[t["task"]], flush=True)
        del f, out
    K.write_json(target, {"encoder_sha256": K.sha(D.CKPT["base"][0]), "tasks": rec, "utc": K.now()})


_VERIFIED = set()


def load_visa(task):
    if task not in _VERIFIED:
        require(K.sha(cache_path(task)) == K.read(K.OUT / "CACHE_visa_dinomaly.json")["tasks"][task]["sha256"],
                f"feature cache changed: {task}")
        _VERIFIED.add(task)
    return torch.load(cache_path(task), weights_only=True, mmap=True)["features"]


# --- training (VisA) ---------------------------------------------------------------------------------------

def run_dir(bench, seed):
    return K.OUT / "runs" / bench / f"DINOMALY_R_seed{seed}"


def train_visa(seed):
    DR.configure()
    NUM.use(True)                                                     # D4'
    require(seed in K.SEEDS, "bad seed")
    ids = K.benchmark_identities("visa")
    stream = [t["task"] for t in ids["tasks"]]
    n = {t["task"]: len(t["train"]) for t in ids["tasks"]}
    run = run_dir("visa", seed)
    run.mkdir(parents=True, exist_ok=True)
    require(not (run / "TRAIN.json").exists(), "run complete")
    torch.cuda.reset_peak_memory_stats()
    dec, stages = None, []
    for s in range(1, len(stream) + 1):
        c = stream[s - 1]
        ck, js = run / f"stage{s}_{c}.pt", run / f"STAGE{s}.json"
        if js.exists():                                               # resume a completed stage
            rec = K.read(js)
            require(K.sha(ck) == rec["checkpoint_sha256"], f"stage {s} checkpoint changed")
            dec = dec or DR.new_decoder(seed)
            dec.load_state_dict(torch.load(ck, weights_only=True)["state"])
            stages.append({**rec, "resumed": True})
            continue
        require(not ck.exists(), f"orphan checkpoint: {ck}")
        dec = dec or DR.new_decoder(seed)
        done = stream[:s - 1]
        replay, idx = None, {}
        if done:
            idx = buffer_index(seed, n, done, stream)
            replay = torch.stack([load_visa(d)[i] for d in done for i in idx[d]])
        current = load_visa(c)
        torch.manual_seed(RB.stage_seeds("R15", seed, s)[0])
        gen = torch.Generator().manual_seed(RB.stage_seeds("R15", seed, s)[1])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        losses, skips = R1.train_stage_guarded(dec, current, ITERS, gen, replay)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        rec = {"index": s, "name": c, "iters": ITERS, "seconds": seconds, "train_images": len(current),
               "loss_curve_every50": [float(np.nanmean(losses[i:i + 50])) for i in range(0, len(losses), 50)],
               "nonfinite_skips": skips,
               "replay": None if replay is None else {
                   "allocation": {d: len(idx[d]) for d in done},
                   "identities": [ids["tasks"][stream.index(d)]["train"][i]["relative_path"] for d in done for i in idx[d]],
                   "feature_bytes": replay.numel() * replay.element_size()},
               "checkpoint": str(ck.relative_to(K.ROOT)),
               "checkpoint_sha256": DR.save_ckpt(dec, ck, {"bench": "visa", "seed": seed, "stage": s, "name": c}),
               "utc": K.now()}
        K.write_json(js, rec)
        stages.append(rec)
        print("visa", seed, s, c, f"{seconds:.0f}s skips {len(skips)} loss {np.nanmean(losses[-50:]):.4f}", flush=True)
        del replay, current
    K.write_json(run / "TRAIN.json", {"seed": seed, "stages": stages, "numerics": "D4prime_fp32core",
                                      "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                                      "persistent_decoder_bytes": DR.state_bytes(dec), "finished_utc": K.now()})


# --- scoring (both benchmarks) -----------------------------------------------------------------------------

def checkpoint(bench, seed, stage):
    if bench == "mvtec":
        rec = K.read(RC.OUT / f"runs/R15_seed{seed}/STAGE{stage}.json")
    else:
        rec = K.read(run_dir("visa", seed) / f"STAGE{stage}.json")
    path = K.ROOT / rec["checkpoint"]
    require(K.sha(path) == rec["checkpoint_sha256"], f"checkpoint changed: {path}")
    return path, rec["checkpoint_sha256"]


@torch.no_grad()
def score(bench, seed):
    """Label-free: for every test image of task j, R_D 256^2 map and image score at stage j (acquisition) and at the
    final stage. Encoder features are recomputed on the fly (frozen encoder, FP32 evaluation as RH-01b)."""
    DR.configure()
    NUM.use(True)
    ids = K.benchmark_identities(bench)
    T, root = len(ids["tasks"]), data_root(bench)
    out = K.OUT / "scores" / bench / f"DINOMALY_R_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "SCORES.json"
    require(not target.exists(), "scored")
    enc = E.load(D.CKPT["base"][0], "base", "cuda")
    decs = {}

    def decoder(stage):
        if stage not in decs:
            path, digest = checkpoint(bench, seed, stage)
            dec = O_decoder()
            dec.load_state_dict(torch.load(path, weights_only=True)["state"])
            decs[stage] = (dec.eval(), digest)
        return decs[stage]

    record = {}
    for j, t in enumerate(ids["tasks"]):
        rows = t["test"]
        for kind, stage in (("acq", j + 1), ("fin", T)):
            dec, digest = decoder(stage)
            maps, img, t0 = [], [], time.perf_counter()
            for i in range(0, len(rows), 8):
                x = torch.stack([E.preprocess(root / r["relative_path"]) for r in rows[i:i + 8]]).cuda()
                f = enc.groups(x).float()
                en, de = dec(f[:, 0], f[:, 1])
                s, im = R.dinomaly(en, de)
                maps.append(s.cpu()); img.append(im.cpu())
            p = out / f"{kind}_{j:02d}.pt"
            require(not p.exists(), f"exists: {p}")
            torch.save({"s256": torch.cat(maps), "image_RD": torch.cat(img),
                        "paths": [r["relative_path"] for r in rows]}, p)
            record[f"{kind}_{j:02d}"] = {"sha256": K.sha(p), "stage": stage, "checkpoint_sha256": digest,
                                         "seconds": time.perf_counter() - t0}
        decs.pop(j + 1, None) if j + 1 != T else None
        print(bench, seed, "scored", t["task"], flush=True)
    K.write_json(target, {"bench": bench, "seed": seed, "files": record, "finished_utc": K.now()})


def O_decoder():
    from exps.dinomaly import official as O
    return O.Decoder(768, 12).cuda()
