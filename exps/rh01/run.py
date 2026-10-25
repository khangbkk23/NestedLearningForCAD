# exps/rh01/run.py
"""RH-01 stages (task file agents/tasks/task_rh_01_replay_headroom_15task.md).

  python -m exps.rh01.run prerun                     # CPU: identities, buffer, exposure, scheduler, bytes, cost
  python -m exps.rh01.run lock
  python -m exps.rh01.run cache                      # GPU feature extraction for the 12 new categories (ext4)
  python -m exps.rh01.run probe                      # GPU resource probe (throughput only)
  python -m exps.rh01.run train --arm R15|J15 --seed 2711
  python -m exps.rh01.run score --seeds 2711         # one-shot per seed batch
"""
from __future__ import annotations

import argparse
import shutil
import time

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly import encoder as E
from exps.dinomaly import official as O
from exps.dinomaly import readout as R
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.rh01 import common as C

OUT = C.OUT
SEEDS = DR.SEEDS
ITERS = DR.ITERS_PER_CLASS                     # 667
JOINT_ITERS = len(C.STREAM) * ITERS            # 10005
TASK = "agents/tasks/task_rh_01_replay_headroom_15task.md"
SOURCES = ["exps/rh01/common.py", "exps/rh01/run.py", "exps/rh01/analysis.py", "exps/rh01/test_rh01.py",
           "exps/dinomaly/official.py", "exps/dinomaly/encoder.py", "exps/dinomaly/common.py",
           "exps/dinomaly/readout.py", "exps/dinomaly/run.py", "exps/dinomaly/analysis.py",
           "exps/task_p/p2/metrics.py", "exps/task_p/p1_readout.py", TASK,
           "results/hope_cad/dinomaly/FROZEN_BASELINE.json"]
PROBE_LIMIT = 1.5                              # s/step; resource-only stop rule


def frozen():
    return D.read(C.DRF / "FROZEN_BASELINE.json")["results_files"]


def n_train(proto):
    return {c: len(proto["train"][c]) for c in C.STREAM}


# --- exposure dry-run (mirrors exps.dinomaly.run.train_stage generator consumption exactly) ------------


def simulate(n_current, iters, gen, n_replay=None):
    cur = DR.Batches(n_current, 8 if n_replay else 16, gen)
    rep = DR.Batches(n_replay, 8, gen) if n_replay else None
    cc, rc = np.zeros(n_current, np.int64), np.zeros(n_replay or 0, np.int64)
    for _ in range(iters):
        np.add.at(cc, cur.next(), 1)
        if rep is not None:
            np.add.at(rc, rep.next(), 1)
            torch.randperm(16, generator=gen)
    return cc, rc


def stage_seeds(arm, seed, stage):
    """(dropout seed, batch-generator seed) per arm and stage."""
    if arm == "R15":
        return seed * 100 + stage, seed * 100 + stage + 50
    return seed * 100 + 1, seed * 100 + 71


def prerun():
    """CPU pre-run checks; written before the lock and repeated (verified) after it."""
    proto = C.identities()
    n = n_train(proto)
    led = {"identities": {c: {"n": n[c], "png_bytes": sum(r["bytes"] for r in proto["train"][c])} for c in C.STREAM},
           "n_train": sum(n.values()), "seeds": {}}
    for seed in SEEDS:
        boundaries, totals = {}, {c: 0 for c in C.STREAM}
        totals["bottle"] += ITERS * 16                                   # stage 1 inherited (DRF-01 A stage 1)
        for s in range(2, len(C.STREAM) + 1):
            c, done = C.STREAM[s - 1], list(C.STREAM[:s - 1])
            idx = C.buffer_index(seed, n, done)
            rows = [proto["train"][d][i] for d in done for i in idx[d]]
            gen = torch.Generator().manual_seed(stage_seeds("R15", seed, s)[1])
            cc, rc = simulate(n[c], ITERS, gen, n_replay=len(rows))
            totals[c] += int(cc.sum())
            for j, r in enumerate(rows):
                totals[r["category"]] += int(rc[j])
            boundaries[s] = {"task": c, "allocation": {d: len(idx[d]) for d in done}, "slots": len(rows),
                             "identities": [r["relative_path"] for r in rows],
                             "png_bytes": sum(r["bytes"] for r in rows), "feature_bytes": len(rows) * 2 * 789 * 768 * 4,
                             "current_exposure_min_mean_max": [int(cc.min()), float(cc.mean()), int(cc.max())],
                             "replay_exposure_min_mean_max": [int(rc.min()), float(rc.mean()), int(rc.max())]}
        gen = torch.Generator().manual_seed(stage_seeds("J15", seed, 1)[1])
        cj, _ = simulate(sum(n.values()), JOINT_ITERS, gen)
        starts = np.cumsum([0] + [n[c] for c in C.STREAM])
        joint = {c: int(cj[starts[k]:starts[k + 1]].sum()) for k, c in enumerate(C.STREAM)}
        led["seeds"][seed] = {"boundaries": boundaries, "R15_sample_gradients_per_class": totals,
                              "J15_sample_gradients_per_class": joint,
                              "R15_total": sum(totals.values()), "J15_total": sum(joint.values())}
        require(all(b["slots"] == C.CAPACITY for b in boundaries.values()), "buffer slots unused")
        require(sum(totals.values()) == len(C.STREAM) * ITERS * 16 and sum(joint.values()) == JOINT_ITERS * 16,
                "sample-gradient accounting")
    sched = {}
    for iters in (ITERS, JOINT_ITERS):
        p = torch.nn.Parameter(torch.zeros(1))
        opt = O.StableAdamW([p], lr=2e-3)
        sch = O.make_scheduler(opt, iters)
        lrs = []
        for _ in range(iters):
            lrs.append(opt.param_groups[0]["lr"])
            sch.step()
        sched[iters] = {"lr_at": {i: lrs[i] for i in (0, 50, 99, 100, iters // 2, iters - 1)},
                        "ramp": round(0.1 * iters)}
    led["scheduler"] = sched
    dec_b = 61425408 * 4
    led["bytes"] = {"decoder_fp32": dec_b, "optimizer_transient": 3 * dec_b,
                    "checkpoint_disk_R15_per_seed": 14 * 245732723, "checkpoint_disk_J15_per_seed": 245732723,
                    "new_feature_cache": sum(n[c] for c in C.STREAM if c not in C.EVAL) * 2 * 789 * 768 * 4,
                    "copied_feature_cache": sum(n[c] for c in C.EVAL) * 2 * 789 * 768 * 4}
    rate = 0.62                                                          # DRF-01 B stage s/step incl. stats
    led["cost_projection_gpu_h"] = {"R15_per_seed": 14 * ITERS * rate / 3600, "J15_per_seed": JOINT_ITERS * 0.57 / 3600,
                                    "cache": sum(n[c] for c in C.STREAM if c not in C.EVAL) * 0.14 / 3600}
    led["cost_projection_gpu_h"]["total_3_seeds"] = 3 * (led["cost_projection_gpu_h"]["R15_per_seed"]
                                                         + led["cost_projection_gpu_h"]["J15_per_seed"]) \
        + led["cost_projection_gpu_h"]["cache"]
    return proto, led


# --- lock --------------------------------------------------------------------------------------------


def lock():
    target = OUT / "PREREG_LOCK.json"
    require(not target.exists(), "already locked")
    proto, led = prerun()
    D.write_json(OUT / "PREREG_PRERUN.json", led)
    D.write_json(OUT / "PROTOCOL.json", {"stream": C.STREAM, "eval": C.EVAL, "capacity": C.CAPACITY, "seeds": SEEDS,
                                         "iters_per_task": ITERS, "joint_iters": JOINT_ITERS, **proto})
    D.write_json(target, {"locked_utc": DR.now(), "sources": {s: sha(D.ROOT / s) for s in SOURCES},
                          "protocol_sha256": sha(OUT / "PROTOCOL.json"),
                          "prerun_sha256": sha(OUT / "PREREG_PRERUN.json"), "encoder_sha256": D.CKPT["base"][1]})
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


# --- cache and probe ---------------------------------------------------------------------------------


def cache():
    DR.configure()
    proto = verify_lock()
    target = OUT / "CACHE.json"
    require(not target.exists(), "cache already built")
    C.CACHE.mkdir(parents=True, exist_ok=True)
    rec, man = {}, frozen()
    for c in C.EVAL:                                      # sha-verified copies of the frozen DRF-01 caches
        src = C.DRF / f"features/train_{c}.pt"
        require(sha(src) == man[f"results/hope_cad/dinomaly/features/train_{c}.pt"]["sha256"], "DRF cache changed")
        dst = C.CACHE / f"train_{c}.pt"
        require(not dst.exists(), f"exists: {dst}")
        shutil.copyfile(src, dst)
        rec[c] = {"n": len(proto["train"][c]), "sha256": sha(dst), "source": "DRF-01 frozen copy"}
        require(rec[c]["sha256"] == sha(src), "copy mismatch")
    enc = E.load(D.CKPT["base"][0], "base", "cuda")
    for c in C.STREAM:
        if c in C.EVAL:
            continue
        t0 = time.perf_counter()
        f = DR.extract(proto["train"][c], enc)
        path = C.CACHE / f"train_{c}.pt"
        require(not path.exists(), f"exists: {path}")
        torch.save({"features": f}, path)
        rec[c] = {"n": len(f), "sha256": sha(path), "seconds": time.perf_counter() - t0}
        print("cache", c, rec[c], flush=True)
        del f
    D.write_json(target, rec)


def probe():
    """Throughput of the J15 (memory-mapped pool) and R15 (current + replay) steps; throwaway decoder."""
    DR.configure()
    proto = verify_lock()
    target = OUT / "PROBE.json"
    require(not target.exists(), "probe is one-shot")
    pool = C.Pool([C.load_features(c) for c in C.STREAM])
    res = {}
    for name, current, replay in (("J15", pool, None),
                                  ("R15", C.load_features("zipper"),
                                   torch.stack([C.load_features(c)[i] for c in ("bottle", "cable", "capsule")
                                                for i in range(8)]))):
        dec = DR.new_decoder(0)
        gen = torch.Generator().manual_seed(1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        DR.train_stage(dec, current, 10, gen, replay)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        DR.train_stage(dec, current, 40, gen, replay)
        torch.cuda.synchronize()
        res[name] = {"s_per_step": (time.perf_counter() - t1) / 40, "warm_s": t1 - t0,
                     "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
        del dec
        torch.cuda.empty_cache()
        print("probe", name, res[name], flush=True)
    res["pass"] = all(v["s_per_step"] <= PROBE_LIMIT for v in res.values())
    D.write_json(target, res)
    require(res["pass"], "resource probe above limit")


# --- training ----------------------------------------------------------------------------------------

MAX_SKIPS, MAX_CONSECUTIVE = 10, 3               # AMENDMENT_1


def train_stage_guarded(dec, current, iters, gen, replay=None):
    """exps.dinomaly.run.train_stage verbatim, plus AMENDMENT_1's skip-step guard: a non-finite loss or gradient norm
    discards the step (no optimizer update; the scheduler still advances). More than MAX_SKIPS skips in total, or
    MAX_CONSECUTIVE in a row, stops the run. With finite steps it is identical to the locked function."""
    opt = O.make_optimizer(dec)
    sch = O.make_scheduler(opt, iters)
    ramp = round(0.1 * iters)
    batch = O.RECIPE["batch_size"]
    cur = DR.Batches(len(current), batch // 2 if replay is not None else batch, gen)
    rep = DR.Batches(len(replay), batch // 2, gen) if replay is not None else None
    dec.train()
    losses, skips, run_ = [], [], 0
    for i in range(iters):
        p = min(O.RECIPE["hm_p_final"] * i / ramp, O.RECIPE["hm_p_final"])
        x = current[cur.next()]
        if rep is not None:
            x = torch.cat([x, replay[rep.next()]])
            x = x[torch.randperm(len(x), generator=gen)]
        x = x.cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)
        total = 0.
        for xb in x.split(DR.MICRO):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                en, de = dec(xb[:, 0], xb[:, 1])
            loss = O.global_cosine_hm_percent([e.float() for e in en], [d.float() for d in de],
                                              p=p, factor=O.RECIPE["hm_factor"]) * len(xb) / len(x)
            loss.backward()
            total += float(loss)
        norm = torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=O.RECIPE["clip_grad_norm"])
        if not (np.isfinite(total) and bool(torch.isfinite(norm))):
            opt.zero_grad(set_to_none=True)
            skips.append({"iteration": i, "loss": total, "grad_norm": float(norm)})
            run_ += 1
            require(len(skips) <= MAX_SKIPS and run_ < MAX_CONSECUTIVE, f"non-finite steps exceed limits: {skips}")
            sch.step()
            losses.append(float("nan"))
            continue
        run_ = 0
        opt.step()
        sch.step()
        losses.append(total)
    return losses, skips


def drf_stage1(seed):
    rec = D.read(C.DRF / f"runs/A_seed{seed}/TRAIN.json")
    st = rec["stages"][0]
    path = D.ROOT / st["checkpoint"]
    require(st["name"] == "bottle" and sha(path) == st["checkpoint_sha256"] == frozen()[st["checkpoint"]]["sha256"],
            "DRF-01 stage-1 checkpoint not frozen")
    return path, st


def train(arm, seed):
    DR.configure()
    proto = verify_lock()
    require(arm in ("R15", "J15") and seed in SEEDS, "bad arm/seed")
    run = OUT / f"runs/{arm}_seed{seed}"
    require(not (run / "TRAIN.json").exists(), "run already completed")
    run.mkdir(parents=True, exist_ok=True)
    held = {c: DR.load_split("heldout", c) for c in C.EVAL}
    n = n_train(proto)
    torch.cuda.reset_peak_memory_stats()
    if arm == "J15":
        require(not any(run.glob("*.pt")), "J15 restarts from scratch: remove the partial run first (recorded)")
        torch.manual_seed(stage_seeds(arm, seed, 1)[0])
        dec = DR.new_decoder(seed)
        gen = torch.Generator().manual_seed(stage_seeds(arm, seed, 1)[1])
        pool = C.Pool([C.load_features(c) for c in C.STREAM])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        losses, skips = train_stage_guarded(dec, pool, JOINT_ITERS, gen)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        path = run / "stage1_joint15.pt"
        digest = DR.save_ckpt(dec, path, {"arm": arm, "seed": seed, "stage": 1})
        stages = [{"index": 1, "name": "joint15", "iters": JOINT_ITERS, "seconds": seconds,
                   "loss_curve_every500": [float(np.nanmean(losses[i:i + 500])) for i in range(0, len(losses), 500)],
                   "loss_last50": float(np.nanmean(losses[-50:])), "nonfinite_skips": skips,
                   "heldout_normal": DR.normal_stats(dec, held),
                   "checkpoint": str(path.relative_to(D.ROOT)), "checkpoint_sha256": digest}]
    else:
        path1, st1 = drf_stage1(seed)
        dec = DR.new_decoder(seed)
        dec.load_state_dict(torch.load(path1, weights_only=True)["state"])
        stages = [{"index": 1, "name": "bottle", "inherited": st1["checkpoint"], "checkpoint": st1["checkpoint"],
                   "checkpoint_sha256": st1["checkpoint_sha256"], "seconds": st1["seconds"], "iters": ITERS}]
        for s in range(2, len(C.STREAM) + 1):
            c = C.STREAM[s - 1]
            ck, js = run / f"stage{s}_{c}.pt", run / f"STAGE{s}.json"
            if js.exists():                                            # resume: completed stage
                rec = D.read(js)
                require(sha(ck) == rec["checkpoint_sha256"], f"stage {s} checkpoint changed")
                dec.load_state_dict(torch.load(ck, weights_only=True)["state"])
                stages.append({**rec, "resumed": True})
                continue
            require(not ck.exists(), f"orphan checkpoint without record: {ck} (decide manually)")
            done = list(C.STREAM[:s - 1])
            idx = C.buffer_index(seed, n, done)
            rows = [proto["train"][d][i] for d in done for i in idx[d]]
            feats = {d: C.load_features(d) for d in done}
            replay = torch.stack([feats[d][i] for d in done for i in idx[d]])
            current = C.load_features(c)
            torch.manual_seed(stage_seeds(arm, seed, s)[0])
            gen = torch.Generator().manual_seed(stage_seeds(arm, seed, s)[1])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            losses, skips = train_stage_guarded(dec, current, ITERS, gen, replay)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            rec = {"index": s, "name": c, "iters": ITERS, "seconds": seconds, "train_images": len(current),
                   "loss_curve_every50": [float(np.nanmean(losses[i:i + 50])) for i in range(0, len(losses), 50)],
                   "loss_last50": float(np.nanmean(losses[-50:])), "nonfinite_skips": skips,
                   "heldout_normal": DR.normal_stats(dec, held),
                   "replay": {"allocation": {d: len(idx[d]) for d in done},
                              "identities": [r["relative_path"] for r in rows],
                              "png_bytes": sum(r["bytes"] for r in rows),
                              "feature_bytes": replay.numel() * replay.element_size()},
                   "checkpoint": str(ck.relative_to(D.ROOT)),
                   "checkpoint_sha256": DR.save_ckpt(dec, ck, {"arm": arm, "seed": seed, "stage": s, "name": c}),
                   "utc": DR.now()}
            D.write_json(js, rec)
            stages.append(rec)
            print(arm, seed, s, c, f"{seconds:.0f}s", "loss", round(rec["loss_last50"], 4),
                  {k: round(v["global_cosine"], 4) for k, v in rec["heldout_normal"].items()}, flush=True)
            del replay, current, feats
    record = {"arm": arm, "seed": seed, "stages": stages, "finished_utc": DR.now(),
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
              "persistent_decoder_bytes": DR.state_bytes(dec)}
    D.write_json(run / "TRAIN.json", record)
    if arm == "J15":
        print(arm, seed, f"{stages[0]['seconds']:.0f}s", {k: round(v["global_cosine"], 4)
                                                         for k, v in stages[0]["heldout_normal"].items()}, flush=True)


# --- scoring -----------------------------------------------------------------------------------------


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
    tag = "_".join(map(str, seeds))
    target = OUT / f"scores/SCORES_{tag}.json"
    require(not target.exists(), "scoring is one-shot per seed batch")
    key = "results/hope_cad/dinomaly/features/dev.pt"
    require(sha(D.ROOT / key) == frozen()[key]["sha256"], "dev features not frozen")
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
        print({k: v for k, v in led.items() if k in ("n_train", "cost_projection_gpu_h", "bytes")})
        for seed, v in led["seeds"].items():
            print(seed, "R15", v["R15_sample_gradients_per_class"], "\n     J15", v["J15_sample_gradients_per_class"])
            for s, b in v["boundaries"].items():
                print("   ", s, b["task"], b["allocation"], b["png_bytes"], b["replay_exposure_min_mean_max"],
                      b["current_exposure_min_mean_max"])
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
