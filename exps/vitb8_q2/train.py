"""VITB8-Q2 training: ViT-B/8 single-stream Dinomaly + Replay-24 + Fixed Q2 from stage 1, seed 2711, 15 stages.

  python -m exps.vitb8_q2.train smoke    # real-feature smoke (few steps, stages 1 and 2 setups); writes SMOKE.json
  python -m exps.vitb8_q2.train lock     # freeze sources / config / seed / metric contract (before any evaluation)
  python -m exps.vitb8_q2.train train    # full stream; resumes from the latest completed stage boundary

Reused unchanged: Replay-24 allocation (exps.rh01.common.buffer_index), R15 stage seeds (exps.rh01b.run.stage_seeds),
Batches / MICRO / configure (exps.dinomaly.run), StableAdamW per-block groups (exps.nlcad.fast.block_optimizer),
WarmCosine + clip 0.1 (exps.dinomaly.official), hard-mining cosine loss, FP32 linear-attention core (exps.rh01.numerics),
FS_B constants / host roles / norm-matched mixing (exps.nl_align_01.g1), ranking loss (exps.nlcad.objectives),
host sets and stage-lawful access (exps.nlcad.bank). The stage loop is exps.nl_align_01.g1.train_stage with the
two-group forward replaced by the single-stream forward and an optional replay (stage 1 has none).
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nl_align_01 import g1 as G
from exps.nlcad import bank as BK
from exps.nlcad.fast import MAX_CONSECUTIVE, MAX_SKIPS, _rng_devices, block_optimizer
from exps.nlcad.objectives import ranking_loss
from exps.rh01 import common as RC
from exps.rh01 import numerics as NUM
from exps.rh01b import run as RB
from exps.vitb8_q2 import common as V

OUT = V.OUT
RUN = OUT / "runs" / f"seed{V.SEED}"
LOCK, PROTOCOL = OUT / "LOCK.json", OUT / "PROTOCOL.json"
HELD = ("bottle", "carpet", "hazelnut")
K_SYN, RHO, N_CUR, N_REP, TAU, N_NEG, EPS = G.K_SYN, G.RHO, G.N_CUR, G.N_REP, G.TAU, G.N_NEG, G.EPS
SOURCES = ["exps/vitb8_q2/common.py", "exps/vitb8_q2/prepare.py", "exps/vitb8_q2/train.py",
           "exps/vitb8_q2/evaluate.py", "exps/vitb8_q2/test_vitb8_q2.py", V.TASK,
           "exps/dinomaly/official.py", "exps/dinomaly/run.py", "exps/dinomaly/readout.py", "exps/dinomaly/common.py",
           "exps/nl_align_01/g1.py", "exps/nlcad/bank.py", "exps/nlcad/fast.py", "exps/nlcad/objectives.py",
           "exps/rh01/common.py", "exps/rh01/numerics.py", "exps/rh01b/run.py", "exps/ptag/synth.py",
           "exps/cadfinal/metrics.py", "exps/cadfinal/common.py", "exps/calm02/native.py", "exps/calm02/extract.py",
           "models/feature_extractors/cadic_vit_v1.py", "dataset/anomaly_generators/perlin.py"]


def cap(default):
    g = float(os.environ.get("VITB8Q2_CAP_GIB", default))
    torch.cuda.set_per_process_memory_fraction(g * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    return g


def env(default_cap="3.5"):
    DR.configure()                    # TF32 off
    g = cap(default_cap)
    NUM.use(True)                     # D4' FP32 linear-attention core (as R15 / FS_B)
    return g


# --- auxiliary Q2 gradient (exps.nl_align_01.g1.Aux, FS_B path, single-stream forward) -----------------------


class Aux:
    def __init__(self, items, hosts, role, n_train, stage, seed=V.SEED):
        c = V.STREAM[stage - 1]
        keys = sorted(k for k in BK.allowed(hosts, seed, n_train, stage) if role.get(k) == "OPT")
        self.pools = {"current": [(a, i, v) for a, i in keys if a == c for v in range(2)],
                      "replay": [(a, i, v) for a, i in keys if a != c for v in range(2)]}
        require(self.pools["current"] and (stage == 1 or self.pools["replay"]), "empty OPT pool")
        require(stage > 1 or not self.pools["replay"], "replay hosts before any completed task")
        self.syn = {k: (items[k][0].clone(), items[k][1].clone()) for p in self.pools.values() for k in p}
        self.host_gen = torch.Generator().manual_seed(seed * 1000 + stage)
        self.neg_gen = torch.Generator().manual_seed(seed * 1000 + 500 + stage)

    def draw(self):
        out = []
        for src, k in (("current", N_CUR), ("replay", N_REP)):
            pool = self.pools[src]
            if pool:                                           # stage 1: no completed task -> current items only
                out += [pool[j] for j in torch.randperm(len(pool), generator=self.host_gen).tolist()[:k]]
        return out

    def backward(self, dec, keys, device="cuda"):
        x = torch.stack([self.syn[k][0] for k in keys]).to(device)
        cov = torch.stack([self.syn[k][1] for k in keys]).to(device)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            en, de = dec(x)
        loss = ranking_loss(V.grid(en, de), cov, TAU, N_NEG, gen=self.neg_gen)
        if loss is None:
            return None
        loss.backward()
        return float(loss)


def train_stage(dec, current, iters, gen, replay=None, aux=None, device="cuda", autocast=True):
    """exps.nl_align_01.g1.train_stage (the FS_B loop) with dec(x) on one feature map and optional replay."""
    opt = block_optimizer(dec)
    sch = O.make_scheduler(opt, iters)
    ramp = round(0.1 * iters)
    batch = O.RECIPE["batch_size"]
    cur = DR.Batches(len(current), batch // 2 if replay is not None else batch, gen)
    rep = DR.Batches(len(replay), batch // 2, gen) if replay is not None else None
    dec.train()
    params = list(dec.parameters())
    drops = [m for m in dec.modules() if isinstance(m, torch.nn.Dropout)]
    dtype = torch.device(device).type
    losses, skips, run_ = [], [], 0
    log = {k: [] for k in ("norm_pre_clip", "nN", "nA", "cos", "scale", "aux_loss")}
    for i in range(iters):
        p = min(O.RECIPE["hm_p_final"] * i / ramp, O.RECIPE["hm_p_final"])
        x = current[cur.next()]
        if rep is not None:
            x = torch.cat([x, replay[rep.next()]])
            x = x[torch.randperm(len(x), generator=gen)]
        x = x.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        total = 0.
        for xb in x.split(DR.MICRO):
            with torch.autocast(dtype, dtype=torch.bfloat16, enabled=autocast):
                en, de = dec(xb)
            loss = O.global_cosine_hm_percent([e.float() for e in en], [d.float() for d in de],
                                              p=p, factor=O.RECIPE["hm_factor"]) * len(xb) / len(x)
            loss.backward()
            total += float(loss)
        if aux is not None and i % K_SYN == 0:
            g_n = [None if q.grad is None else q.grad.detach().clone() for q in params]
            for q in params:
                q.grad = None
            saved = [m.p for m in drops]
            for m in drops:
                m.p = 0.0
            try:
                with torch.random.fork_rng(devices=_rng_devices(device)):
                    al = aux.backward(dec, aux.draw(), device)
            finally:
                for m, v in zip(drops, saved):
                    m.p = v
            if al is None:
                for q in params:
                    q.grad = None
            st = G.mix_(params, g_n, rho=RHO)
            del g_n
            for k in ("nN", "nA", "cos", "scale"):
                log[k].append(st[k])
            log["aux_loss"].append(float("nan") if al is None else al)
        norm = torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=O.RECIPE["clip_grad_norm"])
        log["norm_pre_clip"].append(float(norm))
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
    return losses, skips, log


# --- label-free diagnostics ------------------------------------------------------------------------------------


@torch.no_grad()
def normal_stats(dec, feats, batch=8):
    """Held-out / replay normal features: global cosine loss, mean and q99 of the 28x28 1-cos grid, image scores."""
    was = dec.training
    dec.eval()
    loss, grids, images = 0., [], []
    for xb in feats.split(batch):
        en, de = dec(xb.cuda())
        loss += float(O.global_cosine(en, de)) * len(xb)
        grids.append(V.grid(en, de).cpu())
        images += V.score(en, de)[1].tolist()
    g = torch.cat(grids)
    dec.train(was)
    return {"global_cosine": loss / len(feats), "mean_patch_1mcos": float(g.mean()),
            "q99_patch_1mcos": float(torch.quantile(g.flatten(), 0.99)), "image_score_mean": float(np.mean(images)),
            "image_score_max": float(np.max(images))}


@torch.no_grad()
def synthetic_diag(dec, items, role, stage):
    """Q2 synthetic patch AP on DIAG hosts (never used for gradients) of acquired categories; macro over categories."""
    from sklearn.metrics import average_precision_score
    was = dec.training
    dec.eval()
    out = {}
    for c in V.STREAM[:stage]:
        keys = sorted(k for k in items if k[0] == c and role.get((k[0], k[1])) == "DIAG")
        s, y = [], []
        for j in range(0, len(keys), 8):
            chunk = keys[j:j + 8]
            x = torch.stack([items[k][0] for k in chunk]).cuda()
            cov = torch.stack([items[k][1] for k in chunk])
            g = V.grid(*dec(x)).cpu()
            keep = (cov >= 0.5) | (cov == 0)
            s.append(g[keep]); y.append((cov >= 0.5)[keep])
        s, y = torch.cat(s).numpy(), torch.cat(y).numpy()
        out[c] = float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")
    dec.train(was)
    return {"per_category": out, "macro": float(np.nanmean(list(out.values())))}


def displacement(state, ref):
    return math.sqrt(sum(float((v.double() - ref[k].double()).pow(2).sum()) for k, v in state.items()
                         if v.dtype.is_floating_point))


# --- lock --------------------------------------------------------------------------------------------------------


def protocol():
    return {
        "task": V.TASK, "seed": V.SEED, "stream": list(V.STREAM), "iters_per_stage": V.ITERS,
        "backbone": {"checkpoint": "checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth",
                     "checkpoint_sha256": "755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415",
                     "input": "224x224 PIL bilinear, ImageNet mean/std", "layer": "9 (one-based), CLS removed",
                     "feature": [V.TOKENS, V.DIM], "frozen": True, "precision": "FP32 cached features"},
        "decoder": {"class": "exps.vitb8_q2.common.SingleStreamDecoder", "bottleneck": "bMlp 768-3072-768 dropout 0.2",
                    "blocks": "8 x LinearAttention2 Block, 12 heads, mlp 4, LayerNorm eps 1e-8, qkv_bias",
                    "prefix_tokens": 0, "readout": "mean of the 8 decoder block outputs (one group)",
                    "target": "frozen layer-9 map (stop-gradient)", "init": "official init_trainable after manual_seed(2711)",
                    "numerics": "bf16 autocast training forward, FP32 linear-attention core (D4'), FP32 evaluation"},
        "objective": {"normal": "global_cosine_hm_percent p ramp 0->0.9 over 10% of stage, factor 0.1",
                      "aux": "Fixed Q2 from stage 1: g = g_N + rho*||g_N||/(||g_Q2||+eps)*g_Q2 before clip; "
                             f"rho={RHO}, eps={EPS}, K_SYN={K_SYN}, {N_CUR} current + {N_REP} replay OPT items "
                             f"(stage 1: {N_CUR} current only), tau={TAU}, {N_NEG} negatives, dropout off in aux pass",
                      "generator": "exps.ptag.synth.q2 NativePerlin (DRAEM Perlin + DTD), 2 variants per lawful host"},
        "optimizer": "StableAdamW per-block groups (lr 2e-3 -> 2e-4 WarmCosine, warmup 100, wd 1e-4, amsgrad, "
                     "eps 1e-10), fresh optimizer per stage, clip_grad_norm 0.1, batch 16 (8 current + 8 replay; "
                     "stage 1: 16 current), micro-batch 8",
        "replay": "Replay-24 total, exps.rh01.common.buffer_index(seed 2711): per-class permutations seed+9000, "
                  "remainder priority seed+9001; replay features = cached ViT-B/8 features of the same identities",
        "stage_seeds": "exps.rh01b.run.stage_seeds('R15', 2711, s)",
        "scorer": {"map": "1-cos 28x28 -> 224 bilinear align_corners=True -> 256 bilinear -> Gaussian 5x5 sigma 4",
                   "image": "mean of top 1% of the 256x256 map", "native": "exps.dinomaly.readout.native "
                   "(bilinear, align_corners=False, antialias=False); masks never resized",
                   "pixel_metric": "exact native Pixel AP per category (exps.cadfinal.metrics.exact_ap)",
                   "image_metric": "Image AUROC per category (sklearn)", "macro": "category-balanced mean over acquired",
                   "forgetting": "AP(category at its acquisition stage) - AP(category at stage 15), macro over 14"},
        "evaluation_cohort": "full MVTec AD test set (1,725 images), every acquired category after every stage",
        "historical_reference": {"FS_B_pixel_ap": 0.5791, "FS_B_image_auroc": 0.9492,
                                 "note": "DINOv2-reg ViT-B/14 two-group representation, branched from R15 at stage 6, "
                                         "3-seed aggregate; descriptive only"},
        "one_seed": "not a statistically validated improvement",
    }


def lock():
    require(not LOCK.exists(), "already locked")
    require(not any(OUT.glob("EVAL_STAGE*.json")), "evaluation exists before the lock")
    D.write_json(PROTOCOL, protocol())
    D.write_json(LOCK, {"locked_utc": DR.now(), "sources": {s: sha(V.ROOT / s) for s in SOURCES},
                        "protocol_sha256": sha(PROTOCOL), "cache_sha256": sha(OUT / "CACHE.json"),
                        "bank_sha256": sha(OUT / "BANK.json"), "smoke_sha256": sha(OUT / "SMOKE.json"),
                        "torch": torch.__version__})
    print("locked", sha(LOCK), flush=True)


def verify_lock():
    lk = D.read(LOCK)
    bad = [s for s, d in lk["sources"].items() if sha(V.ROOT / s) != d]
    require(not bad, f"post-lock change: {bad}")
    require(sha(PROTOCOL) == lk["protocol_sha256"] and sha(OUT / "CACHE.json") == lk["cache_sha256"]
            and sha(OUT / "BANK.json") == lk["bank_sha256"], "protocol/cache/bank changed after lock")
    return lk


# --- stage -------------------------------------------------------------------------------------------------------


def stage_data(ids, s):
    c = V.STREAM[s - 1]
    done = list(V.STREAM[:s - 1])
    replay, replay_keys, idx = None, [], {}
    if done:
        idx = RC.buffer_index(V.SEED, ids["n_train"], done)
        replay_keys = [(d, i) for d in done for i in idx[d]]
        replay = torch.stack([V.train_features(d)[i] for d, i in replay_keys])
        require(len(replay_keys) == RC.CAPACITY, "Replay-24 capacity not exactly used")
    return c, V.train_features(c), replay, replay_keys, idx


def save_ckpt(dec, path, meta):
    require(not path.exists(), f"exists: {path}")
    part = path.with_suffix(".part")
    torch.save({"state": {k: v.detach().cpu() for k, v in dec.state_dict().items()}, "meta": meta}, part)
    part.replace(path)
    return sha(path)


def run_stage(dec, s, ids, items, hosts, role, held):
    c, current, replay, replay_keys, idx = stage_data(ids, s)
    ck, js = RUN / f"stage{s:02d}_{c}.pt", RUN / f"STAGE{s:02d}.json"
    dseed, gseed = RB.stage_seeds("R15", V.SEED, s)
    torch.manual_seed(dseed)
    gen = torch.Generator().manual_seed(gseed)
    theta0 = {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()}
    aux = Aux(items, hosts, role, ids["n_train"], s)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    losses, skips, log = train_stage(dec, current, V.ITERS, gen, replay, aux=aux)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    state = {k: v.detach().cpu() for k, v in dec.state_dict().items()}
    require(all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()), "non-finite parameters")
    a = {k: np.asarray(v, dtype=np.float64) for k, v in log.items()}
    al = a["aux_loss"]
    rec = {"index": s, "name": c, "iters": V.ITERS, "seconds": seconds, "train_images": len(current),
           "stage_seeds": [dseed, gseed], "nonfinite_skips": skips,
           "loss_curve_every50": [float(np.nanmean(losses[j:j + 50])) for j in range(0, len(losses), 50)],
           "loss_last50": float(np.nanmean(losses[-50:])),
           "clip_saturation": float(np.mean(a["norm_pre_clip"] > O.RECIPE["clip_grad_norm"])),
           "aux": {"cos_mean": float(np.nanmean(a["cos"])), "nN_mean": float(np.nanmean(a["nN"])),
                   "nA_median": float(np.nanmedian(a["nA"])), "scale_median": float(np.nanmedian(a["scale"])),
                   "aux_loss_mean": float(np.nanmean(al)), "aux_loss_first50": float(np.nanmean(al[:50])),
                   "aux_loss_last50": float(np.nanmean(al[-50:])), "aux_none_frac": float(np.mean(np.isnan(al))),
                   "sat_frac_1e-3": float(np.mean(al < 1e-3)), "pools": {k: len(v) for k, v in aux.pools.items()}},
           "heldout_normal": {h: normal_stats(dec, held[h]) for h in held if h in V.STREAM[:s]},
           "current_normal_first64": normal_stats(dec, current[:64]),
           "replay_normal": None if replay is None else normal_stats(dec, replay),
           "synthetic_q2_diag_ap": synthetic_diag(dec, items, role, s),
           "replay": None if replay is None else {
               "allocation": {d: len(idx[d]) for d in idx}, "slots": len(replay_keys),
               "identities": [ids["train"][d][i]["relative_path"] for d, i in replay_keys],
               "feature_bytes": replay.numel() * replay.element_size()},
           "displacement_from_stage_start": displacement(state, theta0),
           "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
           "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    torch.save(log, RUN / f"LOG{s:02d}.pt")
    rec["checkpoint"] = str(ck.relative_to(V.ROOT))
    rec["checkpoint_sha256"] = save_ckpt(dec, ck, {"seed": V.SEED, "stage": s, "name": c})
    rec["utc"] = DR.now()
    D.write_json(js, rec)
    print("stage", s, c, f"{seconds:.0f}s", "skips", len(skips), "loss", round(rec["loss_last50"], 4),
          "aux", round(rec["aux"]["aux_loss_last50"], 4), "synAP", round(rec["synthetic_q2_diag_ap"]["macro"], 4),
          {h: round(v["global_cosine"], 4) for h, v in rec["heldout_normal"].items()}, flush=True)
    return rec


def train():
    env()
    verify_lock()
    ids = V.identities()
    items, hosts = V.load_bank()
    role = G.roles(hosts)
    held = {h: V.heldout_features(h).clone() for h in HELD}
    RUN.mkdir(parents=True, exist_ok=True)
    require(not (RUN / "TRAIN.json").exists(), "run already completed")
    dec, last = V.new_decoder(V.SEED), 0
    for s in range(1, len(V.STREAM) + 1):                      # resume from the latest completed boundary
        js = RUN / f"STAGE{s:02d}.json"
        if not js.exists():
            break
        rec = D.read(js)
        require(sha(V.ROOT / rec["checkpoint"]) == rec["checkpoint_sha256"], f"stage {s} checkpoint changed")
        last = s
    if last:
        rec = D.read(RUN / f"STAGE{last:02d}.json")
        dec.load_state_dict(torch.load(V.ROOT / rec["checkpoint"], weights_only=True)["state"])
        print("resume after stage", last, flush=True)
    for s in range(last + 1, len(V.STREAM) + 1):
        c = V.STREAM[s - 1]
        for orphan in (RUN / f"stage{s:02d}_{c}.pt", RUN / f"stage{s:02d}_{c}.part"):
            if orphan.exists():                                 # checkpoint without a stage record: incomplete stage
                print("removing incomplete-stage artifact", orphan, flush=True)
                orphan.unlink()
        run_stage(dec, s, ids, items, hosts, role, held)
    recs = [D.read(RUN / f"STAGE{s:02d}.json") for s in range(1, len(V.STREAM) + 1)]
    D.write_json(RUN / "TRAIN.json", {
        "seed": V.SEED, "finished_utc": DR.now(), "train_seconds_total": sum(r["seconds"] for r in recs),
        "peak_allocated_mib_max": max(r["peak_allocated_mib"] for r in recs),
        "persistent_decoder_bytes": V.state_bytes(dec), "nonfinite_skips_total": sum(len(r["nonfinite_skips"]) for r in recs),
        "stages": [{"index": r["index"], "name": r["name"], "checkpoint": r["checkpoint"],
                    "checkpoint_sha256": r["checkpoint_sha256"], "seconds": r["seconds"]} for r in recs]})
    print("train complete", flush=True)


# --- smoke -------------------------------------------------------------------------------------------------------


def smoke(iters=6):
    """Real-feature smoke before the lock: stage-1 and stage-2 setups, a few steps each, on a throwaway decoder."""
    target = OUT / "SMOKE.json"
    require(not target.exists(), "smoke already recorded")
    env()
    ids = V.identities()
    items, hosts = V.load_bank()
    role = G.roles(hosts)
    out = {}
    dec = V.new_decoder(V.SEED)
    for s in (1, 2):
        c, current, replay, replay_keys, idx = stage_data(ids, s)
        aux = Aux(items, hosts, role, ids["n_train"], s)
        before = normal_stats(dec, current[:16])
        gen = torch.Generator().manual_seed(1)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        losses, skips, log = train_stage(dec, current, iters, gen, replay, aux=aux)
        torch.cuda.synchronize()
        spi = (time.perf_counter() - t0) / iters
        after = normal_stats(dec, current[:16])
        diag = synthetic_diag(dec, items, role, s)
        finite = all(bool(torch.isfinite(p).all()) for p in dec.parameters())
        out[f"stage{s}"] = {"category": c, "losses": losses, "skips": skips, "aux_loss": log["aux_loss"], "nA": log["nA"],
                  "nN": log["nN"], "scale": log["scale"], "norm_pre_clip": log["norm_pre_clip"],
                  "seconds_per_iter": spi, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "normal_before": before, "normal_after": after, "synthetic_diag": diag,
                  "pools": {k: len(v) for k, v in aux.pools.items()}, "replay_slots": len(replay_keys),
                  "params_finite": finite}
        require(finite and not skips and all(np.isfinite(losses)), f"smoke stage {s}: non-finite")
        require(all(a > 0 and np.isfinite(a) for a in log["nA"]) and all(n > 0 for n in log["nN"]),
                f"smoke stage {s}: zero / non-finite gradient")
        print("smoke", s, c, f"{spi:.3f}s/it", "loss", [round(x, 4) for x in losses], "aux",
              [round(x, 4) for x in log["aux_loss"]], "synAP", round(diag["macro"], 4), flush=True)
    est = (out["stage1"]["seconds_per_iter"] + 14 * out["stage2"]["seconds_per_iter"]) * V.ITERS / 3600
    out["projected_train_gpu_h"] = est
    D.write_json(target, out)
    print("projected training GPU-h", round(est, 2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("smoke", "lock", "train"))
    a = ap.parse_args()
    {"smoke": smoke, "lock": lock, "train": train}[a.stage]()


if __name__ == "__main__":
    main()
