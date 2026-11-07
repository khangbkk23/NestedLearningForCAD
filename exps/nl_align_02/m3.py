# exps/nl_align_02/m3.py
"""NL-ALIGN-02-MVTec M3: continue the G1 FS_B (fixed Q2) and CN arms through stages 10-15 (owner-authorized after
M2 = Q2_MVTEC_EARLY_INCONCLUSIVE; exploratory, selection-influenced).

Reuses, unchanged, the G1 stage loop (exps.nl_align_01.g1.train_stage / Aux / mix_ / roles) and bank builder
semantics. Writes only under results/hope_cad/nl_align_02_mvtec/m3/ (G1 directories are never touched).

  bank   --seed S  Q2/Q3 bank for categories 10-15 (same hosts rule with stage_max 15; same generators/variants)
  lock             M3_LOCK.json (sources, start checkpoints, bank hashes, owner authorization) before any training
  rerun            runner-equivalence check: FS_B seed 2711 stage 9 from the sealed G1 stage-8 state must be bitwise
                   equal to the sealed G1 stage-9 checkpoint (scratch output, then deleted)
  train  --arm A --seed S   stages 10-15 from the sealed G1 stage-9 state
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
from PIL import Image

from exps.dinomaly import common as D
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nl_align_01 import build_bank as BB
from exps.nl_align_01 import g1 as G
from exps.nlcad import bank as BK
from exps.rh01 import numerics as NUM
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

ROOT = D.ROOT
OUT = ROOT / "results/hope_cad/nl_align_02_mvtec/m3"
LOCK = OUT / "M3_LOCK.json"
FIRST, LAST = 10, 15
ARMS = ("FS_B", "CN")
SEEDS = (2711, 2712, 2713)
SRC = ["exps/nl_align_02/m3.py", "exps/nl_align_01/g1.py", "exps/nl_align_01/build_bank.py", "exps/nlcad/bank.py",
       "exps/nlcad/fast.py", "exps/nlcad/objectives.py", "exps/nlcad/run.py", "exps/rh01/run.py", "exps/rh01/common.py",
       "exps/rh01b/run.py", "exps/rh01b/common.py", "exps/dinomaly/official.py", "exps/dinomaly/run.py",
       "exps/ptag/synth.py"]


# ---- bank extension ----------------------------------------------------------------------------------------

def host_sets_15(seed, n_train):
    """build_bank.host_sets with stage_max 15, restricted to categories 10-15 (categories 1-9: quotas only shrink
    after stage 9 because buffers are prefixes of a fixed permutation, so the G1 bank already holds them)."""
    perms, _ = __import__("exps.rh01.common", fromlist=["buffer_plan"]).buffer_plan(seed, n_train)
    quota = {c: 0 for c in C.STREAM}
    for k in range(1, LAST + 1):
        for c, idx in C.buffer_index(seed, n_train, list(C.STREAM[:k])).items():
            quota[c] = max(quota[c], len(idx))
    out = {}
    for ci, c in enumerate(C.STREAM):
        if ci < G.LAST:
            continue
        fixed = torch.randperm(n_train[c], generator=torch.Generator().manual_seed(31337 + ci)).tolist()
        out[c] = {"current": fixed[:BK.CURRENT_HOSTS], "buffer": perms[c][:quota[c]]}
    return out


def bank(seed):
    from exps.ptag.synth import NativePerlin, q2, q3
    from exps.dinomaly import encoder as E
    DR.configure()
    proto = RB.verify_lock()
    n_train = R1.n_train(proto)
    # the G1 bank must already cover every later buffer of categories 1-9
    _, h9 = G.load_bank(seed, verify=False)
    for k in range(G.LAST + 1, LAST + 1):
        for c, idx in C.buffer_index(seed, n_train, list(C.STREAM[:k])).items():
            if c in h9:
                require(set(idx) <= set(h9[c]["buffer"]) | set(h9[c]["current"]), f"G1 bank misses {c} at stage {k}")
    hosts = host_sets_15(seed, n_train)
    out_dir = OUT / f"bank_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = E.load(D.CKPT["base"][0], "base", "cuda")
    gen = NativePerlin()
    man = {"seed": seed, "categories": list(hosts), "generators": list(BB.GENERATORS), "variants": BB.VARIANTS,
           "hosts": hosts, "files": {}, "encoder_sha256": sha(D.CKPT["base"][0])}
    for c in hosts:
        ci = C.STREAM.index(c)
        idx = sorted(set(hosts[c]["current"]) | set(hosts[c]["buffer"]))
        for g in BB.GENERATORS:
            target = out_dir / f"{g}_{c}.pt"
            require(not target.exists(), f"exists: {target}")
            feats, covs, keys = [], [], []
            for i in idx:
                row = proto["train"][c][i]
                with Image.open(D.source(row)) as im:
                    img = np.array(im.convert("RGB"), dtype=np.uint8)
                for v in range(BB.VARIANTS):
                    o, mask, _ = q2(gen, img, c, ci, i, v) if g == "Q2" else q3(img, ci, i, v)
                    x = BK.preprocess_array(o)[None].cuda()
                    with torch.no_grad():
                        feats.append(enc.groups(x).float().cpu()[0])
                    covs.append(BK.patch_coverage(mask))
                    keys.append((c, i, v))
            torch.save({"features": torch.stack(feats), "coverage": torch.stack(covs), "keys": keys}, target)
            man["files"][f"{g}_{c}"] = {"n": len(keys), "sha256": sha(target)}
            print("bank", seed, g, c, len(keys), flush=True)
    D.write_json(OUT / f"BANK_EXT_seed{seed}.json", man)


def load_bank_15(seed):
    items, hosts = G.load_bank(seed)
    man = D.read(OUT / f"BANK_EXT_seed{seed}.json")
    for g in G.GENS:
        for c in man["categories"]:
            path = OUT / f"bank_seed{seed}" / f"{g}_{c}.pt"
            require(sha(path) == man["files"][f"{g}_{c}"]["sha256"], f"bank file changed: {path}")
            d = torch.load(path, weights_only=False, mmap=True)
            for j, (cat, i, v) in enumerate(d["keys"]):
                items[g][(cat, int(i), int(v))] = (d["features"][j], d["coverage"][j])
    require(set(items["Q2"]) == set(items["Q3"]), "Q2/Q3 bank keys differ")
    for c, h in man["hosts"].items():
        hosts[c] = {k: [int(x) for x in v] for k, v in h.items()}
    return items, hosts


# ---- lock --------------------------------------------------------------------------------------------------

def lock():
    require(not LOCK.exists(), "exists")
    m2 = D.read(ROOT / "results/hope_cad/nl_align_02_mvtec/M2_DECISION.json")
    starts = {f"{a}|{s}": G.checkpoint(a, s, G.LAST)[1] for a in ARMS for s in SEEDS}
    banks = {s: sha(OUT / f"BANK_EXT_seed{s}.json") for s in SEEDS}
    D.write_json(LOCK, {
        "authorization": "owner, 2026-11-05, after M2 = " + m2["label"] + " (answer: 'Chạy M3'); exploratory, "
                         "selection-influenced; not a pass of the M2 rule",
        "m2_decision_sha": sha(ROOT / "results/hope_cad/nl_align_02_mvtec/M2_DECISION.json"),
        "formula": "unchanged G1: K_SYN=1, rho=0.25 norm-matched mixing before the 0.1 clip, StableAdamW, tau=0.05, "
                   "64 negatives, 4 current + 4 replay OPT items; FS_B trains Q2, CN uses the normal loss on the same "
                   "hosts' clean images; per-stage fresh optimizer/scheduler and R15 stage seeds",
        "stages": [FIRST, LAST], "arms": ARMS, "seeds": SEEDS, "start_checkpoints_stage9": starts,
        "bank_ext": banks, "sources": {p: sha(ROOT / p) for p in SRC}, "budget_gpu_h": 12.0,
        "stop_rules": "non-finite parameters or > G1 MAX_SKIPS skips -> stop; no recipe change after any test read",
        "utc": DR.now()})


def verify_lock():
    lk = D.read(LOCK)
    bad = [p for p, d in lk["sources"].items() if sha(ROOT / p) != d]
    require(not bad, f"source changed after M3 lock: {bad}")
    return lk


# ---- training ----------------------------------------------------------------------------------------------

def run_dir(arm, seed):
    return OUT / "runs" / f"{arm}_seed{seed}"


@torch.no_grad()
def replay_normal_stats(dec, replay, keys):
    """Lawful label-free diagnostic: per completed category, mean and q99 of the 28x28 1-cos grid on its buffer."""
    from exps.dinomaly import readout as R
    dec.eval()
    out = {}
    cats = sorted({c for c, _ in keys}, key=C.STREAM.index)
    for c in cats:
        rows = [j for j, (cc, _) in enumerate(keys) if cc == c]
        s = []
        for j in range(0, len(rows), 8):
            x = replay[rows[j:j + 8]].cuda()
            en, de = dec(x[:, 0], x[:, 1])
            s.append(R.grid(en, de).float().cpu())
        s = torch.cat(s)
        out[c] = {"mean": float(s.mean()), "q99": float(torch.quantile(s.flatten(), 0.99)), "n": len(rows)}
    dec.train()
    return out


def _stage(arm, seed, s, dec, items, hosts, role, n, held, run):
    c = C.STREAM[s - 1]
    ck, js = run / f"stage{s}_{c}.pt", run / f"STAGE{s}.json"
    require(not ck.exists() and not js.exists(), f"exists: {ck}")
    done = list(C.STREAM[:s - 1])
    idx = C.buffer_index(seed, n, done)
    replay_keys = [(d, i) for d in done for i in idx[d]]
    replay = torch.stack([C.load_features(d)[i] for d, i in replay_keys])
    current = C.load_features(c)
    dseed, gseed = RB.stage_seeds("R15", seed, s)
    torch.manual_seed(dseed)
    gen = torch.Generator().manual_seed(gseed)
    theta0 = {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()}
    aux = G.Aux(arm, items, hosts, role, seed, n, s, current, replay, replay_keys)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    losses, skips, log = G.train_stage(dec, current, R1.ITERS, gen, replay, aux=aux)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    state = {k: v.detach().cpu() for k, v in dec.state_dict().items()}
    require(all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()), "non-finite parameters")
    torch.save({"state": state, "arm": arm, "seed": seed, "stage": s}, ck)
    torch.save(log, run / f"LOG{s}.pt")
    a = {k: np.asarray(v, dtype=np.float64) for k, v in log.items()}
    al = a["aux_loss"]
    summ = {k: float(np.nanmean(a[k])) for k in ("cos", "nN", "nA", "aux_loss")}
    summ.update({"scale_median": float(np.nanmedian(a["scale"])), "sat_frac_1e-3": float(np.mean(al < 1e-3)),
                 "sat_frac_1e-5": float(np.mean(al < 1e-5)), "nA_median": float(np.nanmedian(a["nA"]))})
    rec = {"arm": arm, "seed": seed, "index": s, "name": c, "seconds": seconds, "nonfinite_skips": skips,
           "loss_curve_every50": [float(np.nanmean(losses[j:j + 50])) for j in range(0, len(losses), 50)],
           "clip_saturation": float(np.mean(a["norm_pre_clip"] > 0.1)), "aux": summ,
           "heldout_normal": DR.normal_stats(dec, held),
           "replay_normal": replay_normal_stats(dec, replay, replay_keys),
           "displacement_from_stage_start": G.displacement(state, theta0),
           "checkpoint": str(ck.relative_to(ROOT)), "checkpoint_sha256": sha(ck),
           "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "utc": DR.now()}
    D.write_json(js, rec)
    print(arm, seed, s, c, f"{seconds:.0f}s", "skips", len(skips), "sat", round(summ["sat_frac_1e-3"], 3),
          {k: round(v["global_cosine"], 4) for k, v in rec["heldout_normal"].items()}, flush=True)
    del replay, current, aux
    return rec


def _env():
    from exps.nlcad.run import cap
    DR.configure()
    cap()
    proto = RB.verify_lock()
    NUM.use(True)
    return R1.n_train(proto)


def train(arm, seed):
    verify_lock()
    require(arm in ARMS and seed in SEEDS, "arm/seed")
    require(D.read(OUT / "RERUN_CHECK.json")["pass"], "runner equivalence first")
    n = _env()
    run = run_dir(arm, seed)
    run.mkdir(parents=True, exist_ok=True)
    items, hosts = load_bank_15(seed)
    role = G.roles(hosts)
    held = {c: DR.load_split("heldout", c) for c in C.EVAL}
    dec = DR.new_decoder(seed)
    path, digest = G.checkpoint(arm, seed, G.LAST)
    require(digest == D.read(LOCK)["start_checkpoints_stage9"][f"{arm}|{seed}"], "start checkpoint")
    dec.load_state_dict(G.load_state(path))
    torch.cuda.reset_peak_memory_stats()
    for s in range(FIRST, LAST + 1):
        js = run / f"STAGE{s}.json"
        if js.exists():
            rec = D.read(js)
            require(sha(ROOT / rec["checkpoint"]) == rec["checkpoint_sha256"], f"stage {s} changed")
            dec.load_state_dict(G.load_state(ROOT / rec["checkpoint"]))
            continue
        _stage(arm, seed, s, dec, items, hosts, role, n, held, run)


def rerun():
    """Runner equivalence: FS_B seed 2711 stage 9 from the sealed G1 stage-8 state through this module's bank loader
    and stage function must reproduce the sealed G1 stage-9 checkpoint bitwise."""
    verify_lock()
    target = OUT / "RERUN_CHECK.json"
    require(not target.exists(), "exists")
    n = _env()
    items, hosts = load_bank_15(2711)
    role = G.roles(hosts)
    held = {c: DR.load_split("heldout", c) for c in C.EVAL}
    dec = DR.new_decoder(2711)
    dec.load_state_dict(G.load_state(G.checkpoint("FS_B", 2711, 8)[0]))
    scratch = OUT / "rerun_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    rec = _stage("FS_B", 2711, 9, dec, items, hosts, role, n, held, scratch)
    mine = G.load_state(ROOT / rec["checkpoint"])
    ref = G.load_state(G.checkpoint("FS_B", 2711, 9)[0])
    diff = max(float((mine[k].double() - ref[k].double()).abs().max()) for k in ref)
    out = {"max_abs_diff": diff, "bitwise": all(torch.equal(mine[k], ref[k]) for k in ref), "pass": diff == 0.0,
           "seconds": rec["seconds"], "reference": G.checkpoint("FS_B", 2711, 9)[1], "utc": DR.now()}
    D.write_json(target, out)
    for p in scratch.glob("*"):
        p.unlink()
    scratch.rmdir()
    print(out)
    require(out["pass"], "M3 runner does not reproduce G1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bank", "lock", "rerun", "train"])
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    a = ap.parse_args()
    if a.cmd == "bank":
        bank(a.seed)
    elif a.cmd == "train":
        train(a.arm, a.seed)
    else:
        {"lock": lock, "rerun": rerun}[a.cmd]()
