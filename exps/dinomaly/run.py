# exps/dinomaly/run.py
"""DRF-01 stages: lock, cache, train, evaluate (gate 0 / gate 1), latency.

  python -m exps.dinomaly.run lock
  python -m exps.dinomaly.run cache
  python -m exps.dinomaly.run train --arm C|A|B --seed 2711
  python -m exps.dinomaly.run evaluate --gate 0|1
"""
from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone

import numpy as np
import torch

from exps.dinomaly import common as C
from exps.dinomaly import encoder as E
from exps.dinomaly import official as O
from exps.dinomaly import readout as R
from exps.dinomaly.common import require

OUT = C.OUT
SEEDS = (2711, 2712, 2713)
ITERS_PER_CLASS = 667                       # 10000 iterations x 16 / 15 classes / 16 (D5)
JOINT_ITERS = 3 * ITERS_PER_CLASS
MICRO = 8
REPLAY_CAPACITY = 24
CAP_GIB = 2.6
TASK = "agents/tasks/task_drf_01_dinomaly_reconstruction_feasibility.md"
SOURCES = ["exps/dinomaly/official.py", "exps/dinomaly/encoder.py", "exps/dinomaly/common.py",
           "exps/dinomaly/readout.py", "exps/dinomaly/run.py", "exps/dinomaly/analysis.py",
           "exps/task_p/p2/metrics.py", "exps/task_p/p1_readout.py", TASK]


def now():
    return datetime.now(timezone.utc).isoformat()


def configure():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)


# --- lock --------------------------------------------------------------------------------------------


def protocol():
    ident = C.identities()
    return {"task": TASK, "seeds": list(SEEDS), "iters_per_class": ITERS_PER_CLASS, "joint_iters": JOINT_ITERS,
            "micro_batch": MICRO, "replay_capacity": REPLAY_CAPACITY, "cap_gib": CAP_GIB, "order": list(C.ORDER),
            "recipe": O.RECIPE, "dinomaly_commit": O.COMMIT, "encoder": {"arch": "base", "sha256": C.CKPT["base"][1],
            "size": E.SIZE, "groups": E.GROUPS}, "train": ident["train"], "heldout": ident["heldout"],
            "evaluation": ident["evaluation"], "sealed_identities_excluded": ident["sealed_count"],
            "p1_protocol_sha256": C.P1_SHA}


def lock():
    require(not (OUT / "PREREG_LOCK.json").exists(), "already locked")
    proto = protocol()
    C.write_json(OUT / "PROTOCOL.json", proto)
    C.write_json(OUT / "PREREG_LOCK.json", {
        "locked_utc": now(), "protocol_sha256": C.sha(OUT / "PROTOCOL.json"),
        "sources": {s: C.sha(C.ROOT / s) for s in SOURCES}, "encoder_sha256": C.sha(C.CKPT["base"][0]),
        "preflight": {f: C.sha(OUT / f) for f in ("PREFLIGHT.json", "PREFLIGHT_2.json")}})
    print("locked", flush=True)


def verify_lock():
    lk = C.read(OUT / "PREREG_LOCK.json")
    digests = dict(lk["sources"])
    for rec in sorted(OUT.glob("AMENDMENT_*.json")):
        for name, a in C.read(rec)["files"].items():
            require(digests[name] == a["locked_sha256"], f"amendment chain broken: {name}")
            digests[name] = a["amended_sha256"]
    for name, digest in digests.items():
        require(C.sha(C.ROOT / name) == digest, f"post-lock change: {name}")
    require(C.sha(OUT / "PROTOCOL.json") == lk["protocol_sha256"], "protocol changed")
    return C.read(OUT / "PROTOCOL.json")


# --- features ----------------------------------------------------------------------------------------


@torch.no_grad()
def extract(rows, enc, batch=8):
    out = []
    for i in range(0, len(rows), batch):
        x = torch.stack([E.preprocess(C.source(r)) for r in rows[i:i + batch]]).cuda()
        out.append(enc.groups(x).float().cpu())
    f = torch.cat(out)
    require(bool(torch.isfinite(f).all()) and f.shape[1:] == (2, 789, 768), "bad features")
    return f


def cache():
    configure()
    proto = verify_lock()
    enc = E.load(C.CKPT["base"][0], "base", "cuda")
    record = {}
    for split in ("train", "heldout"):
        for c in C.CATS:
            path = OUT / f"features/{split}_{c}.pt"
            require(not path.exists(), f"exists: {path}")
            t0 = time.perf_counter()
            f = extract(proto[split][c], enc)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"features": f, "paths": [r["relative_path"] for r in proto[split][c]]}, path)
            record[f"{split}_{c}"] = {"n": len(f), "seconds": time.perf_counter() - t0, "sha256": C.sha(path)}
            print(split, c, record[f"{split}_{c}"], flush=True)
    C.write_json(OUT / "CACHE.json", record)


def load_split(split, c):
    rec = C.read(OUT / "CACHE.json")[f"{split}_{c}"]
    path = OUT / f"features/{split}_{c}.pt"
    require(C.sha(path) == rec["sha256"], f"feature cache changed: {path}")
    return torch.load(path, weights_only=True)["features"]


# --- training ----------------------------------------------------------------------------------------


class Batches:
    """Epoch permutations without replacement, incomplete tail dropped (official DataLoader drop_last)."""

    def __init__(self, n, size, gen):
        require(n >= size, "batch larger than data")
        self.n, self.size, self.gen, self.queue = n, size, gen, []

    def next(self):
        if len(self.queue) < self.size:
            self.queue = torch.randperm(self.n, generator=self.gen).tolist()
            self.queue = self.queue[:self.n - self.n % self.size]
        out, self.queue = self.queue[:self.size], self.queue[self.size:]
        return out


def train_stage(dec, current, iters, gen, replay=None):
    opt = O.make_optimizer(dec)
    sch = O.make_scheduler(opt, iters)
    ramp = round(0.1 * iters)
    batch = O.RECIPE["batch_size"]
    cur = Batches(len(current), batch // 2 if replay is not None else batch, gen)
    rep = Batches(len(replay), batch // 2, gen) if replay is not None else None
    dec.train()
    losses = []
    for i in range(iters):
        p = min(O.RECIPE["hm_p_final"] * i / ramp, O.RECIPE["hm_p_final"])
        x = current[cur.next()]
        if rep is not None:
            x = torch.cat([x, replay[rep.next()]])
            x = x[torch.randperm(len(x), generator=gen)]
        x = x.cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)
        total = 0.
        for xb in x.split(MICRO):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                en, de = dec(xb[:, 0], xb[:, 1])
            loss = O.global_cosine_hm_percent([e.float() for e in en], [d.float() for d in de],
                                              p=p, factor=O.RECIPE["hm_factor"]) * len(xb) / len(x)
            loss.backward()
            total += float(loss)
        require(math.isfinite(total), f"non-finite loss at iteration {i}")
        torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=O.RECIPE["clip_grad_norm"])
        opt.step()
        sch.step()
        losses.append(total)
    return losses


@torch.no_grad()
def normal_stats(dec, held):
    dec.eval()
    out = {}
    for c, f in held.items():
        loss, means, images = 0., [], []
        for xb in f.split(10):
            xb = xb.cuda()
            en, de = dec(xb[:, 0], xb[:, 1])
            loss += float(O.global_cosine(en, de)) * len(xb)
            means += R.grid(en, de).mean(dim=(1, 2)).tolist()
            images += R.dinomaly(en, de)[1].tolist()
        out[c] = {"global_cosine": loss / len(f), "mean_patch_1mcos": float(np.mean(means)),
                  "image_score_RD": images}
    return out


def new_decoder(seed):
    torch.manual_seed(seed)
    dec = O.Decoder(768, 12, checkpoint=True).cuda()
    O.init_trainable(dec)
    return dec


def state_bytes(dec):
    return sum(t.numel() * t.element_size() for t in dec.state_dict().values())


def save_ckpt(dec, path, meta):
    require(not path.exists(), f"exists: {path}")
    torch.save({"state": {k: v.cpu() for k, v in dec.state_dict().items()}, "meta": meta}, path)
    return C.sha(path)


def replay_buffer(proto, seed, done, train):
    """Class-balanced 24-image buffer over completed tasks; nested per-category random subsets."""
    gen = torch.Generator().manual_seed(seed + 9000)
    perms = {c: torch.randperm(len(proto["train"][c]), generator=gen).tolist() for c in C.ORDER}
    k = REPLAY_CAPACITY // len(done)
    idx = {c: perms[c][:k] for c in done}
    feats = torch.cat([train[c][idx[c]] for c in done])
    rows = [proto["train"][c][i] for c in done for i in idx[c]]
    return feats, {"identities": [r["relative_path"] for r in rows], "png_bytes": sum(r["bytes"] for r in rows),
                   "feature_bytes": feats.numel() * feats.element_size()}


def train(arm, seed):
    configure()
    proto = verify_lock()
    require(arm in ("C", "A", "B") and seed in SEEDS, "bad arm/seed")
    run = OUT / f"runs/{arm}_seed{seed}"
    require(not (run / "TRAIN.json").exists(), "run already completed")
    run.mkdir(parents=True, exist_ok=True)
    train_f = {c: load_split("train", c) for c in C.CATS}
    held = {c: load_split("heldout", c) for c in C.CATS}
    torch.cuda.reset_peak_memory_stats()
    record = {"arm": arm, "seed": seed, "started_utc": now(), "stages": []}
    if arm == "C":
        stages = [("joint", C.CATS, JOINT_ITERS)]
        dec = new_decoder(seed)
    elif arm == "A":
        stages = [(c, (c,), ITERS_PER_CLASS) for c in C.ORDER]
        dec = new_decoder(seed)
    else:
        src = OUT / f"runs/A_seed{seed}/stage1_bottle.pt"
        a_rec = C.read(OUT / f"runs/A_seed{seed}/TRAIN.json")
        require(C.sha(src) == a_rec["stages"][0]["checkpoint_sha256"], "A bottle checkpoint changed")
        dec = new_decoder(seed)
        dec.load_state_dict(torch.load(src, weights_only=True)["state"])
        record["inherited"] = {"stage1_bottle_from": str(src.relative_to(C.ROOT)), "sha256": C.sha(src)}
        stages = [(c, (c,), ITERS_PER_CLASS) for c in C.ORDER[1:]]
    for k, (name, cats, iters) in enumerate(stages):
        index = k + 1 if arm != "B" else k + 2
        torch.manual_seed(seed * 100 + index)                       # dropout stream per stage
        gen = torch.Generator().manual_seed(seed * 100 + index + (50 if arm == "B" else 0))
        current = torch.cat([train_f[c] for c in cats])
        replay, rep_info = None, None
        if arm == "B":
            done = list(C.ORDER[:C.ORDER.index(name)])
            replay, rep_info = replay_buffer(proto, seed, done, train_f)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        losses = train_stage(dec, current, iters, gen, replay)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        stats = normal_stats(dec, held)
        path = run / f"stage{index}_{name}.pt"
        digest = save_ckpt(dec, path, {"arm": arm, "seed": seed, "stage": index, "name": name})
        record["stages"].append({"index": index, "name": name, "categories": list(cats), "iters": iters,
                                 "train_images": len(current), "seconds": seconds, "loss_curve_every50":
                                 [float(np.mean(losses[i:i + 50])) for i in range(0, len(losses), 50)],
                                 "loss_last50": float(np.mean(losses[-50:])), "heldout_normal": stats,
                                 "replay": rep_info, "checkpoint": str(path.relative_to(C.ROOT)),
                                 "checkpoint_sha256": digest})
        print(arm, seed, name, f"{seconds:.0f}s", "loss", round(record["stages"][-1]["loss_last50"], 4),
              {c: round(s["global_cosine"], 4) for c, s in stats.items()}, flush=True)
    record.update({"finished_utc": now(), "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                   "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                   "persistent_decoder_bytes": state_bytes(dec)})
    C.write_json(run / "TRAIN.json", record)


# --- evaluation --------------------------------------------------------------------------------------


def checkpoints(gate):
    """(arm, seed, stage index, admitted categories, path) for the gate, each verified against TRAIN.json."""
    out = []
    arms = ("C",) if gate == 0 else ("A", "B")
    for arm in arms:
        for seed in SEEDS:
            rec = C.read(OUT / f"runs/{arm}_seed{seed}/TRAIN.json")
            for st in rec["stages"]:
                path = C.ROOT / st["checkpoint"]
                require(C.sha(path) == st["checkpoint_sha256"], f"checkpoint changed: {path}")
                admitted = list(C.CATS) if arm == "C" else list(C.ORDER[:st["index"]])
                out.append((arm, seed, st["index"], admitted, path))
    return out


def dev_features(proto):
    path = OUT / "features/dev.pt"
    if path.exists():
        rec = C.read(OUT / "DEV_FEATURES.json")
        require(C.sha(path) == rec["sha256"], "dev feature cache changed")
        return torch.load(path, weights_only=True)
    enc = E.load(C.CKPT["base"][0], "base", "cuda")
    rows = [r for c in C.CATS for r in proto["evaluation"][c]]
    f = extract(rows, enc)
    del enc
    torch.cuda.empty_cache()
    data = {"features": f, "paths": [r["relative_path"] for r in rows]}
    torch.save(data, path)
    C.write_json(OUT / "DEV_FEATURES.json", {"sha256": C.sha(path), "utc": now(), "n": len(rows)})
    return data


@torch.no_grad()
def score(gate):
    """One-shot scoring of the gate's checkpoints on the development cohort; saves raw maps."""
    configure()
    proto = verify_lock()
    target = OUT / f"gate{gate}/SCORES.json"
    require(not target.exists(), "gate scoring is one-shot")
    dev = dev_features(proto)
    index = {p: i for i, p in enumerate(dev["paths"])}
    record = {}
    for arm, seed, stage, admitted, path in checkpoints(gate):
        dec = O.Decoder(768, 12).cuda()
        dec.load_state_dict(torch.load(path, weights_only=True)["state"])
        dec.eval()
        rows = [r for c in admitted for r in proto["evaluation"][c]]
        x = dev["features"][[index[r["relative_path"]] for r in rows]]
        s256, img_d, grids = [], [], []
        for xb in x.split(10):
            xb = xb.cuda()
            en, de = dec(xb[:, 0], xb[:, 1])
            s, i = R.dinomaly(en, de)
            s256.append(s.cpu()); img_d.append(i.cpu()); grids.append(R.grid(en, de).cpu())
        out = OUT / f"gate{gate}/maps/{arm}_seed{seed}_stage{stage}.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        require(not out.exists(), f"exists: {out}")
        torch.save({"paths": [r["relative_path"] for r in rows], "s256": torch.cat(s256),
                    "image_RD": torch.cat(img_d), "grid": torch.cat(grids)}, out)
        record[f"{arm}/{seed}/{stage}"] = {"maps": str(out.relative_to(C.ROOT)), "sha256": C.sha(out),
                                           "checkpoint_sha256": C.sha(path), "admitted": admitted}
        print("scored", arm, seed, stage, admitted, flush=True)
    C.write_json(target, record)


@torch.no_grad()
def latency():
    configure()
    proto = verify_lock()
    target = OUT / "LATENCY.json"
    require(not target.exists(), "exists")
    row = proto["heldout"]["bottle"][0]
    enc = E.load(C.CKPT["base"][0], "base", "cuda")
    dec = O.Decoder(768, 12).cuda().eval()
    x = E.preprocess(C.source(row))[None].cuda()
    times = []
    for i in range(25):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        g = enc.groups(x)
        en, de = dec(g[:, 0], g[:, 1])
        s, _ = R.dinomaly(en, de)
        R.native(s[0], (900, 900))
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    C.write_json(target, {"seconds_per_image_batch1_fp32": float(np.median(times[5:])), "repeats": 20,
                          "pipeline": "preprocess excluded; encoder + decoder + R_D map + native upsample 900x900",
                          "encoder_parameters": sum(p.numel() for p in enc.parameters()),
                          "encoder_bytes_fp32": sum(p.numel() * 4 for p in enc.parameters()),
                          "decoder_bytes_fp32": state_bytes(dec)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("lock", "cache", "train", "score", "latency"))
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--gate", type=int, choices=(0, 1))
    a = ap.parse_args()
    if a.stage == "lock":
        lock()
    elif a.stage == "cache":
        cache()
    elif a.stage == "train":
        train(a.arm, a.seed)
    elif a.stage == "score":
        score(a.gate)
    else:
        latency()


if __name__ == "__main__":
    main()
