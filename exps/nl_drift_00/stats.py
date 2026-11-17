"""NL-DRIFT-00 GPU phase: per-image residual sufficient statistics under every relevant decoder checkpoint.

    python -m exps.nl_drift_00.stats

For each category c (stream order) and each boundary t >= a(c), with theta_t = checkpoint of stage t+1:
  raw grids of the trained-once pool T_c, the unseen pool U_c (bottle/carpet/hazelnut) and c's ever-replayed images;
  PQ-decoded grids (c's C2-DOM codebook) of every witness candidate (union over draws of W(4), cohorts T and U).
Per image: n=784, S1 = sum r, S2 = sum r^2 with r = max(0, 1-cos), and S1raw = sum (1-cos) (for PV3), FP64.
Also the FP64 centroid of each category over all R15 train patches. Writes STATS.pt and STATS_LOCK.json
(create-exclusive). Label-free; refuses to run without LOCK.json; custody hook installed before any data access.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from exps.nl_drift_00 import common as C


def image_stats(dec, X, batch=32):
    from exps.vitb8_q2 import common as V
    out = torch.empty(len(X), 3, dtype=torch.float64)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i + batch].cuda(non_blocking=False)
            en, de = dec(xb)
            g = V.grid(en, de).double()                                  # [B,28,28], 1 - cos
            r = g.clamp_min(0.0)
            out[i:i + batch, 0] = r.sum((1, 2)).cpu()
            out[i:i + batch, 1] = r.square().sum((1, 2)).cpu()
            out[i:i + batch, 2] = g.sum((1, 2)).cpu()
    C.require(bool(torch.isfinite(out).all()), "non-finite residual statistics")
    return out


def witness_candidates(c, T, U):
    """Union over draws of W(4), cohorts T then U, in first-appearance order."""
    cand = []
    for X_, P in (("T", T), ("U", U)):
        if c in P:
            for w, _ in C.witness_sets(X_, c, P[c]):
                cand += [p for p in w if p not in cand]
    return cand


def main():
    C.install_custody()
    lk = C.verify_lock()
    C.require(not (C.OUT / "STATS.pt").exists(), "STATS.pt exists (create-exclusive)")
    from exps.nl_sgm_01a import codecs as K
    from exps.vitb8_q2 import common as V
    torch.backends.cuda.matmul.allow_tf32 = False                       # as exps.dinomaly.run.configure
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(6)
    t_start = time.time()
    (C.OUT / "logs").mkdir(parents=True, exist_ok=True)
    log_f = open(C.OUT / "logs/stats.log", "a")

    def log(*a):
        s = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
        print(s, flush=True)
        log_f.write(s + "\n"); log_f.flush()

    # PV1: checkpoints
    ck = {}
    for s in range(1, 16):
        rec = C.stage_record(s)
        path = C.ROOT / rec["checkpoint"]
        C.require(path.parent.resolve() == C.RUN.resolve(), f"checkpoint outside run dir: {path}")
        h = C.sha_file(path)
        C.require(h == rec["checkpoint_sha256"], f"PV1: checkpoint sha mismatch stage {s}")
        ck[s - 1] = (path, h)
    log("PV1 ok: 15 checkpoints")
    # PV2: CALM-02 train caches against the ViT-B/8 cache record
    cache = C.read_json(C.CACHE_JSON)["files"]
    pv2 = {}
    for c in C.ORDER:
        src = cache[f"train_{c}"]["source"]
        want = src.split("sha256:")[1]
        C.require(src.startswith(f"calm02/{c}_train.pt"), f"unexpected cache source {src}")
        got = C.sha_file(C.FEAT / f"{c}_train.pt")
        C.require(got == want, f"PV2: CALM-02 train cache sha mismatch {c}")
        pv2[c] = got
    log("PV2 ok: 15 CALM-02 train caches")

    rows = {}
    for c in C.ORDER:                          # rows only; the mapping is released at once (AMENDMENT_01: RSS)
        obj = torch.load(C.FEAT / f"{c}_train.pt", map_location="cpu", weights_only=False, mmap=True)
        C.require(tuple(obj["features"].shape[1:]) == (C.TOKENS, C.DIM) and obj["features"].dtype == torch.float32,
                  f"feature layout {c}")
        rows[c] = [r["relative_path"] for r in obj["rows"]]
        del obj
    T, U, R, train, held, rep = C.pools(rows)
    log("PV4 pools ok", {c: len(T[c]) for c in C.ORDER}, {c: len(U[c]) for c in U})
    codebooks = torch.load(C.C2DOM_CKPT, map_location="cpu", weights_only=False)["codebooks"]
    n_cand = {c: len(witness_candidates(c, T, U)) for c in C.ORDER}

    stats = {"meta": {"lock_sha256": C.sha_file(C.LOCK), "checkpoints": {t: h for t, (_, h) in ck.items()},
                      "calm02_train_sha256": pv2, "utc_start": C.now()},
             "paths": {}, "raw": {}, "pq": {}, "centroid": {}, "pq_paths": {}, "pq_encode_err": {}}
    dec = V.SingleStreamDecoder(checkpoint=False).cuda().eval()
    gpu_s, n_grids, n_loads = 0.0, 0, 0
    first_cat_wall = None
    for c in C.ORDER:
        tc = time.time()
        a = C.A[c]
        idx = {p: i for i, p in enumerate(rows[c])}
        obj = torch.load(C.FEAT / f"{c}_train.pt", map_location="cpu", weights_only=False, mmap=True)
        C.require([r["relative_path"] for r in obj["rows"]] == rows[c], f"row order changed {c}")
        feats = {c: obj["features"]}
        paths = list(T[c]) + list(U.get(c, [])) + list(R[c])
        C.require(len(set(paths)) == len(paths), f"duplicate identities {c}")
        X = feats[c][torch.tensor([idx[p] for p in paths])].clone()
        r15 = feats[c][torch.tensor([idx[p] for p in train[c]])]
        stats["centroid"][c] = r15.double().mean((0, 1)).numpy()             # over all R15 train patches of c
        del r15
        cand = witness_candidates(c, T, U)
        Xc =feats[c][torch.tensor([idx[p] for p in cand])]
        Xq = torch.empty_like(Xc)
        err = []
        for i in range(len(cand)):
            codes = K.encode(Xc[i], codebooks[a])
            Xq[i] = K.decode(codes, codebooks[a])
            err.append(float((Xq[i].double() - Xc[i].double()).square().sum(1).mean()))
        stats["paths"][c], stats["pq_paths"][c], stats["pq_encode_err"][c] = paths, cand, err
        raw = torch.full((len(paths), 15, 3), float("nan"), dtype=torch.float64)
        pq = torch.full((len(cand), 15, 3), float("nan"), dtype=torch.float64)
        for t in range(a, 15):
            dec.load_state_dict(torch.load(ck[t][0], map_location="cpu", weights_only=True)["state"])
            dec.eval()
            n_loads += 1
            torch.cuda.synchronize(); t0 = time.time()
            raw[:, t] = image_stats(dec, X)
            pq[:, t] = image_stats(dec, Xq)
            torch.cuda.synchronize(); gpu_s += time.time() - t0
            n_grids += len(X) + len(Xq)
        stats["raw"][c], stats["pq"][c] = raw.numpy(), pq.numpy()
        del X, Xc, Xq, feats, obj
        wall = time.time() - t_start
        vram = torch.cuda.max_memory_allocated() / 1e9
        log(c, "images", len(paths), "pq", len(cand), "boundaries", 15 - a, f"cat {time.time() - tc:.0f}s",
            f"wall {wall:.0f}s", f"grids {n_grids}", f"rss {C.peak_rss_gb():.2f}GB", f"vram {vram:.2f}GB")
        C.require(vram <= C.CAPS["vram_gb"], "CAP: VRAM")
        C.require(C.peak_rss_gb() <= C.CAPS["rss_gb"], "CAP: RSS")
        C.require(wall <= C.CAPS["gpu_phase_wall_h"] * 3600, "CAP: GPU phase wall")
        if first_cat_wall is None:                                    # projection: grids remaining at measured rate
            first_cat_wall = wall
            rate = wall / max(n_grids, 1)
            todo = sum((len(T[k]) + len(U.get(k, [])) + len(R[k]) + n_cand[k]) * (15 - C.A[k]) for k in C.ORDER[1:])
            proj = wall + rate * todo
            log("projection", f"{proj:.0f}s", "cap", C.CAPS["gpu_phase_wall_h"] * 3600)
            C.require(proj <= C.CAPS["gpu_phase_wall_h"] * 3600, f"CAP: projected GPU phase wall {proj:.0f}s")
    stats["meta"].update({"utc_end": C.now(), "wall_s": time.time() - t_start, "gpu_forward_s": gpu_s,
                          "n_grids": n_grids, "n_checkpoint_loads": n_loads,
                          "s_per_grid": gpu_s / n_grids, "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                          "rss_peak_gb": C.peak_rss_gb(), "access_log": C.access_log()})
    part = C.OUT / "STATS.part.pt"
    torch.save(stats, part)
    part.replace(C.OUT / "STATS.pt")
    h = C.write_json_x(C.OUT / "STATS_LOCK.json", {"stats_sha256": C.sha_file(C.OUT / "STATS.pt"),
                                                    "lock_sha256": C.sha_file(C.LOCK), **{k: v for k, v in stats["meta"].items()
                                                                                         if k != "access_log"},
                                                    "n_access_log": len(stats["meta"]["access_log"]), "utc": C.now()})
    log("STATS DONE", f"wall {time.time() - t_start:.0f}s", f"gpu {gpu_s:.0f}s", f"grids {n_grids}",
        f"{1000 * gpu_s / n_grids:.1f} ms/grid", "STATS_LOCK", h[:16])


if __name__ == "__main__":
    main()
