"""NL-SGM-01A-C2-DOM sweep — normal-only stages (no test data is opened here).

    python -m exps.nl_sgm_01a_c2dom.stream fit                 # per-category in-domain codebooks + SEL-pool codes
    python -m exps.nl_sgm_01a_c2dom.stream run ARM [--gpu]     # 15-boundary stream, checkpoints, ledger
    python -m exps.nl_sgm_01a_c2dom.stream normdiag            # SEL-probe capacity diagnostic (B1/B2/B4 final)
    python -m exps.nl_sgm_01a_c2dom.stream b8                  # mechanical B8 decision (create-exclusive)

Stream rule (both codecs) at boundary t, category c = ORDER[t]:
  capacity_t from the ledger (PQ: t+1 codebooks charged) -> candidates = c's SEL-pool rows (PQ: decode(encode(x, CB_c)))
  -> kept[t] = candidates (writer order) -> AD-02 per_group_quota(capacity_t, t+1) -> every category with more rows
  than its quota is FPS-subselected (canonical csg01.select_fps decisions) within its own kept rows (FPSRule._requota).
CB_c is fit only on c's FIT-pool rows (image-disjoint from the SEL pool) when c arrives; never refit.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from exps.ad02 import hope_cad_ad02_partition as AP
from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a import exact as X
from exps.nl_sgm_01a_c2dom import common as D
from exps.nl_sgm_01a_c2dom import fps as FPS
from exps.task_p.p0_normal_support import require

CBDIR = D.OUT / "codebooks"


# ------------------------------------------------------------------ codebooks

def do_fit():
    D.verify_lock()
    co, n = D.setup()
    part = D.partition(co, n)
    lk = D.read_json(D.LOCK)
    log = D.Log("fit")
    CBDIR.mkdir(parents=True, exist_ok=True)
    F = C.Features()
    t_all = time.time()
    for t, c in enumerate(D.ORDER):
        p = CBDIR / f"{t:02d}_{c}.pt"
        if (CBDIR / f"{t:02d}_{c}.json").exists():
            log("SKIP (done)", c); continue
        require(not p.exists(), f"partial codebook file {p}: refuse to overwrite")
        fg, sg = D.pool_gids(n, c, part[c]["fit"]), D.pool_gids(n, c, part[c]["sel"])
        require(D.gid_sha(fg) == lk["partition"][c]["fit_gids_sha"] and D.gid_sha(sg) == lk["partition"][c]["sel_gids_sha"],
                f"partition changed {c}")
        require(not bool(torch.isin(fg, sg).any()), "fit/sel identity overlap")
        xf = D.pool_rows(F, part[c]["fit"])
        order = D.fit_order(fg)[:D.FIT_ROWS]
        sample = xf[order].contiguous()
        del xf
        t0 = time.time()
        cb = K.fit(sample, seed=D.SEED, iters=D.FIT_ITERS)
        fit_s = time.time() - t0
        del sample
        xs = D.pool_rows(F, part[c]["sel"])
        F.drop(c)
        t1 = time.time()
        codes = K.encode(xs, cb, chunk=1024)
        enc_s = time.time() - t1
        rec_err = (xs - K.decode(codes, cb)).double()
        nmse = float(rec_err.square().sum() / (xs.double() - xs.double().mean(0)).square().sum())
        del xs, rec_err
        torch.save({"category": c, "t": t, "codebook": cb, "sel_gids": sg, "sel_codes": codes}, p)
        rec = {"category": c, "t": t, "fit_rows": int(len(order)), "fit_pool_rows": int(len(fg)),
               "sel_pool_rows": int(len(sg)), "fit_seconds": fit_s, "encode_seconds": enc_s, "sel_nmse": nmse,
               "codebook_sha256": C.sha(p), "fit_sample_gids_sha": D.gid_sha(fg[order]),
               "peak_rss_gb": D.peak_rss_gb()}
        D.write_json(CBDIR / f"{t:02d}_{c}.json", rec)
        log("FIT", c, json.dumps(rec))
        D.cap_check(t_all, D.CAPS["stream_wall_h"])
    log("FIT DONE", time.time() - t_all)


def load_cb(t, wait_s=0):
    c = D.ORDER[t]
    marker = CBDIR / f"{t:02d}_{c}.json"
    t0 = time.time()
    while not marker.exists():
        require(time.time() - t0 < wait_s, f"codebook {c} not available")
        time.sleep(20)
    rec = D.read_json(marker)
    p = CBDIR / f"{t:02d}_{c}.pt"
    require(C.sha(p) == rec["codebook_sha256"], f"codebook hash mismatch {c}")
    return torch.load(p, weights_only=False), rec


# ------------------------------------------------------------------ stream

class Stream:
    def __init__(self, arm, allow_gpu):
        spec = D.arms(D.BUDGETS.keys())[arm]
        self.arm, self.codec, self.B, self.allow_gpu = arm, spec["codec"], spec["bytes"], allow_gpu
        self.kept = {}                      # k -> (decoded FP32 rows, gids int64, codes uint8 | None), kept order
        self.codebooks = []                 # FP16 [96,256,8] per seen category (PQ)
        self.dir = D.OUT / "streams" / arm
        self.fps_records = []

    def requota(self, cap, t):
        cats = sorted(self.kept)
        quotas = AP.per_group_quota(cap, len(cats))
        for k, q in zip(cats, quotas):
            x, g, cd = self.kept[k]
            if len(x) > q:
                idx, rec = FPS.select(x, q, allow_gpu=self.allow_gpu)
                rec.update(boundary=t, category=k)
                self.fps_records.append(rec)
                ii = torch.as_tensor(idx)
                self.kept[k] = (x[ii].contiguous(), g[ii], None if cd is None else cd[ii])

    def state(self, t):
        cap = D.capacity(self.codec, self.B, t)
        cats = sorted(self.kept)
        g = torch.cat([self.kept[k][1] for k in cats])
        o = torch.cat([torch.full((len(self.kept[k][1]),), k, dtype=torch.int8) for k in cats])
        require(len(g) <= cap, "INTEGRITY: over capacity")
        st = {}
        if self.codec == "fp32":
            x = torch.zeros(cap, 768); x[:len(g)] = torch.cat([self.kept[k][0] for k in cats]); st["x"] = x
        else:
            cd = torch.zeros(cap, K.M, dtype=torch.uint8); cd[:len(g)] = torch.cat([self.kept[k][2] for k in cats])
            st["codes"] = cd
            st["codebooks"] = torch.stack(self.codebooks)
        ids = torch.full((cap,), -1, dtype=torch.int32); ids[:len(g)] = g.to(torch.int32); st["ids"] = ids
        own = torch.full((cap,), -1, dtype=torch.int8); own[:len(g)] = o; st["owner"] = own
        hdr = torch.zeros(16, dtype=torch.int32); hdr[0] = t
        for k in cats:
            hdr[1 + k] = len(self.kept[k][1])
        st["header"] = hdr
        b = D.state_bytes(st)
        require(b == D.ledger_bytes(self.codec, self.B, t) and b <= self.B, f"LEDGER: {b} B != formula or > budget")
        return st, b, len(g)

    def restore(self, ck, t):
        n_rows = int(ck["header"][1:].sum())
        ids, own = ck["ids"][:n_rows].long(), ck["owner"][:n_rows].long()
        if self.codec == "pq_dom":
            self.codebooks = list(ck["codebooks"])
        for k in range(t + 1):
            m = own == k
            if self.codec == "fp32":
                x = ck["x"][:n_rows][m].clone(); cd = None
            else:
                cd = ck["codes"][:n_rows][m].clone(); x = K.decode(cd, self.codebooks[k])
            self.kept[k] = (x, ids[m], cd)


def do_run(arm, allow_gpu):
    D.verify_lock()
    co, n = D.setup()
    part = D.partition(co, n)
    lk = D.read_json(D.LOCK)
    require(arm in lk["arms"], f"{arm} not registered in LOCK")
    if arm.endswith("B8"):
        dec = D.read_json(D.OUT / "B8_DECISION.json")
        require(dec["run_B8"], "B8 not authorized by the locked rule")
    s = Stream(arm, allow_gpu)
    s.dir.mkdir(parents=True, exist_ok=True)
    log = D.Log(f"stream_{arm}")
    F = C.Features()
    t_all = time.time()
    start = 0
    for t in range(15):
        if (s.dir / f"boundary_t{t:02d}.json").exists():
            start = t + 1
    if start:
        ck = torch.load(s.dir / f"ckpt_t{start - 1:02d}.pt", weights_only=False)
        rec = D.read_json(s.dir / f"boundary_t{start - 1:02d}.json")
        require(C.sha(s.dir / f"ckpt_t{start - 1:02d}.pt") == rec["ckpt_sha256"], "resume checkpoint hash mismatch")
        s.restore(ck, start - 1)
        log("RESUME from boundary", start - 1)
    for t in range(start, 15):
        c = D.ORDER[t]
        tb = time.time()
        cap = D.capacity(s.codec, s.B, t)
        sg = D.pool_gids(n, c, part[c]["sel"])
        require(D.gid_sha(sg) == lk["partition"][c]["sel_gids_sha"], f"sel pool changed {c}")
        enc_s = 0.0
        if s.codec == "fp32":
            x = D.pool_rows(F, part[c]["sel"]); cd = None
            F.drop(c)
        else:
            obj, cbrec = load_cb(t, wait_s=6 * 3600)
            require(torch.equal(obj["sel_gids"], sg), "codes/gids mismatch")
            cd = obj["sel_codes"]
            t1 = time.time(); x = K.decode(cd, obj["codebook"]); enc_s = time.time() - t1
            s.codebooks.append(obj["codebook"])
            require(len(s.codebooks) == t + 1, "append-only codebook count")
        prev = {k: set(v[1].tolist()) for k, v in s.kept.items()}
        s.kept[t] = (x, sg, cd)
        nf = len(s.fps_records)
        t2 = time.time()
        s.requota(cap, t)
        fps_s = time.time() - t2
        for k, ids in prev.items():
            require(set(s.kept[k][1].tolist()) <= ids, f"INTEGRITY: category {k} support not a subset of previous")
        st, b, rows = s.state(t)
        p = s.dir / f"ckpt_t{t:02d}.pt"
        require(not p.exists(), f"refuse to overwrite {p}")
        torch.save(st, p)
        recs = s.fps_records[nf:]
        rec = {"arm": s.arm, "t": t, "category": c, "capacity": cap, "rows": rows, "persistent_bytes": b,
               "budget_bytes": s.B, "codebook_bytes": (t + 1) * D.CB_BYTES if s.codec == "pq_dom" else 0,
               "per_category_rows": {D.ORDER[k]: len(v[1]) for k, v in sorted(s.kept.items())},
               "per_category_gids_sha": {D.ORDER[k]: D.gid_sha(v[1]) for k, v in sorted(s.kept.items())},
               "fps_calls": recs, "fps_seconds": fps_s, "decode_seconds": enc_s,
               "boundary_seconds": time.time() - tb, "ckpt_sha256": C.sha(p), **D.cap_check(t_all, D.CAPS["stream_wall_h"]),
               "vram_peak_bytes": max([r.get("vram_peak_bytes", 0) for r in recs] or [0]), "utc": D.now()}
        D.write_json(s.dir / f"boundary_t{t:02d}.json", rec)
        log(s.arm, t, c, "cap", cap, "rows", rows, "bytes", b, "fps_s", round(fps_s, 1),
            "impl", sorted({r["impl"] for r in recs}), "rss", round(rec["peak_rss_gb"], 2))
    if not (s.dir / "STREAM_DONE.json").exists():
        recs = [D.read_json(s.dir / f"boundary_t{t:02d}.json") for t in range(15)]
        D.write_json(s.dir / "STREAM_DONE.json", {
            "arm": arm, "wall_seconds_this_process": time.time() - t_all,
            "fps_seconds_total": sum(r["fps_seconds"] for r in recs),
            "boundary_seconds_total": sum(r["boundary_seconds"] for r in recs),
            "fps_impl_counts": {i: sum(1 for r in recs for f in r["fps_calls"] if f["impl"] == i)
                                for i in ("cpu_canonical", "gpu_certified", "gpu_certified_exact_ties", "cpu_canonical_fallback")},
            "peak_rss_gb": max(r["peak_rss_gb"] for r in recs), "vram_peak_bytes": max(r["vram_peak_bytes"] for r in recs),
            "final_persistent_bytes": recs[-1]["persistent_bytes"], "env": D.env_threads(), "allow_gpu": allow_gpu,
            "utc": D.now()})
    log("DONE", arm)


# ------------------------------------------------------------------ normal-only capacity diagnostic + B8 rule

def supports_at(arm, t, only=None):
    ck = torch.load(D.OUT / "streams" / arm / f"ckpt_t{t:02d}.pt", weights_only=False)
    n_rows = int(ck["header"][1:].sum())
    ids, own = ck["ids"][:n_rows].long(), ck["owner"][:n_rows].long()
    out = {}
    for k in range(t + 1):
        if only is not None and D.ORDER[k] != only:
            continue
        m = own == k
        x = ck["x"][:n_rows][m] if "x" in ck else K.decode(ck["codes"][:n_rows][m], ck["codebooks"][k])
        g = ids[m]
        o = torch.argsort(g)
        out[D.ORDER[k]] = X.RowSet(x[o].contiguous(), g[o])
    return out


def do_normdiag(budgets=D.REGISTERED, name="NORMAL_DIAG.json"):
    D.verify_lock()
    co, n = D.setup()
    part = D.partition(co, n)
    F = C.Features()
    log = D.Log("normdiag")
    target = D.OUT / name
    require(not target.exists(), "one-shot")
    res = {"schema": "c2dom_normal_diag_v1", "probe_cohort": "SEL probes (6 images/category, never written)",
           "budgets": list(budgets), "categories": {}}
    finals = {a: supports_at(a, 14) for b in budgets for a in (f"DOM_PQ_{b}", f"SEL_FP32_{b}")}
    for t, c in enumerate(D.ORDER):
        q = F.images(co[c]["sel"]).reshape(-1, 768).clone()
        xs = D.pool_rows(F, part[c]["sel"]); sg = D.pool_gids(n, c, part[c]["sel"])
        F.drop(c)
        cb, _ = load_cb(t)
        ref = {"fp32": X.RowSet(xs, sg), "pq_dom": X.RowSet(K.decode(cb["sel_codes"], cb["codebook"]).contiguous(), sg)}
        dref = {k: X.nn(q, v)[0] for k, v in ref.items()}
        rc = {"pool_rows": len(sg)}
        for a, sup in finals.items():
            codec = "fp32" if a.startswith("SEL_FP32") else "pq_dom"
            d, _ = X.nn(q, sup[c])
            rc[a] = {"median_ratio_to_pool": float((d / dref[codec]).median()), "rows": len(sup[c].ids),
                     "median_d": float(d.median())}
        rc["pq_pool_vs_fp32_pool_median_ratio"] = float((dref["pq_dom"] / dref["fp32"]).median())
        res["categories"][c] = rc
        log(c, json.dumps({a: round(v["median_ratio_to_pool"], 4) for a, v in rc.items() if isinstance(v, dict)}))
        del ref, xs
    for a in finals:
        res[f"G_{a}"] = float(np.mean([res["categories"][c][a]["median_ratio_to_pool"] - 1 for c in D.ORDER]))
    D.write_json(target, res)
    log("written", target, {k: v for k, v in res.items() if k.startswith("G_")})


def do_b8():
    lk = D.verify_lock()
    target = D.OUT / "B8_DECISION.json"
    require(not target.exists(), "one-shot")
    require(not (D.OUT / "eval").exists(), "B8 decision must precede any real-anomaly evaluation")
    nd = D.read_json(D.OUT / "NORMAL_DIAG.json")
    g2, g4 = nd["G_DOM_PQ_B2"], nd["G_DOM_PQ_B4"]
    limited = g4 >= 0.05 and (g2 - g4) >= 0.25 * g2
    proj = lk["b8_projection_hours"]
    fits = proj <= lk["b8_rule"]["max_projected_hours"]
    D.write_json(target, {"G_DOM_PQ_B2": g2, "G_DOM_PQ_B4": g4, "capacity_limited": bool(limited),
                          "projected_hours": proj, "fits_resource_cap": bool(fits),
                          "run_B8": bool(limited and fits), "rule": lk["b8_rule"], "utc": D.now()})
    print("B8 decision:", limited and fits, "G_B2", g2, "G_B4", g4)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        do_fit()
    elif cmd == "run":
        do_run(sys.argv[2], "--gpu" in sys.argv)
    elif cmd == "normdiag":
        do_normdiag()
    elif cmd == "b8":
        do_b8()
    else:
        raise SystemExit(cmd)
