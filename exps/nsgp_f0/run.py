"""NSGP-F0 execution (normal-only; no MVTec test feature, label or mask is ever opened).

    python -m exps.nsgp_f0.run audit        # A0: provenance + numerical reconstruction of the canonical B0 bank (pre-lock)
    python -m exps.nsgp_f0.run fit T        # static compressors on the bank at boundary T (+ training-only FITQ scores)
    python -m exps.nsgp_f0.run select       # preregistered N0 budget selection from training-only FITQ scores
    python -m exps.nsgp_f0.run eval T       # independent probes (EVAL) + secondary cohorts at boundary T
    python -m exps.nsgp_f0.run geometry     # final-boundary exact Hausdorff, paired bounds, theorem check, rarity

Every output is create-exclusive; a command resumes by skipping completed files.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a import exact as X
from exps.nl_sgm_01a_c2dom import common as D2
from exps.nsgp_f0 import common as S
from exps.task_p.p0_normal_support import require

ARMS_FIT = ("C0_C", "C0_Q", "N0_C", "N0_Q", "P0")


def arm_k(arm):
    fam, b = arm.split("_") if "_" in arm else (arm, None)
    return {"C0": S.K_C0, "N0": S.K_N0}[fam][b] if fam in ("C0", "N0") else None


def load_rows(t, co, n, F):
    b, rec = S.bank(t)
    rows = {}
    for c, v in b.items():
        x = S.o0_rows(F, co, n, c, v["ids"])
        F.drop(c)
        rows[c] = {"ids": v["ids"], "x": x, "b0": S.b0_rows(v), "codes": v["codes"], "codebook": v["codebook"]}
    return rows, rec


# ------------------------------------------------------------------ A0

def do_audit():
    target = S.OUT / "A0_AUDIT.json"
    require(not target.exists(), "one-shot")
    co, n = S.setup()
    part = D2.partition(co, n)
    lk = D2.read_json(D2.LOCK)
    F = C.Features()
    out = {"partition": {}, "boundaries": {}, "normal_diag_reproduction": {}, "sand_sel_reproduction": {}}
    for c in S.ORDER:
        fg, sg = D2.pool_gids(n, c, part[c]["fit"]), D2.pool_gids(n, c, part[c]["sel"])
        probes = set(co[c]["sel"]) | set(co[c]["eval"])
        out["partition"][c] = {
            "fit_gids_sha_matches_c2dom_lock": D2.gid_sha(fg) == lk["partition"][c]["fit_gids_sha"],
            "sel_gids_sha_matches_c2dom_lock": D2.gid_sha(sg) == lk["partition"][c]["sel_gids_sha"],
            "probes_in_writers": len(probes & set(co[c]["writers"])),
            "eval_sel_overlap": len(set(co[c]["sel"]) & set(co[c]["eval"])),
            "fitq_in_sel_pool_or_probes": len(set(S.fitq(co, part, c)) & ({p for _, p in part[c]["sel"]} | probes)),
            "heldout_overlap_with_r15": len(set(S.heldout_paths(c)) & (set(co[c]["writers"]) | probes))
            if c in ("bottle", "carpet", "hazelnut") else None}
        require(all(v in (True, 0, None) for v in out["partition"][c].values()), f"A0 partition/independence {c}")
    sel_pool = {c: {j for j, _ in part[c]["sel"]} for c in S.ORDER}
    for t in range(15):
        b, rec = S.bank(t)
        ck = torch.load(S.MEM_DIR / f"ckpt_t{t:02d}.pt", weights_only=False)
        sb = S.state_bytes(ck)
        r = {"rows": rec["rows"], "capacity": rec["capacity"], "persistent_bytes_ledger": rec["persistent_bytes"],
             "state_bytes_recomputed": sb, "formula": D2.ledger_bytes("pq_dom", D2.BUDGETS["B2"], t),
             "categories": sorted(b, key=S.ORDER.index)}
        require(sb == rec["persistent_bytes"] == r["formula"], f"A0 bytes t={t}")
        require(r["categories"] == list(S.ORDER[:t + 1]), f"A0 future category in bank t={t}")
        if t in S.BOUNDARIES:
            r["codes_reencoded_equal"] = {}
            r["o0_reconstruction_nmse"] = {}
            for c, v in b.items():
                j, _ = S.address(n, c, v["ids"])
                require(set(torch.unique(j).tolist()) <= sel_pool[c], f"A0 bank row outside SEL pool {c}")
                cb = torch.load(D2.OUT / "codebooks" / f"{S.ORDER.index(c):02d}_{c}.pt", weights_only=False)
                require(torch.equal(cb["codebook"], v["codebook"]), f"A0 codebook {c}")
                x = S.o0_rows(F, co, n, c, v["ids"])
                F.drop(c)
                enc = K.encode(x, v["codebook"], chunk=1024)
                r["codes_reencoded_equal"][c] = bool(torch.equal(enc, v["codes"]))
                xb = S.b0_rows(v)
                r["o0_reconstruction_nmse"][c] = float((xb - x).double().square().sum() /
                                                       (x.double() - x.double().mean(0)).square().sum())
                require(r["codes_reencoded_equal"][c], f"A0 stored codes are not encode(O0 row) {c} t={t}")
        out["boundaries"][t] = r
    # stored official C2-DOM normal-only values: median SEL-probe distance to the final DOM_PQ_B2 support
    nd = D2.read_json(D2.OUT / "NORMAL_DIAG.json")["categories"]
    b, _ = S.bank(S.FINAL)
    for c, v in b.items():
        q = F.images(co[c]["sel"]).reshape(-1, S.DIM).clone()
        F.drop(c)
        d, _ = S.score(q, S.b0_rows(v), v["ids"])
        ref = nd[c][S.MEM_ARM]
        out["normal_diag_reproduction"][c] = {"median_d": float(d.median()), "ref": ref["median_d"],
                                              "abs_diff": abs(float(d.median()) - ref["median_d"]), "rows": len(v["ids"])}
        require(out["normal_diag_reproduction"][c]["abs_diff"] <= 1e-12 and ref["rows"] == len(v["ids"]),
                f"A0 NORMAL_DIAG not reproduced {c}")
    # SAND-stored SEL-probe distances to the same bank at the four boundaries (only the 'sel' member is read)
    sd = S.ROOT / "results/hope_cad/sand_final/dist" / S.MEM_ARM
    for t in S.BOUNDARIES:
        b, rec = S.bank(t)
        for c, v in b.items():
            p = sd / f"{c}_{rec['per_category_gids_sha'][c][:24]}.npz"
            if not p.exists():
                continue
            with np.load(p) as z:
                ref = z["sel"].reshape(-1)
            q = F.images(co[c]["sel"]).reshape(-1, S.DIM).clone()
            F.drop(c)
            d, _ = S.score(q, S.b0_rows(v), v["ids"])
            diff = float(np.abs(d.numpy() - ref).max())
            out["sand_sel_reproduction"][f"t{t:02d}_{c}"] = diff
            require(diff <= 1e-12, f"A0 SAND SEL distances not reproduced {c} t={t}: {diff}")
    out["calm02_manifest_sha256"] = S.sha(C.FEATURES / "MANIFEST.json")
    out["c2dom_lock_sha256"] = S.sha(D2.LOCK)
    out["r15_digest"] = C.r15_digest()
    out["verdict_A0"] = "PASS"
    out["utc"] = S.now()
    S.write_json(target, out)
    print("A0 PASS", {c: v["abs_diff"] for c, v in out["normal_diag_reproduction"].items()})


# ------------------------------------------------------------------ fit (training-only)

def fitq_scores(t, rows, states, co, part, F, cats):
    """Training-only queries (FITQ) at boundary t: distances to O0, B0 and every fitted arm."""
    res = {}
    for c in cats:
        q = F.images(S.fitq(co, part, c)).reshape(-1, S.DIM).clone()
        F.drop(c)
        ids = rows[c]["ids"]
        d0, _ = S.score(q, rows[c]["x"], ids)
        r = {"O0": d0.numpy(), "B0": S.score(q, rows[c]["b0"], ids)[0].numpy()}
        for a, xs in states.items():
            r[a] = S.score(q, xs[c], ids)[0].numpy()
        res[c] = r
    return res


def do_fit(t):
    S.verify_lock()
    co, n = S.setup()
    part = D2.partition(co, n)
    log = S.Log(f"fit_t{t:02d}")
    F = C.Features()
    d = S.OUT / "fit" / f"t{t:02d}"
    d.mkdir(parents=True, exist_ok=True)
    rows, rec = load_rows(t, co, n, F)
    cats = list(rows)
    R = sum(len(v["ids"]) for v in rows.values())
    t_all = time.time()
    decoded, diag = {}, {}
    for arm in ARMS_FIT:
        p = d / f"{arm}.pt"
        if not p.exists():
            t0 = time.time()
            if arm.startswith("C0"):
                st, extra = S.build_c0(t, rows, arm_k(arm)), {}
            elif arm.startswith("N0"):
                st, extra = S.train_n0(t, rows, arm_k(arm), log=log)
            else:
                st, sec = S.build_p0(t, rows)
                extra = {"per_category_fit_seconds": sec}
            fit_s = time.time() - t0
            part_ = p.with_suffix(".part")
            torch.save(st, part_)
            part_.replace(p)
            S.write_json(d / f"{arm}.json", {"arm": arm, "t": t, "k": arm_k(arm), "bytes": S.state_bytes(st),
                                            "rows": R, "categories": cats, "fit_seconds": fit_s, "sha256": S.sha(p),
                                            "peak_rss_gb": C.peak_rss_gb(), **extra, "utc": S.now()})
            log("fit", arm, "t", t, "bytes", S.state_bytes(st), f"{fit_s:.0f}s")
        meta = S.read_json(d / f"{arm}.json")
        require(S.sha(p) == meta["sha256"], f"state changed {p}")
        st = torch.load(p, weights_only=False)                    # reload: scores come from the serialized state
        t1 = time.time()
        decoded[arm] = S.DECODERS[arm[:2]](st, cats)
        diag[arm] = {"decode_seconds": time.time() - t1, "bytes": S.state_bytes(st)}
        require(all(torch.isfinite(v).all() for v in decoded[arm].values()), f"non-finite decoded rows {arm}")
    # training-side reconstruction statistics on the bank rows
    recon = {}
    for arm in ("B0",) + ARMS_FIT:
        recon[arm] = {}
        for c in cats:
            xh = rows[c]["b0"] if arm == "B0" else decoded[arm][c]
            e = (xh - rows[c]["x"]).double().norm(dim=1)
            x0 = rows[c]["x"].double()
            recon[arm][c] = {"mean": float(e.mean()), "p95": float(torch.quantile(e, 0.95)),
                             "p99": float(torch.quantile(e, 0.99)), "max": float(e.max()),
                             "nmse": float(e.square().sum() / (x0 - x0.mean(0)).square().sum()),
                             "rel_mean": float((e / x0.norm(dim=1)).mean())}
    fq = fitq_scores(t, rows, decoded, co, part, F, cats)
    np.savez_compressed(d / "FITQ_DISTANCES.npz", **{f"{c}|{a}": v for c, r in fq.items() for a, v in r.items()})
    fstats = {a: {c: S.err_stats(fq[c][a], fq[c]["O0"]) for c in cats} for a in ("B0",) + ARMS_FIT}
    S.write_json(d / "FIT_SUMMARY.json", {
        "t": t, "rows": R, "categories": cats, "bank_bytes_B0": rec["persistent_bytes"], "decode": diag,
        "reconstruction_on_bank": recon, "fitq_score_error": fstats,
        "fitq_macro_p95": {a: float(np.mean([fstats[a][c]["p95_abs"] for c in cats])) for a in fstats},
        "fit_data": "O0 FP32 rows of the stored bank at this boundary (offline/static; not available to a continual "
                    "learner that keeps only PQ codes)", "seconds": time.time() - t_all, "peak_rss_gb": C.peak_rss_gb(),
        "utc": S.now()})
    log("FIT DONE t", t, {a: round(v, 4) for a, v in S.read_json(d / "FIT_SUMMARY.json")["fitq_macro_p95"].items()})


def do_select():
    S.verify_lock()
    target = S.OUT / "SELECTION.json"
    require(not target.exists(), "one-shot")
    require(not (S.OUT / "eval").exists(), "selection must precede every independent-probe evaluation")
    fs = S.read_json(S.OUT / "fit" / f"t{S.FINAL:02d}" / "FIT_SUMMARY.json")
    m = fs["fitq_macro_p95"]
    small = m["N0_C"] <= S.GATE["select_rule_frac"] * m["B0"]
    S.write_json(target, {"rule": "N0* = N0_C (0.75*B0 budget) if its training-only FITQ macro P95 error <= 1.03 x B0's; "
                                  "else N0_Q (B0 budget)", "fitq_macro_p95": m, "selected": "N0_C" if small else "N0_Q",
                          "fit_summary_sha256": S.sha(S.OUT / "fit" / f"t{S.FINAL:02d}" / "FIT_SUMMARY.json"),
                          "eval_outputs_existing": 0, "utc": S.now()})
    print("selected", "N0_C" if small else "N0_Q", m)


# ------------------------------------------------------------------ independent-probe evaluation

def do_eval(t):
    S.verify_lock()
    require((S.OUT / "SELECTION.json").exists(), "N0 selection must exist before probe evaluation")
    co, n = S.setup()
    log = S.Log(f"eval_t{t:02d}")
    F = C.Features()
    d = S.OUT / "eval" / f"t{t:02d}"
    d.mkdir(parents=True, exist_ok=True)
    rows, rec = load_rows(t, co, n, F)
    cats = list(rows)
    fd = S.OUT / "fit" / f"t{t:02d}"
    supp = {"O0": {c: rows[c]["x"] for c in cats}, "B0": {c: rows[c]["b0"] for c in cats}}
    for arm in ARMS_FIT:
        meta = S.read_json(fd / f"{arm}.json")
        require(S.sha(fd / f"{arm}.pt") == meta["sha256"], f"state changed {arm}")
        supp[arm] = S.DECODERS[arm[:2]](torch.load(fd / f"{arm}.pt", weights_only=False), cats)
    sand = S.ROOT / "results/hope_cad/sand_final/dist" / S.MEM_ARM
    for c in cats:
        p = d / f"{c}.npz"
        if p.exists():
            continue
        cohorts = {"EVAL": co[c]["eval"], "SEL": co[c]["sel"]}
        qs = {k: F.images(v).reshape(-1, S.DIM).clone() for k, v in cohorts.items()}
        F.drop(c)
        if c in ("bottle", "carpet", "hazelnut"):
            qs["HELD"] = S.heldout_rows(c)
        out, lat = {}, {}
        ids = rows[c]["ids"]
        for k, q in qs.items():
            for arm, xs in supp.items():
                rs = X.RowSet(xs[c], ids)                    # fixed support set, built before any query
                if k == "EVAL":
                    dd, ii, tt = [], [], []
                    for img in q.split(S.PATCHES):
                        t0 = time.perf_counter()
                        a, b = X.nn(img, rs)
                        tt.append(time.perf_counter() - t0); dd.append(a); ii.append(b)
                    dist, nid = torch.cat(dd), torch.cat(ii)
                    lat[arm] = tt
                else:
                    dist, nid = X.nn(q, rs)
                require(bool(torch.isfinite(dist).all()), f"non-finite distances {arm} {c} {k}")
                out[f"{k}|{arm}|d"] = dist.numpy()
                out[f"{k}|{arm}|id"] = nid.numpy()
        # B0 on the gate cohort must equal the SAND-stored distances to the same bank (only the 'eval' member is read)
        sp = sand / f"{c}_{rec['per_category_gids_sha'][c][:24]}.npz"
        if sp.exists():
            with np.load(sp) as z:
                ref = z["eval"].reshape(-1)
            diff = float(np.abs(out["EVAL|B0|d"] - ref).max())
            require(diff <= 1e-12, f"B0 EVAL distances differ from the SAND record {c}: {diff}")
            out["B0_vs_sand_eval_max_abs_diff"] = np.array(diff)
        for arm, tt in lat.items():
            out[f"latency|{arm}"] = np.asarray(tt)
        part = d / f"{c}.part.npz"
        np.savez_compressed(part, **out)
        part.replace(p)
        e = {a: S.err_stats(out[f"EVAL|{a}|d"], out["EVAL|O0|d"])["p95_abs"] for a in supp if a != "O0"}
        log("eval t", t, c, {a: round(v, 4) for a, v in e.items()})
    S.write_json(d / "DONE.json", {"t": t, "categories": cats, "peak_rss_gb": C.peak_rss_gb(), "utc": S.now()})


# ------------------------------------------------------------------ final-boundary geometry

def do_geometry():
    S.verify_lock()
    target = S.OUT / "GEOMETRY_FINAL.json"
    require(not target.exists(), "one-shot")
    co, n = S.setup()
    F = C.Features()
    t = S.FINAL
    rows, _ = load_rows(t, co, n, F)
    cats = list(rows)
    fd = S.OUT / "fit" / f"t{t:02d}"
    supp = {"B0": {c: rows[c]["b0"] for c in cats}}
    for arm in ARMS_FIT:
        supp[arm] = S.DECODERS[arm[:2]](torch.load(fd / f"{arm}.pt", weights_only=False), cats)
    out = {}
    for c in cats:
        x = rows[c]["x"]
        ids = torch.arange(len(x), dtype=torch.int64)
        # rarity of each stored row: exact distance to its nearest other stored row (FP32 originals)
        rar = X.nn(x, X.RowSet(x, ids), exclude=ids)[0]
        rare = rar >= torch.quantile(rar, 0.9)
        ev = np.load(S.OUT / "eval" / f"t{t:02d}" / f"{c}.npz")
        d0 = ev["EVAL|O0|d"]
        out[c] = {"rare_threshold_nn_dist": float(torch.quantile(rar, 0.9))}
        for arm, xs in supp.items():
            pe = (xs[c] - x).double().norm(dim=1)
            h, h_ab, h_ba = S.hausdorff(x, xs[c])
            e = np.abs(ev[f"EVAL|{arm}|d"] - d0)
            out[c][arm] = {"paired_max": float(pe.max()), "hausdorff": h, "hausdorff_O0_to_arm": h_ab,
                           "hausdorff_arm_to_O0": h_ba, "max_abs_score_err_eval": float(e.max()),
                           "theorem_holds": bool(e.max() <= h + 1e-9 and h <= float(pe.max()) + 1e-9),
                           "recon_rare_mean": float(pe[rare].mean()), "recon_common_mean": float(pe[~rare].mean()),
                           "recon_rare_p95": float(torch.quantile(pe[rare], 0.95))}
            require(out[c][arm]["theorem_holds"], f"score/Hausdorff bound violated {arm} {c}")
        print("geometry", c, {a: round(v["hausdorff"], 3) for a, v in out[c].items() if isinstance(v, dict)}, flush=True)
    S.write_json(target, {"t": t, "categories": out, "utc": S.now(), "peak_rss_gb": C.peak_rss_gb()})


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "audit":
        do_audit()
    elif cmd == "fit":
        do_fit(int(sys.argv[2]))
    elif cmd == "select":
        do_select()
    elif cmd == "eval":
        do_eval(int(sys.argv[2]))
    elif cmd == "geometry":
        do_geometry()
    else:
        raise SystemExit(cmd)
