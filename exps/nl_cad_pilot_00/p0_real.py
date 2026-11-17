"""NL-CAD-PILOT-00 P0 — real-feature mechanical checks and cost measurements (normal train features only).

    python -m exps.nl_cad_pilot_00.p0_real stream      # real fast-level stream on canonical C2-DOM DOM_PQ_B2 banks
    python -m exps.nl_cad_pilot_00.p0_real t6          # gamma=0 bitwise recovery, union consistency, metric exactness
    python -m exps.nl_cad_pilot_00.p0_real bench       # inference latency CPU/GPU, FP32 parity, backbone, readout
    python -m exps.nl_cad_pilot_00.p0_real visa        # size-realistic VisA-train union bank latency
    python -m exps.nl_cad_pilot_00.p0_real metacost    # P5 fallback rates on real data; ES cost components
    python -m exps.nl_cad_pilot_00.p0_real ledger      # end-to-end persistent byte ledger (+ backbone)

Custody: a Python audit hook refuses any open() of a path containing a forbidden pattern (test features, test
directories, masks, ground truth, C2-DOM evaluation cells, dataset manifests listing test data) and records every
project file opened; loaders also log explicitly. Data read: CALM-02 MVTec/VisA ``*_train.pt`` (normal train images),
C2-DOM DOM_PQ_B2 stream checkpoints (built from MVTec train normals), the R15 identity protocol, the ViT-B/8 weights.
Canonical C2-DOM code and artifacts are imported/read only; nothing under the C2-DOM directory is written.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORBID = ("_test.pt", "/test/", "ground_truth", "/masks", "_mask", "/eval/", "DATASET_MANIFEST", "split_csv",
          "image_anno", "/Masks/", "Anomaly")
ACCESS = set()


def _audit(event, args):
    if event != "open" or not args:
        return
    p = args[0]
    if not isinstance(p, (str, bytes, os.PathLike)):
        return
    s = os.path.abspath(os.fsdecode(p))
    if not (s.startswith(str(ROOT)) or s.startswith("/mnt/")) or s.endswith((".py", ".pyc", ".so")):
        return                                     # libraries / system files are outside the custody scope
    if any(f in s for f in FORBID):
        raise PermissionError(f"P0 custody violation: {s}")
    if s.startswith(str(ROOT)) and not s.endswith((".py", ".pyc")):
        ACCESS.add(s)


sys.addaudithook(_audit)

import numpy as np                                                                       # noqa: E402
import torch                                                                             # noqa: E402
import torch.nn.functional as F                                                          # noqa: E402

from exps.csg01.select import random_priorities                                         # noqa: E402
from exps.nl_sgm_01a import am02_run as R2                                              # noqa: E402
from exps.nl_sgm_01a import common as C                                                 # noqa: E402
from exps.nl_sgm_01a import exact as X                                                  # noqa: E402
from exps.nl_sgm_01a_c2dom import common as D                                           # noqa: E402
from exps.nl_sgm_01a_c2dom import stream as CS                                          # noqa: E402
from exps.nl_cad_pilot_00 import es as E                                                # noqa: E402
from exps.nl_cad_pilot_00 import ledger as L                                            # noqa: E402
from exps.nl_cad_pilot_00 import nnx                                                    # noqa: E402
from exps.nl_cad_pilot_00 import scrc as S                                              # noqa: E402

OUT = ROOT / "results/hope_cad/nl_cad_pilot_00/p0"
ARM = "DOM_PQ_B2"
SEED_RES = int.from_bytes(__import__("hashlib").sha256(b"NL-CAD-PILOT-00/residual-sample/v1").digest()[:4], "big")
MVTEC_HW = {"bottle": (900, 900), "cable": (1024, 1024), "capsule": (1000, 1000), "carpet": (1024, 1024),
            "grid": (1024, 1024), "hazelnut": (1024, 1024), "leather": (1024, 1024), "metal_nut": (700, 700),
            "pill": (800, 800), "screw": (1024, 1024), "tile": (840, 840), "toothbrush": (1024, 1024),
            "transistor": (1024, 1024), "wood": (1024, 1024), "zipper": (1024, 1024)}
LOADS = []
CUDA = torch.cuda.is_available()


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    obj = dict(obj)
    obj["_access"] = {"explicit_loads": sorted(set(LOADS)), "audited_project_opens": sorted(ACCESS)}
    obj["_env"] = env()
    p = OUT / name
    C.write_json(p, obj)                                # create-exclusive
    log("wrote", p)


def env():
    e = {"torch": torch.__version__, "threads": torch.get_num_threads(), "cuda": CUDA,
         "tf32": bool(torch.backends.cuda.matmul.allow_tf32), "utc": C.now()}
    if CUDA:
        e["gpu"] = torch.cuda.get_device_name(0)
    return e


def sync():
    if CUDA:
        torch.cuda.synchronize()


def timed(fn, reps=5, warmup=1, cuda=False):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        if cuda:
            sync()
        t0 = time.perf_counter()
        out = fn()
        if cuda:
            sync()
        ts.append(time.perf_counter() - t0)
    return {"median_s": float(np.median(ts)), "min_s": float(min(ts)), "max_s": float(max(ts)), "reps": reps,
            "raw_s": ts}, out


# ------------------------------------------------------------------ canonical banks and features

def canonical_bank(t):
    """Union (task-agnostic) RowSet of the canonical C2-DOM DOM_PQ_B2 bank after 0-based boundary t (read-only)."""
    sdir = D.OUT / "streams" / ARM
    rec = D.read_json(sdir / f"boundary_t{t:02d}.json")
    p = sdir / f"ckpt_t{t:02d}.pt"
    LOADS.append(str(p))
    S.require(C.sha(p) == rec["ckpt_sha256"], f"canonical checkpoint hash mismatch t={t}")
    sup = CS.supports_at(ARM, t)
    cats = [c for c in D.ORDER if c in sup]
    x = torch.cat([sup[c].x for c in cats]).contiguous()
    ids = torch.cat([sup[c].ids for c in cats])
    S.require(bool((ids[1:] > ids[:-1]).all()), "union ids not increasing")
    S.require(len(ids) == rec["rows"], "row count differs from ledger")
    return X.RowSet(x, ids), sup, rec


class Feat:
    def __init__(self):
        self.F = C.Features()
        self.co, self.n = D.setup()

    def images(self, c, paths):
        LOADS.append(str(C.FEATURES / f"{c}_train.pt"))
        return self.F.images(paths)

    def residual_sample(self, c, m=S.M_RES):
        """Hash-first m patches of category c's writer images; returns (x FP32 [m, d], gids, image keys)."""
        w = self.co[c]["writers"]
        base = R2.gid_base(self.n, c)
        g = torch.arange(base, base + len(w) * C.PATCHES)
        pr = random_priorities(SEED_RES, g.numpy().astype(np.int64))
        g = g[torch.as_tensor(np.lexsort((g.numpy(), pr)))[:m]]
        j = (g - base) // C.PATCHES
        p = (g - base) % C.PATCHES
        obj = self.F._load(c)
        LOADS.append(str(C.FEATURES / f"{c}_train.pt"))
        fi = torch.tensor([self.F._index[c][C.guard_path(w[int(k)])] for k in j])
        x = obj["features"][fi, p].clone()
        return x, g, g // C.PATCHES


def visa_rows(c):
    p = ROOT / "results/hope_cad/calm02/features/visa" / f"{c}_train.pt"
    S.require(p.name.endswith("_train.pt"), "custody")
    LOADS.append(str(p))
    return torch.load(p, map_location="cpu", weights_only=False, mmap=True)


# ------------------------------------------------------------------ stream: real fast level on canonical banks

def residuals_all(fe, device="cpu"):
    rdir = OUT / "residuals"
    rdir.mkdir(parents=True, exist_ok=True)
    out, recs = [], []
    for t, c in enumerate(D.ORDER):
        p = rdir / f"R_t{t:02d}.pt"
        if p.exists():
            out.append(torch.load(p)["R"]); continue
        bank, _, brec = canonical_bank(t)
        q, g, qk = fe.residual_sample(c)
        rows = nnx.Rows.from_rowset(bank, device)
        st = {}
        t0 = time.perf_counter()
        d, i = nnx.certified_nn(q, rows, None, q_key=qk, row_key=bank.ids.to(device) // C.PATCHES, stats=st)
        sync()
        nn_s = time.perf_counter() - t0
        d, i = d.cpu(), i.cpu()
        S.require(bool((i >= 0).all()), "residual without neighbour")
        pos = torch.searchsorted(bank.ids, i)
        S.require(torch.equal(bank.ids[pos], i), "id lookup")
        R = q.double() - bank.x[pos].double()
        S.require(bool((d - R.norm(dim=1)).abs().max() <= 1e-9 * d.max()), "residual norm != NN distance")
        own = float(((i // C.PATCHES) != qk).float().mean())
        same_cat = float((pos >= 0).float().mean())
        torch.save({"R": R, "gids": g, "nn_ids": i, "dist": d}, p)
        rec = {"t": t, "category": c, "bank_rows": int(len(bank.ids)), "m": int(len(q)), "nn_seconds": nn_s,
               "device": device, "candidates_per_query": st["candidates"] / st["queries"],
               "leave_image_out_ok": own == 1.0, "median_residual_norm": float(d.median()),
               "trace_C": float(R.square().sum() / len(R))}
        recs.append(rec)
        log("residual", t, c, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
        out.append(R)
        del bank, rows
    return out, recs


def do_stream():
    torch.backends.cuda.matmul.allow_tf32 = False
    fe = Feat()
    t0 = time.time()
    R, rrecs = residuals_all(fe)
    log("residuals done", time.time() - t0)
    Cd = [r.T @ r / len(r) for r in R]
    res = {"schema": "nl_cad_p0_stream_v1", "residual_records": rrecs, "m": S.M_RES, "seed_res": SEED_RES,
           "policies": {}}
    pols = {"cumulative_A1A2": S.Cumulative(), "latest_A3": S.Latest(), "theta0_A5": S.Learned([0, 0, 0, 0]),
            "learned_demo": S.Learned([-1.0, 1.5, -2.0, 0.0])}
    for name, pol in pols.items():
        st = S.FastState.empty()
        Kex = torch.zeros(768, 768, dtype=torch.float64)
        bound, cmax = 0.0, 0.0
        recs = []
        Ufirst = None
        for t in range(1, 16):
            Rt = R[t - 1]
            f = S.novelty_features(st, Rt, t) if t >= 2 else {"nu": float("nan"), "a": float("nan"), "h": None}
            rho = pol(t, f["h"])
            U_prev, _ = S.factors64(st)
            ta = time.perf_counter()
            new, dg = S.prox_update(st, Rt, rho)
            svd_s = time.perf_counter() - ta
            U, lam = S.factors64(new)
            if Ufirst is None:
                Ufirst = U.clone()
            Kex = (Cd[t - 1] + rho * Kex) / (1 + rho)
            bound = rho / (1 + rho) * bound + dg["lam_r1"]
            cmax = max(cmax, float(torch.linalg.matrix_norm(Cd[t - 1], 2)))
            Kt = (U * lam[None]) @ U.T
            evC = torch.linalg.eigvalsh(Cd[t - 1]).flip(0)
            trC = float(evC.sum())
            met = S.make_metric(new, 0.0, 1.0)
            rec = {"t": t, "category": D.ORDER[t - 1], "rho": rho, "nu": f["nu"], "a": f["a"], "h": f["h"],
                   "svd_seconds": svd_s, "lam_1": dg["lam_1"], "lam_32": dg["lam_r"], "lam_33": dg["lam_r1"],
                   "rel_gap_32": dg["rel_gap_r"], "kept_trace_fraction_of_tilde": dg["trace_kept"] / dg["trace_tilde"],
                   "C_top32_energy_fraction": float(evC[:32].sum()) / trC,
                   "C_energy_in_prev_subspace": float(torch.trace(U_prev.T @ Cd[t - 1] @ U_prev)) / trC if U_prev.shape[1] else 0.0,
                   "C_energy_in_new_subspace": float(torch.trace(U.T @ Cd[t - 1] @ U)) / trC,
                   "first_task_energy_retained": float(torch.trace(U.T @ Cd[0] @ U) / torch.trace(Ufirst.T @ Cd[0] @ Ufirst)),
                   "P1_dev": float(torch.linalg.matrix_norm(Kt - Kex, 2)), "P1_bound": bound,
                   "P1_Kex_norm": float(torch.linalg.matrix_norm(Kex, 2)), "P1_cmax": cmax,
                   "P1_dev_rel_Kex_fro": float((Kt - Kex).norm() / Kex.norm()),
                   "mu_t_theta0": met.mu_t, "kappa_t_theta0": met.kappa_t, "kappa_theta0_bound": met.kappa_theta,
                   "tau_theta0": met.tau}
            S.require(rec["P1_dev"] <= bound * (1 + 1e-6) + 1e-9 * cmax, f"P1 bound violated {name} t={t}")
            S.require(rec["P1_Kex_norm"] <= cmax * (1 + 1e-9), "P1 norm bound violated")
            recs.append(rec)
            st = new
            if name == "cumulative_A1A2":
                torch.save(st.tensors(), OUT / "residuals" / f"state_cumulative_t{t:02d}.pt")
        res["policies"][name] = recs
        log(name, "final rho", recs[-1]["rho"], "kappa_t", round(recs[-1]["kappa_t_theta0"], 3),
            "P1 dev/bound", round(recs[-1]["P1_dev"], 4), round(recs[-1]["P1_bound"], 4),
            "first-task retained", round(recs[-1]["first_task_energy_retained"], 3))
    save("STREAM.json", res)


def load_state(t):
    o = torch.load(OUT / "residuals" / f"state_cumulative_t{t:02d}.pt")
    return S.FastState(o["U"], o["lam"], int(o["t"][0]))


def probe_queries(fe, cats=None, per=6):
    out = {}
    for c in (cats or D.ORDER):
        paths = fe.co[c]["eval"][:per]
        out[c] = fe.images(c, paths).reshape(-1, 768).clone()
    return out


# ------------------------------------------------------------------ T6 and metric exactness on real banks

def do_t6():
    torch.backends.cuda.matmul.allow_tf32 = False
    fe = Feat()
    bank, sup, brec = canonical_bank(14)
    st = load_state(15)
    qs = probe_queries(fe)
    res = {"schema": "nl_cad_p0_t6_v1", "bank_rows": int(len(bank.ids)), "bank_persistent_bytes": brec["persistent_bytes"],
           "queries_images": {c: 6 for c in qs}, "categories": {}}
    met0 = S.make_metric(st, 0.0, 0.0)
    S.require(met0.is_identity, "gamma=0 metric is not the identity")
    rows0 = nnx.Rows.transformed(bank.x, bank.ids, met0)
    res["transformed_rows_equal_raw_bitwise"] = bool(torch.equal(rows0.z, bank.x))
    all_ok = True
    for c, q in qs.items():
        s_can, s_scrc = {}, {}
        ta = time.perf_counter()
        d0, i0 = X.nn(q, bank, stats=s_can)
        tc = time.perf_counter() - ta
        ta = time.perf_counter()
        d1, i1 = nnx.certified_nn(q, rows0, met0, stats=s_scrc)
        ts = time.perf_counter() - ta
        # union = min over per-category canonical scorers (lexicographic merge)
        dm, im = None, None
        for k, rs in sup.items():
            dk, ik = X.nn(q, rs)
            dm, im = (dk, ik) if dm is None else X.merge(dm, im, dk, ik)
        hw = MVTEC_HW[c]
        m0 = F.interpolate(d0.float().reshape(-1, 1, 28, 28), size=hw, mode="bilinear", align_corners=False)
        m1 = F.interpolate(d1.float().reshape(-1, 1, 28, 28), size=hw, mode="bilinear", align_corners=False)
        own = torch.bucketize(i0, torch.tensor([R2.gid_base(fe.n, k) for k in D.ORDER[1:]]), right=True)
        rec = {"dist_bitwise": bool(torch.equal(d0, d1)), "ids_equal": bool(torch.equal(i0, i1)),
               "candidates_equal": s_can["candidates"] == s_scrc["candidates"], "candidates": s_can["candidates"],
               "union_equals_merge_of_per_category": bool(torch.equal(d0, dm) and torch.equal(i0, im)),
               "native_maps_bitwise": bool(torch.equal(m0, m1)),
               "image_scores_bitwise": bool(torch.equal(m0.flatten(1).max(1).values, m1.flatten(1).max(1).values)),
               "canonical_seconds": tc, "scrc_gamma0_seconds": ts,
               "nn_in_own_category_fraction": float((own == D.ORDER.index(c)).float().mean())}
        ok = all(rec[k] for k in ("dist_bitwise", "ids_equal", "candidates_equal", "union_equals_merge_of_per_category",
                                  "native_maps_bitwise", "image_scores_bitwise"))
        all_ok &= ok
        res["categories"][c] = rec
        log("T6", c, ok, round(tc, 2), round(ts, 2), "own-cat", round(rec["nn_in_own_category_fraction"], 4))
    res["T6_pass"] = bool(all_ok and res["transformed_rows_equal_raw_bitwise"])
    # metric (gamma = 1, theta_tau = 0) exactness on real data vs an independent all-pairs FP64 oracle (GPU or CPU)
    met = S.make_metric(st, 0.0, 1.0)
    rows = nnx.Rows.transformed(bank.x, bank.ids, met)
    mx = {}
    for c in ("bottle", "screw", "zipper"):
        q = qs[c][:2 * 784]
        stt = {}
        d1, i1 = nnx.certified_nn(q, rows, met, stats=stt)
        dev = "cuda" if CUDA else "cpu"
        z64 = S.apply_sqrt(bank.x.to(dev), met)
        zq = S.apply_sqrt(q.to(dev), met)
        best_d = torch.full((len(q),), math.inf, dtype=torch.float64, device=dev)
        best_i = torch.full((len(q),), -1, dtype=torch.int64, device=dev)
        for a in range(0, len(q), 64):
            Dm = torch.cdist(zq[a:a + 64], z64, compute_mode="donot_use_mm_for_euclid_dist")
            v, j = Dm.min(1)
            best_d[a:a + 64], best_i[a:a + 64] = v, bank.ids.to(dev)[j]
        best_d, best_i = best_d.cpu(), best_i.cpu()
        mism = (best_i != i1)
        # for any id mismatch, both rows must be reference-ties within the oracle's rounding
        tie_ok = True
        if bool(mism.any()):
            pos = torch.searchsorted(bank.ids, best_i[mism])
            dalt = S.metric_dist_pairs(q[mism], bank.x[pos], met)
            tie_ok = bool((dalt >= d1[mism]).all() and ((dalt - d1[mism]).abs() <= 1e-12 * d1[mism]).all())
        mx[c] = {"queries": int(len(q)), "id_mismatch": int(mism.sum()), "mismatches_are_ties": tie_ok,
                 "max_rel_dist_diff_vs_oracle": float(((best_d - d1).abs() / d1).max()),
                 "candidates_per_query": stt["candidates"] / stt["queries"],
                 "max_candidates_per_query": stt["max_candidates_per_query"],
                 "euclid_vs_metric_nn_changed_fraction": None}
        dE, iE = X.nn(q, bank)
        mx[c]["euclid_vs_metric_nn_changed_fraction"] = float((iE != i1).float().mean())
        mx[c]["sandwich_ok"] = bool(((d1 <= dE * (1 + 1e-12)) & (d1 >= math.sqrt(met.mu_t) * dE * (1 - 1e-12))).all())
        log("metric exactness", c, json.dumps(mx[c]))
    res["metric_gamma1_exactness"] = mx
    res["metric_gamma1"] = {"mu_t": met.mu_t, "kappa_t": met.kappa_t, "kappa_theta": met.kappa_theta, "tau": met.tau}
    save("T6_REAL.json", res)
    S.require(res["T6_pass"], "T6 FAILED: gamma=0 score path differs from the canonical scorer")


# ------------------------------------------------------------------ latency

def fp32_fast_nn(q, z, n2):
    T = torch.addmm(n2[None].expand(len(q), -1), q, z.T, beta=1.0, alpha=-2.0)
    v, j = T.min(1)
    return (v.double() + q.double().square().sum(1)).clamp_min(0).sqrt(), j


def bench_bank(bank, st, qimgs, tag):
    """Inference latency on one union bank; qimgs: list of [784, d] query images."""
    out = {"bank_rows": int(len(bank.ids)), "tag": tag}
    met = S.make_metric(st, 0.0, 1.0)
    met0 = S.make_metric(st, 0.0, 0.0)
    q1 = qimgs[0]
    q8 = torch.cat(qimgs[:8])
    # CPU
    tm, _ = timed(lambda: nnx.Rows.transformed(bank.x, bank.ids, met), reps=3)
    out["cpu_bank_transform_per_boundary"] = tm
    rows_m = nnx.Rows.transformed(bank.x, bank.ids, met)
    rows_e = nnx.Rows.from_rowset(bank)
    out["cpu_canonical_exact_nn_1img"], ref = timed(lambda: X.nn(q1, bank), reps=5)
    out["cpu_scrc_gamma0_1img"], _ = timed(lambda: nnx.certified_nn(q1, rows_e, None), reps=5)
    out["cpu_scrc_gamma1_1img"], _ = timed(lambda: nnx.certified_nn(q1, rows_m, met), reps=5)
    out["cpu_scrc_gamma1_8img"], _ = timed(lambda: nnx.certified_nn(q8, rows_m, met), reps=3)
    out["cpu_canonical_exact_nn_8img"], ref8 = timed(lambda: X.nn(q8, bank), reps=3)
    if CUDA:
        dev = "cuda"
        tm, _ = timed(lambda: nnx.Rows.transformed(bank.x, bank.ids, met, device=dev), reps=5, cuda=True)
        out["gpu_bank_transform_per_boundary"] = tm
        gm = nnx.Rows.transformed(bank.x, bank.ids, met, device=dev)
        ge = nnx.Rows.from_rowset(bank, dev)
        out["gpu_scrc_gamma0_1img"], g0 = timed(lambda: nnx.certified_nn(q1, ge, None), reps=20, warmup=3, cuda=True)
        out["gpu_scrc_gamma1_1img"], g1 = timed(lambda: nnx.certified_nn(q1, gm, met), reps=20, warmup=3, cuda=True)
        out["gpu_scrc_gamma0_8img"], g08 = timed(lambda: nnx.certified_nn(q8, ge, None), reps=10, warmup=2, cuda=True)
        out["gpu_scrc_gamma1_8img"], _ = timed(lambda: nnx.certified_nn(q8, gm, met), reps=10, warmup=2, cuda=True)
        # parity of the certified GPU path vs the canonical CPU oracle (gamma = 0)
        out["gpu_certified_vs_cpu_oracle_8img"] = {
            "ids_equal": bool(torch.equal(g08[1].cpu(), ref8[1])),
            "max_rel_dist_diff": float(((g08[0].cpu() - ref8[0]).abs() / ref8[0]).max()),
            "dist_bitwise": bool(torch.equal(g08[0].cpu(), ref8[0]))}
        # uncertified FP32 GEMM path: parity only (never used as the scorer)
        q8d = q8.to(dev)
        out["gpu_fp32_uncertified_8img"], fz = timed(lambda: fp32_fast_nn(q8d, ge.z, ge.n2_32), reps=10, warmup=2, cuda=True)
        fd, fj = fz[0].cpu(), ge.ids[fz[1]].cpu()
        out["fp32_uncertified_vs_fp64_oracle_8img"] = {
            "id_mismatch_fraction": float((fj != ref8[1]).float().mean()),
            "max_rel_dist_err": float(((fd - ref8[0]).abs() / ref8[0]).max()),
            "median_rel_dist_err": float(((fd - ref8[0]).abs() / ref8[0]).median())}
        sc = {}
        nnx.certified_nn(q8, gm, met, stats=sc)
        out["gpu_gamma1_candidates_per_query"] = sc["candidates"] / sc["queries"]
        out["gpu_peak_mem_bytes"] = int(torch.cuda.max_memory_allocated())
        del gm, ge
        torch.cuda.empty_cache()
    sc = {}
    nnx.certified_nn(q8, rows_m, met, stats=sc)
    out["cpu_gamma1_candidates_per_query"] = sc["candidates"] / sc["queries"]
    sc = {}
    X.nn(q8, bank, stats=sc)
    out["cpu_gamma0_candidates_per_query"] = sc["candidates"] / sc["queries"]
    return out


def bench_backbone():
    import timm
    ck = ROOT / "checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth"
    LOADS.append(str(ck))
    sd = torch.load(ck, map_location="cpu", weights_only=True)
    model = timm.create_model("vit_base_patch8_224", pretrained=False, num_classes=0)
    miss = model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("head")}, strict=False)
    n_par = sum(p.numel() for p in model.parameters())
    n_par9 = sum(p.numel() for p in model.patch_embed.parameters()) + model.pos_embed.numel() + model.cls_token.numel() \
        + sum(p.numel() for b in model.blocks[:9] for p in b.parameters())
    out = {"params_total": int(n_par), "params_fp32_bytes": int(n_par * 4), "params_used_through_block9": int(n_par9),
           "missing_keys": len(miss.missing_keys), "unexpected_keys": len(miss.unexpected_keys),
           "checkpoint_file_bytes": ck.stat().st_size}
    model.eval()

    def trunk(x):
        x = model.patch_embed(x)
        x = model._pos_embed(x)
        x = model.norm_pre(x)
        for b in model.blocks[:9]:
            x = b(x)
        return x[:, 1:]

    img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        S.require(trunk(img).shape == (1, 784, 768), "trunk output shape")
        out["cpu_trunk_1img"], _ = timed(lambda: trunk(img), reps=5)
        if CUDA:
            model.cuda()
            ig = img.cuda()
            out["gpu_trunk_1img_fp32"], _ = timed(lambda: trunk(ig), reps=50, warmup=10, cuda=True)
            i8 = torch.randn(8, 3, 224, 224, device="cuda")
            out["gpu_trunk_8img_fp32"], _ = timed(lambda: trunk(i8), reps=20, warmup=5, cuda=True)
            model.cpu()
            torch.cuda.empty_cache()
    return out


def bench_readout():
    d = torch.rand(1, 1, 28, 28)
    out = {}
    for hw in ((1024, 1024), (900, 900)):
        f = lambda: F.interpolate(d, size=hw, mode="bilinear", align_corners=False).flatten(1).max(1).values
        out[f"cpu_{hw[0]}"], _ = timed(f, reps=20, warmup=3)
        if CUDA:
            dg = d.cuda()
            g = lambda: F.interpolate(dg, size=hw, mode="bilinear", align_corners=False).flatten(1).max(1).values
            out[f"gpu_{hw[0]}"], _ = timed(g, reps=50, warmup=5, cuda=True)
    return out


def do_bench():
    torch.backends.cuda.matmul.allow_tf32 = False
    fe = Feat()
    bank, _, brec = canonical_bank(14)
    st = load_state(15)
    qs = probe_queries(fe, per=1)
    qimgs = [qs[c] for c in D.ORDER][:8]
    res = {"schema": "nl_cad_p0_bench_v1", "mvtec_final_union_bank": bench_bank(bank, st, qimgs, "C2-DOM DOM_PQ_B2 t=14"),
           "backbone": bench_backbone(), "readout": bench_readout()}
    log(json.dumps({k: v for k, v in res["mvtec_final_union_bank"].items() if "median" in str(v)})[:2000])
    save("BENCH_MVTEC.json", res)


def do_visa():
    """Size-realistic target-side union bank from VisA train normals (NOT a C2-DOM bank): the B2 DOM-PQ capacity after
    12 tasks with the SCRC overhead charged (105,359 - 975 rows), hash-first rows per category; queries are VisA train
    images excluded from the bank."""
    torch.backends.cuda.matmul.allow_tf32 = False
    from exps.calm02 import data as CD
    N = L.bank_capacity(L.B2, 11, L.SCRC_OVERHEAD)
    per = [N // 12 + (1 if k < N % 12 else 0) for k in range(12)]
    xs, ids, qimgs = [], [], []
    gid = 0
    t0 = time.time()
    for k, c in enumerate(CD.VISA_ORDER):
        o = visa_rows(c)
        f = o["features"]
        n_img = f.shape[0]
        key = sorted(range(n_img), key=lambda i: C.text_sha(f"NL-CAD-PILOT-00/visa-q/{c}/{o['rows'][i]['relative_path']}"))
        qi, bi = key[:1], key[1:]
        qimgs.append(f[qi[0]].clone())
        g = torch.arange(len(bi) * 784)
        pr = random_priorities(SEED_RES + 1, g.numpy().astype(np.int64))
        g = g[torch.as_tensor(np.lexsort((g.numpy(), pr)))[:per[k]]]
        rows = f[torch.tensor(bi)[g // 784], g % 784].clone()
        xs.append(rows)
        ids.append(torch.arange(gid, gid + len(rows)))
        gid += len(rows)
        log("visa rows", c, len(rows), round(time.time() - t0, 1))
    bank = X.RowSet(torch.cat(xs).contiguous(), torch.cat(ids))
    st = load_state(15)
    res = {"schema": "nl_cad_p0_visa_bench_v1", "bank_rows": int(N), "build_seconds": time.time() - t0,
           "note": "size-realistic VisA-train union bank, not a C2-DOM bank; MVTec-stream K_15 used for the metric",
           "bench": bench_bank(bank, st, qimgs, "VisA-train size-realistic")}
    save("BENCH_VISA.json", res)


# ------------------------------------------------------------------ offline meta-training cost components

def kth_certified(q, rows, k, q_chunk=256):
    """Exact (1st, k-th) Euclidean reference distances on rows' device (screen + band + FP64 re-evaluation)."""
    dev = rows.z.device
    q = q.to(dev)
    d1 = torch.empty(len(q), dtype=torch.float64, device=dev)
    dk = torch.empty(len(q), dtype=torch.float64, device=dev)
    ncand = 0
    for a in range(0, len(q), q_chunk):
        Q = q[a:a + q_chunk]
        qn = Q.double().square().sum(1).sqrt()
        band = ((X.screen_band(qn, rows.R) + 4 * X.U32 * (rows.R ** 2 + 2 * qn * rows.R)) * (1 + 1e-6)).float()
        T = torch.addmm(rows.n2_32[None].expand(len(Q), -1), Q, rows.z.T, beta=1.0, alpha=-2.0)
        tk = T.topk(k, dim=1, largest=False).values[:, -1]
        qi, ri = (T <= (tk + band)[:, None]).nonzero(as_tuple=True)
        ncand += len(qi)
        dd = (Q[qi].double() - rows.z[ri].double()).square_().sum(1).sqrt_()
        o = torch.argsort(dd, stable=True)
        o = o[torch.argsort(qi[o], stable=True)]
        cnt = torch.bincount(qi, minlength=len(Q))
        S.require(bool((cnt >= k).all()), "fewer than k candidates")
        start = torch.cumsum(cnt, 0) - cnt
        d1[a:a + len(Q)] = dd[o][start]
        dk[a:a + len(Q)] = dd[o][start + k - 1]
    return d1, dk, ncand / len(q)


def do_metacost():
    torch.backends.cuda.matmul.allow_tf32 = False
    fe = Feat()
    dev = "cuda" if CUDA else "cpu"
    res = {"schema": "nl_cad_p0_metacost_v1"}
    # (1) P5 fallback rate on real data (bank after 6 tasks = the source-episode bank size; and the final bank)
    stream = D.read_json(OUT / "STREAM.json")
    for t_b in (5, 14):
        bank, _, _ = canonical_bank(t_b)
        cats = D.ORDER[:t_b + 1]
        qs = probe_queries(fe, cats=cats, per=2)
        q = torch.cat([qs[c] for c in cats])
        rows = nnx.Rows.from_rowset(bank, dev)
        t0 = time.perf_counter()
        d1, d64, nc = kth_certified(q, rows, 64)
        sync()
        el = time.perf_counter() - t0
        d1, d64 = d1.cpu(), d64.cpu()
        sub = slice(0, 256)
        k_or = X.kth(q[sub], bank, 64)
        n_or, _ = X.nn(q[sub], bank)
        rec = {"bank_rows": int(len(bank.ids)), "queries": int(len(q)), "seconds": el, "cand_per_query": nc,
               "oracle_parity_256_max_rel": float(max(((k_or - d64[sub]).abs() / k_or).max(), ((n_or - d1[sub]).abs() / n_or).max())),
               "ratio_d64_over_d1_quantiles": {str(p): float(torch.quantile(d64 / d1, p)) for p in (0.01, 0.1, 0.5, 0.9, 0.99)}}
        fb = {}
        for gamma in (0.1, 0.25, 0.5, 1.0, 1.5, 2.0):
            for tht in (-2.0, 0.0, 2.0):
                mu_th = (1 + 32 * math.exp(-tht)) ** (-gamma)
                fb[f"gamma={gamma},theta_tau={tht}"] = {"mu_theta": mu_th,
                                                        "fallback_rate_at_mu_theta": float((d64 <= d1 / math.sqrt(mu_th)).float().mean())}
        # actual mu_t of the real cumulative stream at gamma = 1, theta_tau = 0
        mu_t = stream["policies"]["cumulative_A1A2"][t_b]["mu_t_theta0"]
        fb["real_cumulative_state_gamma1_theta0"] = {"mu_t": mu_t,
                                                     "fallback_rate": float((d64 <= d1 / math.sqrt(mu_t)).float().mean())}
        rec["fallback"] = fb
        res[f"p5_bank_t{t_b:02d}"] = rec
        log("P5", t_b, json.dumps(rec["ratio_d64_over_d1_quantiles"]), fb["real_cumulative_state_gamma1_theta0"])
        del rows
        if CUDA:
            torch.cuda.empty_cache()
    # (2) exact full metric search throughput at the source-episode bank size (6 tasks), GPU if present
    bank, _, _ = canonical_bank(5)
    st = load_state(6)
    met = S.make_metric(st, 0.0, 1.0)
    rows = nnx.Rows.transformed(bank.x, bank.ids, met, device=dev)
    qs = probe_queries(fe, per=6)
    q = torch.cat([qs[c] for c in D.ORDER[:2]])                              # 12 images = 9,408 query patches
    tm, _ = timed(lambda: nnx.certified_nn(q, rows, met), reps=3, warmup=1, cuda=CUDA)
    res["full_metric_search"] = {"device": dev, "bank_rows": int(len(bank.ids)), "queries": int(len(q)), **tm,
                                 "seconds_per_query": tm["median_s"] / len(q)}
    S.require(all(res[f"p5_bank_t{t:02d}"]["oracle_parity_256_max_rel"] <= 1e-13 for t in (5, 14)), "kth parity")
    # (2b) target online residual NN (leave-image-out, m = 4096) on the device, final bank; parity with the CPU run
    bank14, _, _ = canonical_bank(14)
    sv = torch.load(OUT / "residuals" / "R_t14.pt")
    xq, gq, kq = fe.residual_sample(D.ORDER[14])
    S.require(torch.equal(sv["gids"], gq), "residual sample not reproducible")
    rows14 = nnx.Rows.from_rowset(bank14, dev)
    tm, rr_ = timed(lambda: nnx.certified_nn(xq, rows14, None, q_key=kq, row_key=bank14.ids.to(dev) // C.PATCHES),
                    reps=3, warmup=1, cuda=CUDA)
    res["residual_nn_final_bank"] = {"device": dev, **tm, "ids_equal_cpu_run": bool(torch.equal(rr_[1].cpu(), sv["nn_ids"])),
                                     "max_rel_dist_diff_cpu_run": float(((rr_[0].cpu() - sv["dist"]).abs() / sv["dist"]).max())}
    del rows14
    # (3) fast-update cost per boundary (SVD) and ES bookkeeping are taken from STREAM.json
    svd = [r["svd_seconds"] for r in stream["policies"]["cumulative_A1A2"]]
    res["prox_update_seconds_median"] = float(np.median(svd))
    R14 = sv["R"]
    prev = load_state(14)
    tg, _ = timed(lambda: R14.T @ R14 / len(R14), reps=3)
    Cd = R14.T @ R14 / len(R14)
    ts_, a = timed(lambda: S.prox_update(prev, R14, 14.0), reps=3)
    te_, b = timed(lambda: S.prox_update_gram(prev, Cd, 14.0), reps=5)
    Ka, Kb = S.dense_K(a[0]), S.dense_K(b[0])
    res["prox_update_paths_real_t15"] = {"svd_path": ts_, "gram_eigh_path": te_, "gram_build_once": tg,
                                         "rel_diff_K": float((Ka - Kb).norm() / Ka.norm())}
    rr = stream["residual_record"] if "residual_record" in stream else stream["residual_records"]
    res["residual_nn_seconds_cpu"] = [r["nn_seconds"] for r in rr]
    save("METACOST.json", res)


def do_ledger():
    res = {"schema": "nl_cad_p0_ledger_v1", "B2_budget": L.B2, "scrc_overhead": L.SCRC_OVERHEAD,
           "fast_state_bytes": L.FAST_BYTES, "slow_state_bytes": L.SLOW_BYTES,
           "mvtec_15": L.ledger(15), "visa_12": L.ledger(12),
           "a6_history_bytes_12": L.a6_history_bytes(12),
           "transient_mvtec15": L.transient(L.ledger(15)["A4"]["bank_rows"]),
           "transient_visa12": L.transient(L.ledger(12)["A4"]["bank_rows"]),
           "per_boundary_A4_rows_mvtec": [L.bank_capacity(L.B2, t, L.SCRC_OVERHEAD) for t in range(15)],
           "per_boundary_A0_rows_mvtec": [L.bank_capacity(L.B2, t) for t in range(15)]}
    # canonical ledger cross-check against the stored C2-DOM stream records
    recs = [D.read_json(D.OUT / "streams" / ARM / f"boundary_t{t:02d}.json") for t in range(15)]
    res["canonical_crosscheck"] = all(r["persistent_bytes"] == L.bank_bytes(L.B2, t) and r["capacity"] == L.bank_capacity(L.B2, t)
                                      for t, r in enumerate(recs))
    st = S.FastState.empty()
    res["fast_state_serialised_tensor_bytes"] = st.nbytes()
    save("LEDGER.json", res)


if __name__ == "__main__":
    torch.set_num_threads(6)
    cmd = sys.argv[1]
    {"stream": do_stream, "t6": do_t6, "bench": do_bench, "visa": do_visa, "metacost": do_metacost,
     "ledger": do_ledger}[cmd]()
