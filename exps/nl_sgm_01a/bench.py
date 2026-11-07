"""NL-SGM-01A CPU resource preflight BM1-BM6 (timings only; every distance/score output is discarded).

Surrogate data: writer images of task 0 (bottle), task 1 (cable) and the largest writer file (screw). Probe
images are never loaded (asserted). No distortion statistic is computed or saved.

    python -m exps.nl_sgm_01a.bench            -> results/hope_cad/nl_sgm_01a/{BENCH,RESOURCE_PROJECTION}.json
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import torch

from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a import exact as X
from exps.nl_sgm_01a import rules as R
from exps.task_p.p0_normal_support import require

CAPS = {"total_wall_h": 16.0, "per_arm_h": 3.0, "rss_gb": 8.0, "disk_gb": 5.0}
SAFETY = 1.3
LOG = []


def log(*a):
    s = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.append(s)


class Data:
    def __init__(self):
        self.co = C.cohorts()
        self.F = C.Features()
        self.probes = {p for v in self.co.values() for p in v["sel"] + v["eval"]}
        self.n = {c: len(self.co[c]["writers"]) for c in C.ORDER}

    def writers(self, c, k=None):
        paths = self.co[c]["writers"][:k]
        require(not set(paths) & self.probes, "probe leaked into the benchmark")
        return paths, self.F.images(paths)


def gid_base(n, c):
    i = C.ORDER.index(c)
    return sum(n[C.ORDER[j]] for j in range(i)) * C.PATCHES


# ------------------------------------------------------------------ BM1 / BM5: exact NN throughput

def bm_exact(d, out):
    _, bx = d.writers("bottle", 6)
    q = bx.reshape(-1, 768)[:4096].contiguous()
    res = {}
    for c in ("screw", "bottle"):
        paths, x = d.writers(c)
        rows = x.reshape(-1, 768)
        rs = X.RowSet(rows, torch.arange(len(rows)))
        ex = torch.arange(4096) if c == "bottle" else None              # bottle: queries are rows -> self-excluded
        st = {}
        t = time.time(); X.nn(q, rs, exclude=ex, stats=st); dt = time.time() - t
        ratio = X.verify_screen_bound(q[:256], rows[:20000])
        res[c] = {"rows": len(rows), "queries": 4096, "seconds": dt, "pairs_per_s": 4096 * len(rows) / dt,
                  "candidates_per_query": st["candidates"] / st["queries"], "screen_bound_ratio_max": ratio,
                  "rss_gb": C.peak_rss_gb()}
        log("BM1", c, res[c])
        d.F.drop(c)
    # BM5: scoring against a 75k-row decoded bank (surrogate: bottle writer rows, PQ-decoded)
    _, bx2 = d.writers("bottle")
    rows = bx2.reshape(-1, 768)[:75121].contiguous()
    cb = K.fit(rows[::3][:100_000], iters=2)
    dec = K.decode(K.encode(rows, cb), cb)
    rs = X.RowSet(dec, torch.arange(len(dec)))
    _, cx = d.writers("cable", 6)
    qc = cx.reshape(-1, 768)[:4096].contiguous()
    st = {}
    t = time.time(); X.nn(qc, rs, stats=st); dt = time.time() - t
    res["bank75k"] = {"rows": len(dec), "queries": 4096, "seconds": dt, "pairs_per_s": 4096 * len(dec) / dt,
                      "candidates_per_query": st["candidates"] / st["queries"]}
    log("BM5", res["bank75k"])
    out["BM1_BM5_exact"] = res


# ------------------------------------------------------------------ BM2: rule update cost

def time_rule(make, d, n0, n1, max_s):
    """Stream bottle writers [0:n0) then cable writers [0:n1); per-image seconds; stops at max_s (lower bound)."""
    arm = make()
    per, t_all = [], time.time()
    stopped = False
    for t, (c, k) in enumerate((("bottle", n0), ("cable", n1))):
        if k == 0:
            continue
        paths, x = d.writers(c, k)
        base = gid_base(d.n, c)
        arm.begin_task(t)
        for j in range(len(paths)):
            s = time.time()
            arm.image(x[j].clone(), torch.arange(base + j * 784, base + j * 784 + 784), t)
            per.append((c, j, time.time() - s))
            if time.time() - t_all > max_s:
                stopped = True; break
        if stopped:
            break
        s = time.time(); arm.end_task(t); per.append((c, "end", time.time() - s))
    return per, stopped, arm


def summarize(per, capacity, stopped):
    img = [(c, j, s) for c, j, s in per if j != "end"]
    fill = math.ceil(capacity / 784)
    fill_s = sum(s for _, _, s in img[:fill])
    sat = [s for _, _, s in img[fill:]]
    new_task = [s for c, _, s in img if c == "cable"]
    return {"images_timed": len(img), "fill_images": fill, "fill_seconds": fill_s,
            "sat_images": len(sat), "sat_mean_s": float(np.mean(sat)) if sat else None,
            "sat_min_s": float(np.min(sat)) if sat else None,
            "new_task_mean_s": float(np.mean(new_task)) if new_task else None,
            "boundary_seconds": [s for _, j, s in per if j == "end"],
            "stopped_at_time_cap": stopped, "rss_gb": C.peak_rss_gb()}


def project_rule(sm, n_images=2919):
    """Projected seconds for the full stream (x SAFETY). Lower bound: saturated *minimum* per image."""
    if sm["sat_mean_s"] is None:
        per = sm["fill_seconds"] / max(1, sm["images_timed"])
        return {"projected_s": per * n_images * SAFETY, "lower_bound_s": per * n_images}
    rest = n_images - sm["fill_images"]
    rate = max(sm["sat_mean_s"], sm["new_task_mean_s"] or 0.0)
    return {"projected_s": (sm["fill_seconds"] + rest * rate) * SAFETY,
            "lower_bound_s": sm["fill_seconds"] + rest * sm["sat_min_s"]}


def bm_rules(d, out):
    a_s4, _ = R.s4_assignment(); a_r4, _ = R.r4_assignment(0)
    stage_a = {
        "A2_cadic_2500": (lambda: R.CodecArm(R.ClosestPairRule(2500, "cadic"), "fp32", owner_bytes=0), 2500),
        "A3_greedy_2500": (lambda: R.CodecArm(R.GreedyRule(2500), "fp32", owner_bytes=0), 2500),
        "A4_quota_2499": (lambda: R.CodecArm(R.ClosestPairRule(2499, "quota"), "fp32"), 2499),
        "A5_S4_2500": (lambda: R.CodecArm(R.ClosestPairRule(2500, "cadic", assignment=a_s4), "fp32", owner_bytes=0), 2500),
        "A6_R4_2500": (lambda: R.CodecArm(R.ClosestPairRule(2500, "cadic", assignment=a_r4), "fp32", owner_bytes=0), 2500),
        "A8_cadic_10000": (lambda: R.CodecArm(R.ClosestPairRule(10000, "cadic"), "fp32", owner_bytes=0), 10000),
        "A9_calm_l1_9684": (lambda: R.CALMRule(), 9684),
    }
    res = {}
    for name, (make, cap) in stage_a.items():
        fill = math.ceil(cap / 784)
        per, stopped, _ = time_rule(make, d, min(d.n["bottle"], fill + 16), 8, max_s=900)
        sm = summarize(per, cap, stopped); sm.update(project_rule(sm))
        res[name] = sm; log("BM2", name, {k: sm[k] for k in ("sat_mean_s", "new_task_mean_s", "projected_s", "lower_bound_s")})
    out["BM2_stage_a_rules"] = res
    # Stage-B rules at 75,121 rows (dominant B5/B6/B7 cost) for every Stage-A candidate X
    cap = K.pq_capacity(C.B1, 1)
    stage_b = {
        "X=A2_cadic": lambda: R.ClosestPairRule(cap, "cadic"),
        "X=A3_greedy": lambda: R.GreedyRule(cap),
        "X=A4_quota": lambda: R.ClosestPairRule(cap, "quota"),
        "X=A5_S4": lambda: R.ClosestPairRule(cap, "cadic", assignment=a_s4),
        "X=A6_R4": lambda: R.ClosestPairRule(cap, "cadic", assignment=a_r4),
    }
    resb = {}
    for name, make in stage_b.items():
        fill = math.ceil(cap / 784)
        per, stopped, _ = time_rule(lambda: R.CodecArm(make(), "fp32", owner_bytes=0), d,
                                    min(d.n["bottle"], fill + 6), 4, max_s=1500)
        sm = summarize(per, cap, stopped); sm.update(project_rule(sm))
        resb[name] = sm; log("BM2b", name, {k: sm[k] for k in ("images_timed", "sat_mean_s", "projected_s", "lower_bound_s", "stopped_at_time_cap")})
    out["BM2_stage_b_rules_75k"] = resb


# ------------------------------------------------------------------ BM3: boundary FPS cost model

def bm_fps(d, out):
    from exps.csg01.select import _dist_to_vector
    res = {}
    for c in ("bottle", "screw"):
        _, x = d.writers(c)
        D = x.reshape(-1, 768).double()
        D2 = (D * D).sum(1)
        t = time.time()
        for j in range(30):
            _dist_to_vector(D, D[j], D2)
        res[c] = {"rows": len(D), "s_per_iteration": (time.time() - t) / 30, "rss_gb": C.peak_rss_gb()}
        del D, D2
        d.F.drop(c)
        log("BM3", c, res[c])
    out["BM3_fps"] = res


def fps_seconds(n, capacity, per_row_s):
    """Exact iteration count of the boundary-FPS rule over the real stream, times the measured cost per
    iteration per candidate row (linear in rows)."""
    total, kept = 0.0, {}
    for k, c in enumerate(C.ORDER):
        rows = n[c] * 784
        quotas = R.AP.per_group_quota(capacity, k + 1)
        total += quotas[k] * rows * per_row_s + rows * 768 * 8 / 5e9          # + FP64 copy
        kept[k] = quotas[k]
        for j in range(k):
            if kept[j] > quotas[j]:
                total += quotas[j] * kept[j] * per_row_s
                kept[j] = quotas[j]
    return total


# ------------------------------------------------------------------ BM4: PQ fit / encode / transport

def bm_pq(d, out):
    _, x = d.writers("bottle")
    rows = x.reshape(-1, 768)
    sample = rows[:100_000].contiguous()
    t = time.time(); cb = K.fit(sample); fit_s = time.time() - t
    log("BM4 fit", fit_s)
    _, sx = d.writers("screw"); srows = sx.reshape(-1, 768)
    t = time.time(); codes = K.encode(srows, cb); enc_s = time.time() - t
    log("BM4 encode", len(srows), enc_s)
    cb2 = K.fit(srows[:100_000].contiguous(), iters=25, seed=7)
    t = time.time(); K.transport(codes[:75121], cb, cb2); tr_s = time.time() - t
    # graph rebuild after transport (CADIC rule, 75k rows)
    rule = R.ClosestPairRule(75121, "cadic")
    rule.begin_task(0)
    dec = K.decode(codes[:75121], cb2)
    m = rule.mems[0]
    m.bank.x[:] = dec; m.bank.x64[:] = dec.double(); m.bank.n2[:] = dec.double().square().sum(1)
    m.bank.owner[:] = 0; m.bank.ids[:] = torch.arange(75121)
    t = time.time(); R.rebuild_graph(m.graph); rb_s = time.time() - t
    out["BM4_pq"] = {"fit_100k_s": fit_s, "encode_rows": len(srows), "encode_s": enc_s,
                     "encode_per_row_s": enc_s / len(srows), "transport_75k_s": tr_s,
                     "graph_rebuild_75k_s": rb_s, "rss_gb": C.peak_rss_gb()}
    log("BM4", out["BM4_pq"])
    d.F.drop()


# ------------------------------------------------------------------ BM6: diagnostics (self-excluded refs, false support, gaps)

def bm_diag(d, out):
    paths, x = d.writers("bottle")
    rows = x.reshape(-1, 768)
    rs = X.RowSet(rows, torch.arange(len(rows)))
    samp = torch.arange(0, len(rows), len(rows) // 4096)[:4096]
    t = time.time(); X.nn(rows[samp].contiguous(), rs, exclude=samp); q50_s = time.time() - t
    t = time.time(); X.kth(rows[samp].contiguous(), rs, 16, exclude=samp); r16_s = time.time() - t
    gx = 0.5 * (rows[samp[:2048]] + rows[samp.flip(0)[:2048]])
    t = time.time(); X.nn(gx.contiguous(), rs); gap_s = time.time() - t
    out["BM6_diag"] = {"rows": len(rows), "q50_4096_s": q50_s, "r16_4096_s": r16_s, "gap_2048_s": gap_s,
                       "rss_gb": C.peak_rss_gb()}
    log("BM6", out["BM6_diag"])
    d.F.drop()


# ------------------------------------------------------------------ projection

def project(bench, n):
    W = {c: n[c] * 784 for c in C.ORDER}
    cum = np.cumsum([W[c] for c in C.ORDER])
    Wtot = int(cum[-1])
    pps = bench["BM1_BM5_exact"]["screw"]["pairs_per_s"]
    pps_bank = bench["BM1_BM5_exact"]["bank75k"]["pairs_per_s"]
    probes = 180 * 784
    oracle_s = probes * Wtot / pps
    q_per_arm = sum((k + 1) for k in range(15)) * 12 * 784
    scored_a = {"A2": 2500, "A3": 2500, "A4": 2499, "A5": 2500, "A6x3": 3 * 2500, "A7": 2499, "A8": 10000, "A9": 9684}
    scored_b = {"B1": 2500, "B2": 2500, "B3": 2500, "B4": 9884, "B5": 75121, "B6": 75121, "B7": 75121,
                "B8": 9884, "X18k": 18368, "R1_B5": 18368, "R1_B6": 18368, "R2_B5": 9884}
    score_s = {k: q_per_arm * v / (pps_bank if v > 20000 else pps) for k, v in {**scored_a, **scored_b}.items()}
    # BM6 diagnostics
    diag = bench["BM6_diag"]
    per_pair = diag["q50_4096_s"] / (4096 * diag["rows"])
    ref_s = sum(4096 * W[c] for c in C.ORDER) * per_pair * (1 + diag["r16_4096_s"] / diag["q50_4096_s"])
    fs_rows = {k: v for k, v in scored_b.items()}
    fs_rows.update({k: v for k, v in scored_a.items()})
    fs_upper = {k: sum(v * int(cum[b]) for b in range(15)) / pps for k, v in fs_rows.items()}   # every row re-checked
    fs_lower = {k: v * Wtot / pps for k, v in fs_rows.items()}                                 # each row once
    gap_s = sum(15 * 4096 * v for v in fs_rows.values()) / pps
    # rule updates
    a = bench["BM2_stage_a_rules"]
    upd_a = {k: v["projected_s"] for k, v in a.items()}
    upd_a["A6_R4_2500"] *= 3                                                                        # three draws
    fps_rate_bottle = bench["BM3_fps"]["bottle"]["s_per_iteration"] / bench["BM3_fps"]["bottle"]["rows"]
    upd_a["A7_fps_2499"] = fps_seconds(n, 2499, fps_rate_bottle) * SAFETY
    b75 = bench["BM2_stage_b_rules_75k"]
    pq = bench["BM4_pq"]
    pq_extra = (15 * pq["fit_100k_s"] + 15 * pq["transport_75k_s"] + 15 * pq["graph_rebuild_75k_s"]
                + 2919 * 784 * pq["encode_per_row_s"])
    per_x = {}
    for name, sm in list(b75.items()) + [("X=A7_fps", None)]:
        if sm is None:
            r75 = fps_seconds(n, 75121, fps_rate_bottle) * SAFETY
            lb75 = fps_seconds(n, 75121, fps_rate_bottle)
        else:
            r75, lb75 = sm["projected_s"], sm["lower_bound_s"]
        # B5, B6, B7 start at 75,121 rows; B7 shrinks to 18,368 (cost bounded below by the 18k rate ~ r75/4)
        b_rules = {"B5": r75, "B6": r75 + pq_extra, "B7": r75 * 0.5, "B4": r75 * 9884 / 75121,
                   "B8": 60.0, "X18k": r75 * 18368 / 75121, "R1_B5": r75 * 18368 / 75121,
                   "R1_B6": r75 * 18368 / 75121, "R2_B5": r75 * 9884 / 75121}
        per_x[name] = {"B_rules_s": b_rules, "B_total_s": sum(b_rules.values()),
                       "max_single_arm_s": max(b_rules.values()), "B5_lower_bound_s": lb75}
    a_total = sum(upd_a.values())
    common_s = oracle_s + sum(score_s.values()) + ref_s + sum(fs_upper.values()) + gap_s
    totals = {name: a_total + common_s + v["B_total_s"] for name, v in per_x.items()}
    worst_x = max(totals, key=totals.get)
    peak_rss = bench["peak_rss_gb"]
    n_scored = len(scored_a) + 2 + len(scored_b)                       # A6 has 3 draws
    disk_gb = (n_scored * q_per_arm * 16                               # per-patch FP64 distance + int64 NN id
               + 2 * probes * 15 * 16                                  # A0 / A1 prefix oracle (+ owners)
               + 15 * sum(v * 100 for v in scored_b.values())          # Stage-B states per boundary (codes, ids)
               + 15 * sum(v * 3100 for v in scored_a.values())) / 1e9  # Stage-A FP32 states per boundary
    gate = {
        "total_wall_h_worst_X": totals[worst_x] / 3600, "worst_X": worst_x,
        "total_wall_h_by_X": {k: v / 3600 for k, v in totals.items()},
        "per_arm_max_h_stage_a": max(upd_a.values()) / 3600,
        "per_arm_max_h_stage_b_by_X": {k: v["max_single_arm_s"] / 3600 for k, v in per_x.items()},
        "peak_rss_gb_measured": peak_rss, "disk_gb_projected": disk_gb,
    }
    fits = {
        "total": gate["total_wall_h_worst_X"] <= CAPS["total_wall_h"],
        "per_arm_a": gate["per_arm_max_h_stage_a"] <= CAPS["per_arm_h"],
        "per_arm_b_all_X": all(v <= CAPS["per_arm_h"] for v in gate["per_arm_max_h_stage_b_by_X"].values()),
        "rss": peak_rss <= CAPS["rss_gb"], "disk": disk_gb <= CAPS["disk_gb"],
    }
    return {
        "caps": CAPS, "safety_factor": SAFETY, "writer_images": n, "writer_patches_total": Wtot,
        "oracle_s": oracle_s, "scoring_s_by_arm": score_s, "reference_scales_s": ref_s,
        "false_support_s_upper_by_arm": fs_upper, "false_support_s_lower_by_arm": fs_lower, "gap_probes_s": gap_s,
        "stage_a_update_s_by_arm": upd_a, "stage_b_by_candidate_X": per_x, "pq_overheads_s": pq_extra,
        "gate": gate, "fits_caps": fits, "all_fit": all(fits.values()),
        "notes": [
            "Rule costs are extrapolated from measured per-image times on bottle (fill + saturated) and the first "
            "cable images; replacement rates later in the stream are unmeasured. lower_bound_s uses the minimum "
            "observed saturated per-image time.",
            "B7 cost assumed 0.5 x the 75k-row rate (capacity shrinks from 75,121 to 18,368 rows).",
            "False-support upper bound re-checks every stored row against W_k at every boundary (exact for B6, "
            "whose rows are all re-decoded at each boundary).",
        ],
    }


def main():
    C.set_threads(6)
    out_dir = C.OUT
    require(not (out_dir / "BENCH.json").exists(), "BENCH.json exists (create-exclusive)")
    d = Data()
    bench = {"utc_start": C.now(), "threads": torch.get_num_threads(), "torch": torch.__version__,
             "cpu": open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": "),
             "cuda_visible": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
             "surrogate_data": "writer images of bottle/cable/screw only; probes asserted absent"}
    t0 = time.time()
    for step in (bm_exact, bm_fps, bm_pq, bm_diag, bm_rules):
        step(d, bench)
        d.F.drop()
    bench["wall_s"] = time.time() - t0
    bench["peak_rss_gb"] = C.peak_rss_gb()
    bench["log"] = LOG
    proj = project(bench, d.n)
    C.write_json(out_dir / "BENCH.json", bench)
    C.write_json(out_dir / "RESOURCE_PROJECTION.json", proj)
    log("ALL_FIT", proj["all_fit"], proj["gate"])


if __name__ == "__main__":
    sys.exit(main())
