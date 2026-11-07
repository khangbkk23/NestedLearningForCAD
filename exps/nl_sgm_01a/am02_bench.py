"""NL-SGM-01A-AM02 resource preflight (timings only; every selection/distance output is discarded).

Measures the canonical csg01.select_fps per-iteration cost on writer rows (CPU; CUDA reported separately with a
small CPU/CUDA index-parity check), re-measures exact-NN throughput, and projects every AM02 arm with the exact
iteration schedule of the real 15-task stream. Probe images are never loaded (asserted by bench.Data).

    python -m exps.nl_sgm_01a.am02_bench  -> results/hope_cad/nl_sgm_01a_am02/{BENCH.json, RESOURCE_PROJECTION.json}
"""
from __future__ import annotations

import time

import numpy as np
import torch

from exps.csg01.select import select_fps
from exps.nl_sgm_01a import am02 as A
from exps.nl_sgm_01a import bench as B
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a import exact as X

OUT = C.ROOT / "results/hope_cad/nl_sgm_01a_am02"
CAPS = {"total_wall_h": 16.0, "per_arm_h": 4.0, "rss_gb": 12.0, "disk_gb": 5.0}
SAFETY = 1.3
ITERS = 150
# Carried from the original preflight log (same code, same machine): PQ fit on 100k rows, encode per row.
CARRIED = {"pq_fit_100k_s": 160.24225330352783, "pq_encode_per_row_s": 0.00020525625220919807,
           "source": "results/hope_cad/nl_sgm_01a/logs/bench_cpu_preflight.log (BM4)"}


def time_fps(D, k):
    t = time.time(); select_fps(D, k)
    if D.is_cuda:
        torch.cuda.synchronize()
    return (time.time() - t) / k


def main():
    C.set_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = B.Data()
    n = d.n
    res = {"cpu_threads": torch.get_num_threads(), "iters_per_point": ITERS}

    _, x = d.writers("screw")
    rows = x.reshape(-1, 768).contiguous()
    pts = []
    for m in (2_499, 37_632, 75_121, 108_192, 166_208, 241_472):
        s = time_fps(rows[:m], ITERS)
        pts.append({"rows": m, "s_per_iteration": s}); B.log("AM-BM1 cpu", pts[-1])
    res["fps_cpu"] = pts
    a, b = np.polyfit([p["rows"] for p in pts], [p["s_per_iteration"] for p in pts], 1)
    res["fps_cpu_fit"] = {"intercept_s": float(b), "per_row_s": float(a)}

    if torch.cuda.is_available():
        cuda = []
        try:
            for m in (75_121, 241_472):
                Dg = rows[:m].cuda()
                time_fps(Dg, 5)                                            # warm-up
                cuda.append({"rows": m, "s_per_iteration": time_fps(Dg, ITERS)}); B.log("AM-BM1 cuda", cuda[-1])
                del Dg
            small = rows[:20_000]
            same = bool(np.array_equal(select_fps(small, 500), select_fps(small.cuda(), 500)))
            res["fps_cuda"] = {"points": cuda, "index_parity_20k_500": same,
                               "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30}
        except Exception as e:                                             # report, never silently switch device
            res["fps_cuda"] = {"error": repr(e)[:300]}
        B.log("AM-BM1 cuda", res["fps_cuda"])
        torch.cuda.empty_cache()

    q = rows[:4096].contiguous()
    rs = X.RowSet(rows[4096:4096 + A.PQ_CAPACITY].contiguous(), torch.arange(A.PQ_CAPACITY))
    t = time.time(); X.nn(q, rs); dt = time.time() - t
    res["exact_nn"] = {"queries": 4096, "rows": A.PQ_CAPACITY, "pairs_per_s": 4096 * A.PQ_CAPACITY / dt}
    B.log("AM-BM2 exact", res["exact_nn"])
    del x, rows, rs
    d.F.drop("screw")
    res["carried"] = CARRIED
    res["peak_rss_gb"] = C.peak_rss_gb()
    res["wall_s"] = time.time() - t0

    # ------------------------------------------------------------ projection (exact stream schedule)
    pr = res["fps_cpu_fit"]["per_row_s"]
    pps = res["exact_nn"]["pairs_per_s"]
    writer_rows = sum(n.values()) * C.PATCHES
    probes_per_cat = 12
    icpt = max(res["fps_cpu_fit"]["intercept_s"], 0.0)
    fps_iters = {cap: sum(_quota_iters(n, cap)) for cap in (A.C1_CAPACITY, A.PQ_CAPACITY)}
    fps = {cap: B.fps_seconds(n, cap, pr) + fps_iters[cap] * icpt for cap in (A.C1_CAPACITY, A.PQ_CAPACITY)}

    def eval_s(cap):            # every boundary: all seen probes (SEL+EVAL) vs memory, global + own-category NN
        return sum(2 * C.PATCHES * probes_per_cat * (t + 1) * cap for t in range(15)) / pps

    encode_all = writer_rows * CARRIED["pq_encode_per_row_s"]
    arms = {
        "C1_FPS_FP32": {"fps_s": fps[A.C1_CAPACITY], "eval_s": eval_s(A.C1_CAPACITY)},
        "C2_FPS_PQ": {"fps_s": fps[A.PQ_CAPACITY], "pq_fit_s": CARRIED["pq_fit_100k_s"], "encode_s": encode_all,
                      "eval_s": eval_s(A.PQ_CAPACITY)},
        "C3_HASH_PQ": {"select_s": 60.0, "pq_fit_s": CARRIED["pq_fit_100k_s"], "encode_s": encode_all,
                       "eval_s": eval_s(A.PQ_CAPACITY)},
        "C4_SAMEID_PQ": {"encode_s": 15 * A.C1_CAPACITY * CARRIED["pq_encode_per_row_s"], "eval_s": eval_s(A.C1_CAPACITY)},
        "A0_oracle_shared": {"nn_s": C.PATCHES * probes_per_cat * len(C.ORDER) * writer_rows / pps},
        "stream_read": {"io_s": writer_rows * 768 * 4 / 0.5e9 * 4},
    }
    for v in arms.values():
        v["raw_s"] = sum(v.values()); v["projected_s"] = v["raw_s"] * SAFETY
    total_h = sum(v["projected_s"] for v in arms.values()) / 3600
    reasons = [f"{k}: {v['projected_s'] / 3600:.2f} h > {CAPS['per_arm_h']} h" for k, v in arms.items()
               if v["projected_s"] / 3600 > CAPS["per_arm_h"]]
    if total_h > CAPS["total_wall_h"]:
        reasons.append(f"total {total_h:.2f} h > {CAPS['total_wall_h']} h")
    if res["peak_rss_gb"] > CAPS["rss_gb"]:
        reasons.append(f"bench peak RSS {res['peak_rss_gb']:.2f} GB > {CAPS['rss_gb']} GB")
    proj = {"version": A.VERSION, "device": "cpu (canonical select_fps; CUDA not used for the projection)",
            "caps": CAPS, "safety_factor": SAFETY, "arm_bytes": A.arm_bytes(), "budget_B1": C.B1,
            "fps_iterations": fps_iters, "arms": arms, "total_projected_h": total_h,
            "gate": {"fits_caps": not reasons, "reasons": reasons},
            "notes": ["fps_s = exact per-boundary (quota x candidate rows) schedule x measured per-row-iteration cost",
                      "eval_s counts global and own-category NN for all seen SEL+EVAL probes at all 15 boundaries",
                      "C3 select_s is a 60 s allowance (hash sort); C4 reuses C1 identities",
                      "outcome_compute_performed: false"]}
    C.write_json(OUT / "BENCH.json", res)
    C.write_json(OUT / "RESOURCE_PROJECTION.json", proj)
    B.log("AM02 projection", {k: round(v["projected_s"] / 3600, 2) for k, v in arms.items()}, "total_h", round(total_h, 2),
          "fits", not reasons, reasons)


def _quota_iters(n, cap):
    kept = {}
    for k, c in enumerate(C.ORDER):
        quotas = B.R.AP.per_group_quota(cap, k + 1)
        kept[k] = min(quotas[k], n[c] * C.PATCHES); yield kept[k]
        for j in range(k):
            if kept[j] > quotas[j]:
                yield quotas[j]; kept[j] = quotas[j]


if __name__ == "__main__":
    main()
