"""NL-CAD-PILOT-00 P0 — cost projection and decision from the measured P0 artifacts.

    python -m exps.nl_cad_pilot_00.analyze

Every projection multiplies a quantity measured in P0 (or a historical C2-DOM measurement, labelled as such) by a
count fixed in the preregistration draft. Nothing here measures; nothing here reads data features.
"""
from __future__ import annotations

import json
from pathlib import Path

from exps.nl_cad_pilot_00 import ledger as L
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a_c2dom import common as D

OUT = Path(__file__).resolve().parents[2] / "results/hope_cad/nl_cad_pilot_00/p0"

# preregistration-draft counts (agents/plans/nl_cad_design_00_pilot_prereg.md, spec 4.4-4.5)
ES_EVALS_PER_FIT = 1 + 60 * 2 * 16 + 1          # initial J, 60 iterations x 2P antithetic, final J
ORDERS_PER_FIT = 8
T_S = 6
PAIRS_PER_EPISODE = T_S * (T_S + 1) // 2        # 21 (t, tau <= t)
FITS_PER_ARM = 3 + 1                             # P1 folds + P2 refit
ES_ARMS = ("A2", "A3", "A4", "A6", "A7", "A8", "A9")
MVTEC_TEST_IMAGES = 1725                         # count from cad_final_benchmark_protocol.md (no test file opened)
QUERIES_PER_SOURCE_CATEGORY = round(MVTEC_TEST_IMAGES / 15) * 784
P1_STREAMS = 3 * (8 + 2)
P2_SOURCE_STREAMS = 8
P2_TARGET_STREAMS = 2
GPU_CAP_H = 6.0


def rd(name):
    return json.loads((OUT / name).read_text())


def main():
    st, t6, bm, bv, mc, lg = (rd(n) for n in ("STREAM.json", "T6_REAL.json", "BENCH_MVTEC.json", "BENCH_VISA.json",
                                              "METACOST.json", "LEDGER.json"))
    hist = [D.read_json(D.OUT / "streams" / "DOM_PQ_B2" / f"boundary_t{t:02d}.json") for t in range(15)]
    rows = [h["rows"] for h in hist]
    fps = [h["fps_seconds"] for h in hist]
    # ---------------------------------------------------------------- ES cost with exact full metric search
    spqr = mc["full_metric_search"]["seconds_per_query"] / mc["full_metric_search"]["bank_rows"]
    qrows_episode = sum(t * rows[t - 1] for t in range(1, T_S + 1)) * QUERIES_PER_SOURCE_CATEGORY
    score_s_per_J = ORDERS_PER_FIT * qrows_episode * spqr
    upd = mc["prox_update_paths_real_t15"]["gram_eigh_path"]["median_s"]
    upd_s_per_J = ORDERS_PER_FIT * T_S * upd
    full_fit_h = ES_EVALS_PER_FIT * (score_s_per_J + upd_s_per_J) / 3600
    total_full_h = full_fit_h * FITS_PER_ARM * len(ES_ARMS)
    # ---------------------------------------------------------------- P5 cached-table route
    fb = mc["p5_bank_t05"]["fallback"]
    f_real = fb["real_cumulative_state_gamma1_theta0"]["fallback_rate"]
    f_min = min(v["fallback_rate_at_mu_theta"] for k, v in fb.items() if k.startswith("gamma="))
    table_h_per_fit = score_s_per_J / 3600       # one theta-independent exact top-64 pass per pair ~ one full J
    p5_fit_h = lambda f: table_h_per_fit + ES_EVALS_PER_FIT * (f * score_s_per_J + upd_s_per_J) / 3600
    # ---------------------------------------------------------------- Level-0 source/target streams (historical)
    l0_episode_s = sum(fps[1:T_S])               # boundaries 1..5 of the canonical stream (boundary 0 has no FPS)
    l0_15_s = sum(fps)
    l0_total_h = ((P1_STREAMS + P2_SOURCE_STREAMS) * l0_episode_s + P2_TARGET_STREAMS * sum(fps[1:12])) / 3600
    # ---------------------------------------------------------------- target-time per-image inference (GPU)
    b = bm["mvtec_final_union_bank"]
    bb = bm["backbone"]
    ro = bm["readout"]
    e2e = {}
    if "gpu_scrc_gamma1_1img" in b:
        for k, nn in (("scrc_gamma1", "gpu_scrc_gamma1_1img"), ("euclid_gamma0", "gpu_scrc_gamma0_1img")):
            e2e[k] = {"backbone_trunk_ms": 1e3 * bb["gpu_trunk_1img_fp32"]["median_s"],
                      "nn_ms": 1e3 * b[nn]["median_s"], "readout_1024_ms": 1e3 * ro["gpu_1024"]["median_s"]}
            e2e[k]["total_ms"] = sum(e2e[k].values())
    replay = {"source": "project-measured Dinomaly-B + Replay-24 on the same RTX 4050 Laptop (DRF-01 LATENCY.json, "
                        "TRAIN; DRF-02 cost table) — different protocol and accuracy metric; cost context only",
              "inference_ms_per_image": 52.0, "persistent_cl_bytes": 245_701_632 + 12_800_000,
              "persistent_cl_bytes_hi": 245_701_632 + 29_000_000, "frozen_encoder_bytes": 346_334_208,
              "update_s_per_task_est": 667 * 0.575}
    scrc = {"persistent_cl_bytes": lg["mvtec_15"]["A4"]["total"], "backbone_file_bytes": L.BACKBONE_FILE_BYTES,
            "backbone_param_bytes": bb["params_fp32_bytes"],
            "update_s_per_boundary_L1_cpu": {"residual_nn": [r["nn_seconds"] for r in st["residual_records"]],
                                             "prox_svd": [r["svd_seconds"] for r in st["policies"]["cumulative_A1A2"]]},
            "update_s_per_boundary_L0_hist_cpu": fps}
    out = {
        "schema": "nl_cad_p0_analysis_v1",
        "es_exact_full_search": {"seconds_per_query_row_gpu": spqr, "query_rows_per_episode": qrows_episode,
                                 "score_seconds_per_J": score_s_per_J, "update_seconds_per_J": upd_s_per_J,
                                 "gpu_h_per_fit": full_fit_h, "gpu_h_all_ES_arms": total_full_h,
                                 "evals_per_fit": ES_EVALS_PER_FIT, "fits_per_arm": FITS_PER_ARM, "arms": len(ES_ARMS)},
        "es_p5_route": {"fallback_rate_real_state_gamma1": f_real, "fallback_rate_min_over_grid": f_min,
                        "gpu_h_per_fit_at_real_rate": p5_fit_h(f_real), "gpu_h_per_fit_at_min_rate": p5_fit_h(f_min),
                        "gpu_h_all_ES_arms_at_real_rate": p5_fit_h(f_real) * FITS_PER_ARM * len(ES_ARMS),
                        "gpu_h_per_fit_if_fallback_zero": p5_fit_h(0.0),
                        "max_fallback_rate_for_cap": max(0.0, (GPU_CAP_H * 3600 / (FITS_PER_ARM * len(ES_ARMS))
                                                               - table_h_per_fit * 3600 - ES_EVALS_PER_FIT * upd_s_per_J)
                                                         / (ES_EVALS_PER_FIT * score_s_per_J))},
        "level0_streams_hist_cpu": {"episode_6_boundary_s": l0_episode_s, "stream_15_s": l0_15_s,
                                    "p1_p2_total_h": l0_total_h, "streams": P1_STREAMS + P2_SOURCE_STREAMS + P2_TARGET_STREAMS},
        "inference_gpu_end_to_end_ms": e2e, "replay_reference": replay, "scrc_costs": scrc,
        "gpu_cap_h": GPU_CAP_H,
    }
    C.write_json(OUT / "ANALYSIS.json", out)
    print(json.dumps({k: v for k, v in out.items() if k in ("es_exact_full_search", "es_p5_route",
                                                            "level0_streams_hist_cpu", "inference_gpu_end_to_end_ms")},
                     indent=1))


if __name__ == "__main__":
    main()
