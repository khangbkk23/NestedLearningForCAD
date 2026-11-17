"""Create-exclusive PROTOCOL.json + LOCK.json for SAND-FINAL (before any SAND code opens test features or masks).

    python -m exps.sand.lock
"""
from __future__ import annotations

from exps.nl_sgm_01a import am03_eval as A3
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a_c2dom import common as D2
from exps.sand import common as S
from exps.task_p.p0_normal_support import require

SOURCES = [
    "AGENTS.md", S.TASK, S.PLAN,
    "exps/sand/__init__.py", "exps/sand/common.py", "exps/sand/score.py", "exps/sand/analyze.py", "exps/sand/lock.py",
    "exps/sand/test_sand.py",
    "exps/vitb8_q2/common.py", "exps/vitb8_q2/train.py", "exps/vitb8_q2/evaluate.py", "exps/vitb8_nl00/run.py",
    "exps/dinomaly/official.py", "exps/dinomaly/run.py", "exps/dinomaly/readout.py", "exps/dinomaly/common.py",
    "exps/nlcad/fast.py", "exps/rh01/common.py", "exps/rh01/numerics.py", "exps/rh01b/run.py",
    "exps/nl_sgm_01a/codecs.py", "exps/nl_sgm_01a/exact.py", "exps/nl_sgm_01a/common.py", "exps/nl_sgm_01a/am02_run.py",
    "exps/nl_sgm_01a/am03_eval.py", "exps/nl_sgm_01a_c2dom/common.py", "exps/nl_sgm_01a_c2dom/fps.py",
    "exps/nl_sgm_01a_c2dom/stream.py", "exps/csg01/select.py", "exps/ad02/hope_cad_ad02_partition.py",
    "exps/cadfinal/metrics.py", "exps/cadfinal/common.py", "exps/calm02/native.py",
    "results/hope_cad/calm02/features/mvtec/MANIFEST.json", "results/hope_cad/nl_sgm_01a_am03/DATASET_MANIFEST.json",
    "results/hope_cad/vitb8_q2/CACHE.json", "results/hope_cad/vitb8_q2/LOCK.json", "results/hope_cad/vitb8_nl00/LOCK.json",
    "results/hope_cad/vitb8_nl00/FINAL.json", "results/hope_cad/nl_sgm_01a_c2_dom_sweep/LOCK.json",
    "results/hope_cad/nl_sgm_01a_c2_dom_sweep/ANALYSIS.json", "results/hope_cad/nl_sgm_01a_c2_dom_sweep/FINAL_VERDICT.json",
]


def main():
    require(not S.LOCK.exists() and not S.PROTOCOL.exists(), "already locked")
    require(not (S.OUT / "A3_s2711").exists(), "scoring artifacts exist before the lock")
    co = C.cohorts()
    require(C.r15_digest()[:8] == C.R15_DIGEST, "R15 cohort digest mismatch")
    a1 = {f"STAGE{s:02d}": S.read_json(S.A1_RUN / f"STAGE{s:02d}.json")["checkpoint_sha256"] for s in range(1, 16)}
    a1_eval = {f"EVAL_STAGE{s:02d}": S.sha(S.A1_OUT / f"EVAL_STAGE{s:02d}.json") for s in range(1, 16)}
    mem = {f"t{t:02d}": S.read_json(S.MEM_DIR / f"boundary_t{t:02d}.json")["ckpt_sha256"] for t in range(15)}
    mem_cells = S.text_sha("\n".join(S.sha(p) for p in sorted(S.MEM_EVAL.glob("stage*.json"))))
    fin = S.read_json(S.A1_OUT / "FINAL.json")["OFF"]
    b2 = S.read_json(D2.OUT / "ANALYSIS.json")["summary"][S.MEM_ARM]
    protocol = {
        "schema": "sand_final_protocol_v1", "task": S.TASK, "plan": S.PLAN, "plan_sha256": S.sha(S.ROOT / S.PLAN),
        "seeds": list(S.SEEDS), "pilot_seed": S.SEEDS[0], "stream": list(S.ORDER),
        "features": "frozen CADIC ViT-B/8 layer 9, CALM-02 MVTec caches ([784,768] per image)",
        "fast": {"class": "exps.vitb8_q2.common.SingleStreamDecoder", "objective": "official global_cosine_hm_percent",
                 "optimizer": "StableAdamW per-block groups, WarmCosine, clip 0.1, fresh per stage, 667 iterations",
                 "q2": "off", "bytes": 245_701_632},
        "slow": {"memory": S.MEM_ARM, "budget_bytes": D2.BUDGETS["B2"], "rule": "locked C2-DOM stream rule",
                 "final_persistent_bytes": b2["bytes"], "calibrator_bytes": 15 * 4 * 8},
        "fusion": {"formula": "g = min((a - mu_theta)/sd_theta, (d - mu_M)/sd_M) + max(mu_theta/sd_theta, mu_M/sd_M)",
                   "primary_readout": "P: 28x28 -> native bilinear (am03_eval.native_maps); image = max",
                   "calibrators": "frozen at acquisition; 6 SEL probes of the category (4,704 patches); population "
                                  "mean/std FP64; never refreshed",
                   "non_gating_variants": ["S1 Dinomaly-chain readout", "D1 refreshed calibrators (not deployable)",
                                           "TH theta channel under readout P"],
                   "probes": {c: {"sel": co[c]["sel"], "eval": co[c]["eval"]} for c in S.ORDER}},
        "comparators": {"A1": {"final_pixel_ap": fin["final_macro_pixel_ap"], "final_image_auroc": fin["final_macro_image_auroc"],
                               "acq_pixel_ap": fin["acq_macro_pixel_ap"], "replay_bytes": 24 * 784 * 768 * 4},
                        "A2": {"final_pixel_ap": b2["final_pixel_ap"], "final_image_auroc": b2["final_image_auroc"],
                               "acq_pixel_ap": b2["acq_pixel_ap"]}},
        "gate_G0": S.GATE0, "gate_G1": S.GATE1, "rho": S.RHO, "bootstrap": S.BOOT, "tolerances": S.TOL, "caps": S.CAPS,
        "replay_seed": "seed * 1000 + 700 + stage", "replay_bm_capacity": 6,
        "phase2_claims": ["BIDIRECTIONAL_NL_SUPPORTED", "ONE_WAY_COUPLING_SUPPORTED", "ENSEMBLE_ONLY", "NOT_SUPPORTED"],
        "inputs": {"a1_checkpoints": a1, "a1_eval": a1_eval, "memory_checkpoints": mem, "memory_cells_digest": mem_cells,
                   "am03_dataset_manifest_sha256": S.sha(A3.OUT / "DATASET_MANIFEST.json"), "r15_digest": C.r15_digest()},
        "exposure": "MVTec test split historically exposed; every MVTec result is exploratory",
    }
    S.OUT.mkdir(parents=True, exist_ok=True)
    S.write_json(S.PROTOCOL, protocol)
    S.write_json(S.LOCK, {"schema": "sand_final_lock_v1", "protocol_sha256": S.sha(S.PROTOCOL),
                          "sources": {s: S.sha(S.ROOT / s) for s in SOURCES}, "utc": S.now(),
                          "note": "Phase 1/2 training and stream code is locked by PHASE1_LOCK / PHASE2_LOCK before "
                                  "those phases run; rules, gates, seeds and the fusion scorer are fixed here"})
    print("LOCK", S.sha(S.LOCK), "PROTOCOL", S.sha(S.PROTOCOL))


if __name__ == "__main__":
    main()
