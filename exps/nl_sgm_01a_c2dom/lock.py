"""Create-exclusive PROTOCOL.json + LOCK.json for the C2-DOM sweep (before any fit, stream or test evaluation).

    python -m exps.nl_sgm_01a_c2dom.lock
"""
from __future__ import annotations

import json

from exps.nl_sgm_01a import am03_eval as A3
from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a_c2dom import analyze as AN
from exps.nl_sgm_01a_c2dom import common as D
from exps.nl_sgm_01a_c2dom import fps as FPS
from exps.task_p.p0_normal_support import require

SOURCES = [
    "AGENTS.md", "agents/tasks/task_nl_sgm_01a_c2_dom_sweep.md",
    "exps/nl_sgm_01a_c2dom/common.py", "exps/nl_sgm_01a_c2dom/fps.py", "exps/nl_sgm_01a_c2dom/stream.py",
    "exps/nl_sgm_01a_c2dom/evaluate.py", "exps/nl_sgm_01a_c2dom/analyze.py", "exps/nl_sgm_01a_c2dom/lock.py",
    "exps/nl_sgm_01a_c2dom/test_c2dom.py",
    "exps/nl_sgm_01a/codecs.py", "exps/nl_sgm_01a/exact.py", "exps/nl_sgm_01a/common.py", "exps/nl_sgm_01a/am02_run.py",
    "exps/nl_sgm_01a/am03_eval.py", "exps/nl_sgm_01a/rules.py", "exps/csg01/select.py", "exps/cadfinal/metrics.py",
    "exps/ad02/hope_cad_ad02_partition.py", "exps/nl_sgm_01a_f1/f2.py",
    "results/hope_cad/calm02/features/mvtec/MANIFEST.json",
    "results/hope_cad/nl_sgm_01a_am03/DATASET_MANIFEST.json", "results/hope_cad/nl_sgm_01a_am03/AM03_LOCK.json",
    "results/hope_cad/nl_sgm_01a_am03/METRICS.json", "results/hope_cad/nl_sgm_01a_am02/LOCK_r2.json",
    "results/hope_cad/nl_sgm_01a_f2/F2_VERDICT.json", "results/hope_cad/nl_sgm_01a_f2/F2_RESULTS.json",
]


def main():
    require(not D.LOCK.exists() and not D.PROTOCOL.exists(), "already locked")
    co, n = D.setup()
    part = D.partition(co, n)
    pmeta = {}
    for c in D.ORDER:
        fg, sg = D.pool_gids(n, c, part[c]["fit"]), D.pool_gids(n, c, part[c]["sel"])
        pmeta[c] = {"writers": n[c], "fit_images": len(part[c]["fit"]), "sel_images": len(part[c]["sel"]),
                    "fit_rows": len(fg), "sel_rows": len(sg), "fit_sample_rows": min(len(fg), D.FIT_ROWS),
                    "fit_gids_sha": D.gid_sha(fg), "sel_gids_sha": D.gid_sha(sg),
                    "fit_paths_sha": C.text_sha("\n".join(p for _, p in part[c]["fit"])),
                    "sel_paths_sha": C.text_sha("\n".join(p for _, p in part[c]["sel"]))}
    ledger = {a: {"codec": s["codec"], "budget_bytes": s["bytes"],
                  "capacity_by_boundary": [D.capacity(s["codec"], s["bytes"], t) for t in range(15)],
                  "persistent_bytes_by_boundary": [D.ledger_bytes(s["codec"], s["bytes"], t) for t in range(15)]}
              for a, s in D.arms(D.BUDGETS.keys()).items()}
    man = json.loads((A3.OUT / "DATASET_MANIFEST.json").read_text())
    protocol = {
        "schema": "c2dom_sweep_protocol_v1", "task": "agents/tasks/task_nl_sgm_01a_c2_dom_sweep.md",
        "question": "Can continual in-domain PQ support memory preserve acquisition, improve retention and stay "
                    "competitive at a reasonable persistent-memory budget? (memory-budget Pareto, B1/B2/B4[/B8])",
        "features": "frozen CADIC ViT-B/8 layer 9, CALM-02 MVTec caches ([784,768] per image)",
        "stream": "15 tasks, alphabetical MVTec order " + ",".join(D.ORDER) + "; normal-only R15 writer cohorts",
        "partition": {"rule": "per category, writer images sorted by sha256('" + D.SPLIT_KEY + "'+path); first floor(n/2) "
                      "-> FIT pool (codebook fitting only), rest -> SEL pool (the only support candidates for EVERY new "
                      "arm, PQ and FP32). Image-level, deterministic, label-free; fit and stored identities are disjoint "
                      "by construction (preserves the F2 support-disjoint principle at every capacity).",
                      "categories": pmeta},
        "codec": {"M": K.M, "K": K.K, "codebook_dtype": "float16", "fit": "k-means++ seed 20261106 (+subspace), 25 Lloyd "
                  "iterations, hash-first <= 100,000 FIT-pool rows (random_priorities(20261106, gid), ties by gid)",
                  "scope": "one codebook per category, fit when the category arrives from its own FIT pool, append-only, "
                           "never refit; codebook c is in-domain for category c only and only decodes c's rows",
                  "note": "codebooks and SEL-pool codes are a deterministic function of category data only, so they are "
                          "computed once and shared by all budgets (identical to fitting inside each stream)"},
        "selection": "FPSRule semantics (AM02 C1/C2): per_group_quota(capacity_t, t+1) over seen categories, remainder "
                     "to earlier tasks; a category above quota is reduced by canonical exps.csg01.select.select_fps on "
                     "its own kept (decoded) rows; PQ selects in decoded space (as C2)",
        "fps_equivalence": "exps/nl_sgm_01a_c2dom/fps.py: canonical CPU function, or GPU FP64 with a per-step "
                           "certificate (gap > 32 (gamma_768 + u) R^2) and canonical CPU fallback on any uncertified "
                           "step; returned indices are the canonical ones. Pre-lock tests compare against select_fps.",
        "capacity_rule": "capacity_t = floor((B - 64 - n_codebooks_t * 393,216) / row_bytes); n_codebooks_t = t+1 (PQ), "
                         "0 (FP32); PQ capacity shrinks as codebooks are appended (no knowledge of the stream length)",
        "ledger": {"charged": "PQ codes uint8[cap,96] | FP32 rows float32[cap,768]; support ids int32[cap]; owner int8[cap]; "
                              "codebooks float16[t+1,96,256,8]; header int32[16]; unused slots charged (fixed capacity)",
                   "row_bytes": {"pq": D.PQ_ROW_BYTES, "fp32": D.FP32_ROW_BYTES}, "header_bytes": D.HEADER_BYTES,
                   "codebook_bytes": D.CB_BYTES, "not_charged": "nothing else is persistent (FPS has no state; seeds are "
                   "constants)", "historical_note": "AM02 C1/C2 ledgers charged rows + owner (+ codebook), not ids",
                   "arms": ledger},
        "budgets": D.BUDGETS, "registered": list(D.REGISTERED),
        "arms": {"new": list(D.arms().keys()), "historical": {"C1_FPS_FP32": "AM02 C1 at B1 (full writer pool), AM03 cells",
                                                              "C2_FPS_PQ": "AM02 C2 bottle-only C0 at B1, AM03 cells"}},
        "b8_rule": {"diagnostic": "NORMAL_DIAG.json: for each category, median over SEL-probe patches (6 never-written "
                                  "images) of d(probe, final support_c) / d(probe, full SEL pool_c in the arm's codec); "
                                  "G_arm = mean over 15 categories of (median ratio - 1)",
                    "run_B8_if": "G_DOM_PQ_B4 >= 0.05 AND (G_DOM_PQ_B2 - G_DOM_PQ_B4) >= 0.25 * G_DOM_PQ_B2 AND projected "
                                 "B8 hours <= max_projected_hours", "max_projected_hours": 10.0,
                    "decided_before": "any real-anomaly evaluation of new arms (evaluate.py refuses without "
                                      "B8_DECISION.json; b8 refuses if eval/ exists)"},
        "scorer": "AM03/F2: exact FP64 1-NN (exact.nn) to the category support at the boundary, AM03 native_maps, image "
                  "= max, exps.cadfinal.metrics.exact_ap, sklearn roc_auc_score; 1,725 MVTec test images; masks only for "
                  "metrics; scorer must reproduce 3 preserved AM03 cells (AP 1e-6, AUROC 1e-9) before any new-arm cell",
        "dataset_manifest": {"am03_dataset_manifest_sha256": C.sha(A3.OUT / "DATASET_MANIFEST.json"),
                             "test_images": sum(len(v["images"]) for v in man["categories"].values()),
                             "train_feature_manifest_sha256": C.sha(C.FEATURES / "MANIFEST.json"),
                             "r15_digest": C.r15_digest()},
        "analysis_rules": AN.RULES,
        "caps": D.CAPS,
        "failure_policy": "a crash/OOM/kill is mechanical: streams resume from the last boundary whose checkpoint hash "
                          "matches its create-exclusive boundary record (state is fully determined by it); eval resumes "
                          "per cell (cells are deterministic functions of checkpoints). Only non-scientific engineering "
                          "fixes are allowed, recorded in AMENDMENTS.md with reason; budgets, rules, seeds and scorer never "
                          "change; no rerun because of results.",
        "devices": "CPU, 6 threads per process; GPU only as the certified FPS accelerator (decisions identical); "
                   "test scoring on CPU",
        "exposure": "MVTec test split historically exposed; any budget recommendation is exploratory",
    }
    D.OUT.mkdir(parents=True, exist_ok=True)
    D.write_json(D.PROTOCOL, protocol)
    lock = {"schema": "c2dom_sweep_lock_v1", "protocol_sha256": C.sha(D.PROTOCOL), "arms": list(D.arms(D.BUDGETS.keys())),
            "partition": pmeta, "b8_rule": protocol["b8_rule"], "b8_projection_hours": 8.5,
            "b8_projection_basis": "FPS work 2.3e11 row-steps at measured ~1.0e7/s (GPU certified) ~6.4 h + ~2 h scoring",
            "sources": {s: C.sha(C.ROOT / s) for s in SOURCES}, "fps_gpu_min_work": FPS.GPU_MIN_WORK, "utc": D.now()}
    D.write_json(D.LOCK, lock)
    print("LOCK", C.sha(D.LOCK), "PROTOCOL", lock["protocol_sha256"])
    for c, v in pmeta.items():
        print(c, v["fit_images"], v["sel_images"], v["sel_rows"])


if __name__ == "__main__":
    main()
