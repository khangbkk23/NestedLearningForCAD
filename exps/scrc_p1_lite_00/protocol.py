"""SCRC-P1-LITE-00 — frozen constants and lock verification (agents/plans/scrc_p1_lite_00_prereg.md).

Nothing here reads experimental data. Every constant is either a literal from the locked prereg / lock JSON or is
recomputed from a locked rule and asserted equal to the literal (``check_constants``).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/hope_cad/scrc_p1_lite_00"
PREREG = ROOT / "agents/plans/scrc_p1_lite_00_prereg.md"
PLAN = ROOT / "agents/plans/scrc_p1_lite_00_implementation_plan.md"
LOCK = ROOT / "agents/plans/scrc_p1_lite_00_lock.json"
LOCK_SHA = "4e8f6899894b46052611b9917c28daa7641a94d39323d5b62c68bbf56427be66"
ENGINEERING_LOCK = OUT / "ENGINEERING_LOCK.json"
SCORES_LOCK = OUT / "SCORES_LOCK.json"
GPU_LEDGER = OUT / "logs" / "GPU_LEDGER.jsonl"

ORDER = ("bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile",
         "toothbrush", "transistor", "wood", "zipper")
PATCHES, DIM = 784, 768

# ------------------------------------------------------------------ bank, support, ledger (prereg §3-§4)
C2DOM = ROOT / "results/hope_cad/nl_sgm_01a_c2_dom_sweep"
ARM_SOURCE = "DOM_PQ_B2"
BOUNDARY = 14
BUDGET = 15_360_000
SCRC_OVERHEAD = 98_464
STATE_PATH = ROOT / "results/hope_cad/nl_cad_pilot_00/p0/residuals/state_cumulative_t15.pt"
CKPT = lambda t: C2DOM / "streams" / ARM_SOURCE / f"ckpt_t{t:02d}.pt"            # noqa: E731
BOUNDARY_REC = lambda t: C2DOM / "streams" / ARM_SOURCE / f"boundary_t{t:02d}.json"  # noqa: E731
QUOTA_FULL = (6246,) * 5 + (6245,) * 10
QUOTA_BM = (6181,) * 5 + (6180,) * 10
ROWS_FULL, ROWS_BM = 93_680, 92_705
BYTES_R0, BYTES_C0, BYTES_M = 15_359_984, 15_261_509, 15_359_973
PQ_ROW_BYTES, CB_BYTES, HEADER_BYTES = 101, 393_216, 64

# ------------------------------------------------------------------ arms (prereg §5)
RANK = 32
GAMMAS = (0.25, 0.5, 1.0)
THETA_TAUS = (2.0, 1.0, 0.0, -1.0)
CONFIGS = {f"M{i + 1:02d}": (g, tt) for i, (g, tt) in enumerate((g, tt) for g in GAMMAS for tt in THETA_TAUS)}
METRIC_ARMS = tuple(CONFIGS)
ARMS = ("R0", "C0") + METRIC_ARMS
PV3_ARM = "C0_via_metric_path"            # gamma = 0 through the metric code path (PV3; hashed, not a scored arm)
# prereg §5 table, printed precision: tau (4 s.f.), c_min, c_max (4 d.p.), kappa_t, kappa_theta (3 d.p.)
TABLE5 = {"M01": (3.214, 0.0081, 0.0571, 1.061, 1.233), "M02": (1.182, 0.0207, 0.1139, 1.129, 1.375),
          "M03": (0.435, 0.0491, 0.1907, 1.236, 1.548), "M04": (0.160, 0.1012, 0.2747, 1.379, 1.750),
          "M05": (3.214, 0.0161, 0.1109, 1.125, 1.519), "M06": (1.182, 0.0410, 0.2148, 1.274, 1.890),
          "M07": (0.435, 0.0957, 0.3450, 1.527, 2.397), "M08": (0.160, 0.1921, 0.4739, 1.901, 3.063),
          "M09": (3.214, 0.0319, 0.2094, 1.265, 2.309), "M10": (1.182, 0.0803, 0.3835, 1.622, 3.574),
          "M11": (0.435, 0.1823, 0.5710, 2.331, 5.745), "M12": (0.160, 0.3473, 0.7232, 3.613, 9.380)}

# ------------------------------------------------------------------ folds, selection, statistics (prereg §8-§10)
FOLD_KEY = "SCRC-P1-LITE-00/fold/"
FOLDS = {"F1": ("bottle", "leather", "wood", "metal_nut", "zipper"),
         "F2": ("tile", "capsule", "pill", "carpet", "transistor"),
         "F3": ("cable", "toothbrush", "grid", "screw", "hazelnut")}
SEL_TIE_EPS = 1e-12
SCREEN_THRESHOLD = 0.005          # exploratory screen (NOT G-C1 +0.010, NOT Gate P1)
MECH_THRESHOLD = 0.005
AUROC_NI = -0.010
BOOT_B = 10_000
BOOT_SEED = 3_665_988_086
BOOT_KEY = "SCRC-P1-LITE-00/bootstrap/v1"
OUTCOMES = ("SCREEN_INVALID", "SCREEN_STOPPED_COST", "SCREEN_FAIL", "SCREEN_WEAK", "SCREEN_PASS")

# ------------------------------------------------------------------ probes (prereg §6.2)
PROBE_IMAGE_KEY = "SCRC-P1-LITE-00/probe/"
PROBE_PATCH_KEY = "SCRC-P1-LITE-00/probe-patch/"
N_PROBE_PATCHES = 64
PV5_ARMS = ("M01", "M12")
PV5_REL_TOL = 1e-12
FALLBACK_K = 64

# ------------------------------------------------------------------ caps (prereg §12)
CAPS = {"phase_S_gpu_h": 1.0, "phase_L_cpu_wall_h": 1.5, "total_wall_h": 3.0, "rss_gb": 8.5, "vram_gb": 4.5,
        "disk_gb": 1.0, "threads": 6}
N_TEST_IMAGES = 1725
U4_GPU_CAP_S = 0.02 * 3600       # checklist A3

# smoke-only native sizes (real Phase L reads hw from the manifest)
MVTEC_HW = {"bottle": (900, 900), "cable": (1024, 1024), "capsule": (1000, 1000), "carpet": (1024, 1024),
            "grid": (1024, 1024), "hazelnut": (1024, 1024), "leather": (1024, 1024), "metal_nut": (700, 700),
            "pill": (800, 800), "screw": (1024, 1024), "tile": (840, 840), "toothbrush": (1024, 1024),
            "transistor": (1024, 1024), "wood": (1024, 1024), "zipper": (1024, 1024)}


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def sha_file(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def sha_tensor(t):
    import torch
    t = t.detach().to("cpu").contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode() + str(tuple(t.shape)).encode())
    h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def theta_gamma(g):
    return math.log((g / 2) / (1 - g / 2))


def folds_from_hash():
    ranked = sorted(ORDER, key=lambda c: sha_text(FOLD_KEY + c))
    return {f"F{i + 1}": tuple(ranked[5 * i:5 * i + 5]) for i in range(3)}


def fold_of(c):
    return next(f for f, cs in FOLDS.items() if c in cs)


def boot_seed():
    return int.from_bytes(hashlib.sha256(BOOT_KEY.encode()).digest()[:4], "big")


def probe_image(c, n):
    return min(range(n), key=lambda j: sha_text(f"{PROBE_IMAGE_KEY}{c}/{j}"))


def probe_patches(c, k=N_PROBE_PATCHES):
    return sorted(range(PATCHES), key=lambda p: sha_text(f"{PROBE_PATCH_KEY}{c}/{p}"))[:k]


def quotas():
    """Recompute both quota vectors from the canonical ledger rule."""
    from exps.ad02 import hope_cad_ad02_partition as AP
    from exps.nl_sgm_01a_c2dom import common as D
    full = AP.per_group_quota(D.capacity("pq_dom", BUDGET, BOUNDARY), BOUNDARY + 1)
    bm = AP.per_group_quota(D.capacity("pq_dom", BUDGET - SCRC_OVERHEAD, BOUNDARY), BOUNDARY + 1)
    return tuple(full), tuple(bm)


def l0_bytes(rows):
    return rows * PQ_ROW_BYTES + (BOUNDARY + 1) * CB_BYTES + HEADER_BYTES


def check_constants():
    require(folds_from_hash() == FOLDS, "fold literal differs from the hash rule")
    require(boot_seed() == BOOT_SEED, "bootstrap seed differs from its rule")
    require(quotas() == (QUOTA_FULL, QUOTA_BM), "quota literal differs from the ledger rule")
    require(sum(QUOTA_FULL) == ROWS_FULL and sum(QUOTA_BM) == ROWS_BM, "row totals")
    require(l0_bytes(ROWS_FULL) == BYTES_R0 and l0_bytes(ROWS_BM) == BYTES_C0
            and BYTES_C0 + SCRC_OVERHEAD == BYTES_M and BYTES_R0 <= BUDGET and BYTES_M <= BUDGET, "ledger")
    require(len(CONFIGS) == 12 and len(ARMS) == 14, "arm count")
    lk = read_lock()
    for k, (g, tt) in CONFIGS.items():
        require(lk["frozen"]["configs"][k] == {"gamma": g, "theta_tau": tt}, f"config {k} differs from lock")
    fz = lk["frozen"]
    for f in FOLDS:
        require(tuple(fz["folds"][f]) == FOLDS[f], f"fold {f} differs from lock")
    require(fz["bootstrap"]["seed"] == BOOT_SEED and fz["bootstrap"]["B"] == BOOT_B, "bootstrap differs from lock")
    require(fz["screen_threshold_pixel_ap"] == SCREEN_THRESHOLD and fz["auroc_noninferiority"] == AUROC_NI, "thresholds")
    require(tuple(fz["metric_support"]["quota_full"]) == QUOTA_FULL
            and tuple(fz["metric_support"]["quota_byte_matched"]) == QUOTA_BM, "quotas differ from lock")
    require(fz["caps"]["phase_S_gpu_h"] == CAPS["phase_S_gpu_h"] and fz["caps"]["vram_gb"] == CAPS["vram_gb"]
            and fz["caps"]["rss_gb"] == CAPS["rss_gb"] and fz["caps"]["disk_gb"] == CAPS["disk_gb"]
            and fz["caps"]["total_wall_h"] == CAPS["total_wall_h"]
            and fz["caps"]["phase_L_cpu_wall_h"] == CAPS["phase_L_cpu_wall_h"], "caps differ from lock")
    require(tuple(fz["outcomes_in_order"]) == OUTCOMES, "outcome order")
    return True


def read_lock():
    require(sha_file(LOCK) == LOCK_SHA, "SCRC-P1-LITE-00 lock JSON changed")
    return json.loads(LOCK.read_text())


def verify_documents():
    """Lock, prereg, plan, frozen inputs, reused code and the old locks (all sha256)."""
    lk = read_lock()
    bad = []
    for group in ("documents", "inputs_sha256", "reused_code_sha256", "old_locks_verified_unchanged_sha256"):
        for p, h in lk[group].items():
            if sha_file(ROOT / p) != h:
                bad.append(p)
    require(not bad, f"hash mismatch: {bad}")
    return {"lock_sha256": LOCK_SHA, "checked": sum(len(lk[g]) for g in ("documents", "inputs_sha256",
                                                                         "reused_code_sha256",
                                                                         "old_locks_verified_unchanged_sha256"))}


NEW_CODE = ("__init__.py", "protocol.py", "custody.py", "support.py", "hybrid.py", "score.py", "evaluate.py",
            "test_p1_lite.py")


PREREG_AMENDMENT_01 = ROOT / "agents/plans/scrc_p1_lite_00_prereg_amendment_01.json"
C2DOM_RESULT_FILES = (            # never opened (PREREG_AMENDMENT_01, finding E1)
    "results/hope_cad/calm02/features/mvtec/MANIFEST.json", "results/hope_cad/nl_sgm_01a_am02/LOCK_r2.json",
    "results/hope_cad/nl_sgm_01a_am03/AM03_LOCK.json", "results/hope_cad/nl_sgm_01a_am03/DATASET_MANIFEST.json",
    "results/hope_cad/nl_sgm_01a_am03/METRICS.json", "results/hope_cad/nl_sgm_01a_f2/F2_RESULTS.json",
    "results/hope_cad/nl_sgm_01a_f2/F2_VERDICT.json")


def verify_c2dom_code_sources():
    """PV1 C2-DOM integrity (PREREG_AMENDMENT_01): the canonical ``verify_lock`` rules restricted to the non-result
    sources of the C2-DOM LOCK; LOCK.json / PROTOCOL.json themselves are checked by ``verify_documents``."""
    lk = json.loads((C2DOM / "LOCK.json").read_text())
    src = {s: h for s, h in lk["sources"].items() if s not in C2DOM_RESULT_FILES}
    require(set(lk["sources"]) - set(src) == set(C2DOM_RESULT_FILES), "C2-DOM LOCK result-file set changed")
    for a in sorted(C2DOM.glob("AMENDMENT_*.json")):
        am = json.loads(a.read_text())
        require(am["protocol_change"] is False, f"invalid C2-DOM amendment {a.name}")
        if am["test_data_opened_before"] is not False:
            require(am.get("post_eval_reporting_only") is True and set(am["sources"]) <= {
                "exps/nl_sgm_01a_c2dom/analyze.py", "exps/nl_sgm_01a_c2dom/common.py"}, f"invalid C2-DOM amendment {a.name}")
        require(not set(am["sources"]) & set(C2DOM_RESULT_FILES), f"C2-DOM amendment {a.name} touches a result file")
        src.update(am["sources"])
    bad = [s for s, h in src.items() if sha_file(ROOT / s) != h]
    require(not bad, f"C2-DOM: source changed after lock: {bad}")
    return {"verified_sources": len(src), "result_files_not_opened": len(C2DOM_RESULT_FILES)}


def verify_engineering_lock():
    """score/evaluate refuse to run unless every hash in ENGINEERING_LOCK.json, updated in order by the
    create-exclusive ENGINEERING_AMENDMENT_NN.json files (``files_sha256``), still matches."""
    require(ENGINEERING_LOCK.exists(), "ENGINEERING_LOCK.json missing")
    el = json.loads(ENGINEERING_LOCK.read_text())
    hashes = dict(el["sha256"])
    for a in sorted(OUT.glob("ENGINEERING_AMENDMENT_*.json")):
        am = json.loads(a.read_text())
        require(am.get("labels_opened_before") is False and am.get("protocol_change") is False,
                f"invalid amendment {a.name}")
        if a.name != "ENGINEERING_AMENDMENT_01.json":       # 01 is itself hashed in the lock and recorded pre-lock files
            hashes.update(am["files_sha256"])
    bad = [p for p, h in hashes.items() if sha_file(ROOT / p) != h]
    require(not bad, f"ENGINEERING_LOCK: changed after lock: {bad}")
    verify_documents()
    el["effective_sha256"] = hashes
    return el
