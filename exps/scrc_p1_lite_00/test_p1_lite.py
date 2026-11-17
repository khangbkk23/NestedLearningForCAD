"""SCRC-P1-LITE-00 pre-run tests U1-U9 (implementation plan §3) and the engineering smoke U10 (amendment 01).

    PYTHONDONTWRITEBYTECODE=1 python -m pytest exps/scrc_p1_lite_00/test_p1_lite.py -q -s -p no:cacheprovider

Custody phase PRE for the whole session: only train-side data (MVTec train eval-probe features, C2-DOM checkpoints
t13/t14, the P0 cumulative state) and synthetic data. U4 is the only test that uses the GPU (checklist A3,
<= 0.02 GPU-h cumulative, enforced through logs/GPU_LEDGER.jsonl). U10 runs on the CPU.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from exps.scrc_p1_lite_00 import custody as CU

CU.install("PRE")

from exps.scrc_p1_lite_00 import evaluate as EV            # noqa: E402
from exps.scrc_p1_lite_00 import hybrid as H               # noqa: E402
from exps.scrc_p1_lite_00 import protocol as P             # noqa: E402
from exps.scrc_p1_lite_00 import score as SC               # noqa: E402
from exps.scrc_p1_lite_00 import support as SU             # noqa: E402

torch.set_num_threads(6)
torch.backends.cuda.matmul.allow_tf32 = False
SCRATCH = Path(os.environ.get("SCRC_SCRATCH", tempfile.gettempdir()))
RESULTS = {}


def _utc():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _record(name, obj):
    """Create-exclusive per-run test record under results/hope_cad/scrc_p1_lite_00/tests/."""
    p = P.OUT / "tests" / f"{name}_{_utc()}.json"
    SC.write_json_x(p, obj)
    RESULTS[name] = str(p)
    return p


@pytest.fixture(scope="module")
def ck14():
    return SU.load_ckpt(P.BOUNDARY)[0]


@pytest.fixture(scope="module")
def supports(ck14):
    return SU.full_support(ck14), SU.byte_matched_support(ck14)


@pytest.fixture(scope="module")
def mets():
    return SU.metrics(SU.load_state())


@pytest.fixture(scope="module")
def probes():
    """1,568 MVTec train eval-probe patches (bottle and screw, first eval image each; not bank members)."""
    from exps.nl_sgm_01a import common as C
    from exps.nl_sgm_01a_c2dom import common as D
    co, _ = D.setup()
    F = C.Features()
    q = torch.cat([F.images([co[c]["eval"][0]]).reshape(-1, P.DIM).clone() for c in ("bottle", "screw")])
    F.drop()
    assert q.shape == (1568, 768) and q.dtype == torch.float32
    return q


# ------------------------------------------------------------------ U1 folds and constants

def test_u1_folds_and_constants():
    f = P.folds_from_hash()
    assert f == P.FOLDS
    allc = [c for cs in f.values() for c in cs]
    assert sorted(allc) == sorted(P.ORDER) and len(set(allc)) == 15
    assert P.check_constants()
    v = P.verify_documents()
    assert v["lock_sha256"] == P.LOCK_SHA
    assert P.boot_seed() == 3665988086
    _record("U1", {"folds": f, "documents": v, "pass": True})


# ------------------------------------------------------------------ U2 FPS prefix property on real data

def test_u2_fps_prefix(ck14):
    from exps.csg01.select import select_fps
    from exps.nl_sgm_01a_c2dom.fps import GPU_MIN_WORK
    ck13, rec13 = SU.load_ckpt(13)
    rec14 = SU.boundary_record(14)
    rows13 = rec13["per_category_rows"]
    cats = sorted(P.ORDER[:14], key=lambda c: (rows13[c], P.ORDER.index(c)))[:2]
    out = {"categories": cats, "per_category": {}}
    for c in cats:
        k = P.ORDER.index(c)
        x13, g13 = SU.owner_rows(ck13, k)
        q, qp = P.QUOTA_FULL[k], P.QUOTA_BM[k]
        assert len(x13) * q < GPU_MIN_WORK                                  # the writer used the canonical CPU rule
        calls = [r for r in rec14["fps_calls"] if r["category"] == k]
        assert len(calls) == 1 and calls[0]["impl"] == "cpu_canonical" and calls[0]["k"] == q
        t0 = time.time()
        idx = select_fps(x13, q)
        _, g14 = SU.owner_rows(ck14, k)
        idx_p = select_fps(x13, qp)
        _, g14p = SU.owner_rows(ck14, k, qp)
        rec = {"n13": len(x13), "q": q, "q_prime": qp, "writer_order_reproduced": bool(torch.equal(g13[idx], g14)),
               "prefix_property": bool(np.array_equal(idx_p, idx[:qp])),
               "byte_matched_equals_writer_decision": bool(torch.equal(g13[idx_p], g14p)),
               "seconds": time.time() - t0}
        out["per_category"][c] = rec
        assert rec["writer_order_reproduced"] and rec["prefix_property"] and rec["byte_matched_equals_writer_decision"]
    out["impl_t14"] = {P.ORDER[r["category"]]: r["impl"] for r in rec14["fps_calls"]}
    out["pass"] = True
    _record("U2", out)


# ------------------------------------------------------------------ U3 supports, ledger, equality with P0 bank

def test_u3_supports(supports):
    S, Sp = supports
    chk = SU.checks(S, Sp)
    assert chk["pass"], chk
    code = ("import json; from exps.nl_cad_pilot_00 import p0_real as R; from exps.scrc_p1_lite_00 import protocol as P;"
            "b,_,_=R.canonical_bank(14); print('HASH', json.dumps({'ids': P.sha_tensor(b.ids), 'x': P.sha_tensor(b.x)}))")
    env = dict(os.environ, PYTHONPATH=str(P.ROOT), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", code], cwd=P.ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    h = json.loads(r.stdout.split("HASH", 1)[1])
    eq = h["ids"] == chk["support_sha"]["S14_ids"] and h["x"] == chk["support_sha"]["S14_x"]
    assert eq
    _record("U3", {**chk, "p0_canonical_bank_hash": h, "S14_equals_p0_canonical_bank": eq})


# ------------------------------------------------------------------ U4 hybrid (GPU) == CPU certified (GPU, A3)

def _gpu_used_s():
    if not P.GPU_LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["gpu_wall_s"] for l in P.GPU_LEDGER.read_text().splitlines() if l.strip())


def _synthetic_ties(seed=7):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(400, 768, generator=g)
    x = torch.cat([base, base[:50], base[100:150] * 1.0]).contiguous()          # exact duplicate rows
    perm = torch.randperm(len(x), generator=g)
    x = x[perm].contiguous()
    ids = torch.arange(len(x)) * 3 + 11
    q = torch.cat([base[:20], (base[20:40] + base[60:80]) / 2, torch.randn(40, 768, generator=g)]).contiguous()
    return q, x, ids


def test_u4_hybrid_gpu_equals_cpu_certified(supports, mets, probes):
    from exps.nl_cad_pilot_00 import nnx
    from exps.nl_cad_pilot_00.scrc import apply_sqrt
    assert torch.cuda.is_available(), "U4 requires CUDA (checklist A3)"
    used = _gpu_used_s()
    assert used < P.U4_GPU_CAP_S - 30, f"A3 GPU budget nearly exhausted ({used:.1f} s used)"
    S, Sp = supports
    arms = {"C0": None, "M01": mets["M01"], "M12": mets["M12"]}
    # CPU references first (outside the GPU clock)
    ref, banks = {}, {}
    for a, m in arms.items():
        banks[a] = H.ScreenRows(Sp.x, Sp.ids, m)
        stc = {}
        ref[a] = nnx.certified_nn(probes, banks[a].cpu, m, stats=stc) + (stc,)
    qs, xs, ids_s = _synthetic_ties()                       # ids increasing; duplicate rows at permuted positions
    syn = {"euclid": (None, H.ScreenRows(xs, ids_s, None)), "M12": (mets["M12"], H.ScreenRows(xs, ids_s, mets["M12"]))}
    syn_ref = {k: nnx.certified_nn(qs, sr.cpu, m) for k, (m, sr) in syn.items()}
    from exps.nl_sgm_01a import exact as X
    syn_brute = {"euclid": X.brute(qs, xs, ids_s), "M12": nnx.brute_metric(qs, xs, ids_s, mets["M12"])}
    # ---------------------------------------------------------------- GPU clock
    t0 = time.time()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    out = {"arms": {}, "synthetic": {}}
    try:
        for a, m in arms.items():
            sr = banks[a].to("cuda")
            st = {}
            ta = time.time()
            d, i = H.hybrid_certified_nn(probes, sr, stats=st)
            dt = time.time() - ta
            sr.release()
            dr, ir, stc = ref[a]
            out["arms"][a] = {"dist_bitwise": bool(torch.equal(d, dr)), "ids_equal": bool(torch.equal(i, ir)),
                              "candidates_gpu": st["candidates"], "candidates_cpu": stc["candidates"],
                              "seconds": dt, "ms_per_image": 1e3 * dt / 2}
        for k, (m, sr) in syn.items():
            d, i = H.hybrid_certified_nn(qs, sr.to("cuda"), q_chunk=37, r_chunk=101)
            sr.release()
            out["synthetic"][k] = {"vs_cpu_certified": bool(torch.equal(d, syn_ref[k][0]) and torch.equal(i, syn_ref[k][1])),
                                   "ids_vs_brute": bool(torch.equal(i, syn_brute[k][1])),
                                   "zero_distance_queries": int((syn_brute[k][0] == 0).sum()),
                                   "zero_distances_exact": bool((d[syn_brute[k][0] == 0] == 0).all()),
                                   "max_rel_vs_brute": float(((d - syn_brute[k][0]).abs()
                                                              / syn_brute[k][0].clamp_min(1e-300))
                                                             [syn_brute[k][0] > 0].max())}
        # screen error on the device (Euclidean rows; metric z-space rows)
        qg = probes[:256].cuda()
        out["screen_error_ratio_euclid"] = nnx.screen_error_ratio(qg, Sp.x.cuda())
        zq = apply_sqrt(probes[:256], mets["M12"]).float().cuda()
        out["screen_error_ratio_M12_zspace"] = nnx.screen_error_ratio(zq, banks["M12"].cpu.z.cuda())
        del qg, zq
        torch.cuda.empty_cache()
        # PV5 oracle machinery on the GPU: timing + agreement on 4 probe queries
        q4 = probes[:4]
        tb = time.time()
        bd, bi = nnx.brute_metric(q4.cuda(), Sp.x.cuda(), Sp.ids.cuda(), mets["M12"])
        torch.cuda.synchronize()
        tbm = time.time() - tb
        dr, ir, _ = ref["M12"]
        out["pv5_brute_metric_gpu"] = {"queries": 4, "seconds": tbm, "s_per_query": tbm / 4,
                                       "id_mismatches": int((bi.cpu() != ir[:4]).sum()),
                                       "max_rel_diff": float(((bd.cpu() - dr[:4]).abs() / dr[:4]).max())}
        out["vram_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    finally:
        torch.cuda.synchronize()
        gpu_s = time.time() - t0
        P.GPU_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with P.GPU_LEDGER.open("a") as f:
            f.write(json.dumps({"test": "U4", "utc": _utc(), "gpu_wall_s": gpu_s}) + "\n")
    out["gpu_wall_s_this_run"] = gpu_s
    out["gpu_wall_s_cumulative"] = _gpu_used_s()
    out["gpu_cap_s"] = P.U4_GPU_CAP_S
    out["device"] = torch.cuda.get_device_name(0)
    ok = (all(v["dist_bitwise"] and v["ids_equal"] for v in out["arms"].values())
          and all(v["vs_cpu_certified"] and v["ids_vs_brute"] and v["max_rel_vs_brute"] <= 1e-12
                  and v["zero_distances_exact"]
                  for v in out["synthetic"].values())
          and out["screen_error_ratio_euclid"] <= 1.0 and out["screen_error_ratio_M12_zspace"] <= 1.0
          and out["pv5_brute_metric_gpu"]["id_mismatches"] == 0
          and out["pv5_brute_metric_gpu"]["max_rel_diff"] <= P.PV5_REL_TOL
          and out["gpu_wall_s_cumulative"] <= P.U4_GPU_CAP_S)
    out["pass"] = ok
    _record("U4", out)
    assert ok, out


# ------------------------------------------------------------------ U5 custody

def test_u5_custody(tmp_path):
    R = P.ROOT
    cases = {
        "PRE": {str(R / "results/hope_cad/calm02/features/mvtec/bottle_test.pt"): False,
                str(R / "results/hope_cad/calm02/features/mvtec/bottle_train.pt"): True,
                str(R / "results/hope_cad/nl_sgm_01a_am03/DATASET_MANIFEST.json"): False,
                str(R / "data/mvtec/bottle/ground_truth/broken_large/000_mask.png"): False,
                str(R / "results/hope_cad/calm02/features/visa/candle_train.pt"): False,
                str(R / "results/hope_cad/nl_sgm_01a_c2_dom_sweep/eval/DOM_PQ_B2/x.json"): False,
                str(R / "results/hope_cad/nl_sgm_01a_c2_dom_sweep/streams/DOM_PQ_B2/ckpt_t14.pt"): True},
        "S": {str(R / "results/hope_cad/calm02/features/mvtec/bottle_test.pt"): True,
              str(R / "results/hope_cad/calm02/features/mvtec/bottle_train.pt"): False,
              str(R / "results/hope_cad/nl_sgm_01a_am03/DATASET_MANIFEST.json"): False,
              str(R / "data/mvtec/bottle/ground_truth/broken_large/000_mask.png"): False,
              str(R / "data/mvtec/bottle/test/broken_large/000.png"): False,
              str(R / "data/visa/candle/Data/Masks/x.png"): False,
              str(R / "results/hope_cad/nl_sgm_01a_am03/cells/C1/x.json"): False,
              str(R / "results/hope_cad/other/METRICS.json"): False},
        "L": {str(R / "results/hope_cad/calm02/features/mvtec/bottle_test.pt"): True,
              str(R / "results/hope_cad/nl_sgm_01a_am03/DATASET_MANIFEST.json"): True,
              str(R / "data/mvtec/bottle/ground_truth/broken_large/000_mask.png"): True,
              str(R / "data/mvtec/bottle/test/broken_large/000.png"): False,
              str(R / "results/hope_cad/other/DATASET_MANIFEST.json"): False,
              str(R / "data/visa/candle/Data/Masks/x.png"): False,
              str(R / "results/hope_cad/nl_sgm_01a_c2_dom_sweep/eval/x.json"): False,
              str(R / "results/hope_cad/calm02/features/mvtec/bottle_train.pt"): False,
              str(R / "results/hope_cad/FINAL/x.json"): False}}
    got = {ph: {p: CU.decide(p, ph)[0] for p in cs} for ph, cs in cases.items()}
    assert got == {ph: cs for ph, cs in cases.items()}
    # the installed hook (phase PRE) refuses a real manifest/VisA open before any byte is read
    assert CU.phase() == "PRE"
    for p in (R / "results/hope_cad/nl_sgm_01a_am03/DATASET_MANIFEST.json",
              R / "results/hope_cad/calm02/features/visa/candle_train.pt",
              R / "results/hope_cad/calm02/features/mvtec/bottle_test.pt"):
        with pytest.raises(PermissionError):
            open(p, "rb")
    # key guard on a synthetic feature file
    f = tmp_path / "synthetic_test.pt"
    torch.save({"features": torch.zeros(2, 784, 768), "rows": [{"relative_path": "bottle/test/broken/000.png"}]}, f)
    try:
        CU.set_phase("S")
        ff = CU.guarded_features(f)
        assert ff.features.shape == (2, 784, 768) and ff["features"].shape == (2, 784, 768)
        with pytest.raises(PermissionError):
            ff["rows"]
        nonexist = R / "data/mvtec/__nonexistent__/ground_truth/x/000_mask.png"
        with pytest.raises(PermissionError):
            open(nonexist, "rb")                                   # phase S: ground truth refused by the hook
        CU.set_phase("L")
        assert CU.guarded_features(f)["rows"][0]["relative_path"].startswith("bottle/")
        with pytest.raises(FileNotFoundError):
            open(nonexist, "rb")                                   # phase L: allowed by custody, file absent
        with pytest.raises(PermissionError):
            open(R / "data/visa/__nonexistent__.png", "rb")
    finally:
        CU.set_phase("PRE")
    assert CU.phase() == "PRE"
    _record("U5", {"decisions": got, "pass": True})


# ------------------------------------------------------------------ U6 gamma = 0 identity (CPU)

def test_u6_gamma0_identity(supports, mets, probes):
    from exps.nl_cad_pilot_00 import nnx
    from exps.nl_sgm_01a import exact as X
    S, Sp = supports
    e = H.ScreenRows(Sp.x, Sp.ids, None)
    g0 = H.ScreenRows(Sp.x, Sp.ids, mets[P.PV3_ARM])
    se, s0 = {}, {}
    d_e, i_e = H.hybrid_certified_nn(probes, e, stats=se)
    d_0, i_0 = H.hybrid_certified_nn(probes, g0, stats=s0)
    d_n, i_n = nnx.certified_nn(probes, e.cpu, None)
    d_x, i_x = X.nn(probes, Sp)
    rec = {"z_equals_raw": bool(torch.equal(g0.cpu.z, Sp.x)),
           "euclid_vs_metric_path": bool(torch.equal(d_e, d_0) and torch.equal(i_e, i_0)),
           "candidates_equal": se["candidates"] == s0["candidates"],
           "hybrid_cpu_vs_nnx": bool(torch.equal(d_e, d_n) and torch.equal(i_e, i_n)),
           "hybrid_cpu_vs_exact_nn": bool(torch.equal(d_e, d_x) and torch.equal(i_e, i_x))}
    rec["pass"] = all(rec.values())
    _record("U6", rec)
    assert rec["pass"], rec


# ------------------------------------------------------------------ U7 metric construction and sandwich (CPU)

def test_u7_metrics_and_sandwich(supports, mets, probes):
    S, Sp = supports
    d_e, _ = H.hybrid_certified_nn(probes, H.ScreenRows(Sp.x, Sp.ids, None))
    out = {}
    for k, (g, tt) in P.CONFIGS.items():
        m = mets[k]
        T = P.TABLE5[k]
        got = (m.tau, float(m.c.min()), float(m.c.max()), m.kappa_t, m.kappa_theta)
        table_ok = (f"{got[0]:.3f}" == f"{T[0]:.3f}" and f"{got[1]:.4f}" == f"{T[1]:.4f}"
                    and f"{got[2]:.4f}" == f"{T[2]:.4f}" and f"{got[3]:.3f}" == f"{T[3]:.3f}"
                    and f"{got[4]:.3f}" == f"{T[4]:.3f}")
        assert m.gamma == g and not m.is_identity
        d_m, _ = H.hybrid_certified_nn(probes, H.ScreenRows(Sp.x, Sp.ids, m))
        r = d_m / d_e
        lo = math.sqrt(m.mu_t)
        sand = bool(((r >= lo * (1 - 1e-12)) & (r <= 1 + 1e-12)).all())
        out[k] = {"table_ok": table_ok, "values": got, "sandwich_ok": sand, "ratio_min": float(r.min()),
                  "ratio_max": float(r.max()), "sqrt_mu_t": lo, "kappa_theta_lt_10": m.kappa_theta < 10}
    ok = all(v["table_ok"] and v["sandwich_ok"] and v["kappa_theta_lt_10"] for v in out.values())
    _record("U7", {"configs": out, "pass": ok})
    assert ok, out


# ------------------------------------------------------------------ U8 metric implementations (synthetic)

def test_u8_metric_functions():
    from sklearn.metrics import average_precision_score, roc_auc_score
    from exps.cadfinal.metrics import exact_ap
    from exps.nl_sgm_01a import am03_eval as A3
    rng = np.random.default_rng(3)
    maps = [rng.random((64, 80), dtype=np.float32) for _ in range(5)]
    masks = [rng.random((64, 80)) < 0.1 for _ in range(5)]
    a = exact_ap(maps, masks)["ap"]
    b = average_precision_score(np.concatenate([m.ravel() for m in masks]), np.concatenate([m.ravel() for m in maps]))
    q = [np.round(m * 20).astype(np.float32) / 20 for m in maps]            # heavy ties
    at = exact_ap(q, masks)["ap"]
    bt = average_precision_score(np.concatenate([m.ravel() for m in masks]), np.concatenate([m.ravel() for m in q]))
    d = torch.rand(3, 784, dtype=torch.float64)
    nm = A3.native_maps(d, (100, 120))
    rec, h = EV.category_metrics(d, np.array([0, 1, 1], dtype=np.int8),
                                 [np.zeros((100, 120), bool), np.ones((100, 120), bool),
                                  np.pad(np.ones((50, 60), bool), ((0, 50), (0, 60)))], (100, 120), hi_pool=1.0)
    out = {"exact_vs_sklearn_no_ties": abs(a - b), "exact_vs_sklearn_ties": abs(at - bt),
           "native_maps_shape": list(nm.shape), "native_maps_dtype": str(nm.dtype),
           "category_metrics_keys": sorted(rec), "hist_binning_error": rec["hist_binning_error"]}
    ok = (abs(a - b) <= 1e-9 and abs(at - bt) <= 1e-9 and nm.shape == (3, 100, 120) and nm.dtype == np.float32
          and 0 <= rec["pixel_ap"] <= 1 and rec["image_auroc"] == roc_auc_score([0, 1, 1], nm.reshape(3, -1).max(1)))
    out["pass"] = bool(ok)
    _record("U8", out)
    assert ok, out


# ------------------------------------------------------------------ U9 statistics and outcome rules (synthetic)

def _tables(effect=None, base_seed=0, auc_shift=0.0):
    rng = np.random.default_rng(base_seed)
    base = {c: 0.5 + 0.3 * rng.random() for c in P.ORDER}
    ap = {a: {c: base[c] + (0.0 if a != "C0" else -0.001) for c in P.ORDER} for a in P.ARMS}
    auc = {a: {c: 0.9 for c in P.ORDER} for a in P.ARMS}
    for j, (c_eff) in (effect or {}).items():
        for c in P.ORDER:
            ap[j][c] += c_eff[c] if isinstance(c_eff, dict) else c_eff
            auc[j][c] += auc_shift
    return ap, auc


def test_u9_statistics():
    out = {}
    # selection tie-break: equal within 1e-12 -> lower ID; strictly better by > 1e-12 -> that ID
    ap, _ = _tables({"M03": 0.01, "M07": 0.01 + 5e-13})
    j, _ = EV.select_config(ap, list(P.ORDER))
    out["tie_lower_id"] = j == "M03"
    ap, _ = _tables({"M03": 0.01, "M07": 0.01 + 5e-11})
    j, _ = EV.select_config(ap, list(P.ORDER))
    out["strict_better"] = j == "M07"
    # selection uses only the 10 non-held-out categories
    eff = {c: (0.05 if c in P.FOLDS["F1"] else 0.0) for c in P.ORDER}
    ap, _ = _tables({"M09": eff, "M02": 0.001})
    sel = EV.fold_selection(ap)
    out["heldout_not_used"] = sel["F1"]["selected"] == "M02" and sel["F2"]["selected"] == "M09"
    # bootstrap determinism and seed
    arr = {"x": np.linspace(-0.01, 0.02, 15)}
    b1, b2 = EV.bootstrap(arr), EV.bootstrap(arr)
    out["bootstrap_deterministic"] = b1 == b2 and b1["x"]["seed"] == P.BOOT_SEED and b1["x"]["B"] == 10_000
    # exact sign flip == brute enumeration
    d = np.random.default_rng(5).normal(0.003, 0.01, 15)
    p, n = EV.sign_flip_p(d)
    brute = [np.mean([s * v for s, v in zip(sg, d)]) for sg in itertools.product((1, -1), repeat=15)]
    obs = np.mean(d)
    pb = np.mean([v >= obs - 1e-15 for v in brute])
    out["sign_flip_vs_brute"] = abs(p - pb) <= 1.0 / n and n == 32768
    # outcome order
    D = EV.decide
    out["order"] = (D(True, True, 1, 1, 0, {"a": 1, "b": 1, "c": 1}, 1) == "SCREEN_INVALID"
                    and D(False, True, 1, 1, 0, {"a": 1, "b": 1, "c": 1}, 1) == "SCREEN_STOPPED_COST"
                    and D(False, False, 0.0049, 1, 0, {"a": 1, "b": 1, "c": 1}, 1) == "SCREEN_FAIL"
                    and D(False, False, 0.02, 0.02, -0.011, {"a": 1, "b": 1, "c": 1}, 1) == "SCREEN_FAIL"
                    and D(False, False, 0.02, 0.004, 0, {"a": 1, "b": 1, "c": 1}, 1) == "SCREEN_WEAK"
                    and D(False, False, 0.02, 0.02, 0, {"a": 1, "b": -1, "c": -1}, 1) == "SCREEN_WEAK"
                    and D(False, False, 0.02, 0.02, 0, {"a": 1, "b": 1, "c": -1}, 0.0) == "SCREEN_WEAK"
                    and D(False, False, 0.005, 0.005, -0.010, {"a": 1, "b": 1, "c": -1}, 1e-9) == "SCREEN_PASS")
    # end to end on synthetic tables
    ap, auc = _tables({"M05": 0.02})
    r = EV.screen(ap, auc)
    out["e2e_pass"] = (r["outcome"] == "SCREEN_PASS" and all(v["selected"] == "M05" for v in r["fold_selection"].values())
                       and abs(r["S_net"] - 0.02) < 1e-12 and abs(r["S_mech"] - 0.021) < 1e-12)
    ap, auc = _tables({})
    out["e2e_fail"] = EV.screen(ap, auc)["outcome"] == "SCREEN_FAIL"
    ap, auc = _tables({"M05": 0.02}, auc_shift=-0.02)
    out["e2e_auroc_fail"] = EV.screen(ap, auc)["outcome"] == "SCREEN_FAIL"
    out["pass"] = all(out.values())
    _record("U9", out)
    assert out["pass"], out


# ------------------------------------------------------------------ U10 engineering smoke (amendment 01; CPU)

def _smoke(out_dir):
    """Run in a fresh CPU-only process (CUDA hidden): Phase S + Phase L code paths on train probes / synthetic labels."""
    out_dir = Path(out_dir)
    t0 = time.time()
    sl = SC.run("smoke", out_dir, "cpu", SC.smoke_queries_factory(2), n_probe_patches=1, n_total_images=30)
    ts = time.time() - t0
    res = EV.run("smoke", out_dir, EV.smoke_labels)
    rec = {"out_dir": str(out_dir), "phase_S_seconds": ts, "phase_L_seconds": res["phase_L_wall_s"],
           "pv": {k: v["pass"] for k, v in sl["pv"].items()}, "pv_all_pass": sl["pv_all_pass"],
           "outcome_on_synthetic_labels": res["outcome"],
           "n_score_files": len(list((out_dir / "scores").rglob("*.pt"))),
           "access_log_phases": sorted({a["phase"] for a in sl["access_log"]}),
           "pv1_c2dom_lock": sl["pv"]["PV1"]["c2dom_lock"],
           "phase_L_cuda_initialized": res["cuda_initialized"],
           "pv5": sl["pv"]["PV5"]["per_arm"],
           "fallback_diag": sl["diagnostics"]["p5_fallback_diagnostic"]["per_arm"],
           "projection": sl["timing"].get("projection_after_first_category"),
           "arm_seconds_total_cpu": sl["diagnostics"]["arm_seconds_total"]}
    print("SMOKE_JSON", json.dumps(rec))


def test_u10_smoke_end_to_end():
    out_dir = SCRATCH / f"scrc_p1_lite_00_smoke_{_utc()}"
    code = f"from exps.scrc_p1_lite_00 import test_p1_lite as T; T._smoke({str(out_dir)!r})"
    env = dict(os.environ, PYTHONPATH=str(P.ROOT), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-c", code], cwd=P.ROOT, env=env, capture_output=True, text=True,
                       timeout=3600)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    rec = json.loads(r.stdout.split("SMOKE_JSON", 1)[1])
    ok = (rec["pv_all_pass"] and rec["outcome_on_synthetic_labels"] in P.OUTCOMES and rec["n_score_files"] == 14 * 15
          and rec["access_log_phases"] == ["PRE"] and rec["phase_L_cuda_initialized"] is False)
    rec["pass"] = bool(ok)
    _record("U10_SMOKE", rec)
    assert ok, rec
