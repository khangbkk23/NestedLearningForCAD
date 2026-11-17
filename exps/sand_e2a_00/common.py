"""SAND-E2A-00 (task agents/tasks/task_sand_e2a_00.md): constants, custody, lock, witness calibrators, gate.

Preregistration: agents/plans/sand_e2a_00_prereg.md (hashed into LOCK.json). SAND-FINAL artifacts are reused
read-only; the canonical SAND / AM03 functions are imported by the runner and never edited (their sources are hashed
into the lock).

    python -m exps.sand_e2a_00.common lock
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/hope_cad/sand_e2a_00"
PREREG = ROOT / "agents/plans/sand_e2a_00_prereg.md"
TASK = ROOT / "agents/tasks/task_sand_e2a_00.md"
LOCK, CALIB, CALIB_LOCK = OUT / "LOCK.json", OUT / "CALIB.json", OUT / "CALIB_LOCK.json"
GATE_FILE, RESULT = OUT / "GATE.json", OUT / "RESULT.json"
CELL_OUT = OUT / "cells"

SAND = ROOT / "results/hope_cad/sand_final"
TAG = "A3_s2711"
GRIDS, CALDIR, CELLS = SAND / TAG / "grids", SAND / TAG / "calib", SAND / TAG / "cells"
DIST = SAND / "dist" / "DOM_PQ_B2"
MEM = ROOT / "results/hope_cad/nl_sgm_01a_c2_dom_sweep/streams/DOM_PQ_B2"
RUN = ROOT / "results/hope_cad/vitb8_nl00/runs/q2off_seed2711"
A1_OUT = ROOT / "results/hope_cad/vitb8_nl00"
MANIFEST = ROOT / "results/hope_cad/nl_sgm_01a_am03/DATASET_MANIFEST.json"
MVTEC = ROOT / "data/mvtec"
HOME_CACHE = Path.home() / "hope_cad_cache"
SAND_PUBLISHED = {"PHASE0_RESULTS.json": "1da1f0cbf1fa88c937e496d1871c34ec195c1e6d773784d51805d8ab9894340a",
                  "PHASE0_VERDICT.json": "5d2a76d785cc563c35d58ede48e4d69c698525bf4ecda0d4e3f1912e2ca2083a",
                  "BYTE_LEDGER.json": "819937ed3438cfc0ae1615c3c5159e0e6aa91e0baa0156b4f70517474c7f49cd"}

ORDER = ("bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw",
         "tile", "toothbrush", "transistor", "wood", "zipper")
NT, TOKENS, DIM, N_SEL = 15, 784, 768, 6
KEYS = [(t, c) for t in range(NT) for c in ORDER[:t + 1]]          # (boundary index t, category); stage = t + 1
CH = ("theta", "m")
CAL_KEYS = ("mu_theta", "sd_theta", "mu_m", "sd_m")

# Final macro values published by SAND-FINAL (PHASE0_RESULTS.json, sha above); known before this design.
PUBLISHED = {"A1": (0.5312241425306976, 0.919951919418113), "A2": (0.5322794415889963, 0.9088841782225171),
             "A3": (0.5366357177250488, 0.9145194873599384), "D1": (0.5759653213602504, 0.9296782619887713),
             "TH_P": (0.5381271260937756, 0.9105765136817214)}
SINGLES = ("A1", "A2", "TH_P")
GATE = {"min_final_pixel_ap_gain": 0.010, "max_final_image_auroc_loss": 0.010, "boot_lower_bound_gt": 0.0,
        "boot_B": 10000, "boot_seed": 20261116, "single_detectors": SINGLES}
TOL = {"d1_calib_rel": 1e-12, "a3_parity": 1e-9, "d1_parity": 1e-6, "published": 1e-12}
CAPS = {"wall_h_total": 3.0, "rss_gb": 8.5}
BYTES = {"fp32_grid": TOKENS * DIM * 4, "a3_total": 318864848, "witness_anchor_per_category": 4 * 8,
         "d1_extra": 216760320}
SOURCES = ("exps/sand_e2a_00/__init__.py", "exps/sand_e2a_00/common.py", "exps/sand_e2a_00/run.py",
           "exps/sand_e2a_00/test_sand_e2a.py")
CANONICAL = ("exps/sand/common.py", "exps/sand/score.py", "exps/nl_sgm_01a/am03_eval.py", "exps/cadfinal/metrics.py",
             "exps/calm02/native.py", "exps/task_p/p1_readout.py", "exps/task_p/p0_normal_support.py")


class ProtocolError(RuntimeError):
    pass


def require(cond, msg):
    if not cond:
        raise ProtocolError(msg)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (tuple, set)):
        return list(o)
    raise TypeError(type(o))


def write_json_x(path, value):
    """Create-exclusive JSON write; returns the file's sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=1, sort_keys=True, allow_nan=True, default=_default)
    return sha_file(path)


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


# ------------------------------------------------------------------ custody (prereg section 6)

_STATE = {"phase": None, "installed": False, "sand_lock_sources": frozenset()}
_LOG = []


def _governed(s):
    roots = (str(ROOT / "results"), str(ROOT / "data"), str(HOME_CACHE))
    return any(s == r or s.startswith(r + "/") for r in roots)


def _under(p, d):
    return str(p).startswith(str(Path(d).resolve()) + "/")


def decide(path, phase):
    """True if `path` may be opened in `phase` ('C' label-free calibration, 'L' label phase). Everything under
    results/, data/ and ~/hope_cad_cache is denied unless listed; other paths (sources, libraries) are ungoverned."""
    try:
        p = Path(os.fsdecode(path)).resolve()
    except Exception:
        return True
    if not _governed(str(p)):
        return True
    if p == OUT.resolve() or _under(p, OUT):
        return True
    if _under(p, GRIDS) or _under(p, CALDIR) or _under(p, DIST):
        return True
    if p.parent == MEM.resolve() and re.fullmatch(r"boundary_t\d\d\.json", p.name):
        return True
    if p.parent == RUN.resolve() and re.fullmatch(r"STAGE\d\d\.json", p.name):
        return True
    if phase != "L":
        return False
    if p in {(SAND / n).resolve() for n in ("LOCK.json", "PROTOCOL.json", *SAND_PUBLISHED)}:
        return True
    if p in _STATE["sand_lock_sources"] or p == MANIFEST.resolve() or _under(p, CELLS):
        return True
    if p.parent == A1_OUT.resolve() and re.fullmatch(r"EVAL_STAGE\d\d\.json", p.name):
        return True
    if _under(p, MVTEC):
        parts = p.relative_to(MVTEC.resolve()).parts
        return len(parts) >= 3 and parts[0] in ORDER and parts[1] == "ground_truth"
    return False


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        ph = _STATE["phase"]
        if not decide(args[0], ph):
            _LOG.append({"denied": os.fsdecode(args[0]), "phase": ph})
            raise PermissionError(f"SAND-E2A-00 custody ({ph}): refused {os.fsdecode(args[0])}")
        p = os.fsdecode(args[0])
        try:
            governed = _governed(str(Path(p).resolve()))
        except Exception:
            governed = False
        if governed:
            _LOG.append({"path": p, "phase": ph})


def install_custody(phase):
    """Phase C: label-free calibration. Phase L0: label phase before CALIB_LOCK is verified (same allow-list as C);
    set_phase('L') then opens the label allow-list."""
    require(phase in ("C", "L0"), "custody starts in phase C or L0")
    _STATE["phase"] = phase
    if not _STATE["installed"]:
        sys.addaudithook(_audit)
        _STATE["installed"] = True


def set_phase(phase):
    """Phase L0 -> L: called only after LOCK and CALIB_LOCK were verified (prereg section 6)."""
    require(_STATE["phase"] == "L0" and phase == "L", "illegal phase transition")
    _STATE["phase"] = "L"
    mark("phase_L_entered")
    srcs = read_json(SAND / "LOCK.json")["sources"]
    _STATE["sand_lock_sources"] = frozenset((ROOT / s).resolve() for s in srcs if s.startswith(("results/", "data/")))


def mark(label):
    _LOG.append({"mark": label, "utc": now()})


def access_log():
    return list(_LOG)


# ------------------------------------------------------------------ inputs and lock

def dist_path(c, t, bnd=None):
    b = bnd if bnd is not None else read_json(MEM / f"boundary_t{t:02d}.json")
    return DIST / f"{c}_{b['per_category_gids_sha'][c][:24]}.npz"


def input_files():
    """Label-free SAND-FINAL inputs: decoder stage records, C2-DOM boundary records, decoder grids, acquisition
    calibrators and the support-distance files the boundaries reference."""
    bnd = [read_json(MEM / f"boundary_t{t:02d}.json") for t in range(NT)]
    files = [MEM / f"boundary_t{t:02d}.json" for t in range(NT)] + [RUN / f"STAGE{t + 1:02d}.json" for t in range(NT)]
    files += [GRIDS / f"stage{t + 1:02d}_{c}.pt" for t, c in KEYS] + [CALDIR / f"{c}.json" for c in ORDER]
    files += sorted({dist_path(c, t, bnd[t]) for t, c in KEYS})
    return files


def write_lock():
    require(not LOCK.exists(), "LOCK.json exists (create-exclusive)")
    require(PREREG.exists() and TASK.exists(), "prereg / task missing")
    lk = {"schema": "sand_e2a_00_lock_v1", "utc": now(),
          "sources": {s: sha_file(ROOT / s) for s in SOURCES},
          "canonical_imports": {s: sha_file(ROOT / s) for s in CANONICAL},
          "prereg": {rel(PREREG): sha_file(PREREG)}, "task": {rel(TASK): sha_file(TASK)},
          "inputs": {rel(p): sha_file(p) for p in input_files()},
          "sand_published": SAND_PUBLISHED, "published_values": PUBLISHED, "gate": GATE, "tol": TOL, "caps": CAPS,
          "bytes": BYTES}
    h = write_json_x(LOCK, lk)
    print("LOCK", h, "inputs", len(lk["inputs"]))
    return h


def verify_lock():
    """Sources, canonical imports, prereg and task unchanged, except via create-exclusive AMENDMENT_NN.json files
    (protocol_change False). An amendment written after labels were opened may touch reporting code only."""
    lk = read_json(LOCK)
    src = {**lk["sources"], **lk["canonical_imports"], **lk["prereg"], **lk["task"]}
    for a in sorted(OUT.glob("AMENDMENT_*.json")):
        am = read_json(a)
        require(am["protocol_change"] is False, f"invalid amendment {a}")
        if am["labels_opened_before"] is not False:
            require(am.get("reporting_only") is True, f"post-label amendment must be reporting-only {a}")
        src.update(am["sources"])
    bad = [s for s, h in src.items() if sha_file(ROOT / s) != h]
    require(not bad, f"LOCK: changed after lock: {bad}")
    return lk


def verify_inputs(lk):
    bad = [p for p, h in lk["inputs"].items() if sha_file(ROOT / p) != h]
    require(not bad, f"inputs changed after lock: {bad[:5]}")
    return len(lk["inputs"])


# ------------------------------------------------------------------ witness calibrators (prereg section 2)

def channel_stats(a, d):
    """Population mean/std (FP64, ddof 0) of the clipped decoder residual and of the support distance over the
    given images' patches; the same operations as exps.sand.common.calibrate."""
    x = np.maximum(np.asarray(a, dtype=np.float64).ravel(), 0.0)
    y = np.asarray(d, dtype=np.float64).ravel()
    require(x.size == y.size and x.size > 1, "witness stats: misaligned")
    return {"theta": (float(x.mean()), float(x.std())), "m": (float(y.mean()), float(y.std()))}


def e2a(cal_a, w_a, w_t, channels=CH):
    """Anchored witness recalibration (NL-DRIFT-00 E2A in calibrator space):
    mu_t = mu_a + (mu_w,t - mu_w,a),  sd_t = sd_a * sd_w,t / sd_w,a, per recalibrated channel; other channels frozen.
    At t = a it returns cal_a exactly."""
    out = {k: float(cal_a[k]) for k in CAL_KEYS}
    for ch in channels:
        (ma, sa), (mt, st) = w_a[ch], w_t[ch]
        require(sa > 0 and st > 0 and np.isfinite([ma, sa, mt, st]).all(), f"degenerate witness stats {ch}")
        out[f"mu_{ch}"] = out[f"mu_{ch}"] + (mt - ma)
        out[f"sd_{ch}"] = out[f"sd_{ch}"] * (st / sa)
    require(all(np.isfinite(out[k]) for k in CAL_KEYS) and out["sd_theta"] > 0 and out["sd_m"] > 0,
            f"degenerate E2A calibrators {out}")
    return out


def tracking_error(cal, ref):
    """Calibrator error against a reference calibrator, per channel: (mu - mu_ref) / sd_ref and log(sd / sd_ref)."""
    return {ch: ((cal[f"mu_{ch}"] - ref[f"mu_{ch}"]) / ref[f"sd_{ch}"], float(np.log(cal[f"sd_{ch}"] / ref[f"sd_{ch}"])))
            for ch in CH}


# ------------------------------------------------------------------ endpoints and gate (prereg sections 3-4)

def forgetting(series):
    """series[(t, c)] -> value for every KEYS cell. Eq. 8 as SAND/C2-DOM: per old category, max over its stages before
    the last minus the final value; returns (macro, per-category)."""
    fm = {c: max(series[(t, c)] for t in range(i, NT - 1)) - series[(NT - 1, c)] for i, c in enumerate(ORDER[:-1])}
    return float(np.mean(list(fm.values()))), fm


def gate(e2a_ap, singles_ap, e2a_auroc, singles_auroc, boot):
    """e2a_ap: per-category final Pixel AP of the primary arm (ORDER order); singles_ap: {arm: per-category list};
    e2a_auroc: macro final Image AUROC; singles_auroc: {arm: macro}; boot(a, b) -> {'point', 'lo', 'hi', ...}."""
    require(set(singles_ap) == set(SINGLES) == set(singles_auroc), "gate: single detectors")
    macro = {a: float(np.mean(v)) for a, v in singles_ap.items()}
    best = max(SINGLES, key=lambda a: macro[a])
    b = boot(list(e2a_ap), list(singles_ap[best]))
    gain = float(np.mean(e2a_ap)) - macro[best]
    au_best = max(singles_auroc.values())
    g1 = gain >= GATE["min_final_pixel_ap_gain"]
    g2 = b["lo"] > GATE["boot_lower_bound_gt"]
    g3 = e2a_auroc >= au_best - GATE["max_final_image_auroc_loss"]
    ok = bool(g1 and g2 and g3)
    return {"best_single_pixel_ap_arm": best, "best_single_pixel_ap": macro[best], "single_macro_pixel_ap": macro,
            "final_pixel_ap_gain": gain, "bootstrap": b, "best_single_image_auroc": au_best,
            "final_image_auroc": e2a_auroc, "G1_gain": bool(g1), "G2_lower_bound": bool(g2), "G3_auroc": bool(g3),
            "thresholds": GATE, "pass": ok, "verdict": "E2A_FUSION_SUPPORTED" if ok else "E2A_FUSION_NOT_SUPPORTED"}


if __name__ == "__main__":
    if sys.argv[1:] == ["lock"]:
        write_lock()
    else:
        raise SystemExit("usage: python -m exps.sand_e2a_00.common lock")
