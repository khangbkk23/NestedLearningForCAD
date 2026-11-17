"""NL-DRIFT-00 (task agents/tasks/task_nl_drift_00.md): constants, identity pools, witness draws, custody, lock.

Preregistration: agents/plans/nl_drift_00_prereg.md (hashed into LOCK.json). Label-free. Reads only train/good
features, the decoder run's checkpoints and stage records, the R15 protocol (train/heldout keys), the ViT-B/8 cache
record and the C2-DOM final checkpoint (PQ codebooks).
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/hope_cad/nl_drift_00"
PREREG = ROOT / "agents/plans/nl_drift_00_prereg.md"
TASK = ROOT / "agents/tasks/task_nl_drift_00.md"
RUN = ROOT / "results/hope_cad/vitb8_nl00/runs/q2off_seed2711"
FEAT = ROOT / "results/hope_cad/calm02/features/mvtec"
CACHE_JSON = ROOT / "results/hope_cad/vitb8_q2/CACHE.json"
RH01B = ROOT / "results/hope_cad/rh01b/PROTOCOL.json"
C2DOM_CKPT = ROOT / "results/hope_cad/nl_sgm_01a_c2_dom_sweep/streams/DOM_PQ_B2/ckpt_t14.pt"
HOME_CACHE = Path.home() / "hope_cad_cache"

ORDER = ("bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw",
         "tile", "toothbrush", "transistor", "wood", "zipper")
A = {c: i for i, c in enumerate(ORDER)}
U_CATS = ("bottle", "carpet", "hazelnut")
TOKENS, DIM = 784, 768

NW = (1, 2, 4)
NW_PRIMARY = 1
DRAWS = 8
N_SUB = 256
G_E3 = 0.5
RIDGE = 1.0
PRIOR_LOGNOISE = 1.0
LOG_BOUNDS = (-14.0, 6.0)
MIN_PREFIX = 10
BOOT_B, BOOT_SEED = 10_000, 20261116
THRESH = {"drift_min_W": 0.5, "track_ratio": 0.5, "learned_rel": 0.9, "pq_rel": 0.10, "pq_p95": 0.25,
          "pv3_abs": 2e-6}
CAPS = {"gpu_phase_wall_h": 0.5, "analysis_wall_h": 0.5, "rss_gb": 8.5, "vram_gb": 4.5, "disk_gb": 0.5}
BYTES = {"fp32_grid": TOKENS * DIM * 4, "fp16_grid": TOKENS * DIM * 2, "pq_grid": TOKENS * 96,
         "pq_codebook": 96 * 256 * 8 * 2, "centroid_fp32": DIM * 4}
POOL_SIZES = {"T": {"bottle": 126, "cable": 212, "capsule": 211, "carpet": 94, "grid": 259, "hazelnut": 96,
                    "leather": 241, "metal_nut": 217, "pill": 264, "screw": 318, "tile": 227, "toothbrush": 58,
                    "transistor": 212, "wood": 246, "zipper": 240},
              "U": {"bottle": 59, "carpet": 180, "hazelnut": 291}}

SOURCES = ("exps/nl_drift_00/__init__.py", "exps/nl_drift_00/common.py", "exps/nl_drift_00/stats.py",
           "exps/nl_drift_00/estimators.py", "exps/nl_drift_00/analyze.py", "exps/nl_drift_00/test_drift.py")


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


def sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json_x(path, value):
    """Create-exclusive JSON write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=1, sort_keys=True, allow_nan=True, default=_default)
    return sha_file(path)


def _default(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


# ------------------------------------------------------------------ custody (prereg section 6)

_ALLOWED_FILES = {RH01B.resolve(), CACHE_JSON.resolve(), C2DOM_CKPT.resolve()}
_LOG = []
_INSTALLED = []


def decide(path):
    """True if the path may be opened. Everything under results/, data/ and ~/hope_cad_cache is denied unless on the
    allow-list; paths elsewhere (sources, libraries, /proc, /tmp) are not governed."""
    try:
        p = Path(os.fsdecode(path)).resolve()
    except Exception:
        return True
    s = str(p)
    governed = (s.startswith(str(ROOT / "results") + "/") or s.startswith(str(ROOT / "data") + "/")
                or s.startswith(str(HOME_CACHE) + "/") or s == str(ROOT / "data") or s == str(HOME_CACHE))
    if not governed:
        return True
    if p in _ALLOWED_FILES:
        return True
    if s.startswith(str(OUT) + "/") or s == str(OUT):
        return True
    if p.parent == FEAT.resolve() and p.name.endswith("_train.pt"):
        return True
    if p.parent == RUN.resolve() and ((p.name.startswith("STAGE") and p.suffix == ".json")
                                      or (p.name.startswith("stage") and p.suffix == ".pt")):
        return True
    return False


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        if not decide(args[0]):
            _LOG.append({"denied": os.fsdecode(args[0])})
            raise PermissionError(f"NL-DRIFT-00 custody: refused {os.fsdecode(args[0])}")
        p = os.fsdecode(args[0])
        if p.startswith(str(ROOT)) or p.startswith(str(HOME_CACHE)):
            _LOG.append({"path": p})


def install_custody():
    if not _INSTALLED:
        sys.addaudithook(_audit)
        _INSTALLED.append(True)


def access_log():
    return list(_LOG)


# ------------------------------------------------------------------ identities (prereg section 1)

def r15_and_heldout():
    proto = read_json(RH01B)
    train = {c: [r["relative_path"] for r in proto["train"][c]] for c in ORDER}
    held = {c: [r["relative_path"] for r in proto["heldout"].get(c, [])] for c in ORDER}
    require(sum(len(v) for v in train.values()) == 3099, "R15 identity count")
    for c in ORDER:
        require(all(p.startswith(f"{c}/train/good/") for p in train[c] + held[c]), f"non-train identity in {c}")
    return train, held


def stage_record(s):
    return read_json(RUN / f"STAGE{s:02d}.json")


def replay_lists():
    """{stage s (1-based): [relative paths]} for s = 1..15 (stage 1 has none)."""
    out = {}
    for s in range(1, 16):
        r = stage_record(s)
        out[s] = list(r["replay"]["identities"]) if r["replay"] else []
    return out


def pools(cache_rows):
    """cache_rows: {c: [relative paths of CALM-02 {c}_train.pt in file order]} -> (T pools, U pools, replayed(c))."""
    train, held = r15_and_heldout()
    rep = replay_lists()
    ever = {p for v in rep.values() for p in v}
    T, U, R = {}, {}, {}
    for c in ORDER:
        r15 = set(train[c])
        require(r15 <= set(cache_rows[c]), f"R15 identity missing from CALM-02 cache {c}")
        T[c] = [p for p in cache_rows[c] if p in r15 and p not in ever]
        R[c] = [p for p in cache_rows[c] if p in r15 and p in ever]
        if c in U_CATS:
            U[c] = [p for p in cache_rows[c] if p not in r15]
            require(not set(U[c]) & ever, f"unseen pool intersects replay {c}")
            require(set(held[c]) <= set(U[c]), f"held-out normals not in unseen pool {c}")
    for X, P in (("T", T), ("U", U)):
        for c, v in P.items():
            require(len(v) == POOL_SIZES[X][c], f"pool size {X}/{c}: {len(v)} != {POOL_SIZES[X][c]}")
    return T, U, R, train, held, rep


def witness_order(cohort, c, d, pool):
    return sorted(pool, key=lambda p: sha_text(f"NL-DRIFT-00/witness/v1|{cohort}|{c}|{d}|{p}"))


def witness_sets(cohort, c, pool):
    """[(W(4) list in witness order, reference list)] for d = 0..DRAWS-1; W(n) = first n of W(4)."""
    out = []
    for d in range(DRAWS):
        o = witness_order(cohort, c, d, pool)
        w = o[:max(NW)]
        ref = [p for p in pool if p not in set(w)]
        require(len(ref) == len(pool) - len(w) and not set(ref) & set(w), "witness/reference overlap")
        out.append((w, ref))
    return out


def subset_seed(cohort, c, d, n):
    return int(sha_text(f"NL-DRIFT-00/subsets/v1|{cohort}|{c}|{d}|{n}")[:16], 16)


# ------------------------------------------------------------------ lock

LOCK = OUT / "LOCK.json"


def write_lock():
    require(not LOCK.exists(), "LOCK exists (create-exclusive)")
    return write_json_x(LOCK, {"task": str(TASK.relative_to(ROOT)), "task_sha256": sha_file(TASK),
                               "prereg": str(PREREG.relative_to(ROOT)), "prereg_sha256": sha_file(PREREG),
                               "sources": {s: sha_file(ROOT / s) for s in SOURCES},
                               "constants": {"NW": NW, "DRAWS": DRAWS, "N_SUB": N_SUB, "G_E3": G_E3, "RIDGE": RIDGE,
                                             "LOG_BOUNDS": LOG_BOUNDS, "MIN_PREFIX": MIN_PREFIX,
                                             "PRIOR_LOGNOISE": PRIOR_LOGNOISE,
                                             "BOOT": [BOOT_B, BOOT_SEED], "THRESH": THRESH, "CAPS": CAPS},
                               "utc": now()})


def verify_lock():
    lk = read_json(LOCK)
    require(sha_file(PREREG) == lk["prereg_sha256"], "LOCK: prereg changed")
    require(sha_file(TASK) == lk["task_sha256"], "LOCK: task changed")
    src = dict(lk["sources"])
    for am in sorted(OUT.glob("AMENDMENT_*.json")):
        a = read_json(am)
        require(a.get("protocol_change") is False, f"invalid amendment {am}")
        src.update(a.get("sources", {}))
    bad = [s for s, h in src.items() if sha_file(ROOT / s) != h]
    require(not bad, f"LOCK: sources changed: {bad}")
    return lk


if __name__ == "__main__":
    if sys.argv[1:] == ["lock"]:
        print("LOCK sha256", write_lock())
    else:
        raise SystemExit("usage: python -m exps.nl_drift_00.common lock")
