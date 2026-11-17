"""NL-SGM-01A-C2-DOM memory-budget sweep: protocol constants, fit/selection partition, capacity ledger.

Normal-only (train/good, R15 identities via exps.nl_sgm_01a.common). Test features and masks are read only by
exps.nl_sgm_01a_c2dom.evaluate, after the LOCK and the B8 decision exist.
"""
from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch

from exps.csg01.select import random_priorities
from exps.nl_sgm_01a import am02_run as R2
from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a import common as C
from exps.task_p.p0_normal_support import require

ROOT = C.ROOT
OUT = ROOT / "results/hope_cad/nl_sgm_01a_c2_dom_sweep"
LOCK = OUT / "LOCK.json"
PROTOCOL = OUT / "PROTOCOL.json"
ORDER = C.ORDER
SEED = C.SEED                                   # 20261106 (C0/F2 k-means++ seed and hash-priority seed)
FIT_ROWS, FIT_ITERS = 100_000, 25
SPLIT_KEY = "NL-SGM-01A-C2DOM/split/v1|"

BUDGETS = {"B1": 7_680_000, "B2": 15_360_000, "B4": 30_720_000, "B8": 61_440_000}
REGISTERED = ("B1", "B2", "B4")                 # B8 only via the locked normal-only rule (B8_DECISION.json)
HEADER_BYTES = 64                                # int32[16]: boundary index + per-category row counts
ID_BYTES, OWNER_BYTES = 4, 1                     # int32 support id + int8 owner, charged per stored row
FP32_ROW_BYTES = 768 * 4 + ID_BYTES + OWNER_BYTES          # 3,077
PQ_ROW_BYTES = K.PQ_ROW_BYTES + ID_BYTES + OWNER_BYTES      # 101
CB_BYTES = K.CODEBOOK_BYTES                                  # 393,216

CAPS = {"stream_wall_h": 10.0, "eval_wall_h": 8.0, "rss_gb": 8.5, "disk_gb": 12.0, "vram_gb": 3.0,
        "total_wall_h": 36.0, "threads": 6, "max_concurrent_processes": 3}


def arms(budgets=REGISTERED):
    out = {}
    for b in budgets:
        out[f"DOM_PQ_{b}"] = {"codec": "pq_dom", "budget": b, "bytes": BUDGETS[b]}
        out[f"SEL_FP32_{b}"] = {"codec": "fp32", "budget": b, "bytes": BUDGETS[b]}
    return out


def capacity(codec, budget_bytes, t):
    """Max stored rows at boundary t (0-based) after charging header, ids, owners and t+1 codebooks (PQ)."""
    if codec == "fp32":
        return int((budget_bytes - HEADER_BYTES) // FP32_ROW_BYTES)
    return int((budget_bytes - HEADER_BYTES - (t + 1) * CB_BYTES) // PQ_ROW_BYTES)


def ledger_bytes(codec, budget_bytes, t):
    cap = capacity(codec, budget_bytes, t)
    if codec == "fp32":
        return cap * FP32_ROW_BYTES + HEADER_BYTES
    return cap * PQ_ROW_BYTES + (t + 1) * CB_BYTES + HEADER_BYTES


def state_bytes(state):
    return int(sum(v.numel() * v.element_size() for v in state.values()))


# ------------------------------------------------------------------ cohorts and partition

def setup(threads=6):
    C.set_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    co = C.cohorts()
    require(C.r15_digest()[:8] == C.R15_DIGEST, "R15 cohort digest mismatch")
    return co, {c: len(co[c]["writers"]) for c in ORDER}


def split_key(path):
    return C.text_sha(SPLIT_KEY + path)


def partition(co, n):
    """{c: {"fit": [(j, path)], "sel": [(j, path)]}}: image-level split of the writer cohort, j = writer index.
    fit = the floor(n/2) writer images with the smallest split key; sel = the rest. Both lists in writer order."""
    out = {}
    for c in ORDER:
        w = co[c]["writers"]
        ranked = sorted(range(len(w)), key=lambda j: split_key(w[j]))
        fit = set(ranked[: len(w) // 2])
        out[c] = {"fit": [(j, w[j]) for j in range(len(w)) if j in fit],
                  "sel": [(j, w[j]) for j in range(len(w)) if j not in fit]}
        require(len(out[c]["fit"]) + len(out[c]["sel"]) == n[c] and not fit & {j for j, _ in out[c]["sel"]},
                "partition overlap")
    return out


def pool_gids(n, c, items):
    base = R2.gid_base(n, c)
    return torch.cat([torch.arange(base + j * C.PATCHES, base + (j + 1) * C.PATCHES) for j, _ in items])


def pool_rows(F, items):
    return F.images([p for _, p in items]).reshape(-1, 768).clone()


def fit_order(gids):
    """Hash-first order of rows (random_priorities(SEED, gid), ties by gid); first FIT_ROWS form the fit sample."""
    p = random_priorities(SEED, gids.numpy().astype(np.int64))
    return torch.as_tensor(np.lexsort((gids.numpy(), p)))


def gid_sha(t):
    return C.text_sha(",".join(str(int(v)) for v in torch.sort(torch.as_tensor(t)).values.tolist()))


# ------------------------------------------------------------------ resources / io

def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def disk_gb():
    return sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e9


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path, value):
    return C.write_json(path, value)


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_lock():
    """LOCK source hashes, overridden only by create-exclusive engineering amendments (AMENDMENT_NN.json, applied in
    order; each records the files it changes, their new hashes, the reason and that no test data had been opened)."""
    lk = read_json(LOCK)
    src = dict(lk["sources"])
    for a in sorted(OUT.glob("AMENDMENT_*.json")):
        am = read_json(a)
        require(am["protocol_change"] is False, f"invalid amendment {a}")
        if am["test_data_opened_before"] is not False:          # post-evaluation: reporting/robustness code only
            require(am.get("post_eval_reporting_only") is True and set(am["sources"]) <= {
                "exps/nl_sgm_01a_c2dom/analyze.py", "exps/nl_sgm_01a_c2dom/common.py"}, f"invalid post-eval amendment {a}")
        src.update(am["sources"])
    bad = [s for s, h in src.items() if C.sha(ROOT / s) != h]
    require(not bad, f"LOCK: source changed after lock: {bad}")
    require(C.sha(PROTOCOL) == lk["protocol_sha256"], "LOCK: protocol changed after lock")
    return lk


class Log:
    def __init__(self, name):
        (OUT / "logs").mkdir(parents=True, exist_ok=True)
        self.f = (OUT / "logs" / f"{name}.log").open("a")

    def __call__(self, *a):
        s = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def cap_check(t0, wall_h_cap):
    wall_h = (time.time() - t0) / 3600
    rss = peak_rss_gb()
    require(wall_h <= wall_h_cap, f"CAP: wall {wall_h:.2f} h > {wall_h_cap} h")
    require(rss <= CAPS["rss_gb"], f"CAP: peak RSS {rss:.2f} GB > {CAPS['rss_gb']} GB")
    d = disk_gb()
    require(d <= CAPS["disk_gb"], f"CAP: disk {d:.2f} GB > {CAPS['disk_gb']} GB")
    return {"wall_h": wall_h, "peak_rss_gb": rss, "disk_gb": d}


def env_threads():
    return {"torch_threads": torch.get_num_threads(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}
