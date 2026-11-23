"""NANM-EVAL-01 shared constants and read-only bindings (task: agents/tasks/task_nanm_eval_01.md).

The canonical oracle (exps/bench_g3/run.py and the primitives it imports) is imported unchanged. `bind()` rebinds only
module-level PATH constants to the NANM-REPRO-01 locations, exactly as exps/nanm_repro_01/run.py does: K: token
caches, per-task normal states and stored score grids are READ-ONLY inputs. This task writes only
results/hope_cad/nanm_eval_01/ (E:, small JSON/logs) and $EVAL01_CACHE (K:, new score grids / per-task states).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/hope_cad/nanm_eval_01'
REPRO = ROOT / 'results/hope_cad/nanm_repro_01'
REPRO_CACHE = Path('/mnt/k/NLCAD/cache/nanm_repro_01')
CACHE = Path(os.environ.get('EVAL01_CACHE', '/mnt/k/NLCAD/cache/nanm_eval_01'))
TASKFILE = ROOT / 'agents/tasks/task_nanm_eval_01.md'
REGFILE = ROOT / 'agents/tasks/task_nanm_eval_01_registration.md'
VHDX = Path('/mnt/d/wsl/Ubuntu_new/ext4.vhdx')
GiB, MiB = 2 ** 30, 2 ** 20
D = 768
KEY_B = 2 * D                                   # fp16 key (canonical BENCH-G3 KEY)
S_B = D * D * 4 + 8                             # canonical G3 S (fp32) + n (fp64)            = 2,359,304
G1_B = D * D * 4 + D * 4 + 8                    # feature covariance + mean + n (BENCH-G3)    = 2,362,376
ENCODER_B = 144_806_400                         # shared frozen encoder (BENCH-G3), reported separately
AGENTS_PIN = 'b88393815586295209c2704a6e16262730f2aba4000a6904b7c5c6d2b65ed622'
DATASETS = ('mvtec', 'visa', 'mpdd', 'realiad')
EXEC_DATASETS = ('mpdd', 'visa')                # datasets with verified existing G3 token caches
SEEDS = (2711, 2712, 2713)
RATIOS = (0.0025, 0.005, 0.01, 0.02)
CAPS_MIB = (16, 32, 64, 128)


def sha(p, buf=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(buf):
            h.update(b)
    return h.hexdigest()


def ssha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def rj(p):
    return json.loads(Path(p).read_text())


def wj(p, obj, exclusive=False):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'x' if exclusive else 'w') as f:
        f.write(json.dumps(obj, indent=1, default=str))


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def storage():
    return {'D_free': shutil.disk_usage('/mnt/d').free, 'E_free': shutil.disk_usage('/mnt/e').free,
            'K_free': shutil.disk_usage('/mnt/k').free, 'VHDX': VHDX.stat().st_size, 'time': time.strftime('%FT%T')}


def bind():
    """Rebind oracle path constants to the REPRO-01 K: caches (read-only use) exactly as exps/nanm_repro_01 does."""
    from exps.nanm_repro_01 import run as RP
    RP.bind()
    from exps.bench_g3 import run as G
    return G


# ------------------------------------------------------------------ task orders (registered)
def orders(base):
    """a = canonical (alphabetical, BENCH-G3), b = reversed, c = salted-SHA permutation (fixed before any metric)."""
    base = tuple(base)
    return {'a': base, 'b': tuple(reversed(base)),
            'c': tuple(sorted(base, key=lambda t: ssha(f'NANM-EVAL-01/order-c/v1|{t}')))}


# ------------------------------------------------------------------ sizing (pure; unit-tested)
def n_of(r, n):
    return math.ceil(r * n)


def ratio_sizes(n_fit, r):
    return [n_of(r, n) for n in n_fit]


def cap_ratio(n_fit, cap_bytes, extra):
    """Largest common ratio r with sum ceil(r n_t) * KEY_B + extra <= cap (BENCH-G3 c_sizes rule, cap as a parameter).
    Returns (r, sizes) or (None, None) if even r -> 0+ does not fit or the cap exceeds all FIT tokens (INFEASIBLE)."""
    fits = lambda r: sum(n_of(r, n) for n in n_fit) * KEY_B + extra <= cap_bytes
    if not fits(1e-12):
        return None, None
    if fits(1.0):
        return None, None                         # cap larger than storing every FIT token: not a memory cut
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo, [n_of(lo, n) for n in n_fit]


def a0b_like_sizes(n_fit, r):
    """Euclidean k-centre sizes buying G3's S bytes on top of ratio r (BENCH-G3 a0b_sizes rule generalised to r):
    common ratio, then trim the overshoot one key at a time from the largest tasks (ties: earlier task)."""
    base = [n_of(r, n) for n in n_fit]
    need = sum(base) + math.ceil(S_B / KEY_B)
    lo, hi = r, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if sum(n_of(mid, n) for n in n_fit) >= need else (mid, hi)
    sizes = [n_of(hi, n) for n in n_fit]
    over = sum(sizes) - need
    assert 0 <= over <= len(sizes), over
    for i in sorted(range(len(sizes)), key=lambda i: (-sizes[i], i))[:over]:
        sizes[i] -= 1
    assert sum(sizes) == need and all(s >= b for s, b in zip(sizes, base))
    return hi, sizes


def bytes_of(arm_kind, n_keys, n_tasks=1):
    """Persistent CL state (BENCH-G3 definition: fp16 keys + method statistic; derived W cache and encoder excluded).
    kinds: 'g3' (keys + S), 'euclid' (keys), 'g1' (keys + feature covariance), 'pertask' (keys + one S per task)."""
    extra = {'g3': S_B, 'euclid': 0, 'g1': G1_B, 'pertask': n_tasks * S_B}[arm_kind]
    return n_keys * KEY_B + extra
