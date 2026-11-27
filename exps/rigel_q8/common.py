"""Shared paths and helpers of RIGEL-Q8-01."""
from __future__ import annotations

import time
from pathlib import Path

from exps.rigel_gg01.dev import sha, wj  # noqa: F401  (same JSON / hash helpers as GG-01)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/hope_cad/rigel_q8'
CACHE = Path('/mnt/k/NLCAD/cache/rigel_q8')
PREREG = ROOT / 'exps/rigel_q8/PREREG.md'
PREREG_SHA = '018ebd6d52dcf5c52fab8a75fdd5c5ff9794e4774bc5e4c4f51bde963f65c6f9'   # ledger 2026-11-26T16:28:35


def prereg_sha() -> str:
    s = sha(PREREG)
    if s != PREREG_SHA:
        raise SystemExit(f'STOP: PREREG.md changed after registration ({s})')
    return s


def log(*a):
    s = time.strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'q8.log', 'a') as f:
        f.write(s + '\n')
