#!/usr/bin/env python
# scripts/exps/npim_cad/preflight.py
"""Thin runner shim for ``exps.npim_cad.preflight``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exps.npim_cad.preflight import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
