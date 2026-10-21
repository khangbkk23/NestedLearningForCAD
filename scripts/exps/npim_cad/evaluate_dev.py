#!/usr/bin/env python
# scripts/exps/npim_cad/evaluate_dev.py
"""Thin runner shim for the ``exps.npim_cad.evaluate_dev`` authorization gate."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exps.npim_cad.evaluate_dev import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
