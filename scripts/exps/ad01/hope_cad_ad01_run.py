# scripts/exps/ad01/hope_cad_ad01_run.py
"""Runner shim for the AD-01 four-arm endpoint evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exps.ad01.hope_cad_ad01_run import main

if __name__ == "__main__":
    main()
