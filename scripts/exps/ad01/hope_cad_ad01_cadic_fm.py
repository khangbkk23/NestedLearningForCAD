# scripts/exps/ad01/hope_cad_ad01_cadic_fm.py
"""Runner shim for the bounded CADIC forgetting slice."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exps.ad01.hope_cad_ad01_cadic_fm import main

if __name__ == "__main__":
    main()
