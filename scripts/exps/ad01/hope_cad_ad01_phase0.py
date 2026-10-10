# scripts/exps/hope_cad_ad01_phase0.py
"""Runner for AD-01 Phase 0 CADIC reference diagnostics (read-only artifacts)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exps.ad01.hope_cad_ad01_phase0 import main

if __name__ == "__main__":
    main()
