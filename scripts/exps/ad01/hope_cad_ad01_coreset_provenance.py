# scripts/exps/hope_cad_ad01_coreset_provenance.py
"""Runner for the AD-01 CADIC coreset provenance diagnostic."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exps.ad01.hope_cad_ad01_coreset_provenance import main

if __name__ == "__main__":
    main()
