# exps/rh01b/common.py
"""RH-01b: identical data, buffer and feature cache to RH-01 (reused from exps.rh01.common); own output directory."""
from __future__ import annotations

from exps.rh01.common import (CACHE, CAPACITY, DRF, EVAL, N_TRAIN, ROOT, STREAM, Pool, buffer_index,  # noqa: F401
                              identities, load_features)

OUT = ROOT / "results/hope_cad/rh01b"
RH01 = ROOT / "results/hope_cad/rh01"            # feature-cache manifest (CACHE.json) and prerun ledger reference
