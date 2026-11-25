"""NANM-G3 scientific contract: frozen constants of the sealed BENCH-G3 system (LOCK ff998675…). No tunable fields.

Claim boundary (D7): NANM = Nested Associative Normality Memory. "Nested" denotes only that the residual geometry is
computed from the exemplar bank written at the same task boundary (PROJECT_MAPPING). No NL-specific benefit, no
per-image update and no CMS equivalence is claimed. Both levels change once per task boundary.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class G3Config:
    dim: int = 768
    grid: int = 28
    image_size: int = 392
    patch: int = 14
    encoder_blocks: tuple = (6, 7, 8, 9)
    prefix_tokens: int = 5
    encoder_batch: int = 8
    ratio: float = 0.01                 # k_t = ceil(ratio * |FIT_t tokens|)
    proj_dim: int = 128
    seed: int = 2711                    # projection + first-centre generator, re-seeded at every write
    shrink_alpha: float = 0.5
    eig_floor: float = 1e-12
    fit_chunk_images: int = 64
    residual_chunk: int = 2048
    query_chunk: int = 512
    sigma: int = 4
    ksize: int = 33                     # 2 * int(4 * sigma + .5) + 1
    top_fraction: float = 0.01

    def contract_sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
