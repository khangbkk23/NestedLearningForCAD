"""NL-SGM-01A-AM02 — minimal historical-support retention experiment: arm definitions (no outcome code here).

Four conceptual controls at the B1 persistent budget (7,680,000 B), all category-balanced
(AD-02 per_group_quota over the categories seen so far, remainder to earlier tasks):

  C1  FPS_FP32      boundary FPS (canonical csg01.select_fps via rules.FPSRule), FP32 rows + 1 B owner -> 2,499 rows
  C2  FPS_PQ        the same FPSRule run on PQ-decoded rows, fixed FP16 codebook C0 fit at the end of task 0
                    (rules.CodecArm kind 'pq_fixed'), 96 B code + 1 B owner -> 75,121 rows
  C3  HASH_PQ       same quotas and codec as C2, within-category bottom-q by fixed hash priority (QuotaHashRule)
  C4  SAMEID_PQ     C1's row identities at every boundary, stored as C0 codes (offline; 2,499 rows, < B1 bytes)

C1-C3 are causal stream arms. C4 is a same-identity diagnostic: it reads FP32 rows only at their write time
(C0 is frozen, so encoding at write time == encoding later) and never selects anything.
"""
from __future__ import annotations

import numpy as np
import torch

from exps.csg01.select import random_priorities
from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a import rules as R
from exps.task_p.p0_normal_support import require

VERSION = "NL-SGM-01A-AM02/v1"
FP32_ROW_BYTES = 768 * 4
OWNER_BYTES = 1
C1_CAPACITY = C.B1 // (FP32_ROW_BYTES + OWNER_BYTES)                 # 2,499
PQ_CAPACITY = K.pq_capacity(C.B1, 1, OWNER_BYTES)                   # 75,121


def arm_bytes():
    return {
        "C1_FPS_FP32": C1_CAPACITY * (FP32_ROW_BYTES + OWNER_BYTES),
        "C2_FPS_PQ": PQ_CAPACITY * (K.PQ_ROW_BYTES + OWNER_BYTES) + K.CODEBOOK_BYTES,
        "C3_HASH_PQ": PQ_CAPACITY * (K.PQ_ROW_BYTES + OWNER_BYTES) + K.CODEBOOK_BYTES,
        "C4_SAMEID_PQ": C1_CAPACITY * (K.PQ_ROW_BYTES + OWNER_BYTES) + K.CODEBOOK_BYTES,
    }


class QuotaHashRule:
    """Category-balanced hash-random allocation: at each boundary every category keeps its bottom-q rows by
    csg01 random_priorities(seed, gid) (ties by gid). Bottom-q of a kept bottom-q' set is the global bottom-q,
    so re-quota never needs evicted rows. Same interface as rules.FPSRule (stores what it is given)."""

    def __init__(self, capacity, seed=C.SEED):
        self.capacity, self.seed = int(capacity), int(seed)
        self.kept = {}
        self.buf_x, self.buf_g = [], []
        self.transient_peak = 0

    def begin_task(self, t):
        self.buf_x, self.buf_g = [], []

    def image(self, x, gids, t):
        self.buf_x.append(x.clone()); self.buf_g.append(gids.clone())
        self.transient_peak = max(self.transient_peak, sum(b.numel() * 4 for b in self.buf_x))

    def _bottom(self, x, g, q):
        p = random_priorities(self.seed, g.numpy().astype(np.int64))
        order = torch.as_tensor(np.lexsort((g.numpy(), p))[:q])
        return x[order], g[order]

    def _requota(self):
        cats = sorted(self.kept)
        require(len(cats) <= self.capacity, "capacity below category count")
        for k, q in zip(cats, R.AP.per_group_quota(self.capacity, len(cats))):
            x, g = self.kept[k]
            if len(x) > q:
                self.kept[k] = self._bottom(x, g, q)

    def end_task(self, t):
        x, g = torch.cat(self.buf_x), torch.cat(self.buf_g)
        self.buf_x, self.buf_g = [], []
        self.kept[t] = (x, g)
        self._requota()

    rows = R.FPSRule.rows
    redecode = R.FPSRule.redecode

    def shrink(self, capacity):
        self.capacity = int(capacity)
        self._requota()


def build(arm):
    if arm == "C1_FPS_FP32":
        return R.CodecArm(R.FPSRule(C1_CAPACITY), "fp32", C.B1)
    if arm == "C2_FPS_PQ":
        return R.CodecArm(R.FPSRule(PQ_CAPACITY), "pq_fixed", C.B1)
    if arm == "C3_HASH_PQ":
        return R.CodecArm(QuotaHashRule(PQ_CAPACITY), "pq_fixed", C.B1)
    raise ValueError(arm)


def same_identity(c1_gids, fp32_rows_at_write, codebook):
    """C4 at one boundary: C1's identities, rows = decode(encode(x_write, C0)). Diagnostic only."""
    require(len(c1_gids) == len(fp32_rows_at_write), "row/id mismatch")
    return K.decode(K.encode(fp32_rows_at_write, codebook), codebook), c1_gids
