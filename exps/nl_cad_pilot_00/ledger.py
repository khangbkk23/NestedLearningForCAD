"""End-to-end persistent-state byte ledger for SCRC and its controls (P0, T9).

Level-0 accounting is the canonical C2-DOM ledger (exps.nl_sgm_01a_c2dom.common: 64 B header, per row 96 B PQ code +
4 B int32 id + 1 B int8 owner, one FP16 PQ codebook of 393,216 B per seen category), imported unchanged. SCRC arms
charge their Level-1/Level-2 state against the same total budget, so their bank capacity is computed with the budget
reduced by that overhead. The backbone is counted separately. Task-agnostic inference has no routing state.
"""
from __future__ import annotations

from exps.nl_sgm_01a_c2dom import common as D
from exps.nl_cad_pilot_00 import scrc as S

B2 = D.BUDGETS["B2"]                                   # 15,360,000 B (C2-DOM registered budget)
FAST_BYTES = S.D_FEAT * S.RANK * 4 + S.RANK * 4        # U FP32 + lam FP32 = 98,432
SLOW_BYTES = 6 * 4 + 8                                 # theta FP32[6] + int64 boundary counter = 32
SCRC_OVERHEAD = FAST_BYTES + SLOW_BYTES                # 98,464
A10_BYTES = S.D_FEAT * S.RANK * 4 + S.RANK * 4         # static learned metric: Q FP32 [d, 32] + beta FP32[32]
BACKBONE_FILE_BYTES = 343_282_995                      # checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth


def bank_capacity(budget, t, overhead=0):
    """C2-DOM DOM-PQ rows at 0-based boundary t with ``overhead`` bytes of non-bank persistent state charged."""
    return int((budget - overhead - D.HEADER_BYTES - (t + 1) * D.CB_BYTES) // D.PQ_ROW_BYTES)


def bank_bytes(budget, t, overhead=0):
    cap = bank_capacity(budget, t, overhead)
    return cap * D.PQ_ROW_BYTES + (t + 1) * D.CB_BYTES + D.HEADER_BYTES


ARM_OVERHEAD = {
    "A0": 0,                          # Euclidean: the overhead bytes are given back to the bank as rows
    "A1": SCRC_OVERHEAD, "A2": SCRC_OVERHEAD, "A3": SCRC_OVERHEAD, "A4": SCRC_OVERHEAD, "A5": SCRC_OVERHEAD,
    "A6": SCRC_OVERHEAD,              # + h history if the causal permutation is adopted (see a6_history_bytes)
    "A7": SCRC_OVERHEAD, "A8": SCRC_OVERHEAD, "A9": SCRC_OVERHEAD,
    "A10": A10_BYTES + 8,             # Q, beta + int64 counter (no theta)
}


def a6_history_bytes(T):
    return 4 * 4 * T


def ledger(T, budget=B2):
    """Per-arm final persistent bytes after T tasks (t = T-1)."""
    t = T - 1
    out = {}
    for arm, ov in ARM_OVERHEAD.items():
        cap = bank_capacity(budget, t, ov)
        bb = bank_bytes(budget, t, ov)
        out[arm] = {"bank_rows": cap, "bank_codes_ids_owner_bytes": cap * D.PQ_ROW_BYTES,
                    "codebook_bytes": (t + 1) * D.CB_BYTES, "header_bytes": D.HEADER_BYTES,
                    "fast_state_bytes": FAST_BYTES if arm not in ("A0", "A10") else 0,
                    "slow_state_bytes": SLOW_BYTES if arm not in ("A0", "A10") else 0,
                    "static_metric_bytes": A10_BYTES + 8 if arm == "A10" else 0,
                    "routing_bytes": 0, "total": bb + ov, "budget": budget, "slack": budget - bb - ov}
    return out


def transient(N_rows, m=S.M_RES, d=S.D_FEAT, r=S.RANK, q_patches=784):
    """Working memory that is never persisted (bytes)."""
    return {"decoded_bank_fp32": N_rows * d * 4, "transformed_bank_fp32": N_rows * d * 4,
            "residual_R_fp32": m * d * 4, "svd_B_fp64": d * (r + m) * 8, "gram_dxd_fp64": d * d * 8,
            "query_patches_fp32": q_patches * d * 4}
