"""SCRC-P1-LITE-00 supports (prereg §4). Label-free; reads only hash-checked canonical C2-DOM checkpoints.

S_14  (R0): every stored row of the canonical DOM_PQ_B2 t=14 checkpoint, task-agnostic union, ids increasing.
S'_14 (C0, M01-M12): per owner k, the first q'_k rows of that owner *in checkpoint storage order* (the order of the
      canonical t=14 FPS selection), q' = per_group_quota(capacity(B - 98,464, 14), 15).
Rows are decoded exactly as ``exps.nl_sgm_01a_c2dom.stream.supports_at`` (``codecs.decode`` with codebook k), so S_14
is bitwise the union used by P0 ``canonical_bank(14)`` (test U3).
"""
from __future__ import annotations

import json

import torch

from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a import exact as X
from exps.scrc_p1_lite_00 import protocol as P


def boundary_record(t):
    return json.loads(P.BOUNDARY_REC(t).read_text())


def load_ckpt(t):
    p = P.CKPT(t)
    rec = boundary_record(t)
    P.require(P.sha_file(p) == rec["ckpt_sha256"], f"checkpoint t={t} hash differs from its boundary record")
    lk = P.read_lock()
    key = str(p.relative_to(P.ROOT))
    if key in lk["inputs_sha256"]:
        P.require(lk["inputs_sha256"][key] == rec["ckpt_sha256"], f"checkpoint t={t} hash differs from the lock")
    return torch.load(p, map_location="cpu", weights_only=False), rec


def owner_rows(ck, k, limit=None):
    """(decoded FP32 rows, gids int64) of owner k in checkpoint storage order (first ``limit`` rows if given)."""
    n = int(ck["header"][1:].sum())
    own = ck["owner"][:n].long()
    pos = torch.nonzero(own == k, as_tuple=False).flatten()
    if limit is not None:
        P.require(len(pos) >= limit, f"owner {k} has {len(pos)} rows < limit {limit}")
        pos = pos[:limit]
    x = K.decode(ck["codes"][:n][pos], ck["codebooks"][k])
    g = ck["ids"][:n][pos].long()
    return x, g


def _union(parts):
    xs, gs = [], []
    for x, g in parts:
        o = torch.argsort(g)
        xs.append(x[o]); gs.append(g[o])
    x, ids = torch.cat(xs).contiguous(), torch.cat(gs)
    P.require(bool((ids[1:] > ids[:-1]).all()), "union ids not strictly increasing")
    return X.RowSet(x, ids)


def full_support(ck=None):
    ck = ck if ck is not None else load_ckpt(P.BOUNDARY)[0]
    parts = [owner_rows(ck, k) for k in range(P.BOUNDARY + 1)]
    P.require(tuple(len(g) for _, g in parts) == P.QUOTA_FULL, "full support per-category rows != quota")
    return _union(parts)


def byte_matched_support(ck=None):
    ck = ck if ck is not None else load_ckpt(P.BOUNDARY)[0]
    parts = [owner_rows(ck, k, P.QUOTA_BM[k]) for k in range(P.BOUNDARY + 1)]
    return _union(parts)


def per_category_counts(rs, n_ids_bounds=None):
    """Rows per owner, from the boundary record's per-category gid ranges (gid = gid_base + j*784 + p)."""
    from exps.nl_sgm_01a import am02_run as R2
    from exps.nl_sgm_01a_c2dom import common as D
    _, n = D.setup()
    bases = torch.tensor([R2.gid_base(n, c) for c in P.ORDER[1:]])
    own = torch.bucketize(rs.ids, bases, right=True)
    return tuple(int((own == k).sum()) for k in range(len(P.ORDER)))


def checks(S, Sp):
    """PV2: subset, counts, sortedness/uniqueness, ledger (prereg §4)."""
    out = {}
    out["full_rows"], out["bm_rows"] = int(len(S.ids)), int(len(Sp.ids))
    out["ids_sorted_unique"] = bool((S.ids[1:] > S.ids[:-1]).all() and (Sp.ids[1:] > Sp.ids[:-1]).all())
    pos = torch.searchsorted(S.ids, Sp.ids)
    out["subset_ids"] = bool((pos < len(S.ids)).all() and torch.equal(S.ids[pos.clamp_max(len(S.ids) - 1)], Sp.ids))
    out["subset_rows_bitwise"] = bool(out["subset_ids"] and torch.equal(S.x[pos], Sp.x))
    out["full_counts"] = per_category_counts(S)
    out["bm_counts"] = per_category_counts(Sp)
    out["counts_ok"] = out["full_counts"] == P.QUOTA_FULL and out["bm_counts"] == P.QUOTA_BM
    out["ledger"] = {"R0": P.l0_bytes(out["full_rows"]), "C0": P.l0_bytes(out["bm_rows"]),
                     "M": P.l0_bytes(out["bm_rows"]) + P.SCRC_OVERHEAD, "budget": P.BUDGET}
    out["ledger_ok"] = (out["ledger"]["R0"] == P.BYTES_R0 and out["ledger"]["C0"] == P.BYTES_C0
                        and out["ledger"]["M"] == P.BYTES_M and out["ledger"]["M"] <= P.BUDGET)
    out["support_sha"] = {"S14_ids": P.sha_tensor(S.ids), "S14_x": P.sha_tensor(S.x),
                          "S14p_ids": P.sha_tensor(Sp.ids), "S14p_x": P.sha_tensor(Sp.x)}
    out["pass"] = bool(out["ids_sorted_unique"] and out["subset_ids"] and out["subset_rows_bitwise"]
                       and out["counts_ok"] and out["ledger_ok"]
                       and out["full_rows"] == P.ROWS_FULL and out["bm_rows"] == P.ROWS_BM)
    return out


def load_state():
    from exps.nl_cad_pilot_00 import scrc as S
    lk = P.read_lock()
    key = str(P.STATE_PATH.relative_to(P.ROOT))
    P.require(P.sha_file(P.STATE_PATH) == lk["inputs_sha256"][key], "state_cumulative_t15 hash differs from lock")
    o = torch.load(P.STATE_PATH, map_location="cpu", weights_only=True)
    st = S.FastState(o["U"], o["lam"], int(o["t"][0]))
    P.require(st.t == 15 and st.U.shape == (P.DIM, P.RANK), "unexpected state")
    return st


def metrics(st):
    """{arm: Metric} for M01-M12 plus the gamma = 0 identity metric used by PV3."""
    from exps.nl_cad_pilot_00 import scrc as S
    out = {k: S.make_metric(st, tt, g) for k, (g, tt) in P.CONFIGS.items()}
    out[P.PV3_ARM] = S.make_metric(st, 0.0, 0.0)
    P.require(out[P.PV3_ARM].is_identity, "gamma = 0 metric is not the identity")
    return out
