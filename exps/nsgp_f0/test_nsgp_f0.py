"""NSGP-F0 focused tests (normal-only: train/good features and stored banks only; no test data).

    python -m pytest exps/nsgp_f0/test_nsgp_f0.py -q

Passing tests demonstrate mechanics, not scientific efficacy.
"""
from __future__ import annotations

import io
import math

import numpy as np
import pytest
import torch

from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a import exact as X
from exps.nl_sgm_01a_c2dom import common as D2
from exps.nsgp_f0 import common as S


@pytest.fixture(scope="module")
def ctx():
    co, n = S.setup()
    return co, n, D2.partition(co, n), C.Features()


def _small_bank(ctx, t=14, cats=("bottle", "screw"), m=400):
    co, n, _, F = ctx
    b, _ = S.bank(t)
    rows = {}
    for c in cats:
        v = b[c]
        sel = torch.arange(0, len(v["ids"]), max(1, len(v["ids"]) // m))[:m]
        ids = v["ids"][sel]
        rows[c] = {"ids": ids, "x": S.o0_rows(F, co, n, c, ids), "codes": v["codes"][sel], "codebook": v["codebook"]}
    return rows


# 1 identity mapping ------------------------------------------------------------------------------------------

def test_bank_ids_map_to_the_same_fp32_vectors_and_categories(ctx):
    co, n, part, F = ctx
    b, _ = S.bank(14)
    for c in ("bottle", "grid", "zipper"):
        v = b[c]
        assert bool((v["ids"][1:] > v["ids"][:-1]).all())                    # no duplicates, canonical order
        j, p = S.address(n, c, v["ids"])
        x = S.o0_rows(F, co, n, c, v["ids"])
        for r in (0, len(x) // 2, len(x) - 1):
            ref = F.image(co[c]["writers"][int(j[r])])[int(p[r])]
            assert torch.equal(x[r], ref)
        assert set(torch.unique(j).tolist()) <= {jj for jj, _ in part[c]["sel"]}
        assert torch.equal(K.encode(x[:300], v["codebook"]), v["codes"][:300])       # codes are encode(O0 row)
        F.drop(c)


# 2 canonical B0 reproduces a stored official normal-only value ---------------------------------------------

def test_b0_reproduces_normal_diag_median(ctx):
    co, _, _, F = ctx
    b, _ = S.bank(14)
    v = b["toothbrush"]
    q = F.images(co["toothbrush"]["sel"]).reshape(-1, 768).clone()
    d, _ = S.score(q, S.b0_rows(v), v["ids"])
    ref = D2.read_json(D2.OUT / "NORMAL_DIAG.json")["categories"]["toothbrush"][S.MEM_ARM]["median_d"]
    assert abs(float(d.median()) - ref) <= 1e-12


# 3 exact NN equals brute-force FP64 -------------------------------------------------------------------------

def test_exact_nn_equals_bruteforce_fp64(ctx):
    rows = _small_bank(ctx)
    co, _, _, F = ctx
    q = F.images(co["bottle"]["eval"][:1]).reshape(-1, 768)[:300].clone()
    x, ids = rows["bottle"]["x"], rows["bottle"]["ids"]
    d, i = S.score(q, x, ids)
    D = torch.cdist(q.double(), x.double())
    ref = D.min(1)
    assert torch.allclose(d, ref.values, rtol=0, atol=1e-9)
    picked = (q.double() - x[torch.searchsorted(ids, i)].double()).norm(dim=1)
    assert torch.allclose(picked, d, rtol=0, atol=1e-12)


# 4 identity reconstruction gives exactly equal scores -------------------------------------------------------

def test_identity_coded_bank_scores_equal_o0(ctx):
    rows = _small_bank(ctx)
    co, _, _, F = ctx
    q = F.images(co["screw"]["eval"][:1]).reshape(-1, 768).clone()
    x, ids = rows["screw"]["x"], rows["screw"]["ids"]
    buf = io.BytesIO(); torch.save({"x": x.clone()}, buf); buf.seek(0)
    x_id = torch.load(buf)["x"]                                         # identity codec: serialized FP32 rows
    assert torch.equal(S.score(q, x, ids)[0], S.score(q, x_id, ids)[0])
    mu, U, _ = S.pca(x, 768)                                            # full-rank PCA, no quantization: ~exact
    xr = (mu + ((x.double() - mu) @ U) @ U.T).float()
    assert float((S.score(q, xr, ids)[0] - S.score(q, x, ids)[0]).abs().max()) < 1e-4


# 5 / 6 point-set stability bound and one-sided failures -----------------------------------------------------

def _sets(seed=0, m=300, d=16):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(m, d, generator=g)
    Q = torch.randn(500, d, generator=g) * 1.3
    return A, Q


def _dd(A, B):
    """Direct-difference FP64 distances (cdist's GEMM form leaves ~1e-7 residue on identical rows)."""
    return (A.double()[:, None] - B.double()[None]).square().sum(-1).sqrt()


def _nn(Q, A):
    return _dd(Q, A).min(1).values


def test_score_error_bounded_by_hausdorff_and_paired_error():
    A, Q = _sets()
    B = A + 0.05 * torch.randn(A.shape, generator=torch.Generator().manual_seed(1))
    e = (_nn(Q, B) - _nn(Q, A)).abs().max()
    Dab = _dd(A, B)
    h = max(Dab.min(1).values.max(), Dab.min(0).values.max())
    paired = (A - B).double().norm(dim=1).max()
    assert e <= h + 1e-12 and h <= paired + 1e-12


def test_deletion_inflates_and_out_of_support_points_suppress():
    A, Q = _sets(seed=2)
    d0 = _nn(Q, A)
    dele = A[: len(A) // 2]                                  # every reconstructed row is exact (paired error 0 on kept)
    d1 = _nn(Q, dele)
    assert bool((d1 >= d0 - 1e-12).all()) and float((d1 - d0).max()) > 0.1
    D = _dd(dele, A)
    assert float(D.min(1).values.max()) == 0.0               # one-sided (arm -> O0) distance is 0, yet scores change
    fake = torch.cat([A, Q[:20] + 1e-3])                     # spurious rows near queries: suppression
    d2 = _nn(Q, fake)
    assert bool((d2 <= d0 + 1e-12).all()) and float((d0 - d2).max()) > 0.1
    D2_ = _dd(A, fake)
    assert float(D2_.min(1).values.max()) == 0.0             # the other one-sided distance is 0 here
    h = max(float(_dd(fake, A).min(1).values.max()), float(_dd(A, fake).min(1).values.max()))
    assert float((d0 - d2).abs().max()) <= h + 1e-12         # the two-sided Hausdorff distance still bounds it


def test_hausdorff_helper_matches_cdist():
    A, _ = _sets(seed=3, d=768, m=200)
    B = A[:150] + 0.01
    h, hab, hba = S.hausdorff(A.float().contiguous(), B.float().contiguous())
    D = torch.cdist(A.double(), B.double())
    assert abs(hab - float(D.min(1).values.max())) < 1e-9 and abs(hba - float(D.min(0).values.max())) < 1e-9
    assert h == max(hab, hba)


# 7 leakage guards -------------------------------------------------------------------------------------------

def test_leakage_guards(ctx):
    co, n, part, F = ctx
    for t in (0, 4, 9):
        b, _ = S.bank(t)
        assert list(b) == list(S.ORDER[:t + 1])                                       # no future category
    for c in S.ORDER:
        fq = set(S.fitq(co, part, c))
        assert len(fq) == S.FITQ_N and fq <= {p for _, p in part[c]["fit"]}
        assert not fq & (set(co[c]["eval"]) | set(co[c]["sel"]))
        assert not (set(co[c]["eval"]) | set(co[c]["sel"])) & set(co[c]["writers"])
    for c in ("bottle", "carpet", "hazelnut"):
        h = set(S.heldout_paths(c))
        assert not h & (set(co[c]["writers"]) | set(co[c]["eval"]) | set(co[c]["sel"]))
    with pytest.raises(Exception):
        F.image("bottle/test/good/000.png")                                          # custody guard


# 8 byte ledger and save/reload ------------------------------------------------------------------------------

def test_byte_ledgers_and_reload(ctx):
    rows = _small_bank(ctx)
    co, _, _, F = ctx
    q = F.images(co["bottle"]["eval"][:1]).reshape(-1, 768)[:200].clone()
    R = sum(len(v["ids"]) for v in rows.values())
    st = S.build_c0(14, rows, 8)
    assert S.state_bytes(st) == S.bytes_c0(R, 2, 8)
    buf = io.BytesIO(); torch.save(st, buf); buf.seek(0)
    st2 = torch.load(buf, weights_only=False)
    a, b = S.decode_c0(st, list(rows)), S.decode_c0(st2, list(rows))
    assert torch.equal(a["bottle"], b["bottle"])
    ids = rows["bottle"]["ids"]
    assert torch.equal(S.score(q, a["bottle"], ids)[0], S.score(q, b["bottle"], ids)[0])
    assert S.bytes_pq(93680, 15) == S.B0_FINAL_BYTES
    assert S.bytes_n0(93680, 15, S.K_N0["C"]) <= 0.75 * S.B0_FINAL_BYTES < S.bytes_n0(93680, 15, S.K_N0["C"] + 1)
    assert S.bytes_n0(93680, 15, S.K_N0["Q"]) <= S.B0_FINAL_BYTES < S.bytes_n0(93680, 15, S.K_N0["Q"] + 1)
    assert S.bytes_c0(93680, 15, S.K_C0["C"]) <= 0.75 * S.B0_FINAL_BYTES < S.bytes_c0(93680, 15, S.K_C0["C"] + 1)
    assert S.bytes_c0(93680, 15, S.K_C0["Q"]) <= S.B0_FINAL_BYTES < S.bytes_c0(93680, 15, S.K_C0["Q"] + 1)


# 9 determinism, finiteness, per-category retrieval ----------------------------------------------------------

def test_c0_determinism_and_retrieval(ctx):
    rows = _small_bank(ctx)
    a, b = S.build_c0(14, rows, 6), S.build_c0(14, rows, 6)
    assert all(torch.equal(a[k], b[k]) for k in a)
    dec = S.decode_c0(a, list(rows))
    for c in rows:
        assert dec[c].shape == rows[c]["x"].shape and bool(torch.isfinite(dec[c]).all())
        err_own = (dec[c] - rows[c]["x"]).norm(dim=1).mean()
        other = [o for o in rows if o != c][0]
        assert err_own < (dec[other][: len(dec[c])] - rows[c]["x"][: len(dec[other])]).norm(dim=1).mean()
    assert torch.equal(a["ids"].long(), torch.cat([rows[c]["ids"] for c in rows]))


def test_quant8_roundtrip_bound():
    u = torch.randn(1000, 5, dtype=torch.float64) * torch.tensor([10., 3., 1., .1, .01], dtype=torch.float64)
    q, s = S.quant8(u)
    assert q.dtype == torch.int8 and int(q.abs().max()) <= 127
    assert bool(((q.double() * s.double() - u).abs() <= s.double() / 2 + 1e-9).all())


# 10 learned decoder mechanics ---------------------------------------------------------------------------------

def _curve_rows(m=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(m, 1, generator=g) * 2 * math.pi
    P = torch.randn(6, 768, generator=g)
    feats = torch.cat([torch.sin(a), torch.cos(a), torch.sin(2 * a), torch.cos(3 * a), a, a * a / 10], 1)
    return {"bottle": {"ids": torch.arange(m), "x": (feats @ P).float().contiguous()}}


def test_n0_starts_at_pca_and_overfits_a_tiny_nonlinear_set():
    rows = _curve_rows()
    st, diag = S.train_n0(0, rows, 2, device="cpu", steps=600)
    mu, U, _ = S.pca(rows["bottle"]["x"], 2)
    X0 = rows["bottle"]["x"].double()
    pca_mse = float(((X0 - mu) - ((X0 - mu) @ U) @ U.T).square().sum()) / X0.numel()
    assert abs(diag["init_mse_pca_unquantized"] - pca_mse) / pca_mse < 1e-5          # W2 = 0: exactly PCA at step 0
    assert diag["trained_mse_continuous"] < 0.5 * diag["init_mse_pca_unquantized"]
    assert diag["persistent_mse_fp16_int8"] < 0.6 * diag["init_mse_pca_unquantized"]
    assert all(math.isfinite(r["mse"]) for r in diag["curve"]) and diag["collapse_min_component_sd"] > 1e-3
    assert S.state_bytes(st) == S.bytes_n0(256, 1, 2)


def test_n0_deterministic_on_cpu():
    rows = _curve_rows(m=64, seed=1)
    a, da = S.train_n0(0, rows, 2, device="cpu", steps=50)
    b, db = S.train_n0(0, rows, 2, device="cpu", steps=50)
    assert all(torch.equal(a[k], b[k]) for k in a) and da["persistent_mse_fp16_int8"] == db["persistent_mse_fp16_int8"]


# gate logic ---------------------------------------------------------------------------------------------------

def test_gate_routes_and_guard():
    eb = np.array([1.0, 2.0, 1.5, 0.01, 1.2])
    ok = S.gate(15_000_000, S.B0_FINAL_BYTES, eb * 0.8, eb)
    assert ok["pass"] and ok["route"] == "quality" and ok["guard"]
    nb = S.gate(15_000_000, S.B0_FINAL_BYTES, eb * 0.95, eb)
    assert not nb["pass"]                                                              # 0.95 > 0.90, too big for 0.75
    comp = S.gate(11_000_000, S.B0_FINAL_BYTES, eb * 1.0, eb)
    assert comp["pass"] and comp["route"] == "compression"
    bad = eb * 0.7
    bad[1] = eb[1] * 1.2                                                               # one material category +20 %
    assert not S.gate(15_000_000, S.B0_FINAL_BYTES, bad, eb)["guard"]
    neg = eb * 0.7
    neg[3] = eb[3] * 5                                                                 # negligible B0 error: exempt
    assert S.gate(15_000_000, S.B0_FINAL_BYTES, neg, eb)["guard"]
    assert not S.gate(15_400_000, 15_500_000, eb * 0.5, eb)["pass"]                   # absolute budget
