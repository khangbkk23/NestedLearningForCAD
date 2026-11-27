"""RIGEL-Q8-01 focused CPU tests.

    python -B -m pytest -q -p no:cacheprovider exps/rigel_q8/test_q8.py
"""
import torch

from exps.rigel_gg01 import gg
from exps.rigel_q8 import q8
from rigel_cad import ops


def _tokens(n, seed=0, d=768):
    g = torch.Generator().manual_seed(seed)
    return ops.unit(torch.randn(n, d, generator=g)).half()


def test_caps_and_payload():
    assert q8.ARMS['Q8T'][2] * q8.KEY8 + q8.S_TRI == 8388392 <= q8.BUDGET
    assert (q8.ARMS['Q8T'][2] + 1) * q8.KEY8 + q8.S_TRI > q8.BUDGET
    assert q8.ARMS['V1'][2] == gg.CAP_SLOW and q8.ARMS['F'][2] == gg.CAP_FAST


def test_quantizer_idempotent_and_accurate():
    x = _tokens(500)
    q, s = q8.quantize(x)
    xh = q8.dequantize(q, s)
    q2, s2 = q8.quantize(xh)
    assert torch.equal(q, q2) and torch.equal(s, s2)
    assert q.dtype == torch.int8 and s.dtype == torch.float16 and int(q.abs().max()) <= 127
    assert float(torch.nn.functional.cosine_similarity(x.float(), xh, dim=1).min()) > 0.999


def test_triangle_roundtrip():
    a = torch.randn(768, 50, dtype=torch.float64)
    S = a @ a.T
    t = q8.tri_pack(S)
    assert t.numel() == q8.N_TRI and t.dtype == torch.float32
    assert torch.equal(q8.tri_unpack(t), S.float().double())


def _fit(x16, seed=2711):
    return gg.fit_task(x16, seed)


def test_v1_arm_equals_gg01_v1_arm():
    """On a small synthetic stream (cap 60), the Q8 study's V1 arm keeps exactly the GG-01 V1 arm's keys."""
    a = q8.Arm('V1', 2711, 'cpu')
    b = gg.ArmMemory('V1', 2711, 'cpu')
    a.cap = b.cap = 60
    for l in range(4):
        x = _tokens(3000, seed=l)
        f = _fit(x)
        a.write(l, f.keys16, f.S)
        b.write(l, f.keys16, None, None)
        assert torch.equal(a.k16, b.keys) and torch.equal(a.task, b.task)


def test_int8_arm_stores_what_it_scores_and_respects_cap():
    a = q8.Arm('Q8T', 2711, 'cpu')
    a.cap = 70
    for l in range(4):
        x = _tokens(3000, seed=10 + l)
        f = _fit(x)
        a.write(l, f.keys16, f.S)
        assert len(a.task) <= 70
        assert torch.equal(a.keys, q8.dequantize(a.q, a.scale))
    S = sum(_fit(_tokens(3000, seed=10 + l)).S for l in range(4))
    assert float((q8.tri_unpack(a.S_tri) - S).abs().max()) <= 1e-5 * float(S.abs().max())   # fp32 storage


# ---------------------------------------------------------------------------------------------- D1 (PREREG_D1.md)
from exps.rigel_q8 import d1  # noqa: E402


def test_int4_pack_and_idempotent():
    x = _tokens(300, seed=5)
    q, s = d1.quantize4(x)
    assert int(q.abs().max()) <= 7
    assert torch.equal(d1.unpack4(d1.pack4(q)), q) and d1.pack4(q).shape[1] == 384
    q2, s2 = d1.quantize4(q8.dequantize(q, s))
    assert torch.equal(q, q2) and torch.equal(s, s2)


def test_d1_caps_payload_and_eviction_flow():
    assert d1.CAP_D1_FAST * q8.KEY8 + d1.CAP_D1_MED * d1.KEY4 + q8.S_TRI == 8388488 <= q8.BUDGET
    assert d1.CAP_Q4T * d1.KEY4 + q8.S_TRI <= q8.BUDGET
    a = d1.ArmD1('D1', 2711, 'cpu')
    a.cap_f, a.cap_m = 40, 60
    seen = 0
    for l in range(5):
        f = _fit(_tokens(3000, seed=20 + l))
        r = a.write(l, f.keys16, f.S)
        seen += len(f.keys16)
        assert r['n_fast'] <= 40 and r['n_medium'] <= 60
        assert r['n_fast'] + r['evicted'] == min(seen, 40) + r['evicted']
        assert torch.equal(a.keys[:r['n_fast']], q8.dequantize(a.fq, a.fs))
    assert a.log[-1]['n_medium'] == 60 and sum(a.log[-1]['retained_per_task_medium']) == 60
