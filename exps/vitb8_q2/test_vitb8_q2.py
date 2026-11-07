"""Focused CPU tests for VITB8-Q2: shapes, gradient flow, replay capacity, Q2 mixing, map geometry, leakage, resume.

  PYTHONPATH=. python -m pytest exps/vitb8_q2/test_vitb8_q2.py -q
"""
from __future__ import annotations

import math

import torch

from exps.dinomaly import official as O
from exps.nl_align_01 import g1 as G
from exps.nlcad import bank as BK
from exps.rh01 import common as RC
from exps.vitb8_q2 import common as V
from exps.vitb8_q2 import train as T

N_TRAIN = {'bottle': 150, 'cable': 224, 'capsule': 219, 'carpet': 100, 'grid': 264, 'hazelnut': 100, 'leather': 245,
           'metal_nut': 220, 'pill': 267, 'screw': 320, 'tile': 230, 'toothbrush': 60, 'transistor': 213, 'wood': 247,
           'zipper': 240}


def _dec():
    return V.new_decoder(V.SEED, device="cpu", ckpt=False)


def test_shapes_and_prefix():
    dec = _dec().eval()
    f = torch.randn(2, 784, 768)
    en, de = dec(f)
    assert len(en) == len(de) == 1 and en[0].shape == de[0].shape == (2, 768, 28, 28)
    assert dec.num_prefix == 0
    assert torch.equal(en[0], f.permute(0, 2, 1).reshape(2, 768, 28, 28))      # target = untouched input map
    try:
        dec(torch.randn(2, 789, 768))
        raise AssertionError("prefix-token input must be rejected")
    except RuntimeError:
        pass
    names = [n for n, _ in dec.named_parameters()]
    assert all(n.startswith("bottleneck.0.") or n.startswith("decoder.") for n in names)
    assert len(dec.decoder) == 8


def test_gradient_flow_and_no_identity_shortcut():
    dec = _dec().train()
    f = torch.randn(2, 784, 768, requires_grad=True)
    en, de = dec(f)
    cos = torch.nn.functional.cosine_similarity(en[0], de[0]).mean()
    assert float(cos) < 0.9                                               # reconstruction is not a copy at init
    loss = O.global_cosine_hm_percent([e.float() for e in en], [d.float() for d in de], p=0.9, factor=0.1)
    loss.backward()
    for n, p in dec.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    assert sum(float(p.grad.abs().sum()) for p in dec.parameters()) > 0
    assert f.grad is not None and float(f.grad.abs().sum()) > 0              # flows through the decoder input path
    # the target branch is stop-gradient: d loss / d f equals the gradient through the decoder alone
    f2 = f.detach().clone().requires_grad_(True)
    torch.manual_seed(0); dec.eval()
    en2, de2 = dec(f2)
    l2 = O.global_cosine([en2[0].detach()], [de2[0]])
    l2.backward()
    assert float(f2.grad.abs().sum()) > 0


def test_replay24_capacity_and_lawfulness():
    for s in range(2, 16):
        done = list(V.STREAM[:s - 1])
        idx = RC.buffer_index(V.SEED, N_TRAIN, done)
        assert set(idx) == set(done) and V.STREAM[s - 1] not in idx
        assert sum(len(v) for v in idx.values()) == 24
        assert all(len(set(v)) == len(v) and all(0 <= i < N_TRAIN[c] for i in v) for c, v in idx.items())


def test_q2_host_access_has_no_future_or_nonbuffer_hosts():
    hosts = BK.host_sets(V.SEED, N_TRAIN)
    for s in range(1, 16):
        keys = BK.allowed(hosts, V.SEED, N_TRAIN, s)
        cats = {c for c, _ in keys}
        assert cats <= set(V.STREAM[:s])
        cur = V.STREAM[s - 1]
        done = list(V.STREAM[:s - 1])
        buf = RC.buffer_index(V.SEED, N_TRAIN, done) if done else {}
        for c, i in keys:
            assert (c == cur and i in hosts[c]["current"]) or (c in buf and i in buf[c])


def test_q2_mixing_rule():
    params = [torch.nn.Parameter(torch.zeros(5)), torch.nn.Parameter(torch.zeros(3))]
    g_n = [torch.randn(5), torch.randn(3)]
    g_a = [torch.randn(5) * 7, torch.randn(3) * 7]
    for p, a in zip(params, g_a):
        p.grad = a.clone()
    ref = [n.clone() for n in g_n]
    st = G.mix_(params, g_n, rho=T.RHO)
    nn_ = math.sqrt(sum(float(x.pow(2).sum()) for x in ref))
    na = math.sqrt(sum(float(x.pow(2).sum()) for x in g_a))
    for p, n, a in zip(params, ref, g_a):
        assert torch.allclose(p.grad, n + 0.25 * nn_ / (na + 1e-12) * a, atol=1e-5)
    assert abs(st["scale"] - 0.25 * nn_ / (na + 1e-12)) < 1e-6
    assert T.RHO == 0.25 and T.EPS == 1e-12 and T.K_SYN == 1


def test_q2_aux_gradient_on_decoder():
    dec = _dec().train()
    cov = torch.zeros(28, 28); cov[5:12, 5:12] = 1.0
    items = {("bottle", i, v): (torch.randn(784, 768), cov) for i in range(16) for v in range(2)}
    hosts = {"bottle": {"current": list(range(16)), "buffer": []}}
    role = G.roles(hosts)
    aux = T.Aux.__new__(T.Aux)
    aux.pools = {"current": [k for k in sorted(items) if role[(k[0], k[1])] == "OPT"], "replay": []}
    aux.syn = items
    aux.host_gen = torch.Generator().manual_seed(1); aux.neg_gen = torch.Generator().manual_seed(2)
    keys = aux.draw()
    assert len(keys) == T.N_CUR and all(role[(k[0], k[1])] == "OPT" for k in keys)
    loss = aux.backward(dec, keys, device="cpu")
    assert loss is not None and math.isfinite(loss)
    g = [p.grad for p in dec.parameters() if p.grad is not None]
    assert g and all(torch.isfinite(x).all() for x in g) and sum(float(x.abs().sum()) for x in g) > 0


def test_score_map_geometry():
    en = torch.ones(1, 768, 28, 28)
    de = en.clone()
    de[0, :, 20, 3] = -1.0                       # one anomalous patch: row 20, col 3
    s, img = V.score([en], [de])
    assert s.shape == (1, 256, 256) and img.shape == (1,)
    r, c = divmod(int(s[0].argmax()), 256)
    assert abs(r - (20 + 0.5) * 256 / 28) < 12 and abs(c - (3 + 0.5) * 256 / 28) < 12
    flat = s.flatten(1)
    assert torch.allclose(img, flat.sort(descending=True)[0][:, :655].mean(1))
    zero, zimg = V.score([en], [en.clone()])
    assert float(zero.abs().max()) < 1e-5 and float(zimg.abs().max()) < 1e-5
    from exps.dinomaly.readout import native
    m = native(s[0], (900, 700))
    rr, cc = divmod(int(m.argmax()), 700)
    assert abs(rr - 20.5 * 900 / 28) < 40 and abs(cc - 3.5 * 700 / 28) < 40


def test_checkpoint_resume_roundtrip(tmp_path):
    dec = _dec().eval()
    p = tmp_path / "stage01_bottle.pt"
    T.save_ckpt(dec, p, {"stage": 1})
    assert not p.with_suffix(".part").exists()
    dec2 = V.SingleStreamDecoder(checkpoint=False)
    dec2.load_state_dict(torch.load(p, weights_only=True)["state"])
    dec2.eval()
    f = torch.randn(1, 784, 768)
    assert torch.equal(dec(f)[1][0], dec2(f)[1][0])
    try:
        T.save_ckpt(dec, p, {"stage": 1})
        raise AssertionError("checkpoint overwrite must be refused")
    except RuntimeError:
        pass


def test_same_init_as_two_group_decoder():
    """Single-stream decoder has the official Decoder's parameters and identical seeded initialisation."""
    a = _dec()
    torch.manual_seed(V.SEED)
    b = O.Decoder(768, 12)
    O.init_trainable(b)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
