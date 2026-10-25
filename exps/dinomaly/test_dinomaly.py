# exps/dinomaly/test_dinomaly.py
"""CPU tests for DRF-01 (no development image, no mask, no checkpoint)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from exps.dinomaly import encoder as E
from exps.dinomaly import official as O
from exps.dinomaly import readout as R
from exps.dinomaly import run


def test_linear_attention_matches_explicit_formula():
    torch.manual_seed(0)
    att = O.LinearAttention2(16, num_heads=2, qkv_bias=True)
    x = torch.randn(2, 7, 16)
    y, _ = att(x)
    q, k, v = att.qkv(x).reshape(2, 7, 3, 2, 8).permute(2, 0, 3, 1, 4)
    phi = lambda t: torch.nn.functional.elu(t) + 1
    w = phi(q) @ phi(k).transpose(-1, -2)                         # [B,H,N,N], not normalised by softmax
    ref = ((w @ v) / w.sum(-1, keepdim=True)).transpose(1, 2).reshape(2, 7, 16)
    assert torch.allclose(y, att.proj(ref), atol=1e-5)


def test_hard_mining_scales_easy_point_gradients():
    torch.manual_seed(1)
    a = [torch.randn(2, 4, 3, 3)]
    b0 = torch.randn(2, 4, 3, 3)
    grads = {}
    for factor in (1.0, 0.1):
        b = b0.clone().requires_grad_(True)
        bb = [b * 1.0]
        O.global_cosine_hm_percent(a, bb, p=0.5, factor=factor).backward()
        grads[factor] = b.grad
    dist = 1 - torch.nn.functional.cosine_similarity(a[0], b0).unsqueeze(1)
    thresh = torch.topk(dist.reshape(-1), k=int(dist.numel() * 0.5))[0][-1]
    easy = (dist < thresh).expand_as(b0)
    assert torch.allclose(grads[0.1][easy], 0.1 * grads[1.0][easy]) and torch.equal(grads[0.1][~easy], grads[1.0][~easy])


def test_two_group_fusion_equals_eight_layer_mean():
    layers = torch.randn(8, 3, 5, 6, dtype=torch.float64)
    g1, g2 = layers[:4].mean(0), layers[4:].mean(0)
    assert torch.allclose((g1 + g2) / 2, layers.mean(0))


def test_checkpointed_decoder_is_exact():
    torch.manual_seed(2)
    d0 = O.Decoder(32, 4, num_prefix=5)
    O.init_trainable(d0)
    d1 = O.Decoder(32, 4, num_prefix=5, checkpoint=True)
    d1.load_state_dict(d0.state_dict())
    g = torch.randn(2, 2, 5 + 16, 32)
    out = []
    for d in (d0, d1):
        d.train()
        d.bottleneck[0].drop.p = 0.0                         # isolate checkpointing from dropout sampling
        en, de = d(g[:, 0], g[:, 1])
        loss = O.global_cosine(en, de)
        loss.backward()
        out.append((de[0].detach(), [p.grad.clone() for p in d.parameters()]))
    assert torch.equal(out[0][0], out[1][0])
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(out[0][1], out[1][1]))


def test_decoder_shapes_and_prefix_removal():
    d = O.Decoder(32, 4).eval()
    en, de = d(torch.randn(3, 5 + 784, 32), torch.randn(3, 5 + 784, 32))
    assert [t.shape for t in en + de] == [torch.Size([3, 32, 28, 28])] * 4


def test_batches_drop_last_epochs():
    gen = torch.Generator().manual_seed(0)
    b = run.Batches(20, 8, gen)
    epoch = b.next() + b.next()
    assert len(set(epoch)) == 16 and b.queue == []
    assert len(set(b.next() + b.next())) == 16


def test_replay_buffer_balanced_and_nested():
    proto = {"train": {c: [{"relative_path": f"{c}/train/good/{i:03d}.png", "bytes": 10} for i in range(n)]
                       for c, n in zip(("bottle", "carpet", "hazelnut"), (150, 100, 100))}}
    train = {c: torch.arange(len(proto["train"][c]), dtype=torch.float32).view(-1, 1, 1, 1) for c in proto["train"]}
    f1, i1 = run.replay_buffer(proto, 2711, ["bottle"], train)
    f2, i2 = run.replay_buffer(proto, 2711, ["bottle", "carpet"], train)
    assert len(i1["identities"]) == len(i2["identities"]) == 24
    assert i2["identities"][:12] == i1["identities"][:12]
    assert sum(p.startswith("carpet") for p in i2["identities"]) == 12 and i2["png_bytes"] == 240


def test_scheduler_matches_official_formula():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = O.StableAdamW([p], lr=2e-3)
    s = O.WarmCosineScheduler(opt, 2e-3, 2e-4, 300, warmup_iters=100)
    lrs = []
    for _ in range(300):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step(); s.step()
    assert lrs[0] == 0 and abs(lrs[99] - 2e-3) < 1e-12 and abs(lrs[100] - 2e-3) < 1e-12 and lrs[-1] > 2e-4


def test_dinomaly_readout_constant_and_topk():
    en = [torch.ones(2, 8, 28, 28)] * 2
    de = [torch.ones(2, 8, 28, 28)] * 2
    s, img = R.dinomaly(en, de)
    assert s.shape == (2, 256, 256) and torch.allclose(img, torch.zeros(2), atol=1e-6)
    de2 = [torch.ones(2, 8, 28, 28)] * 2
    de2[0] = de2[0].clone(); de2[0][:, :, 14, 14] = -1               # one anomalous patch, cos = -1
    s2, img2 = R.dinomaly(en, de2)
    flat = s2.flatten(1)
    k = int(flat.shape[1] * 0.01)
    assert torch.allclose(img2, flat.sort(1, descending=True)[0][:, :k].mean(1))
    peak = torch.nonzero(s2[0] == s2[0].max())[0]
    assert abs(int(peak[0]) - 135) <= 6 and abs(int(peak[1]) - 135) <= 6     # patch (14,14) centre at 392 -> 256
    assert R.native(s2[0], (900, 900)).shape == (900, 900)


def test_encoder_token_layout_random_weights():
    torch.manual_seed(3)
    m = E.DinoV2Reg("small").eval()
    g = m.groups(torch.randn(1, 3, 392, 392))
    assert g.shape == (1, 2, 789, 384)
    assert m.interpolate_pos_encoding(392, 392).shape == (1, 785, 384)
