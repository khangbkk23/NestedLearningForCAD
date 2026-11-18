"""NAR-LITE-01 tests (synthetic CPU; no dataset or checkpoint access)."""
import numpy as np
import torch
import torch.nn.functional as F

from exps.nar_lite_01 import run as L


def test_separable_smoothing_equals_2d_kernel():
    torch.manual_seed(0)
    grid = torch.rand(2, 28, 28)
    ax = torch.arange(L.KSIZE, dtype=torch.float32) - L.KSIZE // 2
    g = torch.exp(-ax ** 2 / (2 * L.SIGMA ** 2)); g = g / g.sum()
    up = F.interpolate(grid[:, None], size=(392, 392), mode='bilinear', align_corners=False)
    ref = F.conv2d(up, (g[:, None] * g[None, :])[None, None], padding=L.KSIZE // 2)[:, 0]
    out = L.smooth392(grid)
    assert out.shape == (2, 392, 392) and L.KSIZE == 33
    assert torch.allclose(out, ref.clamp_min(0), atol=1e-6)


def test_smoothed_readout_uses_exact_native_ap(monkeypatch):
    grid = torch.zeros(2, 28, 28); grid[1, 10:14, 10:14] = 1.
    hws = [(64, 64), (64, 64)]
    masks = [np.zeros((64, 64), bool), np.zeros((64, 64), bool)]
    masks[1][23:32, 23:32] = True
    ap, auc = L.cell_metrics(grid, 'smooth', hws, masks, [0, 1])
    assert auc == 1.0 and 0.5 < ap <= 1.0
    assert L.N.native_map is not L.map_native           # restored after use
    ap_raw, _ = L.cell_metrics(grid, 'raw', hws, masks, [0, 1])
    assert 0 < ap_raw <= 1.0


def test_lite_gate_memory_and_tolerance():
    ref = {'final_pixel_ap': .5665, 'final_image_auroc': .96, 'acquisition_pixel_ap': .56}
    ok = dict(ref, final_pixel_ap=.562)
    assert L.lite_gate(ok, ref, int(.5 * L.REF_BYTES))['passed']
    assert not L.lite_gate(ok, ref, int(.7 * L.REF_BYTES))['passed']
    assert not L.lite_gate(dict(ref, final_pixel_ap=.56), ref, int(.5 * L.REF_BYTES))['passed']


def test_bind():
    L.bind()
    assert L.W.OUT == L.OUT and L.W.CAPS == L.CAPS
