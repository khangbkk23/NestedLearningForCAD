"""NAR-PATCH-01 tests (synthetic CPU; no dataset access)."""
import torch
import torch.nn.functional as F

from exps.dinomaly import official as O
from exps.nar_patch_01 import run as P


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        en = x
        de = torch.einsum('bchw,cd->bdhw', x, self.w).contiguous()
        return [en], [de]


def test_patch_loss_is_mean_patch_cosine_and_differs_from_global():
    torch.manual_seed(0)
    m, x = Toy(), torch.randn(2, 4, 3, 3)
    x[:, :, 0, 0] *= 20                                   # one high-norm patch dominates the global loss
    ref = (1 - F.cosine_similarity(x, torch.einsum('bchw,cd->bdhw', x, m.w), dim=1)).mean()
    assert torch.allclose(P.patch_loss(m, x), ref)
    en, de = m(x)
    assert not torch.allclose(P.patch_loss(m, x), O.global_cosine(en, de))


def test_patch_loss_stops_encoder_gradient():
    m = Toy()
    x = torch.randn(1, 4, 2, 2, requires_grad=True)
    P.patch_loss(m, x).backward()
    g_en = x.grad.clone()
    x.grad = None
    en, de = m(x)
    (1 - F.cosine_similarity(en[0].detach(), de[0], dim=1)).mean().backward()
    assert torch.equal(g_en, x.grad)                      # only the decoder path carries gradient


def test_gate_semantics_reused():
    s = lambda ap=.5, im=.9, acq=.5, fm=.01: {'final_pixel_ap': ap, 'final_image_auroc': im,
                                               'acquisition_pixel_ap': acq, 'pixel_fm_eq8': {'macro': fm}}
    assert P.W.gate(s(), s(), {'point': .01, 'lo': 1e-9})['passed']
    assert not P.W.gate(s(), s(), {'point': .009, 'lo': .001})['passed']
