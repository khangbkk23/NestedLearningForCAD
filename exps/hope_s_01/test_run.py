"""HOPE-S-01 tests (synthetic CPU; no dataset or checkpoint access)."""
import torch
import torch.nn.functional as F

from exps.hope_s_01 import run as H


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.w = torch.nn.Parameter(torch.eye(6) + .3 * torch.randn(6, 6))

    def forward(self, x):                                    # x [B,784,6]
        de = x @ self.w
        f = lambda t: t.permute(0, 2, 1).reshape(t.shape[0], 6, 28, 28).contiguous()
        return [f(x)], [f(de)]


def test_gamma_zero_equals_mean_patch_loss():
    m, x = Toy(), torch.randn(1, 784, 6)
    en, de = m(x)
    ref = (1 - F.cosine_similarity(en[0].detach(), de[0], dim=1)).mean()
    assert torch.equal(H.fit_loss(m, x, 0.), ref)
    assert torch.allclose(H.fit_loss(m, x, 1e-12), ref, atol=1e-6)


def test_gamma_direction_is_derivative_of_fit_gradient():
    m, x = Toy(), torch.randn(1, 784, 6)
    g = 1.3

    def grad(gamma):
        m.zero_grad(); H.fit_loss(m, x, gamma).backward(); return m.w.grad.clone()
    fd = (grad(g + 1e-3) - grad(g - 1e-3)) / 2e-3
    m.zero_grad(); H.gamma_direction_loss(m, x, g).backward()
    assert torch.allclose(m.w.grad, fd, rtol=2e-2, atol=1e-5)


def test_weights_normalised_and_emphasise_surprise():
    e = torch.tensor([[.1, .2, .4, .8]])
    w = H.weights(e, 2.)
    assert torch.allclose(w.mean(), torch.tensor(1.)) and bool((w.diff() > 0).all())


def test_tail_loss_top8():
    m, x = Toy(), torch.randn(1, 784, 6)
    e = H.patch_errors(m, x)
    assert torch.allclose(H.tail_loss(m, x), e.sort(dim=1, descending=True).values[:, :8].mean())


def test_bind_redirects_lite_runner():
    H.bind()
    assert H.L.OUT == H.OUT and H.L.REF == H.OUT and H.W.OUT == H.OUT
