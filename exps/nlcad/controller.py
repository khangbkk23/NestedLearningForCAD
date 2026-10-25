# exps/nlcad/controller.py
"""NL-CAD slow level: a compact blockwise update controller for the Dinomaly decoder.

Slow state phi (one scalar per decoder block) sets update multipliers
    a_b = exp(h_b) / mean_j exp(h_j),   h_b = c * tanh(phi_b),   c = log(2) / 2,
so mean_b a_b = 1 (no global learning-rate change), max_b a_b / min_b a_b <= 2, and phi = 0 gives a_b = 1 exactly
(identity controller). The fast level (deployed decoder) applies a_b to its optimizer update of block b; the slow
level learns phi on its own clock from the consequence of that update on an outer objective (one-step look-ahead
hypergradient; the base update direction is held fixed, so no Hessian is required):

    theta+_b = theta_b + Delta_b,  Delta_b = a_b * Delta_b^(a=1)
    dJ/da_b  = <grad_b J(theta+), Delta_b> / a_b
    dJ/dphi  = sum_b dJ/da_b * da_b/dphi + 2 * lam * phi
"""
from __future__ import annotations

import math

import torch

BLOCKS = ["bottleneck.0"] + [f"decoder.{i}" for i in range(8)]
C_BOUND = math.log(2.0) / 2


def block_of(name):
    for b in BLOCKS:
        if name.startswith(b + "."):
            return b
    raise KeyError(f"parameter outside the controlled blocks: {name}")


class BlockController(torch.nn.Module):
    def __init__(self, lr=0.05, weight_decay=1e-3, c=C_BOUND, betas=(0.9, 0.999)):
        super().__init__()
        self.phi = torch.nn.Parameter(torch.zeros(len(BLOCKS), dtype=torch.float64))
        self.c, self.lam = c, weight_decay
        self.opt = torch.optim.Adam([self.phi], lr=lr, betas=betas)
        self.events = 0

    def multipliers(self):
        h = self.c * torch.tanh(self.phi)
        return torch.exp(h) / torch.exp(h).mean()

    def a(self):
        """Detached multipliers as a {block: float} map; exactly 1.0 for the identity controller."""
        if not bool(self.phi.detach().any()):
            return {b: 1.0 for b in BLOCKS}
        return dict(zip(BLOCKS, self.multipliers().detach().tolist()))

    def grad_phi(self, s, lam=None):
        """dJ/dphi from s[b] = dJ/da_b (chain rule through the normalized, bounded multipliers) + 2*lam*phi."""
        self.opt.zero_grad(set_to_none=True)
        m = self.multipliers()
        sv = torch.tensor([s[b] for b in BLOCKS], dtype=torch.float64)
        lam = self.lam if lam is None else lam
        (torch.dot(sv, m) + lam * (self.phi ** 2).sum()).backward()
        return self.phi.grad.detach().clone()

    def slow_step(self, s):
        """s[b] = dJ/da_b (detached scalars). One Adam step on phi; returns the hypergradient and diagnostics."""
        g = self.grad_phi(s)
        if torch.isfinite(g).all():
            self.opt.step()
            self.events += 1
            ok = True
        else:
            self.opt.zero_grad(set_to_none=True)
            ok = False
        return {"hypergrad": g.tolist(), "applied": ok, "phi": self.phi.detach().tolist(),
                "a": self.multipliers().detach().tolist(),
                "saturation": float((torch.tanh(self.phi.detach()).abs() > 0.95).double().mean())}

    def persistent_bytes(self):
        """phi + Adam moments (FP64) + step counter."""
        return self.phi.numel() * 8 * 3 + 8

    def snapshot(self):
        return {"phi": self.phi.detach().clone(), "opt": self.opt.state_dict(), "events": self.events,
                "c": self.c, "lam": self.lam}

    def restore(self, snap):
        with torch.no_grad():
            self.phi.copy_(snap["phi"])
        self.opt.load_state_dict(snap["opt"])
        self.events, self.c, self.lam = snap["events"], snap["c"], snap["lam"]
