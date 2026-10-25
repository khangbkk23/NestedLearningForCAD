# exps/nlcad/controller_r1.py
"""Proposed revision of the NLCAD-01 slow level (NOT used by Stage C; for a separate, later task).
Kept apart from controller.py so that the Stage C sources stay byte-identical to their launched versions.

SUPERSEDED (2026-10-25, after Stage C finished): the R1 rationale below misread the evidence. The recorded
"hypergrad" includes the L2 term 2*lam*phi; with lam = 1e-3 that term was ~5x the signal (median 2.7e-5 vs 5.3e-6),
so the sign instability was the penalty balancing the signal, not heavy-tailed noise. R1 keeps lam and would converge
to the same near-identity equilibrium. See agents/reports/task_nlcad_01_stage_cd_variant_n.md, section 3."""
from __future__ import annotations

import math

import torch

from exps.nlcad.controller import BLOCKS, C_BOUND, BlockController


class MeanController(BlockController):
    """Revision R1/R2 of the slow level (grounded in Stage C diagnostics, seed 2711 stage 7; no AP used).

    R1: per-event look-ahead hypergradients are heavy-tailed with unstable signs (block sign consistency ~0.5) while
        their means are consistent (|t| = 4-5 over 20 events). Adam's normalized step follows the majority sign, so
        phi random-walked around 0. Here the slow step follows the MEAN hypergradient of a window of W events
        (slower clock: W*K fast steps): phi <- phi - eta * g_bar / rms(g_bar).
    R2 (optional): the global component of dJ/da (sum over blocks) was the most consistent signal (positive in 20/20
        events), and the mean-1 normalization discards it by design. An optional bounded global scale
        gamma = exp(cg * tanh(psi)), cg = ln 2, gamma in [1/2, 2], is learned by the same rule.
    phi = 0 and psi = 0 remain the exact identity."""

    def __init__(self, step=0.1, window=5, weight_decay=1e-3, c=C_BOUND, global_scale=False, cg=math.log(2.0)):
        super().__init__(lr=0.0, weight_decay=weight_decay, c=c)
        self.psi = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64)) if global_scale else None
        self.step_size, self.window, self.cg = step, window, cg
        self.buf, self.updates = [], 0

    def gamma(self):
        return torch.exp(self.cg * torch.tanh(self.psi))[0] if self.psi is not None else torch.ones((), dtype=torch.float64)

    def multipliers(self):
        return super().multipliers() * self.gamma()

    def a(self):
        if not bool(self.phi.detach().any()) and (self.psi is None or not bool(self.psi.detach().any())):
            return {b: 1.0 for b in BLOCKS}
        return dict(zip(BLOCKS, self.multipliers().detach().tolist()))

    def _params(self):
        return [self.phi] + ([self.psi] if self.psi is not None else [])

    def grad_all(self, s, lam=None):
        for q in self._params():
            q.grad = None
        m = self.multipliers()
        sv = torch.tensor([s[b] for b in BLOCKS], dtype=torch.float64)
        lam = self.lam if lam is None else lam
        (torch.dot(sv, m) + lam * sum((q ** 2).sum() for q in self._params())).backward()
        g = torch.cat([q.grad.detach().clone().flatten() for q in self._params()])
        for q in self._params():
            q.grad = None
        return g

    def grad_phi(self, s, lam=None):
        return self.grad_all(s, lam)[:len(BLOCKS)]

    def slow_step(self, s):
        g = self.grad_all(s)
        ok = bool(torch.isfinite(g).all())
        moved = False
        if ok:
            self.buf.append(g)
            self.events += 1
            if len(self.buf) >= self.window:
                gb = torch.stack(self.buf).mean(0)
                rms = gb.pow(2).mean().sqrt()
                if float(rms) > 0:
                    with torch.no_grad():
                        k = 0
                        for q in self._params():
                            n = q.numel()
                            q.sub_(self.step_size * gb[k:k + n] / rms)
                            k += n
                    moved = True
                    self.updates += 1
                self.buf = []
        return {"hypergrad": g.tolist(), "applied": ok, "slow_update": moved, "phi": self.phi.detach().tolist(),
                "psi": None if self.psi is None else float(self.psi.detach()), "gamma": float(self.gamma().detach()),
                "a": self.multipliers().detach().tolist(),
                "saturation": float((torch.tanh(self.phi.detach()).abs() > 0.95).double().mean())}

    def persistent_bytes(self):
        """phi (+ psi) + window buffer (FP64) + counters."""
        n = len(BLOCKS) + (1 if self.psi is not None else 0)
        return n * 8 * (1 + self.window) + 16

    def snapshot(self):
        return {"phi": self.phi.detach().clone(), "psi": None if self.psi is None else self.psi.detach().clone(),
                "buf": [b.clone() for b in self.buf], "events": self.events, "updates": self.updates,
                "step": self.step_size, "window": self.window, "c": self.c, "cg": self.cg, "lam": self.lam}

    def restore(self, snap):
        with torch.no_grad():
            self.phi.copy_(snap["phi"])
            if self.psi is not None:
                self.psi.copy_(snap["psi"])
        self.buf = [b.clone() for b in snap["buf"]]
        self.events, self.updates = snap["events"], snap["updates"]
        self.step_size, self.window, self.c, self.cg, self.lam = (snap["step"], snap["window"], snap["c"], snap["cg"],
                                                                  snap["lam"])
