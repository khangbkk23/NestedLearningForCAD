"""P2e single-variable intervention on the frozen P2d C2 update.

Contract (frozen in agents/tasks/current_task_p2e.md):
  Step 1 direct support update:   w1 = (1-mix1) phi + mix1 theta; gradient at w1;
                                  one bias-corrected Adam step -> phi1.
  Step 2 direct query+replay:     w2 = (1-mix2) phi1 + mix2 theta; gradient of the
                                  current-image query and replay query at w2;
                                  one bias-corrected Adam step -> phi2 (committed).
  Outer update:                   L2 at phi2 trains theta (Adam 1e-4) and, in D2 only,
                                  the controller only for the unused D2 branch.

Scientific execution uses arm E: fixed alpha .001 and mix zero. All executable
code is identical to P2d model.py except the fixed mix literal (.02 -> 0).

The predictor, controller, loss, gradient clipping and readout are imported unchanged
from the pinned P2 modules so the architecture is bitwise the P2 architecture.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch.func import functional_call

from exps.task_p.p2.model import (Controller, Predictor, bounded_gradient, loss,
                                  parameters)


def statistics(fast, slow, moments, names):
    """Detached per-tensor controller context (identical form to P2)."""
    return torch.stack([
        torch.stack([
            fast[name].square().mean().sqrt(),
            (fast[name] - slow[name]).square().mean().sqrt(),
            moments[name][0].square().mean().sqrt(),
            moments[name][1].mean().sqrt(),
        ])
        for name in names
    ]).detach()


def inner_update2(model, fast, slow, controller, features, support, query, replay, moments, cycle,
                  learned=True):
    """One cycle of the frozen two-step update. Returns (phi2, moments, diagnostics)."""
    names = list(slow)

    def gates(sample):
        if learned:
            return controller(sample)
        rate = sample.new_full((len(names),), 0.001)
        mix = sample.new_full((len(names),), 0.0)
        return rate, mix

    def direction(first, second, step):
        return (first / (1 - 0.9 ** step)) / ((second / (1 - 0.999 ** step) + 1e-12).sqrt() + 1e-8)

    def rms_clip(delta):
        return delta * torch.clamp(0.02 / (delta.square().mean().sqrt() + 1e-12), max=1.0)

    sample1 = statistics(fast, slow, moments, names)
    rate1, mix1 = gates(sample1)
    initial1 = OrderedDict((name, (1 - mix1[i]) * fast[name] + mix1[i] * slow[name])
                           for i, name in enumerate(names))
    support_loss = loss(model, initial1, features, support)
    gradients1, norm1 = bounded_gradient(
        torch.autograd.grad(support_loss, tuple(initial1.values()), create_graph=True))
    following1, first1, second1, total = OrderedDict(), {}, {}, None
    for i, (name, gradient) in enumerate(zip(names, gradients1)):
        first = 0.9 * moments[name][0] + 0.1 * gradient
        second = 0.999 * moments[name][1] + 0.001 * gradient.square()
        delta = rms_clip(rate1[i] * direction(first, second, 2 * cycle - 1))
        following1[name] = initial1[name] - delta
        first1[name], second1[name] = first, second
        piece = delta.square().sum()
        total = piece if total is None else total + piece

    steps2 = {name: (first1[name], second1[name]) for name in names}
    sample2 = statistics(following1, slow, steps2, names)
    rate2, mix2 = gates(sample2)
    initial2 = OrderedDict((name, (1 - mix2[i]) * following1[name] + mix2[i] * slow[name])
                           for i, name in enumerate(names))
    stacked = features if replay is None else torch.stack((features, replay))
    query_loss = loss(model, initial2, stacked, query)
    gradients2, norm2 = bounded_gradient(
        torch.autograd.grad(query_loss, tuple(initial2.values()), create_graph=True))
    following, moments_after = OrderedDict(), {}
    for i, (name, gradient) in enumerate(zip(names, gradients2)):
        first = 0.9 * first1[name] + 0.1 * gradient
        second = 0.999 * second1[name] + 0.001 * gradient.square()
        delta = rms_clip(rate2[i] * direction(first, second, 2 * cycle))
        following[name] = initial2[name] - delta
        moments_after[name] = (first.detach(), second.detach())
        piece = delta.square().sum()
        total = piece if total is None else total + piece

    diagnostics = {
        "support_loss": float(support_loss.detach()),
        "query_loss": float(query_loss.detach()),
        "support_gradient_norm": float(norm1.detach()),
        "query_gradient_norm": float(norm2.detach()),
        "inner_delta_norm": float(torch.sqrt(total.detach())),
        "rate1_min": float(rate1.min().detach()), "rate1_max": float(rate1.max().detach()),
        "rate2_min": float(rate2.min().detach()), "rate2_max": float(rate2.max().detach()),
        "mix1_min": float(mix1.min().detach()), "mix1_max": float(mix1.max().detach()),
        "mix2_min": float(mix2.min().detach()), "mix2_max": float(mix2.max().detach()),
        "rate_min": float(torch.minimum(rate1, rate2).min().detach()),
        "rate_max": float(torch.maximum(rate1, rate2).max().detach()),
        "mix_min": float(torch.minimum(mix1, mix2).min().detach()),
        "mix_max": float(torch.maximum(mix1, mix2).max().detach()),
    }
    return following, moments_after, diagnostics


class Learner2:
    """E uses the C2 path with zero mixing and retained slow bookkeeping."""

    def __init__(self, arm, seed, device="cpu", channels=768, latent=64, hidden=128, side=28):
        torch.manual_seed(seed)
        self.model = Predictor(channels, latent, hidden, side).to(device)
        self.controller = Controller().to(device)
        self.arm = arm
        self.cycles = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.controller_optimizer = torch.optim.Adam(self.controller.parameters(), lr=0.001)
        self.fast = OrderedDict((name, value.detach().clone())
                                for name, value in parameters(self.model).items())
        self.moments = {name: (torch.zeros_like(value), torch.zeros_like(value))
                        for name, value in self.fast.items()}

    def train_cycle(self, current, support, query, replay=None):
        self.cycles += 1
        query_stack = current if replay is None else torch.stack((current, replay))
        self.optimizer.zero_grad(set_to_none=True)
        self.controller_optimizer.zero_grad(set_to_none=True)
        following, moments, record = inner_update2(
            self.model, self.fast, parameters(self.model), self.controller, current, support, query,
            replay, self.moments, self.cycles, learned=self.arm == "D2")
        objective = loss(self.model, following, query_stack, query)
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        controller_norm = 0.0
        if self.arm == "D2":
            controller_norm = float(torch.nn.utils.clip_grad_norm_(
                self.controller.parameters(), 1.0, error_if_nonfinite=True))
            self.controller_optimizer.step()
        self.optimizer.step()
        displacement = float(torch.sqrt(sum((following[name].detach() - self.fast[name]).square().sum()
                                            for name in self.fast)))
        self.fast = OrderedDict((name, value.detach().clone()) for name, value in following.items())
        self.moments = moments
        if not all(bool(torch.isfinite(value).all()) for value in self.fast.values()):
            raise RuntimeError("nonfinite fast parameters")
        record.update(outer_loss=float(objective.detach()), outer_gradient_norm=float(norm),
                      controller_gradient_norm=controller_norm,
                      fast_parameter_displacement=displacement,
                      parameter_norm=float(torch.sqrt(sum(value.square().sum()
                                                          for value in self.fast.values()))),
                      fast_slow_distance=float(torch.sqrt(sum(
                          (self.fast[name] - value.detach()).square().sum()
                          for name, value in parameters(self.model).items()))))
        return record

    @torch.no_grad()
    def scores(self, features):
        prediction = functional_call(self.model, self.fast, (features,))
        return (prediction.float() - features.float()).square().sum(-1).sqrt()

    def state(self):
        return {"arm": self.arm, "cycles": self.cycles, "model": self.model.state_dict(),
                "controller": self.controller.state_dict(), "optimizer": self.optimizer.state_dict(),
                "controller_optimizer": self.controller_optimizer.state_dict(),
                "fast": self.fast, "moments": self.moments}

    def restore(self, state):
        self.model.load_state_dict(state["model"])
        self.controller.load_state_dict(state["controller"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.controller_optimizer.load_state_dict(state["controller_optimizer"])
        self.cycles = state["cycles"]
        device = next(self.model.parameters()).device
        self.fast = OrderedDict((name, value.to(device)) for name, value in state["fast"].items())
        self.moments = {name: tuple(value.to(device) for value in pair)
                        for name, pair in state["moments"].items()}

