"""P2f arm F: bounded learned per-tensor consolidation on E's direct fast path.

The fast path is E's two direct steps with the learning rate fixed at .001.  F adds:
  * a per-tensor, per-step gate c = clamp(g, 0, c_max), c_max = .02, with a
    straight-through/projected backward; g is the mix output of the existing P2
    controller (the rate output is unused);
  * an EMA slow state theta <- (1-beta) theta + beta phi2, initialized from the same
    predictor initialization as phi0 and updated once per cycle;
  * a normal-only retention-aware meta-objective for the controller:
        L_meta = L_acq + lambda L_ret + mu L_plas
    with L_acq = E's step-2 current+replay query loss at phi2, L_ret = loss on up to
    8 older replay images at the query positions, and L_plas = current-support loss.

At initialization the controller output is zero and c=0, so F's deployed fast update is
exactly E's.  The slow state is EMA-only and is not trained by the meta-objective.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch.func import functional_call

from exps.task_p.p2.model import (Controller, Predictor, bounded_gradient, loss,
                                  parameters)


class ClampStraightThrough(torch.autograd.Function):
    """Forward clamp; backward passes the effective gate gradient through.

    This is the projected/straight-through alternative to the exact derivative of a
    nonnegative clamp at its zero boundary.  The forward gate is always in [lo, hi].
    """

    @staticmethod
    def forward(ctx, value, lo, hi):
        return value.clamp(lo, hi)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None, None


def statistics(fast, slow, moments, names):
    """Detached per-tensor controller context (identical form to P2/P2e)."""
    return torch.stack([
        torch.stack([
            fast[name].square().mean().sqrt(),
            (fast[name] - slow[name]).square().mean().sqrt(),
            moments[name][0].square().mean().sqrt(),
            moments[name][1].mean().sqrt(),
        ])
        for name in names
    ]).detach()


def _norm(tensors):
    values = [tensor for tensor in tensors if tensor is not None]
    if not values:
        return 0.0
    return float(torch.sqrt(sum(value.square().sum() for value in values)).detach())


def _raw_mix(controller, sample):
    """The controller's raw second output before the P2 exp(.02*) map."""
    return controller.output(torch.tanh(controller.hidden(sample)))[:, 1]


def _gate(controller, sample, gate_mode, fixed_c, c_max):
    if gate_mode == 'learned':
        return ClampStraightThrough.apply(_raw_mix(controller, sample), 0.0, c_max)
    if gate_mode == 'fixed':
        return sample.new_full((sample.shape[0],), float(fixed_c))
    raise RuntimeError(f'unknown gate_mode {gate_mode}')


def inner_update_f(model, fast, slow, controller, features, support, query, replay,
                   moments, cycle, gate_mode='learned', fixed_c=0.0, c_max=.02):
    """One F cycle's two direct fast steps. Returns graph-carrying phi2 and gates."""
    names = list(slow)

    def direction(first, second, step):
        return (first / (1 - 0.9 ** step)) / ((second / (1 - 0.999 ** step) + 1e-12).sqrt() + 1e-8)

    def rms_clip(delta):
        return delta * torch.clamp(0.02 / (delta.square().mean().sqrt() + 1e-12), max=1.0)

    # A detached leaf slow copy is the differentiable value used in w_j.  It is not
    # the EMA parameter and is never optimized; it is present so the fixed-gate modes
    # (where c carries no gradient) still have a valid inner gradient graph.
    slow_graph = OrderedDict((name, value.detach().clone().requires_grad_(True))
                             for name, value in slow.items())
    sample1 = statistics(fast, slow, moments, names)
    c1 = _gate(controller, sample1, gate_mode, fixed_c, c_max)
    initial1 = OrderedDict((name, (1 - c1[i]) * fast[name] + c1[i] * slow_graph[name])
                           for i, name in enumerate(names))
    support_loss = loss(model, initial1, features, support)
    gradients1, norm1 = bounded_gradient(
        torch.autograd.grad(support_loss, tuple(initial1.values()), create_graph=True))
    following1, first1, second1, total = OrderedDict(), {}, {}, None
    for i, (name, gradient) in enumerate(zip(names, gradients1)):
        first = 0.9 * moments[name][0] + 0.1 * gradient
        second = 0.999 * moments[name][1] + 0.001 * gradient.square()
        delta = rms_clip(.001 * direction(first, second, 2 * cycle - 1))
        following1[name] = initial1[name] - delta
        first1[name], second1[name] = first, second
        piece = delta.square().sum()
        total = piece if total is None else total + piece

    steps2 = {name: (first1[name], second1[name]) for name in names}
    sample2 = statistics(following1, slow, steps2, names)
    c2 = _gate(controller, sample2, gate_mode, fixed_c, c_max)
    initial2 = OrderedDict((name, (1 - c2[i]) * following1[name] + c2[i] * slow_graph[name])
                           for i, name in enumerate(names))
    stacked = features if replay is None else torch.stack((features, replay))
    query_loss = loss(model, initial2, stacked, query)
    gradients2, norm2 = bounded_gradient(
        torch.autograd.grad(query_loss, tuple(initial2.values()), create_graph=True))
    following, moments_after = OrderedDict(), {}
    for i, (name, gradient) in enumerate(zip(names, gradients2)):
        first = 0.9 * first1[name] + 0.1 * gradient
        second = 0.999 * second1[name] + 0.001 * gradient.square()
        delta = rms_clip(.001 * direction(first, second, 2 * cycle))
        following[name] = initial2[name] - delta
        moments_after[name] = (first.detach(), second.detach())
        piece = delta.square().sum()
        total = piece if total is None else total + piece

    # Gradient interference: support gradient at w1 vs query gradient at w2.
    flat1 = torch.cat([gradient.reshape(-1) for gradient in gradients1])
    flat2 = torch.cat([gradient.reshape(-1) for gradient in gradients2])
    cosine = float(((flat1 @ flat2) / (norm1 * norm2 + 1e-12)).detach())

    diagnostics = {
        'support_loss': float(support_loss.detach()),
        'query_loss': float(query_loss.detach()),
        'support_gradient_norm': float(norm1.detach()),
        'query_gradient_norm': float(norm2.detach()),
        'inner_delta_norm': float(torch.sqrt(total.detach())),
        'gate1_min': float(c1.min().detach()), 'gate1_max': float(c1.max().detach()),
        'gate2_min': float(c2.min().detach()), 'gate2_max': float(c2.max().detach()),
        'gate_min': float(torch.minimum(c1, c2).min().detach()),
        'gate_max': float(torch.maximum(c1, c2).max().detach()),
        'gate1_mean': float(c1.mean().detach()), 'gate2_mean': float(c2.mean().detach()),
        'gate_nonzero': int((c1 > 0).sum().item() + (c2 > 0).sum().item()),
        'gradient_cosine': cosine,
        'rate_value': 0.001,
    }
    return following, moments_after, diagnostics, c1, c2, stacked


class LearnerF:
    """F: E's direct fast path plus a learned bounded EMA consolidation gate."""

    def __init__(self, arm, seed, device='cpu', channels=768, latent=64, hidden=128, side=28,
                 beta=.02, c_max=.02, meta_lambda=1.0, meta_mu=1.0, controller_lr=1e-2,
                 gate_mode='learned', fixed_c=0.0):
        torch.manual_seed(seed)
        self.model = Predictor(channels, latent, hidden, side).to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.controller = Controller().to(device)
        self.arm = arm
        self.cycles = 0
        self.beta, self.c_max = beta, c_max
        self.meta_lambda, self.meta_mu = meta_lambda, meta_mu
        self.gate_mode, self.fixed_c = gate_mode, fixed_c
        self.controller_optimizer = torch.optim.Adam(self.controller.parameters(), lr=controller_lr)
        self.fast = OrderedDict((name, value.detach().clone())
                                for name, value in parameters(self.model).items())
        self.moments = {name: (torch.zeros_like(value), torch.zeros_like(value))
                        for name, value in self.fast.items()}
        self.initial_slow = OrderedDict((name, value.detach().clone())
                                        for name, value in parameters(self.model).items())

    def train_cycle(self, current, support, query, replay=None, old_replay=None):
        self.cycles += 1
        slow = parameters(self.model)
        following, moments, record, c1, c2, stacked = inner_update_f(
            self.model, self.fast, slow, self.controller, current, support, query, replay,
            self.moments, self.cycles, gate_mode=self.gate_mode, fixed_c=self.fixed_c,
            c_max=self.c_max)
        acquisition = loss(self.model, following, stacked, query)
        retention = (acquisition.new_zeros(()) if old_replay is None else
                     loss(self.model, following, old_replay, query))
        plasticity = loss(self.model, following, current, support)
        meta = acquisition + self.meta_lambda * retention + self.meta_mu * plasticity
        gate_gradient_norm = 0.0
        controller_norm = 0.0
        output_gradient_norm = 0.0
        hidden_gradient_norm = 0.0
        if self.gate_mode == 'learned':
            gate_gradients = torch.autograd.grad(meta, (c1, c2), retain_graph=True,
                                                 allow_unused=True)
            gate_gradient_norm = _norm(gate_gradients)
            self.controller_optimizer.zero_grad(set_to_none=True)
            meta.backward()
            controller_norm = float(torch.nn.utils.clip_grad_norm_(
                self.controller.parameters(), 1.0, error_if_nonfinite=True))
            output_gradient_norm = _norm(parameter.grad for parameter in self.controller.output.parameters())
            hidden_gradient_norm = _norm(parameter.grad for parameter in self.controller.hidden.parameters())
            self.controller_optimizer.step()
        with torch.no_grad():
            for name, theta in slow.items():
                theta.mul_(1 - self.beta).add_(following[name].detach(), alpha=self.beta)
        displacement = float(torch.sqrt(sum((following[name].detach() - self.fast[name]).square().sum()
                                            for name in self.fast)))
        self.fast = OrderedDict((name, value.detach().clone()) for name, value in following.items())
        self.moments = moments
        if not all(bool(torch.isfinite(value).all()) for value in self.fast.values()):
            raise RuntimeError('nonfinite fast parameters')
        if not all(bool(torch.isfinite(value).all()) for value in slow.values()):
            raise RuntimeError('nonfinite slow parameters')
        slow_fast_distance = float(torch.sqrt(sum((self.fast[name] - slow[name].detach()).square().sum()
                                                  for name in self.fast)))
        slow_from_initial = float(torch.sqrt(sum((slow[name].detach() - self.initial_slow[name]).square().sum()
                                                 for name in self.fast)))
        record.update(meta_acquisition=float(acquisition.detach()),
                      meta_retention=float(retention.detach()),
                      meta_plasticity=float(plasticity.detach()),
                      meta_total=float(meta.detach()),
                      gate_gradient_norm=gate_gradient_norm,
                      controller_gradient_norm=controller_norm,
                      controller_output_gradient_norm=output_gradient_norm,
                      controller_hidden_gradient_norm=hidden_gradient_norm,
                      fast_parameter_displacement=displacement,
                      parameter_norm=float(torch.sqrt(sum(value.square().sum()
                                                         for value in self.fast.values()))),
                      slow_fast_distance=slow_fast_distance,
                      slow_from_initial_distance=slow_from_initial)
        return record

    @torch.no_grad()
    def scores(self, features):
        prediction = functional_call(self.model, self.fast, (features,))
        return (prediction.float() - features.float()).square().sum(-1).sqrt()

    @torch.no_grad()
    def slow_scores(self, features):
        prediction = functional_call(self.model, parameters(self.model), (features,))
        return (prediction.float() - features.float()).square().sum(-1).sqrt()

    def state(self):
        return {'arm': self.arm, 'cycles': self.cycles, 'model': self.model.state_dict(),
                'controller': self.controller.state_dict(),
                'controller_optimizer': self.controller_optimizer.state_dict(),
                'fast': self.fast, 'moments': self.moments,
                'gate_mode': self.gate_mode, 'fixed_c': self.fixed_c,
                'beta': self.beta, 'c_max': self.c_max,
                'meta_lambda': self.meta_lambda, 'meta_mu': self.meta_mu}

    def restore(self, state):
        self.model.load_state_dict(state['model'])
        self.controller.load_state_dict(state['controller'])
        self.controller_optimizer.load_state_dict(state['controller_optimizer'])
        self.cycles = state['cycles']
        self.gate_mode = state.get('gate_mode', self.gate_mode)
        self.fixed_c = state.get('fixed_c', self.fixed_c)
        self.beta = state.get('beta', self.beta)
        self.c_max = state.get('c_max', self.c_max)
        self.meta_lambda = state.get('meta_lambda', self.meta_lambda)
        self.meta_mu = state.get('meta_mu', self.meta_mu)
        device = next(self.model.parameters()).device
        self.fast = OrderedDict((name, value.to(device)) for name, value in state['fast'].items())
        self.moments = {name: tuple(value.to(device) for value in pair)
                        for name, pair in state['moments'].items()}
