"""Common bottleneck predictor and differentiable bounded update."""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as functional


class Predictor(nn.Module):
    def __init__(self, channels=768, latent=64, hidden=128, side=28):
        super().__init__()
        self.channels, self.latent, self.side = channels, latent, side
        self.encode = nn.Linear(channels, latent)
        self.hidden = nn.Linear(2 * latent + 2, hidden)
        self.decode = nn.Linear(hidden, channels)
        coordinate = torch.linspace(-1, 1, side)
        vertical, horizontal = torch.meshgrid(coordinate, coordinate, indexing="ij")
        self.register_buffer("coordinates", torch.stack((vertical, horizontal), -1).reshape(-1, 2))

    def forward(self, features):
        unbatched = features.ndim == 2
        if unbatched:
            features = features.unsqueeze(0)
        latent = functional.gelu(self.encode(features))
        grid = latent.transpose(1, 2).reshape(-1, self.latent, self.side, self.side)
        padded = functional.pad(grid, (1, 1, 1, 1), mode="replicate")
        context = (padded[:, :, :-2, 1:-1] + padded[:, :, 2:, 1:-1]
                   + padded[:, :, 1:-1, :-2] + padded[:, :, 1:-1, 2:]) / 4
        context = context.flatten(2).transpose(1, 2)
        position = self.coordinates.expand(features.shape[0], -1, -1)
        prediction = self.decode(functional.gelu(self.hidden(torch.cat((latent, context, position), -1))))
        return prediction[0] if unbatched else prediction


class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 16)
        self.output = nn.Linear(16, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, statistics):
        gates = torch.tanh(self.output(torch.tanh(self.hidden(statistics))))
        rate = 0.001 * torch.exp(gates[:, 0])
        mix = 0.02 * torch.exp(gates[:, 1])
        return rate, mix


def parameters(model):
    return OrderedDict(model.named_parameters())


def loss(model, weights, features, indices):
    prediction = functional_call(model, weights, (features,))
    return (prediction[..., indices, :] - features[..., indices, :]).square().mean()


def bounded_gradient(gradients, limit=1.0):
    norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients))
    scale = torch.clamp(limit / (norm + 1e-12), max=1.0)
    return [gradient * scale for gradient in gradients], norm


def inner_update(model, fast, slow, controller, features, indices, moments, step,
                 learned=True, disable_inner=False):
    names = list(slow)
    statistics = torch.stack([
        torch.stack((fast[name].square().mean().sqrt(),
                     (fast[name] - slow[name]).square().mean().sqrt(),
                     moments[name][0].square().mean().sqrt(),
                     moments[name][1].mean().sqrt()))
        for name in names
    ]).detach()
    if learned:
        rates, mixing = controller(statistics)
    else:
        rates = statistics.new_full((len(names),), 0.001)
        mixing = statistics.new_full((len(names),), 0.02)
    initial = OrderedDict((name, (1 - mixing[index]) * fast[name]
                           + mixing[index] * slow[name]) for index, name in enumerate(names))
    support = loss(model, initial, features, indices)
    gradients = torch.autograd.grad(support, tuple(initial.values()), create_graph=True)
    gradients, norm = bounded_gradient(gradients)
    following, updated_moments = OrderedDict(), {}
    updates = []
    for index, (name, gradient) in enumerate(zip(names, gradients)):
        first = 0.9 * moments[name][0] + 0.1 * gradient
        second = 0.999 * moments[name][1] + 0.001 * gradient.square()
        direction = (first / (1 - 0.9 ** step)) / ((second / (1 - 0.999 ** step) + 1e-12).sqrt() + 1e-8)
        delta = rates[index] * direction
        delta = delta * torch.clamp(0.02 / (delta.square().mean().sqrt() + 1e-12), max=1)
        following[name] = initial[name] if disable_inner else initial[name] - delta
        updated_moments[name] = (first.detach(), second.detach())
        updates.append(delta.detach().square().sum())
    diagnostics = {"support_loss": float(support.detach()), "support_gradient_norm": float(norm.detach()),
                   "inner_delta_norm": float(torch.sqrt(sum(updates))),
                   "rate_min": float(rates.min().detach()), "rate_max": float(rates.max().detach()),
                   "mix_min": float(mixing.min().detach()), "mix_max": float(mixing.max().detach())}
    return following, updated_moments, diagnostics


class Learner:
    def __init__(self, arm, seed, device="cpu", channels=768, latent=64, hidden=128, side=28):
        torch.manual_seed(seed)
        self.model = Predictor(channels, latent, hidden, side).to(device)
        self.controller = Controller().to(device)
        self.arm, self.steps = arm, 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001 if arm in ("A", "B") else 0.0001)
        self.controller_optimizer = torch.optim.Adam(self.controller.parameters(), lr=0.001)
        self.fast = OrderedDict((name, value.detach().clone()) for name, value in parameters(self.model).items())
        self.moments = {name: (torch.zeros_like(value), torch.zeros_like(value)) for name, value in self.fast.items()}

    def train_cycle(self, current, support_indices, query_indices, replay=None):
        self.steps += 1
        query = current.unsqueeze(0) if replay is None or self.arm == "A" else torch.stack((current, replay))
        if self.arm in ("A", "B"):
            records = []
            for features, indices in ((current, support_indices), (query, query_indices)):
                self.optimizer.zero_grad(set_to_none=True)
                objective = loss(self.model, parameters(self.model), features, indices)
                objective.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
                self.optimizer.step()
                records.append((float(objective.detach()), float(norm)))
            return {"support_loss": records[0][0], "query_loss": records[1][0],
                    "support_gradient_norm": records[0][1], "outer_gradient_norm": records[1][1],
                    "controller_gradient_norm": 0.0,
                    "parameter_norm": float(torch.sqrt(sum(value.detach().square().sum() for value in self.model.parameters())))}
        self.optimizer.zero_grad(set_to_none=True)
        self.controller_optimizer.zero_grad(set_to_none=True)
        following, moments, record = inner_update(
            self.model, self.fast, parameters(self.model), self.controller, current, support_indices,
            self.moments, self.steps, learned=self.arm != "C", disable_inner=self.arm == "D_noinner")
        objective = loss(self.model, following, query, query_indices)
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        controller_norm = 0.0
        if self.arm != "C":
            controller_norm = float(torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 1.0,
                                                                  error_if_nonfinite=True))
            self.controller_optimizer.step()
        self.optimizer.step()
        displacement = float(torch.sqrt(sum((following[name].detach()-self.fast[name]).square().sum() for name in self.fast)))
        self.fast = OrderedDict((name, value.detach().clone()) for name, value in following.items())
        self.moments = moments
        if not all(bool(torch.isfinite(value).all()) for value in self.fast.values()):
            raise RuntimeError("nonfinite fast parameters")
        record.update(query_loss=float(objective.detach()), outer_gradient_norm=float(norm),
                      controller_gradient_norm=controller_norm, fast_parameter_displacement=displacement,
                      parameter_norm=float(torch.sqrt(sum(value.square().sum() for value in self.fast.values()))),
                      fast_slow_distance=float(torch.sqrt(sum((self.fast[name]-value.detach()).square().sum()
                                                             for name, value in parameters(self.model).items()))))
        return record

    @torch.no_grad()
    def scores(self, features):
        weights = parameters(self.model) if self.arm in ("A", "B") else self.fast
        prediction = functional_call(self.model, weights, (features,))
        return (prediction.float() - features.float()).square().sum(-1).sqrt()

    def state(self):
        return {"arm": self.arm, "steps": self.steps, "model": self.model.state_dict(),
                "controller": self.controller.state_dict(), "optimizer": self.optimizer.state_dict(),
                "controller_optimizer": self.controller_optimizer.state_dict(),
                "fast": self.fast, "moments": self.moments}

    def restore(self, state):
        self.model.load_state_dict(state["model"])
        self.controller.load_state_dict(state["controller"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.controller_optimizer.load_state_dict(state["controller_optimizer"])
        self.steps = state["steps"]
        device = next(self.model.parameters()).device
        self.fast = OrderedDict((name, value.to(device)) for name, value in state["fast"].items())
        self.moments = {name: tuple(value.to(device) for value in pair) for name, pair in state["moments"].items()}
