"""Exact local partial derivatives and multi-clock mean-gradient commits (v3 §4)."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.func import functional_call
from .state import CMSState, finite_tensors, vector_norm


def init_affine(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


class CMSLevel(nn.Module):
    def __init__(self, dim=256, hidden=512):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        init_affine(self)

    def forward(self, x):
        return x + self.mlp(self.norm(x))


def masked_teacher_loss(prediction, target, mask):
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[0] != 1:
        raise ValueError("one image event requires matching [1,N,C] prediction/target")
    if mask.dtype != torch.bool or mask.shape != (prediction.shape[1],) or not bool(mask.any()):
        raise ValueError("mask must be a nonempty boolean token mask")
    if target.requires_grad:
        raise ValueError("teacher target must be detached")
    return 0.5 * (prediction[:, mask] - target[:, mask]).square().sum(-1).mean()


@dataclass(frozen=True)
class EventResult:
    state: CMSState
    outputs: tuple[torch.Tensor, ...]  # pre-event snapshot at each level
    losses: tuple[torch.Tensor, ...]
    diagnostics: tuple[dict, ...]


class ContinuumMemory(nn.Module):
    def __init__(self, dim=256, hidden=(512, 512), periods=(1, 8), rates=(1e-3, 1e-3)):
        super().__init__()
        if (not hidden or len(hidden) != len(periods) or len(hidden) != len(rates)
                or any(type(p) is not int or p < 1 for p in periods)
                or any(not math.isfinite(b) or b <= 0 for b in rates)):
            raise ValueError("invalid CMS dimensions/clocks/rates")
        self.levels = nn.ModuleList(CMSLevel(dim, h) for h in hidden)
        self.periods, self.rates = tuple(periods), tuple(rates)

    def initial_state(self, *, meta=False):
        weights = tuple({k: p.clone() if meta else p.detach().clone().requires_grad_(True)
                         for k, p in level.named_parameters()} for level in self.levels)
        zeros = tuple({k: torch.zeros_like(v) for k, v in w.items()} for w in weights)
        return CMSState(weights, zeros, (0,) * len(weights), (0,) * len(weights))

    def validate_state(self, state):
        if (type(state.clock) is not int or state.clock < 0
                or any(len(g) != len(self.levels) for g in
                       (state.weights, state.accumulators, state.pending, state.commits))):
            raise ValueError("invalid CMS state clock/level count")
        for i, level in enumerate(self.levels):
            template = dict(level.named_parameters())
            if (type(state.pending[i]) is not int or type(state.commits[i]) is not int
                    or state.pending[i] != state.clock % self.periods[i]
                    or state.commits[i] != state.clock // self.periods[i]):
                raise ValueError("CMS pending/commit clock mismatch")
            for group in (state.weights[i], state.accumulators[i]):
                if group.keys() != template.keys():
                    raise ValueError("CMS parameter names mismatch")
                for k, p in template.items():
                    v = group[k]
                    if v.shape != p.shape or v.dtype != p.dtype or v.device != p.device:
                        raise ValueError("CMS tensor shape/dtype/device mismatch")
                finite_tensors(group.values(), "CMS state")
            if state.pending[i] == 0 and any(bool(v.any()) for v in state.accumulators[i].values()):
                raise ValueError("zero pending count with nonzero accumulator")

    def forward(self, x, state):
        outputs = []
        for level, weights in zip(self.levels, state.weights):
            x = functional_call(level, weights, (x,), strict=True)
            outputs.append(x)
        return tuple(outputs)

    def event(self, x, target, mask, decoder, state, *, meta=False):
        """Compute all gradients before committing any level; caller owns state replacement.

        No input/decoder/history detach in meta mode. Deployment detaches only
        across events. Failures cannot mutate the supplied state or learned initials.
        """
        self.validate_state(state)
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError("CMS event is one image, never a loader batch")
        with torch.enable_grad():
            working = state if meta else state.detached()
            outputs = self.forward(x if meta else x.detach(), working)
            predictions = tuple(decoder(h) for h in outputs)
            losses = tuple(masked_teacher_loss(p, target, mask) for p in predictions)
            finite_tensors((*outputs, *predictions, *losses), "CMS forward/loss")
            grads = tuple(torch.autograd.grad(loss, tuple(w.values()), create_graph=meta,
                                              retain_graph=True)
                          for loss, w in zip(losses, working.weights))
            t = state.clock + 1
            weights, accumulators, pending, commits, diagnostics = [], [], [], [], []
            for i, (w, old, g) in enumerate(zip(working.weights, working.accumulators, grads)):
                finite_tensors(g, "CMS local gradient")
                acc = {k: old[k] + grad for k, grad in zip(w, g)}
                n = state.pending[i] + 1
                update = {k: self.rates[i] * a / n for k, a in acc.items()}
                due = t % self.periods[i] == 0
                proposal = {k: v - update[k] for k, v in w.items()} if due else dict(w)
                finite_tensors((*acc.values(), *update.values(), *proposal.values()), "CMS proposal")
                diagnostics.append(dict(gradient_norm=vector_norm(g), accumulator_norm=vector_norm(acc.values()),
                                        update_norm=vector_norm(update.values()), weight_norm=vector_norm(proposal.values()),
                                        pending_before_commit=n, committed=due))
                weights.append(proposal)
                accumulators.append({k: torch.zeros_like(v) for k, v in acc.items()} if due else acc)
                pending.append(0 if due else n)
                commits.append(state.commits[i] + int(due))
            new = CMSState(tuple(weights), tuple(accumulators), tuple(pending), tuple(commits), t)
            if not meta:
                new = new.detached()
                outputs, losses = tuple(h.detach() for h in outputs), tuple(l.detach() for l in losses)
            return EventResult(new, outputs, losses, tuple(diagnostics))

    def load_state(self, payload):
        if payload.get("schema") != "cms_state_v3":
            raise ValueError("unknown CMS checkpoint schema")
        p = next(self.parameters())
        def move(groups):
            # Do not silently coerce a checkpoint's dtype.
            return tuple({k: v.to(device=p.device).detach().clone() for k, v in w.items()} for w in groups)
        state = CMSState(move(payload["weights"]), move(payload["accumulators"]),
                         tuple(payload["pending"]), tuple(payload["commits"]), payload["clock"])
        self.validate_state(state)
        return state.detached()
