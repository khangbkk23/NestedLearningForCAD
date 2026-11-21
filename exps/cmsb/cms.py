"""CMS-B core: a Continuum Memory System of LoRA levels on the frozen BASE decoder (NL §7.1, Eq. 70-74).

Every adapted Linear computes  y = W0 x + sum_l B_l A_l x  over the ACTIVE levels (head-wise CMS, Agg = sum).
Level l has its own update period C_l (Eq. 71: gradients are accumulated over C_l fast steps and applied once),
its own context (current-task stream, optionally a bounded replay reservoir), its own optimizer and its own reset
rule. Knowledge moves between levels through backpropagation in a shared forward pass (§3.3) and through
re-initialisation of the fast level at the end of its context (Eq. 72).

Inference (task-free, bounded): min over two passes, slow-only and all levels.
Memory is constant in the number of tasks: one (A, B) pair per level per site, plus the replay reservoir.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from exps.dinomaly import official as O
from exps.hope_s_01 import run as H
from exps.nar_pilot_00 import core as K, full as R
from exps.nar_warm_01 import run as W
from exps.vitb8_q2.common import SingleStreamDecoder


@dataclass
class Level:
    name: str
    period: int                 # C_l in fast steps (1 = every step)
    reset_each_task: bool       # re-initialise at the end of its context (Eq. 72) and start fresh moments
    lr: float = 2e-3
    replay: bool = False        # slow-only replay gradients enter this level's accumulator


@dataclass
class Spec:
    levels: tuple = field(default_factory=lambda: (Level('fast', 1, True), Level('slow', 10, False)))
    gamma: float = 2.1
    rank: int = 8
    steps: int = 300
    slow_warmup: int = 20       # slow-level optimizer steps of linear warm-up (first task only)
    replay_bytes: int = 0       # reservoir cap (0 = no replay)
    replay_every: int = 1       # one slow-only replay draw every k fast steps (when replay is on)

    def dict(self):
        d = asdict(self)
        d['levels'] = [asdict(l) for l in self.levels]
        return d


class CMSLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, n_levels: int):
        super().__init__()
        self.base = base
        for q in base.parameters():
            q.requires_grad_(False)
        dev = base.weight.device
        self.A = nn.ParameterList([nn.Parameter(torch.zeros(rank, base.in_features, device=dev)) for _ in range(n_levels)])
        self.B = nn.ParameterList([nn.Parameter(torch.zeros(base.out_features, rank, device=dev)) for _ in range(n_levels)])
        self.active = list(range(n_levels))

    def forward(self, x):
        y = self.base(x)
        for l in self.active:
            y = y + F.linear(F.linear(x, self.A[l].to(x.dtype)), self.B[l].to(x.dtype))
        return y


def build(spec: Spec, base=None):
    """Frozen BASE decoder (regenerable from seed 2711) with CMS adapters on its 34 Linear sites."""
    torch.manual_seed(2711)
    model = SingleStreamDecoder(checkpoint=True).cuda()
    if base is None:
        O.init_trainable(model)
    else:
        model.load_state_dict(base)
    for q in model.parameters():
        q.requires_grad_(False)
    mods = dict(model.named_modules())
    names = [n for n, m in mods.items() if isinstance(m, nn.Linear)]
    sites = {}
    for n in names:
        parent = mods[n.rsplit('.', 1)[0]] if '.' in n else model
        w = CMSLinear(mods[n], spec.rank, len(spec.levels))
        setattr(parent, n.rsplit('.', 1)[-1], w)
        sites[n] = w
    return model, sites


def set_active(sites, levels):
    for w in sites.values():
        w.active = list(levels)


def level_params(sites, l):
    return [p for w in sites.values() for p in (w.A[l], w.B[l])]


@torch.no_grad()
def init_level(sites, l, seed):
    """A ~ U(+-1/sqrt(in)), B = 0: the level's increment starts at exactly zero (same rule as K.init_state)."""
    g = torch.Generator().manual_seed(seed)
    for w in sites.values():
        b = 1 / math.sqrt(w.A[l].shape[1])
        w.A[l].copy_(((torch.rand(w.A[l].shape, generator=g) * 2 - 1) * b).to(w.A[l]))
        w.B[l].zero_()


def get_state(sites):
    return {n: ([a.detach().cpu().clone() for a in w.A], [b.detach().cpu().clone() for b in w.B])
            for n, w in sites.items()}


@torch.no_grad()
def set_state(sites, st, dtype=None):
    for n, w in sites.items():
        for l in range(len(w.A)):
            w.A[l].copy_(st[n][0][l].to(w.A[l]))
            w.B[l].copy_(st[n][1][l].to(w.B[l]))


def state_bytes(st):
    return sum(x.numel() * x.element_size() for a, b in st.values() for x in a + b)


def optimizer(params, lr):
    cfg = R.CFG
    return R.O.StableAdamW([{'params': params}], lr=lr, betas=cfg.betas, weight_decay=cfg.weight_decay,
                           eps=cfg.eps, amsgrad=True)


ITEM_BYTES = 200_000            # per-image JPEG ceiling (392x392)
QUALITY = 100                   # owner decision 2026-11-18: full quality. Normal-only fidelity check (24 FIT images):
                                # JPEG q100 mean token cosine 0.995 (min 0.991) to the training tokens, ~110 KB/image


def jpeg392(path):
    """Normal train image -> 392x392 JPEG bytes <= ITEM_BYTES (quality 100, lowered in steps of 5 if needed)."""
    import io
    from PIL import Image
    from exps.cmsb import dev as D
    a = D.load392(path)
    for q in range(QUALITY, 40, -5):
        b = io.BytesIO()
        Image.fromarray(a).save(b, 'JPEG', quality=q)
        if len(b.getvalue()) <= ITEM_BYTES:
            return b.getvalue()
    raise RuntimeError(f'cannot fit {path} under {ITEM_BYTES} bytes')


def decode(b):
    import io
    import numpy as np
    from PIL import Image
    from exps.cmsb import dev as D
    with Image.open(io.BytesIO(b)) as im:
        return D.normalise(np.array(im.convert('RGB'), dtype=np.uint8))


class Reservoir:
    """Byte-capped reservoir of old-task normal FIT images as JPEG bytes (Algorithm R on the item stream, then
    random eviction until the total is under the cap). Total bytes <= cap at all times, independent of the
    number of tasks."""

    def __init__(self, cap_bytes, seed):
        self.cap_bytes = int(cap_bytes)
        self.items, self.tags, self.seen, self.full, self.total = [], [], 0, False, 0
        self.g = torch.Generator().manual_seed(seed)

    @property
    def cap(self):
        return len(self.items)

    def _pop(self, j):
        self.total -= len(self.items[j])
        self.items.pop(j); self.tags.pop(j)

    def offer(self, b, tag=None):
        K.require(len(b) <= self.cap_bytes, 'item larger than reservoir')
        self.seen += 1
        if not self.full and self.total + len(b) <= self.cap_bytes:
            self.items.append(b); self.tags.append(tag); self.total += len(b)
            return
        self.full = True
        j = int(torch.randint(self.seen, (1,), generator=self.g))
        if j < len(self.items) and self.total + len(b) <= self.cap_bytes:     # accepted and room: grow
            self.items.append(b); self.tags.append(tag); self.total += len(b)
        elif j < len(self.items):
            self.total += len(b) - len(self.items[j])
            self.items[j] = b; self.tags[j] = tag
            while self.total > self.cap_bytes:
                self._pop(int(torch.randint(len(self.items), (1,), generator=self.g)))

    def draw(self):
        return self.items[int(torch.randint(len(self.items), (1,), generator=self.g))]

    def bytes(self):
        K.require(self.total == sum(len(b) for b in self.items), 'reservoir byte ledger')
        return self.total


def train(spec: Spec, seed, tasks, load, log=print, budget=None, encode=None):
    """tasks: ordered names; load(c) -> (features [N,784,768], roles, rows). Returns per-stage states + trace.
    encode(jpeg_bytes) -> [784,768] tokens with the training front end (required when replay is on).

    Fast step i: forward with ALL levels on a current FIT image, backward once. Every level accumulates its
    gradient; level l applies the mean of its accumulator every C_l steps (Eq. 71). Replay (if any) adds a
    slow-only forward/backward on a reservoir draw whose gradient enters only levels with replay=True.
    """
    base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
    model, sites = build(spec, base)
    L = len(spec.levels)
    for l in range(L):
        init_level(sites, l, seed + 1000 * l)
    fast_init = {l: seed + 1000 * l for l, lv in enumerate(spec.levels) if lv.reset_each_task}
    opts = {l: optimizer(level_params(sites, l), lv.lr) for l, lv in enumerate(spec.levels)}
    acc = {l: [torch.zeros_like(p) for p in level_params(sites, l)] for l in range(L)}
    count = {l: 0 for l in range(L)}
    slow_steps = {l: 0 for l in range(L)}
    slow_ids = [l for l, lv in enumerate(spec.levels) if not lv.reset_each_task]
    reservoir = Reservoir(spec.replay_bytes, seed + 77) if spec.replay_bytes else None
    stages, trace = [], []
    model.train()
    K.require(reservoir is None or encode is not None, 'replay needs the training encoder')
    for stage, c in enumerate(tasks):
        feats, roles, rows = load(c)
        torch.manual_seed(seed + 100 * stage)
        gen = torch.Generator().manual_seed(seed + 100 * stage)
        for l in fast_init:                                   # Eq. 72: re-initialise at the end of the context
            init_level(sites, l, fast_init[l])
            opts[l] = optimizer(level_params(sites, l), spec.levels[l].lr)
            for a in acc[l]:
                a.zero_()
            count[l] = 0
        fit_log = []
        for i in range(spec.steps):
            j = roles['FIT'][int(torch.randint(len(roles['FIT']), (1,), generator=gen))]
            set_active(sites, range(L))
            for o in opts.values():
                o.zero_grad(set_to_none=True)
            loss = H.fit_loss(model, feats[j:j + 1].cuda().float(), spec.gamma)
            loss.backward()
            lv_ = float(loss.detach())
            K.require(math.isfinite(lv_), f'non-finite loss {c} {i}')
            grads = {l: [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                         for p in level_params(sites, l)] for l in range(L)}
            if reservoir is not None and reservoir.items and i % spec.replay_every == 0:
                set_active(sites, slow_ids)
                for o in opts.values():
                    o.zero_grad(set_to_none=True)
                H.fit_loss(model, encode(reservoir.draw())[None].cuda().float(), spec.gamma).backward()
                for l in slow_ids:
                    if spec.levels[l].replay:
                        for g, p in zip(grads[l], level_params(sites, l)):
                            if p.grad is not None:
                                g.add_(p.grad)
                set_active(sites, range(L))
            for l, lv in enumerate(spec.levels):
                for a, g in zip(acc[l], grads[l]):
                    a.add_(g)
                count[l] += 1
                if count[l] == lv.period:
                    params = level_params(sites, l)
                    for p, a in zip(params, acc[l]):
                        p.grad = a / lv.period
                    torch.nn.utils.clip_grad_norm_(params, max_norm=R.CFG.clip)
                    if lv.reset_each_task:
                        lr = K.lr_at(R.CFG, i) * lv.lr / R.CFG.lr
                    else:
                        lr = lv.lr * min(1., (slow_steps[l] + 1) / spec.slow_warmup)
                    for grp in opts[l].param_groups:
                        grp['lr'] = lr
                    opts[l].step()
                    slow_steps[l] += 1
                    for a in acc[l]:
                        a.zero_()
                    count[l] = 0
            fit_log.append(lv_)
            if budget is not None and (i + 1) % 50 == 0:
                budget.check()
        if reservoir is not None:                               # the task's FIT stream enters the reservoir
            for j in roles['FIT']:
                reservoir.offer(jpeg392(rows[j]['relative_path']), c)
        stages.append(get_state(sites))
        trace.append({'task': c, 'fit_last10': sum(fit_log[-10:]) / 10,
                      'reservoir_items': len(reservoir.items) if reservoir else 0,
                      'reservoir_bytes': reservoir.bytes() if reservoir else 0})
        log('TRAIN', seed, stage + 1, c, 'fit', round(trace[-1]['fit_last10'], 4),
            'res', trace[-1]['reservoir_items'])
        del feats
    ledger = {'adapter_bytes_fp32': state_bytes(stages[-1]), 'adapter_bytes_bf16': state_bytes(stages[-1]) // 2,
              'replay_bytes': reservoir.bytes() if reservoir else 0, 'replay_cap_items': reservoir.cap if reservoir else 0,
              'replay_items_per_task': ({t: reservoir.tags.count(t) for t in tasks} if reservoir else {})}
    del model, sites
    torch.cuda.empty_cache()
    return stages, trace, ledger


def evaluator(spec: Spec, dtype=torch.bfloat16):
    base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
    model, sites = build(spec, base)
    model.eval().to(dtype)
    return model, sites


@torch.no_grad()
def grids(spec: Spec, ev, st, feats, dtype=torch.bfloat16):
    """[2,N,28,28]: pass 0 = persistent levels only, pass 1 = all levels. Detector score = min over passes."""
    model, sites = ev
    set_state(sites, st)
    persistent = [l for l, lv in enumerate(spec.levels) if not lv.reset_each_task]
    out = torch.empty(2, len(feats), 28, 28)
    for p, act in enumerate((persistent, list(range(len(spec.levels))))):
        set_active(sites, act)
        for j in range(0, len(feats), 4):
            en, de = model(feats[j:j + 4].cuda().to(dtype))
            out[p, j:j + len(en[0])] = (1 - F.cosine_similarity(en[0].float(), de[0].float(), dim=1)).clamp_min(0).cpu()
    set_active(sites, range(len(spec.levels)))
    return out
