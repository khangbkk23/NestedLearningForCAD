# exps/nlcad/fast.py
"""NL-CAD fast level: the Dinomaly decoder trained by its own optimizer on current normals + Replay-24, with the
slow controller's blockwise multipliers applied to each optimizer update and a one-step look-ahead slow event every
K fast steps.

With ctrl=None (or the identity controller) and outer=None this loop is, op for op, exps.rh01.run.train_stage_guarded
(same sampler and generator consumption, loss, micro-batching, clipping and skip guard). The slow event runs the outer
objective with dropout disabled inside a forked RNG and draws query identities from its own generator, so a
non-learning controller (learn=False) leaves the fast trajectory unchanged: the compute-matched fixed-profile control.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from exps.dinomaly import official as O
from exps.dinomaly import run as DR
from exps.dinomaly.common import require
from exps.nlcad.controller import BLOCKS, block_of

MAX_SKIPS, MAX_CONSECUTIVE = 10, 3


def _rng_devices(device):
    return [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []


def slow_event(dec, named, old, a, outer, ctx, ctrl, learn, device):
    """Look-ahead hypergradient at theta+ (the decoder after the fast step just taken)."""
    params = [p for _, p in named]
    blocks = [block_of(n) for n, _ in named]
    drops = [m for m in dec.modules() if isinstance(m, torch.nn.Dropout)]
    saved = [m.p for m in drops]
    for m in drops:
        m.p = 0.0
    for p in params:
        p.grad = None
    try:
        with torch.random.fork_rng(devices=_rng_devices(device)):
            out = outer(dec, ctx)
    finally:
        for m, v in zip(drops, saved):
            m.p = v
    s = {b: 0.0 for b in BLOCKS}
    gq = {b: 0.0 for b in BLOCKS}
    dn = {b: 0.0 for b in BLOCKS}
    for p, o, b in zip(params, old, blocks):
        d = p.detach() - o.to(p.device)                        # Delta_b, streamed per tensor (theta_old kept on CPU)
        dn[b] += float(d.double().pow(2).sum())
        if p.grad is not None:
            s[b] += float((p.grad.double() * d.double()).sum())
            gq[b] += float(p.grad.double().pow(2).sum())
        p.grad = None
    s = {b: v / a[b] for b, v in s.items()}
    ev = {"iter": ctx["iter"], "outer": out["loss"], "dJ_da": s,
          "query_grad_norm": {b: math.sqrt(v) for b, v in gq.items()},
          "update_norm": {b: math.sqrt(v) for b, v in dn.items()}, "a_used": dict(a),
          **{k: v for k, v in out.items() if k != "loss"}}
    finite = math.isfinite(out["loss"]) and all(math.isfinite(v) for v in s.values())
    ev["finite"] = finite
    if ctrl is not None and learn and finite:
        ev.update(ctrl.slow_step(s))
    elif ctrl is not None:
        ev["hypergrad"] = ctrl.grad_phi(s).tolist() if finite else None
        ctrl.opt.zero_grad(set_to_none=True)
    return ev


def block_optimizer(dec):
    """O.make_optimizer with one parameter group per controlled block, in the same parameter order."""
    r = O.RECIPE
    groups = {b: [] for b in BLOCKS}
    for n, q in dec.named_parameters():
        groups[block_of(n)].append(q)
    return O.StableAdamW([{"params": groups[b], "block": b} for b in BLOCKS], lr=r["lr"], betas=r["betas"],
                         weight_decay=r["weight_decay"], amsgrad=r["amsgrad"], eps=r["eps"])


def nl_train_stage(dec, current, iters, gen, replay=None, ctrl=None, outer=None, K=32, learn=True,
                   device="cuda", autocast=True, lr_at=None):
    """lr_at: optional callable(i) -> learning rate for stage iteration i (no-warm-restart control: one schedule
    spanning the stream, supplied by the runner). None keeps the recipe's per-task WarmCosine (warm restart)."""
    opt = block_optimizer(dec)
    sch = O.make_scheduler(opt, iters) if lr_at is None else None
    ramp = round(0.1 * iters)
    batch = O.RECIPE["batch_size"]
    cur = DR.Batches(len(current), batch // 2 if replay is not None else batch, gen)
    rep = DR.Batches(len(replay), batch // 2, gen) if replay is not None else None
    dec.train()
    named = list(dec.named_parameters())
    params = [p for _, p in named]
    blocks = [block_of(n) for n, _ in named]
    dtype = torch.device(device).type
    losses, skips, events, run_ = [], [], [], 0
    for i in range(iters):
        p = min(O.RECIPE["hm_p_final"] * i / ramp, O.RECIPE["hm_p_final"])
        ci = cur.next()
        x = current[ci]
        ri = None
        if rep is not None:
            ri = rep.next()
            x = torch.cat([x, replay[ri]])
            x = x[torch.randperm(len(x), generator=gen)]
        x = x.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        total = 0.
        for xb in x.split(DR.MICRO):
            with torch.autocast(dtype, dtype=torch.bfloat16, enabled=autocast):
                en, de = dec(xb[:, 0], xb[:, 1])
            loss = O.global_cosine_hm_percent([e.float() for e in en], [d.float() for d in de],
                                              p=p, factor=O.RECIPE["hm_factor"]) * len(xb) / len(x)
            loss.backward()
            total += float(loss)
        norm = torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=O.RECIPE["clip_grad_norm"])
        if not (np.isfinite(total) and bool(torch.isfinite(norm))):
            opt.zero_grad(set_to_none=True)
            skips.append({"iteration": i, "loss": total, "grad_norm": float(norm)})
            run_ += 1
            require(len(skips) <= MAX_SKIPS and run_ < MAX_CONSECUTIVE, f"non-finite steps exceed limits: {skips}")
            if sch is not None:
                sch.step()
            losses.append(float("nan"))
            continue
        run_ = 0
        a = ctrl.a() if ctrl is not None else {b: 1.0 for b in BLOCKS}
        base = lr_at(i) if lr_at is not None else sch.get_last_lr()[0]
        for g in opt.param_groups:                              # Delta_b = a_b * Delta_b(a=1): StableAdamW is linear in lr
            g["lr"] = base * a[g["block"]]
        slow = outer is not None and i % K == K - 1
        old = [q.detach().to("cpu", copy=True) for q in params] if slow else None
        opt.step()
        if sch is not None:
            sch.step()
        losses.append(total)
        if slow:
            events.append(slow_event(dec, named, old, a, outer, {"iter": i, "ci": ci, "ri": ri, "p": p},
                                     ctrl, learn, device))
        del old
    return losses, skips, events
