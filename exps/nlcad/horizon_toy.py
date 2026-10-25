# exps/nlcad/horizon_toy.py
"""NLCAD-03 part A: is the endpoint-displacement signal a usable slow-level hypergradient over a K-step fast window?

Toy = the real Dinomaly decoder class at width 32 (4 heads, 21 tokens), FP64, CPU, on synthetic "categories"; the
real inner loop (hard-mined global-cosine loss with micro-batches, dropout, global clip 0.1, StableAdamW with AMSGrad,
per-tensor update clipping, decoupled weight decay, per-block lr = base * a_b), re-implemented functionally so the
whole window can be differentiated with respect to the block multipliers a. Per window of T steps it compares, for the
outer objective J(theta_T) (dropout off, global cosine):

  exact     dJ(theta_T(a))/da            autograd through the unroll (validated against central finite differences)
  endpoint  <grad J(theta_T), theta_T - theta_0>_b / a_b              the proposed long-horizon proxy
  onestep   <grad J(theta_T), theta_T - theta_{T-1}>_b / a_b          the NLCAD-01 slow-event signal
  online    sum_t <grad J(theta_t), theta_t - theta_{t-1}>_b / a_b    hypergradient-descent style sum (T outer passes)

Two outer objectives: J_q (current + replay queries: the NLCAD-01 normal look-ahead) and J_held (unseen samples of
the old categories: retention outside the buffer). Regimes: "restart" (fresh optimizer, warm-up learning rate, as at a
task start) and "mid@lr" (optimizer state carried from 60 steps on the task, constant lr inside the window).

  python -m exps.nlcad.horizon_toy [--seeds 6] [--fd]
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import run as DR
from exps.nlcad.controller import BLOCKS, block_of

OUT = D.ROOT / "results/hope_cad/nlcad"
DT = torch.float64
EMB, HEADS, TOKENS = 32, 4, 21
R = O.RECIPE
B1, B2 = R["betas"]
HORIZONS = (1, 2, 4, 8, 32)
REGIMES = ("restart", "mid@2e-3", "mid@1e-3", "mid@2e-4")
STAGE_ITERS = 667                                                    # real stage length: warm-up and hm ramp shape


def tiny(seed):
    torch.manual_seed(seed)
    dec = O.Decoder(EMB, HEADS).to(DT)
    O.init_trainable(dec)
    return dec


def categories(seed, n_cat=4, n=48):
    """Category c: tokens on a shared + category-specific low-rank structure; group 2 is a category-specific
    nonlinear map of group 1, so the decoder must learn per-category structure from the fused input."""
    g = torch.Generator().manual_seed(seed)
    shared = torch.randn(EMB, 4, generator=g, dtype=DT) / 2
    out = []
    for _ in range(n_cat):
        U = torch.randn(EMB, 6, generator=g, dtype=DT) / 2
        M = torch.randn(EMB, EMB, generator=g, dtype=DT) / EMB ** 0.5
        z, s = torch.randn(n, TOKENS, 6, generator=g, dtype=DT), torch.randn(n, TOKENS, 4, generator=g, dtype=DT)
        x0 = z @ U.T + s @ shared.T + 0.05 * torch.randn(n, TOKENS, EMB, generator=g, dtype=DT)
        out.append(torch.stack([x0, torch.tanh(x0 @ M)], 1))
    return out


def hm_loss(en, de, p, factor=R["hm_factor"]):
    """Value of O.global_cosine_hm_percent and its easy-point gradient masks (constants of the step). The original
    applies the masks with a tensor hook; a hook would also fire on second-order gradients through de, so the mask
    is applied explicitly in inner_grads instead."""
    loss, masks = 0, []
    for a_, b_ in zip(en, de):
        a_ = a_.detach()
        with torch.no_grad():
            point = 1 - F.cosine_similarity(a_, b_, dim=1).unsqueeze(1)
            thresh = torch.topk(point.reshape(-1), k=int(point.numel() * (1 - p)))[0][-1]
            masks.append(torch.where(point < thresh, torch.tensor(factor, dtype=b_.dtype),
                                     torch.tensor(1.0, dtype=b_.dtype)).expand_as(b_))
        loss = loss + torch.mean(1 - F.cosine_similarity(a_.reshape(len(a_), -1), b_.reshape(len(b_), -1)))
    return loss / len(en), masks


def inner_grads(dec, params, x, p, seed, create_graph):
    """The hm-masked gradient G(theta) = (dde/dtheta)^T (mask * dL/dde), as an explicit, twice-differentiable VJP."""
    torch.manual_seed(seed)                                          # dropout masks: identical for every a
    dec.train()
    grads = None
    for xb in x.split(DR.MICRO):
        en, de = functional_call(dec, params, (xb[:, 0], xb[:, 1]))
        loss, masks = hm_loss(en, de, p)
        loss = loss * len(xb) / len(x)
        gy = torch.autograd.grad(loss, de, create_graph=create_graph)
        g = torch.autograd.grad(de, list(params.values()), grad_outputs=[m * u for m, u in zip(masks, gy)],
                                create_graph=create_graph)
        grads = list(g) if grads is None else [u + v for u, v in zip(grads, g)]
    return grads


def _sqrt(x):
    return x.clamp_min(1e-300).sqrt()                                # finite derivative at 0; value change < 1e-150


def opt_step(params, grads, state, lr_block):
    """clip_grad_norm_(0.1) + O.StableAdamW (AMSGrad, per-tensor rms update clip, decoupled wd), functional."""
    total = _sqrt(sum((g * g).sum() for g in grads))
    coef = torch.clamp(R["clip_grad_norm"] / (total + 1e-6), max=1.0)
    new_p, new_s = {}, {}
    for (name, p), g in zip(params.items(), grads):
        g = g * coef
        lr = lr_block[block_of(name)]
        st = state[name]
        step = st["step"] + 1
        p = p * (1 - lr * R["weight_decay"])
        m = st["m"] * B1 + g * (1 - B1)
        v = st["v"] * B2 + g * g * (1 - B2)
        vmax = torch.maximum(st["vmax"], v)
        denom = _sqrt(vmax) / math.sqrt(1 - B2 ** step) + R["eps"]
        r = g / denom
        scale = torch.clamp(_sqrt((r * r).sum()) / math.sqrt(r.numel()) / 1.0, min=1.0)
        p = p - (lr / (1 - B1 ** step) / scale) * m / denom
        new_p[name], new_s[name] = p, {"step": step, "m": m, "v": v, "vmax": vmax}
    return new_p, new_s


def zero_state(params):
    return {k: {"step": 0, "m": torch.zeros_like(v), "v": torch.zeros_like(v), "vmax": torch.zeros_like(v)}
            for k, v in params.items()}


def detach(params, state):
    return ({k: v.detach().requires_grad_() for k, v in params.items()},
            {k: {"step": s["step"], **{q: s[q].detach() for q in ("m", "v", "vmax")}} for k, s in state.items()})


def unroll(dec, params, state, window, a, create_graph):
    """window: list of (x, base_lr, p, seed). Returns the parameter trajectory theta_0..theta_T."""
    traj = [params]
    for x, lr, p, seed in window:
        grads = inner_grads(dec, params, x, p, seed, create_graph)
        params, state = opt_step(params, grads, state, {b: lr * a[i] for i, b in enumerate(BLOCKS)})
        if not create_graph:
            params, state = detach(params, state)
        traj.append(params)
    return traj, state


def outer(dec, params, q):
    dec.eval()
    en, de = functional_call(dec, params, (q[:, 0], q[:, 1]))
    dec.train()
    return O.global_cosine(en, de)


def grad_at(dec, params, q):
    leaf = {k: v.detach().requires_grad_() for k, v in params.items()}
    return dict(zip(leaf, torch.autograd.grad(outer(dec, leaf, q), list(leaf.values()))))


def blockwise(g, d, a):
    s = {b: 0.0 for b in BLOCKS}
    for k in g:
        s[block_of(k)] += float((g[k] * d[k]).sum())
    return np.array([s[b] / float(a[i]) for i, b in enumerate(BLOCKS)])


def setup(seed):
    """Pre-train on three old categories (real optimizer), fix the buffer, queries and the current-task sampler."""
    cats = categories(seed)
    old, cur = cats[:3], cats[3]
    dec = tiny(seed)
    pool = torch.cat([c[:32] for c in old])
    opt = O.make_optimizer(dec)
    sch = O.make_scheduler(opt, 200)
    gen = torch.Generator().manual_seed(seed + 1)
    batches = DR.Batches(len(pool), 16, gen)
    torch.manual_seed(seed + 2)
    for i in range(200):
        opt.zero_grad(set_to_none=True)
        x = pool[batches.next()]
        for xb in x.split(DR.MICRO):
            en, de = dec(xb[:, 0], xb[:, 1])
            (O.global_cosine_hm_percent(en, de, p=min(0.9 * i / 20, 0.9), factor=R["hm_factor"]) * len(xb) / len(x)).backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=R["clip_grad_norm"])
        opt.step()
        sch.step()
    buffer = torch.cat([c[:4] for c in old])
    return {"dec": dec, "params": {k: v.detach().clone() for k, v in dec.named_parameters()}, "cur": cur[:32],
            "buffer": buffer, "q": torch.cat([cur[:8], buffer[[0, 1, 4, 5, 8, 9, 2, 6]]]),
            "held": torch.cat([c[32:37] for c in old] + [old[0][37:38]]), "gen": torch.Generator().manual_seed(seed + 3)}


def batches(env, n, seed0):
    cur = DR.Batches(len(env["cur"]), 8, env["gen"])
    rep = DR.Batches(len(env["buffer"]), 8, env["gen"])
    out = []
    for t in range(n):
        x = torch.cat([env["cur"][cur.next()], env["buffer"][rep.next()]])
        out.append(x[torch.randperm(len(x), generator=env["gen"])])
    return [(x, seed0 + t) for t, x in enumerate(out)]


def window_start(env, regime, T, seed):
    """(theta_0, state_0, window) for a regime. restart: fresh optimizer, real warm-up/hm ramp from stage step 0.
    mid@lr: 60 real steps on the task (warm-up 20, then the recipe lr), then T steps at constant lr."""
    sched = O.WarmCosineScheduler(torch.optim.SGD([torch.zeros(1)], lr=1.0), R["lr"], R["final_lr"], STAGE_ITERS,
                                  R["warmup_iters"]).schedule
    ramp = round(0.1 * STAGE_ITERS)
    params = {k: v.clone().requires_grad_() for k, v in env["params"].items()}
    state = zero_state(params)
    if regime == "restart":
        data = batches(env, T, seed * 1000)
        return params, state, [(x, float(sched[t]), min(0.9 * t / ramp, 0.9), s) for t, (x, s) in enumerate(data)]
    lr = float(regime.split("@")[1])
    warm = batches(env, 60, seed * 1000)
    pre = [(x, R["lr"] * min(1.0, (t + 1) / 20), min(0.9 * t / 20, 0.9), s) for t, (x, s) in enumerate(warm)]
    traj, state = unroll(env["dec"], params, state, pre, torch.ones(len(BLOCKS), dtype=DT), create_graph=False)
    data = batches(env, T, seed * 1000 + 500)
    return traj[-1], state, [(x, lr, 0.9, s) for x, s in data]


def evaluate(env, regime, T, seed, a, fd=False, eps=1e-6):
    dec = env["dec"]
    p0, s0, win = window_start(env, regime, T, seed)
    p0, s0 = detach(p0, s0)
    av = a.clone().requires_grad_()
    traj, _ = unroll(dec, p0, s0, win, av, create_graph=True)
    res = {}
    for name, q in (("J_q", env["q"]), ("J_held", env["held"])):
        J = outer(dec, traj[-1], q)
        exact = torch.autograd.grad(J, av, retain_graph=True)[0].detach().numpy()
        end = {k: v.detach() for k, v in traj[-1].items()}
        gT = grad_at(dec, end, q)
        disp = {k: end[k] - p0[k].detach() for k in end}
        last = {k: end[k] - traj[-2][k].detach() for k in end}
        online = sum(blockwise(grad_at(dec, {k: v.detach() for k, v in traj[t].items()}, q),
                               {k: traj[t][k].detach() - traj[t - 1][k].detach() for k in end}, a)
                     for t in range(1, len(traj)))
        res[name] = {"J": float(J), "exact": exact, "endpoint": blockwise(gT, disp, a), "onestep": blockwise(gT, last, a),
                     "online": online}
    del traj
    if fd:
        for name, q in (("J_q", env["q"]), ("J_held", env["held"])):
            fdv = []
            for i in range(len(BLOCKS)):
                vals = []
                for sgn in (1, -1):
                    ap = a.clone()
                    ap[i] += sgn * eps
                    tr, _ = unroll(dec, p0, s0, win, ap, create_graph=False)
                    with torch.no_grad():
                        vals.append(float(outer(dec, tr[-1], q)))
                fdv.append((vals[0] - vals[1]) / (2 * eps))
            res[name]["fd"] = np.array(fdv)
    return res


def evaluate_chunked(env, regime, T, seed, a, chunk):
    """Design check: slow objective S = sum_w J(theta at the end of window w), windows of `chunk` steps inside a slow
    period of T steps. exact = dS/da through the whole unroll; tbptt = sum_w endpoint proxy of window w (each window's
    start treated as fixed: truncated credit assignment, one anchor and one outer pass per window); endpoint_T =
    endpoint proxy of the whole period applied to S's last term only (what a single long window would use)."""
    dec = env["dec"]
    p0, s0, win = window_start(env, regime, T, seed)
    p0, s0 = detach(p0, s0)
    av = a.clone().requires_grad_()
    traj, _ = unroll(dec, p0, s0, win, av, create_graph=True)
    ends = list(range(chunk, T + 1, chunk))
    res = {}
    for name, q in (("J_q", env["q"]), ("J_held", env["held"])):
        S = sum(outer(dec, traj[e], q) for e in ends)
        exact = torch.autograd.grad(S, av, retain_graph=True)[0].detach().numpy()
        det = [{k: v.detach() for k, v in x.items()} for x in traj]
        tbptt = sum(blockwise(grad_at(dec, det[e], q), {k: det[e][k] - det[e - chunk][k] for k in det[e]}, a)
                    for e in ends)
        whole = sum(blockwise(grad_at(dec, det[e], q), {k: det[e][k] - det[0][k] for k in det[e]}, a) for e in ends)
        res[name] = {"exact": exact, "tbptt": tbptt, "endpoint_from_start": whole}
    return res


def metrics(exact, proxy, a):
    c = lambda u: u - u.mean()
    cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300))
    ge, gp = float(a @ exact), float(a @ proxy)                       # derivative wrt a common log-scale
    return {"cos": cos(proxy, exact), "cos_centred": cos(c(proxy), c(exact)),
            "sign_agree_blocks": float(np.mean(np.sign(proxy) == np.sign(exact))),
            "sign_agree_centred": float(np.mean(np.sign(c(proxy)) == np.sign(c(exact)))),
            "global_exact": ge, "global_proxy": gp, "global_sign_agree": bool(np.sign(ge) == np.sign(gp)),
            "slope": float(proxy @ exact / (exact @ exact + 1e-300))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--fd", action="store_true", help="finite-difference validation subset only")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--profile", choices=("ones", "n1"), default="ones")
    ap.add_argument("--chunk", type=int, default=0, help="design check: truncated windows of this length")
    ap.add_argument("--chunk-T", type=int, default=32)
    a_ = ap.parse_args()
    torch.set_num_threads(a_.threads)
    a = torch.ones(len(BLOCKS), dtype=DT)
    if a_.profile == "n1":                                            # NLCAD-02 open-loop lambda=1e-5 end profile
        a = torch.tensor([0.805, 0.794, 0.989, 1.096, 1.132, 0.948, 1.003, 1.077, 1.156], dtype=DT)
    rows = []
    if a_.chunk:
        envs = {}
        for seed in range(a_.seeds):
            env = envs.setdefault(seed, setup(seed))
            for regime in REGIMES:
                res = evaluate_chunked(env, regime, a_.chunk_T, seed, a, a_.chunk)
                for obj, r in res.items():
                    row = {"seed": seed, "regime": regime, "T": a_.chunk_T, "chunk": a_.chunk, "objective": obj,
                           **{k: r[k].tolist() for k in r}}
                    for k in ("tbptt", "endpoint_from_start"):
                        row[f"m_{k}"] = metrics(r["exact"], r[k], a.numpy())
                    rows.append(row)
                    print(seed, regime, obj, {k: (round(row[f"m_{k}"]["cos_centred"], 3), row[f"m_{k}"]["global_sign_agree"])
                                              for k in ("tbptt", "endpoint_from_start")}, flush=True)
        D.write_json(OUT / f"HORIZON_TOY_CHUNK{a_.chunk}_T{a_.chunk_T}.json", {"a": a.tolist(), "rows": rows})
        return
    grid = ([(s, "mid@2e-3", T) for s in (0, 1) for T in (1, 8, 32)] if a_.fd else
            [(s, r, T) for s in range(a_.seeds) for r in REGIMES for T in HORIZONS])
    envs = {}
    for seed, regime, T in grid:
        env = envs.setdefault(seed, setup(seed))
        res = evaluate(env, regime, T, seed, a, fd=a_.fd)
        for obj, r in res.items():
            row = {"seed": seed, "regime": regime, "T": T, "objective": obj, "J": r["J"],
                   **{k: r[k].tolist() for k in ("exact", "endpoint", "onestep", "online") + (("fd",) if a_.fd else ())}}
            for k in ("endpoint", "onestep", "online"):
                row[f"m_{k}"] = metrics(r["exact"], r[k], a.numpy())
            if a_.fd:
                row["fd_vs_exact_rel"] = float(np.linalg.norm(r["fd"] - r["exact"]) / np.linalg.norm(r["exact"]))
            rows.append(row)
            print(seed, regime, T, obj, "fd_rel" if a_.fd else "", round(row.get("fd_vs_exact_rel", float("nan")), 6),
                  {k: (round(row[f"m_{k}"]["cos_centred"], 3), row[f"m_{k}"]["global_sign_agree"]) for k in ("endpoint", "onestep", "online")},
                  "global_exact", f"{row['m_endpoint']['global_exact']:+.2e}", flush=True)
    tag = "FD" if a_.fd else f"GRID_{a_.profile}"
    D.write_json(OUT / f"HORIZON_TOY_{tag}.json", {"grid": [list(g) for g in grid], "a": a.tolist(), "rows": rows})


if __name__ == "__main__":
    main()
