# exps/nlcad/test_horizon.py
"""NLCAD-03 toy validity: the functional inner loop is the real one, and the exact window hypergradient is exact."""
from __future__ import annotations

import numpy as np
import torch

from exps.dinomaly import official as O
from exps.nlcad import horizon_toy as H
from exps.nlcad.controller import BLOCKS


def _env():
    return H.setup(0)


def test_functional_step_equals_stableadamw_with_clipping():
    env = _env()
    dec = H.tiny(5)
    ref = O.make_optimizer(dec)
    params = {k: v.detach().clone().requires_grad_() for k, v in dec.named_parameters()}
    state = H.zero_state(params)
    a = torch.linspace(0.7, 1.3, len(BLOCKS), dtype=H.DT)
    for g in ref.param_groups:
        g["lr"] = 1e-3
    groups = {}
    for n, q in dec.named_parameters():
        groups.setdefault(H.block_of(n), []).append(q)
    ref = O.StableAdamW([{"params": groups[b], "block": b} for b in BLOCKS], lr=1e-3, betas=O.RECIPE["betas"],
                        weight_decay=O.RECIPE["weight_decay"], amsgrad=True, eps=O.RECIPE["eps"])
    for g in ref.param_groups:
        g["lr"] = 1e-3 * float(a[BLOCKS.index(g["block"])])
    for t, (x, seed) in enumerate(H.batches(env, 5, 77)):
        grads = H.inner_grads(dec, params, x, 0.9, seed, create_graph=False)
        params, state = H.opt_step(params, grads, state, {b: 1e-3 * a[i] for i, b in enumerate(BLOCKS)})
        params, state = H.detach(params, state)
        ref.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        dec.train()
        for xb in x.split(8):
            en, de = dec(xb[:, 0], xb[:, 1])
            (O.global_cosine_hm_percent(en, de, p=0.9, factor=O.RECIPE["hm_factor"]) * len(xb) / len(x)).backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=O.RECIPE["clip_grad_norm"])
        ref.step()
        worst = max(float((params[n] - q).abs().max()) for n, q in dec.named_parameters())
        assert worst < 1e-13, (t, worst)


def test_one_step_window_all_signals_equal_exact():
    env = _env()
    r = H.evaluate(env, "mid@2e-3", 1, 0, torch.ones(len(BLOCKS), dtype=H.DT))
    for obj in ("J_q", "J_held"):
        e = r[obj]["exact"]
        for k in ("endpoint", "onestep", "online"):
            assert np.allclose(r[obj][k], e, rtol=1e-6, atol=1e-14), (obj, k)


def test_window_autograd_matches_finite_differences():
    env = _env()
    r = H.evaluate(env, "mid@2e-3", 4, 1, torch.ones(len(BLOCKS), dtype=H.DT), fd=True)
    for obj in ("J_q", "J_held"):
        e, f = r[obj]["exact"], r[obj]["fd"]
        assert np.linalg.norm(e - f) / np.linalg.norm(e) < 1e-4, obj
