# exps/nlcad/objectives_ret.py
"""NLCAD-05B retention-specific slow objective J_ret (new file; earlier objective files stay hash-locked).

Old-category, category-balanced, cross-image pairwise ranking on synthetic anomalies (Q3 trains, Q2 is a no-grad
diagnostic on the same host identities), evaluated at theta+ by the slow event of exps.nlcad.fast:

    J_ret = mean_{c in C} mean_{h in H_c} mean_{p in P_h, n in N_c} softplus((s_n - s_p) / tau)
    N_c   = union_{h' in H_c} top-m_c background patches of host h',  m_c = ceil(n_neg / |H_c|)

C: old categories (completed tasks) with at least one usable host; H_c: Replay-24 members of c that are not in the
fast step's inner batch and have a synthetic view with a positive patch (one identity = one host, whatever its number
of synthetic variants; one view per host per event, drawn from the objective's own generator); at most max_hosts per
category. P_h: positive patches (coverage >= 0.5) of host h's view. With |H_c| >= 2 every positive meets negatives of
the other hosts (cross-image pairs); a single-host category contributes its within-image pairs only. The per-host
quota m_c keeps one host with a heavy background tail from filling N_c, and the category mean keeps one category from
dominating. No current-task term: acquisition is the fast level's job.

Exact gradient in two passes, so that categories coupled across micro-batches need not share one graph:
  1. no-grad forward of every selected view -> maps S; J(S) and G = dJ/dS on detached leaves (hard negatives selected
     here);
  2. per micro-batch: forward with grad, backward of <maps, G>. With a deterministic forward (dropout is off in the
     slow event) this is dJ/dtheta; the largest |maps_pass2 - S| is returned and checked by the caller.
Synthetic masks only; never a real defect image or label.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from exps.dinomaly import readout as R
from exps.nlcad.objectives import _forward


def category_loss(maps, covs, tau, n_neg):
    """maps, covs: lists of [28, 28] (one per host of one category). Returns (loss, n_pairs, cross_pairs)."""
    pos = [s[c >= 0.5] for s, c in zip(maps, covs)]
    bg = [s[c == 0] for s, c in zip(maps, covs)]
    m = math.ceil(n_neg / len(maps))
    hard = [b.topk(min(m, b.numel())).values for b in bg]
    owner = torch.cat([torch.full((len(h),), j, device=h.device) for j, h in enumerate(hard)])
    neg = torch.cat(hard)
    terms, pairs, cross = [], 0, 0
    for j, p in enumerate(pos):
        terms.append(F.softplus((neg[None, :] - p[:, None]) / tau).mean())
        pairs += p.numel() * neg.numel()
        cross += p.numel() * int((owner != j).sum())
    return torch.stack(terms).mean(), pairs, cross


def retention_loss(maps_by_cat, covs_by_cat, tau=0.05, n_neg=64):
    """Category-balanced J_ret over {category: [maps]} (all categories given have >= 1 host)."""
    per, pairs, cross = {}, 0, 0
    for c in maps_by_cat:
        loss, n, x = category_loss(maps_by_cat[c], covs_by_cat[c], tau, n_neg)
        per[c], pairs, cross = loss, pairs + n, cross + x
    return torch.stack(list(per.values())).mean(), per, pairs, cross


class RetentionRanking:
    """bank: {generator: [{"host": (category, index), "variant": v, "features": [2, 789, 768], "coverage": [28, 28]}]}.
    allowed: bank.allowed(...) for the stage; replay_keys[j]: host key of row j of the fast step's replay tensor."""

    def __init__(self, bank, train_gen, diag_gen, allowed, current_cat, replay_keys, seed=0, tau=0.05, n_neg=64,
                 max_hosts=8, micro=8, device="cuda", autocast=True, check_tol=1e-3):
        self.train_gen, self.diag_gen = train_gen, diag_gen
        self.replay_keys = list(replay_keys)
        self.old = sorted({k[0] for k in self.replay_keys})
        require_old = all(c != current_cat for c in self.old)
        if not require_old:
            raise ValueError("current category inside the replay buffer")
        self.views = {g: {} for g in (train_gen, diag_gen)}     # {gen: {host: [views with a positive patch]}}
        for g in (train_gen, diag_gen):
            for it in bank[g]:
                h = tuple(it["host"])
                if h in allowed and h[0] != current_cat and bool((it["coverage"] >= 0.5).any()):
                    self.views[g].setdefault(h, []).append(it)
        for g in self.views:
            for h in self.views[g]:
                self.views[g][h].sort(key=lambda it: it["variant"])
        self.gen = torch.Generator().manual_seed(seed)
        self.tau, self.n_neg, self.max_hosts, self.micro = tau, n_neg, max_hosts, micro
        self.device, self.autocast, self.check_tol = device, autocast, check_tol

    def hosts(self, ctx):
        """{category: [usable host keys]} after removing the inner batch's replay identities (deterministic order)."""
        inner = {self.replay_keys[int(j)] for j in (ctx.get("ri") or [])}
        out = {}
        for c in self.old:
            hs = [k for k in self.replay_keys if k[0] == c and k not in inner and k in self.views[self.train_gen]]
            if len(hs) > self.max_hosts:
                hs = [hs[j] for j in sorted(torch.randperm(len(hs), generator=self.gen)[:self.max_hosts].tolist())]
            if hs:
                out[c] = hs
        return out

    def select(self, hosts, gen_name):
        """One view per host: a uniform draw among the host's views with a positive patch (own generator)."""
        out = {}
        for c, hs in hosts.items():
            items = []
            for h in hs:
                vs = self.views[gen_name].get(h)
                if vs:
                    items.append(vs[int(torch.randint(len(vs), (1,), generator=self.gen))])
            if items:
                out[c] = items
        return out

    def _maps(self, dec, items, grad):
        x = torch.stack([it["features"] for it in items]).to(self.device)
        with torch.set_grad_enabled(grad):
            en, de = _forward(dec, x, self.device, self.autocast)
            return R.grid(en, de)

    def loss(self, dec, sel, grad):
        """Two-pass exact J_ret; returns (loss, diagnostics). grad=False: pass 1 only."""
        flat = [(c, it) for c in sel for it in sel[c]]
        if not flat:
            return float("nan"), {"categories": 0}
        with torch.no_grad():
            maps = torch.cat([self._maps(dec, [it for _, it in flat[k:k + self.micro]], False)
                              for k in range(0, len(flat), self.micro)])
        leaf = maps.clone().requires_grad_(grad)
        by_cat, cov_by_cat, j = {}, {}, 0
        for c in sel:
            n = len(sel[c])
            by_cat[c] = list(leaf[j:j + n])
            cov_by_cat[c] = [it["coverage"].to(leaf.device) for it in sel[c]]
            j += n
        with torch.set_grad_enabled(grad):
            J, per, pairs, cross = retention_loss(by_cat, cov_by_cat, self.tau, self.n_neg)
        info = {"categories": len(sel), "hosts": sum(len(v) for v in sel.values()),
                "per_category": {c: float(v.detach()) for c, v in per.items()},
                "multi_host_categories": sum(len(v) >= 2 for v in sel.values()),
                "pairs": pairs, "cross_host_pair_fraction": cross / pairs if pairs else 0.0,
                "host_keys": {c: [list(it["host"]) + [it["variant"]] for it in v] for c, v in sel.items()}}
        if grad:
            J.backward()
            G, err = leaf.grad.detach(), 0.0
            for k in range(0, len(flat), self.micro):
                m = self._maps(dec, [it for _, it in flat[k:k + self.micro]], True)
                err = max(err, float((m.detach() - maps[k:k + self.micro]).abs().max()))
                (m * G[k:k + self.micro]).sum().backward()
            info["map_reproduction_max_abs"] = err
            if err > self.check_tol:
                raise RuntimeError(f"two-pass maps differ by {err:.3e} (> {self.check_tol}); gradient would be inexact")
        return float(J.detach()), info

    def __call__(self, dec, ctx):
        hs = self.hosts(ctx)
        loss, info = self.loss(dec, self.select(hs, self.train_gen), grad=True)
        dloss, dinfo = self.loss(dec, self.select(hs, self.diag_gen), grad=False)
        return {"loss": loss, "used": info.get("hosts", 0), "train": info, "diag_loss": dloss,
                "diag": {k: v for k, v in dinfo.items() if k != "host_keys"}}

    def evaluate(self, dec, ctx, name):
        """Diagnostic only (no gradient) at the current state, generator `name`."""
        return self.loss(dec, self.select(self.hosts(ctx), name), grad=False)[0]
