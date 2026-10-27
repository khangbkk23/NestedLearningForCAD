# exps/nlcad/objectives_s.py
"""NLCAD-04 Variant S outer objective (new file; exps/nlcad/objectives.py stays byte-identical for the NLCAD-01/02
locks). Same pairwise synthetic-anomaly ranking loss as objectives.SyntheticRanking, with the two audit fixes:

  1. Host separation: synthetic hosts are disjoint from the fast step's inner batch (current indices ctx["ci"] and the
     buffer members behind ctx["ri"]); a synthetic item is a view of its host, so an inner host would let the
     look-ahead credit reward fitting that very image.
  2. Stratified sampling: n_current items whose host is a current-task host and n_replay items whose host is a buffer
     member of a completed task (default 8 + 8, two micro-batches: the query count and composition of Variant N).

Hosts are only ever drawn from `allowed` (bank.allowed: current-task hosts + the stage's Replay-24 members).
Synthetic masks only; no real defect image or label.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from exps.dinomaly import readout as R
from exps.nlcad.objectives import _forward, ranking_loss


def ranking_terms(s, c, tau):
    """Deterministic all-negatives ranking loss and score statistics of one [28, 28] map (evaluation only)."""
    pos, neg = s[c >= 0.5], s[c == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    return {"loss": F.softplus((neg[None, :] - pos[:, None]) / tau).mean(), "pos_mean": pos.mean(),
            "neg_mean": neg.mean(), "n_pos": pos.numel(), "n_neg": neg.numel()}


class SyntheticRankingS:
    """bank: {generator: [{"host": (category, index), "features": [2, 789, 768], "coverage": [28, 28]}]}.
    current_cat: the current task; replay_keys[j]: host key of row j of the fast step's replay tensor."""

    def __init__(self, bank, train_gen, diag_gen, allowed, current_cat, replay_keys, seed=0, n_current=8, n_replay=8,
                 micro=8, tau=0.05, n_neg=64, device="cuda", autocast=True):
        self.train_gen, self.diag_gen = train_gen, diag_gen
        self.current_cat, self.replay_keys = current_cat, list(replay_keys or [])
        self.pools = {g: {"current": [it for it in bank[g] if it["host"] in allowed and it["host"][0] == current_cat],
                          "replay": [it for it in bank[g] if it["host"] in allowed and it["host"][0] != current_cat]}
                      for g in (train_gen, diag_gen)}
        self.gen = torch.Generator().manual_seed(seed)
        self.n = {"current": n_current, "replay": n_replay}
        self.micro, self.tau, self.n_neg, self.device, self.autocast = micro, tau, n_neg, device, autocast

    def inner_hosts(self, ctx):
        keys = {(self.current_cat, int(j)) for j in (ctx.get("ci") or [])}
        return keys | {self.replay_keys[int(j)] for j in (ctx.get("ri") or [])}

    def sample(self, name, ctx):
        """{source: items}: n per source, uniformly without replacement among allowed items whose host is not inner."""
        inner = self.inner_hosts(ctx)
        out = {}
        for src, k in self.n.items():
            pool = [it for it in self.pools[name][src] if it["host"] not in inner]
            order = torch.randperm(len(pool), generator=self.gen).tolist()[:k]
            out[src] = [pool[j] for j in order]
        return out

    def _loss(self, dec, groups, grad):
        """Per-source mean ranking loss; the objective is the item-weighted mean over sources (one backward per
        micro-batch). Items without a positive patch are skipped and counted."""
        n_items = sum(len(v) for v in groups.values())
        total, info = 0., {}
        for src, items in groups.items():
            src_loss, used, skipped, pos_m, neg_m = 0., 0, 0, 0., 0.
            for k in range(0, len(items), self.micro):
                chunk = items[k:k + self.micro]
                x = torch.stack([it["features"] for it in chunk]).to(self.device)
                cov = torch.stack([it["coverage"] for it in chunk]).to(self.device)
                keep = (cov >= 0.5).flatten(1).any(1)
                skipped += int((~keep).sum())
                with torch.set_grad_enabled(grad):
                    en, de = _forward(dec, x, self.device, self.autocast)
                    s = R.grid(en, de)
                    loss = ranking_loss(s, cov, self.tau, self.n_neg, gen=self.gen)
                    if loss is None:
                        continue
                    w = len(chunk) / n_items
                    if grad:
                        (loss * w).backward()
                with torch.no_grad():
                    pos_m += float(s[cov >= 0.5].mean()) * int(keep.sum())
                    neg_m += float(s[cov == 0].mean()) * int(keep.sum())
                total += float(loss.detach()) * w
                src_loss += float(loss.detach()) * len(chunk)
                used += int(keep.sum())
            info[src] = {"loss": src_loss / len(items) if items else None, "used": used, "skipped": skipped,
                         "pos_mean": pos_m / used if used else None, "neg_mean": neg_m / used if used else None}
        return total, info

    def __call__(self, dec, ctx):
        groups = self.sample(self.train_gen, ctx)
        loss, info = self._loss(dec, groups, grad=True)
        used = sum(v["used"] for v in info.values())
        diag, dinfo = self._loss(dec, self.sample(self.diag_gen, ctx), grad=False)
        return {"loss": loss if used else float("nan"), "used": used, "train_parts": info,
                "cross_generator_loss": diag, "cross_generator_parts": dinfo,
                "hosts": {src: [list(it["host"]) for it in items] for src, items in groups.items()}}

    def evaluate(self, dec, ctx, name):
        """Diagnostic only (no gradient): generator `name` on a fresh stratified sample at the current state."""
        loss, info = self._loss(dec, self.sample(name, ctx), grad=False)
        return loss if sum(v["used"] for v in info.values()) else float("nan")
