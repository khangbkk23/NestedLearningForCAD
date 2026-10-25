# exps/nlcad/objectives.py
"""Outer objectives for the NL-CAD slow level, evaluated at theta+ (after the fast step). Each call performs its own
backward (micro-batched, same autocast as training) so that parameter .grad holds grad J(theta+), and returns
{"loss": float, ...diagnostics}.

Variant N (NormalLookahead): Dinomaly's official global-cosine reconstruction loss on query identities that are
  disjoint from the fast step's inner batch, drawn only from the current task and the current Replay-24 buffer.
  No augmentation; a query identity is a different image, never a view of an inner image.
Variant S (SyntheticRanking): pairwise logistic ranking loss on synthetic anomalies (PTAG Q2 or Q3) generated from
  allowed normal hosts (current task or current buffer); synthetic masks only, never real defect labels. The other
  generator is evaluated without gradient as the cross-generator diagnostic.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from exps.dinomaly import official as O
from exps.dinomaly import readout as R


def _forward(dec, xb, device, autocast):
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=autocast):
        en, de = dec(xb[:, 0], xb[:, 1])
    return [e.float() for e in en], [d.float() for d in de]


class NormalLookahead:
    def __init__(self, current, replay=None, seed=0, n_current=8, n_replay=8, micro=8, device="cuda", autocast=True):
        self.current, self.replay = current, replay
        self.gen = torch.Generator().manual_seed(seed)
        self.n_current = n_current if replay is not None else n_current + n_replay
        self.n_replay = n_replay if replay is not None else 0
        self.micro, self.device, self.autocast = micro, device, autocast

    def _pick(self, n, exclude, k):
        pool = [j for j in range(n) if j not in set(exclude)]
        order = torch.randperm(len(pool), generator=self.gen).tolist()
        return [pool[j] for j in order[:k]]

    def query(self, ctx):
        qc = self._pick(len(self.current), ctx["ci"], self.n_current)
        qr = self._pick(len(self.replay), ctx["ri"] or [], self.n_replay) if self.replay is not None else []
        return qc, qr

    def __call__(self, dec, ctx):
        qc, qr = self.query(ctx)
        x = self.current[qc]
        if qr:
            x = torch.cat([x, self.replay[qr]])
        x = x.to(self.device)
        total, parts = 0., {"current": 0., "replay": 0.}
        for k, xb in enumerate(x.split(self.micro)):
            en, de = _forward(dec, xb, self.device, self.autocast)
            loss = O.global_cosine(en, de) * len(xb) / len(x)
            loss.backward()
            total += float(loss)
        with torch.no_grad():                                   # per-source diagnostic (no extra backward)
            for name, idx, src in (("current", qc, self.current), ("replay", qr, self.replay)):
                if idx:
                    en, de = _forward(dec, src[idx[:self.micro]].to(self.device), self.device, self.autocast)
                    parts[name] = float(O.global_cosine(en, de))
        return {"loss": total, "query_current": qc, "query_replay": qr, "loss_current": parts["current"],
                "loss_replay": parts["replay"]}

    def evaluate(self, dec, ctx):
        """Diagnostic only: query loss at the current state without backward (own generator; no .grad)."""
        qc, qr = self.query(ctx)
        x = self.current[qc]
        if qr:
            x = torch.cat([x, self.replay[qr]])
        x, total = x.to(self.device), 0.
        with torch.no_grad():
            for xb in x.split(self.micro):
                en, de = _forward(dec, xb, self.device, self.autocast)
                total += float(O.global_cosine(en, de)) * len(xb) / len(x)
        return total


def ranking_loss(scores, coverage, tau=0.05, n_neg=64, gen=None):
    """scores, coverage: [B, 28, 28]. Positive patches: coverage >= 0.5; negatives: coverage == 0.
    Mean over images of mean_{p in P, n in sampled N} softplus((s_n - s_p) / tau)."""
    out = []
    for s, c in zip(scores, coverage):
        pos, neg = s[c >= 0.5], s[c == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        if neg.numel() > n_neg:
            neg = neg[torch.randperm(neg.numel(), generator=gen)[:n_neg].to(neg.device)]
        out.append(F.softplus((neg[None, :] - pos[:, None]) / tau).mean())
    return torch.stack(out).mean() if out else None


class SyntheticRanking:
    """bank: {generator_name: list of {"host": (category, index), "features": [2, 789, 768], "coverage": [28, 28]}}."""

    def __init__(self, bank, train_gen, diag_gen, allowed, seed=0, n=8, micro=8, tau=0.05, device="cuda",
                 autocast=True):
        self.bank, self.train_gen, self.diag_gen = bank, train_gen, diag_gen
        self.allowed = allowed                                  # callable(ctx) -> set of allowed host keys
        self.gen = torch.Generator().manual_seed(seed)
        self.n, self.micro, self.tau, self.device, self.autocast = n, micro, tau, device, autocast

    def _sample(self, name, ctx):
        ok = self.allowed(ctx)
        items = [it for it in self.bank[name] if it["host"] in ok]
        order = torch.randperm(len(items), generator=self.gen).tolist()[:self.n]
        return [items[j] for j in order]

    def _loss(self, dec, items, grad):
        total, used = 0., 0
        for k in range(0, len(items), self.micro):
            chunk = items[k:k + self.micro]
            x = torch.stack([it["features"] for it in chunk]).to(self.device)
            cov = torch.stack([it["coverage"] for it in chunk]).to(self.device)
            with torch.set_grad_enabled(grad):
                en, de = _forward(dec, x, self.device, self.autocast)
                loss = ranking_loss(R.grid(en, de), cov, self.tau, gen=self.gen)
                if loss is None:
                    continue
                w = len(chunk) / len(items)
                if grad:
                    (loss * w).backward()
            total += float(loss) * w
            used += len(chunk)
        return total, used

    def __call__(self, dec, ctx):
        items = self._sample(self.train_gen, ctx)
        loss, used = self._loss(dec, items, grad=True)
        diag, _ = self._loss(dec, self._sample(self.diag_gen, ctx), grad=False)
        return {"loss": loss if used else float("nan"), "used": used, "cross_generator_loss": diag}

    def evaluate(self, dec, ctx, name):
        """Diagnostic only (no gradient): ranking loss of generator `name` on allowed hosts at the current state."""
        loss, used = self._loss(dec, self._sample(name, ctx), grad=False)
        return loss if used else float("nan")


class WithDiagnostics:
    """Outer objective plus gradient-free diagnostics evaluated at the same theta+ (they never touch .grad)."""

    def __init__(self, outer, diags):
        self.outer, self.diags = outer, diags

    def __call__(self, dec, ctx):
        out = self.outer(dec, ctx)
        with torch.no_grad():
            for name, fn in self.diags.items():
                out[name] = fn(dec, ctx)
        return out
