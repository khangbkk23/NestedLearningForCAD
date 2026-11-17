"""NAR-PILOT-00 core: Nested Acquisition Rule under structural retention (model-agnostic; no dataset access).

Spec: agents/plans/nl_cad_final_paradigm_spec.md. Prereg: agents/plans/nar_pilot_00_prereg.md.

Level 0 (fast, every step)   a = {(A_g, B_g)}: rank-r increments on frozen linear maps, W_g = W_g^0 + B_g A_g.
                             a_{k+1} = StableAdamW step with group learning rate  lr(k) * exp(lam_g).
Level 1 (slow, every m steps) lam_g <- clamp(lam_g - eta_lam * cos(grad_g L_cred(a_k), a_k,g - a_{k-m},g)):
                             normalized displacement surrogate for a hypergradient, not an exact multi-step
                             derivative through StableAdamW moments, clipping, or the fast trajectory.
Level 2 (slowest, per task)  phi <- phi + beta * (a_T - phi): Reptile recurrence on the fast initialisation.
Retention is structural: a deterministic eval-mode expert with the same frozen base and finished a_c is unchanged.
Task-agnostic aggregation can still change when new experts are added.
With eta_lam = 0 and beta = 0 the procedure is the fixed-rule learner (StableAdamW from the fixed initialisation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


def require(c, msg):
    if not c:
        raise RuntimeError(msg)


@dataclass(frozen=True)
class Cfg:
    rank: int = 8
    steps: int = 300
    lr: float = 2e-3
    final_ratio: float = 0.1
    warmup: int = 20
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 1e-4
    eps: float = 1e-10
    clip: float = 0.1
    m: int = 10                 # fast steps per slow (lam) update
    eta_lam: float = 0.2
    lam_max: float = math.log(8.0)
    beta: float = 0.5           # Reptile rate of the initialisation

    def dict(self):
        return asdict(self)


def lr_at(cfg, i):
    """Linear warm-up to cfg.lr, then cosine to cfg.final_ratio * cfg.lr at cfg.steps."""
    if i < cfg.warmup:
        return cfg.lr * (i + 1) / cfg.warmup
    p = (i - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    lo = cfg.lr * cfg.final_ratio
    return lo + 0.5 * (cfg.lr - lo) * (1 + math.cos(math.pi * p))


class LoRALinear(nn.Module):
    """y = base(x) + B A x with the base frozen. A [r, in], B [out, r]."""

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        for q in base.parameters():
            q.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(rank, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))


def attach(model: nn.Module, names, rank: int):
    """Freeze ``model`` and wrap the named nn.Linear modules. Returns {name: LoRALinear} in the given order."""
    for q in model.parameters():
        q.requires_grad_(False)
    mods = dict(model.named_modules())
    out = {}
    for n in names:
        base = mods[n]
        require(isinstance(base, nn.Linear), f"{n} is not nn.Linear")
        parent = mods[n.rsplit(".", 1)[0]] if "." in n else model
        w = LoRALinear(base, rank)
        setattr(parent, n.rsplit(".", 1)[-1], w)
        out[n] = w
    return out


def init_state(loras, seed: int):
    """Fixed initialisation: A ~ U(-1/sqrt(in), 1/sqrt(in)), B = 0 (the increment is exactly zero)."""
    g = torch.Generator().manual_seed(seed)
    st = {}
    for n, w in loras.items():
        b = 1 / math.sqrt(w.A.shape[1])
        st[n] = ((torch.rand(w.A.shape, generator=g) * 2 - 1) * b, torch.zeros(w.B.shape))
    return st


def get_state(loras):
    return {n: (w.A.detach().cpu().clone(), w.B.detach().cpu().clone()) for n, w in loras.items()}


@torch.no_grad()
def set_state(loras, st):
    for n, w in loras.items():
        w.A.copy_(st[n][0].to(w.A))
        w.B.copy_(st[n][1].to(w.B))


def state_bytes(st, dtype_bytes=4):
    """Payload bytes in the declared serialized dtype (FP32 by default)."""
    return sum(a.numel() + b.numel() for a, b in st.values()) * dtype_bytes


def tensor_state_bytes(st):
    """Actual payload bytes, including FP64 CPU mechanics states."""
    return sum(x.numel() * x.element_size() for pair in st.values() for x in pair)


def slow_bytes(lam, phi):
    return lam.numel() * lam.element_size() + tensor_state_bytes(phi)


def ledger(base, archive, lam, phi):
    """Persistent tensor payload only; transient optimizer/snapshots charged separately."""
    parts = {"shared_base": sum(x.numel() * x.element_size() for x in base.state_dict().values()),
             "finished_adapters": sum(tensor_state_bytes(a) for a in archive),
             "initialization": tensor_state_bytes(phi),
             "log_rates": lam.numel() * lam.element_size()}
    return {**parts, "total": sum(parts.values()), "task_count": len(archive)}


def normal_partition(rows):
    """Deterministic image-disjoint FIT/CREDIT/EVAL roles from current train/good only.

    No tensors are loaded here. Caller must restrict cache access to the current task;
    this validation is a membership guard, not a filesystem security boundary.
    """
    import hashlib
    paths = [r["relative_path"] for r in rows]
    require(len(paths) == len(set(paths)) and len(paths) >= 10, "duplicate or insufficient normal identities")
    require(all(len(p.split("/")) == 4 and p.split("/")[1:3] == ["train", "good"]
                and ".." not in p.split("/") for p in paths), "expected current-task train/good identities")
    require(len({p.split("/")[0] for p in paths}) == 1, "mixed or future tasks in partition")
    order = sorted(range(len(paths)), key=lambda i: hashlib.sha256(
        ("NAR-PILOT-00/roles/v1|" + paths[i]).encode()).hexdigest())
    n = max(1, len(paths) // 5)
    return {"FIT": order[2*n:], "CREDIT": order[n:2*n], "EVAL": order[:n]}


def task_agnostic_grid(expert_grids):
    """Diagnostic minimum over arrived expert maps, no ground-truth task ID.

    Each input is a fixed expert's 1-cos map [B,H,W]. This aggregation does not
    inherit expert functional invariance as the expert set grows.
    """
    require(len(expert_grids) > 0, "no acquired expert")
    require(all(x.ndim == 3 and x.shape == expert_grids[0].shape for x in expert_grids), "expert map shape")
    require(all(bool(torch.isfinite(x).all()) for x in expert_grids), "nonfinite expert map")
    grid = torch.stack(expert_grids).clamp_min(0).amin(0)
    return grid, grid.flatten(1).amax(1)


def reptile(phi, a, beta, perm=None):
    """phi <- phi + beta (a - phi). ``perm`` {name: source name} misassigns the credit (control)."""
    out = {}
    for n in phi:
        src = a[perm[n]] if perm else a[n]
        require(src[0].shape == phi[n][0].shape and src[1].shape == phi[n][1].shape, "permutation changes shapes")
        out[n] = (phi[n][0] + beta * (src[0] - phi[n][0]), phi[n][1] + beta * (src[1] - phi[n][1]))
    return out


def _flat(w):
    return torch.cat([w.A.detach().reshape(-1), w.B.detach().reshape(-1)])


def _flat_grad(w):
    return torch.cat([w.A.grad.reshape(-1), w.B.grad.reshape(-1)])


def acquire(loras, fit_loss, cred_loss, cfg: Cfg, lam, learn_lam: bool, make_opt, perm=None, on_step=None,
            observe_credit=False):
    """Fast acquisition of one task. ``fit_loss(i)`` and ``cred_loss()`` return scalar losses with graphs.

    lam: FP64 tensor [G] (one entry per wrapped matrix, order of ``loras``); returned updated (a copy).
    perm: list[int] group permutation applied to the credit (control arm); None = correct credit.
    make_opt(param_groups) -> optimizer (StableAdamW in the pilot). on_step(k) is called after fast step k (1-based).
    observe_credit=True evaluates the identical credit gradients in fixed/frozen controls without applying them.
    """
    names = list(loras)
    lam = lam.clone().double()
    require(lam.shape == (len(names),), "lam shape")
    opt = make_opt([{"params": [loras[n].A, loras[n].B]} for n in names])
    params = [q for n in names for q in (loras[n].A, loras[n].B)]
    snap = [_flat(loras[n]).clone() for n in names]
    log = {"fit": [], "credit": [], "lam_mean": [], "lam_std": [], "h_mean": [], "h_pos": []}
    for i in range(cfg.steps):
        base = lr_at(cfg, i)
        for g, grp in enumerate(opt.param_groups):
            grp["lr"] = base * math.exp(float(lam[g]))
        opt.zero_grad(set_to_none=True)
        loss = fit_loss(i)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm=cfg.clip)
        lv = float(loss.detach())
        require(math.isfinite(lv) and bool(torch.isfinite(norm)), f"non-finite fast step {i}")
        opt.step()
        log["fit"].append(lv)
        if (learn_lam or observe_credit) and (i + 1) % cfg.m == 0:
            opt.zero_grad(set_to_none=True)
            credit = cred_loss()
            require(bool(torch.isfinite(credit)), f"non-finite credit loss at step {i}")
            credit.backward()
            h = torch.zeros(len(names), dtype=torch.float64)
            for g, n in enumerate(names):
                d = (_flat(loras[n]) - snap[g]).double()
                gr = _flat_grad(loras[n]).double()
                den = d.norm() * gr.norm()
                h[g] = float((d @ gr) / den) if float(den) > 0 else 0.0
            require(bool(torch.isfinite(h).all()), f"non-finite credit at step {i}")
            if perm is not None:
                h = h[torch.as_tensor(perm)]
            if learn_lam:
                lam = (lam - cfg.eta_lam * h).clamp(-cfg.lam_max, cfg.lam_max)
            snap = [_flat(loras[n]).clone() for n in names]
            opt.zero_grad(set_to_none=True)
            log["h_mean"].append(float(h.mean()))
            log["credit"].append(float(credit.detach()))
            log["h_pos"].append(float((h < 0).double().mean()))        # share of groups asked to speed up
            log["lam_mean"].append(float(lam.mean()))
            log["lam_std"].append(float(lam.std()))
        if on_step is not None:
            on_step(i + 1)
    return lam, log
