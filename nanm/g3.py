"""G3: per-task k-centre keys (fast level) + task-boundary residual-covariance whitening (slow statistic).

Composition follows exps/bench_g3/run.py::_normal_ds (fit) and ::whitened / ::score (stage scoring) exactly,
but every numerical primitive is the oracle's own function object:

    keys      A.greedy_coreset          (exps/nanm_r0/audit.py)   seeded Gaussian-projection greedy k-centre
    S, n      B.residual_stats          (exps/nanm_r2b/run.py)    uncentred residual 2nd moment of non-members
    W         B.whiten                  (exps/nanm_r2b/run.py)    inv_sqrt_psd(shrink(S/n, 0.5)), fp64
    distance  R2.sqdist                 (exps/nanm_r2/run.py)
    size      R1.n_of                   (exps/nanm_r1/run.py)     ceil(r * n)

Differences from the oracle are device handling only: the oracle moves query chunks with `.cuda()`; here the
query chunk is moved to the device of `keys`. No scientific constant or operation order is changed.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from exps.nanm_r0 import audit as A
from exps.nanm_r1 import run as R1
from exps.nanm_r2 import run as R2
from exps.nanm_r2b import run as B

D = 768
KEY_BYTES = 2 * D                 # fp16 key, as in bench_g3.KEY
S_BYTES = D * D * 4 + 8           # fp32 S + fp64 count, as in bench_g3.EXTRA_G3 (= 2,359,304)
RATIO = 0.01                      # G3 1 % per-task k-centre ratio
SHRINK = R2.SHRINK                # 0.5
QC = 512                          # query chunk, as in bench_g3.QC


@dataclass
class TaskState:
    """Persistent per-task state written at the task boundary."""
    kc: torch.Tensor      # [n_select] long, k-centre order (prefixes are valid smaller banks)
    keys: torch.Tensor    # [k, D] fp16 unit-norm keys (k-centre prefix of length k)
    S: torch.Tensor       # [D, D] fp64 residual second moment of the task's non-member FIT tokens
    n: int                # number of residual rows in S
    n_fit: int


def task_size(n_fit: int, ratio: float = RATIO) -> int:
    return R1.n_of(ratio, n_fit)


@torch.no_grad()
def fit_task(fit16: torch.Tensor, k: int, seed: int, n_select: int | None = None) -> TaskState:
    """fit16: [N, D] fp16 unit-norm FIT tokens of one task (any device). Mirrors bench_g3._normal_ds for the G3 arm:
    k-centre on fp16 tokens, residual statistic on the fp32 copy against the first k centres."""
    n_select = k if n_select is None else n_select
    if not (0 < k <= n_select <= len(fit16)):
        raise ValueError(f'need 0 < k <= n_select <= N, got k={k} n_select={n_select} N={len(fit16)}')
    kc = A.greedy_coreset(fit16, n_select, seed=seed)
    S, n = B.residual_stats(fit16.float(), kc[:k].to(fit16.device))
    return TaskState(kc=kc, keys=fit16[kc[:k].to(fit16.device)].cpu(), S=S, n=int(n), n_fit=len(fit16))


def stage_whitener(states: list[TaskState]) -> tuple[torch.Tensor, float]:
    """Stage-l whitening from the tasks seen so far: B.whiten(sum S, sum n). Returns (W fp32, condition number)."""
    if not states:
        raise ValueError('no tasks seen')
    W, cond = B.whiten(sum(s.S for s in states), sum(s.n for s in states))
    return W.float(), cond


@torch.no_grad()
def score(q16: torch.Tensor, keys: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Patch energy 0.5 * min_m ||q W - m W||^2 (fp32), identical in operation order to bench_g3.whitened.
    q16: [Nq, D] fp16 unit tokens (CPU); keys: [M, D] fp32 union memory; W: [D, D] fp32 (same device as keys)."""
    mw = keys @ W
    out = torch.empty(len(q16))
    for i in range(0, len(q16), QC):
        out[i:i + QC] = (0.5 * R2.sqdist(q16[i:i + QC].to(keys.device).float() @ W, mw).min(1).values).cpu()
    return out


def union_keys(states: list[TaskState]) -> torch.Tensor:
    """Task-free union memory over all seen tasks (fp32), as in bench_g3.score."""
    return torch.cat([s.keys for s in states]).float()


def persistent_bytes(states: list[TaskState]) -> int:
    """G3 persistent state charged by BENCH-G3: fp16 keys + one fp32 S with its count."""
    return sum(len(s.keys) for s in states) * KEY_BYTES + S_BYTES
