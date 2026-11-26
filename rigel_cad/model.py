"""RIGEL-CAD: a two-level continual anomaly-detection memory under a fixed byte budget.

Two learning levels with different update frequencies (Nested Learning, Def. 2-3 of the NL paper, used at the
mechanism level):

* FastKeyMemory (fast level, written once per task): the task's normal patch tokens are summarised by a 1 % greedy
  k-centre coreset; at each task boundary the union {current keys} u {new coreset} is consolidated back to the key
  capacity by a seeded greedy k-centre (or FIFO in the ablation).
* SlowResidualWhitener (slow level, accumulated over the whole stream): the uncentred second moment of the residuals
  "token - nearest key of its task" is pooled over all tasks; the read-out metric W = shrink(S / n, 0.5)^(-1/2)
  whitens both queries and keys before the nearest-key energy.

Patch energy: e(q) = 0.5 * min_m ||q W - m W||^2. Anomaly map: bilinear 28 -> 392 + Gaussian(sigma 4) smoothing,
image score = mean of the top 1 % map values (see rigel_cad/readout.py).

Persistent state charged against the budget: fp16 keys (1,536 B each) + S (fp32) + count (fp64) = 2,359,304 B.
The frozen encoder is shared and reported separately.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

from rigel_cad import ops
from rigel_cad.config import GRID, KEY_BYTES, TOKEN_DIM, TOKENS_PER_IMAGE, RigelConfig

FORMAT = 'rigel-cad-state/1'


class BudgetViolation(AssertionError):
    """Raised if the persistent state would exceed the configured budget."""


@dataclass
class RigelState:
    """Read-out state of RIGEL-CAD after some task: what is needed to score a query."""
    keys: Optional[torch.Tensor]       # [M, D] fp16 unit keys (CPU); None only inside an unbounded history
    W: Optional[torch.Tensor]          # [D, D] fp32 read-out whitener (CPU); None for the fast-only ablation
    n_tasks: int
    bytes: int                         # persistent bytes charged against the budget


class FastKeyMemory:
    """Fast level: per-task 1 % coresets consolidated to a fixed key capacity at every task boundary."""

    def __init__(self, capacity: Optional[int], selection: str, seed: int, device: str):
        self.capacity, self.selection, self.seed, self.device = capacity, selection, seed, device
        self.keys = torch.empty(0, TOKEN_DIM, dtype=torch.float16, device=device)
        self.blocks: List[torch.Tensor] = []          # only kept when unbounded (union of coresets, CPU)

    @torch.no_grad()
    def write(self, task_keys: torch.Tensor) -> None:
        """task_keys: [k, D] fp16 coreset of the new task (on device), in k-centre order."""
        if self.capacity is None:
            self.blocks.append(task_keys.cpu())
            return
        if self.selection == 'fifo':
            K, C, cap = self.keys, task_keys, self.capacity
            if len(K) + len(C) <= cap:
                self.keys = torch.cat([K, C])
            elif len(C) >= cap:
                self.keys = C[:cap]
            else:
                self.keys = torch.cat([K[len(K) - (cap - len(C)):], C])
            return
        cand = torch.cat([self.keys, task_keys])
        if len(cand) <= self.capacity:
            self.keys = cand
            return
        sel = ops.greedy_kcenter(cand, self.capacity, seed=self.seed)
        keep = torch.arange(len(cand))[sel.sort().values]        # survivors keep their insertion order
        self.keys = cand[keep.to(cand.device)]

    def current(self) -> torch.Tensor:
        if self.capacity is None:
            return torch.cat(self.blocks) if self.blocks else torch.empty(0, TOKEN_DIM, dtype=torch.float16)
        return self.keys.cpu()

    def __len__(self):
        return sum(len(b) for b in self.blocks) if self.capacity is None else len(self.keys)


class SlowResidualWhitener:
    """Slow level: pooled residual second moment S (fp64) and count n; read-out W = shrink(S / n)^(-1/2)."""

    def __init__(self):
        self.S = None
        self.n = 0

    def write(self, S_t: torch.Tensor, n_t: int) -> None:
        self.S = S_t.clone() if self.S is None else self.S + S_t
        self.n += int(n_t)

    def W(self) -> torch.Tensor:
        if self.S is None:
            raise RuntimeError('slow level has no statistic yet (no task seen)')
        W, _ = ops.whitener(self.S, self.n)
        return W.float()

    def condition(self) -> float:
        return ops.whitener(self.S, self.n)[1]


class RigelCAD:
    """Continual anomaly detector. Typical use::

        model = RigelCAD(RigelConfig(budget_mib=8))
        for task_tokens in stream:            # [N_t, 768] fp16 unit tokens of the task's normal training images
            model.fit_task(task_tokens)
        grids = model.patch_energy_grid(test_tokens)       # [n_img, 28, 28]
    """

    def __init__(self, config: Optional[RigelConfig] = None, keep_history: bool = False):
        self.config = config or RigelConfig()
        c = self.config
        self.fast = FastKeyMemory(c.key_capacity, c.selection, c.seed, c.device)
        self.slow = SlowResidualWhitener() if c.slow_level else None
        self.n_tasks = 0
        self.task_names: List[str] = []
        self.update_seconds = 0.0
        self.keep_history = keep_history
        self.history: List[RigelState] = []

    # ------------------------------------------------------------------ learning
    @torch.no_grad()
    def fit_task(self, tokens16: torch.Tensor, name: Optional[str] = None) -> RigelState:
        """Learn one task from its normal training tokens.

        tokens16: [N, 768] fp16 unit-norm patch tokens (all training images of the task, images in sorted path order,
        784 tokens per image), ideally already on config.device.
        """
        if tokens16.dtype != torch.float16 or tokens16.ndim != 2 or tokens16.shape[1] != TOKEN_DIM:
            raise ValueError(f'expected [N, {TOKEN_DIM}] fp16 tokens, got {tuple(tokens16.shape)} {tokens16.dtype}')
        x16 = tokens16.to(self.config.device)
        t0 = time.time()
        k = ops.coreset_size(len(x16))
        kc = ops.greedy_kcenter(x16, k, seed=self.config.seed)
        S_t, n_t = ops.residual_moment(x16.float(), kc[:k].to(x16.device))
        task_keys = x16[kc[:k].to(x16.device)]
        self.fast.write(task_keys)
        if self.slow is not None:
            self.slow.write(S_t, n_t)
        self.update_seconds += time.time() - t0
        self.n_tasks += 1
        self.task_names.append(name or f'task{self.n_tasks - 1}')
        st = self.state()
        if self.keep_history:
            # unbounded memory: store only the prefix length; keys are re-assembled on demand (no per-stage copies)
            self.history.append(RigelState(None, st.W, st.n_tasks, st.bytes) if self.fast.capacity is None else st)
        return st

    # ------------------------------------------------------------------ state
    def persistent_bytes(self) -> int:
        return len(self.fast) * KEY_BYTES + self.config.slow_bytes

    def state(self) -> RigelState:
        b = self.persistent_bytes()
        lim = self.config.budget_bytes
        if lim is not None and b > lim:
            raise BudgetViolation(f'state {b} B > budget {lim} B after task {self.n_tasks}')
        W = self.slow.W() if self.slow is not None else None
        return RigelState(keys=self.fast.current(), W=W, n_tasks=self.n_tasks, bytes=b)

    def state_at(self, task_index: int) -> RigelState:
        """Read-out state right after task `task_index` (requires keep_history=True)."""
        if not self.keep_history:
            raise RuntimeError('construct RigelCAD(keep_history=True) to score past stages')
        st = self.history[task_index]
        if st.keys is None:
            return RigelState(torch.cat(self.fast.blocks[:st.n_tasks]), st.W, st.n_tasks, st.bytes)
        return st

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def patch_energy(self, q16: torch.Tensor, state: Optional[RigelState] = None) -> torch.Tensor:
        """Per-token energy [Nq] (fp32, CPU) of fp16 unit query tokens [Nq, 768]."""
        st = state or self.state()
        keys = st.keys.to(self.config.device).float()
        if st.W is None:
            return ops.energy_euclidean(q16, keys)
        return ops.energy_whitened(q16, keys, st.W.to(self.config.device))

    def patch_energy_grid(self, tokens: torch.Tensor, state: Optional[RigelState] = None) -> torch.Tensor:
        """tokens: [n_img, 784, 768] or [n_img * 784, 768] fp16 unit tokens. Returns [n_img, 28, 28] fp32 energies."""
        q = tokens.reshape(-1, TOKEN_DIM)
        n = len(q) // TOKENS_PER_IMAGE
        return self.patch_energy(q, state).reshape(n, GRID, GRID)

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> Path:
        """Write the persistent state (the only thing a deployment must keep between tasks) to `path`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'format': FORMAT, 'config': self.config.to_dict(), 'n_tasks': self.n_tasks,
                    'task_names': self.task_names,
                    'keys': self.fast.keys.cpu() if self.fast.capacity is not None else None,
                    'blocks': self.fast.blocks if self.fast.capacity is None else None,
                    'S': None if self.slow is None else self.slow.S, 'n': None if self.slow is None else self.slow.n,
                    'persistent_bytes': self.persistent_bytes()}, path)
        return path

    @classmethod
    def load(cls, path, device: Optional[str] = None) -> 'RigelCAD':
        d = torch.load(path, map_location='cpu', weights_only=True)
        if d.get('format') != FORMAT:
            raise ValueError(f'not a RIGEL-CAD state file: {path}')
        cfg = RigelConfig.from_dict(d['config'])
        if device is not None:
            cfg.device = device
        m = cls(cfg)
        m.n_tasks, m.task_names = d['n_tasks'], list(d['task_names'])
        if m.fast.capacity is None:
            m.fast.blocks = list(d['blocks'])
        else:
            m.fast.keys = d['keys'].to(cfg.device)
        if m.slow is not None:
            m.slow.S, m.slow.n = d['S'], d['n']
        return m
