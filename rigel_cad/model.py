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

Persistent state charged against the budget (RigelConfig.storage, see rigel_cad/storage.py):
* 'fp16' (default, v1): fp16 keys (1,536 B each) + S (fp32) + count (fp64) = 2,359,304 B.
* 'int4_triangular' (compact memory): int4 keys with fp16 scale (386 B each) + fp32 upper triangle of S + count
  (1,181,192 B). The storage format changes only how the two levels are stored; the levels, their update schedule,
  the consolidation rule and the read-out are the same.
The frozen encoder is shared and reported separately.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

from rigel_cad import ops, storage
from rigel_cad.config import GRID, TOKEN_DIM, TOKENS_PER_IMAGE, RigelConfig

FORMAT = 'rigel-cad-state/1'               # storage 'fp16' (unchanged since RIGEL-CAD 1.0)
FORMAT_INT4_TRI = 'rigel-cad-state/2'      # storage 'int4_triangular'
INT4_TRI = 'int4_triangular'


class BudgetViolation(AssertionError):
    """Raised if the persistent state would exceed the configured budget."""


@dataclass
class RigelState:
    """Read-out state of RIGEL-CAD after some task: what is needed to score a query."""
    keys: Optional[torch.Tensor]       # [M, D] keys as scored (CPU): fp16 unit keys ('fp16') or fp32 dequantised
                                       # int4 keys ('int4_triangular'); None only inside a stored history
    W: Optional[torch.Tensor]          # [D, D] fp32 read-out whitener (CPU); None for the fast-only ablation
    n_tasks: int
    bytes: int                         # persistent bytes charged against the budget
    storage: str = 'fp16'
    q4: Optional[torch.Tensor] = None      # [M, D / 2] packed int4 codes ('int4_triangular' only)
    scale: Optional[torch.Tensor] = None   # [M] fp16 per-key scales ('int4_triangular' only)


@dataclass
class TaskFit:
    """Budget-independent result of learning one task (RigelCAD.compute_fit)."""
    keys16: torch.Tensor               # [k, D] fp16 coreset tokens in k-centre order (on device)
    S: torch.Tensor                    # [D, D] fp64 residual moment of the task (CPU)
    n: int                             # number of residuals in S
    n_tokens: int                      # number of task tokens


@dataclass
class Prepared:
    """Device-resident read-out of a state (RigelCAD.prepare): keys as scored (fp32), W and keys @ W."""
    keys: torch.Tensor
    W: Optional[torch.Tensor]
    mw: Optional[torch.Tensor]
    n_tasks: int
    bytes: int


class FastKeyMemory:
    """Fast level: per-task 1 % coresets consolidated to a fixed key capacity at every task boundary.

    storage 'fp16': keys are the fp16 coreset tokens. storage 'int4_triangular': each new coreset is quantised to int4
    (rigel_cad.storage.quantize4) when written; the consolidation runs on the dequantised candidates, so the stored
    codes are what is selected and scored.
    """

    def __init__(self, capacity: Optional[int], selection: str, seed: int, device: str, storage: str = 'fp16'):
        self.capacity, self.selection, self.seed, self.device = capacity, selection, seed, device
        self.storage = storage
        self.keys = torch.empty(0, TOKEN_DIM, dtype=torch.float16, device=device)
        self.blocks: List[torch.Tensor] = []          # only kept when unbounded (union of coresets, CPU)
        if storage == INT4_TRI:
            if capacity is None or selection != 'kcenter':
                raise ValueError('int4 keys need a finite capacity and k-centre consolidation')
            self.q = torch.empty(0, TOKEN_DIM, dtype=torch.int8, device=device)     # codes in [-7, 7]
            self.scale = torch.empty(0, dtype=torch.float16, device=device)
            self.keys = torch.empty(0, TOKEN_DIM, dtype=torch.float32, device=device)  # dequantised view

    @torch.no_grad()
    def write(self, task_keys: torch.Tensor) -> None:
        """task_keys: [k, D] fp16 coreset of the new task (on device), in k-centre order."""
        if self.storage == INT4_TRI:
            self._write_int4(task_keys)
            return
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

    @torch.no_grad()
    def _write_int4(self, task_keys: torch.Tensor) -> None:
        qn, sn = storage.quantize4(task_keys)
        cq, cs = torch.cat([self.q, qn]), torch.cat([self.scale, sn])
        cand = storage.dequantize(cq, cs)
        if len(cand) > self.capacity:
            sel = ops.greedy_kcenter(cand, self.capacity, seed=self.seed)
            keep = torch.arange(len(cand))[sel.sort().values].to(cand.device)   # survivors keep insertion order
            cq, cs = cq[keep], cs[keep]
        self.q, self.scale = cq, cs
        self.keys = storage.dequantize(cq, cs)

    def current(self) -> torch.Tensor:
        """Keys as scored (CPU): fp16 ('fp16') or fp32 dequantised ('int4_triangular')."""
        if self.capacity is None:
            return torch.cat(self.blocks) if self.blocks else torch.empty(0, TOKEN_DIM, dtype=torch.float16)
        return self.keys.cpu()

    def packed(self):
        """('int4_triangular') persistent form of the keys: (packed codes [M, D / 2] uint8, scales [M] fp16), CPU."""
        return storage.pack4(self.q).cpu(), self.scale.cpu()

    def load_packed(self, q4: torch.Tensor, scale: torch.Tensor) -> None:
        self.q = storage.unpack4(q4).to(self.device)
        self.scale = scale.to(self.device)
        self.keys = storage.dequantize(self.q, self.scale)

    def __len__(self):
        return sum(len(b) for b in self.blocks) if self.capacity is None else len(self.keys)


class SlowResidualWhitener:
    """Slow level: pooled residual second moment S and count n; read-out W = shrink(S / n)^(-1/2).

    triangular=False ('fp16' storage, v1): S is the fp64 running sum.
    triangular=True ('int4_triangular'): S is kept as its fp32 upper triangle S_tri; each task adds S_t in fp64 to the
    unpacked triangle and the sum is re-rounded to the fp32 triangle.
    """

    def __init__(self, triangular: bool = False):
        self.triangular = triangular
        self.S = None
        self.S_tri = None
        self.n = 0

    def write(self, S_t: torch.Tensor, n_t: int) -> None:
        if self.triangular:
            prev = storage.tri_unpack(self.S_tri) if self.S_tri is not None else 0.
            self.S_tri = storage.tri_pack(prev + S_t)
        else:
            self.S = S_t.clone() if self.S is None else self.S + S_t
        self.n += int(n_t)

    def matrix(self) -> Optional[torch.Tensor]:
        """The accumulated moment as an fp64 [D, D] matrix (None before the first task)."""
        if self.triangular:
            return None if self.S_tri is None else storage.tri_unpack(self.S_tri)
        return self.S

    def W(self) -> torch.Tensor:
        S = self.matrix()
        if S is None:
            raise RuntimeError('slow level has no statistic yet (no task seen)')
        W, _ = ops.whitener(S, self.n)
        return W.float()

    def condition(self) -> float:
        return ops.whitener(self.matrix(), self.n)[1]


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
        self.fast = FastKeyMemory(c.key_capacity, c.selection, c.seed, c.device, c.storage)
        self.slow = SlowResidualWhitener(triangular=c.storage == INT4_TRI) if c.slow_level else None
        self.n_tasks = 0
        self.task_names: List[str] = []
        self.update_seconds = 0.0
        self.keep_history = keep_history
        self.history: List[RigelState] = []
        self._prep = None                  # (fingerprint, Prepared) of the current state, see prepare()

    # ------------------------------------------------------------------ learning
    @torch.no_grad()
    def fit_task(self, tokens16: torch.Tensor, name: Optional[str] = None) -> RigelState:
        """Learn one task from its normal training tokens (= write_task(compute_fit(tokens16), name)).

        tokens16: [N, 768] fp16 unit-norm patch tokens (all training images of the task, images in sorted path order,
        784 tokens per image), ideally already on config.device.
        """
        t0 = time.time()
        fit = self.compute_fit(tokens16)
        return self.write_task(fit, name, seconds=time.time() - t0)

    @torch.no_grad()
    def compute_fit(self, tokens16: torch.Tensor) -> 'TaskFit':
        """The budget-independent part of a task update: the task's 1 % greedy k-centre coreset and its residual
        moment (S_t, n_t). Depends only on the tokens and the seed, not on the memory state or the budget."""
        if tokens16.dtype != torch.float16 or tokens16.ndim != 2 or tokens16.shape[1] != TOKEN_DIM:
            raise ValueError(f'expected [N, {TOKEN_DIM}] fp16 tokens, got {tuple(tokens16.shape)} {tokens16.dtype}')
        x16 = tokens16.to(self.config.device)
        k = ops.coreset_size(len(x16))
        kc = ops.greedy_kcenter(x16, k, seed=self.config.seed)
        S_t, n_t = ops.residual_moment(x16.float(), kc[:k].to(x16.device))
        return TaskFit(keys16=x16[kc[:k].to(x16.device)], S=S_t, n=n_t, n_tokens=len(x16))

    @torch.no_grad()
    def write_task(self, fit: 'TaskFit', name: Optional[str] = None, seconds: Optional[float] = None) -> RigelState:
        """The state-dependent part of a task update: write the task's coreset into the fast level (consolidating to
        the capacity) and its residual moment into the slow level. `seconds`: update time to account (default: the
        time of this call)."""
        t0 = time.time()
        self.fast.write(fit.keys16.to(self.config.device))
        if self.slow is not None:
            self.slow.write(fit.S, fit.n)
        self.update_seconds += (time.time() - t0) if seconds is None else seconds + (time.time() - t0)
        self.n_tasks += 1
        self.task_names.append(name or f'task{self.n_tasks - 1}')
        self._prep = None
        st = self.state()
        if self.keep_history:
            # unbounded memory: store only the prefix length; keys are re-assembled on demand (no per-stage copies).
            # int4 storage: store the packed codes and scales (386 B per key); keys are dequantised on demand.
            if self.fast.capacity is None:
                self.history.append(RigelState(None, st.W, st.n_tasks, st.bytes))
            elif st.storage == INT4_TRI:
                self.history.append(RigelState(None, st.W, st.n_tasks, st.bytes, INT4_TRI, st.q4, st.scale))
            else:
                self.history.append(st)
        return st

    # ------------------------------------------------------------------ state
    def persistent_bytes(self) -> int:
        """Registered (logical) persistent bytes: keys x key_bytes + slow statistic, in the configured storage."""
        return len(self.fast) * self.config.key_bytes + self.config.slow_bytes

    def state(self) -> RigelState:
        b = self.persistent_bytes()
        lim = self.config.budget_bytes
        if lim is not None and b > lim:
            raise BudgetViolation(f'state {b} B > budget {lim} B after task {self.n_tasks}')
        W = self.slow.W() if self.slow is not None else None
        if self.config.storage == INT4_TRI:
            q4, scale = self.fast.packed()
            return RigelState(keys=self.fast.current(), W=W, n_tasks=self.n_tasks, bytes=b, storage=INT4_TRI, q4=q4,
                              scale=scale)
        return RigelState(keys=self.fast.current(), W=W, n_tasks=self.n_tasks, bytes=b)

    def state_at(self, task_index: int) -> RigelState:
        """Read-out state right after task `task_index` (requires keep_history=True)."""
        if not self.keep_history:
            raise RuntimeError('construct RigelCAD(keep_history=True) to score past stages')
        st = self.history[task_index]
        if st.keys is None and st.storage == INT4_TRI:
            keys = storage.dequantize(storage.unpack4(st.q4), st.scale)
            return RigelState(keys, st.W, st.n_tasks, st.bytes, INT4_TRI, st.q4, st.scale)
        if st.keys is None:
            return RigelState(torch.cat(self.fast.blocks[:st.n_tasks]), st.W, st.n_tasks, st.bytes)
        return st

    # ------------------------------------------------------------------ scoring
    def _fingerprint(self):
        """Identity of the current state's tensors (a re-assigned key or moment tensor gives a new fingerprint)."""
        f, s = self.fast, self.slow
        keys = len(f.blocks) if f.capacity is None else (f.keys.data_ptr(), tuple(f.keys.shape))
        slow = None if s is None else (id(s.S), id(s.S_tri), s.n)
        return self.n_tasks, keys, slow

    @torch.no_grad()
    def prepare(self, state: Optional[RigelState] = None) -> 'Prepared':
        """Device-resident read-out of `state` (default: the current state): keys as scored (fp32), W and keys @ W.
        Scoring with it gives exactly the same energies as scoring with the state; it only avoids recomputing W
        (current state), moving the keys to the device and multiplying them by W on every call. The current state's
        read-out is cached by patch_energy(q) until the next write_task / load."""
        if state is None:
            fp = self._fingerprint()
            if self._prep is not None and self._prep[0] == fp:
                return self._prep[1]
            p = self.prepare(self.state())
            self._prep = (fp, p)
            return p
        dev = self.config.device
        keys = state.keys.to(dev).float()
        W = None if state.W is None else state.W.to(dev)
        return Prepared(keys=keys, W=W, mw=None if W is None else keys @ W, n_tasks=state.n_tasks, bytes=state.bytes)

    def clear_cache(self) -> None:
        """Drop the cached device read-out of the current state (frees its GPU memory)."""
        self._prep = None

    @torch.no_grad()
    def patch_energy(self, q16: torch.Tensor, state=None) -> torch.Tensor:
        """Per-token energy [Nq] (fp32, CPU) of fp16 unit query tokens [Nq, 768].

        state: None (current state, cached device read-out), a RigelState (e.g. state_at(l)), or a Prepared."""
        if isinstance(state, Prepared):
            p = state
        elif state is None:
            p = self.prepare()
        else:
            st = state
            keys = st.keys.to(self.config.device).float()
            if st.W is None:
                return ops.energy_euclidean(q16, keys)
            return ops.energy_whitened(q16, keys, st.W.to(self.config.device))
        if p.W is None:
            return ops.energy_euclidean(q16, p.keys)
        return ops.energy_whitened(q16, p.keys, p.W, mw=p.mw)

    def patch_energy_grid(self, tokens: torch.Tensor, state=None) -> torch.Tensor:
        """tokens: [n_img, 784, 768] or [n_img * 784, 768] fp16 unit tokens. Returns [n_img, 28, 28] fp32 energies."""
        q = tokens.reshape(-1, TOKEN_DIM)
        n = len(q) // TOKENS_PER_IMAGE
        return self.patch_energy(q, state).reshape(n, GRID, GRID)

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> Path:
        """Write the persistent state (the only thing a deployment must keep between tasks) to `path`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.config.storage == INT4_TRI:
            q4, scale = self.fast.packed()
            torch.save({'format': FORMAT_INT4_TRI, 'storage': INT4_TRI, 'config': self.config.to_dict(),
                        'n_tasks': self.n_tasks, 'task_names': self.task_names, 'q4': q4, 'scale': scale,
                        'S_tri': None if self.slow.S_tri is None else self.slow.S_tri.cpu(), 'n': self.slow.n,
                        'persistent_bytes': self.persistent_bytes()}, path)
            return path
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
        if d.get('format') not in (FORMAT, FORMAT_INT4_TRI):
            raise ValueError(f'not a RIGEL-CAD state file: {path}')
        cfg = RigelConfig.from_dict(d['config'])
        if (cfg.storage == INT4_TRI) != (d['format'] == FORMAT_INT4_TRI):
            raise ValueError(f'storage {cfg.storage!r} does not match the state format {d["format"]!r}: {path}')
        if device is not None:
            cfg.device = device
        m = cls(cfg)
        m.n_tasks, m.task_names = d['n_tasks'], list(d['task_names'])
        if cfg.storage == INT4_TRI:
            m.fast.load_packed(d['q4'], d['scale'])
            m.slow.S_tri, m.slow.n = d['S_tri'], d['n']
            return m
        if m.fast.capacity is None:
            m.fast.blocks = list(d['blocks'])
        else:
            m.fast.keys = d['keys'].to(cfg.device)
        if m.slow is not None:
            m.slow.S, m.slow.n = d['S'], d['n']
        return m
