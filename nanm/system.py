"""NANMSystem: task-boundary continual normality memory (frozen NANM-G3 semantics).

One state transition only: fit_task(tokens) = one boundary event (write exemplars, update S/n, rebuild W and the
derived cache), computed into new objects and committed by a single assignment. predict() is task-free and pure.
Device semantics follow the oracle: writing, residuals and reading on the bank device (CUDA canonical); geometry on CPU
fp64; query normalisation on CPU (as G.load_test); readout on CPU.
"""
from __future__ import annotations

import torch

from nanm.config import G3Config
from nanm.geometry import ResidualNormalityGeometryEstimator
from nanm.memory import BoundaryExemplarWriter, quota, unit
from nanm.readout import FixedAnomalyReadout
from nanm.retrieval import WhitenedExemplarReader
from nanm.state import ContinualState


class NANMSystem:
    def __init__(self, cfg: G3Config, device):
        self.cfg, self.device = cfg, torch.device(device)
        self.writer = BoundaryExemplarWriter(cfg, self.device)
        self.geometry = ResidualNormalityGeometryEstimator(cfg)
        self.reader = WhitenedExemplarReader(cfg)
        self.readout = FixedAnomalyReadout(cfg)
        self.state = ContinualState(dim=cfg.dim)
        self._cache = None                  # (W64, W32, cond, Wd, mw), derived, never persisted as truth

    @classmethod
    def create(cls, cfg: G3Config | None = None, device='cuda'):
        return cls(cfg or G3Config(), device)

    # ------------------------------------------------------------ P5: persistence (spec §10.2)
    def _meta(self):
        import platform
        from dataclasses import asdict
        from pathlib import Path
        from nanm.state import ENCODER_IDENTITY, _sha
        src = {p.name: _sha(p.read_bytes()) for p in sorted(Path(__file__).parent.glob('*.py'))}
        dev = torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'cpu'
        return {'contract': {'sha256': self.cfg.contract_sha256(), 'fields': asdict(self.cfg)},
                'seed': self.cfg.seed, 'encoder': dict(ENCODER_IDENTITY), 'task_order': [t['label'] for t in
                                                                                      self.state.tasks],
                'ledger': self.state.ledger(),
                'provenance': {'nanm_sources_sha256': src, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                               'device': dev, 'python': platform.python_version(),
                               'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                               'threads': torch.get_num_threads()}}

    def save_state(self, root, _fault=None):
        """Publish the current state as a new immutable checkpoint generation under `root`; returns its name."""
        from nanm.state import publish
        if not self.state.keys:
            raise RuntimeError('nothing to save: no completed task')
        return publish(root, self.state, self._meta(), _fault)

    @classmethod
    def load_state(cls, root, cfg: G3Config | None = None, device='cuda', generation=None, encoder_sha256=None):
        """Load a committed generation (LATEST by default), verify it, and rebuild W and the inference cache."""
        from nanm.state import ENCODER_IDENTITY, read
        cfg = cfg or G3Config()
        state, man = read(root, generation, cfg.contract_sha256(),
                          encoder_sha256 or ENCODER_IDENTITY['checkpoint_sha256'])
        sysm = cls(cfg, device)
        W64, cond = sysm.geometry.geometry(state.S, state.n)
        W32 = W64.float()
        Wd, mw = sysm.reader.cache(state.union_keys(), W32, sysm.device)
        sysm.state, sysm._cache = state, (W64, W32, cond, Wd, mw)
        sysm.load_report = {'generation': man['generation'], 'cond_rebuilt_equals_saved':
                            cond == state.tasks[-1]['cond_T']}
        return sysm

    @torch.no_grad()
    def fit_task(self, tokens, task_label=None, k=None, provenance=None):
        """tokens: [n_img, 784, d] fp32 normal FIT tokens of ONE task, in the caller's registered image order."""
        if tokens.ndim != 3 or tokens.shape[1:] != (self.cfg.grid ** 2, self.cfg.dim):
            raise ValueError('tokens must be [n_img, 784, d]')
        fit16 = self.writer.prepare(tokens)
        rule = 'ceil(ratio*n)' if k is None else 'explicit'
        k = quota(self.cfg.ratio, len(fit16)) if k is None else int(k)
        idx = self.writer.write(fit16, k)
        S_t, n_t = self.geometry.task_statistic(fit16, idx.to(self.device))
        keys_t = fit16[idx.to(self.device)].cpu()
        S = (0 if self.state.S is None else self.state.S) + S_t          # task-order sum, as the oracle's sum()
        n = self.state.n + n_t
        W64, cond = self.geometry.geometry(S, n)
        W32 = W64.float()
        keys = self.state.keys + (keys_t,)
        Wd, mw = self.reader.cache(torch.cat(keys), W32, self.device)
        new = ContinualState(dim=self.cfg.dim, keys=keys, selected=self.state.selected + (idx,), S=S, n=n,
                             tasks=self.state.tasks + ({'label': task_label, 'n_fit_tokens': len(fit16), 'k': k,
                                                        'quota_rule': rule, 'n_res': n_t, 'cond_T': cond,
                                                        'provenance': provenance or {}},))
        self.state, self._cache = new, (W64, W32, cond, Wd, mw)          # single commit point
        return {'stage': new.stage, 'k': k, 'n_res': n_t, 'cond_T': cond}

    @torch.no_grad()
    def predict(self, tokens):
        """tokens: [N, 784, d] fp32 -> energy grids [N, 28, 28], 392 maps, image scores. Task-free; state unchanged."""
        if self._cache is None:
            raise RuntimeError('empty memory: call fit_task first')
        g = self.cfg.grid
        q16 = unit(tokens.reshape(-1, self.cfg.dim)).half()
        e = self.reader.energies(q16, self._cache[3], self._cache[4])
        grid = e.reshape(len(tokens), g, g)
        maps, score = self.readout(grid)
        return {'energy': grid, 'map392': maps, 'image_score': score}

    @property
    def W(self):
        return None if self._cache is None else self._cache[1]
