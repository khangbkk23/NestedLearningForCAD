"""RIGEL-CC-01 support code (SPEC.md): the F4 control, the multi-arm stream, persistence under convention P and
restoration. Candidate arms are the frozen implementations of exps/rigel_q8/q8.py (V1, T16, Q8, Q8T, F8) and
exps/rigel_q8/d1.py (Q4T); nothing here changes how they write or score.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from exps.rigel_q8 import d1, q8

ARMS = ['V1', 'T16', 'Q8', 'Q8T', 'Q4T', 'F8', 'F4']
ELIGIBLE = ['T16', 'Q8', 'Q8T', 'Q4T']
CONTROL = {'T16': 'F', 'Q8': 'F8', 'Q8T': 'F8', 'Q4T': 'F4'}      # matched fast-only control (F not run; T16 none)
CAP_F4 = q8.BUDGET // d1.KEY4                                       # 21,732
KEYB = {'V1': q8.KEY16, 'T16': q8.KEY16, 'Q8': q8.KEY8, 'Q8T': q8.KEY8, 'Q4T': d1.KEY4, 'F8': q8.KEY8, 'F4': d1.KEY4}
SLOWB = {'V1': q8.S_FULL, 'T16': q8.S_TRI, 'Q8': q8.S_FULL, 'Q8T': q8.S_TRI, 'Q4T': q8.S_TRI, 'F8': 0, 'F4': 0}


class ArmF4(d1.ArmQ4):
    """int4 fast-only control: d1.ArmQ4 storage and write rule, no slow statistic (Euclidean scoring), cap B/386."""
    sfmt = None

    def __init__(self, name, seed, device='cuda'):
        super().__init__(name, seed, device)
        self.cap = CAP_F4

    def payload(self):
        return len(self.task) * d1.KEY4

    def update_S(self, S_t):
        pass

    def state_dict(self):
        return {'q4': d1.pack4(self.q).cpu(), 'scale': self.scale.cpu(), 'task': self.task.cpu()}


def make_arm(a, seed, device='cuda'):
    if a == 'Q4T':
        return d1.ArmQ4(a, seed, device)
    if a == 'F4':
        return ArmF4(a, seed, device)
    return q8.Arm(a, seed, device)


class Stream(q8.Stream):
    def __init__(self, arms: List[str], seed: int, device: str = 'cuda'):
        self.arms = {a: make_arm(a, seed, device) for a in arms}
        self.device = device
        self.S, self.n = None, 0
        self.W_full: List[torch.Tensor] = []
        self.W_tri: Dict[str, List[torch.Tensor]] = {a: [] for a in arms if self.arms[a].sfmt == 'tri'}
        self.snap: Dict[str, list] = {a: [] for a in arms}
        self.cond: List[float] = []


def logical_bytes(a: str, n_keys: int) -> int:
    return n_keys * KEYB[a] + SLOWB[a]


# ---------------------------------------------------------------------------------------------- convention P
def persist(st: Stream, a: str) -> Dict[str, torch.Tensor]:
    """Tensors needed to restore arm a (task-id provenance excluded)."""
    m = st.arms[a]
    n = torch.tensor(st.n, dtype=torch.int64)
    if a == 'V1':
        return {'k16': m.k16.cpu(), 'S64': st.S.cpu(), 'n': n}
    if a == 'T16':
        return {'k16': m.k16.cpu(), 'S_tri': m.S_tri.cpu(), 'n': n}
    if a == 'Q8':
        return {'q': m.q.cpu(), 'scale': m.scale.cpu(), 'S32': st.S.float().cpu(), 'n': n}
    if a == 'Q8T':
        return {'q': m.q.cpu(), 'scale': m.scale.cpu(), 'S_tri': m.S_tri.cpu(), 'n': n}
    if a == 'Q4T':
        return {'q4': d1.pack4(m.q).cpu(), 'scale': m.scale.cpu(), 'S_tri': m.S_tri.cpu(), 'n': n}
    if a == 'F8':
        return {'q': m.q.cpu(), 'scale': m.scale.cpu()}
    if a == 'F4':
        return {'q4': d1.pack4(m.q).cpu(), 'scale': m.scale.cpu()}
    raise KeyError(a)


def payload_bytes(sd: Dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in sd.values())


def restore(a: str, sd: Dict[str, torch.Tensor], seed: int, device: str = 'cuda') -> Stream:
    """A one-arm stream whose state equals the persisted arm, ready to continue with Stream.task."""
    st = Stream([a], seed, device)
    m = st.arms[a]
    dev = lambda t: t.to(device)
    if 'k16' in sd:
        m.k16 = dev(sd['k16'])
        m.keys = m.k16.float()
    elif 'q' in sd:
        m.q, m.scale = dev(sd['q']), dev(sd['scale'])
        m.keys = q8.dequantize(m.q, m.scale)
    else:
        m.q, m.scale = dev(d1.unpack4(sd['q4'])), dev(sd['scale'])
        m.keys = q8.dequantize(m.q, m.scale)
    m.task = torch.zeros(len(m.keys), dtype=torch.long, device=device)       # provenance not persisted
    if 'S_tri' in sd:
        m.S_tri = dev(sd['S_tri'])
        st.S = q8.tri_unpack(sd['S_tri'])                                    # only W_full uses it (unused by tri arms)
    if 'S64' in sd:
        st.S = sd['S64'].clone()
    if 'S32' in sd:
        st.S = sd['S32'].double()
    if 'n' in sd:
        st.n = int(sd['n'])
    return st
