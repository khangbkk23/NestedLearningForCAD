"""NL03-MEMORY-PRESSURE-00 core (registration: agents/tasks/task_nl03_pressure_00_registration.md).

Diagnostic only. Every numerical primitive is the canonical oracle's own function object, imported unchanged:
  k-centre A.greedy_coreset | W exps.nanm_eval_01.arms.w_g3 / w_g1 | energy G.whitened / G.euclid | tokens G.load_test
Per-task material = stored canonical normal artifacts (REPRO-01 for MPDD/VisA, EVAL-01 A1 for MVTec), hash-checked by
G.load_norm. No arm reads FIT tokens; the only new token reads are the coverage queries (CREDIT role, §5).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from exps.nanm_eval_01 import common as U

ROOT = U.ROOT
OUT = ROOT / 'results/hope_cad/nl03_pressure_00'
CACHE = Path('/mnt/k/NLCAD/cache/nl03_pressure_00')
D = U.D
MiB = U.MiB
KEY_B = U.KEY_B                                    # 1,536 (fp16 x 768)
SLOW_B = {'g3': U.S_B, 'g1': U.G1_B, 'none': 0}    # 2,359,304 / 2,362,376 / 0
EVICT_META_B = 0                                   # k-centre: no metadata; FIFO: array order only
BUDGETS_MIB = (4, 8, 16)
PRIMARY_MIB = 8
ARMS = ('U', 'Q', 'E', 'E4', 'F', 'G', 'FIFO')
ARM_SLOW = {'U': 'g3', 'Q': 'g3', 'E': 'g3', 'E4': 'g3', 'FIFO': 'g3', 'F': 'none', 'G': 'g1'}
T2_ARMS = ('E', 'Q', 'F', 'E4')
SEEDS = (2711, 2712, 2713)
NL02_SELECTED = ('visa/pcb3', 'visa/macaroni2', 'visa/pcb4', 'mpdd/bracket_brown')
COVER_IMGS = 8
STREAMS = ('visa', 's33')


# ------------------------------------------------------------------ byte accounting (single shared function)
class BudgetViolation(AssertionError):
    pass


def bytes_of(n_keys, slow):
    return int(n_keys) * KEY_B + SLOW_B[slow] + EVICT_META_B


def cap_keys(budget, slow):
    """Largest key count with bytes_of(n, slow) <= budget (None = unbounded)."""
    if budget is None:
        return None
    return (budget - SLOW_B[slow] - EVICT_META_B) // KEY_B


def assert_budget(arm, stage, n_keys, slow, budget):
    b = bytes_of(n_keys, slow)
    if budget is not None and b > budget:
        raise BudgetViolation(f'BUDGET_VIOLATION arm={arm} stage={stage} keys={n_keys} bytes={b} > {budget}')
    return b


# ------------------------------------------------------------------ binding (per-dataset oracle paths)
_PATHS = {}


def bind():
    """EVAL-01 A1 bind (REPRO-01 bind + MVTec dataset dispatch), then remember both path roots."""
    from exps.nanm_eval_01 import mvtec as MV
    from exps.nanm_repro_01 import run as RP
    G = MV.mv_bind()
    if not _PATHS:
        _PATHS['mvtec'] = (MV.G_OUT, MV.G_CACHE)
        _PATHS['repro'] = (RP.OUT, RP.CACHE / 'bench_g3')
    return G


def use(G, ds):
    out, cache = _PATHS['mvtec'] if ds == 'mvtec' else _PATHS['repro']
    G.OUT, G.CACHE = out, cache
    G.FEAT, G.NORM, G.SCORE = cache / 'features', cache / 'normal', cache / 'scores'
    return G


def nrec(G, ds):
    use(G, ds)
    return {ds: U.rj(G.OUT / f'normal_{ds}.json')}


def extract_rec(G, ds):
    use(G, ds)
    return U.rj(G.OUT / 'EXTRACT_COMPLETE.json')


# ------------------------------------------------------------------ streams (registered)
def ssha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def base_order(stream):
    from exps.bench_g3 import run as G0
    from exps.calm02 import data as CD
    if stream == 'visa':
        return [f'visa/{c}' for c in CD.VISA_ORDER]
    if stream == 'mpdd':
        return [f'mpdd/{c}' for c in G0.MPDD_ORDER]
    if stream == 's33':
        return ([f'mvtec/{c}' for c in CD.MVTEC_ORDER] + [f'visa/{c}' for c in CD.VISA_ORDER]
                + [f'mpdd/{c}' for c in G0.MPDD_ORDER])
    raise KeyError(stream)


def orders(stream):
    base = base_order(stream)
    if stream in ('visa', 'mpdd'):                      # exactly EVAL-01's U.orders on the bare task names
        ds = stream
        od = U.orders([t.split('/')[1] for t in base])
        return {k: [f'{ds}/{c}' for c in v] for k, v in od.items()}
    return {'a': list(base), 'b': list(reversed(base)),
            'c': sorted(base, key=lambda t: ssha(f'NL03-PRESSURE-00/order-c/v1|{t}'))}


def logged_stages(stream, T):
    if stream == 's33':
        return [l for l in range(T) if (l + 1) % 3 == 0 or l == T - 1]
    return list(range(T))


def cat_index(stream, task):
    """Registered bootstrap stream index: VisA -> index in CD.VISA_ORDER (as EVAL-01); others -> index in order a."""
    return base_order(stream).index(task) if stream != 'visa' else base_order('visa').index(task)


# ------------------------------------------------------------------ per-task canonical material
def load_material(G, tasks, seed):
    """Canonical stored states: 1 % coreset keys (k-centre order), S1/n1, Sx/s1/n. Returns dict task -> state."""
    out, recs = {}, {}
    for t in tasks:
        ds, c = t.split('/')
        if ds not in recs:
            recs[ds] = nrec(G, ds)
        use(G, ds)
        rec = recs[ds]
        k1 = rec[ds]['sizes']['g3'][list(G.order(ds)).index(c)]
        st = G.load_norm(ds, seed, c, rec)
        assert len(st['keys']) >= k1, (t, len(st['keys']), k1)
        out[t] = {'keys': st['keys'][:k1].contiguous(), 'S1': st['S1'], 'n1': st['n1'], 'Sx': st['Sx'],
                  's1': st['s1'], 'n': st['n'], 'k1': k1}
        del st
    return out


class Pool:
    """All canonical coresets concatenated in stream order. Every bounded memory is a set of pool indices."""

    def __init__(self, order, mat, device='cuda'):
        self.order = list(order)
        sizes = [mat[t]['k1'] for t in order]
        self.off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.keys = torch.cat([mat[t]['keys'] for t in order]).to(device)
        self.task = torch.cat([torch.full((s,), i, dtype=torch.long) for i, s in enumerate(sizes)])
        self.rank = torch.cat([torch.arange(s) for s in sizes])

    def ids(self, l):
        return torch.arange(int(self.off[l]), int(self.off[l + 1]))

    def union(self, l):
        return torch.arange(0, int(self.off[l + 1]))


# ------------------------------------------------------------------ eviction rules
def kc_select(cand, cap, pool, seed):
    """Greedy k-centre (canonical A.greedy_coreset, seed s) over candidate key vectors; survivors keep insertion order.
    |cand| <= cap (or cap None) -> keep all (no k-centre call; gives B = infinity parity)."""
    from exps.nanm_r0 import audit as A
    if cap is None or len(cand) <= cap:
        return cand
    idx = A.greedy_coreset(pool.keys[cand.to(pool.keys.device)], cap, seed=seed)
    return cand[idx.sort().values]


def fifo_select(K, C, cap):
    """Drop oldest keys first; if the new coreset alone fills the cap, keep its k-centre prefix C[:cap]."""
    if cap is None or len(K) + len(C) <= cap:
        return torch.cat([K, C])
    if len(C) >= cap:
        return C[:cap]
    return torch.cat([K[len(K) - (cap - len(C)):], C])


def run_arm(arm, pool, seed, budget, cap_override=None):
    """Memory trajectory of one arm. Returns (K per stage [list of CPU long tensors], eviction log list).
    E and E4 share the trajectory (rule 'E'); F/G use the E rule at their own caps."""
    slow = ARM_SLOW[arm]
    cap = cap_override if cap_override is not None else cap_keys(budget, slow)
    rule = {'U': 'U', 'Q': 'Q', 'E': 'E', 'E4': 'E', 'F': 'E', 'G': 'E', 'FIFO': 'FIFO'}[arm]
    if rule == 'U':
        cap = None
    T = len(pool.order)
    K = torch.empty(0, dtype=torch.long)
    traj, logs = [], []
    ins_stage = torch.full((int(pool.off[-1]),), -1, dtype=torch.long)
    inserted_cum = evicted_cum = 0
    for l in range(T):
        C = pool.ids(l)
        ins_stage[C] = l
        if rule == 'Q':
            cand = pool.union(l)
            Knew = kc_select(cand, cap, pool, seed)
        elif rule == 'FIFO':
            cand = torch.cat([K, C])
            Knew = fifo_select(K, C, cap)
        else:
            cand = torch.cat([K, C])
            Knew = kc_select(cand, cap, pool, seed)
        b = assert_budget(arm, l, len(Knew), slow, budget if rule != 'U' else None)
        keep = torch.zeros(int(pool.off[-1]), dtype=torch.bool); keep[Knew] = True
        ev = cand[~keep[cand]]
        inserted_cum += len(C)
        if rule == 'Q':                                 # hindsight: dropped relative to the seen union
            evicted_cum = len(ev)
            inv = len(Knew) + len(ev) == int(pool.off[l + 1])
        else:
            evicted_cum += len(ev)
            inv = len(Knew) + evicted_cum == inserted_cum
        surv = torch.bincount(pool.task[Knew], minlength=T)[:l + 1]
        logs.append({'stage': l, 'task': pool.order[l], 'inserted': len(C), 'n_keys': len(Knew), 'bytes': b,
                     'evicted_ids': ev.numpy(), 'evicted_task': pool.task[ev].numpy(),
                     'evicted_age': (l - ins_stage[ev]).numpy(), 'survivors_per_task': surv.numpy(),
                     'invariant_ok': bool(inv and int(surv.sum()) == len(Knew) and len(set(Knew.tolist())) == len(Knew))})
        K = Knew
        traj.append(Knew.clone())
    return traj, logs


# ------------------------------------------------------------------ geometry per stage
def whiteners(order, mat):
    """W_acc (canonical pooled), W_cur (current task only, N4 analogue), W_g1 (G1b analogue) for every stage."""
    from exps.nanm_eval_01 import arms as AR
    st = lambda t: {'S1': mat[t]['S1'], 'n1': mat[t]['n1']}
    acc, cur, g1, cond = [], [], [], []
    for l in range(len(order)):
        Wa, ca = AR.w_g3([st(t) for t in order[:l + 1]])
        Wc, cc = AR.w_g3([st(order[l])])
        Wg, cg = AR.w_g1([{'n': mat[t]['n'], 's1': mat[t]['s1'], 'Sx': mat[t]['Sx']} for t in order[:l + 1]])
        acc.append(Wa); cur.append(Wc); g1.append(Wg); cond.append({'acc': ca, 'cur': cc, 'g1': cg})
    return {'acc': acc, 'cur': cur, 'g1': g1, 'cond': cond}


ARM_GEOM = {'U': 'acc', 'Q': 'acc', 'E': 'acc', 'FIFO': 'acc', 'E4': 'cur', 'G': 'g1', 'F': None}


@torch.no_grad()
def energy(G, q16, keys16, W):
    keys = keys16.float()
    return G.euclid(q16, keys) if W is None else G.whitened(q16, keys, W.cuda())


# ------------------------------------------------------------------ coverage queries (CREDIT role, NL-02 exclusion)
def nl02_excluded(G, task):
    """Paths NL-02 assigned to any role (ranks 1-32) in a SELECTED pool; computed from paths only."""
    if task not in NL02_SELECTED:
        return set()
    from exps.nanm_nl_02 import run as R
    ds, c = task.split('/')
    use(G, ds)
    paths = [r['relative_path'] for r in G.train_rows(ds, c)]
    rl = R.roles(paths)
    return {p for v in rl.values() for p in v}


def credit_rows(G, task):
    ds, c = task.split('/')
    use(G, ds)
    rows = G.train_rows(ds, c)
    excl = nl02_excluded(G, task)
    cr = [i for i in G.roles(ds, rows)['CREDIT'] if rows[i]['relative_path'] not in excl]
    return [rows[i]['relative_path'] for i in cr[:COVER_IMGS]]


@torch.no_grad()
def credit_tokens(G, task):
    from exps.nanm_r0 import audit as A
    ds, c = task.split('/')
    want = credit_rows(G, task)
    use(G, ds)
    data = G.load_train(ds, c)
    idx = {r['relative_path']: i for i, r in enumerate(data['rows'])}
    sel = torch.as_tensor([idx[p] for p in want])
    x = A.unit(data['features'][sel].reshape(-1, D).cuda()).half().cpu()
    del data
    return x, want


@torch.no_grad()
def coverage(G, q16, keys16):
    d = (2 * G.euclid(q16, keys16.float())).clamp_min(0).sqrt().numpy()
    return {'median': float(np.median(d)), 'p95': float(np.quantile(d, .95)), 'mean': float(d.mean())}
