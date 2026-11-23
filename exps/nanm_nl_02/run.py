"""NANM-NL-02 Stage 0 driver. Registration: agents/tasks/task_nanm_nl_02_stage0.md (frozen; hash-locked).

    source docs/recovery/recovery_env.sh
    python -B -m exps.nanm_nl_02.run audit     # selection, roles, custody, storage; SELECTED tensors are never decoded
    python -B -m exps.nanm_nl_02.run phase0a   # DESIGN-only: mechanical checks, U, hyperparameters -> DESIGN_SELECTION.json
    python -B -m exps.nanm_nl_02.run lock      # immutable registration / source / data hash lock (create-exclusive)
    python -B -m exps.nanm_nl_02.run phase0b   # refuses unless PHASE0B_AUTHORIZED exists and the lock verifies

Writes only results/hope_cad/nanm_nl_02/ (E:). Reads the NANM-REPRO-01 train caches on K: read-only (mmap).
Canonical G3 (nanm/, exps/bench_g3) is called, never modified; historical G3 states are never opened.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from exps.nanm_nl_02 import core as C
from exps.nanm_r0 import audit as A
from exps.nanm_r2b import run as B
from nanm import g3

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/hope_cad/nanm_nl_02'
CACHE = Path('/mnt/k/NLCAD/cache/nanm_repro_01')
SRC = {'mpdd': CACHE / 'bench_g3/features', 'visa': CACHE / 'vbench_00/features'}
TASKFILE = ROOT / 'agents/tasks/task_nanm_nl_02_stage0.md'
CODE = [ROOT / 'exps/nanm_nl_02/run.py', ROOT / 'exps/nanm_nl_02/core.py', ROOT / 'exps/nanm_nl_02/test_core.py']
ENCODER = ROOT / 'checkpoints/dinov2/dinov2_vitb14_reg4_pretrain.pth'
ENCODER_SHA = '73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71'
VHDX = Path('/mnt/d/wsl/Ubuntu_new/ext4.vhdx')
GiB = 2 ** 30
SEEDS = (2711, 2712, 2713)
PERMS = {'a': (0, 1, 2, 3), 'b': (3, 2, 1, 0), 'c': (1, 3, 0, 2)}
ROLE_SPEC = (('FIT', 16, 784), ('FEEDBACK', 4, 8), ('RESERVE', 4, 8), ('EVAL', 8, 256))
N_FIT = 16 * 784
K_KEYS = g3.task_size(N_FIT)                                   # 126 (canonical 1 %)
STEPS, BATCH, KPROJ = 200, 256, 16
ETA_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
EWC_GRID = (0.1, 1.0, 10.0, 100.0)
EMA_GRID = (0.9, 0.99, 0.999)
F64 = torch.float64
OPENED = []                                                    # every tensor file opened (M2 allowlist audit)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def fsha(p, buf=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(buf):
            h.update(b)
    return h.hexdigest()


def wjson(p, obj, exclusive=False):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'x' if exclusive else 'w') as f:
        f.write(json.dumps(obj, indent=1, default=str))


def storage():
    return {'D_free': shutil.disk_usage('/mnt/d').free, 'E_free': shutil.disk_usage('/mnt/e').free,
            'K_free': shutil.disk_usage('/mnt/k').free, 'VHDX': VHDX.stat().st_size, 'time': time.strftime('%FT%T')}


# ------------------------------------------------------------------ selection and roles
def pool():
    out = [('mpdd', p.name[5:-9]) for p in sorted(SRC['mpdd'].glob('mpdd_*_train.pt'))]
    return out + [('visa', p.name[:-9]) for p in sorted(SRC['visa'].glob('*_train.pt'))]


def fpath(ds, c):
    return SRC[ds] / (f'mpdd_{c}_train.pt' if ds == 'mpdd' else f'{c}_train.pt')


def selection():
    ranked = sorted(pool(), key=lambda t: C.sha256(f'NANM-NL-02/tasks/v1|{t[0]}/{t[1]}'))
    return [f'{d}/{c}' for d, c in ranked[:4]], [f'{d}/{c}' for d, c in ranked[4:8]]


def roles(paths):
    order = sorted(paths, key=lambda p: C.sha256('NANM-NL-02/roles/v1|' + p))
    out, k = {}, 0
    for r, n, _ in ROLE_SPEC:
        out[r] = order[k:k + n]; k += n
    return out


def patch_idx(p, m):
    if m == 784:
        return list(range(784))
    return sorted(range(784), key=lambda j: C.sha256(f'NANM-NL-02/patch/v1|{p}|{j}'))[:m]


def load_task(task, pos, want=('FIT', 'FEEDBACK', 'RESERVE', 'EVAL'), feat_override=None):
    """Role tensors of one task (unit fp16, G3 convention). Image ids are global: pos * 1000 + role-rank."""
    ds, c = task.split('/')
    p = fpath(ds, c); OPENED.append(str(p))
    d = torch.load(p, mmap=True, map_location='cpu', weights_only=False)
    paths = [r['relative_path'] for r in d['rows']]
    feats = d['features'] if feat_override is None else feat_override(paths, d['features'])
    index = {q: i for i, q in enumerate(paths)}
    rl = roles(paths); out = {'n_train': len(paths)}; rank = 0
    for r, n, m in ROLE_SPEC:
        if r not in want:
            rank += n; continue
        vs, ims = [], []
        for q in rl[r]:
            idx = patch_idx(q, m)
            vs.append(A.unit(feats[index[q]][idx].float()).half()); ims.append(torch.full((len(idx),), pos * 1000 + rank))
            rank += 1
        out[r] = torch.cat(vs); out[r + '_img'] = torch.cat(ims)
    return out


# ------------------------------------------------------------------ stream (one order x seed)
class Stream:
    def __init__(self, data, seed, name):
        self.data, self.seed, self.name = data, seed, name
        self.states, self.W, self.ldW, self.Kv, self.Kimg, self.Wq, self.ldWq = [], [], [], [], [], [], []
        Sq, nq = torch.zeros(C.D, C.D, dtype=F64), 0
        for t, td in enumerate(data):
            st, kimg = C.g3_state_fit_only(td['FIT'], td['FIT_img'], seed, k=K_KEYS)
            self.states.append(st)
            W, _ = C.stage_W(self.states)
            self.W.append(W); self.ldW.append(float(torch.linalg.slogdet(W)[1]))
            prev = self.Kimg[-1] if self.Kimg else torch.empty(0, dtype=torch.long)
            self.Kv.append(torch.cat([s.keys.double() for s in self.states]))
            self.Kimg.append(torch.cat([prev, kimg]))
            # C0q: closed-form refit, leave-own-image-out assignment among K_{<=t} under W_t (A = I), additive
            S_t, n_t = self._refit_stat(t)
            Sq += S_t; nq += n_t
            Wq, _ = B.whiten(Sq, nq)
            Wq = Wq.float().double()
            self.Wq.append(Wq); self.ldWq.append(float(torch.linalg.slogdet(Wq)[1]))
        self.W2 = [W @ W for W in self.W]; self.trW2 = [torch.trace(x) for x in self.W2]
        self.Zk = [K @ W for K, W in zip(self.Kv, self.W)]

    def _refit_stat(self, t, chunk=2048):
        v = self.data[t]['FIT'].double(); img = self.data[t]['FIT_img']; W = self.W[t]
        Zk = self.Kv[t] @ W; S = torch.zeros(C.D, C.D, dtype=F64)
        for i in range(0, len(v), chunk):
            d = C.R2.sqdist(v[i:i + chunk] @ W, Zk)
            d.masked_fill_(img[i:i + chunk][:, None] == self.Kimg[t][None], float('inf'))
            r = v[i:i + chunk] - self.Kv[t][d.argmin(1)]
            S += r.T @ r
        return S, len(v)

    def nll(self, t, j, Lmap, ld, role='EVAL'):
        return float(C.nll_mixture(self.data[j][role].double(), self.Kv[t], Lmap, ld).mean())


def metric(stream, t, arm, phi=None):
    if arm == 'C0':
        return stream.W[t], stream.ldW[t]
    if arm == 'C0q':
        return stream.Wq[t], stream.ldWq[t]
    return C.Lmap_phi(stream.W[t], *phi)


# ------------------------------------------------------------------ fast-level training
def reserve_alloc(arm, n_seen):
    spare = 3 * C.PHI_B - C.EXTRA[arm]
    return min(spare // C.PROBE_B, 32 * n_seen) if arm in ('C2', 'C4', 'C5') else 0


def replay_sets(stream, t, arm):
    """Old-task probe sets (v fp64, img) used in the replay term of `arm` at stage t (tasks j < t)."""
    sets = []
    alloc = reserve_alloc(arm, t + 1); taken = 0
    for j in range(t + 1):
        res = stream.data[j]['RESERVE']; k = max(0, min(32, alloc - taken)); taken += k
        if j == t:
            continue
        parts_v, parts_i = [], []
        if arm == 'C2':
            parts_v.append(stream.data[j]['FEEDBACK']); parts_i.append(stream.data[j]['FEEDBACK_img'])
        if k:
            parts_v.append(res[:k]); parts_i.append(stream.data[j]['RESERVE_img'][:k])
        if parts_v:
            sets.append((torch.cat(parts_v).double(), torch.cat(parts_i)))
    return sets


def batch_gen(stream, t, tag=''):
    return torch.Generator().manual_seed(int(C.sha256(f'NANM-NL-02/batch|{stream.name}|{stream.seed}|{t}|{tag}'), 16) % 2 ** 62)


def loss_terms(stream, t, P, Q, groups):
    """groups: list of (v fp64, img) per contributing task; macro weights. Returns (loss, assignment)."""
    W = stream.W[t]
    vq = torch.cat([g[0] for g in groups]); img = torch.cat([g[1] for g in groups])
    w = torch.cat([torch.full((len(g[0]),), 1.0 / (len(groups) * len(g[0])), dtype=F64) for g in groups])
    return C.fast_loss_w(P, Q, vq, vq @ W, img, w, stream.Kv[t], stream.Zk[t], stream.Kimg[t], stream.W2[t], stream.trW2[t])


def probe_grads_sq(stream, t, P, Q, probes):
    """Per-probe squared gradients (empirical Fisher diagonal) of the single-query objective."""
    v, img = probes; FP = torch.zeros_like(P); FQ = torch.zeros_like(Q)
    for i in range(len(v)):
        Pg = P.detach().requires_grad_(True); Qg = Q.detach().requires_grad_(True)
        L, _ = loss_terms(stream, t, Pg, Qg, [(v[i:i + 1], img[i:i + 1])])
        gP, gQ = torch.autograd.grad(L, (Pg, Qg))
        FP += gP ** 2; FQ += gQ ** 2
    return FP / len(v), FQ / len(v)


def proj_basis(stream, t, P, Q):
    vs = [stream.data[j]['FEEDBACK'].double() for j in range(t)]
    ims = [stream.data[j]['FEEDBACK_img'] for j in range(t)]
    v = torch.cat(vs); img = torch.cat(ims); W = stream.W[t]
    zq = v @ W; Zk = stream.Kv[t] @ W
    with torch.no_grad():
        d = C.R2.sqdist(zq + (zq @ Q) @ P.T, Zk + (Zk @ Q) @ P.T)
        d.masked_fill_(img[:, None] == stream.Kimg[t][None], float('inf'))
        R = zq - Zk[d.argmin(1)]
    U, _, _ = torch.linalg.svd(R.T, full_matrices=False)
    return U[:, :KPROJ]


class Aux:
    def __init__(self):
        self.F = None; self.anchor = None; self.ema = None; self.basis = None


def train_stage(stream, t, arm, phi, aux, hp, steps=None, diag=None):
    """One stage of fast updates for arm in {C1..C5}. Returns new phi (and mutates aux)."""
    steps = STEPS if steps is None else steps
    P, Q = phi
    fit = stream.data[t]['FIT']; fimg = stream.data[t]['FIT_img']
    reps = replay_sets(stream, t, arm) if arm in ('C2', 'C4', 'C5') else []
    if arm == 'C4' and t >= 1:
        aux.basis = proj_basis(stream, t, P, Q)
    if arm == 'C5' and aux.ema is None:
        aux.ema = (P.clone(), Q.clone())
    lam = None
    if arm == 'C3' and aux.F is not None:
        lam = hp['ewc_c'] / float(torch.cat([aux.F[0].flatten(), aux.F[1].flatten()]).mean().clamp_min(1e-30))
    g = batch_gen(stream, t)
    for s in range(steps):
        idx = torch.randperm(N_FIT, generator=g)[:BATCH]
        Pg = P.detach().requires_grad_(True); Qg = Q.detach().requires_grad_(True)
        L, _ = loss_terms(stream, t, Pg, Qg, [(fit[idx].double(), fimg[idx])] + reps)
        if lam is not None:
            L = L + 0.5 * lam * ((aux.F[0] * (Pg - aux.anchor[0]) ** 2).sum() + (aux.F[1] * (Qg - aux.anchor[1]) ** 2).sum())
        gP, gQ = torch.autograd.grad(L, (Pg, Qg))
        if diag is not None and s == 0:
            diag.setdefault('first_grad_Q', []).append(float(gQ.norm())); diag.setdefault('first_grad_P', []).append(float(gP.norm()))
            diag.setdefault('first_loss', []).append(float(L.detach()))
        if arm == 'C4' and aux.basis is not None:
            gP = gP - aux.basis @ (aux.basis.T @ gP); gQ = gQ - aux.basis @ (aux.basis.T @ gQ)
        P, Q, h = C.sgd_step(P, Q, gP, gQ, hp['eta'])
        if diag is not None:
            diag['steps'] = diag.get('steps', 0) + 1; diag['rejected'] = diag.get('rejected', 0) + (h < 0)
        if arm == 'C5':
            dcy = hp['ema']; aux.ema = (dcy * aux.ema[0] + (1 - dcy) * P, dcy * aux.ema[1] + (1 - dcy) * Q)
    if diag is not None:
        with torch.no_grad():
            idx = torch.randperm(N_FIT, generator=batch_gen(stream, t, 'final'))[:BATCH]
            diag.setdefault('last_loss', []).append(float(loss_terms(stream, t, P, Q, [(fit[idx].double(), fimg[idx])])[0]))
    if arm == 'C3':
        FP, FQ = probe_grads_sq(stream, t, P, Q, (stream.data[t]['FEEDBACK'].double(), stream.data[t]['FEEDBACK_img']))
        aux.F = (FP, FQ) if aux.F is None else (aux.F[0] + FP, aux.F[1] + FQ)
        aux.anchor = (P.clone(), Q.clone())
    return (P, Q)


def scoring_phi(arm, phi, aux):
    return aux.ema if arm == 'C5' else phi


def run_traj(stream, arm, hp, role='EVAL', diag=None):
    """Full deployable trajectory. Returns NLL matrix ell[t][j] (j <= t) and per-stage phis."""
    ell, phis = [], []
    phi = C.init_phi(stream.seed); aux = Aux()
    for t in range(len(stream.data)):
        if arm in ('C0', 'C0q'):
            Lm, ld = metric(stream, t, arm)
        else:
            phi = train_stage(stream, t, arm, phi, aux, hp, diag=diag)
            sp = scoring_phi(arm, phi, aux)
            s, _ = C.logdet_A(*sp)
            if diag is not None and float(s) <= 0:
                diag['nonpositive_det'] = diag.get('nonpositive_det', 0) + 1
            Lm, ld = metric(stream, t, arm, sp)
        ell.append([stream.nll(t, j, Lm, ld, role) for j in range(t + 1)])
        phis.append(phi)
    return ell, phis


def run_R1(stream, hp, role='EVAL'):
    ell = []
    for t in range(len(stream.data)):
        P, Q = C.init_phi(stream.seed); g = batch_gen(stream, t, 'R1'); per = BATCH // (t + 1)
        for s in range(STEPS * (t + 1)):
            groups = []
            for j in range(t + 1):
                idx = torch.randperm(N_FIT, generator=g)[:per]
                groups.append((stream.data[j]['FIT'][idx].double(), stream.data[j]['FIT_img'][idx]))
            Pg = P.detach().requires_grad_(True); Qg = Q.detach().requires_grad_(True)
            L, _ = loss_terms(stream, t, Pg, Qg, groups)
            gP, gQ = torch.autograd.grad(L, (Pg, Qg))
            P, Q, _ = C.sgd_step(P, Q, gP, gQ, hp['eta'])
        Lm, ld = metric(stream, t, 'R', (P, Q))
        ell.append([stream.nll(t, j, Lm, ld, role) for j in range(t + 1)])
    return ell


def macro_all(ell):
    return [sum(r) / len(r) for r in ell]


def criterion(ell):
    """Selection criterion: mean over stages of the macro NLL over seen tasks. Non-finite -> +inf (invalid, registered)."""
    a = macro_all(ell)
    v = sum(a) / len(a)
    return v if math.isfinite(v) else math.inf


# ------------------------------------------------------------------ gates (pure; unit-tested)
def gate_g0a(cells, U):
    """cells: list of dicts with ACQ[arm] = [ACQ_t for t=0..3] for arms C0, C0q, C1 (R1 entries are ignored)."""
    d_acq = [sum(min(c['ACQ']['C0'][t], c['ACQ']['C0q'][t]) - c['ACQ']['C1'][t] for t in range(4)) / 4 for c in cells]
    d_c0 = [sum(c['ACQ']['C0'][t] - c['ACQ']['C1'][t] for t in range(4)) / 4 for c in cells]
    m = sum(d_acq) / len(cells); pos = sum(x > 0 for x in d_acq)
    ok = m >= 0.10 * U and pos >= 7
    out = 'PASS' if ok else ('NL02_QUADRATIC_SUFFICIENT' if sum(d_c0) / len(cells) >= 0.10 * U else 'NL02_NO_ADAPTATION_VALUE')
    return {'pass': ok, 'outcome': out, 'mean9_delta_acq': m, 'cells_positive': pos, 'delta_acq_cells': d_acq,
            'mean9_delta_c0': sum(d_c0) / len(cells), 'threshold': 0.10 * U}


def gate_g0b(cells, U):
    """cells: dicts with damage[6], rec[X][6], acqcs_gap[X] = mean_t(ACQcs_t(X) - ACQcs_t(C1)) for X in C2..C5."""
    D_c = [sum(c['damage']) / len(c['damage']) for c in cells]; D = sum(D_c) / len(cells)
    X_ok = [x for x in ('C2', 'C3', 'C4', 'C5') if sum(c['acqcs_gap'][x] for c in cells) / len(cells) <= 0.02 * U]
    rec = {x: sum(sum(c['rec'][x]) / len(c['rec'][x]) for c in cells) / len(cells) for x in ('C2', 'C3', 'C4', 'C5')}
    xs = max(X_ok, key=lambda x: (rec[x], -('C2', 'C3', 'C4', 'C5').index(x))) if X_ok else None
    Rec = rec[xs] if xs else 0.0
    un_c = [D_c[i] - (sum(cells[i]['rec'][xs]) / len(cells[i]['rec'][xs]) if xs else 0.0) for i in range(len(cells))]
    ok = D >= 0.10 * U and D - Rec >= 0.05 * U and sum(u > 0 for u in un_c) >= 7
    out = 'PASS' if ok else ('NL02_NO_INTERFERENCE' if D < 0.10 * U else 'NL02_CONVENTIONAL_SUFFICIENT')
    return {'pass': ok, 'outcome': out, 'D': D, 'Rec': Rec, 'X_star': xs, 'guarded': X_ok, 'rec_mean': rec,
            'unrecovered_cells_positive': sum(u > 0 for u in un_c)}


def stage0_verdict(g0_material_fail, ga, gb):
    if g0_material_fail:
        return 'NL02_PROTOCOL_BLOCKED'
    if not ga['pass']:
        return ga['outcome']
    if not gb['pass']:
        return gb['outcome']
    return 'NL02_STAGE0_PASS'


# ------------------------------------------------------------------ phases
def audit():
    sel, des = selection()
    rec = {'storage_before': storage(), 'selected': sel, 'design': des,
           'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           'registration_sha256_at_audit': fsha(TASKFILE), 'code_sha256_at_audit': {p.name: fsha(p) for p in CODE},
           'encoder_checkpoint_sha256': fsha(ENCODER)}
    rec['encoder_ok'] = rec['encoder_checkpoint_sha256'] == ENCODER_SHA
    files, problems, seen_path, seen_content = {}, [], {}, {}
    for grp, tasks in (('SELECTED', sel), ('DESIGN', des)):
        for task in tasks:
            ds, c = task.split('/'); p = fpath(ds, c)
            meta = json.loads(p.with_suffix('.json').read_text())
            paths = sorted(meta['source_images'])                    # metadata only; no tensor decoded here
            rl = roles(paths)
            if len(paths) < sum(n for _, n, _ in ROLE_SPEC):
                problems.append(f'{task}: {len(paths)} images')
            for r, ps in rl.items():
                for q in ps:
                    ch = meta['source_images'][q]
                    if q in seen_path:
                        problems.append(f'path reuse {q}')
                    if ch in seen_content:
                        problems.append(f'content duplicate {q} ~ {seen_content[ch]}')
                    seen_path[q] = (grp, task, r); seen_content[ch] = (grp, task, r, q)
            h = fsha(p)
            files[task] = {'group': grp, 'file': str(p), 'sha256': h, 'json_sha256': meta['sha256'], 'equal': h == meta['sha256'],
                           'json_file_sha256': fsha(p.with_suffix('.json')), 'encoder_sha256': meta.get('encoder_sha256'),
                           'encoder_ok': meta.get('encoder_sha256') == ENCODER_SHA, 'n_train_images': len(paths),
                           'roles': rl}
            log('custody', task, grp, h == meta['sha256'], meta.get('encoder_sha256') == ENCODER_SHA, len(paths))
    rec.update(files=files, roles_problems=problems, custody_pass=all(f['equal'] and f['encoder_ok'] for f in files.values()),
               storage_ok=rec['storage_before']['D_free'] >= 30 * GiB)
    rec['pass'] = rec['custody_pass'] and rec['encoder_ok'] and not problems and rec['storage_ok']
    wjson(RUN / 'AUDIT.json', rec)
    log('AUDIT', 'PASS' if rec['pass'] else 'FAIL', problems[:5])


def design_streams(au, feat_override=None):
    des = au['design']
    data = {t: load_task(t, i, feat_override=feat_override) for i, t in enumerate(des)}
    return {(n, s): Stream([data[des[i]] for i in perm], s, n) for n, perm in PERMS.items() for s in SEEDS}, data


def phase0a():
    au = json.loads((RUN / 'AUDIT.json').read_text())
    assert au['pass'], 'audit failed'
    assert fsha(TASKFILE) == au['registration_sha256_at_audit'], 'registration changed after audit'
    assert {p.name: fsha(p) for p in CODE} == au['code_sha256_at_audit'], 'code changed after audit'
    assert not (RUN / 'DESIGN_SELECTION.json').exists(), 'Phase 0a already ran'
    st0 = storage(); t0 = time.time()
    streams, data = design_streams(au)
    log('DESIGN loaded', list(data))
    mech = {}
    # M4 noise replacement: rebuild DESIGN states with every non-FIT image replaced by noise
    def noisy(paths, feats):
        fit = set().union(*[set(au['files'][t]['roles']['FIT']) for t in au['design']])
        class F:
            def __getitem__(self, i):
                return feats[i] if paths[i] in fit else torch.randn(feats[i].shape, generator=torch.Generator().manual_seed(i))
        return F()
    nd = {t: load_task(t, i, want=('FIT',), feat_override=noisy) for i, t in enumerate(au['design'])}
    ok = True
    for (n, s), stm in streams.items():
        if s != SEEDS[0]:
            continue
        for k, task in enumerate([au['design'][i] for i in PERMS[n]]):
            st2, _ = C.g3_state_fit_only(nd[task]['FIT'], nd[task]['FIT_img'], s, k=K_KEYS)
            ok &= torch.equal(st2.kc, stm.states[k].kc) and torch.equal(st2.S, stm.states[k].S) and st2.n == stm.states[k].n
    mech['M4_noise_replacement_invariant'] = ok
    mech['M2_opened_files'] = sorted(set(OPENED))
    mech['M2_only_allowed'] = all(Path(p).name.endswith('_train.pt') and str(CACHE) in p for p in OPENED) and \
        not any('normal' in p or 'bench_g3/normal' in p for p in OPENED)
    # M5 C0 == zero-step C1 on DESIGN HOLDOUT (canonical scorer)
    stm = streams[('a', SEEDS[0])]
    P0, Q0 = C.init_phi(SEEDS[0])
    q16 = stm.data[0]['EVAL'][:512]
    mech['M5_zero_step_identity'] = torch.equal(C.A_of(P0, Q0), torch.eye(C.D, dtype=F64)) and torch.equal(
        g3.score(q16, stm.Kv[0].float(), stm.W[0].float()), g3.score(q16, stm.Kv[0].float(), C.Lmap_phi(stm.W[0], P0, Q0)[0].float()))
    # M11 reference-measure invariance on real DESIGN data
    gB = torch.Generator().manual_seed(11)
    Bm = torch.eye(C.D, dtype=F64) + 0.02 * torch.randn(C.D, C.D, generator=gB, dtype=F64)
    Binv = torch.linalg.inv(Bm); ldB = float(torch.linalg.slogdet(Bm)[1])
    v = stm.data[0]['EVAL'][:256].double(); K = stm.Kv[0]
    diffs = []
    for Lm, ld in ((stm.W[0], stm.ldW[0]), (torch.eye(C.D, dtype=F64) / math.sqrt(C.iso_sigma2(stm.states[0].S, stm.states[0].n)),
                                            -0.5 * C.D * math.log(C.iso_sigma2(stm.states[0].S, stm.states[0].n)))):
        a = C.nll_mixture(v, K, Lm, ld); b = C.nll_mixture(v @ Bm, K @ Bm, Binv @ Lm, ld - ldB)
        diffs.append(float((b - a - ldB).abs().max()))
    mech['M11_reference_measure_max_abs_err'] = max(diffs); mech['M11_ok'] = max(diffs) < 1e-6
    # U on DESIGN only (single-task states, HOLDOUT role)
    u_vals = []
    for i, task in enumerate(au['design']):
        td = data[task]
        for s in SEEDS:
            st, _ = C.g3_state_fit_only(td['FIT'], td['FIT_img'], s, k=K_KEYS)
            W, _ = C.stage_W([st]); s2 = C.iso_sigma2(st.S, st.n); v = td['EVAL'].double(); K = st.keys.double()
            Li, li = C.maps_for(W, sigma2=s2)
            u_vals.append(float((C.nll_mixture(v, K, Li, li) - C.nll_mixture(v, K, W, float(torch.linalg.slogdet(W)[1]))).mean()))
    U, U_raw, bound = C.effect_scale(u_vals)
    log('U', U, 'raw', U_raw, 'floor binds', bound)
    # hyperparameter selection (DESIGN only)
    sel = {'eta': {}, 'ewc_c': {}, 'ema': {}}; diag_c1 = {}
    for eta in ETA_GRID:
        dg = {}
        sel['eta'][eta] = sum(criterion(run_traj(stm_, 'C1', {'eta': eta}, role='EVAL', diag=dg)[0]) for stm_ in streams.values()) / 9
        diag_c1[eta] = dg; log('eta', eta, sel['eta'][eta], 'rejected', dg.get('rejected'), '/', dg.get('steps'))
    eta = min(ETA_GRID, key=lambda e: (sel['eta'][e], ETA_GRID.index(e)))
    assert math.isfinite(sel['eta'][eta]), 'NL02_PROTOCOL_BLOCKED: every eta diverged on DESIGN'
    for cc in EWC_GRID:
        sel['ewc_c'][cc] = sum(criterion(run_traj(stm_, 'C3', {'eta': eta, 'ewc_c': cc})[0]) for stm_ in streams.values()) / 9
        log('ewc', cc, sel['ewc_c'][cc])
    for dc in EMA_GRID:
        sel['ema'][dc] = sum(criterion(run_traj(stm_, 'C5', {'eta': eta, 'ema': dc})[0]) for stm_ in streams.values()) / 9
        log('ema', dc, sel['ema'][dc])
    ewc_c = min(EWC_GRID, key=lambda e: (sel['ewc_c'][e], EWC_GRID.index(e)))
    ema = min(EMA_GRID, key=lambda e: (sel['ema'][e], EMA_GRID.index(e)))
    c0 = sum(criterion(run_traj(stm_, 'C0', {})[0]) for stm_ in streams.values()) / 9
    c0q = sum(criterion(run_traj(stm_, 'C0q', {})[0]) for stm_ in streams.values()) / 9
    # M6 / M7 / M9 / M8 on DESIGN with the selected eta
    dsel = diag_c1[eta]
    mech['M6_first_grad_Q_min'] = min(dsel['first_grad_Q']); mech['M6_ok'] = mech['M6_first_grad_Q_min'] > 0
    mech['M6_first_grad_P_at_init_max'] = max(dsel['first_grad_P'][::4])
    mech['M7_rejected_frac'] = dsel.get('rejected', 0) / dsel['steps']; mech['M7_nonpositive_det'] = dsel.get('nonpositive_det', 0)
    mech['M7_ok'] = mech['M7_rejected_frac'] <= 0.01 and mech['M7_nonpositive_det'] == 0
    e1, _ = run_traj(stm, 'C1', {'eta': eta}); e2, _ = run_traj(stm, 'C1', {'eta': eta})
    mech['M9_determinism'] = e1 == e2
    # M8 future-noise invariance: perturb tasks >= 2 of stream a, check stages 0..1 unchanged (C2 uses replay)
    stm_n = Stream([stm.data[0], stm.data[1]] + [{**stm.data[k], 'FIT': A.unit(torch.randn(N_FIT, C.D, generator=torch.Generator().manual_seed(k))).half()}
                                                 for k in (2, 3)], stm.seed, stm.name)
    _, ph1 = run_traj(stm, 'C2', {'eta': eta}); _, ph2 = run_traj(stm_n, 'C2', {'eta': eta})
    mech['M8_future_invariance'] = all(torch.equal(ph1[t][0], ph2[t][0]) and torch.equal(ph1[t][1], ph2[t][1]) for t in (0, 1))
    mech['M8_eval_structural'] = 'train_stage/loss_terms receive FIT, FEEDBACK, RESERVE only'
    # M10 bytes
    budget = lambda n: n * K_KEYS * (C.KEY_B + C.KEY_ID_B) + C.S_B + C.META_B + 4 * n + 3 * C.PHI_B + 32 * n * C.PROBE_B
    m10 = {}
    for arm in ('C1', 'C2', 'C3', 'C4', 'C5'):
        ok = True
        for n in range(1, 5):
            fb = 32 * n if arm in ('C2', 'C3', 'C4') else 0
            b = C.ledger(arm, n, [K_KEYS] * n, fb, reserve_alloc(arm, n))['total_charged']
            ok &= b <= budget(n)
        m10[arm] = ok
    mech['M10_bytes'] = m10
    st1 = storage()
    mech['M12_storage'] = {'D_free_min_ok': min(st0['D_free'], st1['D_free']) >= 30 * GiB, 'vhdx_growth': st1['VHDX'] - st0['VHDX']}
    mech['M1_custody'] = au['custody_pass']; mech['M3_roles_disjoint'] = not au['roles_problems']
    flags = [mech['M1_custody'], mech['M2_only_allowed'], mech['M3_roles_disjoint'], mech['M4_noise_replacement_invariant'],
             mech['M5_zero_step_identity'], mech['M6_ok'], mech['M7_ok'], mech['M8_future_invariance'], mech['M9_determinism'],
             all(m10.values()), mech['M11_ok'], mech['M12_storage']['D_free_min_ok'] and mech['M12_storage']['vhdx_growth'] <= 0]
    out = {'U': U, 'U_raw': U_raw, 'U_floor': C.U_FLOOR, 'U_floor_binds': bound, 'u_values': u_vals,
           'selected': {'eta': eta, 'ewc_c': ewc_c, 'ema': ema}, 'criteria': {k: {str(a): b for a, b in v.items()} for k, v in sel.items()},
           'reference_criteria': {'C0': c0, 'C0q': c0q},
           'c1_learning': {str(e): {'first_loss_mean': sum(d['first_loss']) / len(d['first_loss']),
                                    'last_loss_mean': sum(d['last_loss']) / len(d['last_loss'])} for e, d in diag_c1.items()},
           'mechanical': mech, 'mechanical_pass': all(flags), 'storage_before': st0, 'storage_after': st1,
           'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'seconds': round(time.time() - t0, 1),
           'selected_eval_accessed': False}
    wjson(RUN / 'DESIGN_SELECTION.json', out, exclusive=True)
    log('PHASE0A', 'mechanical', 'PASS' if out['mechanical_pass'] else 'FAIL', out['selected'], 'U', U)


def lock():
    au = json.loads((RUN / 'AUDIT.json').read_text()); ds = RUN / 'DESIGN_SELECTION.json'
    assert ds.exists(), 'Phase 0a has not produced DESIGN_SELECTION.json'
    assert fsha(TASKFILE) == au['registration_sha256_at_audit'], 'registration changed after audit'
    assert {p.name: fsha(p) for p in CODE} == au['code_sha256_at_audit'], 'code changed after audit'
    rec = {'registration': {str(TASKFILE.relative_to(ROOT)): fsha(TASKFILE)},
           'source': {str(p.relative_to(ROOT)): fsha(p) for p in CODE},
           'canonical_g3': {f: fsha(ROOT / f) for f in ('nanm/g3.py', 'exps/nanm_r0/audit.py', 'exps/nanm_r2/run.py', 'exps/nanm_r2b/run.py',
                                                        'exps/nanm_r1/run.py')},
           'audit': fsha(RUN / 'AUDIT.json'), 'design_selection': fsha(ds),
           'data': {t: {'pt': f['sha256'], 'json': f['json_file_sha256']} for t, f in au['files'].items()},
           'encoder_checkpoint': au['encoder_checkpoint_sha256'], 'time': time.strftime('%FT%T'),
           'note': 'frozen after DESIGN-only Phase 0a and before any SELECTED tensor is decoded'}
    wjson(RUN / 'PREREG_LOCK.json', rec, exclusive=True)
    log('LOCK written')


def phase0b():
    assert (RUN / 'PHASE0B_AUTHORIZED').exists(), 'Phase 0b is not authorized'
    dsel = json.loads((RUN / 'DESIGN_SELECTION.json').read_text())
    st0 = storage(); out = run_phase0b()
    out['verdict'] = stage0_verdict(not dsel['mechanical_pass'], out['G0a'], out['G0b'])
    out['storage_before'], out['storage_after'] = st0, storage()
    wjson(RUN / 'RESULT.json', out, exclusive=True)
    log('PHASE0B', out['verdict'])


def run_phase0b(selected_data_loader=None):
    """Phase 0b (SELECTED). Verifies the lock, runs all arms, decomposition and gates. Not invoked before GO."""
    lk = json.loads((RUN / 'PREREG_LOCK.json').read_text())
    for f, h in {**lk['registration'], **lk['source'], **lk['canonical_g3']}.items():
        assert fsha(ROOT / f) == h, f'lock mismatch {f}'
    au = json.loads((RUN / 'AUDIT.json').read_text()); dsel = json.loads((RUN / 'DESIGN_SELECTION.json').read_text())
    hp = {**dsel['selected']}; U = dsel['U']
    load = selected_data_loader or (lambda: {t: load_task(t, i) for i, t in enumerate(au['selected'])})
    data = load()
    return evaluate_selected(data, au['selected'], hp, U)


def evaluate_selected(data, sel, hp, U):
    cells, g0a_cells, g0b_cells = {}, [], []
    for n, perm in PERMS.items():
        for s in SEEDS:
            stm = Stream([data[sel[i]] for i in perm], s, n); res = {}
            for arm in ('C0', 'C0q', 'C1', 'C2', 'C3', 'C4', 'C5'):
                res[arm], phis = run_traj(stm, arm, hp)
                if arm == 'C1':
                    c1_phis = phis
            res['R1'] = run_R1(stm, hp)
            damage, rec, acq_gap, geo, orc = [], {x: [] for x in ('C2', 'C3', 'C4', 'C5')}, {x: [] for x in ('C2', 'C3', 'C4', 'C5')}, [], []
            for t in range(1, 4):
                start = c1_phis[t - 1]
                Lb, lb = metric(stm, t - 1, 'phi', start); Lg, lg = metric(stm, t, 'phi', start)
                Ln, ln = metric(stm, t, 'phi', c1_phis[t])
                ctrl = {}
                for x in ('C2', 'C3', 'C4', 'C5'):
                    aux = Aux()
                    if x == 'C3':
                        F = None
                        for j in range(t):
                            FP, FQ = probe_grads_sq(stm, t, *start, (stm.data[j]['FEEDBACK'].double(), stm.data[j]['FEEDBACK_img']))
                            F = (FP, FQ) if F is None else (F[0] + FP, F[1] + FQ)
                        aux.F, aux.anchor = F, start
                    if x == 'C5':
                        aux.ema = start
                    ph = train_stage(stm, t, x, start, aux, hp)
                    ctrl[x] = metric(stm, t, 'phi', scoring_phi(x, ph, aux))
                acq_n = stm.nll(t, t, Ln, ln)
                for x in ctrl:
                    acq_gap[x].append(stm.nll(t, t, *ctrl[x]) - acq_n)
                for j in range(t):
                    before = stm.nll(t - 1, j, Lb, lb); g_ = stm.nll(t, j, Lg, lg); nv = stm.nll(t, j, Ln, ln)
                    damage.append(nv - g_); geo.append(g_ - before)
                    for x in ctrl:
                        rec[x].append(nv - stm.nll(t, j, *ctrl[x]))
                    phR2 = train_stage_task(stm, t, j, C.init_phi(s), hp)
                    orc.append(g_ - stm.nll(t, j, *metric(stm, t, 'phi', phR2)))
            acq = {a: [res[a][t][t] for t in range(4)] for a in ('C0', 'C0q', 'C1')}
            g0a_cells.append({'ACQ': acq})
            g0b_cells.append({'damage': damage, 'rec': rec, 'acqcs_gap': {x: sum(v) / len(v) for x, v in acq_gap.items()}})
            cells[f'{n}|{s}'] = {'ell': res, 'damage': damage, 'geometry': geo, 'oracle': orc, 'rec': rec, 'acqcs_gap': acq_gap}
            log('cell', n, s)
    ga, gb = gate_g0a(g0a_cells, U), gate_g0b(g0b_cells, U)
    r1 = sum(min(sum(c['ell']['C0'][3]) / 4, sum(c['ell']['C0q'][3]) / 4) - sum(c['ell']['R1'][3]) / 4 for c in cells.values()) / 9
    return {'cells': cells, 'G0a': ga, 'G0b': gb, 'R1_headroom_diagnostic': r1}


def train_stage_task(stm, t, j, phi, hp):
    """R2: phi trained on task j's FIT under W_t, K_<=t (privileged diagnostic)."""
    P, Q = phi; g = batch_gen(stm, t, f'R2|{j}')
    for s in range(STEPS):
        idx = torch.randperm(N_FIT, generator=g)[:BATCH]
        Pg = P.detach().requires_grad_(True); Qg = Q.detach().requires_grad_(True)
        L, _ = loss_terms(stm, t, Pg, Qg, [(stm.data[j]['FIT'][idx].double(), stm.data[j]['FIT_img'][idx])])
        gP, gQ = torch.autograd.grad(L, (Pg, Qg))
        P, Q, _ = C.sgd_step(P, Q, gP, gQ, hp['eta'])
    return (P, Q)


if __name__ == '__main__':
    torch.set_grad_enabled(True)
    {'audit': audit, 'phase0a': phase0a, 'lock': lock, 'phase0b': phase0b}[sys.argv[1]]()
