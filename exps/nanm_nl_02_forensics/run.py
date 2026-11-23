"""NANM-NL-02 DESIGN-only objective forensics.

Protocol (frozen before any forensic number): agents/tasks/task_nanm_nl_02_objective_forensics.md.
Imports the locked Stage 0 code (exps/nanm_nl_02/{run,core}.py) read-only; never writes into results/hope_cad/nanm_nl_02.
Only the 4 DESIGN caches are opened (asserted). SELECTED tasks are never loaded.

Phases:  check | task1 | task2z | task2s | task3 | align | summary
"""
from __future__ import annotations

import hashlib
import json
import math
import resource
import sys
import time

import torch

from exps.nanm_nl_02 import core as C
from exps.nanm_nl_02 import run as R

OUT = R.ROOT / 'results/hope_cad/nanm_nl_02_forensics'
PROTO = R.ROOT / 'agents/tasks/task_nanm_nl_02_objective_forensics.md'
F64 = torch.float64
D, K, ALPHA, N_FIT = C.D, R.K_KEYS, C.ALPHA, R.N_FIT
LOG2PI = math.log(2 * math.pi)
OBJS = ('A', 'B', 'C', 'Cp')
ETA_S = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
torch.set_num_threads(12)


def log(*a):
    R.log(*a)


def seed_of(tag):
    return int(hashlib.sha256(('NANM-NL-02/forensics|' + tag).encode()).hexdigest(), 16) % 2 ** 62


# ------------------------------------------------------------------ likelihood pieces
def sqd(a, b):
    return (a.pow(2).sum(1, keepdim=True) + b.pow(2).sum(1)[None] - 2 * a @ b.T).clamp_min(0)


def mapz(z, P, Q):
    return z + (z @ Q) @ P.T


def mix(zq, zk, P, Q, ldW, mask=None):
    """Per-query proper mixture NLL (identical to core.nll_mixture with Lmap = W A^T); mask: True = key excluded,
    the normaliser then counts the remaining keys only (a proper equal-weight mixture over them)."""
    d = sqd(mapz(zq, P, Q), mapz(zk, P, Q))
    if mask is not None:
        d = d.masked_fill(mask, float('inf'))
        lk = torch.log((~mask).sum(1).to(F64))
    else:
        lk = math.log(zk.shape[0])
    _, la = C.logdet_A(P, Q)
    return -torch.logsumexp(-0.5 * d, 1) + lk - ldW - la + 0.5 * D * LOG2PI


def mixE(zq, zk, A, ldW, mask=None):
    """Same, with a full matrix A (for gradient alignment in E-space, A = I + E)."""
    d = sqd(zq @ A.T, zk @ A.T)
    if mask is not None:
        d = d.masked_fill(mask, float('inf'))
        lk = torch.log((~mask).sum(1).to(F64))
    else:
        lk = math.log(zk.shape[0])
    return -torch.logsumexp(-0.5 * d, 1) + lk - ldW - torch.linalg.slogdet(A)[1] + 0.5 * D * LOG2PI


def pq(P, Q, vq, zq, j, Kv, Zk, W2, trW2):
    """Registered penalised quadratic (core.fast_loss_w, uniform weights) with a GIVEN assignment j."""
    rz = zq - Zk[j]
    rv2 = ((vq - Kv[j]) ** 2).sum(1)
    rzA = rz + (rz @ Q) @ P.T
    quad = 0.5 * (1 - ALPHA) * (rzA ** 2).sum(1).mean()
    c = ALPHA * rv2.mean() / D
    trA = trW2 + 2 * (P * (W2 @ Q)).sum() + ((Q.T @ W2 @ Q) * (P.T @ P)).sum()
    _, la = C.logdet_A(P, Q)
    return quad + 0.5 * c * trA - la


def pqE(A, vq, zq, j, Kv, Zk, W):
    rz = zq - Zk[j]
    c = ALPHA * ((vq - Kv[j]) ** 2).sum(1).mean() / D
    return 0.5 * (1 - ALPHA) * ((rz @ A.T) ** 2).sum(1).mean() + 0.5 * c * ((W @ A.T) ** 2).sum() - torch.linalg.slogdet(A)[1]


# ------------------------------------------------------------------ per (cell, stage) context
class Ctx:
    def __init__(self, stm, t):
        self.stm, self.t = stm, t
        self.W, self.ldW, self.Kv, self.Zk = stm.W[t], stm.ldW[t], stm.Kv[t], stm.Zk[t]
        self.kimg, self.W2, self.trW2 = stm.Kimg[t], stm.W2[t], stm.trW2[t]
        td = stm.data[t]; st = stm.states[t]
        self.fit16, self.fimg = td['FIT'], td['FIT_img']
        self.fit = self.fit16.double(); self.zf = self.fit @ self.W
        self.hv = td['EVAL'].double(); self.himg = td['EVAL_img']; self.zh = self.hv @ self.W
        self.member = torch.zeros(N_FIT, dtype=torch.bool); self.member[st.kc[:K]] = True
        assert torch.equal(self.Kv[K * t:K * (t + 1)], st.keys.double()), 'task-t keys not where expected'
        # G3 assignment exactly as B.residual_stats: fp32 raw Euclidean, task-t keys, non-member queries
        x = self.fit16.float(); kk = st.keys.float(); jB = torch.full((N_FIT,), -1, dtype=torch.long)
        for i in range(0, N_FIT, 2048):
            jB[i:i + 2048] = R.C.R2.sqdist(x[i:i + 2048], kk).argmin(1) + K * t
        jB[self.member] = -1; self.jB = jB
        nm = ~self.member
        rv = self.fit[nm] - self.Kv[jB[nm]]
        S_re = rv.T @ rv
        self.g3_S_relerr = float((S_re - st.S.double()).norm() / st.S.double().norm())
        assert self.g3_S_relerr < 1e-4 and int(nm.sum()) == st.n, ('G3 assignment mismatch', self.g3_S_relerr)
        I0 = torch.zeros(D, C.RANK, dtype=F64)
        self.c0 = float(mix(self.zh, self.Zk, I0, I0, self.ldW).mean())
        ref = stm.nll(t, t, self.W, self.ldW)
        assert abs(self.c0 - ref) < 1e-6, ('holdout NLL mismatch', self.c0, ref)
        self.c0q = stm.nll(t, t, stm.Wq[t], stm.ldWq[t])
        self.ref = min(self.c0, self.c0q)
        self.task = None

    def hold(self, P, Q, idx=None):
        zq = self.zh if idx is None else self.zh[idx]
        return mix(zq, self.Zk, P, Q, self.ldW).mean()

    def obj(self, name, P, Q, idx):
        """FIT-only diagnostic objectives on the FIT-token batch idx."""
        if name == 'A':                                   # registered fast loss, exactly as train_stage builds it
            return R.loss_terms(self.stm, self.t, P, Q, [(self.fit16[idx].double(), self.fimg[idx])])[0]
        if name == 'C':
            return mix(self.zf[idx], self.Zk, P, Q, self.ldW, self.fimg[idx][:, None] == self.kimg[None]).mean()
        idx = idx[~self.member[idx]]
        if name == 'B':
            return pq(P, Q, self.fit[idx], self.zf[idx], self.jB[idx], self.Kv, self.Zk, self.W2, self.trW2)
        if name == 'Cp':
            return mix(self.zf[idx], self.Zk, P, Q, self.ldW).mean()
        raise KeyError(name)


def phi_stats(P, Q):
    A = C.A_of(P, Q)
    sv = torch.linalg.svdvals(A)
    s, la = C.logdet_A(P, Q)
    return {'A_minus_I_F': float(torch.sqrt(((P.T @ P) * (Q.T @ Q)).sum().clamp_min(0))),
            'logdet_A': float(la), 'det_sign': float(s), 'cond_A': float(sv[0] / sv[-1]),
            'finite': bool(torch.isfinite(P).all() and torch.isfinite(Q).all())}


def cosv(a, b):
    a = torch.cat([x.flatten() for x in a]); b = torch.cat([x.flatten() for x in b])
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-300))


# ------------------------------------------------------------------ setup
def setup():
    au = json.loads((R.RUN / 'AUDIT.json').read_text())
    sel = json.loads((R.RUN / 'DESIGN_SELECTION.json').read_text())
    U = float(sel['U']) if 'U' in sel else float(sel['effect_scale']['U'])
    streams, data = R.design_streams(au)
    allowed = {str(R.fpath(*t.split('/'))) for t in au['design']}
    assert set(R.OPENED) <= allowed, ('non-DESIGN file opened', set(R.OPENED) - allowed)
    ctxs = {}
    for (n, s), stm in streams.items():
        for t in range(4):
            cx = Ctx(stm, t); cx.task = au['design'][R.PERMS[n][t]]; ctxs[(n, s, t)] = cx
    return au, U, streams, ctxs


def opened_ok(au):
    allowed = {str(R.fpath(*t.split('/'))) for t in au['design']}
    return sorted(set(R.OPENED)), set(R.OPENED) <= allowed


def leaf(P, Q):
    return P.clone().requires_grad_(True), Q.clone().requires_grad_(True)


def adam_fit(loss_fn, seed, steps, lr, trace_every=10):
    P, Q = leaf(*C.init_phi(seed))
    opt = torch.optim.Adam([P, Q], lr=lr)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    tr = []
    for s in range(steps):
        L = loss_fn(P, Q, s)
        opt.zero_grad(); L.backward()
        if s % trace_every == 0 or s == steps - 1:
            tr.append((s, float(L.detach())))
        opt.step(); sch.step()
    P, Q = P.detach(), Q.detach()
    Pg, Qg = leaf(P, Q); Lf = loss_fn(Pg, Qg, steps); gP, gQ = torch.autograd.grad(Lf, (Pg, Qg))
    gn = float(torch.sqrt(gP.pow(2).sum() + gQ.pow(2).sum()))
    return P, Q, tr, float(Lf.detach()), gn


def conv_stats(tr, steps):
    last = [v for s, v in tr if s >= int(0.9 * steps)]
    return {'loss_first': tr[0][1], 'loss_final': tr[-1][1], 'last10pct_change': last[-1] - last[0]}


# ------------------------------------------------------------------ Task 1: capacity reference (privileged)
def task1_one(cx, fold, seed, steps=800, lr=3e-3):
    ids = sorted(set(cx.himg.tolist()))
    trm = torch.isin(cx.himg, torch.tensor(ids[:4] if fold == 0 else ids[4:])); tem = ~trm
    tri, tei = trm.nonzero().squeeze(1), tem.nonzero().squeeze(1)
    P, Q, tr, Lf, gn = adam_fit(lambda P, Q, s: cx.hold(P, Q, tri), seed, steps, lr)
    I0 = torch.zeros(D, C.RANK, dtype=F64)
    with torch.no_grad():
        c0_te, c0_tr = float(cx.hold(I0, I0, tei)), float(cx.hold(I0, I0, tri))
        a_te, a_tr = float(cx.hold(P, Q, tei)), float(cx.hold(P, Q, tri))
    # full-rank closed-form context: whitener of the fit-half nearest-key residuals (raw Euclidean, like G3)
    v = cx.hv[tri]; r = v - cx.Kv[C.R2.sqdist(v, cx.Kv).argmin(1)]
    Wo = R.B.whiten(r.T @ r, len(r))[0].double(); ldo = float(torch.linalg.slogdet(Wo)[1])
    fr_te = float(C.nll_mixture(cx.hv[tei], cx.Kv, Wo, ldo).mean())
    c0q_te = float(C.nll_mixture(cx.hv[tei], cx.Kv, cx.stm.Wq[cx.t], cx.stm.ldWq[cx.t]).mean())
    return {'fold': fold, 'seed': seed, 'steps': steps, 'gain_crossfit': c0_te - a_te, 'gain_insample': c0_tr - a_tr,
            'gain_crossfit_vs_ref': min(c0_te, c0q_te) - a_te, 'fullrank_oracle_gain_crossfit': c0_te - fr_te,
            'final_grad_norm': gn, **conv_stats(tr, steps), **phi_stats(P, Q), 'trace': tr[::8]}


def task1(ctxs):
    out = {}
    for (n, s, t), cx in ctxs.items():
        out[f'{n}|{s}|{t}'] = [task1_one(cx, f, s) for f in (0, 1)]
        log('task1', n, s, t, cx.task, 'gain cf %.2f %.2f' % tuple(r['gain_crossfit'] for r in out[f'{n}|{s}|{t}']))
    rob = {}
    for t in range(4):
        cx = ctxs[('a', R.SEEDS[0], t)]
        rob[t] = {'seeds': [task1_one(cx, f, R.SEEDS[0] + o) for o in (0, 101, 202) for f in (0, 1)],
                  'long1600': [task1_one(cx, f, R.SEEDS[0], steps=1600) for f in (0, 1)]}
        log('task1 robustness t', t, [round(r['gain_crossfit'], 2) for r in rob[t]['seeds']],
            'long', [round(r['gain_crossfit'], 2) for r in rob[t]['long1600']])
    return {'cells': out, 'robustness_a2711': rob}


# ------------------------------------------------------------------ Task 2: objective alignment (FIT only)
def closed_forms(cx):
    stm, t = cx.stm, cx.t
    Sa, na = stm._refit_stat(t)                               # task-t leave-own-image-out (metric) residuals under W_t
    Wa = R.B.whiten(Sa, na)[0].double()
    st = stm.states[t]; Wb = R.B.whiten(st.S, st.n)[0].double()  # task-t G3 whitener
    f = lambda Wx: cx.c0 - stm.nll(t, t, Wx, float(torch.linalg.slogdet(Wx)[1]))
    return {'fullrank_A_opt_gain': f(Wa), 'fullrank_B_opt_gain': f(Wb)}


def task2z_one(cx, name, steps=600, batch=1024, lr=3e-3):
    g = torch.Generator().manual_seed(seed_of(f'z|{cx.stm.name}|{cx.stm.seed}|{cx.t}'))
    batches = [torch.randperm(N_FIT, generator=g)[:batch] for _ in range(steps + 1)]
    P, Q, tr, Lf, gn = adam_fit(lambda P, Q, s: cx.obj(name, P, Q, batches[s]), cx.stm.seed, steps, lr)
    with torch.no_grad():
        h = float(cx.hold(P, Q))
    return {'gain': cx.c0 - h, 'gain_vs_ref': cx.ref - h, 'final_grad_norm': gn, **conv_stats(tr, steps),
            **phi_stats(P, Q), 'trace': tr[::6]}


def task2z(ctxs):
    out = {}
    for (n, s, t), cx in ctxs.items():
        rec = {o: task2z_one(cx, o) for o in OBJS}; rec.update(closed_forms(cx)); rec['task'] = cx.task
        rec['c0'], rec['c0q'] = cx.c0, cx.c0q
        out[f'{n}|{s}|{t}'] = rec
        log('task2z', n, s, t, cx.task, {o: round(rec[o]['gain'], 2) for o in OBJS},
            'FR A %.2f B %.2f' % (rec['fullrank_A_opt_gain'], rec['fullrank_B_opt_gain']))
    return out


def sgd_run(cx, name, eta, steps=200, batch=256, every=20):
    g = torch.Generator().manual_seed(seed_of(f's|{cx.stm.name}|{cx.stm.seed}|{cx.t}'))
    P, Q = C.init_phi(cx.stm.seed); curve = []; rej = 0; fl = []
    for s in range(steps + 1):
        if s % every == 0:
            with torch.no_grad():
                curve.append((s, cx.c0 - float(cx.hold(P, Q))))
        if s == steps:
            break
        idx = torch.randperm(N_FIT, generator=g)[:batch]
        Pg, Qg = leaf(P, Q); L = cx.obj(name, Pg, Qg, idx); gP, gQ = torch.autograd.grad(L, (Pg, Qg))
        fl.append(float(L.detach()))
        P, Q, h = C.sgd_step(P, Q, gP, gQ, eta); rej += h < 0
        if not (torch.isfinite(P).all() and torch.isfinite(Q).all()):
            return {'gain': float('nan'), 'curve': curve, 'rejected': rej, 'finite': False, 'diverged_at': s}
    ok = math.isfinite(curve[-1][1])
    return {'gain': curve[-1][1], 'gain_vs_ref': curve[-1][1] - (cx.c0 - cx.ref), 'curve': curve, 'rejected': rej,
            'finite': ok, 'fit_loss_first20': sum(fl[:20]) / 20, 'fit_loss_last20': sum(fl[-20:]) / 20}


def task2s(ctxs):
    out = {}
    for (n, s, t), cx in ctxs.items():
        out[f'{n}|{s}|{t}'] = {o: {str(e): sgd_run(cx, o, e) for e in ETA_S} for o in OBJS}
        log('task2s', n, s, t, {o: [round(out[f'{n}|{s}|{t}'][o][str(e)]['gain'], 2) for e in ETA_S] for o in OBJS})
    return out


# ------------------------------------------------------------------ Task 3: registered C1 trajectory audit
def big_batch(cx):
    g = torch.Generator().manual_seed(seed_of(f'big|{cx.stm.name}|{cx.stm.seed}|{cx.t}'))
    out = []
    for im in sorted(set(cx.fimg.tolist())):
        ii = (cx.fimg == im).nonzero().squeeze(1)
        out.append(ii[torch.randperm(len(ii), generator=g)[:128]])
    return torch.cat(out)


def audit_traj(stm, ctx_of, eta, every=20):
    P, Q = C.init_phi(stm.seed); stages = []
    for t in range(4):
        cx = ctx_of[t]; bb = big_batch(cx); g = R.batch_gen(stm, t)
        fit16, fimg = stm.data[t]['FIT'], stm.data[t]['FIT_img']
        rec = {'task': cx.task, 'c0': cx.c0, 'c0q': cx.c0q, 'loss': [], 'checks': [], 'rejected': 0, 'halved': 0}
        for s in range(R.STEPS):
            idx = torch.randperm(N_FIT, generator=g)[:R.BATCH]
            Pg, Qg = leaf(P, Q)
            L, _ = R.loss_terms(stm, t, Pg, Qg, [(fit16[idx].double(), fimg[idx])])
            gP, gQ = torch.autograd.grad(L, (Pg, Qg)); rec['loss'].append(float(L.detach()))
            if s % every == 0:
                Ph, Qh = leaf(P, Q); H = cx.hold(Ph, Qh); hP, hQ = torch.autograd.grad(H, (Ph, Qh))
                Pb, Qb = leaf(P, Q); Lb = cx.obj('A', Pb, Qb, bb); bP, bQ = torch.autograd.grad(Lb, (Pb, Qb))
                rec['checks'].append({'step': s, 'hold_gain': cx.c0 - float(H.detach()), 'big_fit_loss': float(Lb.detach()),
                                      'cos_mb_hold': cosv((gP, gQ), (hP, hQ)), 'cos_big_hold': cosv((bP, bQ), (hP, hQ)),
                                      'cos_mb_big': cosv((gP, gQ), (bP, bQ)), **phi_stats(P, Q)})
            P, Q, h = C.sgd_step(P, Q, gP, gQ, eta)
            rec['rejected'] += h < 0; rec['halved'] += h > 0
        with torch.no_grad():
            rec['acq_gain'] = cx.c0 - float(cx.hold(P, Q)); rec['acq_gain_vs_ref'] = cx.ref - float(cx.hold(P, Q))
        rec['final'] = phi_stats(P, Q)
        stages.append(rec)
    return stages, (P, Q)


def task3(streams, ctxs):
    out = {}
    for (n, s), stm in streams.items():
        cof = {t: ctxs[(n, s, t)] for t in range(4)}
        out[f'{n}|{s}'] = {}
        for eta in (1e-4, 1e-3, 3e-3):
            st, phi = audit_traj(stm, cof, eta)
            out[f'{n}|{s}'][str(eta)] = st
            log('task3', n, s, eta, [round(r['acq_gain'], 2) for r in st],
                'cos_big_hold@0', [round(r['checks'][0]['cos_big_hold'], 3) for r in st])
            if (n, s) == ('a', R.SEEDS[0]):
                _, phis = R.run_traj(stm, 'C1', {'eta': eta})
                out[f'{n}|{s}'][str(eta) + '_repro_bitwise'] = bool(torch.equal(phis[-1][0], phi[0]) and torch.equal(phis[-1][1], phi[1]))
                log('task3 repro', eta, out[f'{n}|{s}'][str(eta) + '_repro_bitwise'])
    return out


def align(ctxs):
    out = {}
    for (n, s, t), cx in ctxs.items():
        def gE(fn):
            E = torch.zeros(D, D, dtype=F64, requires_grad=True)
            L = fn(torch.eye(D, dtype=F64) + E); (G,) = torch.autograd.grad(L, (E,)); return G
        I0 = torch.zeros(D, C.RANK, dtype=F64)
        with torch.no_grad():                                      # registered A assignment at A = I (metric = W)
            d = C.R2.sqdist(cx.zf, cx.Zk); d.masked_fill_(cx.fimg[:, None] == cx.kimg[None], float('inf')); jA = d.argmin(1)
        nm = ~cx.member
        G = {'hold': gE(lambda A: mixE(cx.zh, cx.Zk, A, cx.ldW).mean()),
             'A': gE(lambda A: pqE(A, cx.fit, cx.zf, jA, cx.Kv, cx.Zk, cx.W)),
             'B': gE(lambda A: pqE(A, cx.fit[nm], cx.zf[nm], cx.jB[nm], cx.Kv, cx.Zk, cx.W)),
             'C': gE(lambda A: mixE(cx.zf, cx.Zk, A, cx.ldW, cx.fimg[:, None] == cx.kimg[None]).mean()),
             'Cp': gE(lambda A: mixE(cx.zf[nm], cx.Zk, A, cx.ldW).mean())}
        # sanity: E-space hard-NN gradient equals the analytic (1-a) Sz + a c W^2 - I
        rz = cx.zf - cx.Zk[jA]; c = ALPHA * ((cx.fit - cx.Kv[jA]) ** 2).sum(1).mean() / D
        an = (1 - ALPHA) * rz.T @ rz / len(rz) + c * cx.W2 - torch.eye(D, dtype=F64)
        rec = {o: {'cos_hold': float((G[o] * G['hold']).sum() / (G[o].norm() * G['hold'].norm())),
                   'trace_dir': float(torch.trace(G[o])), 'norm': float(G[o].norm())} for o in OBJS}
        rec['hold'] = {'trace_dir': float(torch.trace(G['hold'])), 'norm': float(G['hold'].norm())}
        rec['analytic_A_relerr'] = float((an - G['A']).norm() / G['A'].norm())
        rec['task'] = cx.task
        out[f'{n}|{s}|{t}'] = rec
        log('align', n, s, t, cx.task, {o: round(rec[o]['cos_hold'], 3) for o in OBJS}, 'an err %.1e' % rec['analytic_A_relerr'])
    return out


# ------------------------------------------------------------------ check (smoke + timing, one cell)
def check(ctxs):
    cx = ctxs[('a', R.SEEDS[0], 1)]
    I0 = torch.zeros(D, C.RANK, dtype=F64); P, Q = C.init_phi(cx.stm.seed)
    idx = torch.randperm(N_FIT, generator=torch.Generator().manual_seed(0))[:256]
    # B with the A assignment must equal the registered loss (formula check)
    with torch.no_grad():
        LA = float(cx.obj('A', P, Q, idx))
        d = C.R2.sqdist(cx.zf[idx], cx.Zk); d.masked_fill_(cx.fimg[idx][:, None] == cx.kimg[None], float('inf'))
        Lpq = float(pq(P, Q, cx.fit[idx], cx.zf[idx], d.argmin(1), cx.Kv, cx.Zk, cx.W2, cx.trW2))
    log('check A == pq(jA)', LA, Lpq, abs(LA - Lpq))
    assert abs(LA - Lpq) < 1e-8
    out = {'A_eq_pq': abs(LA - Lpq), 'g3_S_relerr_max': max(c.g3_S_relerr for c in ctxs.values())}
    for name in OBJS:
        t0 = time.time(); r = task2z_one(cx, name, steps=60); out[f'z60_{name}'] = (time.time() - t0, r['gain'])
        log('check z60', name, '%.1fs gain %.3f' % out[f'z60_{name}'])
    t0 = time.time(); r = task1_one(cx, 0, cx.stm.seed, steps=80); out['t1_80'] = (time.time() - t0, r['gain_crossfit'])
    log('check t1 80 steps %.1fs gain %.3f' % out['t1_80'])
    t0 = time.time(); r = sgd_run(cx, 'C', 1e-3); out['s200_C'] = (time.time() - t0, r['gain'])
    log('check sgd200 C %.1fs gain %.3f' % out['s200_C'])
    return out


def main():
    ph = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    proto_sha = R.fsha(PROTO)
    lock_before = R.fsha(R.RUN / 'PREREG_LOCK.json')
    st0 = R.storage(); t0 = time.time()
    au, U, streams, ctxs = setup()
    log('setup', len(ctxs), 'contexts; U', U, 'gate', 0.1 * U)
    res = {'check': check, 'task1': task1, 'task2z': task2z, 'task2s': task2s, 'align': align,
           'task3': lambda c: task3(streams, c)}[ph](ctxs)
    files, ok = opened_ok(au)
    assert ok
    meta = {'phase': ph, 'protocol_sha256': proto_sha, 'prereg_lock_sha256': lock_before,
            'prereg_lock_unchanged': R.fsha(R.RUN / 'PREREG_LOCK.json') == lock_before,
            'code_sha256': R.fsha(R.ROOT / 'exps/nanm_nl_02_forensics/run.py'),
            'locked_code_sha256': {p.name: R.fsha(p) for p in R.CODE}, 'opened_files': files, 'selected_accessed': False,
            'U': U, 'gate': 0.1 * U, 'wall_s': time.time() - t0, 'storage_before': st0, 'storage_after': R.storage(),
            'peak_rss_GB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            'c0_c0q': {f'{n}|{s}|{t}': (cx.c0, cx.c0q, cx.task) for (n, s, t), cx in ctxs.items()}}
    R.wjson(OUT / f'{ph.upper()}.json', {'meta': meta, 'result': res}, exclusive=(ph != 'check'))
    log('done', ph, '%.0fs' % meta['wall_s'])


if __name__ == '__main__':
    main()
