"""CMS-B S3: task-boundary fast-to-slow consolidation (agents/reports/cmsb_s3_plan_2026_11_19.md §2).

Two separate rank-8 LoRA adapters on the frozen BASE decoder (never summed):
  slow theta_s : persistent, deployed (1 decoder pass); persistent StableAdamW O_s (NL §4.5)
  fast theta_f : per task; copied from theta_s at task start (Eq. 28, copy form), 300 steps on current FIT
Boundary consolidation (slow level, once per task, K_c inner steps):
  L_c = L_fit(theta_s; x_c) + 1[t>1] L_fit(theta_s; x_r) + lam_T * L_T(theta_s; x_c)
  L_T = surprise-weighted mean_p (1 - cos(sg(D_{theta_f*}(x_c))_p, D_{theta_s}(x_c)_p))   (fast expert = frozen teacher)
then reservoir write (task t FIT, JPEG q100, byte-capped) and fast reset. lam_T = 1 (BC-CMS) or 0 (BC-REPLAY control).
"""
from __future__ import annotations

import math
import pickle
from collections import defaultdict

import torch
import torch.nn.functional as F

from exps.cmsb import cms as C
from exps.hope_s_01 import run as H
from exps.nar_lite_01 import run as L
from exps.nar_pilot_00 import core as K, full as R
from exps.nar_warm_01 import run as W

GAMMA = 2.1
KF, KC = 300, 150
SLOW_LR, SLOW_WARMUP = 2e-3, 20
REPLAY = 24 * 2**20


def params(loras):
    return [p for w in loras.values() for p in (w.A, w.B)]


def optimizer(ps):
    cfg = R.CFG
    return R.O.StableAdamW([{'params': ps}], lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay,
                           eps=cfg.eps, amsgrad=True)


def weighted(e):
    return (H.weights(e, GAMMA) * e).mean()


def patch_err(a, b):
    return (1 - F.cosine_similarity(a, b, dim=1)).flatten(1)


def flat_grad(loss, ps):
    g = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
    return torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1) for x, p in zip(g, ps)])


class BCLearner:
    def __init__(self, seed, lam_t, kf=KF, kc=KC, replay_bytes=REPLAY, encode=None):
        self.seed, self.lam_t, self.kf, self.kc, self.encode = seed, float(lam_t), kf, kc, encode
        base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
        self.model_s, self.loras_s, _ = R.model_new(base)
        self.model_f, self.loras_f, _ = R.model_new(base)
        K.set_state(self.loras_s, K.init_state(self.loras_s, seed))
        K.set_state(self.loras_f, K.init_state(self.loras_f, seed))
        self.opt_s = optimizer(params(self.loras_s))
        self.steps_s, self.t = 0, 0
        self.reservoir = C.Reservoir(replay_bytes, seed + 77)
        self.counters = defaultdict(lambda: defaultdict(int))
        self.accessed = []

    # ------------------------------------------------------------------ one task
    def task(self, c, feats, roles, rows, diag_every=0, budget=None, teacher_hook=None):
        t, seed, cnt = self.t, self.seed, self.counters[self.t]
        self.accessed.append(c)
        ps_s, ps_f = params(self.loras_s), params(self.loras_f)
        trace = {'task': c, 'fast': [], 'consol': [], 'diag': []}
        # 1. slow -> fast transfer (copy), fresh fast optimizer
        K.set_state(self.loras_f, K.get_state(self.loras_s))
        self.last_fast_init = K.get_state(self.loras_f)
        self.last_slow_before = K.get_state(self.loras_s)
        opt_f = optimizer(ps_f)
        # 2. fast adaptation on current FIT only; slow frozen
        torch.manual_seed(seed + 100 * t)
        gen = torch.Generator().manual_seed(seed + 100 * t + 1)
        self.model_f.train()
        for i in range(self.kf):
            for grp in opt_f.param_groups:
                grp['lr'] = K.lr_at(R.CFG, i)
            opt_f.zero_grad(set_to_none=True)
            j = roles['FIT'][int(torch.randint(len(roles['FIT']), (1,), generator=gen))]
            loss = H.fit_loss(self.model_f, feats[j:j + 1].cuda().float(), GAMMA)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps_f, max_norm=R.CFG.clip)
            K.require(math.isfinite(float(loss)), f'non-finite fast loss {c} {i}')
            opt_f.step(); cnt['dec_fb_current'] += 1; cnt['fast_steps'] += 1
            trace['fast'].append(float(loss))
            if budget is not None and (i + 1) % 50 == 0:
                budget.check()
        opt_f.zero_grad(set_to_none=True)
        del opt_f
        # 3. freeze the trained fast expert as teacher
        self.model_f.eval()
        for p in ps_f:
            p.requires_grad_(False)
        if teacher_hook is not None:                       # tests only
            teacher_hook(self.loras_f)
        self.last_teacher = K.get_state(self.loras_f)
        self.last_slow_after_fast = K.get_state(self.loras_s)
        trace['reservoir_tags_at_consolidation'] = sorted(set(self.reservoir.tags))
        # 4. boundary consolidation of the slow level
        torch.manual_seed(seed + 100 * t + 50)
        gen = torch.Generator().manual_seed(seed + 100 * t + 2)
        self.model_s.train()
        for k in range(self.kc):
            j = roles['FIT'][int(torch.randint(len(roles['FIT']), (1,), generator=gen))]
            x = feats[j:j + 1].cuda().float()
            with torch.no_grad():
                _, de_f = self.model_f(x)
                teacher = de_f[0].detach()
            cnt['teacher_fwd'] += 1
            en, de = self.model_s(x)
            l_fit = weighted(patch_err(en[0].detach(), de[0]))
            l_t = weighted(patch_err(teacher, de[0]))
            loss = l_fit + self.lam_t * l_t if self.lam_t else l_fit
            cnt['dec_fb_current'] += 1
            if diag_every and k % diag_every == 0:
                gf, gt = flat_grad(l_fit, ps_s), flat_grad(l_t, ps_s)
                trace['diag'].append({'k': k, 'rho': float(gf @ gt / (gf.norm() * gt.norm() + 1e-30)),
                                      'teacher_gap': float(patch_err(en[0].detach(), teacher).mean()),
                                      'student_gap_to_teacher': float(patch_err(teacher, de[0]).mean().detach()),
                                      'grad_norm_fit': float(gf.norm()), 'grad_norm_t': float(gt.norm())})
            l_rep = None
            if self.reservoir.items:
                xr = self.encode(self.reservoir.draw())[None].cuda().float()
                l_rep = H.fit_loss(self.model_s, xr, GAMMA)
                loss = loss + l_rep
                cnt['dec_fb_replay'] += 1; cnt['encoder_fwd'] += 1
            self.opt_s.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps_s, max_norm=R.CFG.clip)
            for grp in self.opt_s.param_groups:
                grp['lr'] = SLOW_LR * min(1., (self.steps_s + 1) / SLOW_WARMUP)
            K.require(math.isfinite(float(loss)), f'non-finite consolidation loss {c} {k}')
            self.opt_s.step(); self.steps_s += 1; cnt['slow_steps'] += 1
            trace['consol'].append({'fit': float(l_fit), 't': float(l_t), 'rep': float(l_rep) if l_rep is not None else None})
            if budget is not None and (k + 1) % 50 == 0:
                budget.check()
        self.opt_s.zero_grad(set_to_none=True)
        self.teacher_after = K.get_state(self.loras_f)
        self.teacher_grads_none = all(p.grad is None for p in ps_f)
        # 5. causal reservoir write (after consolidation of task t)
        for j in roles['FIT']:
            self.reservoir.offer(C.jpeg392(rows[j]['relative_path']), c)
        # 6. fast reset (discard the expert)
        for p in ps_f:
            p.requires_grad_(True)
        K.set_state(self.loras_f, K.init_state(self.loras_f, seed))
        self.t += 1
        return trace

    # ------------------------------------------------------------------ persistence / ledger
    def state(self):
        r = self.reservoir
        return {'slow': K.get_state(self.loras_s), 'opt_s': self.opt_s.state_dict(), 'steps_s': self.steps_s, 't': self.t,
                'reservoir': {'items': list(r.items), 'tags': list(r.tags), 'seen': r.seen, 'full': r.full,
                              'total': r.total, 'g': r.g.get_state()}}

    def load(self, st):
        K.set_state(self.loras_s, st['slow'])
        self.opt_s.load_state_dict(st['opt_s'])
        self.steps_s, self.t = st['steps_s'], st['t']
        r, s = self.reservoir, st['reservoir']
        r.items, r.tags, r.seen, r.full, r.total = list(s['items']), list(s['tags']), s['seen'], s['full'], s['total']
        r.g.set_state(s['g'])

    def ledger(self):
        slow = K.tensor_state_bytes(K.get_state(self.loras_s))
        moments = sum(v.numel() * v.element_size() for s in self.opt_s.state.values() for v in s.values()
                      if torch.is_tensor(v))
        r = self.reservoir
        meta = len(pickle.dumps((r.tags, r.seen, r.full, r.total, self.steps_s, self.t))) + r.g.get_state().numel()
        return {'deploy_adapter_bytes_bf16': slow // 2, 'adapter_bytes_bf16': slow // 2, 'slow_adapter_fp32': slow,
                'slow_optimizer_state': moments, 'replay_bytes': r.bytes(), 'replay_items': len(r.items),
                'replay_items_per_task': {t_: r.tags.count(t_) for t_ in dict.fromkeys(r.tags)},
                'metadata_rng': meta, 'continual_state_bytes': slow + moments + r.bytes() + meta,
                'transient_fast_bytes_fp32': K.tensor_state_bytes(K.get_state(self.loras_f)) * 4}


# ---------------------------------------------------------------------- deployment / evaluation
@torch.no_grad()
def stage_grids(stages, feats, dtype=torch.bfloat16):
    """[S,N,28,28]: one decoder pass per image per stage state (deployment = slow adapter only)."""
    base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
    model, loras, _ = R.model_new(base)
    model.eval().to(dtype)
    arch = [{n: (a.to(dtype), b.to(dtype)) for n, (a, b) in s.items()} for s in stages]
    out = L.expert_grids(model, loras, arch, feats, dtype)
    del model, loras
    return out
