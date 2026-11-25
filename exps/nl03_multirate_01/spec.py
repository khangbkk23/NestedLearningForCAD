"""NL03-MULTIRATE-01 registered constants, splits, clocks and byte accounting (single source of truth).

Registration: agents/tasks/task_nl03_multirate_01_registration.md (hash-locked in PREREG_LOCK.json).
Every number here is fixed before any scoring and is never changed after a result is seen (§11 NO-RESCUE).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/hope_cad/nl03_multirate_01'
CACHE = Path('/mnt/k/NLCAD/cache/nl03_multirate_01')
D = 768
MiB = 2 ** 20

# ------------------------------------------------------------------ §4 / §10 development splits (normal only)
N_EVAL, N_CALIB = 32, 16             # per task, capped by availability; §4 minimum on VisA: 16 / 8
MIN_EVAL, MIN_CALIB = 16, 8
SPLIT_RULE = ('canonical roles (salted SHA "NAR-PILOT-00/roles/v1|path", n = N//5): FIT / CREDIT / EVAL. '
              'NL-02 SELECTED-pool paths (ranks 1-32, NANM-NL-02 roles) removed from every split. '
              'FIT_dev = FIT role in chronological (relative_path) order; EVAL = first N_EVAL of the EVAL role; '
              'CALIB = first N_CALIB of the CREDIT role (role order).')

# ------------------------------------------------------------------ §6 clocks
MB_IMAGES = 8                        # one fast step = one minibatch of 8 FIT images (6,272 tokens)
KEY_FRAC = 0.01                      # canonical per-write coreset ratio (ceil(0.01 * tokens))
C_PRIMARY = 8
C_DIAG = (1, 4, 16)

# ------------------------------------------------------------------ §7 bytes
KEY_B = 2 * D                        # fp16 key
COUNT_B = 4                          # int32 count
S_B = D * D * 4 + 8                  # slow S (fp32) + its count (fp64) = 2,359,304 (canonical accounting)
ACC_B = D * D * 4 + 8                # window accumulator of H_in (fp32 sum + fp64 mass), arms with C > 1
BUDGETS = {4: 4 * MiB, 8: 8 * MiB, 16: 16 * MiB}
PRIMARY_MIB = 8

# arm -> (fast rule, counts stored, slow kind, C, transfer, pending charged, window accumulator charged)
ARMS = {
    'C0': dict(fast='task', counts=False, slow='pooled', C=None, transfer=False, pending=False, acc=False,
               note='historical G3-E at task boundaries (reference, not clock-matched)'),
    'C1': dict(fast='mb', counts=False, slow='pooled', C=1, transfer=False, pending=False, acc=False,
               note='minibatch fast level + commutative pooled S, evicted freed'),
    'C2': dict(fast='mb', counts=True, slow='ema', C=1, transfer=True, pending=True, acc=False,
               note='BMRM C=1 (single clock, transfer, matched decay)'),
    'C3': dict(fast='mb', counts=False, slow='ema', C=8, transfer=False, pending=False, acc=True,
               note='C=8, beta(8), H_in only, evicted freed'),
    'C4': dict(fast='mb', counts=False, slow='ema', C=1, transfer=False, pending=False, acc=False,
               note='per-step EMA of H_in, beta(1)'),
    'C5': dict(fast='mb', counts=False, slow='none', C=None, transfer=False, pending=False, acc=False,
               note='flat streaming Euclidean exemplar memory, full budget as keys'),
    'P': dict(fast='mb', counts=True, slow='ema', C=8, transfer=True, pending=True, acc=True, note='BMRM C=8 (primary)'),
    'P4': dict(fast='mb', counts=True, slow='ema', C=4, transfer=True, pending=True, acc=True, note='BMRM C=4 (diag)'),
    'P16': dict(fast='mb', counts=True, slow='ema', C=16, transfer=True, pending=True, acc=True, note='BMRM C=16 (diag)'),
}
CONVENTIONAL = ('C1', 'C4', 'C5')    # CB is chosen among these (§14)
MATCHED = ('C1', 'C2', 'C3', 'C4', 'C5')


def slow_bytes(arm):
    a = ARMS[arm]
    return (S_B if a['slow'] != 'none' else 0) + (ACC_B if a['acc'] else 0)


def per_key_bytes(arm):
    return KEY_B + (COUNT_B if ARMS[arm]['counts'] else 0)


def total_cap(arm, budget):
    """Max number of stored keys (fast K plus pending P) within the budget; None = unbounded."""
    if budget is None:
        return None
    return (budget - slow_bytes(arm)) // per_key_bytes(arm)


def bytes_of(arm, n_fast, n_pending):
    return (n_fast + n_pending) * per_key_bytes(arm) + slow_bytes(arm)


class BudgetViolation(AssertionError):
    pass


def assert_budget(arm, where, n_fast, n_pending, budget):
    b = bytes_of(arm, n_fast, n_pending)
    if budget is not None and b > budget:
        raise BudgetViolation(f'BUDGET_VIOLATION arm={arm} at={where} fast={n_fast} pending={n_pending} '
                              f'bytes={b} > {budget}')
    return b


def tau_of(n_fit_images):
    """tau = mean fast updates per task, from manifests (image counts) only."""
    steps = [math.ceil(n / MB_IMAGES) for n in n_fit_images]
    return sum(steps) / len(steps)


def beta(C, tau):
    return math.exp(-C / tau)


# ------------------------------------------------------------------ §10 synthetic generator (SYNTHETIC_DEV), fixed
SYNTH = dict(version='NL03-MULTIRATE-01/synth/v1', size=392, n_patches=(1, 2), area_frac=(0.004, 0.02),
             aspect=(0.5, 2.0), center_box=(0.15, 0.85), feather_sigma_px=2.0, mask_thr=0.5,
             source='EVAL image (i+1) mod n of the same task, patch taken from an independent random location',
             blend='alpha = Gaussian-feathered ellipse; out = (1-alpha)*target + alpha*source in [0,1] RGB at 392x392, '
                   'then canonical ImageNet normalisation', per_image=1)

# ------------------------------------------------------------------ §10 thresholds / metrics
THRESHOLD_RULE = 'theta_t = max image score (A1 top-1%) over the task CALIB images, at the stage task t is acquired'
BOOT_B = 2000
BOOT_SEED = 20261124
SEEDS = (2711, 2712, 2713)
NL02_SELECTED = ('visa/pcb3', 'visa/macaroni2', 'visa/pcb4', 'mpdd/bracket_brown')

# ------------------------------------------------------------------ §11 gates
M2 = dict(auroc=0.010, ap=0.010, fpr=0.02, cur_auroc_drop=0.005, cur_fpr_rise=0.01)
M3_MIN_POS = 7
DEV_GPU_CEILING_S = 4 * 3600
CONF_GPU_CEILING_S = 8 * 3600


def ssha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def image_root(ds):
    return ROOT / 'data' / ds


def splits(G, task):
    """Development splits of one task from identities only (no tensor read)."""
    from exps.nl03_pressure_00 import core as PC
    ds, c = task.split('/')
    PC.use(G, ds)
    rows = G.train_rows(ds, c)
    paths = [r['relative_path'] for r in rows]
    rl = G.roles(ds, rows)
    ex = PC.nl02_excluded(G, task)
    fit = sorted(paths[i] for i in rl['FIT'] if paths[i] not in ex)
    ev = [paths[i] for i in rl['EVAL'] if paths[i] not in ex][:N_EVAL]
    cal = [paths[i] for i in rl['CREDIT'] if paths[i] not in ex][:N_CALIB]
    return {'FIT': fit, 'EVAL': ev, 'CALIB': cal}


def data_gate(tasks):
    """§4: primary stream VisA needs >= 16 EVAL and >= 8 CALIB per task, all disjoint and SELECTED-free."""
    rows = {t: (v['EVAL'], v['CALIB'], v['disjoint'], v['selected_free'], v.get('cache', {}).get('rows_match', True))
            for t, v in tasks.items()}
    visa = {t: r for t, r in rows.items() if t.startswith('visa/')}
    ok = all(e >= MIN_EVAL and c >= MIN_CALIB and d and s and m for e, c, d, s, m in visa.values()) and len(visa) == 12
    short = {t: {'EVAL': r[0], 'CALIB': r[1]} for t, r in rows.items() if r[0] < MIN_EVAL or r[1] < MIN_CALIB}
    return {'pass': bool(ok), 'primary': 'visa (12 tasks)', 'below_minimum_any_stream': short,
            'full_fit_features': 'all train/good tokens cached (fp32, 784x768 per image) for visa, mvtec, mpdd'}
