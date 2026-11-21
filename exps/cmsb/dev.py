"""CMS-B development sets (agents/tasks/task_cms_bounded_00.md).

Dev-S: synthetic proxy on held-out VisA train normals (VBENCH EVAL role minus TCS-00 FIT/CREDIT), 40 hosts per class:
20 normal, 10 DRAEM Perlin+DTD (PTAG Q2), 10 CutPaste (PTAG Q3); synthesis, masks and A1 metrics at 392x392.
Dev-R: VisA-DEV real (240 images, frozen cad_final_v1 rule), native-resolution masks.
Features are bf16 lite tokens (deployment front end). Used for every CMS-B gate; never a training target.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from exps.calm02 import data as CD, native as N
from exps.dinomaly import encoder as E
from exps.nar_lite_01 import run as L
from exps.nar_pilot_00 import core as K, full as R
from exps.tcs_00 import run as T
from exps.vbench_00 import run as V

CACHE = Path.home() / 'hope_cad_cache/cmsb'
VISA, ORDER = T.VISA, T.ORDER
N_NORMAL, N_Q2, N_Q3 = 20, 10, 10
DTYPE = torch.bfloat16
SALT = 'CMSB/devs/v1|'


def _h(salt, p):
    return hashlib.sha256((salt + p).encode()).hexdigest()


# ---------------------------------------------------------------- identities (paths only)
def hosts(c, ids=None):
    """Held out from every trained detector in this lineage: VBENCH EVAL role minus TCS-00 FIT and CREDIT."""
    ids = ids or CD.identities('visa')
    rows = V.train_rows(c, ids)
    ev = {rows[i]['relative_path'] for i in T.roles_for(rows)['EVAL']}
    trs = T.train_rows(c, ids)
    tr = T.roles_for(trs)
    used = {trs[i]['relative_path'] for i in tr['FIT'] + tr['CREDIT']}
    cand = sorted(ev - used, key=lambda p: _h(SALT, p))
    need = N_NORMAL + N_Q2 + N_Q3
    K.require(len(cand) >= need, f'too few held-out hosts: {c}')
    return cand[:need]


def manifest_sha(ids=None):
    ids = ids or CD.identities('visa')
    return hashlib.sha256('\n'.join(p for c in ORDER for p in hosts(c, ids)).encode()).hexdigest()


# ---------------------------------------------------------------- synthesis (train normals only)
def load392(rel):
    with Image.open(VISA / rel) as im:
        return np.array(im.convert('RGB').resize((E.SIZE, E.SIZE), Image.BILINEAR), dtype=np.uint8)


def normalise(a):
    """Identical to E.preprocess after its resize."""
    x = torch.from_numpy(np.asarray(a, dtype=np.float32) / 255.).permute(2, 0, 1)
    return (x - torch.tensor(E.MEAN).view(3, 1, 1)) / torch.tensor(E.STD).view(3, 1, 1)


def synthesise(c, paths, gen):
    from exps.ptag import synth as S
    ci = ORDER.index(c)
    images, masks, kinds = [], [], []
    for k, p in enumerate(paths):
        a = load392(p)
        if k < N_NORMAL:
            images.append(a); masks.append(np.zeros((E.SIZE, E.SIZE), bool)); kinds.append('normal')
        elif k < N_NORMAL + N_Q2:
            out, m, _ = S.q2(gen, a, c, ci, k, 0)
            images.append(out); masks.append(m.astype(bool)); kinds.append('Q2')
        else:
            out, m, _ = S.q3(a, ci, k, 0)
            images.append(out); masks.append(m.astype(bool)); kinds.append('Q3')
    return images, masks, kinds


@torch.no_grad()
def lite_tokens(enc, tensors):
    f = torch.empty(len(tensors), 784, 768, dtype=DTYPE)
    for i in range(0, len(tensors), 8):
        x = torch.stack(tensors[i:i + 8]).cuda().to(DTYPE)
        f[i:i + len(x)] = L.tokens(enc, x).cpu()
    K.require(bool(torch.isfinite(f).all()), 'feature finiteness')
    return f


def build(log=print):
    """Writes CACHE/devs_<c>.pt and CACHE/devr_<c>.pt (idempotent); returns {name: sha256}."""
    from exps.ptag import synth as S
    CACHE.mkdir(parents=True, exist_ok=True)
    ids = CD.identities('visa')
    enc = L.lite_encoder(DTYPE)
    gen = S.NativePerlin()
    dev = T.dev_rows(ids)
    out = {}
    for c in ORDER:
        ps, pr = CACHE / f'devs_{c}.pt', CACHE / f'devr_{c}.pt'
        if not ps.exists():
            t0 = time.monotonic()
            paths = hosts(c, ids)
            images, masks, kinds = synthesise(c, paths, gen)
            R.save(ps, {'features': lite_tokens(enc, [normalise(a) for a in images]),
                        'masks': torch.from_numpy(np.stack(masks)), 'kinds': kinds, 'paths': paths,
                        'labels': [int(k != 'normal') for k in kinds]})
            log('DEVS', c, len(paths), round(time.monotonic() - t0, 1), 's')
        if not pr.exists():
            rows = dev[c]
            R.save(pr, {'features': lite_tokens(enc, [E.preprocess(VISA / r['relative_path']) for r in rows]),
                        'rows': rows})
            log('DEVR', c, len(rows))
        out[ps.name], out[pr.name] = R.sha(ps), R.sha(pr)
    del enc
    torch.cuda.empty_cache()
    return out


def load(kind, c, shas=None):
    p = CACHE / f'{kind}_{c}.pt'
    if shas is not None:
        K.require(R.sha(p) == shas[p.name], f'dev cache changed: {p.name}')
    return torch.load(p, weights_only=True)


# ---------------------------------------------------------------- metrics
def geometry_r(rows):
    hws = [N.native_hw(VISA / r['relative_path']) for r in rows]
    masks = [N.native_mask(VISA / r['mask_path'] if r['mask_path'] else None, hw) for r, hw in zip(rows, hws)]
    return hws, masks, [r['label'] for r in rows]


def cell(grid, hws, masks, labels, idx=None):
    """A1 readout metrics for [N,28,28] grids (optionally a subset)."""
    sm = L.smooth392(grid)
    if idx is not None:
        sm, hws, masks, labels = sm[idx], [hws[i] for i in idx], [masks[i] for i in idx], [labels[i] for i in idx]
    ap, auc = T.a1_metrics(sm, hws, masks, labels)
    return {'pixel_ap': ap, 'image_auroc': auc, 'n': len(labels)}


def metrics_s(grid, d):
    hws = [(E.SIZE, E.SIZE)] * len(d['labels'])
    masks = [m.numpy() for m in d['masks']]
    res = {'all': cell(grid, hws, masks, d['labels'])}
    for q in ('Q2', 'Q3'):
        idx = [i for i, k in enumerate(d['kinds']) if k in ('normal', q)]
        res[q] = cell(grid, hws, masks, d['labels'], idx)
    return res
