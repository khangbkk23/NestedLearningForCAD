"""NL03-MULTIRATE-01 §10(a): SYNTHETIC_DEV anomalies on held-out EVAL normal images (CutPaste / NSA-style patch
blending from another normal image of the same category). Spec fixed in spec.SYNTH; never tuned.

Pipeline per EVAL image i (392x392, canonical front end):
  target = PIL RGB bilinear resize to 392 of EVAL[i] (float [0,1]); source = same of EVAL[(i+1) % n]
  for each of 1-2 patches: ellipse (area 0.4-2 % of the image, aspect 0.5-2, random rotation), target centre and an
  independent source centre in [0.15, 0.85]^2; alpha = Gaussian-feathered (sigma 2 px) ellipse;
  target <- (1 - alpha) target + alpha * shift(source)
  mask = (combined alpha >= 0.5); tokens = canonical encoder (b14.tokens) of the ImageNet-normalised image, unit fp16.
RNG: numpy default_rng(int(sha256(version|relative_path)[:16], 16)) - independent of arms, stages and outcomes.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from exps.nl03_multirate_01 import spec as SP

SZ = SP.SYNTH['size']


def _img(path):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert('RGB').resize((SZ, SZ), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.


def _norm(x):
    from exps.dinomaly import encoder as E
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)
    return (t - torch.tensor(E.MEAN).view(3, 1, 1)) / torch.tensor(E.STD).view(3, 1, 1)


def rng_of(rel):
    return np.random.default_rng(int(SP.ssha(f"{SP.SYNTH['version']}|{rel}")[:16], 16))


def make(target, source, rng):
    """Returns (image [SZ,SZ,3] float32, mask [SZ,SZ] uint8, params)."""
    from scipy.ndimage import gaussian_filter
    s = SP.SYNTH
    out = target.copy()
    mask = np.zeros((SZ, SZ), bool)
    yy, xx = np.mgrid[0:SZ, 0:SZ].astype(np.float32)
    params = []
    for _ in range(int(rng.integers(s['n_patches'][0], s['n_patches'][1] + 1))):
        area = rng.uniform(*s['area_frac']) * SZ * SZ
        asp = float(np.exp(rng.uniform(np.log(s['aspect'][0]), np.log(s['aspect'][1]))))
        rx, ry = np.sqrt(area * asp / np.pi), np.sqrt(area / (np.pi * asp))
        th = rng.uniform(0, np.pi)
        cx, cy = rng.uniform(*s['center_box'], 2) * SZ
        sx, sy = rng.uniform(*s['center_box'], 2) * SZ
        c, si = np.cos(th), np.sin(th)
        u = (xx - cx) * c + (yy - cy) * si
        v = -(xx - cx) * si + (yy - cy) * c
        hard = ((u / rx) ** 2 + (v / ry) ** 2 <= 1).astype(np.float32)
        alpha = np.clip(gaussian_filter(hard, s['feather_sigma_px']), 0, 1)
        ys = np.clip(np.round(yy - cy + sy).astype(np.int64), 0, SZ - 1)
        xs = np.clip(np.round(xx - cx + sx).astype(np.int64), 0, SZ - 1)
        shifted = source[ys, xs]
        out = (1 - alpha[..., None]) * out + alpha[..., None] * shifted
        mask |= alpha >= s['mask_thr']
        params.append({'area_px': float(area), 'aspect': asp, 'theta': float(th), 'c': [float(cx), float(cy)],
                       's': [float(sx), float(sy)]})
    return out.astype(np.float32), mask.astype(np.uint8), params


def path_of(task):
    ds, c = task.split('/')
    return SP.CACHE / 'synth' / f'{ds}_{c}.pt'


@torch.no_grad()
def build_task(G, enc, task):
    """Generate + encode the synthetic EVAL set of one task (idempotent; never overwrites)."""
    from exps.nanm_r0 import audit as A
    from exps.nar_b14_01 import run as b14
    p = path_of(task)
    if p.exists():
        return json.loads(p.with_suffix('.json').read_text())
    ds, c = task.split('/')
    sp = SP.splits(G, task)
    ev, cal = sp['EVAL'], sp['CALIB']
    root = SP.image_root(ds)
    t0 = time.time()
    imgs, masks, params = [], [], []
    for i, rel in enumerate(ev):
        tgt, src = _img(root / rel), _img(root / ev[(i + 1) % len(ev)])
        im, m, pr = make(tgt, src, rng_of(rel))
        imgs.append(_norm(im)); masks.append(m); params.append(pr)

    def enc_all(xs):                                     # same front end and batch size (8) for every query set
        out = []
        for i in range(0, len(xs), 8):
            x = torch.stack(xs[i:i + 8]).cuda()
            out.append(A.unit(b14.tokens(enc, x).reshape(-1, SP.D)).half().reshape(len(x), 784, SP.D).cpu())
        return torch.cat(out)
    toks = enc_all(imgs)
    clean = enc_all([_norm(_img(root / r)) for r in ev])
    calib = enc_all([_norm(_img(root / r)) for r in cal])
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'rows': ev, 'calib_rows': cal, 'tokens': toks, 'clean': clean, 'calib': calib,
                'masks': torch.from_numpy(np.stack(masks)), 'params': params}, p)
    meta = {'task': task, 'n': len(ev), 'sha256': _fsha(p), 'seconds': time.time() - t0,
            'mask_frac_mean': float(np.mean([m.mean() for m in masks])), 'spec': SP.SYNTH,
            'source_images': {r: _fsha(root / r) for r in ev}, 'encoder_sha256': b14.ENC_SHA}
    p.with_suffix('.json').write_text(json.dumps(meta, indent=1))
    return meta


def _fsha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


_LOADED = {}


def load(task):
    if task in _LOADED:
        return _LOADED[task]
    p = path_of(task)
    meta = json.loads(p.with_suffix('.json').read_text())
    assert _fsha(p) == meta['sha256'], f'synth cache changed {task}'
    _LOADED[task] = torch.load(p, weights_only=False)
    return _LOADED[task]


@torch.no_grad()
def clean_parity(G, enc, task, n=2):
    """Front-end check: an EVAL image through this pipeline without a patch == its canonical cached tokens."""
    from exps.nanm_r0 import audit as A
    from exps.nar_b14_01 import run as b14
    from exps.nl03_pressure_00 import core as PC
    ds, c = task.split('/')
    ev = SP.splits(G, task)['EVAL'][:n]
    x = torch.stack([_norm(_img(SP.image_root(ds) / r)) for r in ev]).cuda()
    mine = A.unit(b14.tokens(enc, x).reshape(-1, SP.D)).half().cpu()
    PC.use(G, ds)
    data = G.load_train(ds, c)
    idx = {r['relative_path']: i for i, r in enumerate(data['rows'])}
    ref = A.unit(data['features'][torch.as_tensor([idx[r] for r in ev])].reshape(-1, SP.D).cuda()).half().cpu()
    return {'task': task, 'bitwise': bool(torch.equal(mine, ref)),
            'max_abs': float((mine.float() - ref.float()).abs().max())}
