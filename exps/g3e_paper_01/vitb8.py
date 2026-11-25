"""G3E-PAPER-01 PART B: matched-backbone (CADIC) token contract and one-time feature extraction to K:.

CONTRACT (documented before running; MATCHED_BACKBONE_ABLATION):
- Backbone: timm `vit_base_patch8_224`, ImageNet-21k weights (augreg in21k), frozen; local checkpoint
  checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth, sha256 755e9f5f…a415 (343,282,995 B), already
  on disk (no download); loaded strictly by the repository's CADIC adapter models/feature_extractors/cadic_vit_v1.py.
- Layer: output of transformer block 9 (one-based; blocks[8]), the CADIC paper's choice (layer 9 of 12), before the
  final norm; CLS removed -> 784 = 28 x 28 patch tokens x 768.
- Front end: PIL RGB, direct square bilinear resize to 224 x 224 (full field of view, same geometry as the canonical
  DINOv2 392 front end, so the 28 x 28 grid and the A1 readout are unchanged), ToTensor, ImageNet mean/std
  (0.485, 0.456, 0.406)/(0.229, 0.224, 0.225) = the repository's existing CADIC contract (timm's augreg default would be
  0.5/0.5; recorded as a known ambiguity, not tuned).
- Token contract for G3: per-token L2 normalisation (A.unit), stored fp16 [n_img, 784, 768], exactly as the DINOv2
  caches are consumed by G3. Train rows: sorted relative_path (= final.train_rows_sorted); test rows: the canonical
  test row order (G.test_rows), checked against the DINOv2 benchmark grids' row order.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from exps.nl03_multirate_01 import spec as SP

ROOT = SP.ROOT
CKPT = ROOT / 'checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth'
CKPT_SHA = '755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415'
CACHE = Path('/mnt/k/NLCAD/cache/g3e_paper_01/vitb8')
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CONTRACT = {'model': 'timm vit_base_patch8_224 (ImageNet-21k augreg)', 'checkpoint': str(CKPT.relative_to(ROOT)),
            'checkpoint_sha256': CKPT_SHA, 'layer': 'block 9 one-based (blocks[8]) output, pre final norm',
            'tokens': '784 = 28x28 patch tokens (CLS dropped) x 768', 'input': 'PIL RGB bilinear direct resize 224x224',
            'normalisation': 'ImageNet mean/std (repository CADIC contract)', 'token_norm': 'L2 per token (A.unit), fp16',
            'label': 'MATCHED_BACKBONE_ABLATION'}


def _fsha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def _prep(path):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert('RGB').resize((224, 224), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.).permute(2, 0, 1)
    return (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)


def extractor():
    from models.feature_extractors.cadic_vit_v1 import CADICViTConfig, CADICViTFeatureExtractor
    assert _fsha(CKPT) == CKPT_SHA, 'ViT-B/8 checkpoint hash'
    cfg = CADICViTConfig(checkpoint_path=str(CKPT), checkpoint_identity=f'sha256:{CKPT_SHA}', layer_number=9,
                         layer_indexing='one_based', mean=MEAN, std=STD)
    return CADICViTFeatureExtractor(cfg, device='cuda')


def path_of(ds, c, split):
    return CACHE / f'{ds}_{c}_{split}.pt'


@torch.no_grad()
def extract(G, ds, guard=None):
    """All train (sorted) and test (canonical order) images of every category of ds; skip existing files."""
    from exps.nanm_r0 import audit as A
    from exps.nl03_pressure_00 import core as PC
    PC.use(G, ds)
    root = Path(G.data_root(ds))
    enc = None
    pool = ThreadPoolExecutor(6)
    secs = 0.0
    for c in G.order(ds):
        for split in ('train', 'test'):
            p = path_of(ds, c, split)
            if p.exists():
                continue
            rows = (sorted(r['relative_path'] for r in G.train_rows(ds, c)) if split == 'train'
                    else [r['relative_path'] for r in G.test_rows(ds, c)])
            enc = enc or extractor()
            t0 = time.time()
            out = []
            for i in range(0, len(rows), 32):
                xs = list(pool.map(_prep, [root / r for r in rows[i:i + 32]]))
                f = enc.extract_patch_features(torch.stack(xs).cuda())
                out.append(A.unit(f.reshape(-1, SP.D).float()).half().reshape(len(xs), 784, SP.D).cpu())
            feats = torch.cat(out)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'rows': rows, 'features': feats}, p)
            meta = {'sha256': _fsha(p), 'n': len(rows), 'contract': CONTRACT, 'seconds': time.time() - t0}
            p.with_suffix('.json').write_text(json.dumps(meta, indent=1))
            secs += time.time() - t0
            print(time.strftime('%H:%M:%S'), 'VITB8', ds, c, split, len(rows), '%.0fs' % (time.time() - t0), flush=True)
            if guard:
                guard(secs)
    pool.shutdown()
    return secs


def _load(ds, c, split):
    p = path_of(ds, c, split)
    assert _fsha(p) == json.loads(p.with_suffix('.json').read_text())['sha256'], f'vitb8 cache changed {p.name}'
    return torch.load(p, weights_only=True)


def fit_tokens(ds, c, rows_sorted):
    d = _load(ds, c, 'train')
    assert d['rows'] == list(rows_sorted), 'train rows differ'
    return d['features'].reshape(-1, SP.D).cuda(), len(d['rows'])


def load_test(ds, c):
    d = _load(ds, c, 'test')
    return d['features'].reshape(-1, SP.D), d['rows']
