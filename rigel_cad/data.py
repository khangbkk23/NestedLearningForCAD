"""Dataset identities for RIGEL-CAD streams (paths, labels, mask paths; no pixels are read here).

Supported layouts:
* MVTec AD and MPDD: <root>/<category>/{train/good, test/<defect>, ground_truth/<defect>/<stem>_mask.png}, *.png.
* VisA: <root>/split_csv/1cls.csv (object, split, label, image, mask) with paths relative to <root>.
* MVTec AD 2: <root>/<scenario>/{train, validation, test_public, test_private, test_private_mixed}/...

Row order is part of the method (greedy k-centre starts from a seeded random token index): training images are used
in sorted relative-path order, test images in the order below (identical to the frozen research code).
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MVTEC_CATEGORIES = ('bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                    'tile', 'toothbrush', 'transistor', 'wood', 'zipper')
VISA_CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2',
                   'pcb3', 'pcb4', 'pipe_fryum')
MPDD_CATEGORIES = ('bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes')
CATEGORIES = {'mvtec': MVTEC_CATEGORIES, 'visa': VISA_CATEGORIES, 'mpdd': MPDD_CATEGORIES}
AD2_SPLITS = ('train', 'validation', 'test_public', 'test_private', 'test_private_mixed')
AD2_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
ORDER_SALT = 'NANM-EVAL-01/order-c/v1'


@dataclass(frozen=True)
class Sample:
    relative_path: str
    label: int = 0                          # 1 = anomalous (test only)
    mask_path: Optional[str] = None         # relative to the dataset root


def stream_orders(dataset: str) -> Dict[str, Tuple[str, ...]]:
    """Registered task orders: a = alphabetical (canonical), b = reversed, c = salted-SHA256 permutation."""
    base = CATEGORIES[dataset]
    key = lambda t: hashlib.sha256(f'{ORDER_SALT}|{t}'.encode()).hexdigest()
    return {'a': tuple(base), 'b': tuple(reversed(base)), 'c': tuple(sorted(base, key=key))}


def _mvtec_style(root: Path, cat: str):
    train = [Sample(p.relative_to(root).as_posix()) for p in sorted((root / cat / 'train' / 'good').glob('*.png'))]
    test = []
    for d in sorted(p for p in (root / cat / 'test').iterdir() if p.is_dir()):
        for p in sorted(d.glob('*.png')):
            lab = int(d.name != 'good')
            mask = f'{cat}/ground_truth/{d.name}/{p.stem}_mask.png' if lab else None
            test.append(Sample(p.relative_to(root).as_posix(), lab, mask))
    return train, test


_VISA_ROWS: Dict[str, list] = {}


def _visa(root: Path, cat: str):
    key = str(root)
    if key not in _VISA_ROWS:
        with open(root / 'split_csv' / '1cls.csv') as f:
            _VISA_ROWS[key] = list(csv.DictReader(f))
    mine = sorted((r for r in _VISA_ROWS[key] if r['object'] == cat), key=lambda r: r['image'])
    train = [Sample(r['image']) for r in mine if r['split'] == 'train']
    if not all(r['label'] == 'normal' for r in mine if r['split'] == 'train'):
        raise ValueError('VisA train split must be normal-only')
    test = [Sample(r['image'], int(r['label'] == 'anomaly'), r['mask'] or None) for r in mine if r['split'] == 'test']
    return train, test


def category_split(dataset: str, root, cat: str) -> Tuple[List[Sample], List[Sample]]:
    """(train samples sorted by path, test samples in canonical order) of one category."""
    root = Path(root)
    train, test = _visa(root, cat) if dataset == 'visa' else _mvtec_style(root, cat)
    train = sorted(train, key=lambda s: s.relative_path)
    if not train or not test:
        raise ValueError(f'empty split for {dataset}/{cat} under {root}')
    if {s.relative_path for s in train} & {s.relative_path for s in test}:
        raise ValueError(f'train/test overlap in {dataset}/{cat}')
    return train, test


# ---------------------------------------------------------------------- MVTec AD 2
def ad2_scenarios(root) -> Tuple[str, ...]:
    root = Path(root)
    return tuple(sorted(p.name for p in root.iterdir() if p.is_dir() and (p / 'train').is_dir()))


def ad2_files(root, scenario: str, split: str) -> List[str]:
    """Image files of one scenario/split (ground truth excluded), sorted, relative to root."""
    root = Path(root)
    base = root / scenario / split
    fs = sorted(p for p in base.rglob('*') if p.is_file() and p.suffix.lower() in AD2_EXT
                and 'ground_truth' not in p.relative_to(base).parts)
    return [str(p.relative_to(root)) for p in fs]


def ad2_mask_of(root, rel: str) -> Optional[Path]:
    """test_public ground truth <scenario>/test_public/ground_truth/bad/<stem>_mask.png (None for good images)."""
    root = Path(root)
    gt = root / Path(rel).parts[0] / 'test_public' / 'ground_truth'
    stem = Path(rel).stem
    if not gt.exists():
        return None
    c = [q for q in gt.rglob('*') if q.is_file() and q.stem in (stem, stem + '_mask')]
    return c[0] if c else None
