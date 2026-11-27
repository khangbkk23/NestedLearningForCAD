"""End-to-end RIGEL-CAD runs: continual stream on MVTec AD / VisA / MPDD, and the MVTec AD 2 protocol."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from rigel_cad import ad2 as AD2
from rigel_cad import data as DATA
from rigel_cad import metrics as M
from rigel_cad import readout
from rigel_cad.config import TOKEN_DIM, RigelConfig
from rigel_cad.encoders import build_encoder, file_sha256
from rigel_cad.model import RigelCAD


def storage_suffix(config: RigelConfig) -> str:
    """Result / state file-name suffix of the storage format ('' for the v1 default, so v1 file names are unchanged and
    compact-memory runs never overwrite them)."""
    return '' if config.storage == 'fp16' else '_int4tri'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


class TokenStore:
    """Encodes images to unit fp16 tokens [n, 784, 768]; optionally caches them as <cache>/<key>.pt (+ sha256)."""

    def __init__(self, backbone: str, device: str, cache_dir=None, batch: Optional[int] = None):
        self.backbone, self.device = backbone, device
        self.cache = Path(cache_dir) / backbone if cache_dir else None
        self.batch = batch or (32 if backbone == 'vit_b8_in21k' else 8)
        self._enc = None

    def encoder(self):
        if self._enc is None:
            self._enc = build_encoder(self.backbone, self.device)
        return self._enc

    def get(self, key: str, root, rel_paths) -> torch.Tensor:
        rel_paths = list(rel_paths)
        p = self.cache / f'{key}.pt' if self.cache else None
        if p is not None and p.exists():
            meta = json.loads(p.with_suffix('.json').read_text())
            if file_sha256(p) != meta['sha256']:
                raise RuntimeError(f'token cache changed: {p}')
            d = torch.load(p, weights_only=True)
            if d['rows'] != rel_paths:
                raise RuntimeError(f'token cache rows differ: {p}')
            return d['tokens']
        root = Path(root)
        t0 = time.time()
        tok = self.encoder().encode_paths([root / r for r in rel_paths], self.batch)
        if p is not None:
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'rows': rel_paths, 'tokens': tok}, p)
            write_json(p.with_suffix('.json'), {'sha256': file_sha256(p), 'n': len(rel_paths), 'backbone': self.backbone,
                                                'seconds': time.time() - t0})
        log('ENCODE', key, len(rel_paths), f'{time.time() - t0:.0f}s')
        return tok


# ====================================================================== MVTec AD / VisA / MPDD stream
def run_stream(dataset: str, data_root, config: RigelConfig, order: str = 'a', out_dir='results/rigel_cad',
               cache_dir=None, bootstrap: Optional[bool] = None, save_grids: bool = False) -> dict:
    """Learn the categories of `dataset` one after another (task order `order`), score every category's test set
    with the state after every task from its acquisition on, and write RESULTS_<tag>.json."""
    bootstrap = (order == 'a') if bootstrap is None else bootstrap
    tasks = DATA.stream_orders(dataset)[order]
    budget = 'inf' if config.budget_mib is None else f'{config.budget_mib:g}'
    tag = (f'{dataset}_{order}{config.seed}_{config.backbone}_B{budget}'
           f'{"" if config.slow_level else "_fastonly"}{"" if config.selection == "kcenter" else "_" + config.selection}'
           f'{storage_suffix(config)}')
    out_dir = Path(out_dir)
    store = TokenStore(config.backbone, config.device, cache_dir)
    model = RigelCAD(config, keep_history=True)
    splits = {c: DATA.category_split(dataset, data_root, c) for c in tasks}
    t0 = time.time()
    bytes_per_stage = []
    for c in tasks:
        train, _ = splits[c]
        tok = store.get(f'{dataset}/{c}_train', data_root, [s.relative_path for s in train])
        model.fit_task(tok.reshape(-1, TOKEN_DIM).to(config.device), f'{dataset}/{c}')
        bytes_per_stage.append(model.history[-1].bytes)
        log('TASK', tag, c, 'keys', len(model.fast), 'bytes', bytes_per_stage[-1])
        del tok
        torch.cuda.empty_cache() if config.device == 'cuda' else None
    per_task, boots, lat = {}, {}, []
    T = len(tasks)
    for j, c in enumerate(tasks):
        _, test = splits[c]
        q = store.get(f'{dataset}/{c}_test', data_root, [s.relative_path for s in test])
        grids = []
        for l in range(j, T):
            t1 = time.time()
            grids.append(model.patch_energy_grid(q, model.state_at(l)))
            lat.append((time.time() - t1) / len(test))
        grid = torch.stack(grids)
        if save_grids:
            p = out_dir / 'grids' / tag / f'{c}.pt'
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'rows': [s.relative_path for s in test], 'grid': grid, 'stages': list(range(j, T))}, p)
        cat = M.Category(data_root, test)
        bw = M.boot_weights(len(cat.y), j) if bootstrap else None
        per_task[f'{dataset}/{c}'], b = M.evaluate_category(cat, grid, list(range(j, T)), bw)
        if b is not None:
            boots[f'{dataset}/{c}'] = b
        log('EVAL', tag, c, {k: round(v, 4) for k, v in per_task[f'{dataset}/{c}'][T - 1].items()})
    order_names = [f'{dataset}/{c}' for c in tasks]
    res = {'tag': tag, 'method': 'RIGEL-CAD', 'storage': config.storage, 'config': config.to_dict(), 'dataset': dataset,
           'order': order_names,
           **M.stream_summary(order_names, per_task),
           'per_task_stage': per_task, 'bytes_per_stage': bytes_per_stage, 'bytes_max': max(bytes_per_stage),
           'update_seconds': model.update_seconds, 'latency_ms_per_img_p50': float(np.median(lat) * 1e3),
           'seconds': time.time() - t0}
    if boots:
        res['pixel_ap_ci95'] = M.ci95([boots[t]['ap'] for t in order_names])
        res['image_auroc_ci95'] = M.ci95([boots[t]['auroc'] for t in order_names])
        np.savez_compressed(out_dir / f'BOOT_{tag}.npz', **{f'{t}||{m}': v[m] for t, v in boots.items() for m in v})
    write_json(out_dir / f'RESULTS_{tag}.json', res)
    model.save(out_dir / 'states' / f'{tag}.rigel')
    log('DONE', tag, 'pixel AP %.4f image AUROC %.4f' % (res['pixel_ap'], res['image_auroc']))
    return res


# ====================================================================== MVTec AD 2
def ad2_stream_summary(per_scenario: dict, scens) -> dict:
    """Forgetting and acquisition of test_public AU-PRO 0.05 from a per-stage report {scenario: {stage: cell}}
    (stage keys int or str). FM as in NL03-MULTIRATE-01 ad2_aupro.py: mean over scenarios but the last of
    max over stages j..T-2 minus final."""
    T = len(scens)
    c = lambda s, l: (per_scenario[s].get(l) or per_scenario[s].get(str(l)))['au_pro_0.05']
    return {'fm_au_pro_0.05': float(np.mean([max(c(s, l) for l in range(j, T - 1)) - c(s, T - 1)
                                             for j, s in enumerate(scens[:-1])])),
            'acq_au_pro_0.05': float(np.mean([c(s, j) for j, s in enumerate(scens)]))}


def run_ad2(data_root, config: RigelConfig, out_dir='results/rigel_cad/ad2', cache_dir=None,
            per_stage: bool = False, write_submission: bool = True) -> dict:
    """Continual stream over the AD 2 scenarios; thresholds from validation at acquisition; local test_public AU-PRO
    and SegF1 (good images scored against an all-normal mask, bad images against their ground truth); submission folder
    for test_private and test_private_mixed (final-stage maps) unless write_submission=False."""
    root, out_dir = Path(data_root), Path(out_dir)
    scens = DATA.ad2_scenarios(root)
    T = len(scens)
    store = TokenStore(config.backbone, config.device, cache_dir)
    model = RigelCAD(config, keep_history=True)
    theta, theta_stats = {}, {}
    t0 = time.time()
    hws_of = lambda rels: [M.native_hw(root / r) for r in rels]
    for l, s in enumerate(scens):
        tr = DATA.ad2_files(root, s, 'train')
        model.fit_task(store.get(f'ad2/{s}_train', root, tr).reshape(-1, TOKEN_DIM).to(config.device), s)
        va = DATA.ad2_files(root, s, 'validation')
        g = model.patch_energy_grid(store.get(f'ad2/{s}_validation', root, va))
        theta[s], theta_stats[s] = AD2.validation_threshold(g, hws_of(va))
        log('AD2 TASK', s, 'theta %.3f' % theta[s])
    sfx = storage_suffix(config)
    sub = out_dir / f'submission{sfx}'
    report, mask_audit = {}, {}
    for j, s in enumerate(scens):
        tp = DATA.ad2_files(root, s, 'test_public')
        q, hws = store.get(f'ad2/{s}_test_public', root, tp), hws_of(tp)
        mpaths = [DATA.ad2_mask_of(root, r) for r in tp]
        masks = [M.native_mask(m, hw) for m, hw in zip(mpaths, hws)]
        is_bad = [Path(r).parts[2] == 'bad' for r in tp]
        mask_audit[s] = {'n_good': is_bad.count(False), 'n_bad': is_bad.count(True),
                         'good_with_defect_pixels': sum(bool(m.any()) for m, b in zip(masks, is_bad) if not b),
                         'bad_with_defect_pixels': sum(bool(m.any()) for m, b in zip(masks, is_bad) if b)}
        assert mask_audit[s]['good_with_defect_pixels'] == 0, mask_audit[s]
        stages = range(j, T) if per_stage else [T - 1]
        report[s] = {}
        for l in stages:
            maps = readout.anomaly_maps(model.patch_energy_grid(q, model.state_at(l)))
            acc, f1 = M.ProAccumulator(), M.SegF1()
            for m, hw, gt in zip(maps, hws, masks):
                nat = readout.to_native_cv2(m, hw)
                acc.add(nat.astype(np.float16), gt)
                if l == T - 1:
                    f1.add(nat > theta[s], gt)
            cell = {'au_pro_0.05': acc.au_pro(0.05), 'au_pro_0.30': acc.au_pro(0.30)}
            if l == T - 1:
                cell['seg_f1'] = f1.value()
            report[s][l] = cell
        for split in ('test_private', 'test_private_mixed') if write_submission else ():
            rels = DATA.ad2_files(root, s, split)
            if rels:
                g = model.patch_energy_grid(store.get(f'ad2/{s}_{split}', root, rels))
                AD2.write_maps(sub, s, split, rels, hws_of(rels), g, theta[s])
        log('AD2 EVAL', s, report[s][T - 1])
    fin = {s: report[s][T - 1] for s in scens}
    res = {'method': 'RIGEL-CAD', 'storage': config.storage, 'protocol': 'MVTec AD 2 continual stream (scenarios alphabetical, train/ only)',
           'config': config.to_dict(), 'scenarios': list(scens), 'theta': theta, 'theta_stats': theta_stats,
           'test_public': {'au_pro_0.05': float(np.mean([fin[s]['au_pro_0.05'] for s in scens])),
                           'au_pro_0.30': float(np.mean([fin[s]['au_pro_0.30'] for s in scens])),
                           'seg_f1': float(np.mean([fin[s]['seg_f1'] for s in scens])), 'per_scenario': report,
                           'mask_audit': mask_audit,
                           **(ad2_stream_summary(report, list(scens)) if per_stage else {})},
           'submission_dir': str(sub) if write_submission else None, 'bytes': model.persistent_bytes(), 'seconds': time.time() - t0}
    write_json(out_dir / f'RESULTS_ad2{sfx}.json', res)
    model.save(out_dir / 'states' / f'ad2{sfx}.rigel')
    return res
