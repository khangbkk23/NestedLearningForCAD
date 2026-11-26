"""Command line: python -m rigel_cad {stream,ad2} ... (see rigel_cad/README.md)."""
from __future__ import annotations

import argparse

from rigel_cad.config import DEFAULT_SEED, RigelConfig


def _budget(v: str):
    return None if v.lower() in ('none', 'inf', 'unbounded') else float(v)


def _config(a) -> RigelConfig:
    return RigelConfig(budget_mib=a.budget_mib, slow_level=not a.fast_only, selection=a.selection, seed=a.seed,
                       backbone=a.backbone, device=a.device)


def main(argv=None):
    p = argparse.ArgumentParser(prog='rigel_cad', description='RIGEL-CAD continual anomaly detection')
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('stream', 'ad2'):
        s = sub.add_parser(name)
        s.add_argument('--data-root', required=True)
        s.add_argument('--out', default='results/rigel_cad' if name == 'stream' else 'results/rigel_cad/ad2')
        s.add_argument('--cache-dir', default=None, help='token cache directory (optional)')
        s.add_argument('--budget-mib', type=_budget, default=8.0, help="persistent-state budget, or 'none'")
        s.add_argument('--seed', type=int, default=DEFAULT_SEED)
        s.add_argument('--fast-only', action='store_true', help='ablation: no slow level (Euclidean keys only)')
        s.add_argument('--selection', choices=('kcenter', 'fifo'), default='kcenter')
        s.add_argument('--backbone', choices=('dinov2_vitb14_reg4', 'vit_b8_in21k'), default='dinov2_vitb14_reg4')
        s.add_argument('--device', default='cuda')
        if name == 'stream':
            s.add_argument('--dataset', choices=('mvtec', 'visa', 'mpdd'), required=True)
            s.add_argument('--order', choices=('a', 'b', 'c'), default='a')
            s.add_argument('--no-bootstrap', action='store_true')
            s.add_argument('--save-grids', action='store_true')
        else:
            s.add_argument('--per-stage', action='store_true', help='test_public AU-PRO at every stage (forgetting)')
    a = p.parse_args(argv)
    from rigel_cad import pipeline
    if a.cmd == 'stream':
        pipeline.run_stream(a.dataset, a.data_root, _config(a), a.order, a.out, a.cache_dir,
                            bootstrap=False if a.no_bootstrap else None, save_grids=a.save_grids)
    else:
        pipeline.run_ad2(a.data_root, _config(a), a.out, a.cache_dir, per_stage=a.per_stage)


if __name__ == '__main__':
    main()
