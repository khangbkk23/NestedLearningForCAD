"""Exact 16-event resource spike. --reduced is explicitly noncanonical smoke only."""
import argparse
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from models.hope_cad_v2.predictor import HOPEFeaturePredictor
from models.hope_cad_v2.visual_block import VisualBlockConfig
from training.hope_outer_learning_v2 import make_schedule, episode_loss
from scripts.hope_cad_v2.synthetic import feature_fixture
from scripts.hope_cad_v2.artifacts import provenance, write_json


def ram_bytes():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--reduced', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(103)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = dict(**provenance(), scope='synthetic exact 16-support meta graph; no CAD performance claim',
                  canonical=not args.reduced, reportable=False, status='running', support_events=16,
                  create_graph=True, input_requires_frozen_backbone=False)
    if not args.reduced and (not args.device.startswith('cuda') or not torch.cuda.is_available()
                            or torch.cuda.get_device_properties(args.device).total_memory < 12*1024**3):
        report.update(status='pending_external_gpu', reason='canonical resource gate requires one CUDA GPU with >=12GiB; no reduced substitute')
        write_json(args.output, report)
        print(report['reason'])
        return 2
    config = VisualBlockConfig(8,8,8,4,4,4) if args.reduced else VisualBlockConfig()
    model = HOPEFeaturePredictor(config).to(args.device)
    data = feature_fixture(config, args.device)
    schedule = make_schedule(data, 0, steps=1)
    report.update(metadata=model.metadata(), schedule=schedule, ram_before_bytes=ram_bytes())
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    try:
        loss = episode_loss(model, data, schedule['episodes'][0])
        if args.device.startswith('cuda'): torch.cuda.synchronize(args.device)
        report['forward_seconds'] = time.perf_counter()-started
        backward = time.perf_counter()
        loss.backward()
        if args.device.startswith('cuda'): torch.cuda.synchronize(args.device)
        report['backward_seconds'] = time.perf_counter()-backward
        report['gradients'] = {name: dict(connected=p.grad is not None,
                                  finite=p.grad is not None and bool(torch.isfinite(p.grad).all()),
                                  norm=p.grad.double().norm().item() if p.grad is not None and bool(torch.isfinite(p.grad).all()) else None)
                               for name, p in model.named_parameters() if p.requires_grad}
        report['status'] = 'passed' if bool(torch.isfinite(loss)) and all(g['finite'] for g in report['gradients'].values()) else 'failed_nonfinite_or_disconnected'
        report['loss'] = loss.item() if bool(torch.isfinite(loss)) else None
    except (RuntimeError, FloatingPointError, ValueError) as error:
        report.update(status='failed', reason=str(error))
    report.update(total_seconds=time.perf_counter()-started, ram_after_bytes=ram_bytes(), reads=data.reads)
    if args.device.startswith('cuda'):
        report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
    write_json(args.output, report)
    print(report['status'])
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__': raise SystemExit(main())
