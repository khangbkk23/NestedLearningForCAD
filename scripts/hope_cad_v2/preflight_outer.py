"""Real-prefix exact-meta/Adam checkpoint smoke; never selects a scientific model."""
import argparse
from pathlib import Path
import random
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from models.hope_cad_v2.predictor import HOPEFeaturePredictor
from models.hope_cad_v2.visual_block import VisualBlockConfig
from scripts.hope_cad_v2.calibrate_features import bundle_provider, canonical_resource_block
from scripts.hope_cad_v2.diagnostics import visual_diagnostics
from training.benchmark_engine_v1 import _state_equal
from training.hope_feature_contract_v3 import load_context
from training.hope_outer_learning_v2 import fit, make_schedule, capture_rng, restore_rng
from training.hope_runtime_v3 import (configure_runtime, atomic_json, atomic_torch_save,
                                      resource_usage, runtime_identity, source_identity)
from dataset.benchmark_protocol_v2 import file_hash


def verify_outer_resume(data, output_dir, *, device='cpu', config=None):
    """Two full exact-16 Adam steps, then reload step 1 and repeat step 2.

    Use the production fit/serialization/RNG code with an explicitly smoke-only
    step cap. The complete 200-episode schedule is preserved; no recipe or
    deployment initialization is exported. The experiment starts fresh later.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = config or VisualBlockConfig()
    rng = capture_rng()
    report = dict(passed=False, reportable=False, smoke=True,
                  selected_initialization_exported=False, support_length=16,
                  method='M', beta=1e-3, outer_lr=1e-4, completed_outer_steps=2,
                  scope='engineering only: real-prefix exact-meta Adam and checkpoint/RNG resume')
    started = time.perf_counter()
    try:
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        schedule = make_schedule(data, 0)
        atomic_json(schedule, root/'schedule.json')
        model = HOPEFeaturePredictor(config, 'M', 1e-3).to(device)
        report['canonical'] = config.canonical
        def save_full(payload):
            if payload['step'] == 1:
                atomic_torch_save(payload, root/'interrupted_step1.pt')
            if payload['step'] == 2:
                atomic_torch_save(payload, root/'uninterrupted_final.pt')
        def diagnose(step, current):
            x, _, _ = data.get(data.ids('meta_support')[0], 0, 'meta_support')
            atomic_json(visual_diagnostics(current, x), root/f'visual_step_{step}.json')
        full = fit(model, data, schedule, smoke=True, max_steps=2, checkpoints=(0, 2),
                   checkpoint_callback=save_full, diagnostic_callback=diagnose)
        atomic_torch_save(full, root/'smoke_fit.pt')
        if full['status'] != 'fit_complete':
            raise RuntimeError(f"real-prefix smoke fit failed: {full['first_failure']}")
        del model, full
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        resumed_model = HOPEFeaturePredictor(config, 'M', 1e-3).to(device)
        resume = torch.load(root/'interrupted_step1.pt', map_location=device, weights_only=False)
        resumed = fit(resumed_model, data, schedule, smoke=True, max_steps=2, checkpoints=(0, 2),
                      resume=resume,
                      checkpoint_callback=lambda p: atomic_torch_save(p, root/'resumed_final.pt'))
        if resumed['status'] != 'fit_complete':
            raise RuntimeError(f"resumed smoke fit failed: {resumed['first_failure']}")
        expected = torch.load(root/'uninterrupted_final.pt', map_location='cpu', weights_only=False)
        actual = torch.load(root/'resumed_final.pt', map_location='cpu', weights_only=False)
        for key in ('weights', 'optimizer', 'rng', 'schedule', 'metadata', 'step',
                    'feature_reads_total', 'feature_reads_at_start'):
            if not _state_equal(expected[key], actual[key]):
                raise RuntimeError(f'real-prefix outer resume mismatch: {key}')
        for key in ('training', 'short_losses', 'shortlist'):
            if not _state_equal(expected['result'][key], actual['result'][key]):
                raise RuntimeError(f'real-prefix outer resume mismatch: result.{key}')
        report.update(passed=True, exact_weights_optimizer_rng_resume=True,
                      schedule_digest=schedule['digest'], training=resumed['training'],
                      short_losses=resumed['short_losses'],
                      uninterrupted_fit_seconds=expected['elapsed_seconds'],
                      checkpoint_sha256={p.name: file_hash(p) for p in root.glob('*.pt')})
    finally:
        restore_rng(rng)
    report.update(elapsed_seconds=time.perf_counter()-started, resources=resource_usage(device))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', default='conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml')
    p.add_argument('--baseline-method', default='conf/benchmarks/methods/cadic_matched_prefix_v2.yaml')
    for key in ('lock-dir', 'bundle', 'output-dir'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--local-cpu-audit', action='store_true', help='CPU audit only; cannot satisfy Kaggle acceptance')
    args = p.parse_args()
    configure_runtime()
    report = dict(passed=False, reportable=False, status='failed',
                  kaggle_gate_eligible=False, source_sha256=source_identity(), runtime=runtime_identity())
    started = time.perf_counter()
    try:
        local = args.local_cpu_audit and args.device == 'cpu'
        memory = (torch.cuda.get_device_properties(args.device).total_memory
                  if args.device.startswith('cuda') and torch.cuda.is_available() else None)
        blocked = canonical_resource_block(args.device, torch.cuda.is_available(), memory)
        if blocked and not local:
            raise ValueError(blocked)
        if memory is not None:
            torch.cuda.reset_peak_memory_stats(args.device)
        context = load_context(args.protocol, args.baseline_method, args.lock_dir)
        bundle = torch.load(args.bundle, map_location='cpu', weights_only=False)
        data = bundle_provider(bundle, args.device, context)
        report.update(verify_outer_resume(data, args.output_dir, device=args.device))
        report.update(status='passed', kaggle_gate_eligible=not local,
                      feature_identity=bundle['feature_identity'], bundle_sha256=file_hash(args.bundle),
                      roles_digest=context['roles']['digest'])
    except (ValueError, RuntimeError, FloatingPointError, OSError) as error:
        report.update(passed=False, status='failed', error_type=type(error).__name__, error=str(error))
    report['total_seconds'] = time.perf_counter()-started
    atomic_json(report, args.output_dir/'preflight_outer.json')
    print(report['status'], flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
