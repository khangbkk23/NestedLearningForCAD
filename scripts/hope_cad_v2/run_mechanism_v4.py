"""Evaluate the held-out normal probe only after independently locked selections."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from dataset.hope_mask_views import patch_masks
from scripts.hope_cad_v2.calibrate_features_v4 import bundle_provider
from training.hope_feature_contract_v3 import load_context, validate_bundle
from training.hope_initialization_v4 import validate_initialization, predictor_from_initialization
from training.hope_outer_learning_v4 import make_schedule
from training.hope_mechanism_v4 import mechanism_report
from training.hope_runtime_v4 import configure_runtime, atomic_json, resource_usage, source_identity
from dataset.benchmark_protocol_v2 import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', default='conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml')
    p.add_argument('--baseline-method', default='conf/benchmarks/methods/cadic_matched_prefix_v2.yaml')
    p.add_argument('--lock-dir', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--probe-bundle', type=Path, required=True)
    p.add_argument('--selection-root', type=Path, required=True)
    p.add_argument('--methods', nargs='+', default=['M','C0','C1','C2'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    configure_runtime()
    report = dict(status='failed', engineering_passed=False, reportable=False,
                  official_test_accessed=False, source_sha256=source_identity())
    try:
        context = load_context(args.protocol,args.baseline_method,args.lock_dir)
        models, receipts = {}, {}
        for method in args.methods:
            folder = args.selection_root/method
            payload = torch.load(folder/'deployment_initialization.pt', map_location='cpu', weights_only=False)
            receipts[method] = validate_initialization(payload, context, selection_path=folder/'selection.pt')
            if receipts[method]['method'] != method:
                raise ValueError('selection folder/method mismatch')
            models[method] = predictor_from_initialization(payload,args.device)
        data = bundle_provider(torch.load(args.bundle,map_location='cpu',weights_only=False),args.device,context)
        cache = validate_bundle(torch.load(args.probe_bundle,map_location='cpu',weights_only=False),context,role='probe')
        masks = patch_masks(device=args.device)
        probe = [(identity, [(row['tensors'][f'masked_{view}'].to(args.device),
                             row['tensors']['teacher'].to(args.device), masks[view]) for view in range(4)])
                 for identity,row in sorted(cache['images'].items())]
        if str(args.device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(args.device)
        report.update(mechanism_report(models,data,make_schedule(data,0),probe))
        report.update(selection_receipts=receipts, resources=resource_usage(args.device),
                      probe_bundle_sha256=file_hash(args.probe_bundle),
                      calibration_bundle_sha256=file_hash(args.bundle),
                      W7_authorized=False, review_required=True)
    except (ValueError,RuntimeError,FloatingPointError) as error:
        report.update(status='failed',engineering_passed=False,error=str(error))
    atomic_json(report,args.output)
    print(report['status'], flush=True)
    return 0 if report['engineering_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
