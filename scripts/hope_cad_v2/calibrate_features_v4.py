"""W4 orchestration over an explicit role-scoped frozen-feature bundle.

Real bundle production/ViT cache integration is W2/W5. Never substitute synthetic
features for missing real assets. --synthetic-smoke has its own nonreportable run.
"""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from dataset.hope_mask_views import patch_masks
from models.hope_cad_v2.predictor import HOPEFeaturePredictorV4 as HOPEFeaturePredictor
from models.hope_cad_v2.visual_block import VisualBlockConfig
from models.hope_cad_v2.state import tensor_digest
from training.hope_outer_learning_v4 import PrefixFeatures, make_schedule, export_initialization
from training.hope_selection_v4 import search
from scripts.hope_cad_v2.synthetic import feature_fixture
from scripts.hope_cad_v2.artifacts import provenance, write_json
from training.hope_runtime_v4 import configure_runtime, runtime_identity, source_identity, atomic_torch_save, SPECS
from training.hope_feature_contract_v3 import load_context, validate_bundle
from training.hope_initialization_v4 import selection_receipt, validate_initialization
from training.hope_search_journal_v4 import SearchJournal
from dataset.benchmark_manifest_v1 import canonical_digest
from dataset.benchmark_protocol_v2 import file_hash
from scripts.hope_cad_v2.diagnostics_v4 import visual_diagnostics
import json
import time
from training.hope_runtime_v4 import resource_usage

MIN_SCIENTIFIC_GPU_BYTES = 12 * 1024**3


def canonical_resource_block(device, cuda_available, total_memory_bytes):
    if not str(device).startswith('cuda') or not cuda_available:
        return 'scientific W6 selection requires a CUDA GPU with at least 12 GiB; no reduced substitute'
    if total_memory_bytes is None or total_memory_bytes < MIN_SCIENTIFIC_GPU_BYTES:
        available = 0 if total_memory_bytes is None else int(total_memory_bytes)
        return (f'scientific W6 selection requires >=12 GiB CUDA memory; '
                f'available={available} bytes; no reduced substitute')
    return None


def bundle_provider(bundle, device, context=None):
    if context is None:
        raise ValueError('a byte-verified shared protocol context is required')
    validate_bundle(bundle, context)
    if bundle.get('schema') != 'hope_frozen_prefix_features_v3' or not bundle.get('feature_identity'):
        raise ValueError('invalid frozen feature bundle')
    required = ('backbone_sha256','preprocessing_sha256','mask_sha256','source_sha256')
    if any(not bundle['feature_identity'].get(k) for k in required):
        raise ValueError('feature identity is incomplete')
    allowed = {r['relative_path'] for role in ('meta_support','meta_query','normal_validation')
               for r in bundle['role_manifest']['roles'][role]}
    if set(bundle['images']) != allowed:
        raise ValueError('feature bundle must contain precisely the permitted calibration identities')
    for identity,row in bundle['images'].items():
        tensors = row['tensors']
        if set(tensors) != {'masked_0','masked_1','masked_2','masked_3','teacher'}:
            raise ValueError('invalid image feature keys')
        if row['sha256'] != tensor_digest((tensors,)):
            raise ValueError('cached tensor hash mismatch')
        for value in tensors.values():
            if value.shape != (1,784,768) or value.dtype != torch.float32 or value.requires_grad or not bool(torch.isfinite(value).all()):
                raise ValueError('invalid frozen feature shape/dtype/graph/finiteness')
        norms = tensors['teacher'].norm(dim=-1)
        if not bool(((norms-1).abs() < 1e-5).all()):
            raise ValueError('expected normalized frozen teacher target')
    masks = patch_masks(device=device)
    def provider(identity, view):
        row = bundle['images'][identity]['tensors']
        return row[f'masked_{view}'].to(device), row['teacher'].to(device), masks[view]
    return PrefixFeatures(bundle['role_manifest'], provider)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--bundle', type=Path)
    source.add_argument('--synthetic-smoke', action='store_true')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--method', choices=('M','C0','C1','C2','C3','C4'), default='M')
    p.add_argument('--selected-m', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--protocol', default='conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml')
    p.add_argument('--baseline-method', default='conf/benchmarks/methods/cadic_matched_prefix_v2.yaml')
    p.add_argument('--lock-dir', type=Path)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    configure_runtime()
    started = time.perf_counter()
    if str(args.device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(args.device)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise ValueError('use a new output directory; never overwrite a selection history')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    evidence = provenance()
    if not args.synthetic_smoke:
        memory_bytes = (torch.cuda.get_device_properties(args.device).total_memory
                        if str(args.device).startswith('cuda') and torch.cuda.is_available()
                        else None)
        blocked = canonical_resource_block(args.device, torch.cuda.is_available(), memory_bytes)
        if blocked:
            write_json(args.output_dir/'source.json', evidence)
            write_json(args.output_dir/'selection.json', dict(
                status='pending_external_gpu', reportable=False, smoke=False,
                method=args.method, selection_rule_id='prefix_warm_joint_stress_v4',
                required_gpu_bytes=MIN_SCIENTIFIC_GPU_BYTES,
                available_gpu_bytes=memory_bytes, reason=blocked))
            print(blocked)
            return 2
    config = VisualBlockConfig(4,3,4,2,4,4) if args.synthetic_smoke else VisualBlockConfig()
    if args.synthetic_smoke:
        data = feature_fixture(config,args.device)
        feature_identity = {'synthetic':True}
        context = None
    else:
        if args.lock_dir is None:
            p.error('--lock-dir is required for scientific calibration')
        context = load_context(args.protocol, args.baseline_method, args.lock_dir)
        bundle = torch.load(args.bundle,map_location='cpu',weights_only=False)
        data = bundle_provider(bundle,args.device,context)
        feature_identity = {**bundle['feature_identity'], 'bundle_sha256': file_hash(args.bundle)}
    steps = 1 if args.synthetic_smoke else 400
    schedule = make_schedule(data,0,steps=steps)
    identity = dict(method=args.method, smoke=args.synthetic_smoke, source=source_identity(),
                    runtime=runtime_identity(), features=feature_identity, schedule_digest=schedule['digest'],
                    selected_m_sha256=file_hash(args.selected_m) if args.selected_m else None)
    lock = dict(identity=identity, digest=canonical_digest(identity))
    lock_path = args.output_dir/'selection_lock.json'
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding='utf-8')) != lock:
            raise ValueError('selection resume source/runtime/data/recipe mismatch')
    elif args.resume and any(args.output_dir.glob('*.pt')):
        raise ValueError('unlocked historical selection cannot be resumed')
    else:
        write_json(lock_path, lock)
    if (args.output_dir/'selection.pt').is_file():
        previous = torch.load(args.output_dir/'selection.pt', map_location='cpu', weights_only=False)
        if previous['status'] != 'selected':
            print('selection_failed (preserved; failed candidates are not retried)')
            return 1
        # Recreate missing final exports after a crash using the completed outcome.
    else:
        previous = None
    write_json(args.output_dir/'source.json',evidence)
    write_json(args.output_dir/'schedule.json',schedule)
    selected_m = torch.load(args.selected_m,map_location='cpu',weights_only=False) if args.selected_m else None
    if selected_m is not None and selected_m.get('feature_identity') != feature_identity:
        raise ValueError('selected M used different frozen features/backbone/provenance')
    if selected_m is not None and context is not None:
        validate_initialization(torch.load(args.selected_m.parent/'deployment_initialization.pt',
                                map_location='cpu', weights_only=False), context,
                                selection_path=args.selected_m)
    factory = lambda method,beta,selected: HOPEFeaturePredictor(config,method,beta,selected).to(args.device)
    options = dict(max_steps=400,checkpoints=(0,25,50,100,200,300,400))
    if args.synthetic_smoke: options.update(max_steps=1,checkpoints=(0,1))
    outcome = previous or search(factory,data,schedule,method=args.method,selected_m=selected_m,
                     smoke=args.synthetic_smoke,fit_options=options,
                     journal=SearchJournal(args.output_dir,data,device=args.device))
    outcome['feature_identity'] = feature_identity
    outcome['provenance'] = evidence
    if args.selected_m is not None:
        outcome['inherited_m_selection_sha256'] = file_hash(args.selected_m)
    if previous is None:
        atomic_torch_save(outcome,args.output_dir/'selection.pt')
    summaries = []
    for index,c in enumerate(outcome['candidates']):
        if 'gate' in c: write_json(args.output_dir/f'candidate_{index}_gate.json',c['gate'])
        summaries.append({k:v for k,v in c.items() if k not in ('shortlist','gate')})
        summaries[-1]['shortlist'] = {k:v for k,v in c['shortlist'].items() if k != 'weights'} if c['shortlist'] else None
    selected = outcome['selected']
    write_json(args.output_dir/'selection.json',dict(status=outcome['status'],method=args.method,
               selection_rule_id='prefix_warm_joint_stress_v4',reportable=False,smoke=args.synthetic_smoke,
               selected_beta=selected['beta'] if selected and args.method not in ('C0','C1') else None,
               selected_outer_lr=selected['outer_lr'] if selected else None,
               selected_step=selected['shortlist']['step'] if selected else None,
               candidates=summaries,feature_identity=feature_identity))
    if selected:
        model = factory(args.method,selected['beta'],selected['beta'] if args.method=='C4' else None)
        model.load_state_dict(selected['shortlist']['weights'],strict=True)
        hashes = source_identity()
        record = dict(roles_digest=data.manifest['digest'],schedule_digest=schedule['digest'],
                      source_sha256=hashes,
                      spec_sha256={k:hashes[k] for k in SPECS},
                      feature_identity=feature_identity)
        if not args.synthetic_smoke:
            record['selection_receipt'] = selection_receipt(outcome,args.output_dir/'selection.pt')
        exported = export_initialization(model,record)
        if context is not None:
            validate_initialization(exported, context, selection_path=args.output_dir/'selection.pt')
        atomic_torch_save(exported,args.output_dir/'deployment_initialization.pt')
        x,_,_ = data.get(data.ids('meta_support')[0],0,'meta_support')
        write_json(args.output_dir/'selected_diagnostics.json', dict(
            method=args.method, selected_step=selected['shortlist']['step'],
            data_scope='meta_support', identity=data.ids('meta_support')[0], view=0,
            diagnostics=visual_diagnostics(model,x)))
    print(outcome['status'])
    write_json(args.output_dir/'resources.json', dict(**resource_usage(args.device),
               elapsed_this_session_seconds=time.perf_counter()-started,
               cache_bytes=args.bundle.stat().st_size if args.bundle else 0,
               candidate_count=len(outcome['candidates']),
               complete_artifact_bytes=sum(p.stat().st_size for p in args.output_dir.rglob('*') if p.is_file())))
    return 0 if selected else 1


if __name__ == '__main__': raise SystemExit(main())
