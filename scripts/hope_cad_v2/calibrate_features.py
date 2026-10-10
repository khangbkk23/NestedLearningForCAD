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
from models.hope_cad_v2.predictor import HOPEFeaturePredictor
from models.hope_cad_v2.visual_block import VisualBlockConfig
from models.hope_cad_v2.state import tensor_digest
from training.hope_outer_learning_v2 import PrefixFeatures, make_schedule, export_initialization
from training.hope_selection_v3 import search
from scripts.hope_cad_v2.synthetic import feature_fixture
from scripts.hope_cad_v2.artifacts import provenance, write_json


def bundle_provider(bundle, device):
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
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('use a new output directory; never overwrite a selection history')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    config = VisualBlockConfig(4,3,4,2,4,4) if args.synthetic_smoke else VisualBlockConfig()
    if args.synthetic_smoke:
        data = feature_fixture(config,args.device)
        feature_identity = {'synthetic':True}
    else:
        bundle = torch.load(args.bundle,map_location='cpu',weights_only=False)
        data = bundle_provider(bundle,args.device)
        feature_identity = dict(**bundle['feature_identity'], bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest())
    steps = 1 if args.synthetic_smoke else 200
    schedule = make_schedule(data,0,steps=steps)
    evidence = provenance()
    write_json(args.output_dir/'source.json',evidence)
    write_json(args.output_dir/'schedule.json',schedule)
    selected_m = torch.load(args.selected_m,map_location='cpu',weights_only=False) if args.selected_m else None
    if selected_m is not None and selected_m.get('feature_identity') != feature_identity:
        raise ValueError('selected M used different frozen features/backbone/provenance')
    factory = lambda method,beta,selected: HOPEFeaturePredictor(config,method,beta,selected).to(args.device)
    def save_checkpoint(payload):
        method = payload['metadata']['dynamics']['method']
        beta = payload['metadata']['dynamics']['rates'][0]
        path = args.output_dir/f'fit_{method}_b{beta}_lr{payload["outer_lr"]}_resume.pt'
        temp = path.with_suffix('.tmp')
        payload['provenance'] = evidence
        torch.save(payload,temp)
        temp.replace(path)
    options = dict(checkpoint_callback=save_checkpoint)
    if args.synthetic_smoke: options.update(max_steps=1,checkpoints=(0,1))
    outcome = search(factory,data,schedule,method=args.method,selected_m=selected_m,
                     smoke=args.synthetic_smoke,fit_options=options)
    outcome['feature_identity'] = feature_identity
    outcome['provenance'] = evidence
    torch.save(outcome,args.output_dir/'selection.pt')
    summaries = []
    for index,c in enumerate(outcome['candidates']):
        if 'gate' in c: write_json(args.output_dir/f'candidate_{index}_gate.json',c['gate'])
        summaries.append({k:v for k,v in c.items() if k not in ('shortlist','gate')})
        summaries[-1]['shortlist'] = {k:v for k,v in c['shortlist'].items() if k != 'weights'} if c['shortlist'] else None
    selected = outcome['selected']
    write_json(args.output_dir/'selection.json',dict(status=outcome['status'],method=args.method,
               selection_rule_id='prefix_cms_stress_v3',reportable=False,smoke=args.synthetic_smoke,
               selected_beta=selected['beta'] if selected else None,
               selected_step=selected['shortlist']['step'] if selected else None,
               candidates=summaries,feature_identity=feature_identity))
    if selected:
        model = factory(args.method,selected['beta'],selected['beta'] if args.method=='C4' else None)
        model.load_state_dict(selected['shortlist']['weights'],strict=True)
        record = dict(roles_digest=data.manifest['digest'],schedule_digest=schedule['digest'],
                      source_sha256=evidence['source_sha256'],
                      spec_sha256={k:v for k,v in evidence['source_sha256'].items() if k.startswith('docs/')},
                      feature_identity=feature_identity)
        torch.save(export_initialization(model,record),args.output_dir/'deployment_initialization.pt')
    print(outcome['status'])
    return 0 if selected else 1


if __name__ == '__main__': raise SystemExit(main())
