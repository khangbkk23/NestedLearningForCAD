"""A1 reuses C1 learned weights and selected M beta, with its own fixed stress gate."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from dataset.benchmark_manifest_v1 import canonical_digest
from dataset.benchmark_protocol_v2 import file_hash
from scripts.hope_cad_v2.calibrate_features import bundle_provider
from training.hope_feature_contract_v3 import load_context
from training.hope_initialization_v3 import validate_initialization, selection_receipt
from training.hope_outer_learning_v2 import export_initialization, make_schedule
from training.hope_cms_stress_v3 import run_stress_gate
from training.hope_runtime_v3 import configure_runtime, atomic_torch_save, atomic_json, source_identity, runtime_identity, SPECS
from models.hope_cad_v2.predictor import HOPEFeaturePredictor
import json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',default='conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml')
    p.add_argument('--baseline-method',default='conf/benchmarks/methods/cadic_matched_prefix_v2.yaml')
    for name in ('lock-dir','bundle','selection-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    configure_runtime()
    context=load_context(args.protocol,args.baseline_method,args.lock_dir)
    parents={}
    for name in ('M','C1'):
        folder=args.selection_root/name
        payload=torch.load(folder/'deployment_initialization.pt',map_location='cpu',weights_only=False)
        receipt=validate_initialization(payload,context,selection_path=folder/'selection.pt')
        parents[name]=(payload,receipt)
    data=bundle_provider(torch.load(args.bundle,map_location='cpu',weights_only=False),args.device,context)
    schedule=make_schedule(data,0)
    beta=parents['M'][1]['beta']
    model=HOPEFeaturePredictor(method='A1',beta=beta).to(args.device)
    model.load_state_dict(parents['C1'][0]['slow_and_initial_weights'],strict=True)
    folder=args.selection_root/'A1'; folder.mkdir(parents=True,exist_ok=True)
    lock=dict(parents={k:v[1]['digest'] for k,v in parents.items()},source=source_identity(),runtime=runtime_identity())
    lock_path=folder/'selection_lock.json'
    if lock_path.exists() and json.loads(lock_path.read_text())!=lock:
        raise ValueError('A1 resume parents/source/runtime mismatch')
    atomic_json(lock,lock_path)
    complete=folder/'selection.pt'
    if complete.exists():
        outcome=torch.load(complete,map_location='cpu',weights_only=False)
    else:
        def record_outcome(gate):
            candidate=dict(status='fit_complete',method='A1',beta=beta,outer_lr=parents['C1'][1]['outer_lr'],
                role_digest=data.manifest['digest'],gate=gate,
                shortlist=dict(step=parents['C1'][1]['selected_step'],weights=parents['C1'][0]['slow_and_initial_weights'],loss=None))
            result=dict(status='selected' if gate['passed'] else 'selection_failed',smoke=False,
                         selected=candidate if gate['passed'] else None,candidates=[candidate],common_seed=0,
                         schedule_digest=schedule['digest'],selection_rule_id='prefix_cms_stress_v3',
                         feature_identity=parents['C1'][0]['provenance']['feature_identity'],
                         inherited=lock['parents'],additional_optimizer_steps=0)
            atomic_torch_save(result,complete)
            return result
        checkpoint=folder/'gate_resume.pt'
        gate=run_stress_gate(model,data,
            resume=torch.load(checkpoint,map_location=args.device,weights_only=False) if checkpoint.exists() else None,
            checkpoint_callback=lambda value:atomic_torch_save(value,checkpoint), failure_callback=record_outcome)
        outcome=record_outcome(gate)
    atomic_json(dict(status=outcome['status'],method='A1',selected_beta=beta,reportable=False,
        additional_optimizer_steps=0,inherited=lock['parents']),folder/'selection.json')
    atomic_json(outcome['candidates'][0]['gate'],folder/'candidate_0_gate.json')
    if outcome['status']!='selected': return 1
    hashes=source_identity()
    record=dict(roles_digest=data.manifest['digest'],schedule_digest=schedule['digest'],source_sha256=hashes,
                spec_sha256={k:hashes[k] for k in SPECS},feature_identity=outcome['feature_identity'],
                selection_receipt=selection_receipt(outcome,complete),inherited=lock['parents'])
    initial=export_initialization(model,record)
    validate_initialization(initial,context,selection_path=complete)
    atomic_torch_save(initial,folder/'deployment_initialization.pt')
    print('A1 selected: C1 weights + M beta; no additional fit',flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
