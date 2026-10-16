"""Fixed-recipe refits for seeds 1/2. Validation never reselects steps or rates."""
from pathlib import Path
import json
import math
import random

import numpy as np
import torch

from dataset.benchmark_manifest_v1 import canonical_digest
from dataset.benchmark_protocol_v2 import file_hash
from models.hope_cad_v2.predictor import HOPEFeaturePredictorV4
from models.hope_cad_v2.state import tensor_digest
from training.hope_feature_contract_v3 import expected_feature_identity,validate_feature_identity
from training.hope_initialization_v4 import weights_digest
from training.hope_runtime_v4 import atomic_json,atomic_torch_save,learning_source_identity,SPECS
from training.hope_outer_learning_v4 import make_schedule,fit,export_initialization,restore_rng
from training.hope_cms_stress_v4 import run_stress_gate,CONSTANTS
from training.hope_w7_contract import lock_json
from training.hope_w7_engine import SessionLimit


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_refit(initial, context, *, fit_path=None):
    """Validate the distinct refit receipt, never relabel a seed-1 fit as seed-0 selection."""
    from training.hope_initialization_v4 import predictor_from_initialization
    receipt = initial.get('w7_refit_receipt',{})
    if (initial.get('schema')!='hope_deployment_initialization_v4'
            or initial.get('selection_rule_id')!='prefix_warm_joint_stress_v4'
            or not initial.get('metadata',{}).get('canonical')
            or receipt.get('schema')!='hope_w7_fixed_refit_v1' or receipt.get('seed') not in (1,2)
            or receipt.get('status')!='accepted' or receipt.get('selection_performed') is not False
            or receipt.get('digest')!=canonical_digest({k:v for k,v in receipt.items() if k!='digest'})
            or receipt.get('weights_sha256')!=weights_digest(initial['slow_and_initial_weights'])
            or receipt.get('method')!=initial['metadata']['dynamics']['method']):
        raise ValueError('invalid fixed-refit initialization receipt')
    parent = receipt['frozen_recipe']
    if parent['digest']!=canonical_digest({k:v for k,v in parent.items() if k!='digest'}):
        raise ValueError('invalid parent seed-0 recipe')
    if context.get('w7_recipes',{}).get(receipt['method'])!=parent:
        raise ValueError('refit parent is not the reviewed W6 recipe')
    if (parent['method']!=receipt['method'] or parent['beta']!=receipt['beta']
            or parent['outer_lr']!=receipt['outer_lr'] or parent['selected_step']!=receipt['steps']
            or receipt['steps']!=400):
        raise ValueError('refit differs from the concrete frozen W6 recipe')
    provenance = initial['provenance']
    validate_feature_identity(provenance['feature_identity'],expected_feature_identity(context))
    if any(provenance['source_sha256'].get(p)!=h for p,h in learning_source_identity().items()):
        raise ValueError('refit learning-source drift')
    if (receipt['roles_digest']!=context['roles']['digest'] or initial['cursor']!=0
            or initial['cms']['clock']!=0 or any(initial['cms']['pending']) or any(initial['cms']['commits'])
            or tensor_digest(tuple(initial['cms']['weights']))!=initial['initial_cms_sha256']):
        raise ValueError('refit must export fresh zero-clock deployment state')
    gate = receipt['gate']
    method = receipt['method']
    if method in ('C0','C1'):
        if gate!={'passed':True,'status':'not_applicable_static'}:
            raise ValueError('invalid static refit acceptance')
    else:
        commits = [1024] if method=='C3' else [1024,1024] if method=='C4' else [1024,128]
        if (gate.get('passed') is not True or gate.get('constants')!=CONSTANTS
                or gate.get('events_completed')!=1024 or len(gate.get('trace',[]))!=1024
                or gate.get('commits')!=commits or gate.get('roles_digest')!=context['roles']['digest']):
            raise ValueError('fixed refit failed/incomplete acceptance gate')
    if fit_path is not None and file_hash(fit_path)!=receipt.get('fit_payload_sha256'):
        raise ValueError('fixed fit payload checksum mismatch')
    model = predictor_from_initialization(initial)
    state = model.cms.load_state(initial['cms'])
    if (tensor_digest(state.weights)!=tensor_digest(model.cms.initial_state().weights)
            or any(bool(v.any()) for acc in state.accumulators for v in acc.values())):
        raise ValueError('refit export contains adapted state')
    return receipt


def fixed_refit(data, context, recipe, root, *, seed, device, source,
                deadline=None, c1_initial=None):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    method,beta = recipe['method'],recipe['beta']
    identity = dict(seed=seed,recipe=recipe,source=source,feature_identity=expected_feature_identity(context))
    lock_json(root/'refit_lock.json',identity)
    initial_path = root/'deployment_initialization.pt'
    if initial_path.exists():
        initial = torch.load(initial_path,map_location='cpu',weights_only=False)
        validate_refit(initial,context)
        return initial
    if (root/'failure.json').exists():
        raise ValueError('preserved fixed-recipe failure; no retry/reselection allowed')
    if seed not in (1,2) or recipe['selected_step']!=400:
        raise ValueError('unregistered seed or recipe; this suite binds the reviewed 400-step recipes')
    seed_all(seed)
    schedule = make_schedule(data,seed,steps=400)
    atomic_json(schedule,root/'schedule.json')
    model = HOPEFeaturePredictorV4(method=method,beta=beta,
                                 selected_m_beta=beta if method=='C4' else None).to(device)

    def fail(value):
        atomic_torch_save(value,root/'failure.pt')
        atomic_json(dict(status='acceptance_failed',method=method,seed=seed,
                         first_failure=value.get('first_failure')),root/'failure.json')

    def check_time():
        if deadline is not None:
            import time
            if time.monotonic()>=deadline:
                raise SessionLimit('fixed refit saved at an atomic boundary')

    inherited = None
    checkpoint = root/'fit_resume.pt'
    complete = root/'fit_complete.pt'
    if method=='A1':
        if c1_initial is None:
            raise ValueError('A1 seed refit requires the accepted C1 initialization of the SAME seed')
        inherited = validate_refit(c1_initial,context)
        if inherited['method']!='C1' or inherited['seed']!=seed:
            raise ValueError('A1 C1 parent method/seed mismatch')
        model.load_state_dict(c1_initial['slow_and_initial_weights'],strict=True)
        restore_rng(c1_initial['rng'])
        atomic_torch_save(dict(status='reused_C1',seed=seed,parent_digest=inherited['digest'],
                              additional_optimizer_steps=0),complete)
    else:
        def save_fit(value):
            atomic_torch_save(value,checkpoint)
            if value['step']%25==0:
                print(f'[W7 REFIT] {method} seed={seed} fixed_step={value["step"]}/400',flush=True)
            check_time()
        if complete.exists():
            result = torch.load(complete,map_location='cpu',weights_only=False)
            terminal = torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(terminal['weights'],strict=True)
            restore_rng(terminal['rng'])
        else:
            result = fit(model,data,schedule,outer_lr=recipe['outer_lr'],max_steps=400,
                         resume=torch.load(checkpoint,map_location=device,weights_only=False) if checkpoint.exists() else None,
                         checkpoint_callback=save_fit,failure_callback=fail)
            atomic_torch_save(result,complete)
        if result['status']!='fit_complete':
            fail(result)
            raise ValueError('fixed-recipe fit failed')
        # Deliberately retain TERMINAL model weights, NOT the normal-validation shortlist.
    gate_path = root/'acceptance_gate.pt'
    if gate_path.exists():
        gate = torch.load(gate_path,map_location='cpu',weights_only=False)
    elif not model.dynamics.cms_updates:
        gate = dict(passed=True,status='not_applicable_static')
        atomic_torch_save(gate,gate_path)
    else:
        gate_resume = root/'gate_resume.pt'
        def save_gate(value):
            atomic_torch_save(value,gate_resume)
            check_time()
        gate = run_stress_gate(model,data,
            resume=torch.load(gate_resume,map_location=device,weights_only=False) if gate_resume.exists() else None,
            checkpoint_callback=save_gate,failure_callback=fail)
        atomic_torch_save(gate,gate_path)
    if not gate['passed']:
        fail(gate)
        raise ValueError('fixed-recipe gate failed; no alternate rate/step may rescue it')
    receipt = dict(schema='hope_w7_fixed_refit_v1',status='accepted',seed=seed,method=method,
        beta=beta,outer_lr=recipe['outer_lr'],steps=recipe['selected_step'],selection_performed=False,
        frozen_recipe=recipe,weights_sha256=weights_digest(model.state_dict()),gate=gate,
        roles_digest=data.manifest['digest'],schedule_digest=schedule['digest'],
        fit_payload_sha256=file_hash(complete),inherited_C1_digest=inherited['digest'] if inherited else None,
        additional_optimizer_steps=0 if method=='A1' else 400)
    receipt['digest'] = canonical_digest(receipt)
    provenance = dict(roles_digest=data.manifest['digest'],schedule_digest=schedule['digest'],
        source_sha256=source,spec_sha256={p:source[p] for p in SPECS},
        feature_identity=expected_feature_identity(context),selection_receipt=recipe)
    initial = export_initialization(model,provenance)
    initial['w7_refit_receipt'] = receipt
    validate_refit(initial,context,fit_path=complete)
    atomic_torch_save(initial,initial_path)
    atomic_json(dict(seed=seed,method=method,status='accepted',selection_performed=False,
                     recipe=recipe,receipt_digest=receipt['digest']),root/'refit_summary.json')
    return initial
