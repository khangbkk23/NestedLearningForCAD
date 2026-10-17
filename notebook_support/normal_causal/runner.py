"""Finite normal-only worker. Frozen v4 dependencies are imported, never patched."""
import argparse,json,time,gc,tempfile
from pathlib import Path
import torch
from .contract import *
from .experiments import capacity_arm,latent_oracle,continuation,load,boundary,BudgetPause,state_fingerprint
from .features import NormalFeatures,encode_rows
from models.hope_cad_v2.predictor import HOPEFeaturePredictorV4
from models.hope_cad_v2.visual_block import VisualBlockConfig
from models.hope_cad_v2.state import tensor_digest
from models.feature_extractors.cadic_vit_v1 import CADICViTConfig,CADICViTFeatureExtractor
from training.hope_runtime_v4 import configure_runtime,resource_usage
from training.hope_outer_learning_v4 import episode_loss,capture_rng,restore_rng,make_schedule
from training.hope_feature_contract_v3 import load_context
from scripts.hope_cad_v2.calibrate_features_v4 import bundle_provider


def parent_model(path,device):
    payload=load(path);session=payload['adapter']['session'];metadata=session['metadata']
    if payload['task_id'] not in PROTOCOL['checkpoint_indices']:raise ValueError('unregistered parent task')
    if metadata['dynamics']['method']!='M' or metadata['dynamics']['rates']!=[.01,.01] and tuple(metadata['dynamics']['rates'])!=(.01,.01):raise ValueError('wrong parent M dynamics')
    model=HOPEFeaturePredictorV4(VisualBlockConfig(**metadata['visual_config']),method='M',beta=.01).to(device).eval().requires_grad_(False)
    model.load_state_dict(session['model'],strict=True)
    if canon(model.metadata())!=canon(metadata):raise ValueError('checkpoint metadata mismatch')
    state=model.cms.load_state(session['cms'])
    if session['cursor']!=state.clock:raise ValueError('parent event cursor differs from CMS clock')
    return model,state,session['cursor']


def preflight(work,prepared,w6,device,deadline):
    target=work/'preflight.json'
    if target.exists():
        old=json.loads(target.read_text())
        if old['status']!='passed':raise ValueError('failed preflight cannot be silently retried')
        return
    boundary(deadline);started=time.monotonic()
    # New fixture tests, then real canonical shapes/gradients/peak memory.
    import unittest
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/normal_causal'))
    checked=unittest.TextTestRunner(verbosity=1).run(suite)
    if not checked.wasSuccessful() or checked.testsRun<10:raise RuntimeError('normal diagnostic contract tests failed/missing')
    torch.cuda.reset_peak_memory_stats(device)
    context=load_context('conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml','conf/benchmarks/methods/cadic_matched_prefix_v2.yaml',w6/'baseline_lock')
    data=bundle_provider(load(prepared['prefix_cache']),device,context)
    parent=load(w6/FIT_ENTRY);model=HOPEFeaturePredictorV4(beta=.01).to(device).eval();model.load_state_dict(parent['weights'])
    schedule=make_schedule(data,0,steps=500);rng=capture_rng();t0=time.monotonic()
    try:
        loss=episode_loss(model,data,schedule['episodes'][400]);loss.backward()
        for name,param in model.named_parameters():
            if param.requires_grad and (param.grad is None or not bool(torch.isfinite(param.grad).all())):raise RuntimeError('canonical outer gradient failed: '+name)
        torch.cuda.synchronize();outer_seconds=time.monotonic()-t0
    finally:restore_rng(rng)
    value=float(loss.detach());del loss,model,data,parent;gc.collect();torch.cuda.empty_cache()
    model,state,cursor=parent_model(work/'parents/recovery/reference_w7/runs/M_seed0/states/task_01.pt',device)
    train=json.loads((work/'parents/recovery/reference_w7/runs/M_seed0/manifest_train.json').read_text());samples=json.loads((work/'parents/recovery/diagnostics/M_seed0/sample_manifest.json').read_text())
    rows=legal_rows(train,samples,1);allowed={r['relative_path']:r for r in rows['fit']+rows['query']+rows['past']}
    extractor=CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=prepared['checkpoint'],checkpoint_identity='timm/vit_base_patch8_224.augreg_in21k'),device)
    provider=NormalFeatures(prepared['dataset'],extractor,Path(prepared['prefix_cache']).parent/'normal_cache',work/'feature_receipts',device)
    one=encode_rows(model,provider,rows['fit'][:1],allowed)[0]
    groups={'fixture_real_image':one}
    t0=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='normal_causal_preflight_') as tmp:
        for arm in PROTOCOL['arms']:
            capacity_arm(model,state,groups,[one]*4,cursor,arm,Path(tmp)/arm,identity={'canonical_preflight':True},steps=1,checkpoints=[0,1],deadline=deadline)
        latent_oracle(model,state,one[0],Path(tmp)/'oracle',identity={'canonical_preflight':True},steps=1,deadline=deadline)
    profile=dict(status='passed',tests=checked.testsRun,canonical=True,outer_one_episode_forward_backward_seconds=outer_seconds,
        outer_example_loss=value,capacity_four_arms_one_step_and_oracle_seconds=time.monotonic()-t0,
        elapsed_seconds=time.monotonic()-started,resource=resource_usage(device),
        estimate_rule='100 measured outer episodes plus validation, 1152 encoded capacity events, 3072 decoder-only latent steps; timing is a profile, not a wall-clock guarantee',reportable=False)
    atomic_json(profile,target);print('[PREFLIGHT PASS]',json.dumps(profile),flush=True)
    del extractor,model,state,one;gc.collect();torch.cuda.empty_cache()


def run(work,w6,prepared,phase,device,deadline):
    preflight(work,prepared,w6,device,deadline)
    if phase=='preflight':return 'preflight_passed'
    ref=work/'parents/recovery';train=json.loads((ref/'reference_w7/runs/M_seed0/manifest_train.json').read_text());samples=json.loads((ref/'diagnostics/M_seed0/sample_manifest.json').read_text())
    base=dict(lock_digest=json.loads((work/'diagnostic_lock.json').read_text())['digest'])
    # Continuation first: still only task-0 prefix, with the exact reviewed W6 cache.
    if not (work/'continuation/result.json').exists():
        boundary(deadline);print('[CONTINUATION] selected M recipe only, 400 -> 500; no new selection',flush=True)
        context=load_context('conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml','conf/benchmarks/methods/cadic_matched_prefix_v2.yaml',w6/'baseline_lock')
        data=bundle_provider(load(prepared['prefix_cache']),device,context);parent=load(w6/FIT_ENTRY)
        model=HOPEFeaturePredictorV4(beta=.01).to(device).eval()
        continuation(model,data,parent,work/'continuation',identity={**base,'parent_fit_sha256':sha(w6/FIT_ENTRY)},deadline=deadline)
        del model,data,parent;gc.collect();torch.cuda.empty_cache()
    extractor=CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=prepared['checkpoint'],checkpoint_identity='timm/vit_base_patch8_224.augreg_in21k'),device)
    provider=NormalFeatures(prepared['dataset'],extractor,Path(prepared['prefix_cache']).parent/'normal_cache',work/'feature_receipts',device)
    for checkpoint in PROTOCOL['checkpoint_indices']:
        boundary(deadline);print('[CAPACITY] checkpoint',checkpoint,flush=True)
        parent_path=ref/f'reference_w7/runs/M_seed0/states/task_{checkpoint:02d}.pt';model,state,cursor=parent_model(parent_path,device)
        original=state_fingerprint(state);weights=tensor_digest((dict(model.named_parameters()),));rng=capture_rng()
        try:
            rows=legal_rows(train,samples,checkpoint);allowed={r['relative_path']:r for key in ('fit','query','past') for r in rows[key]}
            fit=encode_rows(model,provider,rows['fit'],allowed);query=encode_rows(model,provider,rows['query'],allowed);past=encode_rows(model,provider,rows['past'],allowed)
            flatten=lambda items:[e for image in items for e in image]
            groups={'current_fit':flatten(fit),'current_aux_query':flatten(query),'past_aux_query':flatten(past)}
            folder=work/'capacity'/f'task_{checkpoint:02d}'
            identity=dict(**base,checkpoint=checkpoint,parent_checkpoint_sha256=sha(parent_path),rows=rows,parent_state_digest=original)
            locked_json(folder/'case_lock.json',identity)
            for arm in PROTOCOL['arms']:
                result=capacity_arm(model,state,groups,fit,cursor,arm,folder/arm,identity=identity,deadline=deadline)
                print('[ARM COMPLETE]',checkpoint,arm,'query loss',result['trace'][-1]['groups']['current_aux_query']['loss'],flush=True)
            for image,events in enumerate(query):
                for view,event in enumerate(events):
                    latent_oracle(model,state,event,folder/'latent'/f'{image}_{view}',identity=dict(**identity,image=rows['query'][image],view=view),deadline=deadline)
            if original!=state_fingerprint(state) or weights!=tensor_digest((dict(model.named_parameters()),)):raise RuntimeError('parent model was mutated')
            atomic_json(dict(status='complete',checkpoint=checkpoint,parent_unchanged=True,fit_images=4,query_images=4,past_tasks=rows['past_tasks']),folder/'completion.json')
        finally:restore_rng(rng)
        del model,state,fit,query,past,groups;gc.collect();torch.cuda.empty_cache()
    results=list((work/'capacity').glob('task_*/[!l]*/result.json'))
    oracles=list((work/'capacity').glob('task_*/latent/*/result.json'))
    if len(results)!=12 or len(oracles)!=48:raise ValueError('incomplete diagnostic cells')
    report=dict(status='complete_pending_review',capacity_fits=len(results),latent_same_image_fits=len(oracles),continuations=1,
        capacity={str(p.relative_to(work)):json.loads(p.read_text()) for p in results},
        latent_oracles={str(p.relative_to(work)):json.loads(p.read_text()) for p in oracles},
        continuation=json.loads((work/'continuation/result.json').read_text()),
        scope='normal-only causal probes; development diagnosis, not new anomaly benchmark',
        no_method_selected=True,no_deployable_weights_exported=True,official_test_accessed=False,visa_accessed=False,reportable=False)
    atomic_json(report,work/'summary.json');return 'complete_pending_review'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=json.loads(Path(args.config).read_text());work,w6=Path(config['work']),Path(config['w6']);configure_runtime()
    deadline=config['started']+config['session_hours']*3600-180
    try:
        status=run(work,w6,config['prepared'],config['phase'],'cuda',deadline)
        atomic_json(dict(status=status,reportable=False),work/'handoff.json')
    except BudgetPause as e:
        atomic_json(dict(status='paused_at_atomic_boundary',reason=str(e),reportable=False),work/'handoff.json')
    except BaseException as e:
        atomic_json(dict(status='failed_preserve_evidence',error=repr(e),reportable=False),work/'handoff.json');raise
if __name__=='__main__':main()
