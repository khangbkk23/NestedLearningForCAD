"""Recovery controls use the original W7 run identities; only final-row evaluation is requested."""
from pathlib import Path
from copy import deepcopy
import argparse
import json
import sys
import time
import traceback
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from notebook_support.w7_resume_fix.launcher import install_json_lock_fix
install_json_lock_fix()
from scripts.hope_cad_v2.run_w7 import reviewed_context,recipes_from_w6,adapter_from_initial
from training.hope_w7_contract import job_by_id,job_protocol,validate_w6_review
from training.hope_w7_adapter import CompactCADIC,baseline_config
from training.hope_w7_engine import W7Engine,SessionLimit
from training.hope_outer_learning_v4 import restore_rng
from training.hope_w7_refit import seed_all
from training.hope_runtime_v4 import configure_runtime,runtime_identity,resource_usage
from dataset.benchmark_protocol_v2 import PrefixContinualProtocol
from models.benchmark_registry_v2 import build_adapter
from .common import (JOBS,PROTOCOL,SUITE_DIGEST,M_RUN_DIGEST,atomic_json,canon,digest,locked_json,
                     source_identity,verify_reference,ObservedAdapter,check_deadline,sha)
from .diagnostics import run_diagnostics
from .cadic_accel import accelerate,BACKEND


def setup(work,w6):
    work,w6=Path(work),Path(w6)
    source=source_identity();verify_reference(work/'reference_w7')
    context=reviewed_context(w6);recipes=recipes_from_w6(w6,context)
    suite=json.loads((work/'reference_w7/suite_lock.json').read_text())
    if suite['digest']!=SUITE_DIGEST or digest(suite['identity'])!=SUITE_DIGEST:raise ValueError('wrong parent suite')
    identity=suite['identity']
    if (identity['source_sha256']!=source['frozen'] or identity['recipes']!=canon(recipes)
        or identity['runtime']!=runtime_identity() or identity['parent_baseline_lock']!=context['lock']['digest']):
        raise ValueError('frozen source/recipe/runtime/protocol differs from completed W7.2')
    if runtime_identity()!=json.loads((w6/'pipeline_lock.json').read_text())['runtime']:
        raise ValueError('requires W6-tested runtime; do not install/upgrade packages to bypass this')
    study=locked_json(work/'recovery_lock.json',dict(schema='hope_recovery_stage1_lock_v1',
        source=source,parent_suite=SUITE_DIGEST,parent_M_run=M_RUN_DIGEST,
        normal_manifest=context['manifest']['digest'],runtime=runtime_identity()))
    from training.hope_w7_io import destination
    for name,expected in source['extras'].items():
        target=destination(work/'operations_source',name)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha(target)!=expected:raise ValueError('archived recovery helper changed')
        raw=(ROOT/name).read_bytes()
        if __import__('hashlib').sha256(raw).hexdigest()!=expected:raise ValueError('recovery source changed during setup')
        target.write_bytes(raw)
    atomic_json(dict(schema='recovery_ops_source_archive_v1',files=source['extras'],
        frozen_reference_digest=digest(source['frozen']),protocol=PROTOCOL),work/'operations_source/identity.json')
    context['w7_recipes']=recipes
    # Original suite lock travels unchanged so controls can later reuse their states/final cells in full W7.
    locked_json(work/'suite_lock.json',identity)
    return context,suite,study


def make_adapter(job,context,w6,suite,device):
    protocol,manifest=job_protocol(context,job)
    method=baseline_config(context,job) if job['method']=='CADIC' else deepcopy(context['method'])
    seed_all(job['seed'])
    if job['method']=='CADIC':
        adapter=CompactCADIC(accelerate(build_adapter(method,device)),method['extractor']['checkpoint_sha256'])
        initial_identity=dict(baseline_config=method['id'],budget=job['budget'])
    else:
        from training.hope_initialization_v4 import validate_initialization
        p=Path(w6)/'selection'/job['method']/'deployment_initialization.pt'
        initial=torch.load(p,map_location='cpu',weights_only=False)
        receipt=validate_initialization(initial,context,selection_path=p.parent/'selection.pt')
        initial_identity=dict(reused_W6_seed0=True,receipt=receipt)
        identity=dict(protocol_id=protocol['id'],manifest_digest=manifest['digest'],
            parent_baseline_lock_digest=context['lock']['digest'],suite_digest=suite['digest'],job=job)
        adapter=adapter_from_initial(initial,context,identity,device)
        restore_rng(initial['rng'])
    dataset=PrefixContinualProtocol(protocol,manifest,method,job['seed'])
    run_identity=dict(suite_digest=suite['digest'],job=job,manifest_digest=manifest['digest'],initialization=initial_identity,
        effective_training_transport_batch_size=getattr(adapter,'train_transport_batch_size',1),
        evaluation_loader_batch_size=method['runtime']['batch_size'],effective_method=adapter.method_metadata(),runtime=runtime_identity())
    return adapter,dataset,run_identity,method,protocol,manifest


def run_control(name,context,suite,study,args,deadline):
    job=job_by_id(name);folder=args.work_dir/'runs'/name
    locked_json(folder/'job_lock.json',dict(suite_digest=suite['digest'],job=job,
        manifest_digest=job_protocol(context,job)[1]['digest'],runtime=runtime_identity()))
    adapter,dataset,run_identity,method,protocol,manifest=make_adapter(job,context,args.w6_dir,suite,args.device)
    try:
        run=locked_json(folder/'run_lock.json',run_identity)
        for filename,value in [('manifest_train.json',manifest),('protocol_resolved.json',protocol),('baseline_method_resolved.json',method)]:
            atomic_json(value,folder/filename)
        observed=ObservedAdapter(adapter,folder/'normal_training_observations')
        engine=W7Engine(observed,dataset,folder,run['digest'],device=args.device,deadline=deadline)
        engine.train()
        check_deadline(deadline)
        metrics=engine.final()
        atomic_json(dict(schema='recovery_control_final_v1',status='final_row_complete',job=job,
            suite_digest=suite['digest'],run_digest=run['digest'],final_complete=True,matrix_complete=False,
            algorithm_changed=False,calibration_refitted=False,metrics=metrics,
            execution_optimization=BACKEND if job['method']=='CADIC' else None,
            resources=resource_usage(args.device),profile=adapter.profile_metadata(),storage=adapter.memory_stats(),
            reuse='HOPE original identity; CADIC has explicit cache execution identity; verified bridge required for full W7',reportable=False),
            folder/'recovery_completion.json')
        if job['method']!='CADIC':
            run_diagnostics(adapter,dataset,engine,args.work_dir/'diagnostics'/name,name,study['digest'],deadline=deadline)
        print('[RECOVERY FINAL]',name,json.dumps(metrics),flush=True)
    finally:adapter.close()
    atomic_json(dict(job=name,run_digest=run['digest'],cleanup_complete=True),folder/'lifecycle_completion.json')


def reference_diagnostics(context,suite,study,args,deadline):
    job=job_by_id('M_seed0')
    adapter,dataset,run_identity,_,_,_=make_adapter(job,context,args.w6_dir,suite,args.device)
    try:
        # The canonical JSON transport is the ONLY adaptation; compare full original identity/digest.
        parent=json.loads((args.work_dir/'reference_w7/runs/M_seed0/run_lock.json').read_text())
        if parent!=dict(identity=canon(run_identity),digest=digest(run_identity)):
            raise ValueError('reference M run identity differs from original frozen runner')
        engine=W7Engine(adapter,dataset,args.work_dir/'reference_w7/runs/M_seed0',parent['digest'],
            device=args.device,deadline=deadline)
        return run_diagnostics(adapter,dataset,engine,args.work_dir/'diagnostics/M_seed0','M_seed0',study['digest'],deadline=deadline)
    finally:adapter.close()


def comparison(work):
    """Descriptive only. Do not select or rank hyperparameters using these test outputs."""
    work=Path(work);rows={}
    p=work/'reference_w7/runs/M_seed0/metrics/final_macro.json'
    if p.exists():rows['M_seed0']=json.loads(p.read_text())
    for name in JOBS:
        p=work/'runs'/name/'metrics/final_macro.json'
        if p.exists():rows[name]=json.loads(p.read_text())
    deltas={}
    if 'CADIC10000_seed0' in rows:
        baseline=rows['CADIC10000_seed0']
        for name,values in rows.items():
            if name!='CADIC10000_seed0':deltas[name]={k:values[k]-baseline[k] for k in ('i_auroc','p_aupr','i_ap','p_auroc')}
    contrasts={}
    for label,a,b in [('A1_minus_C1','A1_seed0','C1_seed0'),('M_minus_A1','M_seed0','A1_seed0')]:
        if a in rows and b in rows:contrasts[label]={k:rows[a][k]-rows[b][k] for k in ('i_auroc','p_aupr','i_ap','p_auroc')}
    result=dict(schema='recovery_descriptive_comparison_v1',rows=rows,deltas_vs_matched_CADIC=deltas,
        descriptive_control_contrasts=contrasts,
        status='complete' if len(rows)==4 else 'partial',same_frozen_suite=True,
        paired_confidence_intervals_not_computed=True,selection_feedback=False,
        CADIC_execution_optimization=BACKEND,reference_parity_required=True,
        claim='development seed0 diagnosis; not superiority/noninferiority or full W7',reportable=False)
    atomic_json(result,work/'comparison.json');return result


def execute(args,deadline):
    context,suite,study=setup(args.work_dir,args.w6_dir)
    if args.phase=='preflight':
        from .preflight import run_preflight
        return run_preflight(args,context,suite,study,deadline)
    from .preflight import validate_preflight
    validate_preflight(args.work_dir,study)
    reference_diagnostics(context,suite,study,args,deadline)
    for name in args.jobs:
        check_deadline(deadline);run_control(name,context,suite,study,args,deadline)
        comparison(args.work_dir)
    result=comparison(args.work_dir)
    completed=[]
    for name in JOBS:
        final=args.work_dir/'runs'/name/'recovery_completion.json'
        diag=args.work_dir/'diagnostics'/name/'summary.json'
        lifecycle=args.work_dir/'runs'/name/'lifecycle_completion.json'
        if (final.exists() and lifecycle.exists() and json.loads(lifecycle.read_text()).get('cleanup_complete') is True
            and (name.startswith('CADIC') or diag.exists())):completed.append(name)
    return dict(status='recovery_stage1_complete_pending_review' if len(completed)==3 else 'requested_jobs_complete_remaining_pending',
        completed_controls=completed,comparison_status=result['status'],reportable=False,
        next='review artifacts before ANY method revision; no automatic W7.3/VisA')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('preflight','controls'),required=True)
    p.add_argument('--jobs',nargs='+',choices=JOBS,default=list(JOBS))
    p.add_argument('--work-dir',type=Path,required=True);p.add_argument('--w6-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--session-hours',type=float,default=10.5)
    p.add_argument('--session-start-monotonic',type=float)
    args=p.parse_args(argv)
    if not .05<=args.session_hours<=11:p.error('session-hours must be .05..11 with export reserve')
    if len(set(args.jobs))!=len(args.jobs):p.error('duplicate jobs')
    configure_runtime();start=args.session_start_monotonic or time.monotonic()
    if start>time.monotonic():p.error('future session start')
    deadline=start+args.session_hours*3600-180;args.work_dir.mkdir(parents=True,exist_ok=True)
    code=0
    try:
        check_deadline(deadline);handoff=execute(args,deadline)
    except SessionLimit as error:
        handoff=dict(status='paused_at_atomic_boundary',reason=str(error),reportable=False)
    except Exception as error:
        traceback.print_exc();code=1
        handoff=dict(status='failed_preserve_evidence',error=repr(error),reportable=False)
    handoff.update(phase=args.phase,jobs=args.jobs,elapsed_seconds=time.monotonic()-start,
        algorithm='frozen v4; recovery stage1 only',session_hours=args.session_hours)
    atomic_json(handoff,args.work_dir/'handoff.json');print(json.dumps(handoff,indent=2),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())