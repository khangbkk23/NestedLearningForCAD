"""Three-part frozen-v4 MVTec benchmark. Never performs test-driven selection."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch

from dataset.benchmark_manifest_v1 import canonical_digest
from dataset.benchmark_protocol_v2 import PrefixContinualProtocol,file_hash
from models.hope_cad_v2.benchmark_adapter_v4 import HOPEBenchmarkAdapterV4
from models.benchmark_registry_v2 import build_adapter
from training.hope_feature_contract_v3 import load_context
from training.hope_initialization_v4 import validate_initialization
from training.hope_outer_learning_v4 import restore_rng
from training.hope_runtime_v4 import configure_runtime,runtime_identity,atomic_json,resource_usage
from training.hope_w7_contract import (METHODS,MVTEC_COUNTS,jobs,job_by_id,job_protocol,source_hashes,
                                      lock_json,validate_w6_review)
from training.hope_w7_engine import W7Engine,SessionLimit
from training.hope_w7_refit import fixed_refit,validate_refit,seed_all
from training.hope_w7_adapter import CompactCADIC,baseline_config
from training.hope_w7_io import export_archive


def reviewed_context(w6):
    validate_w6_review(w6)
    context=load_context('conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml',
                         'conf/benchmarks/methods/cadic_matched_prefix_v2.yaml',Path(w6)/'baseline_lock')
    if context['protocol']['dataset']['name']!='MVTec AD' or len(context['protocol']['task_order'])!=15:
        raise ValueError('W7 is the full 15-category MVTec protocol')
    # Bind completeness to the actual reviewed train manifest; reject missing AND extra train files.
    root=Path(context['protocol']['dataset']['root'])
    found={p.relative_to(root).as_posix() for name in context['protocol']['task_order']
           for p in (root/name/'train/good').glob('*.png')}
    expected={r['relative_path'] for r in context['manifest']['entries']}
    if found!=expected:
        raise ValueError('current normal training inventory differs from reviewed W6')
    for name,counts in MVTEC_COUNTS.items():
        if sum(row['task_name']==name for row in context['manifest']['entries'])!=counts[0]:
            raise ValueError('reviewed train manifest is not the complete official MVTec split')
    return context


def recipes_from_w6(w6,context):
    recipes={}
    for name in METHODS:
        folder=Path(w6)/'selection'/name
        initial=torch.load(folder/'deployment_initialization.pt',map_location='cpu',weights_only=False)
        recipes[name]=validate_initialization(initial,context,selection_path=folder/'selection.pt')
        if recipes[name]['selected_step']!=400:
            raise ValueError('this suite is bound to the reviewed all-400-step recipes')
    return recipes


def calibration_ledger(w6):
    records=[]
    for method in METHODS:
        outcome=torch.load(Path(w6)/'selection'/method/'selection.pt',map_location='cpu',weights_only=False)
        for candidate in outcome['candidates']:
            gate=candidate.get('gate',{})
            records.append(dict(method=method,beta=candidate['beta'],outer_lr=candidate['outer_lr'],
                optimizer_steps=len(candidate.get('training',[])),
                fit_seconds=candidate.get('elapsed_seconds',0.0) if method!='A1' else 0.0,
                gate_seconds=gate.get('elapsed_seconds',0.0),gate_events=gate.get('events_completed',0),
                feature_reads=candidate.get('feature_reads',{}),
                inherited_C1_without_additional_fit=method=='A1'))
    return dict(scope='original W6 search cost counted once, not free seed0 initialization',
                candidates=records,total_fit_seconds=sum(r['fit_seconds'] for r in records),
                total_gate_seconds=sum(r['gate_seconds'] for r in records))


def adapter_from_initial(initial,context,protocol_identity,device):
    extractor=context['method']['extractor']
    return HOPEBenchmarkAdapterV4(initial,checkpoint_path=extractor['checkpoint_path'],
        checkpoint_sha256=extractor['checkpoint_sha256'],checkpoint_identity=extractor['checkpoint_identity'],
        device=device,protocol_identity=protocol_identity,context=context)


def execute(args,deadline):
    work=Path(args.work_dir)
    context=reviewed_context(args.w6_dir)
    recipes=recipes_from_w6(args.w6_dir,context)
    context['w7_recipes']=recipes
    source=source_hashes()
    runtime=runtime_identity()
    expected_runtime=json.loads((Path(args.w6_dir)/'pipeline_lock.json').read_text())['runtime']
    if runtime!=expected_runtime:
        raise ValueError('W7 reference runtime differs from reviewed W6; run a documented parity gate first')
    suite_identity=dict(schema='hope_w7_suite_v1',design_id='hope_cad_v4_warm_joint_cms_row_retention',
        jobs=jobs(),recipes=recipes,source_sha256=source,runtime=runtime,
        w6_archive_sha256=validate_w6_review(args.w6_dir)['archive_sha256'],
        parent_baseline_lock=context['lock']['digest'],
        primary_comparison='M vs CADIC10000',seeds=[0,1,2],
        bootstrap=dict(replicates=2000,seed=20261009,primary_percentiles=[1.25,98.75],
                       secondary_percentiles=[2.5,97.5],unit='paired whole image, category and label stratified'),
        exposure='W6 normal probe development-exposed; disclose historical official-test exposure',
        policy='no model/recipe changes or test-based early stopping between 7.1/7.2/7.3')
    suite=lock_json(work/'suite_lock.json',suite_identity)
    atomic_json(calibration_ledger(args.w6_dir),work/'w6_compute_ledger.json')
    job=job_by_id(args.job)
    if args.phase in ('7.1','7.2') and job['id']!='M_seed0':
        raise ValueError('7.1/7.2 are reserved for M seed 0')
    preflight=work/'preflight.json'
    if args.phase!='preflight':
        if not preflight.exists():
            raise ValueError('run/import the W7 preflight before official-test evaluation')
        report=json.loads(preflight.read_text())
        if not report.get('passed') or report.get('suite_lock_digest')!=suite['digest']:
            raise ValueError('W7 preflight does not match this source/runtime/suite')
    if args.phase=='preflight':
        from scripts.hope_cad_v2.preflight_w7 import run_preflight
        return run_preflight(args,context,suite)
    protocol,manifest=job_protocol(context,job)
    folder=work/'runs'/job['id']
    method=(baseline_config(context,job) if job['method']=='CADIC' else deepcopy(context['method']))
    seed_all(job['seed'])
    # A partial refit must be locked to source/runtime even before an adapter/run exists.
    lock_json(folder/'job_lock.json',dict(suite_digest=suite['digest'],job=job,
                                         manifest_digest=manifest['digest'],runtime=runtime))
    if job['method']=='CADIC':
        adapter=CompactCADIC(build_adapter(method,args.device),method['extractor']['checkpoint_sha256'])
        initial_identity=dict(baseline_config=method['id'],budget=job['budget'])
    else:
        if job['seed']==0:
            path=Path(args.w6_dir)/'selection'/job['method']/'deployment_initialization.pt'
            initial=torch.load(path,map_location='cpu',weights_only=False)
            receipt=validate_initialization(initial,context,selection_path=path.parent/'selection.pt')
            initial_identity=dict(reused_W6_seed0=True,receipt=receipt)
        else:
            from scripts.hope_cad_v2.calibrate_features_v4 import bundle_provider
            bundle_path=Path(args.w6_dir)/'prefix_features.pt'
            if not bundle_path.exists():
                raise ValueError('seed refit needs verified W6 prefix_features.pt; import with --features')
            data=bundle_provider(torch.load(bundle_path,map_location='cpu',weights_only=False),args.device,context)
            c1=None
            if job['method']=='A1':
                if args.c1_evidence is None:
                    raise ValueError('supply the C1 W7 evidence for this seed; A1 never fits again')
                import io
                from training.hope_w7_statistics import EvidenceView
                parent=EvidenceView(args.c1_evidence)
                try:
                    if parent.suite!=suite:
                        raise ValueError('A1 C1 parent belongs to a different frozen suite')
                    c1=torch.load(io.BytesIO(parent.read(f'runs/C1_seed{job["seed"]}/refit/deployment_initialization.pt')),
                                  map_location='cpu',weights_only=False)
                finally:
                    parent.close()
            initial=fixed_refit(data,context,recipes[job['method']],folder/'refit',seed=job['seed'],
                device=args.device,source=source,deadline=deadline,c1_initial=c1)
            initial_identity=dict(reused_W6_seed0=False,receipt=validate_refit(initial,context))
            del data
        identity=dict(protocol_id=protocol['id'],manifest_digest=manifest['digest'],
                      parent_baseline_lock_digest=context['lock']['digest'],suite_digest=suite['digest'],job=job)
        adapter=adapter_from_initial(initial,context,identity,args.device)
        restore_rng(initial['rng'])
    try:
        dataset=PrefixContinualProtocol(protocol,manifest,method,job['seed'])
        run=lock_json(folder/'run_lock.json',dict(suite_digest=suite['digest'],job=job,
            manifest_digest=manifest['digest'],initialization=initial_identity,
            effective_training_transport_batch_size=getattr(adapter,'train_transport_batch_size',1),
            evaluation_loader_batch_size=method['runtime']['batch_size'],
            effective_method=adapter.method_metadata(),runtime=runtime))
        atomic_json(manifest,folder/'manifest_train.json')
        atomic_json(protocol,folder/'protocol_resolved.json')
        atomic_json(method,folder/'baseline_method_resolved.json')
        engine=W7Engine(adapter,dataset,folder,run['digest'],device=args.device,deadline=deadline)
        if args.phase=='7.1':
            engine.train()
            print('[W7 FINAL]',json.dumps(engine.final()),flush=True)
            status='7.1_complete_pending_review'
        elif args.phase=='7.2':
            if not (folder/'metrics/final_macro.json').exists():
                raise ValueError('7.2 requires completed 7.1; it never trains/reconstructs missing final metrics')
            engine.matrix()
            from scripts.hope_cad_v2.diagnose_w7 import run_interventions
            run_interventions(adapter,context,folder/'mechanism_interventions.json',
                              past_checkpoint=folder/'states/task_00.pt')
            status='7.2_complete_pending_review'
        else:
            if job['id']=='M_seed0':
                if not (folder/'metrics/continual_summary.json').exists():
                    raise ValueError('M seed0 must be reused from completed 7.2, not rerun in 7.3')
            else:
                engine.train()
                print('[W7 FINAL]',json.dumps(engine.final()),flush=True)
                engine.matrix()
            status='7.3_job_complete_pending_suite'
        atomic_json(dict(status='complete',job=job,suite_digest=suite['digest'],run_digest=run['digest'],
                         final_complete=True,matrix_complete=(folder/'metrics/continual_summary.json').exists(),
                         resources=resource_usage(args.device),profile=adapter.profile_metadata(),
                         storage=adapter.memory_stats(),reportable=False),folder/'completion.json')
        return dict(status=status,job=job['id'],suite_digest=suite['digest'],
                    scientific_status='benchmark output requires review; no automatic superiority claim',reportable=False)
    finally:
        adapter.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('preflight','7.1','7.2','7.3'),required=True)
    p.add_argument('--job',choices=tuple(job['id'] for job in jobs()),default='M_seed0')
    p.add_argument('--w6-dir',type=Path,required=True)
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--session-hours',type=float,default=10.5)
    p.add_argument('--session-start-monotonic',type=float)
    p.add_argument('--c1-evidence',type=Path)
    p.add_argument('--skip-export',action='store_true',help='notebook finally cell still exports evidence')
    args=p.parse_args()
    if not 0.05<=args.session_hours<=11:
        p.error('session hours must be within 0.05..11, with export reserve')
    configure_runtime()
    start=args.session_start_monotonic if args.session_start_monotonic is not None else time.monotonic()
    if start>time.monotonic():
        p.error('session start cannot be in the future')
    deadline=start+args.session_hours*3600-180
    args.work_dir.mkdir(parents=True,exist_ok=True)
    code=0
    try:
        if time.monotonic()>=deadline:
            raise SessionLimit('bootstrap consumed the session budget')
        handoff=execute(args,deadline)
    except SessionLimit as error:
        handoff=dict(status='paused_at_atomic_boundary',phase=args.phase,job=args.job,
                     reason=str(error),reportable=False)
    except Exception as error:
        traceback.print_exc()
        code=1
        handoff=dict(status='failed_preserve_evidence',phase=args.phase,job=args.job,
                     error=repr(error),reportable=False)
    handoff.update(requested_phase=args.phase,session_hours=args.session_hours,
                   elapsed_since_notebook_start_seconds=time.monotonic()-start,
                   budget_includes_bootstrap=args.session_start_monotonic is not None)
    atomic_json(handoff,args.work_dir/'handoff.json')
    import uuid
    atomic_json(handoff,args.work_dir/'runs'/args.job/'sessions'/f'{uuid.uuid4().hex}.json')
    print(json.dumps(handoff,indent=2),flush=True)
    if not args.skip_export:
        print(export_archive(args.work_dir,args.work_dir.parent/'hope_cad_w7_evidence.zip'),flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
