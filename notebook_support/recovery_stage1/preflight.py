"""Normal-only real-RGB transport/observer/diagnostic check plus full-budget CADIC profiling."""
from pathlib import Path
import gc
import json
import os
import subprocess
import sys
import time
import torch
from torch.utils.data import DataLoader,Subset
from dataset.benchmark_protocol_v2 import HashedNormalDataset
from training.benchmark_engine_v1 import _clone_state,_state_equal
from training.hope_outer_learning_v4 import capture_rng,restore_rng
from training.hope_w7_engine import W7Engine,SessionLimit,synchronize
from training.hope_runtime_v4 import resource_usage
from scripts.hope_cad_v2.run_w7 import adapter_from_initial
from models.benchmark_registry_v2 import build_adapter
from training.hope_w7_adapter import CompactCADIC,baseline_config
from training.hope_w7_contract import job_by_id
from .common import ObservedAdapter,atomic_json,check_deadline
from .diagnostics import measure_normal
from .cadic_accel import BACKEND


class NormalData:
    def __init__(self,context,count,device):
        self.rows=context['roles']['roles']['meta_support'][:count]
        self.task_names=['normal_prefix_0','normal_prefix_1']
        pre=context['method']['preprocessing']
        self.data=HashedNormalDataset(context['protocol']['dataset']['root'],self.rows,
            pre['image_size'],pre['mean'],pre['std'],True)
    def indices(self,task):return range(min(6,len(self.rows))) if task==0 else range(min(6,len(self.rows)),len(self.rows))
    def training_paths(self,task):return [self.rows[i] for i in self.indices(task)]
    def build_train_loader(self,task):return DataLoader(Subset(self.data,list(self.indices(task))),batch_size=1,
        shuffle=False,generator=torch.Generator().manual_seed(0))


def validate_preflight(work,study):
    from .common import digest
    work=Path(work)
    def read(name):
        p=work/name
        if not p.is_file():raise ValueError('preflight missing supporting receipt: '+name)
        return json.loads(p.read_text())
    gate=read('recovery_preflight.json');frozen=read('verification_frozen.json')
    recovery=read('verification_recovery.json');resource=read('cadic_resource_probe.json')
    source=study['identity']['source']
    if (gate.get('schema')!='recovery_stage1_preflight_v1' or gate.get('passed') is not True
        or gate.get('study_digest')!=study['digest'] or gate.get('official_test_accessed') is not False
        or gate.get('calibration_refitted') is not False):raise ValueError('invalid recovery preflight identity/scope')
    if (not frozen.get('success') or frozen.get('tests_run')!=130 or frozen.get('failures') or frozen.get('errors')
        or frozen.get('source_sha256')!=source['frozen']):raise ValueError('frozen regression receipts do not match source')
    if (not recovery.get('success') or recovery.get('tests_run',0)<45 or recovery.get('failures') or recovery.get('errors')
        or recovery.get('operations_source_digest')!=digest(source)):raise ValueError('recovery regression receipt changed')
    if (resource!=gate.get('CADIC') or resource.get('status')!='passed' or resource.get('full_budget_exercised') is not True
        or resource.get('normal_train_images_completed')!=16 or resource.get('official_test_accessed') is not False
        or resource.get('algorithm_changed') is not False or not resource.get('checkpoint_roundtrip_exact')
        or not resource.get('state_rng_readonly') or resource.get('cleanup_complete') is not True):raise ValueError('full-budget CADIC resource gate not passed')
    binding=resource.get('binding',{})
    if (binding.get('fit_task_and_snapshot_shared') is not True or binding.get('registry_wrapper') is not True
        or binding.get('cache_backend')!=BACKEND or resource.get('cache_progress',{}).get('closest_pair_calls',0)<=0):
        raise ValueError('CADIC resource did not exercise the shared registered cache')
    parity=resource.get('reference_parity',{})
    fields=('passed','pair_exact','decisions_and_counters_exact','features_exact','pixel_and_image_scores_exact','main_state_rng_unchanged')
    if (resource.get('execution_optimization')!=BACKEND or parity.get('backend')!=BACKEND
        or parity.get('bank_count')!=10000 or parity.get('candidate_vectors')!=4
        or any(parity.get(k) is not True for k in fields)):
        raise ValueError('CADIC optimized execution lacks full-budget reference parity')
    stats=resource.get('coreset',{})
    if any(stats.get(k)!=v for k,v in {'count':10000,'budget':10000,'dim':768,'update_batches':16}.items()):
        raise ValueError('CADIC preflight changed bank geometry or update unit')
    methods=gate.get('methods',{})
    if set(methods)!={'M','C1','A1'} or any(not v.get('paired_state_rng_score_resume') or not v.get('diagnostic_state_rng_unchanged')
        or v.get('normal_events_per_history')!=9 for v in methods.values()):raise ValueError('normal RGB parity gate incomplete')
    return gate


def run_preflight(args,context,suite,study,deadline):
    work=args.work_dir
    existing=work/'recovery_preflight.json'
    if existing.exists():
        gate=json.loads(existing.read_text())
        if gate.get('passed') and gate.get('study_digest')==study['digest']:
            validate_preflight(work,study)
            return dict(status='recovery_preflight_passed',reused_verified_gate=True,reportable=False)
        raise ValueError('preflight receipt mismatch')
    check_deadline(deadline)
    temporary=work/'test_tmp';temporary.mkdir(parents=True,exist_ok=True)
    test_env=dict(os.environ,TEMP=str(temporary),TMP=str(temporary),PYTHONDONTWRITEBYTECODE='1')
    for command in ([sys.executable,'-B','scripts/hope_cad_v2/verify_w7.py','--output',str(work/'verification_frozen.json')],
                    [sys.executable,'-B','-m','notebook_support.recovery_stage1.verify','--output',str(work/'verification_recovery.json')]):
        print('[RECOVERY PREFLIGHT]', ' '.join(command),flush=True)
        if subprocess.call(command,cwd=Path(__file__).resolve().parents[2],env=test_env):raise ValueError('preflight regression suite failed')
    data=NormalData(context,9,args.device);methods={}
    for method in ('M','C1','A1'):
        check_deadline(deadline)
        initial=torch.load(args.w6_dir/'selection'/method/'deployment_initialization.pt',map_location='cpu',weights_only=False)
        identity=dict(recovery_normal_preflight=True,method=method)
        base=work/'normal_preflight'/method
        # Preflight attempts have independent state; a paused normal fixture is rerun, not relabeled.
        attempt=0
        while (base/str(attempt)).exists():attempt+=1
        base=base/str(attempt)
        plain=adapter_from_initial(initial,context,identity,args.device)
        try:
            restore_rng(initial['rng']);t=time.perf_counter()
            W7Engine(plain,data,base/'plain',study['digest'],device=args.device,deadline=deadline).train()
            synchronize(args.device)
            expected=_clone_state(plain.state_dict());rng=capture_rng()
            batch=next(iter(data.build_train_loader(0)))
            score=plain.score_batch(batch)
            if not _state_equal(rng,capture_rng()):raise RuntimeError('normal score consumed RNG')
            seconds=time.perf_counter()-t
        finally:plain.close()
        del plain;gc.collect()
        partial=adapter_from_initial(initial,context,identity,args.device)
        try:
            restore_rng(initial['rng'])
            observer=ObservedAdapter(partial,base/'observations')
            try:W7Engine(observer,data,base/'observed',study['digest'],device=args.device,deadline=deadline).train(stop_after_events=5)
            except SessionLimit:
                if deadline is not None and time.monotonic()>=deadline:raise
        finally:partial.close()
        del observer,partial;gc.collect()
        resumed=adapter_from_initial(initial,context,identity,args.device)
        try:
            observer=ObservedAdapter(resumed,base/'observations')
            W7Engine(observer,data,base/'observed',study['digest'],device=args.device,deadline=deadline).train()
            if not _state_equal(expected,_clone_state(resumed.state_dict())) or not _state_equal(rng,capture_rng()):
                raise RuntimeError('observer/atomic resume changed real RGB state or RNG')
            if not _state_equal(score,resumed.score_batch(batch)):raise RuntimeError('observer/resume changed scores')
            diagnostic=measure_normal(resumed,batch['images'][:1])
            if abs(diagnostic['image_score']-float(score['image_scores'][0]))>1e-5:
                raise RuntimeError('diagnostic score differs from actual adapter score')
            methods[method]=dict(normal_events_per_history=9,paired_state_rng_score_resume=True,
                diagnostic_state_rng_unchanged=True,mean_seconds_per_image=seconds/9,
                normal_example=diagnostic)
        finally:resumed.close()
        del observer,resumed;gc.collect()
    check_deadline(deadline)
    from .cadic_profile import bounded_probe
    budget=min(600.,max(0.,deadline-time.monotonic())) if deadline is not None else 600.
    check_deadline(deadline)
    baseline=bounded_probe([sys.executable,'-B','-m','notebook_support.recovery_stage1.cadic_profile',
        '--w6-dir',str(args.w6_dir),'--output',str(work/'cadic_resource_probe.json'),'--device',args.device],
        work/'cadic_resource_probe.json',budget)
    report=dict(schema='recovery_stage1_preflight_v1',passed=True,study_digest=study['digest'],
        methods=methods,CADIC=baseline,official_test_accessed=False,calibration_refitted=False,
        resources=resource_usage(args.device),reportable=False)
    atomic_json(report,existing)
    return dict(status='recovery_preflight_passed',study_digest=study['digest'],official_test_accessed=False,
        next='controls notebook; import this exact preflight output',reportable=False)