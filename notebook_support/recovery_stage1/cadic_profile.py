"""Bounded resource probe of the UNCHANGED full K=10000 CADIC image-update route."""
from pathlib import Path
import argparse
import json
import time
import traceback
import torch
from torch.utils.data import DataLoader
from dataset.benchmark_protocol_v2 import HashedNormalDataset
from training.hope_runtime_v4 import configure_runtime
from training.hope_outer_learning_v4 import capture_rng
from training.benchmark_engine_v1 import _clone_state,_state_equal
from training.hope_w7_contract import job_by_id
from training.hope_w7_adapter import CompactCADIC,baseline_config
from training.hope_w7_engine import synchronize
from models.benchmark_registry_v2 import build_adapter
from scripts.hope_cad_v2.run_w7 import reviewed_context
from .common import atomic_json
from .cadic_accel import accelerate,reference_parity,BACKEND


def run(w6,output,device):
    context=reviewed_context(w6);config=baseline_config(context,job_by_id('CADIC10000_seed0'))
    cadic=CompactCADIC(accelerate(build_adapter(config,device)),config['extractor']['checkpoint_sha256'])
    report=dict(schema='recovery_cadic_resource_probe_v1',status='running',official_test_accessed=False,
        normal_train_images_completed=0,budget=10000,algorithm_changed=False,execution_optimization=BACKEND,binding=cadic.binding_profile())
    atomic_json(report,output)
    try:
        pre=context['method']['preprocessing'];rows=context['roles']['roles']['meta_support'][:16]
        samples=HashedNormalDataset(context['protocol']['dataset']['root'],rows,pre['image_size'],pre['mean'],pre['std'],True)
        loader=DataLoader(samples,batch_size=config['runtime']['batch_size'],shuffle=False,generator=torch.Generator().manual_seed(0))
        started=time.perf_counter();batches=[]
        for batch in loader:
            print('[CADIC RESOURCE] starting exact batch of',len(batch['images']),'normal images; watchdog is active',flush=True)
            t=time.perf_counter();result=cadic.fit_task(0,'normal_preflight',[batch]);synchronize(device)
            if result['coreset']!=cadic.coreset.stats():raise RuntimeError('fit return and snapshot coreset disagree')
            batches.append(time.perf_counter()-t)
            report.update(normal_train_images_completed=report['normal_train_images_completed']+len(batch['images']),
                batch_seconds=batches,train_seconds=time.perf_counter()-started,coreset=cadic.coreset.stats(),cache_progress=cadic.coreset.cache_profile(),binding=cadic.binding_profile())
            atomic_json(report,output)
        if cadic.coreset.count!=10000:raise ValueError('full-budget route was not exercised')
        if cadic.coreset.cache_profile()['closest_pair_calls']<=0:raise RuntimeError('optimized cache was not used by real training')
        before=_clone_state(cadic.state_dict());rng=capture_rng()
        batch=next(iter(loader));batch={k:v[:1] for k,v in batch.items()}
        t=time.perf_counter();score=cadic.score_batch(batch);synchronize(device);score_seconds=time.perf_counter()-t
        if not _state_equal(before,_clone_state(cadic.state_dict())) or not _state_equal(rng,capture_rng()):
            raise RuntimeError('CADIC scoring changed state/RNG')
        cadic.load_state_dict(before)
        if not _state_equal(score,cadic.score_batch(batch)):raise RuntimeError('CADIC state roundtrip changed scores')
        normal_features=cadic.extractor.extract_patch_features(batch['images'].to(cadic.device))
        parity=reference_parity(cadic.coreset,normal_features[0,::97][:4],normal_features)
        report.update(status='passed',reference_parity=parity,full_budget_exercised=True,checkpoint_roundtrip_exact=True,state_rng_readonly=True,
            normal_score_seconds=score_seconds,profile=cadic.profile_metadata(),storage=cadic.memory_stats(),
            caveat='16 first-prefix normal images only; not full-stream runtime or AD performance')
        atomic_json(report,output)
    finally:cadic.close()
    report['cleanup_complete']=True
    atomic_json(report,output)
    return report


def bounded_probe(command,output,timeout):
    import subprocess
    output=Path(output)
    print('[CADIC RESOURCE] Hard resource-probe limit:',timeout,'seconds; timeout blocks long controls',flush=True)
    try:
        completed=subprocess.run(command,timeout=timeout)
    except subprocess.TimeoutExpired as error:
        prior=json.loads(output.read_text()) if output.exists() else {}
        prior.update(schema='recovery_cadic_resource_probe_v1',status='resource_timeout',timeout_seconds=timeout,
            official_test_accessed=False,algorithm_changed=False,needs_engineering_review=True,
            instruction='Preserve ZIP. Do not run long CADIC by disabling this gate; review exact implementation cost.')
        atomic_json(prior,output)
        raise ValueError('CADIC full-budget resource probe exceeded limit; evidence saved for engineering review') from error
    if completed.returncode:raise ValueError('CADIC resource probe process failed')
    report=json.loads(output.read_text())
    if report.get('status')!='passed' or not report.get('full_budget_exercised') or not report.get('cleanup_complete'):raise ValueError('CADIC resource probe incomplete')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--w6-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');args=p.parse_args()
    configure_runtime()
    try:run(args.w6_dir,args.output,args.device)
    except Exception as error:
        traceback.print_exc()
        prior=json.loads(args.output.read_text()) if args.output.exists() else {}
        prior.update(schema='recovery_cadic_resource_probe_v1',status='failed',error=repr(error),official_test_accessed=False)
        atomic_json(prior,args.output);return 1
    return 0

if __name__=='__main__':raise SystemExit(main())