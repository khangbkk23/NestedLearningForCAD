"""Single registered job per invocation; explicit phase handoffs, no metric feedback."""
import argparse,json,time,gc,unittest
from pathlib import Path
import torch
from models.feature_extractors.cadic_vit_v1 import CADICViTConfig,CADICViTFeatureExtractor
from models.hope_cad_v5_1.contract import ROOT,CONTRACT,digest,design_digest
from training.benchmark_config_v1 import load_configs
from dataset.benchmark_protocol_v2 import build_prefix_manifest
from training.runtime import atomic_json,resource_usage
from training.support_benchmark_v5_1.contract import study,study_digest,source_digest,session_for,job,lock
from training.support_benchmark_v5_1.data import NormalFeatures,EvaluationFeatures,test_manifest
from training.support_benchmark_v5_1.engine import train,evaluate,Pause,boundary,read,save_or_verify
from notebook_support.support_v5_1.common import configure_runtime
from notebook_support.support_v5_1.runner import preflight as core_preflight

def run(c):
    configure_runtime();work=Path(c['work']);cfg=study();mode,seed=job(c['job']);device='cuda';deadline=c['started']+c['session_hours']*3600-180
    p,m=load_configs(ROOT/'conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml',ROOT/'conf/benchmarks/methods/cadic_matched_prefix_v2.yaml')
    # Partition order is identical across seeds. Common manifest uses seed0; session seed is separate.
    manifest=build_prefix_manifest(p,seed=0,source_commit='source-file-map:'+source_digest())
    if len(manifest['online'])!=cfg['online_images'] or manifest['task_order']!=cfg['task_order']:raise ValueError('Canonical online protocol mismatch')
    rows=manifest['online'];tasks=manifest['task_order'];runtime=c['prepared']['runtime']
    identity=dict(schema='support_v5_1_benchmark_suite_v1',study_digest=study_digest(),source_digest=source_digest(),design_digest=design_digest(),normal_rows_digest=digest(rows),runtime=runtime,backbone_sha256=CONTRACT['backbone_sha256'])
    suite=lock(work/'suite_lock.json',identity);save_or_verify(work/'manifest_train.json',manifest)
    save_or_verify(work/'data_rights.json',dict(offline_calibration=False,search=False,task_id_to_learner=False,official_test_for_selection=False,one_image_causal_delivery=True,observed_context_replay=True,test_and_GT_to_evaluator_only=True,pilot_learned_state_reused=False))
    extractor=CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=c['prepared']['checkpoint'],checkpoint_identity='timm/vit_base_patch8_224.augreg_in21k'),device)
    try:
        provider=NormalFeatures(c['prepared']['dataset'],extractor,work/'runs'/c['job']/'feature_receipts',device)
        pf=work/'benchmark_preflight.json'
        if not pf.exists():
            if c['phase']!='preflight':raise ValueError('Run new benchmark preflight first, then attach its PASS output')
            boundary(deadline)
            tests=unittest.defaultTestLoader.discover(str(ROOT/'tests/support_benchmark_v5_1'))
            result=unittest.TextTestRunner(verbosity=1).run(tests)
            if not result.wasSuccessful() or result.testsRun<12 or result.skipped:raise RuntimeError('Benchmark preflight tests failed/missing/skipped')
            # Existing accepted core +24 discarded normals, including full-budget pending/resume.
            core_preflight(work,rows,provider,deadline)
            atomic_json(dict(status='passed',suite_digest=suite['digest'],tests=result.testsRun,discarded_normal_clone=True,official_test_accessed=False,core_receipt_sha256=__import__('hashlib').sha256((work/'preflight.json').read_bytes()).hexdigest()),pf)
        elif read(pf)['status']!='passed' or read(pf)['suite_digest']!=suite['digest']:raise ValueError('Preflight identity changed')
        if c['phase']=='preflight':return 'preflight_passed'
        if c['phase'] in ('7.1','7.2') and seed!=0:raise ValueError('7.1/7.2 are seed0 only; use7.3 for others')
        folder=work/'runs'/c['job'];ji=dict(job=c['job'],mode=mode,seed=seed,suite_digest=suite['digest'],rows_digest=digest(rows),fresh_initialization=True)
        factory=lambda:session_for(c['job'],device)
        if mode in ('NESTED','FIRST_ORDER'):
            first=session_for('NESTED_seed'+str(seed),device);second=session_for('FIRST_ORDER_seed'+str(seed),device)
            a,b=first.initialization_hashes(),second.initialization_hashes()
            if a!=b:raise ValueError('Exact/FO seed initialization mismatch')
            lock(folder/'matched_initialization.json',dict(seed=seed,NESTED=a,FIRST_ORDER=b))
            del first,second
        if c['phase']=='7.2':
            if not (folder/'final_complete.json').exists():raise ValueError('7.2 requires completed7.1 of the same job; cannot train or reconstruct missing states')
            lock(folder/'job_lock.json',dict(**ji,rows=rows,task_order=tasks))
        else:
            actor=train(folder,rows,tasks,ji,provider,factory,deadline)
            del actor;gc.collect();torch.cuda.empty_cache()
        # Test manifest is created only after complete training of selected job.
        tm=test_manifest(c['prepared']['dataset'],tasks)
        if len(tm['rows'])!=cfg['test_images']:raise ValueError('Expected1725official test images')
        save_or_verify(work/'test_manifest.json',tm)
        ep=EvaluationFeatures(c['prepared']['dataset'],extractor,device)
        evaluate(folder,tasks,tm['rows'],factory,dict(provider=ep,test_digest=tm['digest']),matrix=c['phase'] in ('7.2','7.3'),deadline=deadline)
        atomic_json(dict(gpu=resource_usage(device),timing_scope='end-to-end with IO, no fair standalone compute claim',official_test_for_selection=False),folder/'resources.json')
        return c['phase']+'_complete_pending_review'
    finally:extractor.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);c=json.loads(Path(ap.parse_args().config).read_text());work=Path(c['work'])
    try:status=run(c);atomic_json(dict(status=status,job=c['job'],reportable=False,VisA_authorized=False),work/'handoff.json')
    except (Pause,__import__('training.hope_support_v5_1',fromlist=['PilotPause']).PilotPause) as e:atomic_json(dict(status='paused_at_atomic_boundary',job=c['job'],reason=str(e),reportable=False),work/'handoff.json')
    except BaseException as e:atomic_json(dict(status='failed_preserve_evidence',job=c['job'],error=repr(e),reportable=False),work/'handoff.json');raise
if __name__=='__main__':main()
