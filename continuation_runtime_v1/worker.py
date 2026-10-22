"""Continuation7.1 only: strict migrated owner, frozen training and final15 evaluator."""
import argparse,gc,json,time
from pathlib import Path
import torch
from training.runtime import atomic_json,resource_usage
from training.support_migration_v5_1.contract import contract
from training.support_migration_v5_1.lineage import Execution
from training.support_throughput_v5_1.contract import contract as profile_contract
from training.support_benchmark_v5_1.contract import session_for,study
from training.support_benchmark_v5_1.data import NormalFeatures,EvaluationFeatures,test_manifest
from training.support_benchmark_v5_1.engine import boundary,save_or_verify,Pause
from notebook_support.support_v5_1.common import configure_runtime,runtime_identity
from models.feature_extractors.cadic_vit_v1 import CADICViTConfig,CADICViTFeatureExtractor
from .common import read,verify_delivery,verify_publication,stage_tar

def continue_execution(e,c,provider,factory,evaluate_after_train):
    """Dependency-injected seam, also used for meaningful CPU interrupted/resumed tests."""
    actor=e.train(provider,factory,deadline=c['deadline'])
    del actor;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    boundary(c['deadline'])
    evaluate_after_train()
    final=read(e.folder/'final_complete.json')
    if final['cell_count']!=15 or final['job_digest']!=e.identities['job']['digest']:raise ValueError('Incomplete/wrong final15 owner')
    return '7.1_complete_pending_review'

def run(c):
    configure_runtime();source,op=verify_delivery();runtime=runtime_identity()
    if source!=c['source'] or op!=c['orchestration'] or runtime!=c['runtime'] or not torch.cuda.is_available():raise ValueError('Worker source/runtime changed')
    verify_publication(Path(c['work']))
    root=Path(c['work'])/'benchmark';e=Execution(root,source,runtime,contract(),profile_contract()['parent'])
    if len(e.rows)!=3599 or len(e.tasks)!=15:raise ValueError('Wrong registered dataset scope')
    extractor=CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=c['checkpoint'],checkpoint_identity='timm/vit_base_patch8_224.augreg_in21k'),'cuda')
    try:
        folder=e.folder
        normal=NormalFeatures(c['dataset'],extractor,folder/'feature_receipts','cuda')
        factory=lambda:session_for('NESTED_seed0','cuda')
        def evaluation():
            if not (folder/'train_complete.json').exists():raise ValueError('Test staging forbidden before train_complete')
            boundary(c['deadline'])
            if c['dataset_archive']:
                print('[CONTINUATION] staging test/GT bytes after completed training',flush=True)
                stage_tar(c['dataset_archive'],c['dataset'],evaluation=True)
            tm=test_manifest(c['dataset'],e.tasks)
            if len(tm['rows'])!=study()['test_images']:raise ValueError('Wrong test manifest size')
            save_or_verify(root/'test_manifest.json',tm)
            ep=EvaluationFeatures(c['dataset'],extractor,'cuda')
            e.evaluate(tm['rows'],factory,dict(provider=ep,test_digest=tm['digest']),matrix=False,deadline=c['deadline'])
        return continue_execution(e,c,normal,factory,evaluation)
    finally:
        atomic_json(dict(gpu=resource_usage('cuda'),scope='worker including IO; not paired speedup experiment',official_test_for_selection=False),Path(c['session'])/'resources.json')
        extractor.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);c=read(p.parse_args().config);session=Path(c['session']);started=time.monotonic()
    try:status=run(c)
    except (Pause,__import__('training.hope_support_v5_1',fromlist=['PilotPause']).PilotPause) as exc:
        status='paused_at_atomic_boundary';atomic_json(dict(reason=str(exc)),session/'pause.json')
    except BaseException as exc:
        atomic_json(dict(status='failed_preserve_evidence',error=repr(exc),worker_seconds=time.monotonic()-started),session/'worker_status.json');raise
    atomic_json(dict(status=status,worker_seconds=time.monotonic()-started),session/'worker_status.json')
    print('[CONTINUATION STATUS]',status,flush=True)
if __name__=='__main__':main()
