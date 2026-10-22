"""Only score saved task states; no training entry point or normal provider."""
import argparse,gc,time
from pathlib import Path
import torch
from training.runtime import atomic_json,resource_usage
from training.support_benchmark_v5_1 import engine
from training.support_benchmark_v5_1.contract import session_for
from training.support_benchmark_v5_1.data import EvaluationFeatures,test_manifest
from training.support_throughput_v5_1.replay import assert_equal
from training.support_nn_write_v5_1.candidate import activation
from notebook_support.support_v5_1.common import configure_runtime,runtime_identity
from models.feature_extractors.cadic_vit_v1 import CADICViTConfig,CADICViTFeatureExtractor
from .common import read,verify_source,operation,verify_frozen,open_owner,load_task,complete,sha

def readonly_preflight(owner,rows,factory,provider):
    """Two discarded loaded actors, one legal test row per first/last checkpoint."""
    reports=[]
    for i in (0,len(owner.tasks)-1):
        actor=factory();checkpoint=load_task(owner,i,actor)
        row=next(r for r in rows if r['task_id']==i)
        before=actor.state_dict();raw,_=provider.get(row)
        with activation(True),torch.no_grad():scores=actor.score(raw)
        if any(not bool(torch.isfinite(v).all()) for v in scores) or scores[0].shape!=(1,actor.image_tokens):raise ValueError('Readonly score preflight failed')
        assert_equal(before,actor.state_dict(),'readonly matrix preflight')
        clone=factory();clone.load_state_dict(before);assert_equal(before,clone.state_dict(),'task-state roundtrip')
        reports.append(dict(checkpoint=i,task=i,identity=row['relative_path'],checkpoint_sha256=checkpoint,scientific_state_unchanged=True))
        del actor,clone,raw,before;gc.collect()
    return dict(status='passed',discarded_owners=4,scored_images=2,official_test_used_for_learning=False,probes=reports)

def evaluate_matrix(owner,rows,factory,provider,test_digest,deadline):
    """NESTED Execution includes its legacy resolver; CADICOwner uses the original one."""
    from training.support_migration_v5_1.lineage import Execution
    if isinstance(owner,Execution):
        owner.evaluate(rows,factory,dict(provider=provider,test_digest=test_digest),matrix=True,deadline=deadline)
    else:owner.evaluate(rows,factory,dict(provider=provider,test_digest=test_digest),deadline=deadline)
    return complete(owner.folder,owner.tasks)

def run(c):
    configure_runtime();op,refs,_=verify_source();runtime=runtime_identity();job=c['job']
    if c['phase']!='7.2' or c['operation']!=operation(op,job) or runtime!=c['runtime'] or runtime!=refs['runtime'] or not torch.cuda.is_available():raise ValueError('Worker matrix source/runtime/job changed')
    work,session=Path(c['work']),Path(c['session']);preserved=verify_frozen(work,job,refs)
    owner=open_owner(work,job,runtime,refs);engine.boundary(c['deadline'])
    extractor=CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=c['checkpoint'],checkpoint_identity='timm/vit_base_patch8_224.augreg_in21k'),'cuda')
    began=time.monotonic()
    try:
        tm=test_manifest(c['dataset'],owner.tasks)
        if sha(work/'benchmark/test_manifest.json')!=refs['test_manifest_sha256'] or tm!=read(work/'benchmark/test_manifest.json'):raise ValueError('Official image/mask bytes differ from frozen7.1')
        provider=EvaluationFeatures(c['dataset'],extractor,'cuda');factory=lambda:session_for(job,'cuda')
        pf=work/'matrix_preflight.json'
        if pf.exists():
            if read(pf).get('status')!='passed' or read(pf).get('job')!=job or read(pf).get('operation')!=c['operation']:raise ValueError('Matrix preflight identity changed')
        else:
            engine.boundary(c['deadline']);proof=readonly_preflight(owner,tm['rows'],factory,provider)
            atomic_json(dict(**proof,job=job,operation=c['operation']),pf)
        print('[MATRIX] readonly evaluation',job,'120 total cells; reuse existing completed cells; no training',flush=True)
        evaluate_matrix(owner,tm['rows'],factory,provider,tm['digest'],c['deadline'])
        return '7.2_complete_pending_review'
    finally:
        checked=verify_frozen(work,job,refs)
        count=len(list((owner.folder/'cells').glob('*/completion.json')))
        atomic_json(dict(job=job,completed_cells=count,total_cells=120,original_final_cells=15,train_repeated=False,immutable_parent_payloads_checked=checked,initial_preservation_checks=preserved),session/'matrix_progress.json')
        atomic_json(dict(evaluation_session_seconds=time.monotonic()-began,gpu=resource_usage('cuda'),scope='readonly evaluation incl feature extraction/metrics/checkpoints; no train'),session/'resources.json')
        extractor.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);c=read(p.parse_args().config);session=Path(c['session']);began=time.monotonic()
    try:status=run(c)
    except engine.Pause as exc:
        status='paused_at_atomic_boundary';atomic_json(dict(reason=str(exc)),session/'pause.json')
    except BaseException as exc:
        atomic_json(dict(status='failed_preserve_evidence',error=repr(exc),worker_seconds=time.monotonic()-began),session/'worker_status.json');raise
    atomic_json(dict(status=status,worker_seconds=time.monotonic()-began),session/'worker_status.json');print('[MATRIX STATUS]',c['job'],status,flush=True)
if __name__=='__main__':main()
