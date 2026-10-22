"""Fresh CADIC, exact shared core and optimizedNN; no NESTED state/metrics imports."""
import argparse,gc,json,time,os
from pathlib import Path
import torch
from training.runtime import atomic_json,resource_usage
from training.support_benchmark_v5_1 import engine
from training.support_benchmark_v5_1.contract import session_for,lock
from training.support_benchmark_v5_1.data import NormalFeatures,EvaluationFeatures,test_manifest
from training.support_nn_write_v5_1.candidate import activation
from training.support_throughput_v5_1.replay import assert_equal,state_digest
from notebook_support.support_v5_1.common import configure_runtime,runtime_identity
from models.feature_extractors.cadic_vit_v1 import CADICViTConfig,CADICViTFeatureExtractor
from models.hope_cad_v5_1.contract import digest
from .common import verify_source,read,identity,stage_tar,RECEIPTS,ROOT,sha

class MatchedFeatures:
    """Return features only if their original frozen receipt equals the locked NESTED hash."""
    def __init__(self,provider,receipts,expected):self.provider,self.receipts,self.expected=provider,Path(receipts),expected
    def get(self,row):
        key=digest(row)
        if row.get('split')!='train' or '/train/good/' not in row['relative_path'] or key not in self.expected:raise ValueError('Unregistered normal feature row')
        value=self.provider.get(row);receipt=read(self.receipts/(key+'.json'))
        if receipt['row']!=row or receipt['tensor_sha256']!=self.expected[key]:raise ValueError('Raw ViT features differ from NESTED reference')
        return value

def preflight(folder,rows,provider,factory,owner_digest,deadline):
    """Two discarded owners; exact reference/candidate state and readonly score, no official test."""
    folder=Path(folder);receipt=folder/'preflight.json';row_digest=digest(rows[:2])
    if receipt.exists():
        value=read(receipt)
        if value.get('status')!='passed' or value.get('operation_digest')!=owner_digest or value.get('row_digest')!=row_digest or value.get('images')!=2 or value.get('discarded_owners')!=2 or value.get('official_test_accessed') is not False:raise ValueError('Preflight receipt identity changed')
        return value
    a,b=factory(),factory()
    if a.cursor or b.cursor or a.mode!='CADIC' or b.mode!='CADIC' or a.learner is not None or b.learner is not None:raise ValueError('Fresh discarded CADIC owners required')
    events=[]
    for row in rows[:2]:
        engine.boundary(deadline);raw=provider.get(row)
        with activation(False):left=a.observe(raw,row['relative_path'])
        with activation(True):right=b.observe(raw,row['relative_path'])
        assert_equal(left,right,'CADIC preflight event');assert_equal(a.state_dict(),b.state_dict(),'CADIC preflight state')
        before=b.state_dict()
        with activation(True):
            score=b.score(raw);loaded=factory();loaded.load_state_dict(before)
        assert_equal(before,b.state_dict(),'readonlyscore');assert_equal(loaded.state_dict(),before,'strictCADICroundtrip')
        if not all(bool(torch.isfinite(v).all()) for v in score) or score[0].shape!=(1,b.image_tokens):raise ValueError('Invalid readonly CADIC score/geometry')
        events.append(left)
    out=dict(status='passed',operation_digest=owner_digest,row_digest=row_digest,images=2,discarded_owners=2,probe_bank_count=int(b.bank.features.shape[0]),full_budget_test=False,official_test_accessed=False,exact_reference_candidate_events=True,exact_state_roundtrip=True,score_readonly=True,event_digest=digest(events),state_digest=state_digest(b.state_dict()))
    atomic_json(out,receipt);return out

def run_job(folder,rows,tasks,job_identity,provider,factory,evaluation,deadline,timing=None):
    """The frozen engine owns checkpoints; evaluation callback cannot run before true train completion."""
    folder=Path(folder)
    began=time.monotonic()
    try:
        with activation(True):actor=engine.train(folder,rows,tasks,job_identity,provider,factory,deadline=deadline)
    finally:
        if timing is not None:timing['train_session_seconds']=time.monotonic()-began
    del actor;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    engine.boundary(deadline)
    if not (folder/'train_complete.json').exists():raise ValueError('Evaluation before completed training forbidden')
    evaluation()
    final=read(folder/'final_complete.json');job=read(folder/'job_lock.json')
    if final['cell_count']!=len(tasks) or final['job_digest']!=job['digest']:raise ValueError('Final metrics incomplete/wrong owner')
    return '7.1_complete_pending_review'

def run(c):
    configure_runtime();op,ref,manifest=verify_source();runtime=runtime_identity()
    if op!=c['operation'] or runtime!=c['runtime'] or not torch.cuda.is_available():raise ValueError('Worker source/runtime differs')
    if c['phase']!='7.1' or c['job']!='CADIC_seed0':raise ValueError('Only matchedCADIC_seed0/7.1')
    work,session=Path(c['work']),Path(c['session']);root=work/'benchmark';root.mkdir(exist_ok=True)
    suite,ji=identity(op,ref,runtime);lock(root/'suite_lock.json',suite['identity'])
    engine.save_or_verify(root/'manifest_train.json',manifest)
    engine.save_or_verify(root/'data_rights.json',dict(offline_calibration=False,search=False,official_test_for_selection=False,task_id_to_learner=False,pilot_learned_state_reused=False,NESTED_learned_state_reused=False,CADIC_cost_clone_reused=False,matched_normal_rows_digest=ref['normal_rows_digest'],feature_hashes_bound=True))
    extractor=CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=c['checkpoint'],checkpoint_identity='timm/vit_base_patch8_224.augreg_in21k'),'cuda')
    factory=lambda:session_for('CADIC_seed0','cuda');folder=root/'runs/CADIC_seed0';expected=read(RECEIPTS/'feature_hashes.json')
    timing=dict(initial_cursor=0,train_session_seconds=0.,evaluation_seconds=0.,preflight_seconds=0.,scope='measured orchestrator wall time, with init/hash/IO; fresh vs resumed disclosed, not paper purekernel')
    if (folder/'active.json').exists():timing['initial_cursor']=read(folder/'active.json')['cursor']
    try:
        began=time.monotonic()
        pf_provider=MatchedFeatures(NormalFeatures(c['dataset'],extractor,root/'preflight_features','cuda'),root/'preflight_features',expected)
        preflight(root,manifest['online'],pf_provider,factory,digest(op),c['deadline'])
        timing['preflight_seconds']=time.monotonic()-began
        provider=MatchedFeatures(NormalFeatures(c['dataset'],extractor,folder/'feature_receipts','cuda'),folder/'feature_receipts',expected)
        def evaluation():
            start=time.monotonic()
            try:
                engine.boundary(c['deadline'])
                if c['dataset_archive']:stage_tar(c['dataset_archive'],c['dataset'],evaluation=True)
                tm=test_manifest(c['dataset'],manifest['task_order'])
                # Original atomic_json serialization is part of reference equality; compare semanticdigest too.
                if tm['digest']!=ref['test_manifest_digest'] or len(tm['rows'])!=1725:raise ValueError('Test manifest differs from NESTED reference')
                engine.save_or_verify(root/'test_manifest.json',tm)
                if sha(root/'test_manifest.json')!=ref['test_manifest_sha256']:raise ValueError('Test manifest bytes differ from NESTED')
                ep=EvaluationFeatures(c['dataset'],extractor,'cuda')
                with activation(True):engine.evaluate(folder,manifest['task_order'],tm['rows'],factory,dict(provider=ep,test_digest=tm['digest']),matrix=False,deadline=c['deadline'])
            finally:timing['evaluation_seconds']=time.monotonic()-start
        return run_job(folder,manifest['online'],manifest['task_order'],ji,provider,factory,evaluation,c['deadline'],timing)
    finally:
        if (folder/'active.json').exists():timing['final_cursor']=read(folder/'active.json')['cursor']
        atomic_json(timing,session/'timing_components.json')
        atomic_json(dict(gpu=resource_usage('cuda'),scope='actual process incl IO/checkpoints, separate preflight/eval receipts; not purepapertraining'),session/'resources.json')
        extractor.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);c=read(p.parse_args().config);session=Path(c['session']);start=time.monotonic()
    try:status=run(c)
    except engine.Pause as exc:
        status='paused_at_atomic_boundary';atomic_json(dict(reason=str(exc)),session/'pause.json')
    except BaseException as exc:
        atomic_json(dict(status='failed_preserve_evidence',error=repr(exc),worker_seconds=time.monotonic()-start),session/'worker_status.json');raise
    atomic_json(dict(status=status,worker_seconds=time.monotonic()-start),session/'worker_status.json');print('[MATCHED CADIC STATUS]',status,flush=True)
if __name__=='__main__':main()
