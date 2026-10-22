"""One fresh subprocess/owner per fixed pass. GPU execution is user-triggered only."""
import argparse, io, json, time
from pathlib import Path
import torch
from models.hope_cad_v5_1.contract import digest, design_digest
from training.runtime import atomic_json, atomic_torch_save, resource_usage
from training.support_benchmark_v5_1.contract import session_for, source_digest, study_digest
from training.support_benchmark_v5_1.data import NormalFeatures
from models.feature_extractors.cadic_vit_v1 import CADICViTConfig, CADICViTFeatureExtractor
from notebook_support.support_v5_1.common import configure_runtime, runtime_identity
from .contract import contract, verify_source, verify_parent, Evidence, sha
from .replay import bounded_replay, assert_equal, state_digest
from .instrumentation import Profiler

def run(c):
    began=time.monotonic(); configure_runtime(); cfg=contract(); f=cfg['profile']
    source=verify_source(); work=Path(c['work']); name=c['pass']; folder=work/'passes'/name
    folder.mkdir(parents=True,exist_ok=False)
    if name not in f['passes']:raise ValueError('Unknown pass')
    runtime=runtime_identity()
    with Evidence(c['parent']) as e:
        parent=verify_parent(e,cfg); rows=parent['rows']; expected_runtime=parent['suite']['identity']['runtime']
        if runtime!=expected_runtime or not torch.cuda.is_available():raise ValueError('Pinned parent T4 runtime differs')
        if source_digest()!=cfg['frozen_reference']['source_digest'] or study_digest()!=cfg['study_digest'] or design_digest()!=cfg['method_digest']:
            raise ValueError('Frozen source/study/method differs')
        actor=session_for('NESTED_seed0','cuda')
        payload=torch.load(io.BytesIO(e.read(cfg['parent']['profile_start_entry'])),map_location='cpu',weights_only=True)
        if set(payload)!={'schema','job_digest','task_id','cursor','session'} or payload['schema']!='support_benchmark_task_v1' or payload['job_digest']!=cfg['parent']['job_digest'] or payload['task_id']!=0 or payload['cursor']!=179:
            raise ValueError('Wrong predecessor wrapper')
        actor.load_state_dict(payload['session'])
        if actor.learner.labels!=165 or actor.learner.slow_updates!=20 or actor.cursor!=179:raise ValueError('Predecessor clocks changed')
        chain=digest('empty event stream')
        for i in range(179):
            event=e.json(f'runs/NESTED_seed0/events/{i:05d}.json')
            if event['cursor']!=i or event['identity']!=rows[i]['relative_path']:raise ValueError('Prefix chronology changed')
            chain=digest([chain,event])
        expected={i:e.json(f'runs/NESTED_seed0/events/{i:05d}.json') for i in range(179,195)}
        # Read-only expected store never prepopulates fresh provider/event output.
        receipts={digest(rows[i]):e.json('runs/NESTED_seed0/feature_receipts/'+digest(rows[i])+'.json') for i in range(179,195)}
    extractor=CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=c['checkpoint'],checkpoint_identity='timm/vit_base_patch8_224.augreg_in21k'),'cuda')
    provider=NormalFeatures(c['dataset'],extractor,folder/'feature_receipts','cuda')
    provider.job_digest=cfg['parent']['job_digest']
    bootstrap=dict(seconds=time.monotonic()-began,gpu=resource_usage('cuda'),runtime=runtime,source=source,
        cold_derived_cache=True,warmup_arrivals=0,OS_page_cache='uncontrolled; pass order reported',
        predecessor_cursor=179,prefix_chain=chain,learned_state_adopted=False)
    atomic_json(bootstrap,folder/'bootstrap.json')
    attempted=[]
    try:
        if name=='P1':
            with Profiler('cuda') as profiler:
                result,state=bounded_replay(actor,provider,rows,folder,start=179,end=195,chain=chain,
                    deadline=c['deadline'],attempt=attempted.append,device='cuda',profiler=profiler,expected_event=expected.__getitem__)
        else:
            result,state=bounded_replay(actor,provider,rows,folder,start=179,end=195,chain=chain,
                deadline=c['deadline'],attempt=attempted.append,device='cuda',expected_event=expected.__getitem__)
        audit_start=time.monotonic()
        result['attempted_event_ids']=attempted
        result['bootstrap_seconds']=bootstrap['seconds']
        if result['status']=='complete':
            if state['learner']['labels']!=181 or state['learner']['slow_updates']!=22 or len(state['learner']['records'])!=5 or state['pending'] is None:
                raise ValueError('Replay endpoint clocks/pending changed')
            slow=[];previous=20
            for i in range(179,195):
                actual_event=json.loads((folder/'events'/f'{i:05d}.json').read_text())
                if actual_event['slow_updates']>previous:slow.append(i)
                previous=actual_event['slow_updates']
                key=digest(rows[i]); actual=json.loads((folder/'feature_receipts'/(key+'.json')).read_text())
                assert_equal(actual,receipts[key],'feature_receipt')
            if slow!=[181,189]:raise ValueError('Slow commit event indices changed')
            final=work/'reference_final_state.pt'
            if name=='U1':atomic_torch_save(state,final)
            else:
                reference=torch.load(final,map_location='cpu',weights_only=True)
                assert_equal(reference,state)
            result['parity']=dict(parent_events_exact=True,raw_feature_receipts_exact=True,
                final_state_exact_vs_U1=True,full_intermediate_state_dumped=False,
                gradient_parity='not part of profiler G1; G2/G3 later',slow_event_indices=slow)
            # Actual periodic writes remain in timing. Keep hashes/receipts; final U1 state supports cross-session comparison.
            active=folder/'active.pt'
            if active.exists():
                result['diagnostic_checkpoint_removed_after_measurement']=dict(cursor=192,sha256=sha(active),reason='disposable replay; reference_final_state.pt retained separately; never benchmark resume')
                active.unlink()
        result['post_measurement_audit_seconds']=time.monotonic()-audit_start
        result['worker_total_seconds']=time.monotonic()-began
        atomic_json(result,folder/'measurement.json')
        atomic_json(dict(attempted_event_ids=attempted,worker_elapsed_seconds=time.monotonic()-began),folder/'attempts.json')
        if result['status']=='complete':
            files={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()}
            atomic_json(dict(schema='throughput_pass_completion_v1',pass_id=name,files=files,
                final_state_digest=state_digest(state),reference_final_sha256=sha(work/'reference_final_state.pt')),
                folder/'complete.json')
        return result['status']
    finally:
        extractor.close()
        if not (folder/'attempts.json').exists():
            atomic_json(dict(attempted_event_ids=attempted,worker_elapsed_seconds=time.monotonic()-began),folder/'attempts.json')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True)
    c=json.loads(Path(ap.parse_args().config).read_text())
    try:
        status=run(c)
        print('[PROFILE PASS]',c['pass'],status,flush=True)
    except BaseException as exc:
        atomic_json(dict(status='failed',error=repr(exc)),Path(c['work'])/'passes'/c['pass']/'failure.json')
        raise
if __name__=='__main__':main()
