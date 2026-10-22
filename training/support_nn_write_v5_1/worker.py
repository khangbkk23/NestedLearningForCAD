"""One reference/candidate pass in a fresh process, capture and diagnostic receipt."""
import argparse,io,json,time
from pathlib import Path
import torch
from training.runtime import atomic_json,atomic_torch_save,resource_usage
from notebook_support.support_v5_1.common import configure_runtime,runtime_identity
from training.support_throughput_v5_1.contract import contract as original_contract,Evidence,verify_parent,digest,sha
from training.support_throughput_v5_1.replay import assert_equal,state_digest
from training.support_benchmark_v5_1.contract import session_for,source_digest,study_digest
from training.support_benchmark_v5_1.data import NormalFeatures
from models.hope_cad_v5_1.contract import design_digest
from models.feature_extractors.cadic_vit_v1 import CADICViTConfig,CADICViTFeatureExtractor
from .contract import contract,verify_source
from .candidate import activation
from .replay import replay
from .diagnostics import recompute

def run(c):
    t=time.monotonic();configure_runtime();cfg=contract();original=original_contract();source=verify_source()
    name=c['pass'];role=cfg['roles'][name];folder=Path(c['work'])/'passes'/name;folder.mkdir(parents=True,exist_ok=False)
    runtime=runtime_identity();attempted=[];extractor=None
    try:
        with Evidence(c['parent']) as e:
            p=verify_parent(e,original);rows=p['rows']
            if runtime!=p['suite']['identity']['runtime'] or not torch.cuda.is_available():raise ValueError('Exact reviewed T4 runtime required')
            if source_digest()!=original['frozen_reference']['source_digest'] or study_digest()!=original['study_digest'] or design_digest()!=original['method_digest']:raise ValueError('Scientific/source identity changed')
            task=torch.load(io.BytesIO(e.read(original['parent']['profile_start_entry'])),map_location='cpu',weights_only=True)
            if task['schema']!='support_benchmark_task_v1' or task['cursor']!=179 or task['job_digest']!=original['parent']['job_digest'] or task['task_id']!=0:raise ValueError('Wrong predecessor')
            chain=digest('empty event stream')
            for i in range(179):
                event=e.json(f'runs/NESTED_seed0/events/{i:05d}.json')
                if event['cursor']!=i or event['identity']!=rows[i]['relative_path']:raise ValueError('Prefix chronology differs')
                chain=digest([chain,event])
            events={i:e.json(f'runs/NESTED_seed0/events/{i:05d}.json') for i in range(179,195)}
            receipts={digest(rows[i]):e.json('runs/NESTED_seed0/feature_receipts/'+digest(rows[i])+'.json') for i in range(179,195)}
        with activation(role=='OPTIMIZED'):
            actor=session_for('NESTED_seed0','cuda');actor.load_state_dict(task['session'])
            if actor.cursor!=179 or actor.learner.labels!=165 or actor.learner.slow_updates!=20:raise ValueError('Predecessor clocks changed')
            extractor=CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=c['checkpoint'],checkpoint_identity='timm/vit_base_patch8_224.augreg_in21k'),'cuda')
            provider=NormalFeatures(c['dataset'],extractor,folder/'feature_receipts','cuda')
            provider.job_digest=original['parent']['job_digest'];provider.prefix_chain=chain
            bootstrap=dict(seconds=time.monotonic()-t,runtime=runtime,source=source,role=role,cold_derived_cache=True,warmup_arrivals=0,predecessor_cursor=179,prefix_chain=chain,gpu=resource_usage('cuda'))
            atomic_json(bootstrap,folder/'bootstrap.json')
            m,state,captures=replay(actor,provider,rows,folder,deadline=c['deadline'],attempt=attempted.append)
            audit=time.monotonic()
            if m['status']=='complete':
                for i in range(179,195):
                    assert_equal(json.loads((folder/'events'/f'{i:05d}.json').read_text()),events[i],'parent_event')
                    key=digest(rows[i]);assert_equal(json.loads((folder/'feature_receipts'/(key+'.json')).read_text()),receipts[key],'feature_receipt')
                if state_digest(state)!=cfg['G1_final_state_digest']:raise ValueError('Endpoint differs from reviewed G1 state195')
                if state['learner']['labels']!=181 or state['learner']['slow_updates']!=22 or len(state['learner']['records'])!=5 or state['pending'] is None or state['counters']['censored']:raise ValueError('End clocks/pending changed')
                final=Path(c['work'])/'reference_final_state.pt'
                if name=='R1':atomic_torch_save(state,final)
                else:assert_equal(torch.load(final,map_location='cpu',weights_only=True),state,'reference_final_state')
                if time.monotonic()>=c['deadline']:m['status']='incomplete_diagnostic_budget'
                else:
                    outputs={}
                    for i,cap in captures.items():
                        if time.monotonic()>=c['deadline']:m['status']='incomplete_diagnostic_budget';break
                        outputs[i]=recompute(cap,lambda:session_for('NESTED_seed0','cuda'),'cuda')
                    if len(outputs)==2 and m['status']=='complete':
                        atomic_torch_save(outputs,folder/'gradients.pt')
                        reference=Path(c['work'])/'reference_gradients.pt'
                        if name=='R1':atomic_torch_save(outputs,reference)
                        else:assert_equal(torch.load(reference,map_location='cpu',weights_only=True),outputs,'raw_gradient_and_steps')
                        m['gradient_digest']=state_digest(outputs)
                        m['parity']=dict(parent_events_exact=True,feature_receipts_exact=True,full_final_state_exact=True,
                            G1_endpoint_exact=True,raw_actual_meta_gradients_and_steps_exact=True,gradient_capture_events=[181,189])
                active=folder/'active.pt'
                if active.exists() and m['status']=='complete':
                    m['disposable_periodic_checkpoint']=dict(cursor=192,sha256=sha(active),removed_after_measurement=True);active.unlink()
            m.update(attempted_event_ids=attempted,role=role,bootstrap_seconds=bootstrap['seconds'],post_measurement_seconds=time.monotonic()-audit,worker_total_seconds=time.monotonic()-t)
            m['post_measurement_gpu_including_diagnostics']=resource_usage('cuda')
            atomic_json(m,folder/'measurement.json')
            atomic_json(dict(attempted_event_ids=attempted,worker_elapsed_seconds=time.monotonic()-t),folder/'attempts.json')
            if m['status']=='complete':
                files={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()}
                atomic_json(dict(schema='nn_write_pass_completion_v1',pass_id=name,role=role,files=files,
                    reference_final_sha256=sha(Path(c['work'])/'reference_final_state.pt'),reference_gradients_sha256=sha(Path(c['work'])/'reference_gradients.pt'),final_state_digest=m['final_state_digest'],gradient_digest=m['gradient_digest']),folder/'complete.json')
            return m['status']
    finally:
        if extractor is not None:extractor.close()
        if not (folder/'attempts.json').exists():atomic_json(dict(attempted_event_ids=attempted,worker_elapsed_seconds=time.monotonic()-t),folder/'attempts.json')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);c=json.loads(Path(ap.parse_args().config).read_text())
    try:print('[NN PASS]',c['pass'],run(c),flush=True)
    except BaseException as e:
        atomic_json(dict(status='failed',error=repr(e)),Path(c['work'])/'passes'/c['pass']/'failure.json');raise
if __name__=='__main__':main()
