"""Four passes and descriptive speed/memory gates; never adopt cloned state."""
import json,math,sys,time
from pathlib import Path
from training.runtime import atomic_json
from notebook_support.common.process import stream_process
from training.support_throughput_v5_1.contract import sha,safe
from .contract import contract,PASSES

def verify_complete(work,name):
    work=Path(work);folder=work/'passes'/name;c=json.loads((folder/'complete.json').read_text())
    if c['schema']!='nn_write_pass_completion_v1' or c['pass_id']!=name or c['role']!=contract()['roles'][name]:raise ValueError('Pass identity differs')
    actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    if actual!=set(c['files'])|{'complete.json'}:raise ValueError('Complete pass inventory changed')
    for n,h in c['files'].items():
        if sha(safe(folder,n))!=h:raise ValueError('Pass payload SHA differs')
    for n,key in [('reference_final_state.pt','reference_final_sha256'),('reference_gradients.pt','reference_gradients_sha256')]:
        if sha(work/n)!=c[key]:raise ValueError('Reference diagnostic payload changed')
    m=json.loads((folder/'measurement.json').read_text())
    required={'parent_events_exact','feature_receipts_exact','full_final_state_exact','G1_endpoint_exact','raw_actual_meta_gradients_and_steps_exact'}
    if m['status']!='complete' or m['start_cursor']!=179 or m['completed_images']!=16 or m['end_cursor']!=195 or m['attempted_event_ids']!=list(range(179,195)) or m['capture_intervals']!=4 or any(m.get('parity',{}).get(k) is not True for k in required) or m['parity'].get('gradient_capture_events')!=[181,189] or m['finalized'] or not m['global_rng_unchanged'] or m['role']!=c['role']:raise ValueError('Pass incomplete or parity absent')
    if m['final_state_digest']!=contract()['G1_final_state_digest']:raise ValueError('Endpoint differs from reviewed G1')
    if m['final_state_digest']!=c['final_state_digest'] or m['gradient_digest']!=c['gradient_digest']:raise ValueError('Pass state/gradient digest differs')
    return m

def gates(values,c):
    times={n:m['pipeline_seconds_excluding_diagnostics'] for n,m in values.items()}
    if set(times)!=set(PASSES) or any(not math.isfinite(v) or v<=0 for v in times.values()):raise ValueError('Invalid timing')
    if len({m['final_state_digest'] for m in values.values()})!=1 or len({m['gradient_digest'] for m in values.values()})!=1:raise ValueError('Candidate state/gradient mismatch')
    r=(times['R1']+times['R2'])/2;o=(times['O1']+times['O2'])/2;ratio=o/r
    rvar=abs(times['R1']-times['R2'])/r;ovar=abs(times['O1']-times['O2'])/o
    pairs=[times['O1']/times['R1'],times['O2']/times['R2']]
    # Algebraically the registered relative-noise rule, avoiding cancellation at equality.
    noise_seconds=max(abs(times['R1']-times['R2']),r*abs(times['O1']-times['O2'])/o)
    speed=ratio<=1 and all(v<1 for v in pairs) and r-o>noise_seconds
    ref_peak=max(values[n]['gpu']['peak_allocated_bytes'] for n in ['R1','R2'])
    opt_peak=max(values[n]['gpu']['peak_allocated_bytes'] for n in ['O1','O2'])
    memory=opt_peak<=ref_peak+c['candidate_peak_headroom_bytes']
    return dict(status='validation_complete_pending_review',times=times,aggregate_candidate_over_reference_ratio=ratio,
        paired_ratios=pairs,within_reference_variation=rvar,within_candidate_variation=ovar,speed_gate_passed=speed,
        mean_gain_seconds=r-o,noise_envelope_reference_seconds=noise_seconds,
        memory_gate_passed=memory,reference_peak_allocated_bytes=ref_peak,candidate_peak_allocated_bytes=opt_peak,
        numerical_parity_passed=True,all_engineering_gates_passed=speed and memory,
        interpretation='eligible_pending_review' if speed and memory else 'exact_parity_speed_or_memory_inconclusive',
        statistical_CI=False,benchmark_adoption=False,AD_superiority=False)

def run(c,*,process=stream_process,now=time.monotonic):
    cfg=contract();work=Path(c['work']);path=work/'ledger.json'
    ledger=json.loads(path.read_text()) if path.exists() else dict(schema='nn_write_budget_ledger_v1',charged_seconds=0.,reserved_transitions=0,passes={},sessions=[])
    if ledger.get('schema')!='nn_write_budget_ledger_v1' or not isinstance(ledger.get('passes'),dict) or set(ledger['passes'])-set(PASSES):raise ValueError('Wrong validation ledger')
    spent=ledger['charged_seconds'];reserve=cfg['export_reserve_seconds']
    if not isinstance(spent,(float,int)) or not math.isfinite(spent) or not 0<=spent<=cfg['cumulative_budget_seconds']:raise ValueError('Invalid/exhausted budget')
    if type(ledger['reserved_transitions']) is not int or ledger['reserved_transitions']!=16*len(ledger['passes']) or not 0<=ledger['reserved_transitions']<=64:raise ValueError('Invalid reserved attempts')
    stopped=False
    for n in PASSES:
        if n not in ledger['passes']:stopped=True;continue
        if stopped or ledger['passes'][n].get('reserved_attempts')!=16 or ledger['passes'][n]['status'] not in ('running','complete','failed','incomplete'):raise ValueError('Invalid pass chronology')
        if ledger['passes'][n]['status']!='complete':stopped=True
    deadline=c['started']+cfg['cumulative_budget_seconds']-spent-reserve
    session=dict(start_utc=c['session_utc'],previous_charged_seconds=spent,passes=[]);ledger['sessions'].append(session)
    def save():
        ledger['charged_seconds']=spent+now()-c['started']+reserve;atomic_json(ledger,path)
    try:
        for n in PASSES:
            if n in ledger['passes']:
                if ledger['passes'][n]['status']=='failed':raise ValueError('Previous failure requires review; no retry')
                if ledger['passes'][n]['status']!='complete':return 'validation_incomplete_budget'
                verify_complete(work,n);continue
            if now()>=deadline:return 'validation_incomplete_budget'
            ledger['reserved_transitions']+=16;ledger['passes'][n]=dict(status='running',reserved_attempts=16);session['passes'].append(n);save()
            config=dict(**c['prepared'],work=str(work),deadline=deadline);config['pass']=n
            p=work/'configs'/(n+'.json');atomic_json(config,p)
            print('[NN PASS START]',n,cfg['roles'][n],'remaining',round(deadline-now()),'s',flush=True)
            code=process([sys.executable,'-B','-m','training.support_nn_write_v5_1.worker','--config',str(p)],work/'logs'/(n+'.log'),cwd=Path(__file__).resolve().parents[2])
            attempts=work/'passes'/n/'attempts.json'
            if attempts.exists():ledger['passes'][n]['actual_attempted_event_ids']=json.loads(attempts.read_text())['attempted_event_ids']
            if code:ledger['passes'][n]['status']='failed';save();raise RuntimeError('Validation worker failed: '+n)
            if not (work/'passes'/n/'complete.json').exists():ledger['passes'][n]['status']='incomplete';save();return 'validation_incomplete_budget'
            verify_complete(work,n);ledger['passes'][n]['status']='complete';save()
        summary=gates({n:verify_complete(work,n) for n in PASSES},cfg)
        atomic_json(summary,work/'summary.json');return summary['status']
    finally:
        session['elapsed_before_export_seconds']=now()-c['started'];session['export_reserve_charged_seconds']=reserve
        session['accounting']='actual elapsed before export+180s conservative reserve; actual export timing printed separately'
        save()
