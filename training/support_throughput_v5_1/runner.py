"""Orchestrate U1/P1/U2 with conservative crash accounting and no partial-pass retry."""
import json,time,sys,math
from pathlib import Path
from training.runtime import atomic_json
from notebook_support.common.process import stream_process
from .contract import contract,sha,digest,PASSES

def verify_complete(work,name):
    work=Path(work);folder=work/'passes'/name;p=folder/'complete.json'
    if not p.is_file():raise ValueError('Pass is incomplete: '+name)
    v=json.loads(p.read_text())
    if v['schema']!='throughput_pass_completion_v1' or v['pass_id']!=name:raise ValueError('Pass completion differs')
    actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    if actual!=set(v['files'])|{'complete.json'}:raise ValueError('Completed pass inventory changed')
    if sha(work/'reference_final_state.pt')!=v['reference_final_sha256']:raise ValueError('Reference final state changed')
    for n,h in v['files'].items():
        from .contract import safe
        if sha(safe(folder,n))!=h:raise ValueError('Completed pass payload changed: '+name+'/'+n)
    m=json.loads((folder/'measurement.json').read_text())
    if m['status']!='complete' or m['start_cursor']!=179 or m['end_cursor']!=195 or m['completed_images']!=16 or m['parity']['parent_events_exact'] is not True or m['final_state_digest']!=v['final_state_digest']:
        raise ValueError('Incomplete parity pass')
    if not math.isfinite(m['pipeline_seconds']) or m['pipeline_seconds']<=0:raise ValueError('Invalid pass timing')
    return m

def run(c,*,process=stream_process,now=time.monotonic):
    work=Path(c['work']);cfg=contract();ledger_path=work/'ledger.json'
    if ledger_path.exists():ledger=json.loads(ledger_path.read_text())
    else:ledger=dict(schema='throughput_budget_ledger_v1',charged_seconds=0.,reserved_transitions=0,passes={},sessions=[])
    if ledger.get('schema')!='throughput_budget_ledger_v1' or not isinstance(ledger.get('passes'),dict) or set(ledger['passes'])-set(PASSES):raise ValueError('Invalid profiler ledger schema/passes')
    previous_charge=ledger['charged_seconds']
    limit=cfg['profile']['cumulative_resume_budget_seconds'];reserve=cfg['profile']['export_reserve_seconds']
    if not isinstance(previous_charge,(int,float)) or not math.isfinite(previous_charge) or previous_charge<0 or previous_charge>limit:raise ValueError('Corrupt/exhausted budget ledger')
    if type(ledger['reserved_transitions']) is not int or not 0<=ledger['reserved_transitions']<=48 or ledger['reserved_transitions']!=16*len(ledger['passes']):raise ValueError('Attempt reservation ledger differs')
    seen_incomplete=False
    for name in PASSES:
        if name not in ledger['passes']:seen_incomplete=True;continue
        if seen_incomplete or ledger['passes'][name].get('reserved_attempts')!=16 or ledger['passes'][name].get('status') not in ('running','complete','failed','incomplete'):raise ValueError('Pass chronology/reservation changed')
        if ledger['passes'][name]['status']!='complete':seen_incomplete=True
    deadline=c['started']+limit-previous_charge-reserve
    session=dict(start_utc=c['session_utc'],previous_charged_seconds=previous_charge,
        export_reserve_charged_seconds=reserve,bootstrap_seconds=now()-c['started'],passes=[])
    ledger['sessions'].append(session)
    def save():
        ledger['charged_seconds']=previous_charge+(now()-c['started'])+reserve
        atomic_json(ledger,ledger_path)
    try:
        for name in PASSES:
            if name in ledger['passes']:
                if ledger['passes'][name]['status']!='complete':
                    if ledger['passes'][name]['status']=='failed':raise ValueError('Prior failed pass requires review; no automatic retry')
                    return 'profile_incomplete_budget' # Crash/partial is never replayed automatically.
                verify_complete(work,name);continue
            if now()>=deadline:return 'profile_incomplete_budget'
            if ledger['reserved_transitions']+16>48:raise ValueError('Attempt budget exhausted')
            # Reserve worst-case all16 BEFORE starting child: a hard kill cannot erase work from cap.
            ledger['reserved_transitions']+=16
            ledger['passes'][name]=dict(status='running',reserved_attempts=16,actual_attempted_event_ids=None)
            session['passes'].append(name);save()
            child=dict(work=str(work),pass_id=name,**c['prepared'],deadline=deadline)
            child['pass']=child.pop('pass_id')
            path=work/'configs'/(name+'.json');atomic_json(child,path)
            print('[PROFILE PASS START]',name,'179→195; cumulative budget remaining',round(deadline-now()),'s',flush=True)
            code=process([sys.executable,'-B','-m','training.support_throughput_v5_1.worker','--config',str(path)],
                work/'logs'/(name+'.log'),cwd=Path(__file__).resolve().parents[2])
            attempts=work/'passes'/name/'attempts.json'
            if attempts.exists():ledger['passes'][name]['actual_attempted_event_ids']=json.loads(attempts.read_text())['attempted_event_ids']
            if code:
                ledger['passes'][name]['status']='failed';save();raise RuntimeError('Profile worker failed: '+name)
            if not (work/'passes'/name/'complete.json').exists():
                ledger['passes'][name]['status']='incomplete';save();return 'profile_incomplete_budget'
            m=verify_complete(work,name)
            ledger['passes'][name].update(status='complete',measurement_sha256=sha(work/'passes'/name/'measurement.json'))
            save()
        values={n:verify_complete(work,n) for n in PASSES}
        if len({v['final_state_digest'] for v in values.values()})!=1 or len({v['trace_digest'] for v in values.values()})!=1:
            raise ValueError('U/P/U state or trace differs')
        u1,u2=values['U1']['pipeline_seconds'],values['U2']['pipeline_seconds']
        if min(u1,u2)<=0:raise ValueError('Invalid timing')
        atomic_json(dict(status='profile_complete_pending_review',pipeline_seconds={n:v['pipeline_seconds'] for n,v in values.items()},
            instrumentation_overhead_ratio=values['P1']['pipeline_seconds']/((u1+u2)/2)-1,
            reference_variation_ratio=abs(u1-u2)/((u1+u2)/2),same_known_window=True,scoped_exact_parity=True,
            statistical_CI=False,candidate_speedup_measured=False,AD_metrics=False,checkpoint376_modified=False,
            final_state_digest=values['U1']['final_state_digest']),work/'summary.json')
        return 'profile_complete_pending_review'
    finally:
        session['elapsed_before_export_seconds']=now()-c['started']
        session['budget_accounting']='actual elapsed before export + conservative180s export reserve; actual export seconds printed in Kaggle log; if export exceeds reserve stop for review'
        save()
