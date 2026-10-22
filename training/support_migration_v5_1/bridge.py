"""Lockstep shadow continuation; no publish before all eight transitions agree."""
import time
from pathlib import Path
from training.runtime import atomic_json
from training.support_nn_write_v5_1.candidate import activation
from training.support_throughput_v5_1.replay import assert_equal,state_digest,synchronize,rng_fingerprint
from models.hope_cad_v5_1.contract import digest

class Incomplete(Exception):pass

def shadow(reference,candidate,provider,rows,folder,*,start,end,chain,deadline,device='cuda',now=time.monotonic):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    if start>=end or end>=len(rows) or reference.cursor!=start or candidate.cursor!=start:raise ValueError('Wrong bridge cursor/full manifest')
    assert_equal(reference.state_dict(),candidate.state_dict(),'initial_T4_pending_revalidated_state')
    if state_digest(reference.state_dict())!=state_digest(candidate.state_dict()):raise ValueError('Initial scientific bits differ')
    if reference.seen_ids!=[r['relative_path'] for r in rows[:start]] or reference.finalized:raise ValueError('Wrong predecessor observed identities')
    initial=reference.state_dict();pending=initial['pending'];rng=rng_fingerprint(device)
    pending_check=dict(loaded_state_digest=state_digest(initial),pending_present=pending is not None,
        prediction=None if pending is None else pending['prediction'],denominator=None if pending is None else pending['decision_d'],original_loader_revalidated_both=True)
    atomic_json(pending_check,folder/'pending_check.json')
    ledger=dict(schema='nn_write_bridge_attempts_v1',arrivals=[],owner_transitions=[],completed_pairs=[],times=dict(provider=0.,reference=0.,candidate=0.,comparison=0.),status='running')
    peak=0;began=now()
    def save():atomic_json(ledger,folder/'attempts.json')
    save()
    try:
        for i in range(start,end):
            if now()>=deadline:raise Incomplete('Bridge budget exhausted; preserve parent, no automatic retry')
            # Durable reservation precedes any raw read or owner transition, including rolled-back calls.
            ledger['arrivals'].append(i);save();t=now();raw=provider.get(rows[i]);synchronize(device);ledger['times']['provider']+=now()-t
            raw_digest=state_digest(raw)
            events=[]
            for name,owner,opt in [('reference',reference,False),('candidate',candidate,True)]:
                if now()>=deadline:raise Incomplete('Bridge budget exhausted before owner; no publication')
                ledger['owner_transitions'].append(dict(cursor=i,owner=name));save();synchronize(device);t=now()
                with activation(opt):event=owner.observe(raw,rows[i]['relative_path'])
                synchronize(device);ledger['times'][name]+=now()-t;events.append(event)
                if state_digest(raw)!=raw_digest:raise ValueError('Owner mutated shared raw arrival')
            t=now();assert_equal(events[0],events[1],'bridge_event_'+str(i))
            rs=reference.state_dict();cs=candidate.state_dict();assert_equal(rs,cs,'bridge_full_state_'+str(i))
            bit_digest=state_digest(rs)
            if bit_digest!=state_digest(cs):raise ValueError('State logical bits differ')
            if reference.finalized or candidate.finalized:raise ValueError('Bridge cannot finalize stream')
            chain=digest([chain,events[1]]);peak=max(peak,candidate.memory_profile()['persistent_tensor_bytes'])
            ledger['times']['comparison']+=now()-t
            atomic_json(events[1],folder/'events'/f'{i:05d}.json')
            atomic_json(dict(cursor=i,event_digest=digest(events[1]),state_digest=bit_digest,trace_digest=chain),folder/'pairs'/f'{i:05d}.json')
            ledger['completed_pairs'].append(i);save()
            print('[BRIDGE]',candidate.cursor,'/',len(rows),'exact; labels',candidate.counters['labels_completed'],flush=True)
        if rng_fingerprint(device)!=rng:raise ValueError('Bridge modified global RNG')
        ledger['status']='complete'
        return dict(state=candidate.state_dict(),trace_digest=chain,peak_persistent_tensor_bytes=peak,pending_check=pending_check,ledger=ledger)
    finally:
        ledger['elapsed_seconds']=now()-began;save()
