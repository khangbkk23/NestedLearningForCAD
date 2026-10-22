"""Global-clock bounded replay; no full benchmark launcher and no finalization."""
import hashlib, math, time
from pathlib import Path
import torch
import random
import numpy as np
from training.runtime import atomic_json, atomic_torch_save
from models.hope_cad_v5_1.contract import digest
from .instrumentation import scope

def synchronize(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)

def rng_fingerprint(device='cpu'):
    n=np.random.get_state()
    value=dict(python=random.getstate(),numpy=[n[0],n[1].tolist(),*n[2:]],torch=torch.get_rng_state())
    if str(device).startswith('cuda'):value['cuda0']=torch.cuda.get_rng_state(device)
    return state_digest(value)

def state_digest(value):
    """Typed logical content, independent of torch.save storage IDs/container bytes."""
    h=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            t=v.detach().cpu().contiguous()
            h.update(('tensor:'+str(t.dtype)+':'+str(tuple(t.shape))+':').encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,dict):
            h.update(b'dict{')
            for k in sorted(v,key=lambda x:(type(x).__name__,str(x))):add(k);add(v[k])
            h.update(b'}')
        elif isinstance(v,(tuple,list)):
            h.update((type(v).__name__+'[').encode())
            for x in v:add(x)
            h.update(b']')
        else:
            h.update((type(v).__name__+':'+repr(v)+';').encode())
    add(value);return h.hexdigest()

def assert_equal(a,b,path='state'):
    if type(a) is not type(b): raise ValueError('Parity type mismatch: '+path)
    if torch.is_tensor(a):
        if a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a.cpu(),b.cpu()):raise ValueError('Parity tensor mismatch: '+path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('Parity keys mismatch: '+path)
        for k in a:assert_equal(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise ValueError('Parity length mismatch: '+path)
        for i,(x,y) in enumerate(zip(a,b)):assert_equal(x,y,path+'/'+str(i))
    elif a!=b:raise ValueError('Parity scalar mismatch: '+path)

class Incomplete(Exception): pass

def bounded_replay(actor,provider,rows,folder,*,start,end,chain,deadline,attempt,
                   device='cpu',profiler=None,expected_event=None,now=time.monotonic):
    """Only mathematical/reference hot work is timed. Audits are after final sync."""
    folder=Path(folder)
    if actor.cursor!=start or not 0<=start<end<len(rows):raise ValueError('Invalid bounded replay cursor/full manifest')
    if actor.seen_ids!=[r['relative_path'] for r in rows[:start]]:raise ValueError('Predecessor seen IDs mismatch')
    if actor.finalized:raise ValueError('Cannot replay finalized owner')
    for i in range(start,end):
        if (folder/'events'/f'{i:05d}.json').exists():raise ValueError('Pass output must be fresh')
    before_rng=rng_fingerprint(device)
    synchronize(device)
    start_mem={}
    if str(device).startswith('cuda'):
        start_mem=dict(allocated_bytes=torch.cuda.memory_allocated(device),reserved_bytes=torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
    began=now();peak=0;events=[];partial=None
    try:
        for i in range(start,end):
            if now()>=deadline:raise Incomplete('Budget reached inside pass; no automatic replay or stitched timing')
            attempt(i)  # In-memory count; durable pass reservation was published before timing.
            if profiler:profiler.cursor=i
            row=rows[i]
            with scope(profiler,'arrival',cuda=True):
                with scope(profiler,'provider_delivery'):
                    x=provider.get(row)
                event=actor.observe(x,row['relative_path'])
                with scope(profiler,'event_publication_and_chain'):
                    atomic_json(event,folder/'events'/f'{i:05d}.json')
                    chain=digest([chain,event])
                with scope(profiler,'persistent_memory_accounting'):
                    peak=max(peak,actor.memory_profile()['persistent_tensor_bytes'])
                if actor.cursor%8==0:
                    with scope(profiler,'periodic_checkpoint'):
                        state=dict(schema='support_benchmark_active_v1',job_digest=provider.job_digest,
                            cursor=actor.cursor,session=actor.state_dict(),trace_digest=chain,
                            peak_persistent_tensor_bytes=peak,elapsed_seconds=now()-began)
                        atomic_torch_save(state,folder/'active.pt')
                        with (folder/'active.pt').open('rb') as f:checksum=hashlib.file_digest(f,'sha256').hexdigest()
                        atomic_json(dict(cursor=actor.cursor,sha256=checksum),folder/'active.json')
            events.append(event)
            if actor.cursor%8==0:
                print('[BENCH TRAIN]',folder.name,actor.cursor,'/',len(rows),'labels',event['learner_labels'],flush=True)
    except BaseException as exc:
        partial=exc
    finally:
        synchronize(device);seconds=now()-began
        if profiler:profiler.enabled=False
    gpu=dict(start=start_mem)
    if str(device).startswith('cuda'):
        gpu.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    # No per-event full-state dumps/audits inside primary timing.
    scientific=actor.state_dict()
    if rng_fingerprint(device)!=before_rng:raise ValueError('Replay/instrumentation changed global RNG')
    result=dict(status='incomplete' if partial else 'complete',start_cursor=start,end_cursor=actor.cursor,
        completed_images=len(events),pipeline_seconds=seconds,peak_persistent_tensor_bytes=peak,gpu=gpu,
        final_state_digest=state_digest(scientific),trace_digest=chain,
        timing_scope='provider/observe/events/chain/memory accounting/checkpoints and reference-cadence progress prints; final audits/export excluded',
        finalized=actor.finalized)
    result['global_rng_unchanged']=True
    result['work_counters']=dict(proposal_replacements=sum(e['proposal']['replaced'] for e in events),
        committed_replacements=sum(e['proposal']['replaced'] for e in events if e['accepted']),
        vetoed_replacements=sum(e['proposal']['replaced'] for e in events if not e['accepted']),
        completed_credits=sum(e['credit'] is not None for e in events))
    if partial:result['error']=repr(partial)
    if profiler:result['profile']=profiler.report(seconds)
    atomic_json(result,folder/'measurement.json')
    if partial:
        if isinstance(partial,Incomplete):return result,scientific
        raise partial
    if actor.cursor!=end or actor.finalized:raise ValueError('Bounded replay finalized or changed cursor')
    if expected_event:
        for event in events:assert_equal(event,expected_event(event['cursor']),'event/'+str(event['cursor']))
    return result,scientific
