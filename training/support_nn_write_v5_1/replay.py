"""Identically segmented R/O timing; diagnostic copies outside primary intervals."""
from pathlib import Path
import time,hashlib
import torch
from training.runtime import atomic_json,atomic_torch_save
from training.support_throughput_v5_1.replay import (synchronize,state_digest,rng_fingerprint,assert_equal)
from models.hope_cad_v5_1.contract import digest
from .diagnostics import Segments,Captures

def replay(actor,provider,rows,folder,*,deadline,attempt,device='cuda',start=179,end=195,capture_events=(181,189),now=time.monotonic):
    folder=Path(folder)
    if actor.cursor!=start or end>=len(rows) or actor.finalized or actor.seen_ids!=[r['relative_path'] for r in rows[:start]]:raise ValueError('Wrong full-manifest predecessor')
    if any((folder/'events'/f'{i:05d}.json').exists() for i in range(start,end)):raise ValueError('Pass output must be fresh')
    rng=rng_fingerprint(device);segments=Segments(device);capture=Captures(segments,capture_events)
    before={}
    if str(device).startswith('cuda'):
        synchronize(device);before=dict(allocated_bytes=torch.cuda.memory_allocated(device),reserved_bytes=torch.cuda.memory_reserved(device));torch.cuda.reset_peak_memory_stats(device)
    chain=provider.prefix_chain;peak=0;events=[];error=None;wall=now();segments.start()
    with capture.installed(actor.learner):
        try:
            for i in range(start,end):
                if now()>=deadline:break
                attempt(i);capture.cursor=i
                event=actor.observe(provider.get(rows[i]),rows[i]['relative_path'])
                atomic_json(event,folder/'events'/f'{i:05d}.json');chain=digest([chain,event])
                peak=max(peak,actor.memory_profile()['persistent_tensor_bytes'])
                if actor.cursor%8==0:
                    atomic_torch_save(dict(schema='support_benchmark_active_v1',job_digest=provider.job_digest,
                        cursor=actor.cursor,session=actor.state_dict(),trace_digest=chain,
                        peak_persistent_tensor_bytes=peak,elapsed_seconds=now()-wall),folder/'active.pt')
                    with (folder/'active.pt').open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
                    atomic_json(dict(cursor=actor.cursor,sha256=h),folder/'active.json')
                events.append(event)
                if actor.cursor%8==0:print('[NN VALIDATION]',folder.name,actor.cursor,'/',len(rows),'labels',event['learner_labels'],flush=True)
        except BaseException as exc:error=exc
        finally:segments.finish()
    result=dict(status='complete' if actor.cursor==end and error is None else 'incomplete',start_cursor=start,end_cursor=actor.cursor,
        completed_images=len(events),pipeline_seconds_excluding_diagnostics=segments.pipeline_seconds,
        diagnostic_capture_seconds=segments.excluded_seconds,capture_intervals=segments.exclusions,
        total_replay_wall_seconds=now()-wall,peak_persistent_tensor_bytes=peak,
        gpu=dict(start=before),trace_digest=chain,finalized=actor.finalized,
        timing_scope='segmented pipeline; identical synchronized pre/post learning captures excluded; bootstrap/audit/recompute/export charged separately')
    if str(device).startswith('cuda'):result['gpu'].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    result['global_rng_unchanged']=rng_fingerprint(device)==rng
    if not result['global_rng_unchanged']:raise ValueError('Replay/capture modified global RNG')
    state=actor.state_dict();result['final_state_digest']=state_digest(state)
    atomic_json(result,folder/'measurement.json')
    if capture.values:atomic_torch_save(capture.values,folder/'capture_inputs.pt')
    if error is not None:raise error
    if actor.finalized:raise ValueError('Bounded replay finalized')
    if result['status']=='complete' and set(capture.values)!=set(capture_events):raise ValueError('Missing scheduled gradient captures')
    return result,state,capture.values
