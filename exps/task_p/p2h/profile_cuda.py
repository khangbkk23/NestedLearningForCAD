"""G2: bounded, complete full-shape CUDA trajectory; runs only after G1b."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import gc
import io
import json
import resource
import time
from dataclasses import is_dataclass,fields
import torch
from .core import State,Observation,E_operator
from .learner import Learner
from .data import SyntheticStream,Replay,partition
from .preflight import configure,device_row
from .design import OUT,save,sha


def bytes_of(value):
    seen=set(); total=0
    def visit(x):
        nonlocal total
        if torch.is_tensor(x):
            key=(str(x.device),x.untyped_storage().data_ptr())
            if key not in seen:
                total+=x.untyped_storage().nbytes(); seen.add(key)
        elif isinstance(x,dict):
            for v in x.values(): visit(v)
        elif isinstance(x,(list,tuple)):
            for v in x: visit(v)
        elif is_dataclass(x):
            for f in fields(x): visit(getattr(x,f.name))
    visit(value)
    return total


def prepare_window(stream,replay,anchor_rng,start):
    # Harness-only tape: learners receive one arrived row, never this future table.
    anchors=[device_row(row) for row in replay.anchors(anchor_rng)]
    rows=[]; x=None
    for t in range(start,start+16):
        event,local=divmod(t,2)
        if local==0: x_cpu=stream.arrive(event); x=x_cpu.to('cuda')
        s,q=partition(event,local,784,'cuda')
        old=replay.sample(event)
        rows.append((x,s,q,None if old is None else old.features.to('cuda'),
                     event,None if old is None else old.identity))
        if local==1: replay.insert(Observation(event,x_cpu,q.cpu()))
    return anchors,rows


def run_window(learner,anchors,tape,resume=False):
    learner.begin(anchors); checkpoint_bytes=0
    for i,(x,s,q,r,identity,replay_id) in enumerate(tape):
        assert replay_id is None or replay_id<identity
        learner.replay_identity=-1 if replay_id is None else replay_id
        learner.step(x,s,q,r,identity)
        if resume and i==5:
            buffer=io.BytesIO(); torch.save(learner.payload(),buffer)
            checkpoint_bytes=buffer.tell(); buffer.seek(0)
            # Preserve CPU RNG tensors and original device tensors separately.
            data=torch.load(buffer,weights_only=False)
            restored=Learner(1901,device='cuda',eps_abs=learner.eps_abs,eps_rel=learner.eps_rel)
            restored.restore(data)
            del learner,data,buffer
            learner=restored
    assert learner.window is None and learner.trace[-1]['forward_parity']
    return learner,checkpoint_bytes


def main():
    assert json.loads((OUT/'G1b.json').read_text())['passed'],'G1b has not passed'
    configure()
    assert not (OUT/'G2_started.json').exists(),'G2 already started'
    save(OUT/'G2_started.json',dict(utc=time.time(),seed=1901,max_measured_windows=3,G3_untouched=True))
    start=time.perf_counter()
    eps=json.loads((OUT/'guard_calibration.json').read_text())
    learner=Learner(1901,device='cuda',eps_abs=eps['eps_abs'],eps_rel=eps['eps_rel'])
    stream=SyntheticStream(1901,'D8'); replay=Replay(1901)
    anchor_rng=torch.Generator().manual_seed(12901)
    anchors,tape=prepare_window(stream,replay,anchor_rng,0)
    learner,_=run_window(learner,anchors,tape) # one unmeasured warmup window
    del tape,anchors
    gc.collect(); torch.cuda.synchronize()
    records=[]; failure=None
    for window in range(3):
        anchors,tape=prepare_window(stream,replay,anchor_rng,16+window*16)
        entry=learner.live.detached()
        reference=entry.detached()
        torch.cuda.synchronize(); reference_start=time.perf_counter()
        for x,s,q,r,identity,replay_id in tape:
            reference,_,_=E_operator(learner.model,reference,x,s,q,r,False)
        torch.cuda.synchronize(); reference_s=time.perf_counter()-reference_start
        del reference
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); measured_start=time.perf_counter()
        before=dict(learner.meter)
        learner,checkpoint_bytes=run_window(learner,anchors,tape,resume=True)
        torch.cuda.synchronize(); candidate_s=time.perf_counter()-measured_start
        allocated=torch.cuda.max_memory_allocated(); reserved=torch.cuda.max_memory_reserved()
        tape_bytes=bytes_of((anchors,tape))
        del tape,anchors,entry
        gc.collect(); torch.cuda.synchronize()
        released_allocated=torch.cuda.memory_allocated()
        persistent=bytes_of((learner.model.state_dict(),learner.live,learner.M4,
                            learner.M16,learner.short_memory,learner.sketch.index))
        record=dict(window=window+1,cycles=16,reference_seconds=reference_s,
                    candidate_seconds=candidate_s,time_ratio=candidate_s/reference_s,
                    peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,
                    after_release_allocated_bytes=released_allocated,
                    enumerated_persistent_tensor_bytes=persistent,
                    unexplained_allocated_bytes=released_allocated-persistent,
                    tape_tensor_bytes=tape_bytes,checkpoint_bytes=checkpoint_bytes,
                    host_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    replay_host_bytes=bytes_of(replay.rows),
                    counts={k:v-before.get(k,0) for k,v in learner.meter.items()},
                    forward_parity=True,resume=True)
        records.append(record); save(OUT/f'G2_window_{window+1}.json',record)
        print(json.dumps(record),flush=True)
        if reserved>4.5*1024**3: failure='peak_reserved_above_4.5_GiB'
        elif candidate_s>10*reference_s: failure='wall_time_above_10x_stripped_E'
        elif len(records)>1 and record['unexplained_allocated_bytes']>records[0]['unexplained_allocated_bytes']+1024**2:
            failure='unreleased_memory_growth'
        elif time.perf_counter()-start>600: failure='600_second_profile_timeout'
        if failure: break
    save(OUT/'G2.json',dict(passed=failure is None,first_failure=failure,windows=records,
         seconds=time.perf_counter()-start,G3_untouched=True,
         verdict=None if failure is None else 'RESOURCE_INFEASIBLE'))
    if failure: raise SystemExit(3)


if __name__=='__main__': main()
