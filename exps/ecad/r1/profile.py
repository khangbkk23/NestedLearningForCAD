"""Synthetic-only six-arm K=2500 CUDA preflight, hard process wall deadline.

CPU gate runs first. No checkpoint, feature-cache or dataset reader is imported.
CUDA measurements remain UNVERIFIED until this actually runs on accessible GPU.
"""
import argparse
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import torch
from .controls import ARMS,MemoryArm,byte_capacities
from .distance import Work,Euclidean,global_nn,configure


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True)


def synthetic(event,high=False,dim=768):
    g=torch.Generator().manual_seed(41011+event)
    x=torch.randn(784,dim,generator=g)
    return x*3+20 if high else x


def sync(work=None):
    torch.cuda.synchronize()
    if work is not None:work.explicit_cuda_syncs+=1


def worker(name,out,capacity=2500):
    configure();torch.set_num_threads(2)
    arm=MemoryArm(name,capacity=capacity,device='cuda')
    records=[]
    def one(label,high=False):
        full_start=time.monotonic()
        x=synthetic(arm.event,high).to('cuda')
        work=Work();sync(work);torch.cuda.reset_peak_memory_stats()
        rss_before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        allocated_before=torch.cuda.memory_allocated()
        start=time.monotonic();stats=arm.update(x,event=arm.event,work=work);sync(work)
        update=time.monotonic()-start
        prepare_start=time.monotonic();support=arm.bank;sync()
        prepare_seconds=time.monotonic()-prepare_start
        query_work=Work();sync(query_work);start=time.monotonic()
        score=global_nn(x,support,Euclidean(work=query_work));sync(query_work)
        nn=time.monotonic()-start
        transfer_start=time.monotonic();score.cpu()
        transfer_seconds=time.monotonic()-transfer_start
        wall=time.monotonic()-full_start
        stats={k:int(v) if isinstance(v,torch.Tensor) else v for k,v in stats.items() if k!='decisions'}
        record={'label':label,'event':arm.event,'occupied':len(arm.bank),
                'selection_seconds':update,'global_784_nn_seconds':nn,'cached_image_wall_seconds':wall,
                'inference_bank_preparation_seconds':prepare_seconds,
                'result_transfer_seconds':transfer_seconds,
                'encoder_inclusive_seconds':None,'update_work':asdict(work),'query_work':asdict(query_work),
                'stats':stats,'peak_allocated':torch.cuda.max_memory_allocated(),
                'peak_reserved':torch.cuda.max_memory_reserved(),
                'transient_cuda_allocated_upper_bound':max(0,torch.cuda.max_memory_allocated()-allocated_before),
                'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'rss_peak_growth_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss_before}
        records.append(record);write(out/f'{label}.json',record)
    one('fill')
    while len(arm.bank)<arm.capacity:
        arm.update(synthetic(arm.event).to('cuda'),event=arm.event)
        if arm.event>40:raise RuntimeError('failed to reach actual saturation')
    assert len(arm.bank)==capacity
    one('saturated_typical_1');one('saturated_typical_2')
    one('saturated_high_1',True);one('saturated_high_2',True)
    # Exact online restore includes live distance caches; synthetic continuation
    # is a correctness check outside the timing samples, not raw-feature replay.
    state=arm.snapshot();stream=io.BytesIO();torch.save(state,stream);stream.seek(0)
    resumed=MemoryArm.restore(torch.load(stream,weights_only=True),'cuda')
    x=synthetic(arm.event).to('cuda')
    arm.update(x,event=arm.event);resumed.update(x,event=resumed.event);sync()
    torch.testing.assert_close(arm.bank,resumed.bank,rtol=0,atol=0)
    # A separate diagnostic trace includes instrumentation overhead. Actual CUDA
    # runtime sync APIs are counted from the trace, alongside scalar reads and
    # dynamic nonzero compactions. Do not substitute these timings above.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        arm.update(synthetic(arm.event).to('cuda'),event=arm.event);sync()
    sync_events={e.key:e.count for e in prof.key_averages()
                 if 'synchronize' in e.key.lower() or '_local_scalar_dense' in e.key or 'nonzero' in e.key}
    prof.export_chrome_trace(str(out/'synchronization_trace.json'))
    write(out/'summary.json',{'arm':name,'records':records,'ledger':arm.ledger(),
                             'resume':'PASS','diagnostic_sync_events':sync_events,
                             'lifecycle_peak_allocated':max([r['peak_allocated'] for r in records]+[torch.cuda.max_memory_allocated()]),
                             'lifecycle_peak_reserved':max([r['peak_reserved'] for r in records]+[torch.cuda.max_memory_reserved()]),
                             'lifecycle_peak_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                             'p95_certified':False,'sample_count_per_saturated_geometry':2})


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--capacity',type=int,default=2500)
    p.add_argument('--worker',choices=ARMS);args=p.parse_args()
    if args.worker:
        if not 16<=args.capacity<=2500:raise ValueError('bounded K=2500 or smaller matched byte capacity only')
        worker(args.worker,args.out,args.capacity);return
    args.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();deadline=start+600
    gate=subprocess.run([sys.executable,'-m','unittest','exps.ecad.r1.test_correctness','exps.ecad.r1.test_endpoints','-v'],capture_output=True,text=True,timeout=60)
    (args.out/'cpu_gate.log').write_text(gate.stdout+gate.stderr)
    result={'stage_cap_seconds':600,'synthetic_only':True,'cpu_gate_returncode':gate.returncode,
            'python':sys.version,'torch':torch.__version__,'cuda_build':torch.version.cuda,'arms':{}}
    if gate.returncode:
        result['status']='CPU_GATE_FAILED';write(args.out/'result.json',result);return
    if not torch.cuda.is_available():
        result.update(status='CUDA_RESOURCE_BLOCKED',cuda_available=False,device_count=torch.cuda.device_count())
        result['arms']={ledger+'_'+n:{'status':'NOT_RUN_NO_CUDA','capacity':2500 if ledger=='equal_descriptor' else byte_capacities()[n],
                                    'latency':None,'peak_cuda_bytes':None,'sync_count':None}
                        for ledger in ('equal_descriptor','equal_byte') for n in ARMS}
        result['elapsed_seconds']=time.monotonic()-start
        write(args.out/'result.json',result);print(json.dumps(result,indent=2));return
    result['device']=torch.cuda.get_device_name(0)
    result['device_total_bytes']=torch.cuda.get_device_properties(0).total_memory
    jobs=[(ledger,n,2500 if ledger=='equal_descriptor' else byte_capacities()[n])
          for ledger in ('equal_descriptor','equal_byte') for n in ARMS]
    for ledger,name,capacity in jobs:
        job=ledger+'_'+name
        remaining=deadline-time.monotonic()-3
        if remaining<=0:
            result['arms'][job]={'status':'NOT_STARTED_GLOBAL_CAP'};continue
        out=args.out/job;out.mkdir()
        try:
            run=subprocess.run([sys.executable,'-m','exps.ecad.r1.profile','--worker',name,'--capacity',str(capacity),'--out',str(out)],
                               capture_output=True,text=True,timeout=min(45,remaining))
            (out/'worker.log').write_text(run.stdout+run.stderr)
            result['arms'][job]={'status':'COMPLETE' if run.returncode==0 else 'FAILED','returncode':run.returncode,'capacity':capacity}
        except subprocess.TimeoutExpired as e:
            (out/'worker.log').write_text(str(e))
            result['arms'][job]={'status':'TIMEOUT','cap_seconds':min(45,remaining),'capacity':capacity}
    result['elapsed_seconds']=time.monotonic()-start
    result['status']='COMPLETE' if all(v['status']=='COMPLETE' for v in result['arms'].values()) else 'INCOMPLETE_PRESERVED_FAILURES'
    write(args.out/'result.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
