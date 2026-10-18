"""Causal prepare/train/witness/serialization for engineering and locked evaluation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import gc
import hashlib
import json
import resource
import time
from pathlib import Path
import torch
from ..core import Observation,clone_map
from ..preflight import tensor_file,device_row
from ..design import REGIMES,sha,save
from .data import SyntheticStream,Replay,WitnessEvaluator,partition
from .learner import Learner
from .design import OUT,MATRIX,validate_execution,budget_elapsed


def tensor_digest(value):
    digest=hashlib.sha256()
    def visit(obj):
        if torch.is_tensor(obj):
            x=obj.detach().cpu().contiguous()
            digest.update(str((tuple(x.shape),str(x.dtype))).encode()); digest.update(x.numpy().tobytes())
        elif isinstance(obj,dict):
            for k,v in sorted(obj.items()): digest.update(str(k).encode()); visit(v)
        elif isinstance(obj,(list,tuple)):
            for v in obj: visit(v)
        elif obj is not None: digest.update(str(obj).encode())
    visit(value)
    return digest.hexdigest()


def configure():
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()


def run(seed,regime,arm,stage='dryrun',cycles=None,reuse_prefix=False):
    assert arm in MATRIX[regime]
    if stage=='evaluation':
        validate_execution(); assert seed in range(3101,3111)
    else: assert stage in ['dryrun','reusecheck'] and seed in range(1901,1911)
    directory=OUT/stage/f'{regime}_{seed}_{arm}'
    directory.mkdir(parents=True,exist_ok=False)
    save(directory/'started.json',dict(seed=seed,regime=regime,arm=arm,utc=time.time()))
    started=time.perf_counter(); prior_seconds=budget_elapsed(); torch.cuda.reset_peak_memory_stats()
    stream=SyntheticStream(seed,regime); replay=Replay(seed)
    rng=torch.Generator().manual_seed(seed+11000)
    evaluator=WitnessEvaluator(seed,stream); evaluator.admit('A',0)
    learner=Learner(seed,device='cuda',arm=arm)
    total=(160 if regime=='R' else 128) if cycles is None else cycles
    endpoint=REGIMES[regime][-1]; onset=64+REGIMES[regime][0]
    acquisition=None; retained=None; new=None; recurrence=None
    start_cycle=0; shared_seconds=0.; inherited_meter={}
    if reuse_prefix:
        from .prefix import path as prefix_path,restore as restore_prefix
        if prefix_path(stage,seed,arm).exists():
            assert regime!='D8'
            stream,replay,rng,evaluator,acquisition,shared_seconds=restore_prefix(
                stage,seed,regime,arm,learner,directory)
            start_cycle=64; inherited_meter=dict(learner.meter)
    causal=[]; window_start=0; deployed_source=None; version_records=[]
    if start_cycle:
        causal=json.loads((OUT/stage/f'D8_{seed}_{arm}'/'trace.json').read_text())['causal_replay'][:64]
    for t in range(start_cycle,total):
        event,local=divmod(t,2)
        if t==(64 if regime=='C' else onset): evaluator.admit('B',t)
        if local==0: x_cpu=stream.arrive(event); x=x_cpu.to('cuda')
        s,q=partition(event,local,784,'cuda'); old=replay.sample(event)
        old_x=None if old is None else old.features.to('cuda')
        learner.replay_identity=-1 if old is None else old.identity
        if learner.window is None:
            learner.begin([device_row(a) for a in replay.anchors(rng)])
            window_start=t
            learner.mask_before=(onset-t if arm=='post_onset_only' and t<=onset<t+16 else 0)
            if arm=='deployed_ablation':
                source=OUT/stage/f'{regime}_{seed}_candidate'/f'window_{t//16:02d}.pt'
                deployed_source=torch.load(source,weights_only=False,map_location='cpu')
        if arm=='deployed_ablation':
            # Causal schedule: only the version available at this cycle is read.
            chunk=(t-window_start)//4
            state=deployed_source['versions']
            learner.M4=type(learner.M4)((n,p.to('cuda')) for n,p in state['M4'][chunk].items())
            learner.M16=type(learner.M16)((n,p.to('cuda')) for n,p in state['M16'][chunk].items())
        record=learner.step(x,s,q,old_x,event)
        if arm=='deployed_ablation':
            record['memory_available_after']=window_start+4*((t-window_start)//4)
            assert record['memory_available_after']<=t
        record['live_sha256']=tensor_digest((learner.live.phi,learner.live.moments,learner.live.clock))
        if (t+1)%4==0:
            record['memory_sha256']=tensor_digest((learner.M4,learner.M16))
            version_records.append(dict(clock=t+1,M4=clone_map(learner.M4),M16=clone_map(learner.M16)))
        causal.append(dict(clock=t+1,current=event,replay=None if old is None else old.identity))
        if local==1: replay.insert(Observation(event,x_cpu,q.cpu()))
        if t+1 in {64,endpoint,128,160}:
            tensor_file(directory/f'endpoint_{t+1}.pt',learner.live.payload())
        if t+1==64: acquisition=evaluator.evaluate('A',learner.model,learner.live)
        if t+1==endpoint: retained=evaluator.evaluate('A',learner.model,learner.live)
        if t+1==128: new=evaluator.evaluate('B',learner.model,learner.live)
        if t+1==160: recurrence=evaluator.evaluate('A',learner.model,learner.live)
        if (t+1)%16==0:
            tensor_file(directory/f'window_{t//16:02d}.pt',dict(
                versions=learner.last_versions,writes=version_records,
                M4=clone_map(learner.M4),M16=clone_map(learner.M16)))
            learner.last_versions=None; version_records=[]; deployed_source=None
        if reuse_prefix and t+1==64:
            from .prefix import capture
            capture(stage,seed,arm,learner,stream,replay,rng,evaluator,acquisition,time.perf_counter()-started)
        if torch.cuda.max_memory_reserved()>4.5*1024**3: raise RuntimeError('RESOURCE_INFEASIBLE: VRAM')
        if prior_seconds+time.perf_counter()-started>21600: raise RuntimeError('RESOURCE_INFEASIBLE:6hour')
    torch.cuda.synchronize()
    witness_hash=tensor_digest(evaluator.samples)
    witness_dir=OUT/stage/'witnesses'; witness_dir.mkdir(exist_ok=True)
    witness_path=witness_dir/f'{regime}_{seed}.pt'; witness_meta=witness_dir/f'{regime}_{seed}.json'
    if not witness_path.exists():
        tensor_file(witness_path,evaluator.samples)
        save(witness_meta,dict(tensor_sha256=witness_hash,file_sha256=sha(witness_path)))
    else: assert json.loads(witness_meta.read_text())['tensor_sha256']==witness_hash
    row=dict(seed=seed,regime=regime,arm=arm,cycles=total,acquisition=acquisition,
        retained=retained,new_normal=new,recurrence=recurrence,
        excess=None if retained is None or acquisition is None else max(retained-acquisition,0.),
        raw_excess=None if retained is None or acquisition is None else retained-acquisition,
        traces=learner.trace,causal_replay=causal,meter=dict(learner.meter),
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        host_maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        witness_tensor_sha256=witness_hash,seconds=time.perf_counter()-started,
        persistent_memory_bytes=sum(p.numel()*p.element_size() for m in [learner.M4,learner.M16] for p in m.values()),
        witness_loss_forwards=16*sum(v is not None for v in [acquisition,retained,new,recurrence]),
        shared_prefix_cycles=start_cycle,shared_prefix_seconds=shared_seconds,
        logical_seconds=time.perf_counter()-started+shared_seconds,
        physical_meter={k:v-inherited_meter.get(k,0) for k,v in learner.meter.items()},
        physical_witness_forwards=16*(sum(v is not None for v in [acquisition,retained,new,recurrence])-(1 if start_cycle else 0)))
    save(directory/'trace.json',row)
    tensor_file(directory/'final.pt',learner.payload())
    save(directory/'cost.json',dict(seconds=time.perf_counter()-started))
    save(directory/'hashes.json',{p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(directory.glob('*')) if p.is_file()})
    del learner,stream,replay,evaluator,version_records,deployed_source
    gc.collect(); torch.cuda.empty_cache()
    print(json.dumps({k:row[k] for k in ['seed','regime','arm','seconds','excess','new_normal','peak_reserved']}),flush=True)
    return row
