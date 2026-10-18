"""Three released CUDA windows, same derivatives; time waiver only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import gc
import io
import json
import resource
import time
import torch
from ..profile_cuda import prepare_window,bytes_of
from ..core import E_operator
from ..data import Replay
from ..preflight import configure
from .data import SyntheticStream
from .learner import Learner
from .design import OUT,ORIGINAL,save


def window(learner,stream,replay,rng,start,measured):
    anchors,tape=prepare_window(stream,replay,rng,start)
    reference=learner.live.detached()
    torch.cuda.synchronize(); then=time.perf_counter()
    for x,s,q,r,i,ri in tape: reference,_,_=E_operator(learner.model,reference,x,s,q,r,False)
    torch.cuda.synchronize(); refseconds=time.perf_counter()-then
    del reference
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); then=time.perf_counter()
    before=dict(learner.meter); learner.begin(anchors); checkpoint=0
    for t,(x,s,q,r,i,ri) in enumerate(tape):
        learner.replay_identity=-1 if ri is None else ri
        learner.step(x,s,q,r,i)
        if t==5:
            buffer=io.BytesIO(); torch.save(learner.payload(),buffer); checkpoint=buffer.tell(); buffer.seek(0)
            restored=Learner(1901,device='cuda')
            restored.restore(torch.load(buffer,weights_only=False))
            learner=restored
    assert learner.window is None and learner.trace[-1]['forward_parity']
    torch.cuda.synchronize()
    result=dict(seconds=time.perf_counter()-then,reference_seconds=refseconds,
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        tape_bytes=bytes_of((anchors,tape)),checkpoint_bytes=checkpoint,
        meter={k:v-before.get(k,0) for k,v in learner.meter.items()})
    learner.last_versions=None
    return learner,result


def main():
    configure(); started=time.perf_counter()
    assert (OUT/'design_lock.json').exists()
    assert json.loads((OUT/'control_tests.json').read_text())['passed']
    save(OUT/'profile_started.json',dict(utc=time.time(),G3_untouched=True))
    learner=Learner(1901,device='cuda'); stream=SyntheticStream(1901,'D8'); replay=Replay(1901)
    rng=torch.Generator().manual_seed(12901)
    learner,_=window(learner,stream,replay,rng,0,False)
    gc.collect(); torch.cuda.empty_cache()
    records=[]; failure=None
    for j in range(3):
        learner,r=window(learner,stream,replay,rng,16+16*j,True)
        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        persistent=bytes_of((learner.model.state_dict(),learner.live,learner.M4,learner.M16,
            learner.short_memory,learner.short16,learner.sketch.index))
        allocated=torch.cuda.memory_allocated()
        r.update(window=j+1,after_release_allocated=allocated,persistent_tensor_bytes=persistent,
            unexplained_allocation=allocated-persistent,host_maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        records.append(r); print(json.dumps(r),flush=True)
        if r['peak_reserved']>4.5*1024**3: failure='VRAM_cap'
        if len(records)>1 and allocated>records[0]['after_release_allocated']+1024**2*j: failure='unreleased_growth'
        if time.perf_counter()-started>600: failure='profile_timeout'
        if failure: break
    passed=failure is None and len(records)==3
    save(OUT/'profile.json',dict(passed=passed,failure=failure,windows=records,
        verdict=None if passed else 'RESOURCE_INFEASIBLE',G3_untouched=True))
    save(OUT/'profile'/'cost.json',dict(seconds=time.perf_counter()-started))
    if not passed: raise SystemExit(3)


if __name__=='__main__': main()
