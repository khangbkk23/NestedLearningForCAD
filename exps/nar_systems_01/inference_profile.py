"""Fixed-workload, inference-only NAR profiler. No training/test masks."""
import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.profiler import profile,ProfilerActivity,record_function
from dataset.benchmark_protocol_v1 import _ImageDataset
from exps.nar_pilot_00 import core as K
from exps.nar_pilot_00 import full as R
from exps.nar_pilot_00.detector import Detector
from exps.nar_systems_01.resident import ResidentDetector
from exps.nar_systems_01.audit import OUT

MODES=('oracle','resident','resident_deferred')
SOURCES=('exps/nar_systems_01/inference_profile.py','exps/nar_systems_01/resident.py',
 'exps/nar_systems_01/test_resident.py','agents/plans/nar_systems_01_protocol.md')

def stats(xs):
    return {'mean_ms':float(np.mean(xs)*1000),'p50_ms':float(np.percentile(xs,50)*1000),
      'p95_ms':float(np.percentile(xs,95)*1000),'n':len(xs)}

def lock():
    R.verify_training()
    assert json.loads((OUT/'ARTIFACT_RECHECK.json').read_text())['all_passed']
    R.write_json(OUT/'PROFILE_LOCK.json',{'sources':{s:R.sha(R.ROOT/s) for s in SOURCES},
      'input_seal':R.sha(R.OUT/'FINAL_AUDIT.json'),'model':R.sha(R.OUT/'nested/stage15.pt'),
      'modes':MODES,'seconds_cap':600,'queries':'first20 bottle/train/good; 2 passes in reverse mode order',
      'parity_queries':'first EVAL normal/category for both arms; no test features or masks'})

def verify():
    d=json.loads((OUT/'PROFILE_LOCK.json').read_text())
    legacy='exps/nar_systems_01/profile.py'
    for p,h in d['sources'].items():
        source=OUT/'PROFILE_RUN_SOURCE.py' if p==legacy else R.ROOT/p
        assert R.sha(source)==h
    if legacy in d['sources']:
        rename=json.loads((OUT/'SOURCE_RENAME.json').read_text())
        assert rename['new_source_sha256']==R.sha(Path(__file__))
        assert rename['profile_lock_sha256']==R.sha(OUT/'PROFILE_LOCK.json')
    assert R.sha(R.OUT/'FINAL_AUDIT.json')==d['input_seal']
    return d

def build(base,archive,mode):
    m,lo,_=R.model_new(base)
    if mode=='oracle': return Detector(m,lo,archive)
    return ResidentDetector(m,lo,archive,mode=='resident_deferred')

@contextmanager
def event_range(name,events):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    with record_function(name):
        a.record(); yield; b.record()
    events.append((name,a,b))

def traced(detector,x,hw,path):
    """Instrument an equal-arithmetic scan; hooks/ranges excluded from latency tests."""
    events=[]; handles=[]; stacks={}
    def before(name):
        def hook(m,args):
            scope=record_function(name); scope.__enter__()
            a=torch.cuda.Event(enable_timing=True); a.record()
            stacks.setdefault(name,[]).append((scope,a))
        return hook
    def after(name):
        def hook(m,args,out):
            scope,a=stacks[name].pop(); b=torch.cuda.Event(enable_timing=True); b.record()
            events.append((name,a,b)); scope.__exit__(None,None,None)
        return hook
    for name,m in detector.model.named_modules():
        tag=None
        if name=='bottleneck.0' or (name.startswith('decoder.') and name.count('.')==1): tag='block:'+name
        elif isinstance(m,K.LoRALinear): tag='lora:'+name
        elif isinstance(m,nn.Linear): tag='base:'+name
        if tag:
            handles.extend([m.register_forward_pre_hook(before(tag)),m.register_forward_hook(after(tag))])
    grids=[]
    try:
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
            for state in detector.archive if not isinstance(detector,ResidentDetector) else detector.resident:
                with event_range('adapter_loading_binding',events):
                    K.set_state(detector.loras,state) if not isinstance(detector,ResidentDetector) else detector.bind(state)
                en,de=detector.model(x)
                with event_range('score_cosine',events):
                    grid=(1-F.cosine_similarity(en[0].float(),de[0].float(),dim=1)).clamp_min(0)
                if not getattr(detector,'deferred_checks',False):
                    with event_range('finite_check_expert',events): assert bool(torch.isfinite(grid).all())
                grids.append(grid)
            with event_range('aggregation',events):
                if getattr(detector,'deferred_checks',False):
                    experts=torch.stack(grids); assert bool(torch.isfinite(experts).all())
                    g=experts.clamp_min(0).amin(0); image=g.flatten(1).amax(1)
                else: g,image=K.task_agnostic_grid(grids)
            with event_range('native_interpolation',events):
                native=F.interpolate(g[:,None],size=hw,mode='bilinear',align_corners=False)[:,0]
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(path))
        grouped={}
        for name,a,b in events: grouped[name]=grouped.get(name,0)+a.elapsed_time(b)
        ops=[]
        for e in prof.key_averages(group_by_input_shape=True):
            ops.append({'name':e.key,'count':e.count,'shapes':str(e.input_shapes),
                'cpu_us':e.cpu_time_total,'self_cpu_us':e.self_cpu_time_total,
                'device_us':getattr(e,'device_time_total',0),'self_device_us':getattr(e,'self_device_time_total',0)})
        return {'module_stream_ms':grouped,'operators':ops,'trace_file':path.name},(native,image)
    finally:
        for h in handles: h.remove()

@torch.no_grad()
def run():
    R.configure(); verify(); R.verify_training()
    assert not (OUT/'PROFILE.json').exists()
    start=time.monotonic()
    def cap():
        assert time.monotonic()-start<600,'profile wall cap'
        assert torch.cuda.max_memory_allocated()<5*2**30,'VRAM cap'
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8.5e9,'RSS cap'
    base=torch.load(R.OUT/'BASE.pt',weights_only=True)
    state=torch.load(R.OUT/'nested/stage15.pt',weights_only=True)
    data=torch.load(R.OUT/'features/bottle_train.pt',weights_only=True,mmap=True)
    rows=data['rows'][:20]; ds=_ImageDataset(R.DATA,rows,224,(.5,)*3,(.5,)*3,training=True)
    encoder=R.make_encoder(); references={}; measurements={m:[] for m in MODES}; parity={m:[] for m in MODES[1:]}
    startup={m:[] for m in MODES}; peaks={m:[] for m in MODES}; traces={}
    for rep,modes in enumerate((MODES,MODES[::-1])):
        for mode in modes:
            torch.cuda.synchronize(); t=time.perf_counter(); det=build(base,state['archive'],mode)
            torch.cuda.synchronize(); startup[mode].append(time.perf_counter()-t)
            for _ in range(3): det.predict(data['features'][:1].cuda(),(900,900))
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            for i in range(20):
                t=time.perf_counter(); image=ds[i]['images'][None]
                with Image.open(R.DATA/rows[i]['relative_path']) as im: hw=(im.height,im.width)
                prep=time.perf_counter()-t
                torch.cuda.synchronize(); t=time.perf_counter(); feature=encoder.extract_patch_features(image.cuda()).float()
                torch.cuda.synchronize(); enc=time.perf_counter()-t
                t=time.perf_counter(); result=det.predict(feature,hw); torch.cuda.synchronize(); seconds=time.perf_counter()-t
                output=tuple(x.cpu().clone() for x in result)
                if mode=='oracle' and rep==0: references[i]=output
                else:
                    exact=all(torch.equal(a,b) for a,b in zip(output,references[i]))
                    difference=max(float((a-b).abs().max()) for a,b in zip(output,references[i]))
                    if mode!='oracle': parity[mode].append({'exact':exact,'max_abs':difference})
                    assert exact,'score parity failed; no tolerance substitution'
                measurements[mode].append({'pre':prep,'encoder':enc,'detector':seconds,'total':prep+enc+seconds})
                cap()
            peaks[mode].append(torch.cuda.max_memory_allocated())
            R.log('TIMED',rep,mode)
            if rep==0 and mode in ('oracle','resident_deferred'):
                feature=data['features'][:1].cuda()
                expected=det.predict(feature,(900,900))
                details,actual=traced(det,feature,(900,900),OUT/f'TRACE_{mode}.json')
                assert all(torch.equal(a,b) for a,b in zip(actual,expected))
                traces[mode]=details
                R.log('TRACED',mode)
            del det,feature,result,output; gc.collect(); torch.cuda.empty_cache()
    del encoder; torch.cuda.empty_cache()
    # Diverse, normal-only parity; no labels/masks or test feature tensors.
    diverse={}
    for arm in ('nested','fixed'):
        state=torch.load(R.OUT/arm/'stage15.pt',weights_only=True)
        oracle=build(base,state['archive'],'oracle')
        ref=[]; probes=[]
        for c in R.ORDER:
            d=torch.load(R.OUT/'features'/f'{c}_train.pt',weights_only=True,mmap=True)
            j=K.normal_partition(d['rows'])['EVAL'][0]; x=d['features'][j:j+1].clone(); probes.append(x)
            ref.append(tuple(y.cpu() for y in oracle.predict(x.cuda(),(224,224))))
        del oracle; gc.collect(); torch.cuda.empty_cache()
        for mode in MODES[1:]:
            det=build(base,state['archive'],mode); differences=[]
            for x,y in zip(probes,ref):
                z=tuple(t.cpu() for t in det.predict(x.cuda(),(224,224)))
                assert all(torch.equal(a,b) for a,b in zip(y,z))
                differences.append(max(float((a-b).abs().max()) for a,b in zip(y,z)))
                cap()
            diverse[arm+':'+mode]={'queries':15,'max_abs':max(differences),'bitwise':True}
            del det; gc.collect(); torch.cuda.empty_cache()
        R.log('DIVERSE_PARITY',arm)
    summarized={m:{k:stats([x[k] for x in measurements[m]]) for k in ('pre','encoder','detector','total')} for m in MODES}
    for mode in MODES:
        summarized[mode].update({'peak_cuda_allocated_bytes':max(peaks[mode]),'startup_seconds':startup[mode],
            'resident_factor_runtime_bytes':0 if mode=='oracle' else 50872320,
            'complete_persistent_tensor_bytes':643197200,'changed_serialized_deployment_bytes':0})
    R.write_json(OUT/'PROFILE.json',{'lock_sha256':R.sha(OUT/'PROFILE_LOCK.json'),'device':torch.cuda.get_device_name(0),
       'modes':summarized,'raw':measurements,'parity':parity,'diverse_parity':diverse,'traces':traces,
       'seconds':time.monotonic()-start,'rss_peak':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
       'limitations':'Same arithmetic; bitwise certified on fixed normal/synthetic workload, not exhaustively on official test. Instrumented event durations include dispatch gaps and are inclusive/nonadditive. 40 timings/mode reuse 20 identities.'})
    R.log('PROFILE_COMPLETE',json.dumps(summarized))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=('lock','run')); a=p.parse_args()
    lock() if a.phase=='lock' else run()
