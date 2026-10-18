"""G1b engineering-only signal/power probe, on the authorized CUDA device."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call
from .core import E_operator,State,Observation,objective_loss,equal_state
from .learner import Learner
from .data import SyntheticStream,Replay,WitnessEvaluator,partition
from .statistics import mde,primary
from .design import ROOT,OUT,sha,save


def tensor_file(path,value):
    with Path(path).open('xb') as stream: torch.save(value,stream)


def configure():
    assert json.loads((OUT/'G1.json').read_text())['passed'],'G1 has not passed'
    lock=json.loads((OUT/'design_lock.json').read_text())
    assert sha(OUT/'protocol_design.json')==lock['protocol_sha256']
    config=json.loads((OUT/'protocol_design.json').read_text())
    assert sha(ROOT/'exps/task_p/p2h/PROTOCOL.md')==config['mathematics_protocol_sha256']
    for name,digest in config['canonical_source_sha256'].items(): assert sha(ROOT/name)==digest
    assert not (OUT/'execution_lock.json').exists(),'G1b is engineering-only before EL'
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available(),'CUDA access unavailable'


def device_row(row):
    return Observation(row.identity,row.features.to('cuda'),row.query.to('cuda'))


def calibrate():
    stream=SyntheticStream(1901,'D8'); replay=Replay(1901)
    reference=Learner(1901,device='cuda',arm='E')
    selected=set(np.linspace(0,127,100,dtype=int).tolist())
    rows=[]; states=[]; x=None
    for t in range(128):
        event,local=divmod(t,2)
        if local==0: x_cpu=stream.arrive(event); x=x_cpu.to('cuda')
        s,q=partition(event,local,784,'cuda')
        old=replay.sample(event)
        old_x=None if old is None else old.features.to('cuda')
        reference.live,_,_=E_operator(reference.model,reference.live,x,s,q,old_x,False,reference.meter)
        if t in selected:
            with torch.no_grad():
                first=objective_loss(reference.model,reference.live.phi,x,q)
                second=objective_loss(reference.model,reference.live.phi,x,q)
                prediction=functional_call(reference.model,reference.live.phi,(x,))
                double=(prediction[q].double()-x[q].double()).square().mean()
            a,b,c=float(first),float(second),float(double)
            absolute=max(abs(a-b),abs(a-c)); relative=absolute/max(abs(c),1e-30)
            rows.append(dict(clock=t+1,loss32=a,repeat32=b,loss64_reduction=c,
                             absolute=absolute,relative=relative))
            states.append(dict(state=reference.live.payload(),x=x_cpu.clone(),query=q.cpu(),clock=t+1))
        if local==1: replay.insert(Observation(event,x_cpu,q.cpu()))
    p_abs=float(np.quantile([r['absolute'] for r in rows],.95))
    p_rel=float(np.quantile([r['relative'] for r in rows],.95))
    assert len(rows)==100
    tensor_file(OUT/'calibration_states.pt',states)
    result=dict(states=100,seed=1901,p95_abs=p_abs,p95_rel=p_rel,
                eps_abs=max(10*p_abs,1e-7),eps_rel=max(10*p_rel,1e-5),
                passed=p_rel<=1e-4,rows=rows,
                state_file_sha256=sha(OUT/'calibration_states.pt'))
    save(OUT/'guard_calibration.json',result)
    del states,reference
    torch.cuda.empty_cache()
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    return result


def run_stream(seed,regime,eps,arm='candidate'):
    assert seed in range(1901,1911),'no G3 seed before EL'
    stream=SyntheticStream(seed,regime); replay=Replay(seed)
    anchor_rng=torch.Generator().manual_seed(seed+11000)
    evaluator=WitnessEvaluator(seed,stream); evaluator.admit('A',0)
    candidate=Learner(seed,device='cuda',arm=arm,diagnostics=True,
                      eps_abs=eps['eps_abs'],eps_rel=eps['eps_rel'])
    E=candidate.live.detached()
    endpoint=json.loads((OUT/'protocol_design.json').read_text())['regimes_LPMW_endpoint'][regime][-1]
    onset=64+json.loads((OUT/'protocol_design.json').read_text())['regimes_LPMW_endpoint'][regime][0]
    acquisition={}; retained={}; end_new={}; effective=[]; identities=[]
    started=time.perf_counter(); x=None
    first_failure=None; vetoes=[]
    for t in range(128):
        event,local=divmod(t,2)
        if t==onset: evaluator.admit('B',t)
        if local==0: x_cpu=stream.arrive(event); x=x_cpu.to('cuda')
        s,q=partition(event,local,784,'cuda')
        old=replay.sample(event); old_x=None if old is None else old.features.to('cuda')
        candidate.replay_identity=-1 if old is None else old.identity
        identities.append(dict(clock=t+1,current=event,replay=None if old is None else old.identity))
        if candidate.window is None:
            candidate.begin([device_row(row) for row in replay.anchors(anchor_rng)])
        # Independent local E reference detects corrections lost to FP32 rounding.
        local_base,_,_=E_operator(candidate.model,candidate.live,x,s,q,old_x,False)
        E,_,_=E_operator(candidate.model,E,x,s,q,old_x,False)
        record=candidate.step(x,s,q,old_x,event)
        change=float(torch.sqrt(sum((candidate.live.phi[n]-local_base.phi[n]).square().sum() for n in local_base.phi)))
        effective.append(dict(clock=t+1,accepted_displacement=change,lambda_value=record['lambda_value']))
        if local==1: replay.insert(Observation(event,x_cpu,q.cpu()))
        if t+1==64:
            acquisition=dict(candidate=evaluator.evaluate('A',candidate.model,candidate.live),
                             E=evaluator.evaluate('A',candidate.model,E))
        if t+1==endpoint:
            retained=dict(candidate=evaluator.evaluate('A',candidate.model,candidate.live),
                          E=evaluator.evaluate('A',candidate.model,E))
        if t+1==128:
            if 'B' not in evaluator.samples: evaluator.admit('B',t+1)
            end_new=dict(candidate=evaluator.evaluate('B',candidate.model,candidate.live),
                         E=evaluator.evaluate('B',candidate.model,E))
        for d in candidate.decompositions:
            if d['kind'] in ['actual_short','actual_long'] and d['active']>0 and d['ratio']<.05:
                first_failure=dict(reason='actual_active_retention_ratio_below_0.05',diagnostic=d)
                break
        if first_failure is not None: break
        if t%16==15:
            print(json.dumps(dict(seed=seed,regime=regime,arm=arm,clock=t+1,
                  memory4=float(sum(v.square().sum() for v in candidate.M4.values()).sqrt()),
                  memory16=float(sum(v.square().sum() for v in candidate.M16.values()).sqrt()),
                  elapsed=time.perf_counter()-started)),flush=True)
    torch.cuda.synchronize()
    directory=OUT/'engineering'/f'{regime}_{seed}_{arm}'
    directory.mkdir(parents=True,exist_ok=False)
    trace=dict(seed=seed,regime=regime,arm=arm,completed_cycles=candidate.live.clock,
               traces=candidate.trace,decompositions=candidate.decompositions,
               effective_corrections=effective,causal_replay=identities,meter=dict(candidate.meter),
               acquisition=acquisition,retained=retained,new_normal=end_new,
               first_failure=first_failure,seconds=time.perf_counter()-started,
               cpu_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               cuda_peak_allocated=torch.cuda.max_memory_allocated(),
               cuda_peak_reserved=torch.cuda.max_memory_reserved(),G3_untouched=True)
    ratios=[b['correction']/max(b['base'],1e-30) for row in candidate.trace[16:] for b in row['block_norms'].values()]
    trace['median_correction_base_ratio']=None if not ratios else float(np.median(ratios))
    trace['effective_after16']=sum(row['accepted_displacement']>0 for row in effective[16:])
    if not first_failure:
        if not candidate.first_short or not candidate.first_long:
            trace['first_failure']=dict(reason='no_actual_active_retention_headroom_for_both_contracts')
        elif not ratios or np.median(ratios)<=1e-6 or trace['effective_after16']==0:
            trace['first_failure']=dict(reason='negligible_or_undeployed_correction')
    if acquisition and retained:
        trace['excess']={n:max(retained[n]-acquisition[n],0.) for n in ['E','candidate']}
        trace['raw_excess']={n:retained[n]-acquisition[n] for n in ['E','candidate']}
    save(directory/'trace.json',trace)
    tensor_file(directory/'state.pt',candidate.payload())
    save(directory/'artifact_hashes.json',{'trace':sha(directory/'trace.json'),'state':sha(directory/'state.pt')})
    del candidate,E,stream,replay,evaluator
    torch.cuda.empty_cache()
    return trace


def main():
    configure(); started=time.perf_counter()
    assert not (OUT/'G1b_started.json').exists(),'preflight already started; preserve the one authorized attempt'
    save(OUT/'G1b_started.json',dict(utc=time.time(),engineering_seeds=list(range(1901,1911)),G3_untouched=True))
    eps=calibrate(); outcomes=[]; failure=None
    if not eps['passed']: failure=dict(reason='guard_calibration_noise')
    for regime in ['D8','D16']:
        if failure: break
        for seed in range(1901,1911):
            row=run_stream(seed,regime,eps); outcomes.append(row)
            if row['first_failure']:
                failure=dict(regime=regime,seed=seed,**row['first_failure']); break
            if time.perf_counter()-started>1800:
                failure=dict(reason='bounded_G1b_timeout'); break
        if failure: break
    power={}
    if not failure:
        for regime in ['D8','D16']:
            selected=[r for r in outcomes if r['regime']==regime]
            power[regime]=mde([r['excess']['E'] for r in selected],[r['excess']['candidate'] for r in selected])
            if not power[regime]['passed']:
                failure=dict(regime=regime,reason='MDE_exceeds_5_percent_or_no_headroom',power=power[regime]); break
    # The cap oracle is diagnostic only; it can never rescue a failed G1b.
    oracle=[]
    if not failure:
        for seed in range(1901,1911):
            row=run_stream(seed,'D8',eps,'relaxed_cap'); oracle.append(row)
            if row['first_failure']: failure=dict(reason='relaxed_cap_diagnostic_failed',seed=seed); break
    save(OUT/'G1b.json',dict(passed=failure is None,first_failure=failure,power=power,
         engineering_runs=[dict(regime=r['regime'],seed=r['seed'],arm=r['arm'],
                               completed_cycles=r['completed_cycles'],first_failure=r['first_failure']) for r in outcomes+oracle],
         seconds=time.perf_counter()-started,G3_untouched=True,
         verdict=None if failure is None else 'INCONCLUSIVE_UNDERPOWERED'))
    print('G1B',json.dumps(dict(passed=failure is None,first_failure=failure,power=power)),flush=True)
    if failure: raise SystemExit(2)


if __name__=='__main__': main()
