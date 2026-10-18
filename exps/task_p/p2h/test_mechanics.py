"""G1 CPU gate: independent reference parity, numerical derivatives and invariants."""
from __future__ import annotations
import io
import json
import time
from collections import OrderedDict
import numpy as np
import torch
from exps.task_p.p2e.model import Learner2
from .core import *
from .data import Replay,SyntheticStream,WitnessEvaluator,partition
from .learner import Learner
from .statistics import paired_benefit,primary,holm
from .design import OUT,save

SMALL=dict(channels=3,latent=2,hidden=3,side=2)
DETAILS={}


def expect_rejection(call):
    try: call()
    except (AssertionError,RuntimeError): return
    raise AssertionError('invalid access/restore unexpectedly accepted')


def metric_orientation():
    E=np.linspace(1,1.9,10); candidate=E-.2
    result=primary(E,candidate,np.ones(10),np.ones(10))
    assert result['benefit']>0 and result['lower']>0 and result['passed']
    assert not primary(E,E+.2,np.ones(10),np.ones(10))['passed']
    assert not primary(E,candidate,np.ones(10),np.ones(10)*1.02)['passed']
    for contrast in ['C1','C2','C3','C4']:
        assert paired_benefit(E,candidate)['benefit']>0,contrast
        assert paired_benefit(candidate,E)['upper']<0,contrast
    assert paired_benefit(E,E)['benefit']==0
    assert holm({'a':.01,'b':.04})=={'a':.02,'b':.04}
    DETAILS['known_value_benefit']=result


def exact_E_and_zero_memory():
    reference=Learner2('E',1901,device='cpu')
    state=State.initial(reference.model)
    zero=new_memory(state); sketch=Sketch(state)
    rng=torch.Generator().manual_seed(1902)
    past=[]; current=None
    for cycle in range(16):
        event,local=divmod(cycle,2)
        if local==0: current=torch.randn(784,768,generator=rng)*.1
        s,q=partition(event,local,784)
        replay=None if not past else past[event%len(past)]
        before=state
        state,d,record=E_operator(reference.model,state,current,s,q,replay)
        reference_record=reference.train_cycle(current,s,q,replay)
        assert record['support_loss']==reference_record['support_loss']
        assert record['query_loss']==reference_record['query_loss']
        assert state.clock==reference.cycles
        for name,p in state.phi.items():
            assert p.detach().numpy().tobytes()==reference.fast[name].numpy().tobytes(),(cycle,name,'fast')
            for a,b in zip(state.moments[name],reference.moments[name]):
                assert a.detach().numpy().tobytes()==b.numpy().tobytes(),(cycle,name,'moment')
        r,_=correction(before,state,d,zero,zero,sketch)
        assert equal_state(apply_correction(state,r),state)
        if local==1: past.append(current.clone())
    DETAILS['E_parity_cycles']=16
    DETAILS['E_parameter_count']=sum(p.numel() for p in state.phi.values())


def bounds_guard_versions():
    torch.manual_seed(1903)
    model=Predictor(**SMALL).double(); previous=State.initial(model)
    x=torch.randn(4,3,dtype=torch.float64)*.2; s,q=partition(0,0,4)
    base,d,_=E_operator(model,previous,x,s,q)
    M=new_memory(base); sketch=Sketch(base)
    for p in M.values(): p.normal_()
    r,norms=correction(previous,base,d,M,M,sketch)
    for n in r: assert norms[n]['correction']<=.25*norms[n]['base']+1e-15
    zero,_=correction(base,base,d,M,M,sketch)
    assert all(torch.count_nonzero(v)==0 for v in zero.values())
    value=objective_loss(model,base.phi,x,q)
    g=torch.autograd.grad(value,tuple(base.phi.values()))
    bad=OrderedDict((n,torch.zeros_like(p)) for n,p in base.phi.items())
    bad['decode.bias']=100*g[-1]
    accepted,lam,record=guard(model,base,bad,[Observation(0,x,q)],1e-12,1e-12)
    assert lam==0 and accepted is base and all(v>0 for v in record['full_delta'])
    copied=clone_map(M); old={n:p.clone() for n,p in M.items()}
    written=write_memory(M,[torch.ones_like(p) for p in M.values()])
    assert float(map_norm(written))<=1+1e-12
    for n in M:
        assert torch.equal(M[n],old[n]) and torch.equal(copied[n],old[n])
        assert M[n].data_ptr()!=written[n].data_ptr()
    DETAILS['guard_fallback_lambda']=lam


def finite_difference_hypergradients():
    torch.manual_seed(1904)
    model=Predictor(**SMALL).double(); entry=State.initial(model)
    sketch=Sketch(entry); rng=torch.Generator().manual_seed(1905)
    tape=[torch.randn(4,3,generator=rng,dtype=torch.float64)*.2 for _ in range(16)]
    fixed=[new_memory(entry) for _ in range(4)]
    nominal=new_memory(entry)
    for memory in fixed+[nominal]:
        for value in memory.values(): value.normal_(std=.03)
    anchors=[Observation(0,tape[0],torch.tensor([1,3]))]
    with torch.no_grad(): c=[objective_loss(model,entry.phi,anchors[0].features,anchors[0].query).detach()]
    rows=[]
    for length in [2,4,16]:
        # Detached entries and fixed M4 versions define the exact truncated partial.
        def evaluate(memory):
            state=entry.detached(); ref=entry.detached(); recent=[]; terms=[]
            for t in range(length):
                s,q=partition(t//2,t%2,4)
                ref,_,_=E_operator(model,ref,tape[t],s,q,None,False)
                base,d,_=E_operator(model,state,tape[t],s,q,None,True)
                m4,m16=(memory,fixed[0]) if length<=4 else (fixed[t//4],memory)
                r,_=correction(state,base,d,m4,m16,sketch)
                state=apply_correction(base,r)
                recent.append(Observation(t,tape[t],q))
                if (length==16 and t+1 in [4,8,16]) or (length<=4 and t+1==length):
                    terms.append(retention_objective(model,state,ref,anchors,c,recent)['total'])
            return sum(terms)/len(terms)
        memory=clone_map(nominal,True)
        value=evaluate(memory)
        gradients=torch.autograd.grad(value,tuple(memory.values()))
        norm=float(torch.sqrt(sum(g.square().sum() for g in gradients)))
        for block in range(6):
            perturb=[torch.zeros_like(p) for p in memory.values()]
            perturb[block].normal_(generator=rng)
            size=torch.sqrt(sum(p.square().sum() for p in perturb))
            perturb=[p/size for p in perturb]
            analytical=float(sum((g*v).sum() for g,v in zip(gradients,perturb)))
            eps=1e-5
            plus=OrderedDict((n,(p.detach()+eps*v).requires_grad_()) for (n,p),v in zip(memory.items(),perturb))
            minus=OrderedDict((n,(p.detach()-eps*v).requires_grad_()) for (n,p),v in zip(memory.items(),perturb))
            numeric=float((evaluate(plus)-evaluate(minus))/(2*eps))
            error=abs(numeric-analytical)
            relative=error/max(abs(numeric),abs(analytical),1e-12)
            assert error<=1e-6 if norm<=1e-8 else relative<=1e-3,(length,block,analytical,numeric,relative)
            rows.append(dict(cycles=length,block=block,analytical=analytical,numerical=numeric,relative_error=relative,absolute_error=error))
    DETAILS['finite_differences']=rows


def cross_cycle_break():
    # Independent scalar delayed-credit example: the first write influences only a later loss.
    m=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    first=torch.tensor(.4,dtype=torch.float64)+m
    final=(first-.1*first.square()).square()
    intact=float(torch.autograd.grad(final,m)[0])
    severed=first.detach().requires_grad_()
    broken_final=(severed-.1*severed.square()).square()+0*m
    broken=float(torch.autograd.grad(broken_final,m)[0])
    assert intact!=0 and broken==0
    DETAILS['graph_break']=dict(intact=intact,broken=broken)


def pass_parity_and_resume():
    learner=Learner(1906,dimensions=SMALL,dtype=torch.float64)
    rng=torch.Generator().manual_seed(1907)
    images=[torch.randn(4,3,generator=rng,dtype=torch.float64)*.2 for _ in range(8)]
    anchor=Observation(0,torch.randn(4,3,generator=rng,dtype=torch.float64)*.2,torch.tensor([1,3]))
    learner.begin([anchor]); restored=None
    for t in range(16):
        s,q=partition(10+t//2,t%2,4)
        learner.step(images[t//2],s,q,None,10+t//2)
        if restored is not None: restored.step(images[t//2],s,q,None,10+t//2)
        if t==5:
            buffer=io.BytesIO(); torch.save(learner.payload(),buffer); buffer.seek(0)
            data=torch.load(buffer,weights_only=False)
            restored=Learner(1908,dimensions=SMALL,dtype=torch.float64)
            restored.restore(data)
            assert equal_state(learner.live,restored.live)
            invalid=Learner(1908,dimensions=SMALL,dtype=torch.float64,eps_rel=.1)
            expect_rejection(lambda:invalid.restore(data))
    assert learner.window is None and restored.window is None
    assert equal_state(learner.live,restored.live)
    for level in ['M4','M16']:
        for n,p in getattr(learner,level).items(): assert torch.equal(p,getattr(restored,level)[n])
    assert learner.trace[-1]['forward_parity'] and restored.trace[-1]['forward_parity']
    DETAILS['mid_window_resume_cycle']=6


def causal_access_and_rng():
    rng=torch.Generator().manual_seed(1909)
    replay=Replay(1909,3); anchor_rng=torch.Generator().manual_seed(12909)
    for event in range(12):
        x=torch.randn(4,3,generator=rng)
        for cycle in range(2):
            row=replay.sample(event)
            assert row is None or row.identity<event
        replay.insert(Observation(event,x,torch.tensor([1,3])))
        assert all(a.identity<=event for a in replay.anchors(anchor_rng))
    expect_rejection(lambda:replay.sample(0))
    data=replay.state(); second=Replay.restore(data)
    assert replay.sample(12).identity==second.sample(12).identity
    stream=SyntheticStream(1901,'D16',channels=6,side=2,rank=2)
    expect_rejection(lambda:stream.arrive(1))
    evaluator=WitnessEvaluator(1901,stream)
    expect_rejection(lambda:evaluator.admit('B',77))
    assert stream.distribution(77)=='A' and stream.distribution(78)=='B'
    for event in range(40): stream.arrive(event)
    restored=SyntheticStream.restore(stream.state())
    assert torch.equal(stream.arrive(40),restored.arrive(40))
    expect_rejection(lambda:SyntheticStream(3101,'D8'))
    DETAILS['G3_seed_use']=0


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started=time.perf_counter(); rows=[]
    tests=[metric_orientation,exact_E_and_zero_memory,bounds_guard_versions,
           finite_difference_hypergradients,cross_cycle_break,pass_parity_and_resume,causal_access_and_rng]
    for test in tests:
        start=time.perf_counter()
        try:
            test(); status='PASS'; error=None
        except Exception as exc:
            import traceback
            status='FAIL'; error=traceback.format_exc()
        row=dict(test=test.__name__,status=status,seconds=time.perf_counter()-start,error=error)
        rows.append(row); print(json.dumps(row),flush=True)
        if status=='FAIL': break
        if time.perf_counter()-started>600:
            rows.append(dict(test='G1_timeout',status='FAIL',error='600 seconds exceeded')); break
    passed=len(rows)==len(tests) and all(r['status']=='PASS' for r in rows)
    save(OUT/'G1.json',dict(passed=passed,tests=rows,details=DETAILS,seconds=time.perf_counter()-started,
                          G3_untouched=True,device='cpu',verdict=None if passed else 'IMPLEMENTATION_INVALID'))
    if not passed: raise SystemExit(1)


if __name__=='__main__': main()
