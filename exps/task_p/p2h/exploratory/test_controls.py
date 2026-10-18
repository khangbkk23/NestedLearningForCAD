"""Engineering-only numerical/causal controls before any evaluation RNG."""
import io
import json
import traceback
import torch
from ..core import *
from ..learner import Learner as Original
from ..data import SyntheticStream as OriginalStream
from ..statistics import paired_benefit,primary,holm
from .data import SyntheticStream,partition
from .learner import Learner
from .design import OUT,POSITIVE,save

SMALL=dict(channels=3,latent=2,hidden=3,side=2)


def run(arm):
    stream=SyntheticStream(1902,'D8',channels=3,side=2,rank=1)
    model=Learner(1902,dimensions=SMALL,arm=arm,dtype=torch.float64)
    original=Original(1902,dimensions=SMALL,dtype=torch.float64) if arm=='candidate' else None
    resumed=None; previous=None; correction_opportunities=0
    for t in range(32):
        event,local=divmod(t,2)
        if local==0: x=stream.arrive(event).double()
        s,q=partition(event,local,4)
        anchors=[] if previous is None else [Observation(event-1,previous,q)]
        for obj in [o for o in [model,original,resumed] if o is not None]:
            if obj.window is None: obj.begin(anchors)
            obj.replay_identity=-1 if previous is None else event-1
            obj.step(x,s,q,previous,event)
        if t==22:
            buffer=io.BytesIO(); torch.save(model.payload(),buffer); buffer.seek(0)
            resumed=Learner(1902,dimensions=SMALL,arm=arm,dtype=torch.float64)
            resumed.restore(torch.load(buffer,weights_only=False))
        if resumed is not None: assert equal_state(model.live,resumed.live),(arm,t,'resume')
        if original is not None:
            assert equal_state(model.live,original.live),('candidate parity',t)
            assert all(torch.equal(getattr(model,m)[n],getattr(original,m)[n]) for m in ['M4','M16'] for n in model.M4)
        if arm=='norm_matched_random':
            for norms in model.trace[-1]['random_norms'].values():
                assert abs(norms['learned']-norms['random'])<=1e-12+1e-6*norms['learned']
                correction_opportunities+=norms['learned']>0
        if local==1: previous=x
    if model.wide:
        assert sum(p.numel() for m in [model.M4,model.M16] for p in m.values())==16*sum(p.numel() for p in model.live.phi.values())
    if arm=='norm_matched_random': assert correction_opportunities>0
    if arm=='extra_replay': assert model.meter['normal_gradient_calls']==32*12
    return dict(resume=True,clock=model.live.clock,meter=dict(model.meter),
                candidate_parity=original is not None,wide_capacity=model.wide,
                random_matched_opportunities=correction_opportunities)


def semantic_tests():
    a=OriginalStream(1901,'D16',channels=3,side=2,rank=1)
    b=SyntheticStream(1901,'D16',channels=3,side=2,rank=1)
    for event in range(64): assert torch.equal(a.arrive(event),b.arrive(event))
    # Reject evaluation-range construction before PRNG, not a sample of an evaluation stream.
    try: SyntheticStream(3101,'D8')
    except (FileNotFoundError,AssertionError): pass
    else: raise AssertionError('unlocked evaluation seed accepted')
    E=[1.]*10; better=[.8]*10
    assert paired_benefit(E,better)['benefit']>0
    assert primary(E,better,E,E)['passed']
    assert not primary(better,E,E,E)['passed']
    assert holm({'a':.001,'b':.03,'c':.2})=={'a':.003,'b':.06,'c':.2}
    # Mask changes direct derivatives, retains forwards, and empty mask is identity.
    torch.manual_seed(1905)
    model=Predictor(**SMALL).double(); entry=State.initial(model); sketch=Sketch(entry)
    x=torch.randn(4,3,dtype=torch.float64); s,q=partition(0,0,4)
    gradients=[]; forwards=[]
    for mask in [0,2]:
        leaf=new_memory(entry,grad=True); fixed=new_memory(entry); state=entry.detached()
        for t in range(4):
            base,d,_=E_operator(model,state,x,s,q,graph=True)
            r,_=correction(state,base,d,fixed,clone_map(leaf) if t<mask else leaf,sketch)
            state=apply_correction(base,r)
        value=objective_loss(model,state.phi,x,q)
        gradients.append(torch.autograd.grad(value,tuple(leaf.values())))
        forwards.append(state.detached())
    assert equal_state(*forwards)
    assert any(not torch.equal(a,b) for a,b in zip(*gradients))
    return dict(generator_bitwise=True,metric_signs=True,mask_forward_parity=True,mask_gradient_effect=True)


def main():
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    results={}; failed=None
    try:
        results['semantics']=semantic_tests()
        for arm in POSITIVE:
            results[arm]=run(arm)
            print(arm,'PASS',flush=True)
    except Exception: failed=traceback.format_exc(); print(failed,flush=True)
    save(OUT/'control_tests.json',dict(passed=failed is None,results=results,failure=failed,G3_untouched=True))
    if failed: raise SystemExit(1)


if __name__=='__main__': main()
