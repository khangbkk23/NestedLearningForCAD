"""Additional CPU adversarial checks on the actual E/optimizer-memory path."""
import io
import json
from collections import OrderedDict
import torch
from .core import *
from .learner import Learner
from .data import SyntheticStream,Replay,partition
from .design import OUT,save

SMALL=dict(channels=3,latent=2,hidden=3,side=2)


def actual_delayed_path():
    torch.manual_seed(1908)
    model=Predictor(**SMALL).double(); entry=State.initial(model)
    sketch=Sketch(entry); fixed=new_memory(entry)
    x=torch.randn(4,3,dtype=torch.float64)*.2
    target=torch.randn(4,3,dtype=torch.float64)*.2
    norms=[]
    for broken in [False,True]:
        leaf=new_memory(entry,grad=True); state=entry.detached()
        for t in range(4):
            s,q=partition(t//2,t%2,4)
            base,d,_=E_operator(model,state,x,s,q,graph=True)
            r,_=correction(state,base,d,fixed,leaf if t==0 else fixed,sketch)
            state=apply_correction(base,r)
            if broken and t==0: state=state.detached()
        value=objective_loss(model,state.phi,target,torch.tensor([1,3]))
        value=value+sum(p.sum()*0 for p in leaf.values())
        g=torch.autograd.grad(value,tuple(leaf.values()))
        norms.append(float(torch.sqrt(sum(p.square().sum() for p in g))))
    assert norms[0]>1e-8 and norms[1]==0,norms
    return dict(intact_first_read_gradient=norms[0],broken_first_read_gradient=norms[1])


def write_after_read():
    torch.manual_seed(1909)
    model=Predictor(**SMALL).double(); entry=State.initial(model)
    sketch=Sketch(entry); memory=new_memory(entry); fixed=new_memory(entry)
    x=torch.randn(4,3,dtype=torch.float64)*.2; s,q=partition(0,0,4)
    for p in memory.values(): p.normal_(std=.01)
    snapshots=clone_map(memory); versions={n:p._version for n,p in memory.items()}
    base,d,_=E_operator(model,entry,x,s,q,graph=True)
    r,_=correction(entry,base,d,memory,fixed,sketch)
    before=apply_correction(base,r).detached()
    written=write_memory(memory,[torch.ones_like(p)*.1 for p in memory.values()])
    base2,d2,_=E_operator(model,entry,x,s,q,graph=True)
    r2,_=correction(entry,base2,d2,memory,fixed,sketch)
    after=apply_correction(base2,r2).detached()
    assert equal_state(before,after)
    assert all(torch.equal(memory[n],snapshots[n]) and memory[n]._version==versions[n]
               and memory[n].data_ptr()!=written[n].data_ptr() for n in memory)
    return dict(bitwise_forward_after_write=True,old_versions_unmodified=True)


def harness_resume():
    def fresh():
        return dict(learner=Learner(1906,dimensions=SMALL),
                    stream=SyntheticStream(1906,'D8',channels=3,side=2,rank=1),
                    replay=Replay(1906),anchor=torch.Generator().manual_seed(12906),x=None)
    def step(h,t):
        event,local=divmod(t,2)
        if local==0: h['x']=h['stream'].arrive(event)
        learner=h['learner']; replay=h['replay']; x=h['x']
        s,q=partition(event,local,4)
        old=replay.sample(event)
        learner.replay_identity=-1 if old is None else old.identity
        if learner.window is None: learner.begin(replay.anchors(h['anchor']))
        learner.step(x,s,q,None if old is None else old.features,event)
        if local==1: replay.insert(Observation(event,x,q))
    original=fresh(); resumed=None
    for t in range(32):
        step(original,t)
        if resumed is not None:
            step(resumed,t)
            a,b=original['learner'],resumed['learner']
            assert equal_state(a.live,b.live)
            if a.window is not None:
                for name in ['trial','reference','counter']:
                    assert equal_state(a.window[name],b.window[name]),name
                for x,y in zip(a.window['milestones'],b.window['milestones']):
                    assert equal_state(x,y)
                for x,y in zip(a.window['versions'],b.window['versions']):
                    assert all(torch.equal(x[n],y[n]) for n in x)
        if t==22: # partial chunk/window and first cycle of an uninserted image
            assert original['learner'].live.clock==23
            saved=dict(learner=original['learner'].payload(),stream=original['stream'].state(),
                       replay=original['replay'].state(),anchor=original['anchor'].get_state(),x=original['x'])
            buffer=io.BytesIO(); torch.save(saved,buffer); buffer.seek(0)
            data=torch.load(buffer,weights_only=False)
            resumed=fresh(); resumed['learner'].restore(data['learner'])
            resumed.update(stream=SyntheticStream.restore(data['stream']),
                           replay=Replay.restore(data['replay']),x=data['x'])
            resumed['anchor'].set_state(data['anchor'])
    a,b=original['learner'],resumed['learner']
    assert equal_state(a.live,b.live)
    assert all(torch.equal(getattr(a,m)[n],getattr(b,m)[n]) for m in ['M4','M16'] for n in a.M4)
    assert original['replay'].sample(16).identity==resumed['replay'].sample(16).identity
    assert original['replay'].seen==resumed['replay'].seen==16
    assert all('witness' not in key.lower() for key in a.payload())
    return dict(resume_completed_cycle=23,bitwise_final=True,causal_replay_rng=True,
                current_image_preserved=True,witness_in_learner=False)


def main():
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    results={}
    for test in [actual_delayed_path,write_after_read,harness_resume]:
        try:
            results[test.__name__]=dict(passed=True,details=test())
        except Exception:
            import traceback
            results[test.__name__]=dict(passed=False,error=traceback.format_exc()); break
    passed=len(results)==3 and all(r['passed'] for r in results.values())
    save(OUT/'G1_contract_checks.json',dict(passed=passed,tests=results,G3_untouched=True))
    print(json.dumps(results),flush=True)
    if not passed: raise SystemExit(1)


if __name__=='__main__': main()
