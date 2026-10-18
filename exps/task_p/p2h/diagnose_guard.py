"""Passive G1b engineering instrumentation of guard vetoes and shadow divergence."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
import time
import torch
from .core import Observation,objective_loss,equal_state
from .data import SyntheticStream,Replay,partition
from .learner import Learner
from .preflight import configure,device_row
from .design import OUT,save


class ObservedLearner(Learner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.observations=[]

    @torch.no_grad()
    def observe(self):
        w=self.window
        assert w is not None
        losses={name:[float(objective_loss(self.model,state.phi,a.features,a.query))
                      for a in w['anchors']]
                for name,state in [('live',self.live),('trial',w['trial']),
                                   ('reference',w['reference']),('counter',w['counter'])]}
        difference=torch.sqrt(sum((self.live.phi[n]-w['trial'].phi[n]).square().sum() for n in self.live.phi))
        scale=torch.sqrt(sum(p.square().sum() for p in self.live.phi.values()))
        rows=self.trace[-4:]
        veto_cycles=sum(r['lambda_value']<1 for r in rows)
        counter_mean=sum(losses['counter'])/max(len(losses['counter']),1)
        live_mean=sum(losses['live'])/max(len(losses['live']),1)
        self.observations.append(dict(clock=self.live.clock,anchor_count=len(w['anchors']),losses=losses,
                 parameter_divergence=float(difference),relative_divergence=float(difference/scale.clamp_min(1e-12)),
                 veto_cycles=veto_cycles,rejected_but_better_at_milestone=veto_cycles if counter_mean<live_mean else 0,
                 counter_trial_forward_parity=equal_state(w['counter'],w['trial'])))

    def long_pass(self,zero=False):
        result=super().long_pass(zero)
        if not zero: self.observe() # before window state is released; no optimizer mutation
        return result


def main():
    configure()
    assert json.loads((OUT/'G1b.json').read_text())['passed']
    eps=json.loads((OUT/'guard_calibration.json').read_text())
    learner=ObservedLearner(1901,device='cuda',eps_abs=eps['eps_abs'],eps_rel=eps['eps_rel'])
    stream=SyntheticStream(1901,'D8'); replay=Replay(1901)
    rng=torch.Generator().manual_seed(12901)
    start=time.perf_counter()
    for t in range(80):
        event,local=divmod(t,2)
        if local==0: x_cpu=stream.arrive(event); x=x_cpu.to('cuda')
        s,q=partition(event,local,784,'cuda')
        old=replay.sample(event); learner.replay_identity=-1 if old is None else old.identity
        if learner.window is None: learner.begin([device_row(row) for row in replay.anchors(rng)])
        learner.step(x,s,q,None if old is None else old.features.to('cuda'),event)
        if learner.window is not None and len(learner.window['tape'])%4==0: learner.observe()
        if local==1: replay.insert(Observation(event,x_cpu,q.cpu()))
    torch.cuda.synchronize()
    veto=sum(r['veto_cycles'] for r in learner.observations)
    rejected_better=sum(r['rejected_but_better_at_milestone'] for r in learner.observations)
    result=dict(seed=1901,regime='D8',cycles=80,scope='passive engineering diagnostic; not another G3 or MDE sample',
                passed=all(r['counter_trial_forward_parity'] for r in learner.observations),
                observations=learner.observations,veto_rate=None if veto==0 else rejected_better/veto,
                veto_cycles=veto,maximum_relative_divergence=max(r['relative_divergence'] for r in learner.observations),
                meter=dict(learner.meter),seconds=time.perf_counter()-start,G3_untouched=True)
    # Original seed1901 candidate final checkpoint is unchanged. This observer never trains on witnesses.
    save(OUT/'G1b_guard_diagnostic.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='observations'}),flush=True)
    if not result['passed']: raise SystemExit(1)


if __name__=='__main__': main()
