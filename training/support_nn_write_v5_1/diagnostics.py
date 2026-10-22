"""Synchronized, excluded pre/post-credit captures; exact recomputation on disposable learners."""
import time
from contextlib import contextmanager
import torch
from models.hope_cad_v5_1.learner import NestedUtilityLearner,fast_step,prediction
from training.support_throughput_v5_1.replay import synchronize,state_digest,rng_fingerprint,assert_equal

class Segments:
    def __init__(self,device,clock=time.perf_counter):
        self.device,self.clock=device,clock;self.pipeline_seconds=0.;self.excluded_seconds=0.;self.started=None;self.exclusions=0
    def start(self):synchronize(self.device);self.started=self.clock()
    def exclude(self,fn):
        synchronize(self.device);t=self.clock();self.pipeline_seconds+=t-self.started
        try:return fn()
        finally:
            synchronize(self.device);end=self.clock();self.excluded_seconds+=end-t;self.exclusions+=1;self.started=end
    def finish(self):
        synchronize(self.device);end=self.clock();self.pipeline_seconds+=end-self.started;self.started=None

class Captures:
    def __init__(self,segments,events=(181,189)):
        self.segments=segments;self.events=set(events);self.cursor=None;self.values={}
    @contextmanager
    def installed(self,learner):
        original=learner.observe;had='observe' in vars(learner);previous=vars(learner).get('observe')
        def wrapped(record):
            current=self.cursor
            if current in self.events:
                def before():
                    if current in self.values:raise ValueError('Duplicate diagnostic capture')
                    self.values[current]=dict(cursor=current,pre=learner.state_dict(),record=learner._cpu(record))
                self.segments.exclude(before)
            result=original(record)
            if current in self.events:self.segments.exclude(lambda:self.values[current].update(post=learner.state_dict()))
            return result
        learner.observe=wrapped
        try:yield self
        finally:
            if had:learner.observe=previous
            else:del learner.observe

def recompute(capture,factory,device):
    """No image arrival. Each capture contains the factual completed causal record."""
    rng=rng_fingerprint(device);actor=factory();learner=actor.learner
    learner.load_state_dict(capture['pre'])
    record=dict(capture['record'])
    for k in ['tokens','stats']:record[k]=record[k].to(device)
    updated,loss,denom,actual_grads=fast_step(learner.model,learner.phi,record,exact=False)
    actual=dict(loss=loss,denominator=denom,gradients=dict(zip(learner.phi,actual_grads)),
        increments={k:(v.double()/denom).detach() for k,v in zip(learner.phi,actual_grads)},
        updated_phi=updated)
    records=[*learner.records,record]
    if len(records)!=8 or learner.block_initial is None:raise ValueError('Capture must precede scheduled slow commit')
    phi={k:v.detach().clone().requires_grad_(True) for k,v in learner.block_initial.items()}
    support=[]
    for r in records[:4]:
        previous=phi;phi,l,d,g=fast_step(learner.model,phi,r,exact=True)
        support.append(dict(loss=l,denominator=d,gradients=dict(zip(previous,g)),
            increments={k:v.double()/d for k,v in zip(previous,g)},updated_phi=phi))
    query_losses=torch.stack([.5*(prediction(learner.model,phi,r)[0]-r['target']).square() for r in records[4:]])
    outer=query_losses.mean();params=learner.model.slow_parameters()
    grads=torch.autograd.grad(outer,[p for _,p in params])
    details=learner.observe(record)
    actual_post=learner.state_dict();assert_equal(actual_post,capture['post'],'diagnostic_post_learning')
    out=NestedUtilityLearner._cpu(dict(actual=actual,support=support,query_losses=query_losses,
        meta_loss=outer,raw_slow_gradients=dict(zip([n for n,_ in params],grads)),
        post_learning=actual_post,learning_details=details))
    if rng_fingerprint(device)!=rng:raise ValueError('Diagnostic recomputation changed global RNG')
    return out
