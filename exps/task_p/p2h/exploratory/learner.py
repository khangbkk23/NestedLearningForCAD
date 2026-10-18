"""Complete causal controls; imports preserved exact-E primitives only."""
from collections import OrderedDict
import math
import torch
from ..core import *
from ..learner import Learner as Original
from ..design import ARMS


def write_single(memory,gradients):
    limit=math.sqrt(2)
    norm=torch.sqrt(sum(g.square().sum() for g in gradients))
    scale=torch.clamp(limit/(norm+1e-12),max=1.)
    proposed=OrderedDict((n,p-.01*scale*g) for (n,p),g in zip(memory.items(),gradients))
    clipping=torch.clamp(limit/(map_norm(proposed)+1e-12),max=1.)
    return OrderedDict((n,(p*clipping).detach().clone()) for n,p in proposed.items())


class Learner(Original):
    def __init__(self,*args,arm='candidate',**kwargs):
        assert arm in ARMS and arm!='relaxed_cap'
        super().__init__(*args,arm='candidate',**kwargs)
        self.arm=arm
        self.wide=arm in ['single_memory_16','M4_only_16','M16_only_16']
        if self.wide:
            self.sketch=Sketch(self.live,16)
            self.M4=new_memory(self.live,16) if arm!='M16_only_16' else OrderedDict()
            self.M16=new_memory(self.live,16) if arm=='M16_only_16' else OrderedDict()
        if arm in ['E','immediate_protection','extra_replay']:
            self.M4,self.M16=OrderedDict(),OrderedDict()
        self.mask_before=0
        self.last_versions=None

    def constants(self):
        c=super().constants(); c.update(extension='P2h-X',columns=16 if self.wide else 8)
        return c

    def begin(self,anchors):
        assert self.window is None and self.live.clock%16==0
        assert all(a.identity>=0 for a in anchors)
        entry=self.live.detached()
        with torch.no_grad():
            entries=[objective_loss(self.model,entry.phi,a.features,a.query,self.meter).detach() for a in anchors]
        if self.arm=='reset_memory': self.M4,self.M16=new_memory(entry),new_memory(entry)
        self.window=dict(entry=entry,anchors=anchors,entries=entries,trial=entry.detached(),
                         reference=entry.detached(),counter=entry.detached(),versions=[],versions16=[],
                         milestones=[],tape=[],recent=[])
        self.short_memory=clone_map(self.M16 if self.arm=='M16_only_16' else self.M4,True)
        self.short16=clone_map(self.M16,True) if self.arm=='collapsed_clock' else self.M16

    def _propose(self,previous,base,direction,M4=None,M16=None):
        a=self.M4 if M4 is None else M4
        b=self.M16 if M16 is None else M16
        if self.wide:
            active=b if self.arm=='M16_only_16' else a
            inactive=OrderedDict((n,torch.zeros_like(p)) for n,p in active.items())
            if self.arm!='M16_only_16' and b: inactive=b
            a,b=active,inactive
        return correction(previous,base,direction,a,b,self.sketch,.25)

    def _advance(self,state,row,M4,M16,graph):
        x,s,q,replay,identity=row
        base,d,_=E_operator(self.model,state,x,s,q,replay,graph,self.meter)
        r,_=self._propose(state,base,d,M4,M16)
        after=apply_correction(base,r)
        return after if graph else after.detached()

    def _ordinary(self,old,base,row):
        x,s,q,replay,identity=row
        if self.arm=='immediate_protection':
            anchors=self.window['anchors']
            if not anchors: return OrderedDict((n,torch.zeros_like(p)) for n,p in base.phi.items())
            value=sum(objective_loss(self.model,base.phi,a.features,a.query,self.meter) for a in anchors)/len(anchors)
            gs=torch.autograd.grad(value,tuple(base.phi.values()))
            self.meter['normal_gradient_calls']+=1
            r=OrderedDict()
            for (n,p),g in zip(base.phi.items(),gs):
                b=(p-old.phi[n]).detach(); g=g.detach()
                v=-torch.relu((g*b).sum())*g/(g.square().sum()+1e-12)
                r[n]=v*torch.clamp(.25*b.norm()/(v.norm()+1e-30),max=1.)
            return r
        extra=base.detached()
        for _ in range(5): extra,_,_=E_operator(self.model,extra,x,s,q,replay,False,self.meter)
        return OrderedDict((n,(extra.phi[n]-p).detach()) for n,p in base.phi.items())

    def step(self,x,support,query,replay,identity):
        w=self.window; assert w is not None
        h=len(w['tape'])+1; row=(x,support,query,replay,identity)
        assert isinstance(identity,int) and identity>=0
        w['tape'].append(row); w['recent'].append(Observation(identity,x,query))
        old=self.live
        base,d,record=E_operator(self.model,old,x,support,query,replay,False,self.meter)
        if self.arm in ['E','immediate_protection','extra_replay']:
            r=OrderedDict((n,torch.zeros_like(p)) for n,p in base.phi.items())
            norms={n:dict(base=float((p-old.phi[n]).detach().norm()),correction=0.) for n,p in base.phi.items()}
        else: r,norms=self._propose(old,base,d)
        if self.arm in ['immediate_protection','extra_replay']:
            r=self._ordinary(old,base,row)
            norms={n:dict(base=float((base.phi[n]-old.phi[n]).norm()),correction=float(p.norm())) for n,p in r.items()}
        if self.arm=='norm_matched_random':
            r=OrderedDict((n,(lambda g,p:g*(p.detach().norm()/(g.norm()+1e-30)))(
                torch.randn(p.shape,generator=self.random,dtype=p.dtype).to(p.device),p)) for n,p in r.items())
            record['random_norms']={n:dict(learned=norms[n]['correction'],random=float(p.norm())) for n,p in r.items()}
        checks=[Observation(identity,x,query)]
        if replay is not None: checks.append(Observation(self.replay_identity,replay,query))
        checks.extend(w['anchors'])
        if self.arm in ['E','zero_memory','deployed_ablation']:
            accepted,lam,gr=base,0.,dict(full_delta=[],accepted_delta=[])
        elif self.arm=='disabled_guard':
            accepted,lam,gr=apply_correction(base,r),1.,dict(full_delta=[],accepted_delta=[])
        else: accepted,lam,gr=guard(self.model,base,r,checks,self.eps_abs,self.eps_rel,self.meter)
        self.live=accepted.detached()
        record.update(clock=self.live.clock,lambda_value=lam,block_norms=norms,guard=gr,
                      memory4_norm=float(map_norm(self.M4)) if self.M4 else 0.,
                      memory16_norm=float(map_norm(self.M16)) if self.M16 else 0.)
        self.trace.append(record)
        if self.arm in ['E','deployed_ablation','immediate_protection','extra_replay']:
            if h==16: self.window=None
            return record
        w['reference'],_,_=E_operator(self.model,w['reference'],x,support,query,replay,False,self.meter)
        w['counter']=self._advance(w['counter'],row,self.M4,self.M16,False)
        if self.arm=='short_credit': w['trial']=w['trial'].detached()
        short4=self.short_memory if self.arm!='M16_only_16' else self.M4
        short16=self.short_memory if self.arm=='M16_only_16' else self.short16
        w['trial']=self._advance(w['trial'],row,short4,short16,True)
        if h%4: return record
        terms=retention_objective(self.model,w['trial'],w['reference'],w['anchors'],w['entries'],w['recent'],self.meter)
        if self.arm=='collapsed_clock':
            joint=OrderedDict([('4/'+n,p) for n,p in self.short_memory.items()]+[('16/'+n,p) for n,p in self.short16.items()])
            gs=gradient(terms['total'],joint,self.meter)
            n=len(self.short_memory); g4,g16short=gs[:n],gs[n:]
            proposed16=write_memory(self.short16,g16short)
        else: g4=gradient(terms['total'],self.short_memory,self.meter)
        cap=math.sqrt(2) if self.arm=='single_memory_16' else 1.
        proposed4=write_single(self.short_memory,g4) if self.arm=='single_memory_16' else write_memory(self.short_memory,g4,cap)
        w['versions'].append(clone_map(self.short_memory))
        w['versions16'].append(clone_map(self.short16))
        w['milestones'].append(w['trial'].detached())
        record['short_objective']={k:float(v.detach()) if torch.is_tensor(v) else v for k,v in terms.items()}
        record['short_write_norm']=float(torch.sqrt(sum((proposed4[n]-self.short_memory[n]).square().sum() for n in proposed4)))
        if h==16:
            glong,long_record=self.long_pass(); record.update(long_record)
            if self.arm=='single_memory_16':
                proposed4=write_single(self.short_memory,[a+b for a,b in zip(g4,glong)])
            elif self.arm=='M16_only_16': proposed16=write_memory(self.M16,glong)
            elif self.arm not in ['collapsed_clock','M4_only_16']: proposed16=write_memory(self.M16,glong)
            self.last_versions=dict(M4=w['versions'],M16=w['versions16'])
        if self.arm not in ['zero_memory','M16_only_16']: self.M4=proposed4
        if self.arm=='collapsed_clock' or (h==16 and self.arm not in ['zero_memory','single_memory_16','M4_only_16']):
            self.M16=proposed16
        if h==16: self.window=None
        else:
            self.short_memory=clone_map(self.M16 if self.arm=='M16_only_16' else self.M4,True)
            self.short16=clone_map(self.M16,True) if self.arm=='collapsed_clock' else self.M16
            w['trial']=w['trial'].detached()
        return record

    def long_pass(self):
        w=self.window
        additive=self.arm in ['single_memory_16','M4_only_16','collapsed_clock']
        columns=16 if self.wide else 8
        memory=new_memory(w['entry'],columns,True) if additive else clone_map(self.M16,True)
        state=w['entry'].detached(); reference=w['entry'].detached(); recent=[]; objectives=[]
        for t,row in enumerate(w['tape']):
            x,s,q,replay,identity=row
            reference,_,_=E_operator(self.model,reference,x,s,q,replay,False,self.meter)
            if self.arm=='short_credit': state=state.detached()
            if self.arm=='M16_only_16': a,b=self.M4,memory
            elif self.arm=='collapsed_clock':
                a=w['versions'][t//4]
                b=OrderedDict((n,p+memory[n]) for n,p in w['versions16'][t//4].items())
            else:
                a=w['versions'][t//4]; b=memory
            if self.arm=='post_onset_only' and t<self.mask_before: b=clone_map(b)
            state=self._advance(state,row,a,b,True)
            recent.append(Observation(identity,x,q))
            if (t+1)%4==0:
                assert equal_state(state.detached(),w['milestones'][t//4]),('PassA/B mismatch',self.arm,t)
                terms=retention_objective(self.model,state,reference,w['anchors'],w['entries'],recent,self.meter)
                if t+1 in [4,8,16]: objectives.append(terms['total'])
        g=gradient(sum(objectives)/3,memory,self.meter)
        return g,dict(forward_parity=True,long_gradient_norm=float(torch.sqrt(sum(p.square().sum() for p in g))))

    def payload(self):
        result=super().payload()
        result.update(mask_before=self.mask_before)
        return result

    def restore(self,data):
        assert data['constants']==self.constants()
        self.model.load_state_dict(data['model'])
        device=next(self.model.parameters()).device
        self.live=State.restore(data['live'],device)
        self.M4=OrderedDict((n,p.to(device).detach().clone()) for n,p in data['M4'].items())
        self.M16=OrderedDict((n,p.to(device).detach().clone()) for n,p in data['M16'].items())
        self.mask_before=data['mask_before']
        self.window=data['window']
        if self.window is not None:
            w=dict(self.window)
            for k in ['entry','reference','counter','trial']: w[k]=State.restore(w[k],device)
            w['milestones']=[State.restore(s,device) for s in w['milestones']]
            self.window=w
            if self.arm not in ['E','deployed_ablation','immediate_protection','extra_replay']:
                self.short_memory=clone_map(self.M16 if self.arm=='M16_only_16' else self.M4,True)
                self.short16=clone_map(self.M16,True) if self.arm=='collapsed_clock' else self.M16
                h=len(w['tape']); start=(h//4)*4
                current=w['entry'].detached() if start==0 else w['milestones'][start//4-1].detached()
                for row in w['tape'][start:]:
                    if self.arm=='short_credit': current=current.detached()
                    a=self.M4 if self.arm=='M16_only_16' else self.short_memory
                    b=self.short_memory if self.arm=='M16_only_16' else self.short16
                    current=self._advance(current,row,a,b,True)
                assert equal_state(current,w['trial']),('resume mismatch',self.arm)
                w['trial']=current
        self.trace=data['trace']; self.decompositions=data['decompositions']
        self.first_short=data['first_short']; self.first_long=data['first_long']
        self.random.set_state(data['random']); self.meter.update(data['meter'])
