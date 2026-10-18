"""Causal live path, four short graphs and reconstructed sixteen-cycle graph."""
from __future__ import annotations
from collections import OrderedDict
import torch
from .core import (Predictor,State,Meter,Sketch,Observation,new_memory,clone_map,map_norm,
                   E_operator,correction,apply_correction,guard,objective_loss,
                   retention_objective,gradient,write_memory,decompose,equal_state)


class Learner:
    def __init__(self,seed=1901,device='cpu',dimensions=None,arm='candidate',
                 eps_abs=1e-7,eps_rel=1e-5,diagnostics=False,dtype=torch.float32):
        implemented={'E','zero_memory','candidate','short_credit','reset_memory',
                     'disabled_guard','norm_matched_random','relaxed_cap'}
        if arm not in implemented:
            raise NotImplementedError(f'P2h arm not implemented/validated: {arm}; no silent candidate alias')
        self.dimensions=dimensions or dict(channels=768,latent=64,hidden=128,side=28)
        torch.manual_seed(seed)
        self.model=Predictor(**self.dimensions).to(device=device,dtype=dtype)
        self.live=State.initial(self.model)
        self.M4=new_memory(self.live); self.M16=new_memory(self.live)
        self.sketch=Sketch(self.live)
        self.arm=arm; self.rho=1. if arm=='relaxed_cap' else .25
        self.cap=10. if arm=='relaxed_cap' else 1.
        self.eps_abs,self.eps_rel=eps_abs,eps_rel
        self.diagnostics=diagnostics; self.first_short=False; self.first_long=False
        self.meter=Meter(); self.window=None
        self.trace=[]; self.decompositions=[]
        self.random=torch.Generator().manual_seed(seed+13000)

    def constants(self):
        return dict(dimensions=self.dimensions,arm=self.arm,rho=self.rho,cap=self.cap,
                    eps_abs=self.eps_abs,eps_rel=self.eps_rel,
                    sketch='sha256/p2h/sketch/v1',short=4,long=16)

    def begin(self,anchors):
        assert self.window is None and self.live.clock%16==0
        assert all(a.identity>=0 for a in anchors),'witness anchor'
        entry=self.live.detached()
        with torch.no_grad():
            entries=[objective_loss(self.model,entry.phi,a.features,a.query,self.meter).detach() for a in anchors]
        if self.arm=='reset_memory':
            self.M4=new_memory(entry); self.M16=new_memory(entry)
        self.window=dict(entry=entry,anchors=anchors,entries=entries,
                         trial=entry.detached(),reference=entry.detached(),
                         counter=entry.detached(),versions=[],milestones=[],tape=[],recent=[])
        self.short_memory=clone_map(self.M4,True)

    def _propose(self,previous,base,direction,M4=None,M16=None):
        return correction(previous,base,direction,self.M4 if M4 is None else M4,
                          self.M16 if M16 is None else M16,self.sketch,self.rho)

    def _advance(self,state,row,M4,M16,graph):
        x,s,q,replay,identity=row
        base,direction,_=E_operator(self.model,state,x,s,q,replay,graph,self.meter)
        r,_=self._propose(state,base,direction,M4,M16)
        after=apply_correction(base,r)
        return after if graph else after.detached()

    def step(self,x,support,query,replay,identity):
        assert self.window is not None,'window anchors must be supplied causally'
        w=self.window; h=len(w['tape'])+1
        assert isinstance(identity,int) and identity>=0
        row=(x,support,query,replay,identity)
        w['tape'].append(row); w['recent'].append(Observation(identity,x,query))
        old=self.live
        base,direction,record=E_operator(self.model,old,x,support,query,replay,False,self.meter)
        r,norms=self._propose(old,base,direction)
        if self.arm=='norm_matched_random':
            learned=r
            r=OrderedDict()
            for n,p in learned.items():
                g=torch.randn(p.shape,generator=self.random,dtype=p.dtype).to(p.device)
                r[n]=g*(p.detach().norm()/(g.norm()+1e-30))
        checks=[Observation(identity,x,query)]
        # Replay query uses the current query, exactly as E. Its identity is supplied by harness.
        if replay is not None:
            checks.append(Observation(self.replay_identity,replay,query))
        checks.extend(w['anchors'])
        if self.arm in ['E','zero_memory']:
            accepted,lam,guard_record=base,0.,dict(full_delta=[],accepted_delta=[])
        elif self.arm=='disabled_guard':
            accepted,lam,guard_record=apply_correction(base,r),1.,dict(full_delta=[],accepted_delta=[])
        else:
            accepted,lam,guard_record=guard(self.model,base,r,checks,self.eps_abs,self.eps_rel,self.meter)
        self.live=accepted.detached()
        record.update(clock=self.live.clock,lambda_value=lam,block_norms=norms,guard=guard_record,
                      memory4_norm=float(map_norm(self.M4)),memory16_norm=float(map_norm(self.M16)))
        self.trace.append(record)
        if self.arm=='E':
            if h==16: self.window=None
            return record
        w['reference'],_,_=E_operator(self.model,w['reference'],x,support,query,replay,False,self.meter)
        w['counter']=self._advance(w['counter'],row,self.M4,self.M16,False)
        if self.arm=='short_credit': w['trial']=w['trial'].detached()
        w['trial']=self._advance(w['trial'],row,self.short_memory,self.M16,True)
        if h%4:
            return record
        terms=retention_objective(self.model,w['trial'],w['reference'],w['anchors'],w['entries'],w['recent'],self.meter)
        if self.diagnostics and not self.first_short and terms['active']>0:
            d=decompose(terms,self.short_memory,self.meter)
            d.update(kind='actual_short',clock=self.live.clock,milestone=h)
            self.decompositions.append(d); self.first_short=True
            self.zero_short_diagnostic(h)
        g4=gradient(terms['total'],self.short_memory,self.meter)
        proposed4=write_memory(self.short_memory,g4,self.cap)
        w['versions'].append(clone_map(self.short_memory))
        w['milestones'].append(w['trial'].detached())
        record['short_objective']={k:float(v.detach()) if torch.is_tensor(v) else v for k,v in terms.items()}
        record['short_write_norm']=float(torch.sqrt(sum((proposed4[n]-self.M4[n]).square().sum() for n in self.M4)))
        if h==16:
            g16,long_record=self.long_pass()
            proposed16=write_memory(self.M16,g16,self.cap)
            record.update(long_record)
            record['long_write_norm']=float(torch.sqrt(sum((proposed16[n]-self.M16[n]).square().sum() for n in self.M16)))
            if self.arm!='zero_memory': self.M4,self.M16=proposed4,proposed16
            self.window=None
        else:
            if self.arm!='zero_memory': self.M4=proposed4
            self.short_memory=clone_map(self.M4,True)
            w['trial']=w['trial'].detached()
        return record

    def long_pass(self,zero=False):
        w=self.window
        memory=new_memory(w['entry'],grad=True) if zero else clone_map(self.M16,True)
        versions=[new_memory(w['entry']) for _ in range(4)] if zero else w['versions']
        state=w['entry'].detached(); reference=w['entry'].detached()
        objectives=[]; recent=[]; states=[]
        for t,row in enumerate(w['tape']):
            x,s,q,replay,identity=row
            reference,_,_=E_operator(self.model,reference,x,s,q,replay,False,self.meter)
            if self.arm=='short_credit': state=state.detached()
            state=self._advance(state,row,versions[t//4],memory,True)
            recent.append(Observation(identity,x,q))
            if (t+1)%4==0:
                states.append(state.detached())
                if not zero:
                    assert equal_state(states[-1],w['milestones'][t//4]),'Pass A/B forward mismatch'
                terms=retention_objective(self.model,state,reference,w['anchors'],w['entries'],recent,self.meter)
                if t+1 in [4,8,16]: objectives.append(terms)
        total={key:sum(t[key] for t in objectives)/3 for key in ['total','acquisition','retention','harm']}
        total['active']=max(t['active'] for t in objectives)
        first_actual=self.diagnostics and not zero and not self.first_long and total['active']>0
        if self.diagnostics and ((zero and True) or first_actual):
            d=decompose(total,memory,self.meter)
            d.update(kind='zero_counterfactual_long' if zero else 'actual_long',clock=self.live.clock)
            self.decompositions.append(d)
            if not zero: self.first_long=True
        g=gradient(total['total'],memory,self.meter)
        if first_actual:
            self.long_pass(zero=True)
        return g,{'long_objective':{k:float(v.detach()) if torch.is_tensor(v) else v for k,v in total.items()},
                  'forward_parity':True}

    def zero_short_diagnostic(self,h):
        w=self.window; state=w['entry'].detached(); reference=w['entry'].detached()
        zero=new_memory(state); leaf=new_memory(state,grad=True); recent=[]
        start=h-4
        for t,row in enumerate(w['tape'][:h]):
            x,s,q,replay,identity=row
            reference,_,_=E_operator(self.model,reference,x,s,q,replay,False,self.meter)
            if t<start:
                state=reference.detached()
            else:
                state=self._advance(state,row,leaf,zero,True)
            recent.append(Observation(identity,x,q))
        terms=retention_objective(self.model,state,reference,w['anchors'],w['entries'],recent,self.meter)
        d=decompose(terms,leaf,self.meter)
        d.update(kind='zero_counterfactual_short',clock=self.live.clock,milestone=h)
        self.decompositions.append(d)

    def payload(self):
        # No evaluator object, phase identity or witness enters this object.
        def state(s): return s.payload()
        window=None
        if self.window is not None:
            w=self.window
            window={k:v for k,v in w.items() if k not in ['entry','trial','reference','counter','milestones']}
            window.update({k:state(w[k]) for k in ['entry','trial','reference','counter']})
            window['milestones']=[state(s) for s in w['milestones']]
        return dict(constants=self.constants(),model=self.model.state_dict(),live=state(self.live),
                    M4=clone_map(self.M4),M16=clone_map(self.M16),window=window,
                    trace=self.trace,decompositions=self.decompositions,
                    first_short=self.first_short,first_long=self.first_long,
                    random=self.random.get_state(),meter=dict(self.meter))

    def restore(self,data):
        assert data['constants']==self.constants(),'restore attempted to override frozen constants'
        self.model.load_state_dict(data['model'])
        device=next(self.model.parameters()).device
        self.live=State.restore(data['live'],device)
        self.M4=clone_map(OrderedDict((n,p.to(device)) for n,p in data['M4'].items()))
        self.M16=clone_map(OrderedDict((n,p.to(device)) for n,p in data['M16'].items()))
        self.window=data['window']
        if self.window is not None:
            w=dict(self.window)
            for k in ['entry','reference','counter','trial']: w[k]=State.restore(w[k],device)
            w['milestones']=[State.restore(s,device) for s in w['milestones']]
            self.window=w
            self.short_memory=clone_map(self.M4,True)
            # Recreate only the current chunk's graph from its detached entry and arrived tape.
            h=len(w['tape']); start=(h//4)*4
            current=w['entry'].detached() if start==0 else w['milestones'][start//4-1].detached()
            for row in w['tape'][start:]:
                current=self._advance(current,row,self.short_memory,self.M16,True)
            assert equal_state(current,w['trial']),'resume reconstruction mismatch'
            w['trial']=current
        self.trace=data['trace']; self.decompositions=data['decompositions']
        self.first_short=data['first_short']; self.first_long=data['first_long']
        self.random.set_state(data['random']); self.meter.update(data['meter'])
