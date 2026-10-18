"""Differentiable exact-E recurrence, immutable optimizer memories and guard."""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict, Counter
from dataclasses import dataclass

import torch
from exps.task_p.p2.model import Predictor, bounded_gradient, loss, parameters


@dataclass
class State:
    phi: OrderedDict
    moments: dict
    clock: int = 0

    @classmethod
    def initial(cls, model):
        phi=OrderedDict((n,p.detach().clone().requires_grad_(True)) for n,p in parameters(model).items())
        return cls(phi,{n:(torch.zeros_like(p),torch.zeros_like(p)) for n,p in phi.items()})

    def detached(self):
        return State(OrderedDict((n,p.detach().clone().requires_grad_(True)) for n,p in self.phi.items()),
                     {n:tuple(v.detach().clone() for v in pair) for n,pair in self.moments.items()},self.clock)

    def payload(self):
        return {'phi':{n:p.detach().cpu() for n,p in self.phi.items()},
                'moments':{n:tuple(p.detach().cpu() for p in pair) for n,pair in self.moments.items()},
                'clock':self.clock}

    @classmethod
    def restore(cls, data, device='cpu'):
        return cls(OrderedDict((n,p.to(device).clone().requires_grad_(True)) for n,p in data['phi'].items()),
                   {n:tuple(p.to(device).clone() for p in pair) for n,pair in data['moments'].items()},data['clock'])


def equal_state(a,b):
    return a.clock==b.clock and all(torch.equal(a.phi[n],b.phi[n]) and
        all(torch.equal(x,y) for x,y in zip(a.moments[n],b.moments[n])) for n in a.phi)


class Meter(Counter):
    def adjoint(self, gradient):
        self['inner_gradient_adjoint_sites']+=1
        return gradient


def objective_loss(model, phi, x, indices, meter=None):
    if meter is not None:
        batch=1 if x.ndim==2 else len(x)
        meter['loss_forwards']+=1
        meter['image_forward_accesses']+=batch
        meter['input_patches']+=batch*x.shape[-2]
        meter['queried_patches']+=batch*len(indices)
    return loss(model,phi,x,indices)


def E_operator(model, state, x, support, query, replay=None, graph=False, meter=None):
    """Same arithmetic order/clips as pinned P2e, moments connected only in shadows."""
    phi=state.phi
    moments=state.moments
    record={}
    direction2=None
    for j in [1,2]:
        data=x if j==1 or replay is None else torch.stack((x,replay))
        indices=support if j==1 else query
        value=objective_loss(model,phi,data,indices,meter)
        gradients=torch.autograd.grad(value,tuple(phi.values()),create_graph=graph)
        if meter is not None:
            meter['normal_gradient_calls']+=1
            if graph and gradients[0].requires_grad:
                gradients[0].register_hook(meter.adjoint)
        gradients,norm=bounded_gradient(gradients)
        updated=OrderedDict(); after={}; directions={}
        step=2*state.clock+j
        for (name,p),g in zip(phi.items(),gradients):
            first=.9*moments[name][0]+.1*g
            second=.999*moments[name][1]+.001*g.square()
            d=(first/(1-.9**step))/((second/(1-.999**step)+1e-12).sqrt()+1e-8)
            delta=p.new_full((),.001)*d
            delta=delta*torch.clamp(.02/(delta.square().mean().sqrt()+1e-12),max=1.)
            updated[name]=p-delta
            after[name]=(first,second)
            directions[name]=d
        phi,moments=updated,after
        record['support_loss' if j==1 else 'query_loss']=float(value.detach())
        record['support_gradient_norm' if j==1 else 'query_gradient_norm']=float(norm.detach())
        direction2=directions
    result=State(phi,moments,state.clock+1)
    if not graph:
        result=result.detached()
        direction2={n:d.detach() for n,d in direction2.items()}
    return result,direction2,record


def clone_map(memory, grad=False):
    return OrderedDict((n,p.detach().clone().requires_grad_(grad)) for n,p in memory.items())


def new_memory(state, columns=8, grad=False):
    return OrderedDict((n,p.new_zeros(p.numel(),columns).requires_grad_(grad)) for n,p in state.phi.items())


def map_norm(memory):
    return torch.sqrt(sum(p.square().sum() for p in memory.values()))


def write_memory(memory, gradients, cap=1., rate=.01):
    norm=torch.sqrt(sum(g.square().sum() for g in gradients))
    scale=torch.clamp(1./(norm+1e-12),max=1.)
    proposed=OrderedDict((n,p-rate*scale*g) for (n,p),g in zip(memory.items(),gradients))
    clipping=torch.clamp(cap/(map_norm(proposed)+1e-12),max=1.)
    return OrderedDict((n,(p*clipping).detach().clone()) for n,p in proposed.items())


class Sketch:
    def __init__(self,state,columns=8):
        self.columns=columns; self.index={}
        for name,p in state.phi.items():
            seed=int.from_bytes(hashlib.sha256(b'p2h/sketch/'+name.encode()).digest()[:8],'little')
            groups=[([],[]) for _ in range(columns-1)]
            for i in range(p.numel()):
                h=(seed+(i+1)*2654435761)%(1<<64)
                bucket=h%(columns-1)
                groups[bucket][0].append(i)
                groups[bucket][1].append(1. if ((h>>32)&1)==0 else -1.)
            self.index[name]=[(torch.tensor(ids,device=p.device,dtype=torch.long),
                               torch.tensor(signs,device=p.device,dtype=p.dtype)) for ids,signs in groups]

    def key(self,name,direction):
        flat=direction.flatten()
        z=torch.stack([(flat[index]*sign).sum()/math.sqrt(flat.numel()) for index,sign in self.index[name]])
        return torch.cat((z.new_ones(1),z))/torch.sqrt(1+z.square().sum())


def correction(previous,base,direction,M4,M16,sketch,rho=.25):
    result=OrderedDict(); norms={}
    for name,p in base.phi.items():
        b=p-previous.phi[name]
        amplitude=torch.linalg.vector_norm(b)
        key=sketch.key(name,direction[name])
        u=(M4[name]+M16[name])@key
        r=torch.zeros_like(p) if float(amplitude.detach())==0 else (
            rho*amplitude*u/torch.sqrt(1+u.square().sum())).reshape_as(p)
        result[name]=r
        norms[name]={'base':float(amplitude.detach()),'correction':float(r.detach().norm()),
                     'u':float(u.detach().norm())}
    return result,norms


def apply_correction(base,r,scale=1.):
    if scale==0:
        return base
    return State(OrderedDict((n,p+scale*r[n]) for n,p in base.phi.items()),base.moments,base.clock)


def gradient(value,memory,meter=None,retain_graph=False):
    if meter is not None: meter['meta_gradient_calls']+=1
    return torch.autograd.grad(value,tuple(memory.values()),retain_graph=retain_graph)


@dataclass
class Observation:
    identity: int
    features: torch.Tensor
    query: torch.Tensor


def guard(model,base,r,checks,eps_abs=1e-7,eps_rel=1e-5,meter=None):
    unique=OrderedDict()
    for row in checks:
        assert isinstance(row.identity,int) and row.identity>=0, 'witness/non-training identity'
        unique.setdefault(row.identity,row)
    with torch.no_grad():
        references=[objective_loss(model,base.phi,j.features,j.query,meter) for j in unique.values()]
        if not all(bool(torch.isfinite(v)) for v in references): raise RuntimeError('nonfinite reference')
        full_delta=None
        for lam in [1.,.5,.25]:
            trial=apply_correction(base,r,lam)
            losses=[objective_loss(model,trial.phi,j.features,j.query,meter) for j in unique.values()]
            if not all(bool(torch.isfinite(v)) for v in losses): raise RuntimeError('nonfinite proposal')
            changes=[float(a-b) for a,b in zip(losses,references)]
            if full_delta is None: full_delta=changes
            if all(bool(a<=b+eps_abs+eps_rel*b.abs()) for a,b in zip(losses,references)):
                return trial,lam,{'full_delta':full_delta,'accepted_delta':changes}
    return base,0.,{'full_delta':full_delta,'accepted_delta':[0.]*len(references)}


def retention_objective(model,trial,reference,anchors,entries,recent,meter=None):
    acquisition=[]
    for row in recent[-4:]:
        a=objective_loss(model,trial.phi,row.features,row.query,meter)
        with torch.no_grad():
            b=objective_loss(model,reference.phi,row.features,row.query,meter).clamp_min(1e-6)
        acquisition.append(a/(b+1e-8))
    A=torch.stack(acquisition).mean()
    excess=[]
    for row,c in zip(anchors,entries):
        if float(c)<=1e-8: continue
        value=objective_loss(model,trial.phi,row.features,row.query,meter)
        with torch.no_grad():
            ref=objective_loss(model,reference.phi,row.features,row.query,meter)
            threshold=torch.minimum(c,ref)
        excess.append(torch.relu((value-threshold)/(c.clamp_min(1e-6)+1e-8)))
    if excess:
        e=torch.stack(excess)
        retention=e.mean()+torch.max(e,dim=0).values
        active=float(e.detach().max())
    else:
        retention=A*0
        active=0.
    harm=torch.relu(A-1).square()
    return {'total':A+retention+harm,'acquisition':A,'retention':retention,'harm':harm,'active':active}


def decompose(terms,memory,meter=None):
    parts={k:gradient(terms[k],memory,meter,retain_graph=True) for k in ['acquisition','retention','harm']}
    norms={k:float(torch.sqrt(sum(v.detach().square().sum() for v in g))) for k,g in parts.items()}
    dot=sum((a.detach()*b.detach()).sum() for a,b in zip(parts['acquisition'],parts['retention']))
    norms.update(ratio=norms['retention']/max(norms['acquisition'],1e-30),
                 cosine=float(dot)/max(norms['retention']*norms['acquisition'],1e-30),
                 active=terms['active'],memory_norm=float(map_norm(memory).detach()))
    return norms
