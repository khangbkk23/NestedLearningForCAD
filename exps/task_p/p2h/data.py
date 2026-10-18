"""Causal synthetic normal data; no paths or APIs for real data."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from .core import Observation
from .design import REGIMES


def partition(event,local_cycle,patches,device='cpu'):
    assert local_cycle in (0,1)
    parity=(event+local_cycle)%2
    return (torch.arange(parity,patches,2,device=device),
            torch.arange(1-parity,patches,2,device=device))


def orthogonal(matrix):
    q,r=torch.linalg.qr(matrix,mode='reduced')
    signs=torch.sign(torch.diag(r)); signs[signs==0]=1
    return q*signs


class SyntheticStream:
    def __init__(self,seed,regime,channels=768,side=28,rank=8):
        assert 1901<=seed<=1910 or 3101<=seed<=3110
        if 3101<=seed<=3110:
            raise RuntimeError('G3 generation requires separate locked evaluator, not an engineering API')
        self.seed,self.regime,self.channels,self.side=seed,regime,channels,side
        self.rank=rank; self.next_event=0
        self.params_rng=torch.Generator().manual_seed(seed+1000)
        self.images_rng=torch.Generator().manual_seed(seed+2000)
        self.muA=torch.randn(channels,generator=self.params_rng)/math.sqrt(channels)
        self.VA=orthogonal(torch.randn(channels,rank,generator=self.params_rng))
        a=torch.randint(1,5,(rank,),generator=self.params_rng)
        b=torch.randint(1,5,(rank,),generator=self.params_rng)
        phi=2*math.pi*torch.rand(rank,generator=self.params_rng)
        u,v=torch.meshgrid(torch.linspace(-1,1,side),torch.linspace(-1,1,side),indexing='ij')
        self.waves=torch.cos(math.pi*(u.flatten()[:,None]*a+v.flatten()[:,None]*b)+phi)
        self.muB=self.VB=None

    def distribution(self,cycle):
        L=REGIMES[self.regime][0]
        if self.regime=='C' or cycle<64+L or cycle>=128:
            return 'A'
        return 'B'

    def _B(self):
        if self.muB is not None: return
        mu=torch.randn(self.channels,generator=self.params_rng)/math.sqrt(self.channels)
        self.muB=mu-self.muA*torch.dot(mu,self.muA)/self.muA.square().sum()
        v=torch.randn(self.channels,self.rank,generator=self.params_rng)
        self.VB=orthogonal(v-self.VA@(self.VA.T@v))

    def sample(self,distribution,rng):
        if distribution=='B': self._B()
        mu,V=(self.muA,self.VA) if distribution=='A' else (self.muB,self.VB)
        z=torch.randn(self.rank,generator=rng)
        noise=torch.randn(self.side*self.side,self.channels,generator=rng)
        return mu+(self.waves*z)@V.T+.1*noise

    def arrive(self,event):
        assert event==self.next_event,'future/repeated arrival'
        self.next_event+=1
        return self.sample(self.distribution(2*event),self.images_rng)

    def state(self):
        return dict(self.__dict__,params_rng=self.params_rng.get_state(),images_rng=self.images_rng.get_state())

    @classmethod
    def restore(cls,data):
        obj=cls.__new__(cls); obj.__dict__.update(data)
        obj.params_rng=torch.Generator().set_state(data['params_rng'])
        obj.images_rng=torch.Generator().set_state(data['images_rng'])
        return obj


class Replay:
    def __init__(self,seed,capacity=24):
        self.capacity,self.seen=capacity,0
        self.rng=torch.Generator().manual_seed(seed+9000)
        self.rows=[]

    def sample(self,current_identity):
        assert all(row.identity<current_identity for row in self.rows),'self/future replay'
        if not self.rows: return None
        return self.rows[int(torch.randint(len(self.rows),(1,),generator=self.rng))]

    def insert(self,row):
        assert row.identity==self.seen,'noncausal insertion'
        self.seen+=1
        copy=Observation(row.identity,row.features.detach().clone(),row.query.detach().clone())
        if len(self.rows)<self.capacity:
            self.rows.append(copy)
        else:
            i=int(torch.randint(self.seen,(1,),generator=self.rng))
            if i<self.capacity: self.rows[i]=copy

    def anchors(self,rng):
        rows=sorted(self.rows,key=lambda row:row.identity)
        if len(rows)<=4: return list(rows)
        chosen=[]
        for group in range(4):
            lo=group*len(rows)//4; hi=(group+1)*len(rows)//4
            i=int(torch.randint(lo,hi,(1,),generator=rng))
            chosen.append(rows[i])
        return chosen

    def state(self):
        return dict(capacity=self.capacity,seen=self.seen,rows=self.rows,rng=self.rng.get_state())

    @classmethod
    def restore(cls,data):
        obj=cls.__new__(cls)
        obj.capacity,obj.seen,obj.rows=data['capacity'],data['seen'],data['rows']
        obj.rng=torch.Generator().set_state(data['rng'])
        return obj


class WitnessEvaluator:
    """Evaluator object deliberately never belongs to Learner or its checkpoint."""
    def __init__(self,seed,stream):
        self.rng=torch.Generator().manual_seed(seed+30000)
        self.stream=stream; self.samples={}

    def admit(self,phase,completed_cycles):
        earliest=0 if phase=='A' else (64 if self.stream.regime=='C' else 64+REGIMES[self.stream.regime][0])
        assert completed_cycles>=earliest,'future-phase witnesses'
        if phase in self.samples: return
        distribution='A' if phase=='A' or self.stream.regime=='C' else 'B'
        self.samples[phase]=[(self.stream.sample(distribution,self.rng),
                              torch.arange(1,self.stream.side*self.stream.side,2)) for _ in range(16)]

    @torch.no_grad()
    def evaluate(self,phase,model,state):
        from .core import objective_loss
        device=next(iter(state.phi.values())).device
        return sum(float(objective_loss(model,state.phi,x.to(device),q.to(device)))
                   for x,q in self.samples[phase])/len(self.samples[phase])
