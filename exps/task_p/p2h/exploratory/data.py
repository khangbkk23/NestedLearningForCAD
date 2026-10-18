"""Identical generator; evaluation range opens only after valid new lock."""
import math
import torch
from ..data import SyntheticStream as Base,orthogonal,Replay,WitnessEvaluator,partition
from .design import validate_execution


class SyntheticStream(Base):
    def __init__(self,seed,regime,channels=768,side=28,rank=8):
        assert seed in range(1901,1911) or seed in range(3101,3111)
        if seed>=3101: validate_execution()
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
