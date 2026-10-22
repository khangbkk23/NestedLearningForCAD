"""Raw CADIC geometry stays exact; only proposal admission is learned."""
from copy import deepcopy
import math,random,torch
from dataclasses import asdict
from hope_cad.baselines.cadic.cache import CachedPairCoreset
from hope_cad.baselines.cadic.bank import CADICPatchCoresetConfig
from hope_cad.models.primitives.state import finite_tensors

class ForkableCoreset(CachedPairCoreset):
    def fork(self):
        """Clone scientific state and valid tile minima without a quadratic rebuild."""
        other=type(self)(self.config,device=self.device)
        other.features=self.features.detach().clone()
        for name in ('seen_features','accepted_features','replaced_features','rejected_features','update_batches'):
            setattr(other,name,getattr(self,name))
        other._distance_profile=deepcopy(self._distance_profile)
        other._cache_counts=deepcopy(self._cache_counts);other._cache_peak_bytes=self._cache_peak_bytes
        if self._cache_signature==self._signature():
            other._tiles=list(self._tiles);other._adjacent=[list(v) for v in self._adjacent];other._dirty=set(self._dirty)
            for name in ('_tile_values','_tile_pairs','_triangle'):
                value=getattr(self,name);setattr(other,name,value.clone() if value is not None else None)
            other._cache_signature=other._signature()
        return other

@torch.no_grad()
def risk(features,bank):
    if features.ndim==3:
        if features.shape[0]!=1:raise ValueError('one image per coverage event')
        features=features[0]
    if features.ndim!=2 or features.shape[1]!=bank.config.dim or not len(features) or not bank.count:raise ValueError('invalid coverage geometry')
    finite_tensors((features,bank.features),'coverage input')
    distances,_=bank._nearest(features,bank.features);sq=distances.square()
    value=.5*(sq.mean()+sq.max());finite_tensors((value,),'coverage risk')
    return float(value)

@torch.no_grad()
def anchor_risk(anchors,bank):
    return sum(risk(a,bank) for a in anchors)/len(anchors) if anchors else 0.

def advantage(before,after):
    if not math.isfinite(before+after) or min(before,after)<0:raise ValueError('invalid counterfactual risks')
    return (before-after)/(before+after+1e-8)


class _AnchorReservoir:
    """Uniform image reservoir, eight uniform patches/image; no task identity."""
    def __init__(self,seed=0,capacity=32,patches=8):
        self.capacity,self.patches=capacity,patches;self.rng=random.Random(seed+73471);self.seen=0;self.images=[];self.identities=[]
    def update(self,features,identity):
        f=features.reshape(-1,features.shape[-1]);self.seen+=1
        positions=self.rng.sample(range(len(f)),min(self.patches,len(f)))
        sample=f[positions].detach().clone()
        slot=len(self.images) if len(self.images)<self.capacity else self.rng.randrange(self.seen)
        if slot<self.capacity:
            if slot==len(self.images):self.images.append(sample);self.identities.append(identity)
            else:self.images[slot]=sample;self.identities[slot]=identity
    def state_dict(self):return dict(capacity=self.capacity,patches=self.patches,seen=self.seen,images=[v.detach().cpu().clone() for v in self.images],identities=list(self.identities),rng=self.rng.getstate())
    def load_state_dict(self,payload,device):
        if payload['capacity']!=self.capacity or payload['patches']!=self.patches:raise ValueError('anchor config changed')
        self.seen=payload['seen'];self.images=[v.to(device).detach().clone() for v in payload['images']];self.identities=list(payload['identities']);self.rng.setstate(payload['rng'])
        if len(self.images)>self.capacity or len(self.identities)!=len(self.images) or self.seen<len(self.images):raise ValueError('corrupt anchor reservoir')

class FixedContextProjection:
    def __init__(self,raw_dim=768,dim=32,seed=20261017,device='cpu'):
        gen=torch.Generator().manual_seed(seed)
        self.matrix=(torch.randint(0,2,(raw_dim,dim),generator=gen).float()*2-1).to(device)/math.sqrt(dim)
    @torch.no_grad()
    def __call__(self,features):
        raw=features.reshape(1,-1,self.matrix.shape[0]);u=raw@self.matrix
        return (u/torch.sqrt(u.square().mean()+1e-6)).detach()

"""Inherited raw geometry, with a separate FP64 causal decision scale."""
import random
import math
import torch
from hope_cad.models.validation import keys, integer, tensor, scalar


def causal_scale(current0, current1, anchor0, anchor1, weight=1.):
    for v in (current0, current1, anchor0, anchor1):
        scalar(v, 'counterfactual risk', 0.)
    c = 1. + current0 + current1 + weight * (anchor0 + anchor1)
    numerator = (current0 + weight * anchor0) - (current1 + weight * anchor1)
    scalar(c, 'causal scale', 1.)
    scalar(numerator, 'causal numerator')
    return c, numerator / c


@torch.no_grad()
def context_statistics(features, old, new, result, anchors_seen, stream_cursor,
                       anchor0, anchor1, anchor_weight=1.):
    """Inherited 16 statistics; cast R only AFTER its FP32 half multiplication."""
    f=features.reshape(-1,features.shape[-1])
    d0,_=old._nearest(f,old.features);d1,_=new._nearest(f,new.features)
    scale=f.norm(dim=-1).mean().clamp_min(1e-8)
    r0=float(.5*(d0.square().mean()+d0.square().max()))
    r1=float(.5*(d1.square().mean()+d1.square().max()))
    value=lambda v:torch.tensor(v,device=f.device,dtype=torch.float32)
    stats=torch.stack((d0.mean()/scale,torch.quantile(d0,.95)/scale,d0.max()/scale,d0.min()/scale,
        d0.std(unbiased=False)/scale,d1.mean()/scale,d1.max()/scale,value(advantage(r0,r1)),
        value(advantage(anchor0,anchor1) if anchor_weight else 0.),value(old.count/old.config.budget),
        (old.features!=new.features).any(dim=-1).float().mean(),value(result['replaced']/len(f)),
        value(math.log1p(stream_cursor)/math.log1p(1e6)),value(min(anchors_seen,32)/32),
        f.norm(dim=-1).mean()/old.features.norm(dim=-1).mean().clamp_min(1e-8),
        value(advantage(r0+anchor_weight*anchor0,r1+anchor_weight*anchor1))))
    from hope_cad.models.primitives.state import finite_tensors
    finite_tensors((stats,),'causal statistics')
    return stats.detach(),dict(current0=r0,current1=r1,anchor0=anchor0,anchor1=anchor1)


class AnchorReservoir(_AnchorReservoir):
    def validate(self, payload, *, raw_dim, patches, seen_ids, expected_seen):
        keys(payload, ('capacity', 'patches', 'seen', 'images', 'identities', 'rng'), 'anchors')
        if payload['capacity'] != self.capacity or payload['patches'] != self.patches:
            raise ValueError('anchor topology changed')
        integer(payload['seen'], 'anchor seen')
        if payload['seen'] != expected_seen or not isinstance(payload['images'], list):
            raise ValueError('anchor clock changed')
        if len(payload['images']) != min(self.capacity, expected_seen):
            raise ValueError('anchor occupancy inconsistent')
        identities = payload['identities']
        if not isinstance(identities, list) or len(identities) != len(payload['images']) or len(set(identities)) != len(identities):
            raise ValueError('invalid anchor identities')
        if any(not isinstance(v, str) or v not in seen_ids for v in identities):
            raise ValueError('anchor identity was not observed')
        for image in payload['images']:
            tensor(image, (patches, raw_dim), torch.float32, 'anchor patches')
        try:
            random.Random().setstate(payload['rng'])
        except (ValueError, TypeError) as error:
            raise ValueError('invalid private anchor RNG') from error
