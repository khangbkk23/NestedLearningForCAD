"""Inherited raw geometry, with a separate FP64 causal decision scale."""
import random
import math
import torch
from models.hope_cad_v5.geometry import (ForkableCoreset, risk, anchor_risk,
    FixedContextProjection, advantage, AnchorReservoir as OriginalReservoir)
from .validation import keys, integer, tensor, scalar


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
    from models.hope_cad_v2.state import finite_tensors
    finite_tensors((stats,),'causal statistics')
    return stats.detach(),dict(current0=r0,current1=r1,anchor0=anchor0,anchor1=anchor1)


class AnchorReservoir(OriginalReservoir):
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
