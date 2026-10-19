"""Pure endpoint functions for the unexecuted normal-only protocol.

Arrays are [whole images, 784 patches]. The primary tail is nested: upper
one-eighth patches per image, then upper one-eighth images per category.
No feature tensor, class label, clustering or anomaly score is read here.
"""
import math
import numpy as np


def upper_tail(values, fraction=.125, axis=-1):
    a=np.asarray(values,dtype=np.float64)
    if a.shape[axis]==0 or not np.isfinite(a).all():raise ValueError('empty/nonfinite endpoint')
    n=max(1,math.ceil(a.shape[axis]*fraction))
    return np.sort(a,axis=axis).take(range(a.shape[axis]-n,a.shape[axis]),axis=axis).mean(axis=axis)


def old_tail(acquisition,current,scale,indices=None):
    a,b=np.asarray(acquisition),np.asarray(current)
    if a.shape!=b.shape or a.ndim!=2 or scale<=0:raise ValueError('bad paired endpoint')
    signed=(b-a)/scale
    image_signed=upper_tail(signed)
    image_positive=upper_tail(np.maximum(signed,0))
    ix=np.arange(len(a)) if indices is None else np.asarray(indices)
    return {'signed':float(upper_tail(image_signed[ix])),
            'positive':float(upper_tail(image_positive[ix]))}


def acquisition(distances,scale,indices=None):
    d=np.asarray(distances,dtype=np.float64)
    if d.ndim!=2 or scale<=0 or not np.isfinite(d).all():raise ValueError('bad acquisition endpoint')
    ix=np.arange(len(d)) if indices is None else np.asarray(indices)
    image_mean=d.mean(1)/scale
    image_q95=np.sort(d,axis=1)[:,math.ceil(.95*d.shape[1])-1]/scale
    return {'mean':float(image_mean[ix].mean()),'q95':float(image_q95[ix].mean())}


def benefit(reference,candidate):
    absolute=reference-candidate
    return {'absolute':absolute,'relative':absolute/reference if reference>=.02 else None,
            'status':'HEADROOM' if reference>=.02 else 'NO_HEADROOM'}


def acquisition_regression(reference,candidate):
    return {'absolute':candidate-reference,
            'relative':(candidate-reference)/reference if reference>=1e-6 else None,
            'status':'REGULAR' if reference>=1e-6 else 'ZERO_REFERENCE'}


def pre_return_risk(orders,indices=None):
    """Mean over fixed orders of maximum old-category nested tail at t=300.

    Each order maps its TWO older category names to acquisition/current/scale.
    Category keys are evaluator-only. They never enter an allocator or scorer.
    """
    values=[]
    for old in orders:
        if len(old)!=2:raise ValueError('exactly two old categories required')
        values.append(max(old_tail(v['acquisition'],v['current'],v['scale'],
                                   None if indices is None else indices[c])['positive']
                          for c,v in old.items()))
    return float(np.mean(values))


def whole_image_draws(category_sizes,draws=2000,seed=9011):
    rng=np.random.default_rng(seed)
    for _ in range(draws):
        # One shared image-index draw/category reused by every arm/order/boundary.
        yield {c:rng.integers(n,size=n) for c,n in category_sizes.items()}


def simultaneous_basic(points,replicates):
    points,replicates=np.asarray(points),np.asarray(replicates)
    if replicates.ndim!=2 or replicates.shape[1:]!=points.shape:raise ValueError('family shape mismatch')
    radius=float(np.quantile(np.abs(replicates-points).max(1),.95,method='higher'))
    return np.stack((points-radius,points+radius),axis=1)
