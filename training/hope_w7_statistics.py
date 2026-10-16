"""Paired whole-image bootstrap. Exact point metrics stay in the benchmark engine.

Pixel bootstrap uses per-image histograms only after 32 exact weighted checks.
No pixel is treated as an independent resampling unit. Categories/seeds stay fixed.
"""
import io
import json
from pathlib import Path
import zipfile
import hashlib

import numpy as np

from dataset.benchmark_manifest_v1 import canonical_digest
from training.hope_w7_engine import atomic_npz
from training.hope_runtime_v4 import atomic_json


def image_counts(labels,category,replicates=2000):
    labels=np.asarray(labels)
    if set(np.unique(labels))!={0,1}:
        raise ValueError('bootstrap needs both official image labels')
    # Same category seed and canonical image order for every method/checkpoint/seed.
    rng=np.random.default_rng(np.random.SeedSequence([20261009,category]))
    counts=np.zeros((replicates,len(labels)),dtype=np.int32)
    for label in (0,1):
        ix=np.flatnonzero(labels==label)
        counts[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=replicates)
    return counts


def weighted_curves(positive,negative):
    """Inputs are weighted counts in DESCENDING score groups, with ties grouped."""
    positive,negative=np.atleast_2d(positive),np.atleast_2d(negative)
    tp=np.cumsum(positive,axis=-1);fp=np.cumsum(negative,axis=-1)
    total_p,total_n=tp[:,-1:],fp[:,-1:]
    if np.any(total_p<=0) or np.any(total_n<=0):
        raise ValueError('undefined weighted binary metric')
    precision=np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=tp+fp>0)
    ap=np.sum((positive/total_p)*precision,axis=-1)
    previous_fp=np.concatenate([np.zeros_like(fp[:,:1]),fp[:,:-1]],axis=-1)
    previous_tp=np.concatenate([np.zeros_like(tp[:,:1]),tp[:,:-1]],axis=-1)
    auc=np.sum((fp-previous_fp)*(tp+previous_tp)/2,axis=-1)/(total_p[:,0]*total_n[:,0])
    return auc,ap


def exact_bootstrap(scores,labels,counts):
    scores=np.asarray(scores).reshape(-1);labels=np.asarray(labels).reshape(-1)
    order=np.argsort(-scores,kind='stable')
    sorted_scores=scores[order]
    starts=np.r_[0,np.flatnonzero(sorted_scores[1:]!=sorted_scores[:-1])+1]
    weighted=np.asarray(counts,dtype=float)[:,order]
    positive=np.add.reduceat(weighted*labels[order],starts,axis=1)
    negative=np.add.reduceat(weighted*(1-labels[order]),starts,axis=1)
    return weighted_curves(positive,negative)


def pixel_bootstrap(maps,masks,counts,*,exact=False,max_bins=262144):
    maps=np.asarray(maps);masks=np.asarray(masks)
    n=len(maps);pixels=maps[0].size
    flat=maps.reshape(-1);truth=masks.reshape(-1).astype(np.uint8)
    order=np.argsort(-flat,kind='stable')
    starts=np.r_[0,np.flatnonzero(flat[order][1:]!=flat[order][:-1])+1]
    image_ids=(order//pixels).astype(np.int32)
    ordered_truth=truth[order]
    def exact_values(weights):
        values=[]
        # One replicate at a time bounds memory for exact validation/fallback.
        for w in weights:
            per_pixel=w[image_ids].astype(float)
            pos=np.add.reduceat(per_pixel*ordered_truth,starts)
            neg=np.add.reduceat(per_pixel*(1-ordered_truth),starts)
            auc,ap=weighted_curves(pos,neg)
            values.append((auc[0],ap[0]))
        return np.asarray(values)
    if exact:
        values=exact_values(counts)
        return values[:,1],dict(mode='exact',replicates=len(counts),validation_error=0.0)
    validation_count=min(32,len(counts))
    expected=exact_values(counts[:validation_count])[:,1]
    low,high=float(flat.min()),float(flat.max())
    bins=4096
    while bins<=max_bins:
        span=max(high-low,np.finfo(float).tiny)
        bucket=np.clip(((maps.astype(float)-low)/span*(bins-1)).astype(np.int64),0,bins-1)
        positives=np.empty((n,bins));negatives=np.empty((n,bins))
        for image in range(n):
            b=bucket[image].reshape(-1);y=masks[image].reshape(-1)
            positives[image]=np.bincount(b,weights=y,minlength=bins)[::-1]
            negatives[image]=np.bincount(b,weights=1-y,minlength=bins)[::-1]
        def approximate(weights):
            result=[]
            for start in range(0,len(weights),32):
                w=weights[start:start+32].astype(float)
                _,ap=weighted_curves(w@positives,w@negatives)
                result.extend(ap.tolist())
            return np.asarray(result)
        got=approximate(counts[:validation_count])
        error=float(np.max(np.abs(got-expected)))
        if error<=2e-4:
            return approximate(counts),dict(mode='per_image_histogram',bins=bins,
                validation_replicates=validation_count,validation_error=error,tolerance=2e-4)
        bins*=4
    raise ValueError('pixel bootstrap histogram exceeds registered error tolerance; use exact CPU mode, no CI claim yet')


def cell_bootstrap(arrays,counts,*,exact_pixels=False):
    i_auroc,_=exact_bootstrap(arrays['image_scores'],arrays['labels'],counts)
    p_aupr,accuracy=pixel_bootstrap(arrays['maps'],arrays['masks'],counts,exact=exact_pixels)
    return dict(i_auroc=i_auroc,p_aupr=p_aupr),accuracy


class EvidenceView:
    """Read a saved W7 ZIP directly; do not expand dozens of multi-GB outputs."""
    def __init__(self,path):
        self.path=Path(path);self.archive=None
        if self.path.is_file():
            self.archive=zipfile.ZipFile(self.path)
            names=self.archive.namelist()
            if len(names)!=len(set(names)):
                raise ValueError('duplicate evidence entries')
            self.manifest=json.loads(self.archive.read('artifact_manifest.json'))
        else:
            self.manifest=json.loads((self.path/'artifact_manifest.json').read_text())
        if self.manifest.get('schema')!='hope_w7_evidence_v1':
            raise ValueError('not W7 evidence')
        self.suite=self.json('suite_lock.json')
        if self.suite['digest']!=canonical_digest(self.suite['identity']):
            raise ValueError('invalid W7 suite digest')
    def read(self,name):
        from training.hope_w7_io import destination
        destination(Path('/safe'),name)
        if name not in self.manifest['files']:
            raise ValueError('unlisted W7 evidence file')
        raw=self.archive.read(name) if self.archive else (self.path/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=self.manifest['files'][name]:
            raise ValueError('W7 evidence checksum mismatch')
        return raw
    def json(self,name): return json.loads(self.read(name))
    def completions(self):
        return [n for n in self.manifest['files'] if n.startswith('runs/') and n.endswith('/completion.json')]
    def cell(self,job,checkpoint,task):
        prefix=f'runs/{job}/scores/{checkpoint:02d}_{task:02d}/'
        progress=self.json(prefix+'progress.json');result=self.json(prefix+'result.json')
        if result['status']!='complete' or progress['identity']!=result['identity']:
            raise ValueError('incomplete raw evaluation cell')
        values={k:[] for k in ('image_scores','maps','masks','labels','identities')}
        offset=0
        for shard in progress['shards']:
            if shard['start']!=offset:
                raise ValueError('noncontiguous raw evaluation shards')
            raw=self.read(prefix+shard['file'])
            if hashlib.sha256(raw).hexdigest()!=shard['sha256']:
                raise ValueError('score shard differs from its cell receipt')
            with np.load(io.BytesIO(raw),allow_pickle=False) as arrays:
                for key in values: values[key].append(arrays[key])
            offset=shard['end']
        values={k:np.concatenate(v) for k,v in values.items()}
        if len(values['identities'])!=offset or offset!=progress['offset']:
            raise ValueError('incomplete raw image inventory')
        return values
    def close(self):
        if self.archive:self.archive.close()


def continual_bootstrap(matrix):
    # [checkpoint, task, replicate]; same replicate weights at EVERY checkpoint.
    n=matrix.shape[0]
    final=matrix[-1].mean(axis=0)
    bwt=(matrix[-1,:-1]-np.stack([matrix[j,j] for j in range(n-1)])).mean(axis=0)
    forgetting=np.stack([matrix[j:n-1,j].max(axis=0)-matrix[-1,j] for j in range(n-1)]).mean(axis=0)
    return dict(final=final,bwt=bwt,forgetting=forgetting)


def paired_interval(values,primary=True):
    percentiles=[1.25,98.75] if primary else [2.5,97.5]
    low,high=np.percentile(values,percentiles)
    return dict(lower=float(low),upper=float(high),percentiles=percentiles,
                confidence=.975 if primary else .95,
                unit='paired whole images within fixed category/image-label strata; averaged over fixed trained seeds')
