"""CPU-only paired whole-image bootstrap; never selects or modifies a model."""
import argparse,json,hashlib,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
from sklearn.metrics import roc_auc_score
from .contract import study,digest,EVIDENCE_SCHEMA
from .engine import maps_from_patches

def multiplicities(labels,replicates,task_id,seed=20261020):
    labels=np.asarray(labels);rng=np.random.default_rng(np.random.SeedSequence([seed,task_id]));out=np.zeros((replicates,len(labels)),dtype=np.int32)
    if set(labels.tolist())!={0,1}:raise ValueError('Both image strata required')
    for label in (0,1):
        positions=np.flatnonzero(labels==label)
        for b in range(replicates):out[b]+=np.bincount(rng.choice(positions,len(positions),replace=True),minlength=len(labels)).astype(np.int32)
    return out

class PixelAP:
    """Sort once, preserve sklearn tie handling, apply whole-image multiplicities."""
    def __init__(self,maps,masks):
        if maps.shape!=masks.shape or maps.ndim!=3 or not np.isfinite(maps).all():raise ValueError('Invalid pixel arrays')
        n,h,w=maps.shape;flat=maps.reshape(-1);order=np.argsort(-flat,kind='stable')
        self.image=(order//(h*w)).astype(np.int32);self.positive=masks.reshape(-1)[order].astype(bool)
        sorted_scores=flat[order];self.ends=np.r_[np.flatnonzero(sorted_scores[1:]!=sorted_scores[:-1]),len(order)-1]
    def ap(self,weights):
        mass=np.asarray(weights,dtype=np.float64)[self.image];tp=np.cumsum(mass*self.positive);den=np.cumsum(mass)
        if tp[-1]==0:return 0. # Locked bootstrap degenerate convention; never change/resample draw.
        tp=tp[self.ends];den=den[self.ends];precision=np.divide(tp,den,out=np.zeros_like(tp),where=den>0)
        return float(np.sum(np.diff(np.r_[0.,tp])*precision)/tp[-1])

class Evidence:
    """Verified selected score payloads; does not extract/load learned checkpoints."""
    def __init__(self,path):
        self.path=Path(path);self.archive=None if self.path.is_dir() else zipfile.ZipFile(path)
        raw=(self.path/'artifact_manifest.json').read_bytes() if self.archive is None else self.archive.read('artifact_manifest.json')
        self.manifest=json.loads(raw)
        if self.manifest['schema']!=EVIDENCE_SCHEMA:raise ValueError('Wrong benchmark evidence schema')
        for n in self.manifest['files']:
            p=PurePosixPath(n)
            if p.is_absolute() or '..' in p.parts or ':' in n or '\\' in n:raise ValueError('Unsafe evidence path')
        if self.archive is not None:
            names=self.archive.namelist()
            if len(names)!=len(set(names)) or set(names)!=set(self.manifest['files'])|{'artifact_manifest.json'}:raise ValueError('Evidence inventory mismatch')
    def bytes(self,n):
        p=self.path/n
        if self.archive is None and (not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(self.path.resolve())):raise ValueError('Missing/symlinked/escaping payload')
        raw=p.read_bytes() if self.archive is None else self.archive.read(n)
        if hashlib.sha256(raw).hexdigest()!=self.manifest['files'][n]:raise ValueError('Selected evidence checksum changed')
        return raw
    def json(self,n):return json.loads(self.bytes(n))
    def final(self,name,task_id):
        import io
        base=f'runs/{name}/cells/14_{task_id:02d}/';done=self.json(base+'completion.json');locked=self.json(base+'cell_lock.json')
        if done['cell_digest']!=locked['digest']:raise ValueError('Cell digest changed')
        payload={}
        for n,h in sorted(done['chunks'].items()):
            raw=self.bytes(base+n)
            if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Score hash binding changed')
            with np.load(io.BytesIO(raw),allow_pickle=False) as arrays:
                for k in arrays.files:payload.setdefault(k,[]).append(arrays[k])
        return {k:np.concatenate(v) for k,v in payload.items()},locked['identity']['rows']

def aggregate(paths,output,*,statistics=False,compare='CADIC'):
    cfg=study();jobs={};source=None;test_digest=None
    for p in paths:
        e=Evidence(p);identity=e.json('suite_lock.json')['identity']
        if digest(identity)!=e.json('suite_lock.json')['digest']:raise ValueError('Suite identity digest changed')
        if source is None:source=identity
        if identity!=source:raise ValueError('Mixed source/study/runtime/data identity')
        if identity['study_digest']!=__import__('training.support_benchmark_v5_1.contract',fromlist=['study_digest']).study_digest():raise ValueError('Wrong study')
        td=e.json('test_manifest.json')['digest']
        if test_digest is None:test_digest=td
        if td!=test_digest:raise ValueError('Mixed official test identities')
        names={n.split('/')[1] for n in e.manifest['files'] if n.startswith('runs/') and n.endswith('/metrics/final_macro.json')}
        for name in names:
            if name in jobs:raise ValueError('Supply only latest output per job: '+name)
            value=e.json(f'runs/{name}/metrics/final_macro.json')
            if value.get('task_count')!=15 or any(not np.isfinite(value[k]) or not 0<=value[k]<=1 for k in ('macro_final_i_auroc','macro_final_p_ap','i_ap','p_auroc')):raise ValueError('Incomplete/invalid final endpoint')
            jobs[name]=(e,value)
    registered={m+'_seed'+str(s) for m in cfg['methods'] for s in cfg['seeds']}
    if not set(jobs).issubset(registered):raise ValueError('Unregistered result')
    matrices=set()
    for name,(e,_) in jobs.items():
        marker=f'runs/{name}/matrix_complete.json'
        if marker in e.manifest['files']:
            done=e.json(marker)
            if done['cell_count']!=120 or done['status']!='matrix_complete':raise ValueError('Incomplete matrix marker')
            for i in range(15):
                for j in range(i+1):e.json(f'runs/{name}/cells/{i:02d}_{j:02d}/completion.json')
            matrices.add(name)
    out=dict(study_id=cfg['study_id'],source_identity=source,final_metrics={j:v for j,(_,v) in jobs.items()},complete_final_jobs=len(jobs),required_jobs=len(registered),missing_jobs=sorted(registered-set(jobs)),complete_study=registered.issubset(matrices),scientific_superiority_validated=False,scope='Descriptive subset unless every registered job/matrix is present; selected payload hashes/coverage checked, not an independent full score recomputation; normal pilot is not anomaly evidence')
    if statistics:
        if compare not in cfg['methods'] or compare=='NESTED':raise ValueError('Invalid registered comparison')
        required={m+'_seed'+str(s) for m in ('NESTED',compare) for s in cfg['seeds']}
        if not required.issubset(jobs):raise ValueError('Paired CI requires all three finite seeds for both methods')
        b=cfg['bootstrap']['replicates'];samples=np.zeros((2,b,2));point=np.zeros((2,2));draw_hash=hashlib.sha256()
        for ti in range(15):
            reference=None;weights=None
            for mi,method in enumerate(('NESTED',compare)):
                for seed in cfg['seeds']:
                    name=method+'_seed'+str(seed);data,rows=jobs[name][0].final(name,ti)
                    binding=digest(dict(rows=rows,labels=data['labels'].tolist(),mask_sha256=hashlib.sha256(data['masks'].tobytes()).hexdigest()))
                    if reference is None:
                        reference=binding;weights=multiplicities(data['labels'],b,ti,cfg['bootstrap']['seed']);draw_hash.update(weights.tobytes())
                    if binding!=reference:raise ValueError('Methods/seeds do not have paired images/GT')
                    ap=PixelAP(maps_from_patches(data['patches']),data['masks'])
                    observed=np.array([roc_auc_score(data['labels'],data['image_scores']),ap.ap(np.ones(len(rows)))])
                    point[mi]+=observed/45
                    for ri,w in enumerate(weights):
                        samples[mi,ri,0]+=roc_auc_score(data['labels'],data['image_scores'],sample_weight=w)/45
                        samples[mi,ri,1]+=ap.ap(w)/45
                    print('[BOOTSTRAP CPU]',name,'task',ti,'complete',flush=True)
        delta=samples[0]-samples[1];ci=np.percentile(delta,cfg['bootstrap']['ci_percentiles'],axis=0)
        out['paired_statistics']=dict(comparison='NESTED-'+compare,point_difference=(point[0]-point[1]).tolist(),endpoints=['macro_final_i_auroc','macro_final_p_ap'],ci=ci.T.tolist(),replicates=b,draw_sha256=draw_hash.hexdigest(),sampler_seed=cfg['bootstrap']['seed'],noninferior_both=bool((ci[0]>-.02).all()),superior_both=bool((ci[0]>0).all()),scope='MVTec development, fixed3seed average, not population-seed uncertainty or VisA confirmation')
    Path(output).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8');return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evidence',nargs='+',required=True);ap.add_argument('--output',required=True);ap.add_argument('--statistics',action='store_true');ap.add_argument('--compare',default='CADIC')
    a=ap.parse_args();aggregate(a.evidence,a.output,statistics=a.statistics,compare=a.compare)
if __name__=='__main__':main()
