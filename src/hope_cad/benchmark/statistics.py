"""CPU-only paired whole-image bootstrap; never selects or modifies a model."""
import argparse,json,hashlib,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
from sklearn.metrics import roc_auc_score
from hope_cad.benchmark.study import study,digest,EVIDENCE_SCHEMA
from hope_cad.benchmark.engine import maps_from_patches

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
    def __init__(self,path,*,schema=EVIDENCE_SCHEMA):
        self.path=Path(path);self.archive=None if self.path.is_dir() else zipfile.ZipFile(path)
        raw=(self.path/'artifact_manifest.json').read_bytes() if self.archive is None else self.archive.read('artifact_manifest.json')
        self.manifest=json.loads(raw)
        if self.manifest['schema']!=schema:raise ValueError('Wrong benchmark evidence schema')
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

def aggregate(paths,output,*,statistics=False,compare='CADIC',research=False):
    cfg=study();jobs={};source=None;test_digest=None;primary='NESTED';schema=EVIDENCE_SCHEMA
    if research:
        from hope_cad.research import contract,study_digest,method_identity,job
        r=contract();s=r['study'];primary='H32_EXACT';schema='hope_hq_fulltrain_evidence_v1'
        cfg=dict(study_id=s['id'],methods=s['variants'],seeds=s['learning_seeds'],bootstrap=dict(replicates=s['statistics']['draws'],seed=s['statistics']['seed'],ci_percentiles=s['statistics']['CI_percentiles']))
    for p in paths:
        e=Evidence(p,schema=schema);identity=e.json('suite_lock.json')['identity']
        if digest(identity)!=e.json('suite_lock.json')['digest']:raise ValueError('Suite identity digest changed')
        if source is None:source=identity
        if identity!=source:raise ValueError('Mixed source/study/runtime/data identity')
        if research:
            if identity['research_study_digest']!=study_digest():raise ValueError('Wrong research study')
        elif identity['study_digest']!=__import__('hope_cad.benchmark.study',fromlist=['study_digest']).study_digest():raise ValueError('Wrong study')
        td=e.json('test_manifest.json')['digest']
        if test_digest is None:test_digest=td
        if td!=test_digest:raise ValueError('Mixed official test identities')
        names={n.split('/')[1] for n in e.manifest['files'] if n.startswith('runs/') and n.endswith('/metrics/final_macro.json')}
        for name in names:
            if name in jobs:raise ValueError('Supply only latest output per job: '+name)
            value=e.json(f'runs/{name}/metrics/final_macro.json')
            if research:
                variant,seed=job(name);jl=e.json(f'runs/{name}/job_lock.json')
                expected=dict(job=name,mode=variant,seed=seed,scope='fresh_fulltrain_3629',
                    method_digest=digest(method_identity(variant)),suite_digest=digest(identity))
                if digest(jl['identity'])!=jl['digest'] or any(jl['identity'].get(k)!=v or type(jl['identity'].get(k)) is not type(v) for k,v in expected.items()):raise ValueError('Unregistered per-job research method/cohort identity')
                trained=e.json(f'runs/{name}/train_complete.json');final=e.json(f'runs/{name}/final_complete.json')
                if trained.get('job_digest')!=jl['digest'] or trained.get('images')!=3629 or trained.get('finalized') is not True or final.get('job_digest')!=jl['digest'] or final.get('status')!='final_complete' or final.get('cell_count')!=15:raise ValueError('Research training/final completion identity changed')
                for ti in range(15):research_cell(e,name,14,ti,jl['digest'],td)
            if value.get('task_count')!=15 or any(not np.isfinite(value[k]) or not 0<=value[k]<=1 for k in ('macro_final_i_auroc','macro_final_p_ap','i_ap','p_auroc')):raise ValueError('Incomplete/invalid final endpoint')
            jobs[name]=(e,value)
    registered={m+'_seed'+str(s) for m in cfg['methods'] for s in cfg['seeds']}
    if research:
        s=contract()['study'];registered=set(s['C_full_seed0_jobs'])|{m+'_seed0' for m in s['D_conditional_remaining_seed0_controls']}|set(s['D_conditional_replicates'])
    if not set(jobs).issubset(registered):raise ValueError('Unregistered result')
    matrices=set()
    for name,(e,_) in jobs.items():
        marker=f'runs/{name}/matrix_complete.json'
        if marker in e.manifest['files']:
            done=e.json(marker)
            if done['cell_count']!=120 or done['status']!='matrix_complete':raise ValueError('Incomplete matrix marker')
            if research and done.get('job_digest')!=e.json(f'runs/{name}/job_lock.json')['digest']:raise ValueError('Research matrix job identity changed')
            for i in range(15):
                for j in range(i+1):
                    if research:research_cell(e,name,i,j,done['job_digest'],test_digest)
                    else:e.json(f'runs/{name}/cells/{i:02d}_{j:02d}/completion.json')
            matrices.add(name)
    out=dict(study_id=cfg['study_id'],source_identity=source,final_metrics={j:v for j,(_,v) in jobs.items()},complete_final_jobs=len(jobs),required_jobs=len(registered),missing_jobs=sorted(registered-set(jobs)),complete_study=registered.issubset(matrices),scientific_superiority_validated=False,scope='Descriptive subset unless every registered job/matrix is present; selected payload hashes/coverage checked, not an independent full score recomputation; normal pilot is not anomaly evidence')
    if statistics:
        if compare not in cfg['methods'] or compare==primary:raise ValueError('Invalid registered comparison')
        required={m+'_seed'+str(s) for m in (primary,compare) for s in cfg['seeds']}
        if not required.issubset(jobs):raise ValueError('Paired CI requires all three finite seeds for both methods')
        b=cfg['bootstrap']['replicates'];samples=np.zeros((2,b,2));point=np.zeros((2,2));draw_hash=hashlib.sha256()
        for ti in range(15):
            reference=None;weights=None
            for mi,method in enumerate((primary,compare)):
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
        out['paired_statistics']=dict(comparison=primary+'-'+compare,point_difference=(point[0]-point[1]).tolist(),endpoints=['macro_final_i_auroc','macro_final_p_ap'],ci=ci.T.tolist(),replicates=b,draw_sha256=draw_hash.hexdigest(),sampler_seed=cfg['bootstrap']['seed'],noninferior_both=bool((ci[0]>-.02).all()),superior_both=bool((ci[0]>0).all()),scope='MVTec development, fixed3seed average, not population-seed uncertainty or VisA confirmation')
    Path(output).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8');return out


def research_cell(e,name,checkpoint,task,job_digest,test_digest):
    """Bind every selected completion to its registered job and evaluator cell."""
    base=f'runs/{name}/cells/{checkpoint:02d}_{task:02d}/'
    locked=e.json(base+'cell_lock.json');done=e.json(base+'completion.json');ci=locked['identity']
    expected=dict(job_digest=job_digest,test_manifest_digest=test_digest,checkpoint_id=checkpoint,task_id=task)
    if digest(ci)!=locked['digest'] or done.get('cell_digest')!=locked['digest'] or any(ci.get(k)!=v for k,v in expected.items()):raise ValueError('Research cell job/evaluator identity changed')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evidence',nargs='+',required=True);ap.add_argument('--output',required=True);ap.add_argument('--statistics',action='store_true');ap.add_argument('--compare',default='CADIC')
    a=ap.parse_args();aggregate(a.evidence,a.output,statistics=a.statistics,compare=a.compare)
if __name__=='__main__':main()
