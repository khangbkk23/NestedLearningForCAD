"""First two-arm full NAR benchmark. Reuses the registered NAR architecture.

Stages: lock, smoke, train, score, metrics. GPU stages require owner authorization.
Training and smoke open normal training inputs only. Official test inputs and
masks are confined to score/metrics, after BOTH arms finish and are sealed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exps.dinomaly import official as O
from exps.nar_pilot_00 import core as K
from exps.nar_pilot_00.detector import Detector, prefix_grids, fm_eq8
from exps.nar_pilot_00.run import sha, save, write_json, assert_cache
from exps.vitb8_q2.common import SingleStreamDecoder
from exps.calm02.data import CHECKPOINT, CHECKPOINT_SHA, MVTEC_ORDER

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/hope_cad/nar_full_15'
DATA=ROOT/'data/mvtec'
ARMS=('nested','fixed')
ORDER=tuple(MVTEC_ORDER)
CFG=K.Cfg()
SEED=2711
BUDGET=9000
SOURCES=('exps/nar_pilot_00/core.py','exps/nar_pilot_00/detector.py',
 'exps/nar_pilot_00/full.py','exps/nar_pilot_00/run.py','exps/vitb8_q2/common.py',
 'exps/dinomaly/official.py','exps/calm02/data.py','exps/calm02/native.py',
 'exps/cadfinal/metrics.py','exps/task_p/p1_readout.py',
 'models/feature_extractors/cadic_vit_v1.py','dataset/benchmark_protocol_v1.py',
 'agents/plans/nar_full_15_protocol.md')


def read(p):
    return json.loads(Path(p).read_text())


def log(*args):
    print(*args,flush=True)


def configure(gpu=True):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if gpu:
        K.require(torch.cuda.is_available(),'CUDA unavailable; no CPU substitution for benchmark')
        torch.cuda.set_per_process_memory_fraction(5*2**30/torch.cuda.get_device_properties(0).total_memory)


def cap(start,seconds=BUDGET):
    K.require(time.monotonic()-start<seconds,'wall budget exceeded')
    K.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8.5e9,'RSS budget exceeded')
    if torch.cuda.is_initialized():
        K.require(torch.cuda.max_memory_allocated()<=5*2**30,'CUDA budget exceeded')


def verify_source():
    lk=read(OUT/'SOURCE_LOCK.json')
    K.require(all(sha(ROOT/p)==h for p,h in lk['sources'].items()),'source lock changed')
    return lk


def lock():
    K.require(not (OUT/'SOURCE_LOCK.json').exists(),'source lock already exists')
    K.require(sha(CHECKPOINT)==CHECKPOINT_SHA,'encoder checkpoint identity')
    norm=read(ROOT/'results/hope_cad/norm_00/RESULT.json')
    K.require(norm['verdict']=='NORMALIZATION_MATTERS_GO','registered frontend decision')
    write_json(OUT/'SOURCE_LOCK.json',{'schema':'nar_full_15_v1','seed':SEED,'cfg':CFG.dict(),
        'arms':ARMS,'order':ORDER,'sources':{p:sha(ROOT/p) for p in SOURCES},
        'encoder_sha256':CHECKPOINT_SHA,'normalization':'INC','budget_seconds':BUDGET,
        'norm_result_sha256':sha(ROOT/'results/hope_cad/norm_00/RESULT.json'),
        'torch':torch.__version__,'purpose':'exposed-MVTec exploratory benchmark'})
    log('SOURCE_LOCK',sha(OUT/'SOURCE_LOCK.json'))


def rows_for(c,split):
    if split=='train':
        return [{'relative_path':p.relative_to(DATA).as_posix()}
                for p in sorted((DATA/c/'train/good').glob('*.png'))]
    rows=[]
    for folder in sorted((DATA/c/'test').iterdir()):
        if folder.is_dir():
            for p in sorted(folder.glob('*.png')):
                anomaly=folder.name!='good'
                rows.append({'relative_path':p.relative_to(DATA).as_posix(),'label':int(anomaly),
                 'mask_path':f'{c}/ground_truth/{folder.name}/{p.stem}_mask.png' if anomaly else None})
    return rows


def make_encoder():
    from models.feature_extractors.cadic_vit_v1 import CADICViTConfig,CADICViTFeatureExtractor
    return CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=str(CHECKPOINT),
        checkpoint_identity='sha256:'+CHECKPOINT_SHA,mean=(.5,)*3,std=(.5,)*3),'cuda')


def cache(c,split,encoder):
    K.require(c in ORDER and split in ('train','test'),'cache cohort')
    if split=='test':
        verify_training()
    target=OUT/'features'/f'{c}_{split}.pt'
    meta=target.with_suffix('.json')
    if target.exists() or meta.exists():
        K.require(target.exists() and meta.exists(),'partial cache; preserve and audit')
        rec=read(meta)
        K.require(sha(target)==rec['sha256'],'cache hash')
        K.require(rec['source_lock_sha256']==sha(OUT/'SOURCE_LOCK.json'),'cache provenance')
        return target
    from dataset.benchmark_protocol_v1 import _ImageDataset
    rows=rows_for(c,split)
    if split=='train': K.normal_partition(rows)
    ds=_ImageDataset(DATA,rows,224,(.5,)*3,(.5,)*3,training=True)
    start=time.monotonic()
    features=torch.empty(len(rows),784,768,dtype=torch.float32)
    with torch.no_grad():
        for i in range(0,len(rows),8):
            x=torch.stack([ds[j]['images'] for j in range(i,min(i+8,len(rows)))]).cuda()
            features[i:i+len(x)]=encoder.extract_patch_features(x).float().cpu()
    K.require(bool(torch.isfinite(features).all()),'feature finiteness')
    target.parent.mkdir(parents=True,exist_ok=True)
    save(target,{'rows':rows,'features':features})
    rec={'sha256':sha(target),'source_lock_sha256':sha(OUT/'SOURCE_LOCK.json'),
        'n':len(rows),'split':split,'category':c,'normalization':'INC','seconds':time.monotonic()-start,
        'source_images':{r['relative_path']:sha(DATA/r['relative_path']) for r in rows}}
    write_json(meta,rec)
    log('CACHE',c,split,len(rows),round(rec['seconds'],2),'s')
    return target


def model_new(base=None):
    torch.manual_seed(SEED)
    model=SingleStreamDecoder(checkpoint=True).cuda()
    if base is None: O.init_trainable(model)
    else: model.load_state_dict(base)
    names=[n for n,m in model.named_modules() if isinstance(m,nn.Linear)]
    base_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    loras=K.attach(model,names,CFG.rank)
    return model,loras,base_state


def loss(model,x):
    en,de=model(x)
    return O.global_cosine(en,de)


@torch.no_grad()
def normal_eval(model,features,indices):
    was=model.training
    model.eval()
    values=[float(loss(model,features[j:j+1].cuda())) for j in indices[:4]]
    model.train(was)
    return float(np.mean(values))


def fit_task(model,loras,features,roles,stage,lam,learn,cfg=CFG):
    torch.manual_seed(SEED+100*stage)
    gen=torch.Generator().manual_seed(SEED+100*stage)
    def fit(i):
        j=roles['FIT'][int(torch.randperm(len(roles['FIT']),generator=gen)[0])]
        return loss(model,features[j:j+1].cuda())
    credit_n=0
    def credit():
        nonlocal credit_n
        j=roles['CREDIT'][credit_n%len(roles['CREDIT'])]
        credit_n+=1
        model.eval()
        v=loss(model,features[j:j+1].cuda())
        model.train()
        return v
    start=time.monotonic()
    def observe(k):
        cap(start)
        if k%50==0: log('FAST',stage,k)
    opt=lambda groups:O.StableAdamW(groups,lr=cfg.lr,betas=cfg.betas,
        weight_decay=cfg.weight_decay,eps=cfg.eps,amsgrad=True)
    model.train()
    torch.cuda.synchronize()
    lam,trace=K.acquire(loras,fit,credit,cfg,lam,learn,opt,on_step=observe,observe_credit=True)
    torch.cuda.synchronize()
    K.require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'parameter finiteness')
    return lam,trace,time.monotonic()-start


def state_equal(a,b):
    return a.keys()==b.keys() and all(torch.equal(a[n][i],b[n][i]) for n in a for i in (0,1))


def smoke():
    configure()
    verify_source()
    K.require(not (OUT/'SMOKE.json').exists(),'smoke already completed; preserve it')
    encoder=make_encoder()
    paths=[cache(c,'train',encoder) for c in ORDER[:2]]
    del encoder
    torch.cuda.empty_cache()
    model,loras,base=model_new()
    phi=K.init_state(loras,SEED)
    lam=torch.zeros(len(loras),dtype=torch.float64)
    archive=[]
    records=[]
    total_start=time.monotonic()
    for stage,path in enumerate(paths):
        data=torch.load(path,weights_only=True,mmap=True)
        roles=assert_cache(data,ORDER[stage]); features=data['features']
        K.set_state(loras,phi)
        before=normal_eval(model,features,roles['EVAL'])
        prior=K.get_state(loras)
        lam_prior=lam.clone()
        cfg=CFG if stage==0 else replace(CFG,steps=30)
        lam,trace,wall=fit_task(model,loras,features,roles,stage,lam,True,cfg)
        endpoint=K.get_state(loras)
        after=normal_eval(model,features,roles['EVAL'])
        archive.append(endpoint)
        next_phi=K.reptile(phi,endpoint,CFG.beta)
        records.append({'task':ORDER[stage],'steps':cfg.steps,'before':before,'after':after,'seconds':wall,
            'fit_first':trace['fit'][:5],'fit_last':trace['fit'][-5:],
            'adapter_changed':not state_equal(prior,endpoint),'phi_changed':not state_equal(phi,next_phi),
            'lambda_changed':not torch.equal(lam_prior,lam),'lambda_max':float(lam.abs().max())})
        phi=next_phi
    x=features[roles['EVAL'][0]:roles['EVAL'][0]+1].cuda()
    detector=Detector(model,loras,archive)
    grids=detector.expert_grids(x)
    prediction,images=detector.predict(x,(320,256))
    reuse=prefix_grids(grids)
    prefix_equal=all(torch.equal(reuse[t],K.task_agnostic_grid(list(grids[:t+1]))[0]) for t in range(len(archive)))
    save(OUT/'SMOKE_STATE.pt',{'model':model.state_dict(),'archive':archive,'phi':phi,'lam':lam})
    state=torch.load(OUT/'SMOKE_STATE.pt',weights_only=True)
    model.load_state_dict(state['model'])
    detector=Detector(model,loras,state['archive'])
    reopened,reimage=detector.predict(x,(320,256))
    parity=torch.equal(reopened,prediction) and torch.equal(reimage,images)
    # Batch-four forward estimate on normal inputs only.
    xb=x.repeat(4,1,1)
    model.eval()
    with torch.no_grad():
        model(xb); torch.cuda.synchronize(); start=time.monotonic()
        for _ in range(5): model(xb)
        torch.cuda.synchronize(); forward_s=(time.monotonic()-start)/20
    passed=(records[0]['after']<0.5 and records[0]['after']<=.9*records[0]['before']
        and records[1]['after']<records[1]['before'] and parity and prefix_equal
        and all(r['adapter_changed'] and r['phi_changed'] and r['lambda_changed'] for r in records)
        and float(prediction.std())>1e-6)
    out={'passed':bool(passed),'records':records,'checkpoint_predictions_bitwise':parity,
        'prefix_reuse_bitwise':prefix_equal,'map_std':float(prediction.std()),'map_min':float(prediction.min()),
        'map_max':float(prediction.max()),'seconds':time.monotonic()-total_start,
        'forward_per_image_s':forward_s,'cuda_peak':torch.cuda.max_memory_allocated(),
        'rss_peak':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'source_lock_sha256':sha(OUT/'SOURCE_LOCK.json'),'device':torch.cuda.get_device_name(0)}
    write_json(OUT/'SMOKE.json',out)
    log('SMOKE',json.dumps(out))
    if not passed: return
    del state,model,loras,detector,features,data
    torch.cuda.empty_cache()
    from exps.calm02.native import native_ap
    rng=np.random.default_rng(SEED)
    scores=rng.random((8,784)).astype('float32')
    masks=[rng.random((1024,1024))>.98 for _ in range(8)]
    start=time.monotonic()
    native_ap(scores,[(1024,1024)]*8,masks)
    metric_s=(time.monotonic()-start)/8
    training=1.25*sum(r['seconds'] for r in records)/330*(2*15*CFG.steps)
    infer=1.25*2*15*1725*forward_s
    metrics=1.25*2*sum(range(1,16))*(1725/15)*metric_s
    resources={'training_estimate_s':training,'inference_estimate_s':infer,'metrics_estimate_s':metrics,
        'total_estimate_s':training+infer+metrics,'both_arms_budget_s':BUDGET,
        'passed':training+infer+metrics<BUDGET and out['cuda_peak']<=5*2**30 and out['rss_peak']<8.5e9,
        'metric_s_per_1024_image':metric_s,'gpu':out['device']}
    write_json(OUT/'RESOURCE.json',resources)
    log('RESOURCE',json.dumps(resources))


def train():
    configure(); verify_source()
    K.require(read(OUT/'SMOKE.json')['passed'],'detector smoke failed')
    K.require(read(OUT/'RESOURCE.json')['passed'],'insufficient both-arm budget')
    K.require(not (OUT/'TRAIN_LOCK.json').exists(),'training already started; preserve partial states')
    write_json(OUT/'TRAIN_LOCK.json',{'source_lock_sha256':sha(OUT/'SOURCE_LOCK.json'),
        'smoke_sha256':sha(OUT/'SMOKE.json'),'resource_sha256':sha(OUT/'RESOURCE.json')})
    encoder=make_encoder()
    # Frozen extraction only; learner will open each task tensor at arrival.
    for c in ORDER: cache(c,'train',encoder)
    del encoder
    torch.cuda.empty_cache()
    start=time.monotonic()
    base_saved=None
    for arm in ARMS:
        model,loras,base=model_new(base_saved)
        if base_saved is None:
            save(OUT/'BASE.pt',base); base_saved=base
        fixed=K.init_state(loras,SEED); phi=fixed
        lam=torch.zeros(len(loras),dtype=torch.float64); archive=[]
        for stage,c in enumerate(ORDER):
            path=OUT/'features'/f'{c}_train.pt'
            K.require(sha(path)==read(path.with_suffix('.json'))['sha256'],'training input hash')
            data=torch.load(path,weights_only=True,mmap=True); roles=assert_cache(data,c)
            K.set_state(loras,phi if arm=='nested' else fixed)
            before=normal_eval(model,data['features'],roles['EVAL'])
            lam,trace,wall=fit_task(model,loras,data['features'],roles,stage,lam,arm=='nested')
            endpoint=K.get_state(loras); archive.append(endpoint)
            after=normal_eval(model,data['features'],roles['EVAL'])
            if arm=='nested': phi=K.reptile(phi,endpoint,CFG.beta)
            prefix=OUT/arm; prefix.mkdir(exist_ok=True)
            target=prefix/f'stage{stage+1:02}.pt'
            save(target,{'archive':archive,'phi':phi,'lam':lam,'cfg':CFG.dict(),'stage':stage+1,
                'arm':arm,'base_sha256':sha(OUT/'BASE.pt')})
            rec={'stage':stage+1,'category':c,'before':before,'after':after,'seconds':wall,
                'state_sha256':sha(target),'feature_sha256':sha(path),'trace':trace,
                'adapter_bytes':K.tensor_state_bytes(endpoint),'archive_bytes':sum(K.tensor_state_bytes(a) for a in archive),
                'policy_bytes':K.slow_bytes(lam,phi),'cuda_peak':torch.cuda.max_memory_allocated(),
                'rss_peak':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
            write_json(prefix/f'stage{stage+1:02}.json',rec)
            log('TRAIN',arm,stage+1,c,'loss',round(before,6),'->',round(after,6),'seconds',round(wall,2))
            del data
            cap(start)
        del model,loras,archive
        torch.cuda.empty_cache()
    manifest={str(p.relative_to(OUT)):sha(p) for arm in ARMS for p in sorted((OUT/arm).glob('stage*.pt'))}
    write_json(OUT/'TRAIN_COMPLETE.json',{'checkpoints':manifest,'base_sha256':sha(OUT/'BASE.pt'),
        'train_seconds':time.monotonic()-start,'train_lock_sha256':sha(OUT/'TRAIN_LOCK.json')})


def verify_training():
    verify_source()
    complete=read(OUT/'TRAIN_COMPLETE.json')
    K.require(len(complete['checkpoints'])==30,'both arms must complete before test access')
    K.require(sha(OUT/'BASE.pt')==complete['base_sha256'],'base changed')
    for p,h in complete['checkpoints'].items(): K.require(sha(OUT/p)==h,'stage checkpoint changed')
    return complete


def score():
    configure(); complete=verify_training()
    K.require(not (OUT/'SCORES_COMPLETE.json').exists(),'scores complete; preserve')
    encoder=make_encoder()
    for c in ORDER: cache(c,'test',encoder)
    del encoder
    torch.cuda.empty_cache()
    base=torch.load(OUT/'BASE.pt',weights_only=True)
    start=time.monotonic()
    timing={}
    for arm in ARMS:
        final=torch.load(OUT/arm/'stage15.pt',weights_only=True)
        # Prove final archives equal each acquisition payload before prefix reuse.
        for t in range(15):
            stage=torch.load(OUT/arm/f'stage{t+1:02}.pt',weights_only=True)
            K.require(state_equal(stage['archive'][t],final['archive'][t]),'expert function changed after acquisition')
        model,loras,_=model_new(base); model.eval()
        arm_time=time.monotonic()
        for c in ORDER:
            data=torch.load(OUT/'features'/f'{c}_test.pt',weights_only=True,mmap=True)
            f=data['features']; n=len(f); all_scores=torch.empty(15,n,28,28)
            for t,state in enumerate(final['archive']):
                K.set_state(loras,state)
                with torch.no_grad():
                    for j in range(0,n,4):
                        en,de=model(f[j:j+4].cuda())
                        all_scores[t,j:j+4]=(1-F.cosine_similarity(en[0].float(),de[0].float(),dim=1)).clamp_min(0).cpu()
            prefixes=prefix_grids(all_scores)
            target=OUT/arm/f'scores_{c}.pt'
            save(target,{'prefix_grids':prefixes,'rows':data['rows'],'base_sha256':complete['base_sha256']})
            log('SCORE',arm,c,n)
            cap(start,BUDGET-complete['train_seconds'])
        timing[arm]=time.monotonic()-arm_time
        del model,loras
    write_json(OUT/'SCORES_COMPLETE.json',{'seconds':time.monotonic()-start,'arms_seconds':timing,
        'files':{str(p.relative_to(OUT)):sha(p) for a in ARMS for p in (OUT/a).glob('scores_*.pt')}})


def metrics():
    configure(gpu=False); complete=verify_training()
    scored=read(OUT/'SCORES_COMPLETE.json')
    from exps.calm02.native import native_ap,native_mask,native_hw
    from sklearn.metrics import roc_auc_score
    from exps.nl_sgm_01a.am03_eval import bootstrap_delta
    start=time.monotonic(); summaries={}
    for arm in ARMS:
        cells={}
        for i,c in enumerate(ORDER):
            file=OUT/arm/f'scores_{c}.pt'
            K.require(sha(file)==scored['files'][str(file.relative_to(OUT))],'score hash')
            payload=torch.load(file,weights_only=True)
            rows=payload['rows']; grids=payload['prefix_grids']
            hws=[native_hw(DATA/r['relative_path']) for r in rows]
            masks=[native_mask(DATA/r['mask_path'] if r['mask_path'] else None,hw) for r,hw in zip(rows,hws)]
            labels=[r['label'] for r in rows]
            for t in range(i,15):
                cellpath=OUT/arm/f'metric_{t+1:02}_{c}.json'
                if cellpath.exists():
                    cells[(t,c)]=read(cellpath); continue
                grid=grids[t].reshape(len(rows),784).numpy()
                ap=native_ap(grid,hws,masks)['ap']
                auc=float(roc_auc_score(labels,grid.max(axis=1)))
                row={'stage':t+1,'category':c,'pixel_ap':ap,'image_auroc':auc,'scores_sha256':sha(file)}
                write_json(cellpath,row); cells[(t,c)]=row
                log('METRIC',arm,t+1,c,'AP',round(ap,6),'AUROC',round(auc,6))
                cap(start,BUDGET-complete['train_seconds']-scored['seconds'])
        per={c:{'acquisition':cells[(i,c)],'final':cells[(14,c)]} for i,c in enumerate(ORDER)}
        summaries[arm]={'per_category':per,
          'final_pixel_ap':float(np.mean([per[c]['final']['pixel_ap'] for c in ORDER])),
          'final_image_auroc':float(np.mean([per[c]['final']['image_auroc'] for c in ORDER])),
          'acquisition_pixel_ap':float(np.mean([per[c]['acquisition']['pixel_ap'] for c in ORDER])),
          'pixel_fm_eq8':fm_eq8(cells,ORDER,'pixel_ap'),'image_fm_eq8':fm_eq8(cells,ORDER,'image_auroc')}
    a,b=summaries['nested'],summaries['fixed']
    ci=bootstrap_delta([a['per_category'][c]['final']['pixel_ap'] for c in ORDER],
        [b['per_category'][c]['final']['pixel_ap'] for c in ORDER],seed=20261117,B=2000)
    added=(ci['point']>=.01 and ci['lo']>0 and a['final_image_auroc']>=b['final_image_auroc']-.01
        and a['acquisition_pixel_ap']>=b['acquisition_pixel_ap']-.01
        and a['pixel_fm_eq8']['macro']<=b['pixel_fm_eq8']['macro']+.01)
    out={'arms':summaries,'paired_pixel_ap_delta':ci,'NL_ADDED_VALUE':bool(added),
       'NAR_DETECTOR_FUNCTIONAL':True,'NAR_FULL_15_COMPLETED':True,
       'COMPETITIVE_WITH_LITERATURE':'UNRESOLVED_UNMATCHED_PROTOCOLS',
       'cost':{'train_seconds':complete['train_seconds'],'score_seconds':scored['seconds'],
         'metrics_seconds':time.monotonic()-start,'artifact_disk_bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
         'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},
       'exposure':'historically exposed official MVTec AD; exploratory only'}
    write_json(OUT/'RESULT.json',out)
    log('RESULT',json.dumps({a:{k:v for k,v in s.items() if k!='per_category'} for a,s in summaries.items()}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('lock','smoke','train','score','metrics'))
    p.add_argument('--owner-authorized',action='store_true')
    a=p.parse_args()
    if a.phase in ('smoke','train','score'):
        K.require(a.owner_authorized,'owner authorization required for GPU benchmark')
    globals()[a.phase]()


if __name__=='__main__': main()
