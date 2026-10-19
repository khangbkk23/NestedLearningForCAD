"""Posthoc attribution/uncertainty only. No alternate scorer or parameter search."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn.functional import interpolate
from threadpoolctl import threadpool_limits
from sklearn.metrics import average_precision_score
from exps.ecad.r1.distance import configure
from exps.ecad.r1.train_normal import ROOT,file_sha
from exps.task_p.p1_data import native_mask
from exps.task_p.p1_bootstrap_bounds import CorrectedBoundedPixelBootstrap
from .mechanism import CCNM
from .evaluate_anomaly import PRIOR,write

EVAL=ROOT/'results/hope_cad/ecad/ccnm/ccnm_03_anomaly_dev_20261018'
CATS=('bottle','carpet','hazelnut')


def bounded(positive,total,weights):
    obj=CorrectedBoundedPixelBootstrap.__new__(CorrectedBoundedPixelBootstrap)
    obj.positive,obj.total=positive,total
    low,high=[],[]
    for start in range(0,len(weights),16):
        lo,hi=obj.bounds(weights[start:start+16])
        low.extend(lo);high.extend(hi)
    return np.array(low),np.array(high)


def validate_bounds():
    # Exhaust actual image multiplicities on a synthetic tied fixture.
    scores=np.array([[.1,.1,.9,.9],[.1,.1,.9,.9]])
    labels=np.array([[0,1,0,1],[0,0,1,1]])
    pos=np.array([[1.,1.],[2.,0.]])
    total=np.full((2,2),2.)
    for weights in (np.array([[1.,1.]]),np.array([[2.,0.]]),np.array([[0.,2.]])):
        lo,hi=bounded(pos,total,weights)
        ids=np.repeat(np.arange(2),weights[0].astype(int))
        exact=average_precision_score(labels[ids].ravel(),scores[ids].ravel())
        assert lo[0]<=exact<=hi[0]


def attribution(protocol,device):
    feature_pins={r['relative_path']:r for r in protocol['feature_pins']}
    fit=json.loads((PRIOR/'contract.json').read_text())
    result=[]
    for boundary in (200,300,350):
        cp=next(c for c in protocol['checkpoints'] if c['boundary']==boundary)
        path=PRIOR/f'ccnm_{boundary:03}.pt';assert file_sha(path)==cp['sha256']
        m=CCNM.from_state_dict(torch.load(path,weights_only=True),device=device)
        rows=json.loads((EVAL/f'predictions_{boundary}.json').read_text())['evaluations']
        z=np.load(EVAL/f'predictions_{boundary}.npz');s,ids=z['scores'],z['ids']
        examples=[]
        for i,row in enumerate(rows):
            if row['category']!='carpet':continue
            pin=feature_pins[row['relative_path']];f=ROOT/pin['cache_path']
            assert file_sha(f)==pin['file_sha256']
            x=torch.load(f,weights_only=True).to(device)
            assert hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()==pin['feature_sha256']
            c=m.context(x)
            selected=torch.from_numpy(ids[i,1]).to(device)
            ix=torch.searchsorted(m.model.serials,selected)
            assert torch.equal(m.model.serials[ix],selected)
            geom=(x.double()-m.model.bank[ix].double()).square().sum(1)
            context=(c.double()-m.contexts[ix].double()).square().sum(1)*m.alpha.double().square()
            direct=(geom+context).sqrt()
            saved_cc=torch.from_numpy(s[i,1]).to(device)
            cc_error=(direct.float()-saved_cc).abs()
            cc_bad=cc_error>2e-6+2e-6*direct.float().abs()
            rawsel=torch.from_numpy(ids[i,0]).to(device)
            rawix=torch.searchsorted(m.model.serials,rawsel)
            rawgeom=(x.double()-m.model.bank[rawix].double()).square().sum(1)
            raw_reference=rawgeom.sqrt().float()
            raw_error=(raw_reference-torch.from_numpy(s[i,0]).to(device)).abs()
            raw_bad=raw_error>2e-6+2e-6*raw_reference.abs()
            rp,cpix=int(s[i,0].argmax()),int(s[i,1].argmax())
            geometric_cost=float(geom[cpix]-rawgeom[cpix])
            context_cost=float(context[cpix])
            previous=next(e for e in fit['updates'] if e['event']==int(selected[cpix])//784)
            raworigin=fit['updates'][int(rawsel[cpix])//784]
            ex={'path':row['relative_path'],'label':row['label'],
                'raw_max':float(s[i,0,rp]),'context_max':float(s[i,1,cpix]),
                'raw_at_context_max':float(s[i,0,cpix]),
                'raw_argmax_rc':list(divmod(rp,28)),'context_argmax_rc':list(divmod(cpix,28)),
                'selected_patch_geometric_distance':float(geom[cpix].sqrt()),
                'selected_context_penalty_distance':float(context[cpix].sqrt()),
                'geometric_reselection_cost_squared':geometric_cost,
                'context_penalty_squared':context_cost,
                'context_fraction_of_squared_increase':context_cost/(geometric_cost+context_cost),
                'context_nn_origin_category':previous['category'],
                'raw_nn_origin_category_at_context_max':raworigin['category'],
                'nn_reselected_at_context_max':bool(selected[cpix]!=rawsel[cpix]),
                'normal_tail_p99':float(np.quantile(s[i,1],.99)),
                'context_uplift_patch_mean':float((s[i,1]-s[i,0]).mean())}
            ex['numerical_audit']={'context_max_absolute_error':float(cc_error.max()),
                'raw_max_absolute_error':float(raw_error.max()),
                'context_outside_registered_tolerance':int(cc_bad.sum()),
                'raw_outside_registered_tolerance':int(raw_bad.sum()),
                'context_error_at_context_peak':float(cc_error[cpix]),
                'query_feature_norm_max':float(x.norm(dim=1).max())}
            examples.append(ex)
        norm=[r for r in examples if r['label']==0];anom=[r for r in examples if r['label']==1]
        wrong,recovered=[],[]
        for a in anom:
            for n in norm:
                if a['raw_max']>n['raw_max'] and a['context_max']<n['context_max']:
                    wrong.append({'normal':n['path'],'anomaly':a['path'],
                                  'raw_margin':a['raw_max']-n['raw_max'],
                                  'context_margin':a['context_max']-n['context_max']})
                if a['raw_max']<n['raw_max'] and a['context_max']>n['context_max']:
                    recovered.append({'normal':n['path'],'anomaly':a['path']})
        result.append({'boundary':boundary,'examples':examples,'new_wrong_pairs':wrong,'recovered_pairs':recovered,
            'normal_summary':{'mean_raw_max':float(np.mean([r['raw_max'] for r in norm])),
                'mean_context_max':float(np.mean([r['context_max'] for r in norm])),
                'argmax_changed_images':sum(r['raw_argmax_rc']!=r['context_argmax_rc'] for r in norm),
                'reselected_at_max_images':sum(r['nn_reselected_at_context_max'] for r in norm),
                'mean_context_share_of_squared_increase':float(np.mean([r['context_fraction_of_squared_increase'] for r in norm])),
                'mean_max_uplift':float(np.mean([r['context_max']-r['raw_max'] for r in norm])),
                'mean_anomaly_max_uplift':float(np.mean([r['context_max']-r['raw_max'] for r in anom]))}})
        assert file_sha(path)==cp['sha256']
        print(json.dumps({'attribution_boundary':boundary,'new_wrong_pairs':len(wrong),'recovered':len(recovered),
                          'normal_summary':result[-1]['normal_summary']}),flush=True)
        z.close()
    return result


def uncertainty(protocol,device):
    rows=json.loads((EVAL/'predictions_350.json').read_text())['evaluations']
    z=np.load(EVAL/'predictions_350.npz');scores=z['scores'][:,:2]
    metrics=json.loads((EVAL/'metrics.json').read_text())
    per_category=[];macro_low=np.zeros(1000);macro_hi=np.zeros(1000);macro_img=np.zeros(1000)
    for ci,category in enumerate(CATS):
        ix=[i for i,r in enumerate(rows) if r['category']==category]
        admitted=[rows[i] for i in ix];y=np.array([r['label'] for r in admitted])
        rng=np.random.default_rng(40401+ci)
        weights=np.zeros((1000,20),dtype=np.int64)
        for label in (0,1):
            group=np.flatnonzero(y==label)
            draws=rng.choice(group,size=(1000,len(group)),replace=True)
            for b in range(1000):weights[b]+=np.bincount(draws[b],minlength=20)
        images=scores[ix].max(2)
        auc=[]
        for arm in range(2):
            pos=images[y==1,arm];neg=images[y==0,arm]
            pairs=(pos[:,None]>neg[None]).astype(float)+.5*(pos[:,None]==neg[None])
            auc.append(np.einsum('bi,ij,bj->b',weights[:,y==1],pairs,weights[:,y==0])/100)
        delta_auc=auc[1]-auc[0]
        hw=tuple(admitted[0]['hw'])
        patch=torch.from_numpy(scores[ix]).to(device)
        maps=interpolate(patch.permute(1,0,2).reshape(40,1,28,28),size=hw,mode='bilinear',
                         align_corners=False,antialias=False).reshape(2,20,*hw)
        truth=torch.stack([torch.from_numpy(native_mask(ROOT/'data/mvtec',r)) for r in admitted]).to(device)
        lo,hi=maps.min(),maps.max();bins=65536
        offset=torch.arange(20,device=device,dtype=torch.int64)[:,None,None]*bins
        allbounds=[];pointbounds=[]
        for arm in range(2):
            index=((maps[arm]-lo)/(hi-lo)*(bins-1)).long().clamp_(0,bins-1)+offset
            total=torch.bincount(index.flatten(),minlength=20*bins).reshape(20,bins).flip(1).cpu().numpy().astype(float)
            positive=torch.bincount(index[truth],minlength=20*bins).reshape(20,bins).flip(1).cpu().numpy().astype(float)
            del index
            low,high=bounded(positive,total,weights)
            pointlo,pointhi=bounded(positive,total,np.ones((1,20)))
            point=next(r['native_pixel_ap'] for r in metrics['groups'] if
                r['boundary']==350 and r['category']==category and r['arm']==('raw','context')[arm])
            assert pointlo[0]<=point<=pointhi[0],(category,arm,pointlo,point,pointhi)
            allbounds.append((low,high));pointbounds.append([float(pointlo[0]),float(pointhi[0])])
        delta_lo=allbounds[1][0]-allbounds[0][1]
        delta_hi=allbounds[1][1]-allbounds[0][0]
        raw=next(r for r in metrics['groups'] if r['boundary']==350 and r['category']==category and r['arm']=='raw')
        cc=next(r for r in metrics['groups'] if r['boundary']==350 and r['category']==category and r['arm']=='context')
        row={'category':category,'replicates':1000,'seed':40401+ci,
            'native_AP_delta':cc['native_pixel_ap']-raw['native_pixel_ap'],
            'native_AP_delta_conservative_95':[float(np.quantile(delta_lo,.025)),float(np.quantile(delta_hi,.975))],
            'image_AUROC_delta':cc['image_auroc']-raw['image_auroc'],
            'image_AUROC_delta_95':np.quantile(delta_auc,[.025,.975]).tolist(),
            'point_AP_bounds_raw_context':pointbounds,
            'max_per_arm_bootstrap_AP_bound_width':float(max(np.max(high-low) for low,high in allbounds))}
        per_category.append(row);macro_low+=delta_lo/3;macro_hi+=delta_hi/3;macro_img+=delta_auc/3
        print(json.dumps(row),flush=True)
        del maps,truth,patch
    z.close()
    return {'posthoc':True,'unit':'whole image, stratified by category and label; paired across arms',
        'replicates':1000,'bins':65536,'per_category':per_category,
        'macro_native_AP_delta':float(np.mean([r['native_AP_delta'] for r in per_category])),
        'macro_native_AP_delta_conservative_95':[float(np.quantile(macro_low,.025)),float(np.quantile(macro_hi,.975))],
        'macro_image_AUROC_delta':float(np.mean([r['image_AUROC_delta'] for r in per_category])),
        'macro_image_AUROC_delta_95':np.quantile(macro_img,[.025,.975]).tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda',choices=('cuda','cpu'));args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    configure();torch.set_num_threads(2);threadpool_limits(limits=2)
    start=time.time();validate_bounds()
    protocol=json.loads((EVAL/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert file_sha(ROOT/name)==digest
    expected=json.loads((EVAL/'protocol.sha256.json').read_text())['sha256']
    assert file_sha(EVAL/'protocol.json')==expected
    prediction=json.loads((EVAL/'predictions_complete.json').read_text())
    for name,digest in prediction['files'].items():assert file_sha(EVAL/name)==digest
    write(args.out/'analysis_protocol.json',{'source_sha256':file_sha(__file__),
        'parent_protocol_sha256':expected,'bootstrap_replicates':1000,'seeds':[40401,40402,40403],
        'bins':65536,'posthoc':True,'parameter_changes':False,'model_predictions_replaced':False})
    root=attribution(protocol,args.device);write(args.out/'attribution.json',root)
    intervals=uncertainty(protocol,args.device);write(args.out/'uncertainty.json',intervals)
    write(args.out/'status.json',{'status':'DONE','elapsed_seconds':time.time()-start,
        'direct_FP64_selected_scores':'REGISTERED_TOLERANCE_VIOLATION_FOUND_AND_RECORDED',
        'synthetic_bootstrap_bounds':'PASS',
        'original_sources_predictions_checkpoints':'UNCHANGED','exposure':'consumed development only'})
    print(json.dumps({'status':'DONE','seconds':time.time()-start,'uncertainty':intervals}),flush=True)


if __name__=='__main__':main()
