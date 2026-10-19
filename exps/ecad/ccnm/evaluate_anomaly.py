"""Locked CCNM anomaly readout measurement on the exposed P1 DEV whitelist.

No training, extraction, split discovery, sealed reader, or parameter search.
Predictions are locked before native annotations are opened numerically.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from torch.nn.functional import interpolate
from exps.ecad.r1.distance import Euclidean, configure
from exps.ecad.r1.train_normal import ROOT, CP_SHA, file_sha
from exps.task_p.p1_data import source_path, native_mask
from .mechanism import CCNM
from .test_mechanism import equal_state

P1 = ROOT/'results/hope_cad/task_p/p1'
PRIOR = ROOT/'results/hope_cad/ecad/ccnm/ccnm_01_8011_20261018'
P1_SHA = '11c1cc03b20cb1f79a6e8b7e472bb6f44758b43046b61043368ea11b4ce1e08f'
CATEGORIES = ('bottle','carpet','hazelnut')
ARMS = ('raw','context','shuffled','routing5')


def write(path, value):
    text = json.dumps(value,indent=2,sort_keys=True,allow_nan=False)
    with path.open('x') as f:f.write(text)


def admitted_rows(protocol):
    """Read only the historical admitted field; no sealed membership discovery."""
    rows = protocol['evaluations']
    if len(rows)!=60 or len({r['relative_path'] for r in rows})!=60:
        raise ValueError('incorrect fixed development whitelist')
    if len({r['sha256'] for r in rows})!=60:
        raise ValueError('duplicate development content')
    for c in CATEGORIES:
        for label in (0,1):
            if sum(r['category']==c and r['label']==label for r in rows)!=10:
                raise ValueError('changed development class allocation')
    for r in rows:
        p=Path(r['relative_path']).parts
        if (len(p)!=4 or p[0]!=r['category'] or p[1]!='test'
                or r['role']!='development_eval' or r['label']!=int(p[2]!='good')):
            raise ValueError('invalid admitted development record')
    return rows


def cache_identity(row):
    key=hashlib.sha256(row['relative_path'].encode()).hexdigest()
    path=P1/'evaluation_features'/f'{key}.pt'
    meta=json.loads(path.with_suffix('.json').read_text())
    if (meta['source']!=row or meta['protocol_sha256']!=P1_SHA
            or meta['checkpoint_sha256']!=CP_SHA or file_sha(path)!=meta['file_sha256']):
        raise ValueError('evaluation feature provenance mismatch')
    return path,meta


class FrozenReadout:
    def __init__(self, model):self.model=model

    @torch.no_grad()
    def score(self, features):
        m=self.model
        values=[m.score(features,mode=mode) for mode in ARMS[:3]]
        origins,counts=torch.unique_consecutive(m.model.serials//m.model.patches,return_counts=True)
        starts=torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1]))
        context=m.context(features)
        distances=(m.contexts[starts]-context).square().sum(1)
        chosen=origins[torch.argsort(distances,stable=True)[:min(5,len(origins))]]
        keep=torch.isin(m.model.serials//m.model.patches,chosen)
        bank,serials=m.model.bank[keep],m.model.serials[keep]
        d=Euclidean(m.model.chunk).matrix(features,bank)
        score,ix=d.min(1)
        values.append({'scores':score,'ids':serials[ix]})
        scores=torch.stack([v['scores'] for v in values])
        ids=torch.stack([v['ids'] for v in values])
        if not bool(torch.isfinite(scores).all()):raise ValueError('nonfinite patch scores')
        return scores,ids


@torch.no_grad()
def ranking_metrics(scores,labels,scan=1048576):
    """Exact AP/ROC at grouped FP32 score ties, bounded cumulative scan memory."""
    scores,labels=scores.reshape(-1).float(),labels.reshape(-1).bool()
    if len(scores)!=len(labels) or len(scores)==0 or scan<1:
        raise ValueError('invalid ranking input')
    if not bool(torch.isfinite(scores).all()):raise ValueError('nonfinite ranking scores')
    values,order=torch.sort(scores,descending=True,stable=True)
    positive=labels[order].to(torch.int64).cumsum(0)
    del order
    p=int(positive[-1]);n=len(scores)-p
    if not p or not n:raise ValueError('ranking requires both classes')
    ap=torch.zeros((),dtype=torch.float64,device=scores.device)
    area=ap.clone();last_tp=ap.clone();last_fp=ap.clone()
    for start in range(0,len(values),scan):
        end=min(start+scan,len(values));local=values[start:end]
        ends=torch.ones(len(local),dtype=torch.bool,device=scores.device)
        ends[:-1]=local[:-1]!=local[1:]
        if end<len(values):ends[-1]=local[-1]!=values[end]
        if not bool(ends.any()):continue
        rank=torch.arange(start+1,end+1,device=scores.device,dtype=torch.int64)[ends].double()
        tp=positive[start:end][ends].double();fp=rank-tp
        previous_tp=torch.cat((last_tp[None],tp[:-1]))
        previous_fp=torch.cat((last_fp[None],fp[:-1]))
        ap+=((tp-previous_tp)*tp/rank).sum()
        area+=((fp-previous_fp)*(tp+previous_tp)*.5).sum()
        last_tp,last_fp=tp[-1],fp[-1]
    return {'ap':float(ap/p),'auroc':float(area/(p*n)),
            'prevalence':p/len(scores),'positive':p,'total':len(scores)}


def synthetic_gate(device):
    from sklearn.metrics import average_precision_score,roc_auc_score
    g=torch.Generator().manual_seed(30301)
    for case in range(4):
        s=torch.randint(0,7,(53,),generator=g).float()
        y=torch.randint(0,2,(53,),generator=g).bool()
        for chunk in (1,7,100):
            m=ranking_metrics(s.to(device),y.to(device),scan=chunk)
            assert abs(m['ap']-average_precision_score(y.numpy(),s.numpy()))<1e-12
            assert abs(m['auroc']-roc_auc_score(y.numpy(),s.numpy()))<1e-12
    m=CCNM(width=4,capacity=31,dim=8,patches=16,candidates=5,device=device)
    for t in range(8):m.update(torch.randn(16,8,generator=g).to(device)+t,event=t)
    query=torch.randn(16,8,generator=g).to(device)
    before=m.state_dict();reader=FrozenReadout(m);s,ids=reader.score(query)
    for i,mode in enumerate(ARMS[:3]):
        v=m.score(query,mode=mode)
        assert torch.equal(s[i],v['scores']) and torch.equal(ids[i],v['ids'])
    # Independent direct-FP64 routing5 geometry and stable prototype selection.
    origin=m.model.serials//16;unique=torch.unique_consecutive(origin)
    ctx=torch.stack([m.contexts[origin==o][0] for o in unique])
    chosen=unique[torch.argsort((ctx-m.context(query)).square().sum(1),stable=True)[:5]]
    keep=torch.isin(origin,chosen)
    direct=(query[:,None].double()-m.model.bank[keep][None].double()).square().sum(2).sqrt()
    torch.testing.assert_close(s[3],direct.min(1).values.float(),rtol=2e-6,atol=2e-6)
    assert torch.equal(ids[3],m.model.serials[keep][direct.argmin(1)])
    assert equal_state(before,m.state_dict())
    patch=torch.randn(1,1,28,28,generator=g)
    cpu=interpolate(patch,size=(37,43),mode='bilinear',align_corners=False,antialias=False)
    actual=interpolate(patch.to(device),size=(37,43),mode='bilinear',align_corners=False,antialias=False).cpu()
    torch.testing.assert_close(cpu,actual,atol=2e-6,rtol=2e-6)
    return {'status':'PASS','exact_AP_AUROC_ties_against_sklearn':True,
            'routing5_direct_fp64':True,'original_readout_parity':True,'read_only':True,
            'native_geometry_parity':True,'device':device}


def analyze(args,contract):
    """Only this phase reads numeric masks; predictions are already immutable."""
    prediction_pin=json.loads((args.out/'predictions_complete.json').read_text())
    for name,digest in prediction_pin['files'].items():
        assert file_sha(args.out/name)==digest
    dataset=(ROOT/'data/mvtec').resolve()
    mask_cache={}
    for row in contract['evaluations']:
        mask_cache[row['relative_path']]=native_mask(dataset,row)
    result={'status':'DONE','exposure':'DEVELOPMENT EVIDENCE ONLY; consumed P1 cohort; not independent confirmation',
        'arms':ARMS,'groups':[],'macro':[],'retention':[],'learner_updates':0,
        'threshold_fitting':False,'scientific_parameters_changed':False,'confidence_intervals':None}
    with (args.out/'metrics_progress.jsonl').open('x') as log:
        for boundary in (100,200,300,350):
            meta=json.loads((args.out/f'predictions_{boundary}.json').read_text())
            saved=np.load(args.out/f'predictions_{boundary}.npz')
            rows=meta['evaluations'];scores=saved['scores']
            for category in CATEGORIES[:min(boundary//100,3)]:
                ix=[i for i,r in enumerate(rows) if r['category']==category]
                group_rows=[rows[i] for i in ix]
                hw=tuple(group_rows[0]['hw'])
                assert len(ix)==20 and all(tuple(r['hw'])==hw for r in group_rows)
                native_labels=torch.stack([torch.from_numpy(mask_cache[r['relative_path']]) for r in group_rows]).to(args.device)
                labels=torch.tensor([r['label'] for r in group_rows],device=args.device,dtype=torch.bool)
                patches=torch.from_numpy(scores[ix]).to(args.device)
                # [images,arms,784] -> [arms,images,H,W], common native rule.
                maps=interpolate(patches.permute(1,0,2).reshape(len(ARMS)*20,1,28,28),size=hw,
                    mode='bilinear',align_corners=False,antialias=False).reshape(len(ARMS),20,*hw)
                for ai,arm in enumerate(ARMS):
                    metrics=ranking_metrics(maps[ai],native_labels)
                    image_scores=patches[:,ai].max(1).values
                    image=ranking_metrics(image_scores,labels,scan=64)
                    normal=image_scores[~labels].cpu().numpy()
                    row={'boundary':boundary,'category':category,'arm':arm,
                        'native_pixel_ap':metrics['ap'],'native_pixel_auroc':metrics['auroc'],
                        'image_auroc':image['auroc'],'prevalence':metrics['prevalence'],
                        'native_pixels':metrics['total'],'anomaly_pixels':metrics['positive'],
                        'normal_image_max_mean':float(normal.mean()),'normal_image_max_p95':float(np.quantile(normal,.95)),
                        'normal_native_mean':float(maps[ai,~labels].double().mean()),
                        'defect_native_mean':float(maps[ai][native_labels].double().mean())}
                    result['groups'].append(row);log.write(json.dumps(row)+'\n');log.flush()
                    print(json.dumps({k:row[k] for k in ('boundary','category','arm','image_auroc','native_pixel_ap')}),flush=True)
                del maps,native_labels,patches
            saved.close()
            for arm in ARMS:
                rows0=[r for r in result['groups'] if r['boundary']==boundary and r['arm']==arm]
                result['macro'].append({'boundary':boundary,'arm':arm,'categories':len(rows0),
                    'image_auroc':float(np.mean([r['image_auroc'] for r in rows0])),
                    'native_pixel_ap':float(np.mean([r['native_pixel_ap'] for r in rows0]))})
    for r in result['groups']:
        acquired=next(x for x in result['groups'] if x['category']==r['category'] and x['arm']==r['arm'])
        raw=next(x for x in result['groups'] if x['category']==r['category'] and x['boundary']==r['boundary'] and x['arm']=='raw')
        result['retention'].append({'boundary':r['boundary'],'category':r['category'],'arm':r['arm'],
            'acquisition_boundary':acquired['boundary'],
            'native_pixel_ap_loss':acquired['native_pixel_ap']-r['native_pixel_ap'],
            'image_auroc_loss':acquired['image_auroc']-r['image_auroc'],
            'native_pixel_ap_minus_raw':r['native_pixel_ap']-raw['native_pixel_ap'],
            'image_auroc_minus_raw':r['image_auroc']-raw['image_auroc']})
    write(args.out/'metrics.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    configure();torch.set_num_threads(2)
    started=time.time()
    def timeout(*_):raise TimeoutError('30-minute execution safety cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
    status={'status':'IN_PROGRESS','started_epoch':started,'cost_preflight':False,'formal_report':False}
    try:
        assert file_sha(P1/'protocol_resolved.json')==P1_SHA
        old=json.loads((P1/'protocol_resolved.json').read_text())
        rows=admitted_rows(old)
        prior=json.loads((PRIOR/'summary.json').read_text())
        pins=[]
        for r in rows:
            path,meta=cache_identity(r)
            assert file_sha(source_path(ROOT/'data/mvtec',r['relative_path']))==r['sha256']
            pins.append({'relative_path':r['relative_path'],'file_sha256':meta['file_sha256'],
                'feature_sha256':meta['feature_sha256'],'cache_path':str(path.relative_to(ROOT))})
        contract={'kind':'fixed readout anomaly DEVELOPMENT evaluation; no scientific tuning',
            'exposure':'Previously consumed P1/MVTec v1 development; NOT independent',
            'evaluations':rows,'feature_pins':pins,'p1_protocol_sha256':P1_SHA,'encoder_sha256':CP_SHA,
            'checkpoints':prior['checkpoints'],'arms':ARMS,'routing_k':5,'shuffle_seed':52011,
            'map':'FP32 bilinear native H,W;align_corners=False;antialias=False;no smoothing',
            'image_score':'max784 patch distances','pixel_metric':'exact non-interpolated AP; FP32 ties grouped',
            'source_sha256':{str(f.relative_to(ROOT)):file_sha(f) for f in sorted(Path(__file__).parent.glob('*.py'))}}
        write(args.out/'protocol.json',contract)
        write(args.out/'protocol.sha256.json',{'sha256':file_sha(args.out/'protocol.json')})
        write(args.out/'correctness_gate.json',synthetic_gate(args.device))
        features={}
        for row,pin in zip(rows,pins):
            x=torch.load(ROOT/pin['cache_path'],weights_only=True,map_location='cpu')
            if x.shape!=(784,768) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()):
                raise ValueError('incorrect frozen features')
            assert hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()==pin['feature_sha256']
            features[row['relative_path']]=x
        access=[]
        for boundary in (100,200,300,350):
            cp=next(c for c in prior['checkpoints'] if c['boundary']==boundary)
            path=PRIOR/f'ccnm_{boundary:03}.pt';assert file_sha(path)==cp['sha256']
            m=CCNM.from_state_dict(torch.load(path,weights_only=True),device=args.device)
            before=m.state_dict();reader=FrozenReadout(m)
            eligible=[r for r in rows if r['category'] in CATEGORIES[:min(boundary//100,3)]]
            scores,ids=[],[]
            for r in eligible:
                values,selected=reader.score(features[r['relative_path']].to(args.device))
                scores.append(values.cpu().numpy());ids.append(selected.cpu().numpy())
                access.append({'boundary':boundary,'relative_path':r['relative_path'],'role':'development_eval'})
            assert equal_state(before,m.state_dict()) and file_sha(path)==cp['sha256']
            np.savez_compressed(args.out/f'predictions_{boundary}.npz',scores=np.stack(scores),ids=np.stack(ids))
            write(args.out/f'predictions_{boundary}.json',{'boundary':boundary,'arms':ARMS,'evaluations':eligible,
                'checkpoint_sha256':cp['sha256'],'state_read_only':True})
            print(json.dumps({'predictions_complete':boundary,'images':len(eligible)}),flush=True)
        write(args.out/'access.json',access)
        write(args.out/'predictions_complete.json',{'images':len(access),'numeric_masks_read':0,
            'files':{f.name:file_sha(f) for f in args.out.glob('predictions_*') if f.is_file()}})
        result=analyze(args,contract)
        status.update({'status':'DONE','evaluation_images':len(access),'native_metric_groups':len(result['groups']),
            'exposure':result['exposure'],'elapsed_seconds':time.time()-started})
    except (Exception,KeyboardInterrupt) as e:
        status.update({'status':'FAILED','failure_type':type(e).__name__,'failure':str(e),'elapsed_seconds':time.time()-started})
        raise
    finally:
        signal.alarm(0);write(args.out/'execution_status.json',status)


if __name__=='__main__':main()
