"""Artifact verification: all image metrics and two full native sklearn oracles."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score
from torch.nn.functional import interpolate
from exps.ecad.r1.distance import configure
from exps.ecad.r1.train_normal import ROOT,file_sha
from exps.task_p.p1_data import native_mask
from .evaluate_anomaly import ARMS,PRIOR,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda');args=p.parse_args()
    configure();torch.set_num_threads(2)
    protocol=json.loads((args.run/'protocol.json').read_text())
    expected=json.loads((args.run/'protocol.sha256.json').read_text())['sha256']
    assert file_sha(args.run/'protocol.json')==expected
    for name,digest in protocol['source_sha256'].items():assert file_sha(ROOT/name)==digest
    prediction=json.loads((args.run/'predictions_complete.json').read_text())
    assert prediction['images']==180 and prediction['numeric_masks_read']==0
    for name,digest in prediction['files'].items():assert file_sha(args.run/name)==digest
    metrics=json.loads((args.run/'metrics.json').read_text())
    assert metrics['status']=='DONE' and len(metrics['groups'])==36
    access=json.loads((args.run/'access.json').read_text())
    permitted={r['relative_path'] for r in protocol['evaluations']}
    assert len(access)==180 and {r['relative_path'] for r in access}==permitted
    cache={}
    checks=[]
    for boundary in (100,200,300,350):
        cp=next(c for c in protocol['checkpoints'] if c['boundary']==boundary)
        path=PRIOR/f'ccnm_{boundary:03}.pt';assert file_sha(path)==cp['sha256']
        bank_ids=set(torch.load(path,weights_only=True)['allocator']['serials'].tolist())
        records=json.loads((args.run/f'predictions_{boundary}.json').read_text())['evaluations']
        with np.load(args.run/f'predictions_{boundary}.npz') as data:
            scores,ids=data['scores'],data['ids']
            assert scores.shape==ids.shape==(len(records),4,784)
            assert np.isfinite(scores).all() and set(ids.flat)<=bank_ids
            cache[boundary]=(records,scores.copy())
            for row in [r for r in metrics['groups'] if r['boundary']==boundary]:
                ix=[i for i,r in enumerate(records) if r['category']==row['category']]
                y=np.array([records[i]['label'] for i in ix])
                s=scores[ix,ARMS.index(row['arm'])].max(1)
                assert abs(roc_auc_score(y,s)-row['image_auroc'])<1e-12
                checks.append({'boundary':boundary,'category':row['category'],'arm':row['arm'],'image_auroc_sklearn':'PASS'})
    native=[]
    for boundary,category,arm in ((100,'bottle','raw'),(350,'hazelnut','context')):
        records,scores=cache[boundary]
        ix=[i for i,r in enumerate(records) if r['category']==category]
        rows=[records[i] for i in ix];hw=tuple(rows[0]['hw'])
        patch=torch.from_numpy(scores[ix,ARMS.index(arm)]).to(args.device)
        values=interpolate(patch.reshape(20,1,28,28),size=hw,mode='bilinear',
            align_corners=False,antialias=False)[:,0].cpu().numpy().reshape(-1)
        truth=np.stack([native_mask(ROOT/'data/mvtec',r) for r in rows]).reshape(-1)
        ap=float(average_precision_score(truth,values));auc=float(roc_auc_score(truth,values))
        row=next(r for r in metrics['groups'] if (r['boundary'],r['category'],r['arm'])==(boundary,category,arm))
        assert abs(ap-row['native_pixel_ap'])<1e-10
        assert abs(auc-row['native_pixel_auroc'])<1e-10
        native.append({'boundary':boundary,'category':category,'arm':arm,'pixels':len(values),
            'native_pixel_ap_sklearn':ap,'native_pixel_auroc_sklearn':auc,
            'ap_absolute_error':abs(ap-row['native_pixel_ap']),
            'auroc_absolute_error':abs(auc-row['native_pixel_auroc'])})
        del values,truth,patch
    result={'status':'PASS','image_groups':checks,'full_native_oracle_groups':native,
        'prediction_hashes_verified':True,'source_hashes_verified':True,
        'checkpoint_hashes_verified':True,'whitelist_only_access':True,
        'auditor_source_sha256':file_sha(__file__)}
    write(args.run/'verification.json',result)
    print(json.dumps({'status':'PASS','image_metric_groups':len(checks),'full_native_oracles':native},indent=2))


if __name__=='__main__':main()
