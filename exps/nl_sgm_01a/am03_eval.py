"""NL-SGM-01A-AM03 real-anomaly evaluation on preserved AM02 support checkpoints.

This module never trains or changes AM02 artifacts. It verifies AM03_LOCK.json, loads frozen
ViT-B/8 test features, scores exact category-specific SC-SN-01 support distances, and computes
native Pixel AP/Image AUROC plus registered diagnostics.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a import exact as X

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/hope_cad/nl_sgm_01a_am03'
AM02=ROOT/'results/hope_cad/nl_sgm_01a_am02'
LOCKP=OUT/'AM03_LOCK.json'
ARMS=('C1_FPS_FP32','C2_FPS_PQ','C4_SAMEID_PQ')
ORDER=('bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper')

def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def req(x,msg):
 if not x: raise RuntimeError(msg)

def lock():
 d=json.loads(LOCKP.read_text()); req(d['schema']=='nl_sgm_01a_am03_lock_v1','bad AM03 lock schema'); req(d['protocol_sha256']==sha(OUT/'AM03_PROTOCOL.json'),'protocol changed after lock'); return d

def load_features(c):
 d=torch.load(ROOT/f'results/hope_cad/calm02/features/mvtec/{c}_test.pt',map_location='cpu',weights_only=False)
 req(d['features'].dtype==torch.float32 and tuple(d['features'].shape[1:])==(784,768),f'bad test features {c}')
 return d['features'].contiguous()

def load_masks(c):
 d=json.loads((OUT/'DATASET_MANIFEST.json').read_text())['categories'][c]
 labels=np.array([int(x['label']) for x in d['images']],dtype=np.int8)
 masks=[]
 for x in d['images']:
  p=ROOT/'data/mvtec'/x['relative_path']
  if x['label']:
   mp=ROOT/'data/mvtec'/x['mask_path']
   with Image.open(mp) as im: m=np.asarray(im.convert('L'))>0
   req(tuple(m.shape)==tuple(x['hw']) and int(m.sum())==int(x['mask_positive_pixels']),f'mask mismatch {mp}')
  else: m=np.zeros(tuple(x['hw']),dtype=bool)
  masks.append(m)
 return labels,masks,d['images']

def load_support(arm,t):
 if arm=='C1_FPS_FP32':
  d=torch.load(AM02/f'stages/C1_FPS_FP32/ckpt_t{t:02d}.pt',map_location='cpu',weights_only=False)
  x=d['x'].float(); owner=d['owner'].long(); gids=d['gids'].long()
 elif arm=='C2_FPS_PQ':
  d=torch.load(AM02/f'stages/C2_FPS_PQ/ckpt_t{t:02d}.pt',map_location='cpu',weights_only=False)
  cb=d['codebooks']; cb=cb[0] if cb.ndim==4 else cb
  x=K.decode(d['codes'],cb); owner=d['owner'].long(); gids=d['gids'].long()
 else:
  d=torch.load(AM02/f'stages/c4/ckpt_t{t:02d}.pt',map_location='cpu',weights_only=False)
  cb=torch.load(AM02/'stages/c4/C0.pt',map_location='cpu',weights_only=False)['codebook']
  x=K.decode(d['codes'],cb); owner=d['owner'].long(); gids=d['gids'].long()
 req(x.dtype==torch.float32 and x.ndim==2 and x.shape[1]==768,'bad decoded support')
 out={}
 for i,c in enumerate(ORDER):
  mask=owner==i
  if bool(mask.any()):
   gg=gids[mask]; xx=x[mask]; order=torch.argsort(gg)
   out[c]=X.RowSet(xx[order].contiguous(),gg[order])
 return out,int(x.numel()*x.element_size())

def native_maps(dist,hw):
 return F.interpolate(dist.float().reshape(len(dist),1,28,28),size=tuple(hw),mode='bilinear',align_corners=False,antialias=False)[:,0].numpy()

def eval_cell(features,labels,masks,rows,hw,batch_images=4):
 image_scores=[]; pixel_maps=[]; normal_means=[]; normal_q99=[]; anomaly_means=[]; anomaly_bg_means=[]
 t0=time.time()
 for a in range(0,len(features),batch_images):
  q=features[a:a+batch_images].reshape(-1,768)
  d,_=X.nn(q,rows)
  d=d.reshape(-1,784)
  maps=native_maps(d,hw)
  pixel_maps.append(maps)
  image_scores.extend(maps.reshape(len(maps),-1).max(1).tolist())
  for j,m in enumerate(maps):
   mask=masks[a+j]
   normal_means.append(float(m[~mask].mean()))
   normal_q99.append(float(np.quantile(m[~mask],.99)))
   anomaly_means.append(float(m[mask].mean()) if mask.any() else float('nan'))
   anomaly_bg_means.append(float(m[~mask].mean()) if mask.any() else float('nan'))
 maps=np.concatenate(pixel_maps,axis=0)
 flat_s=maps.reshape(-1); flat_y=np.concatenate([m.reshape(-1) for m in masks]).astype(np.uint8)
 req(np.isfinite(flat_s).all(),'nonfinite native scores')
 ap=float(average_precision_score(flat_y,flat_s)); auc=float(roc_auc_score(labels,np.asarray(image_scores))) if len(np.unique(labels))==2 else float('nan')
 per_image_ap=[float(average_precision_score(m.reshape(-1).astype(np.uint8),s.reshape(-1))) if m.any() else float('nan') for s,m in zip(maps,masks)]
 return {'pixel_ap':ap,'image_auroc':auc,'n_images':len(labels),'n_anomalous':int(labels.sum()),'normal_score_mean':float(np.mean(normal_means)),'normal_score_q99_mean':float(np.mean(normal_q99)),'anomaly_score_mean':float(np.nanmean(anomaly_means)),'anomaly_background_mean':float(np.nanmean(anomaly_bg_means)),'normal_means':normal_means,'normal_q99':normal_q99,'anomaly_means':anomaly_means,'image_scores':image_scores,'labels':labels.tolist(),'per_image_pixel_ap':per_image_ap,'seconds':time.time()-t0}

def macro(per,stage,arm):
 rows=[per[f'{arm}|{stage}|{c}'] for c in ORDER[:stage+1]]
 return {'pixel_ap':float(np.mean([x['pixel_ap'] for x in rows])),'image_auroc':float(np.mean([x['image_auroc'] for x in rows])),'normal_score_q99_mean':float(np.mean([x['normal_score_q99_mean'] for x in rows if x['normal_score_q99_mean'] is not None])) if any(x['normal_score_q99_mean'] is not None for x in rows) else None,'anomaly_score_mean':float(np.mean([x['anomaly_score_mean'] for x in rows if x['anomaly_score_mean'] is not None])) if any(x['anomaly_score_mean'] is not None for x in rows) else None}

def bootstrap_delta(vals_a,vals_b,seed=20261108,B=2000):
 rng=np.random.default_rng(seed); d=np.asarray(vals_a)-np.asarray(vals_b); n=len(d); sm=np.array([d[rng.integers(0,n,n)].mean() for _ in range(B)])
 return {'point':float(d.mean()),'lo':float(np.quantile(sm,.025)),'hi':float(np.quantile(sm,.975)),'B':B,'seed':seed,'unit':'category'}

def run_full():
 lock(); torch.set_num_threads(6); torch.backends.cuda.matmul.allow_tf32=False; req(not torch.cuda.is_initialized(),'CUDA initialized')
 manifest=json.loads((OUT/'DATASET_MANIFEST.json').read_text())
 per={}; stage=[]; resources=[]; t_all=time.time(); celldir=OUT/'cells'; celldir.mkdir(exist_ok=True)
 for arm in ARMS:
  for t in range(15):
   t0=time.time(); supports,bytes_=load_support(arm,t)
   for c in ORDER[:t+1]:
    cp=celldir/arm/f'stage{t+1:02d}_{c}.json'; cp.parent.mkdir(exist_ok=True)
    if cp.exists():
     cell=json.loads(cp.read_text()); per[f'{arm}|{t}|{c}']=cell; print('RESUME',arm,t+1,c,flush=True)
    else:
     features=load_features(c); labels,masks,items=load_masks(c); hw=tuple(manifest['categories'][c]['images'][0]['hw'])
     full=eval_cell(features,labels,masks,supports[c],hw)
     cell={k:v for k,v in full.items() if k not in ('normal_means','normal_q99','anomaly_means','image_scores','labels','per_image_pixel_ap')}
     cp.write_text(json.dumps(cell,sort_keys=True)+'\n'); per[f'{arm}|{t}|{c}']=cell
     print('CELL',arm,t+1,c,'AP',round(cell['pixel_ap'],6),'AUROC',round(cell['image_auroc'],6),'sec',round(cell['seconds'],1),flush=True)
    stage.append({'arm':arm,'stage':t+1,'category':c,'pixel_ap':cell['pixel_ap'],'image_auroc':cell['image_auroc'],'n_images':cell['n_images'],'n_anomalous':cell['n_anomalous'],'normal_score_q99_mean':cell.get('normal_score_q99_mean'),'anomaly_score_mean':cell.get('anomaly_score_mean'),'anomaly_background_mean':cell.get('anomaly_background_mean'),'seconds':cell['seconds'],'support_bytes':bytes_})
   resources.append({'arm':arm,'stage':t+1,'seconds':time.time()-t0,'support_bytes':bytes_})
 # compact per-cell artifact strips large arrays to separate no file; retain image arrays in memory for comparisons, but save no maps.
 compact={k:{x:v for x,v in d.items() if x not in ('normal_means','normal_q99','anomaly_means','image_scores','labels','per_image_pixel_ap')} for k,d in per.items()}
 metrics={'schema':'nl_sgm_01a_am03_metrics_v1','lock_sha256':sha(LOCKP),'per_cell':compact,'resources':resources,'stage_macro':{}}
 for arm in ARMS:
  for t in range(15): metrics['stage_macro'][f'{arm}|{t+1}']=macro(per,t,arm)
 # acquisition/final and Eq8 data
 acq={}; fin={}; fm={}
 for arm in ARMS:
  for i,c in enumerate(ORDER):
   a=per[f'{arm}|{i}|{c}']; f=per[f'{arm}|14|{c}']; acq[f'{arm}|{c}']=a; fin[f'{arm}|{c}']=f
   if i>=14:
    fm[f'{arm}|{c}']={'pixel_best':f['pixel_ap'],'pixel_final':f['pixel_ap'],'pixel_delta':0.0,'image_best':f['image_auroc'],'image_final':f['image_auroc'],'image_delta':0.0}
   else:
    hist=[per[f'{arm}|{t}|{c}']['pixel_ap'] for t in range(i,14)]
    histi=[per[f'{arm}|{t}|{c}']['image_auroc'] for t in range(i,14)]
    fm[f'{arm}|{c}']={'pixel_best':max(hist),'pixel_final':f['pixel_ap'],'pixel_delta':max(hist)-f['pixel_ap'],'image_best':max(histi),'image_final':f['image_auroc'],'image_delta':max(histi)-f['image_auroc']}
 metrics['acquisition']={k:{x:v for x,v in d.items() if x not in ('normal_means','normal_q99','anomaly_means','image_scores','labels','per_image_pixel_ap')} for k,d in acq.items()}
 metrics['final']={k:{x:v for x,v in d.items() if x not in ('normal_means','normal_q99','anomaly_means','image_scores','labels','per_image_pixel_ap')} for k,d in fin.items()}
 metrics['forgetting']=fm
 # comparisons category-wise, final/acq/fm; category-level bootstrap
 comps=[]
 for left,right,label in [('C1_FPS_FP32','C2_FPS_PQ','C1_minus_C2'),('C1_FPS_FP32','C4_SAMEID_PQ','C1_minus_C4'),('C4_SAMEID_PQ','C2_FPS_PQ','C4_minus_C2')]:
  for metric in ('pixel_ap','image_auroc'):
   for phase,src in [('acquisition',acq),('final',fin)]:
    va=[src[f'{left}|{c}'][metric] for c in ORDER]; vb=[src[f'{right}|{c}'][metric] for c in ORDER]; ci=bootstrap_delta(va,vb); comps.append({'comparison':label,'phase':phase,'metric':metric,**ci})
   keymetric='pixel_delta' if metric=='pixel_ap' else 'image_delta'; va=[fm[f'{left}|{c}'][keymetric] for c in ORDER[:14]]; vb=[fm[f'{right}|{c}'][keymetric] for c in ORDER[:14]]; ci=bootstrap_delta(va,vb); comps.append({'comparison':label,'phase':'FM_Eq8','metric':metric,**ci})
 metrics['comparisons']=comps
 (OUT/'METRICS.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n')
 with (OUT/'STAGE_METRIC_MATRIX.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(stage[0])); w.writeheader(); w.writerows(stage)
 with (OUT/'MATCHED_COMPARISONS.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(comps[0])); w.writeheader(); w.writerows(comps)
 with (OUT/'PER_CATEGORY_TABLE.csv').open('w',newline='') as f:
  fields=['arm','category','acquisition_pixel_ap','final_pixel_ap','pixel_fm_delta','acquisition_image_auroc','final_image_auroc','image_fm_delta']
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
  for arm in ARMS:
   for c in ORDER[:14]:
    w.writerow({'arm':arm,'category':c,'acquisition_pixel_ap':acq[f'{arm}|{c}']['pixel_ap'],'final_pixel_ap':fin[f'{arm}|{c}']['pixel_ap'],'pixel_fm_delta':fm[f'{arm}|{c}']['pixel_delta'],'acquisition_image_auroc':acq[f'{arm}|{c}']['image_auroc'],'final_image_auroc':fin[f'{arm}|{c}']['image_auroc'],'image_fm_delta':fm[f'{arm}|{c}']['image_delta']})
 (OUT/'RESOURCE_LEDGER.json').write_text(json.dumps({'schema':'nl_sgm_01a_am03_resource_ledger_v1','lock_sha256':sha(LOCKP),'total_seconds':time.time()-t_all,'peak_rss_gb':None,'device':'cpu','threads':torch.get_num_threads(),'stages':resources,'notes':'Peak RSS is recorded from process resource usage by the execution wrapper; no GPU initialized.'},indent=2,sort_keys=True)+'\n')
 print('DONE total_sec',time.time()-t_all,flush=True)

def main():
 cmd=sys.argv[1] if len(sys.argv)>1 else 'full'
 if cmd=='full': run_full()
 elif cmd=='smoke':
  lock(); arm='C1_FPS_FP32'; supports,b=load_support(arm,0); c=ORDER[0]; f=load_features(c)[:2]; labels,masks,_=load_masks(c); hw=tuple(json.loads((OUT/'DATASET_MANIFEST.json').read_text())['categories'][c]['images'][0]['hw']); print(eval_cell(f,labels[:2],masks[:2],supports[c],hw,batch_images=1))
 else: raise SystemExit(cmd)
if __name__=='__main__': main()

def run_final_diagnostics():
 lock(); torch.set_num_threads(6); torch.backends.cuda.matmul.allow_tf32=False; req(not torch.cuda.is_initialized(),'CUDA initialized')
 manifest=json.loads((OUT/'DATASET_MANIFEST.json').read_text()); celldir=OUT/'cells'; t0=time.time()
 for arm in ARMS:
  supports,bytes_=load_support(arm,14)
  for c in ORDER:
   features=load_features(c); labels,masks,items=load_masks(c); hw=tuple(manifest['categories'][c]['images'][0]['hw'])
   full=eval_cell(features,labels,masks,supports[c],hw)
   cell={k:v for k,v in full.items() if k not in ('normal_means','normal_q99','anomaly_means','image_scores','labels','per_image_pixel_ap')}
   p=celldir/arm/f'stage15_{c}.json'; p.write_text(json.dumps(cell,sort_keys=True)+'\n')
   print('FINAL_DIAG',arm,c,'normal_q99',cell['normal_score_q99_mean'],'anomaly_mean',cell['anomaly_score_mean'],'sec',round(cell['seconds'],1),flush=True)
 print('FINAL_DIAG_DONE',time.time()-t0,flush=True)

if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='final_diagnostics': run_final_diagnostics()
