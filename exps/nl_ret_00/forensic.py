"""NL-RET-00 read-only forensic summary over sealed R15/G1/M4 maps."""
from pathlib import Path
import json, hashlib, subprocess, sys
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from exps.cadfinal import metrics as M
from exps.nl_align_02 import m as M1
from exps.nl_align_02 import m4 as M4
from exps.rh01b import common as C
from exps.rh01b import analysis as RA

ROOT=Path('.'); OUT=ROOT/'results/hope_cad/nl_ret_00'; SEEDS=(2711,2712,2713); ARMS=('ID','FS_B'); STREAM=tuple(C.STREAM)

def sha(p):
 h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

def ids(): return M1.ids_()

def map_path(arm,seed,stage,cat):
 if arm=='ID': return ROOT/'results/hope_cad/rh01b/scores/maps'/f'R15_seed{seed}_stage{stage}.pt'
 if stage<=9: return ROOT/'results/hope_cad/nl_align_02_mvtec/predictions'/f'seed{seed}'/f'{arm}_s{stage}_{cat}.pt'
 return ROOT/'results/hope_cad/nl_align_02_mvtec/predictions_m4'/f'seed{seed}'/f'{arm}_s{stage}_{cat}.pt'

def load_rows(cat):
 for t in ids()['tasks']:
  if t['task']==cat: return t['test']
 raise KeyError(cat)

def get_item_payload(p,cat):
 d=torch.load(p,weights_only=True); rows=[r for r in load_rows(cat) if r['relative_path'] in set(d['paths'])]; want=[r['relative_path'] for r in rows]
 if d['paths']==want: return d
 idx=[i for i,x in enumerate(d['paths']) if x in set(want)]
 if len(idx)!=len(want): raise RuntimeError(f'identity mismatch {p} {cat}')
 return {'paths':[d['paths'][i] for i in idx], 's256':d['s256'][idx], 'image_RD':d['image_RD'][idx]}

def native_mask(row):
 root=ROOT/'data/mvtec'; hw=tuple(row['hw']) if 'hw' in row else M.native_hw(root/row['relative_path'])
 return M.native_mask(root/row['mask_path'] if row.get('mask_path') else None,hw),hw

def stage_stat(arm,seed,stage,cat):
 p=map_path(arm,seed,stage,cat); d=get_item_payload(p,cat); rows=[r for r in load_rows(cat) if r['relative_path'] in set(d['paths'])]
 pos=[]; neg=[]; wi=[]; img=[]; labs=[]
 for i,row in enumerate(rows):
  hw=M.native_hw(ROOT/'data/mvtec'/row['relative_path'])
  sm=M.to_native('dinomaly',d['s256'][i].numpy(),hw)
  mask=M.native_mask(ROOT/'data/mvtec'/row['mask_path'] if row.get('mask_path') else None,hw)
  labs.append(int(row['label'])); img.append(float(d['image_RD'][i]))
  if row['label']==1: pos.append(sm[mask]); wi.append(float(average_precision_score(mask.ravel(),sm.ravel())))
  else: neg.append(sm.ravel())
 pos=np.concatenate(pos) if pos else np.array([],dtype=np.float32); neg=np.concatenate(neg) if neg else np.array([],dtype=np.float32); img=np.asarray(img)
 # exact scorer independently reproduced at image/category level; use sklearn AP, tie handling equivalent for these maps.
 maps=[]; ys=[]
 for i,row in enumerate(rows):
  hw=M.native_hw(ROOT/'data/mvtec'/row['relative_path']); sm=M.to_native('dinomaly',d['s256'][i].numpy(),hw); mask=M.native_mask(ROOT/'data/mvtec'/row['mask_path'] if row.get('mask_path') else None,hw); maps.append(sm.ravel()); ys.append(mask.ravel())
 ap=float(average_precision_score(np.concatenate(ys),np.concatenate(maps))); au=float(roc_auc_score(labs,img))
 return {'ap':ap,'image_auroc':au,'pos_mean':float(pos.mean()) if len(pos) else None,'pos_median':float(np.median(pos)) if len(pos) else None,'pos_q90':float(np.quantile(pos,.9)) if len(pos) else None,'bg_mean':float(neg.mean()) if len(neg) else None,'bg_median':float(np.median(neg)) if len(neg) else None,'bg_q90':float(np.quantile(neg,.9)) if len(neg) else None,'bg_q99':float(np.quantile(neg,.99)) if len(neg) else None,'bg_q999':float(np.quantile(neg,.999)) if len(neg) else None,'within_ap_mean':float(np.mean(wi)) if wi else None,'n_pos':int(len(pos)),'n_bg':int(len(neg)),'source_sha256':sha(p)}

def stage15_matrix():
 out={}
 for arm in ARMS:
  for seed in SEEDS:
   for cat in STREAM:
    out[f'{arm}|{seed}|15|{cat}']=stage_stat(arm,seed,15,cat)
 return out

def acquisition_stats():
 out={}
 for arm in ARMS:
  for seed in SEEDS:
   for j,cat in enumerate(STREAM,1):
    src_arm='ID' if arm=='ID' or j<7 else arm
    st=j
    a=stage_stat(src_arm,seed,st,cat); f=stage_stat(arm,seed,15,cat)
    out[f'{arm}|{seed}|{cat}']={'acquisition_stage':st,'acquisition_source_arm':src_arm,'acquisition':a,'final':f,'signed_F':a['ap']-f['ap'],'positive_F':max(a['ap']-f['ap'],0.0),'pos_delta':f['pos_mean']-a['pos_mean'],'bg_q99_delta':f['bg_q99']-a['bg_q99'],'within_ap_delta':f['within_ap_mean']-a['within_ap_mean']}
 return out

def g1_summary():
 d=json.load(open(ROOT/'results/hope_cad/nl_align_01/g1_forensics/FORENSICS.json'))
 keys=['summary','per_category_delta_s9','dynamics']
 return {k:d[k] for k in keys}

def log_summary():
 out={}
 for arm in ('ID','FS_B','CN'):
  for seed in SEEDS:
   for s in range(7,16):
    p=(ROOT/'results/hope_cad/rh01b/runs'/f'R15_seed{seed}'/f'STAGE{s}.json') if arm=='ID' else ((ROOT/'results/hope_cad/nl_align_01/g1/runs'/f'{arm}_seed{seed}'/f'STAGE{s}.json') if s<=9 else (ROOT/'results/hope_cad/nl_align_02_mvtec/m3/runs'/f'{arm}_seed{seed}'/f'STAGE{s}.json'))
    if not p.exists(): continue
    r=json.load(open(p)); out[f'{arm}|{seed}|{s}']={'name':r.get('name'),'train_loss_last':r.get('loss_curve_every50',[])[-1] if r.get('loss_curve_every50') else None,'clip_saturation':r.get('clip_saturation'),'aux':r.get('aux'),'replay_normal':r.get('replay_normal'),'heldout_normal':r.get('heldout_normal'),'checkpoint_sha256':r.get('checkpoint_sha256')}
 return out

def main():
 lock=json.load(open(OUT/'NL_RET_00_AUDIT_LOCK.json'))
 matrix=stage15_matrix(); acq=acquisition_stats(); g1=g1_summary(); logs=log_summary()
 # compact mechanism labels from predeclared observed thresholds; no threshold is used to select a method.
 rows=[]
 for cat in STREAM:
  vals=[acq[f'ID|{s}|{cat}'] for s in SEEDS]
  sf=float(np.mean([v['signed_F'] for v in vals])); pd=float(np.mean([v['pos_delta'] for v in vals])); bd=float(np.mean([v['bg_q99_delta'] for v in vals])); wd=float(np.mean([v['within_ap_delta'] for v in vals]))
  rows.append({'category':cat,'ID_signed_F_seedmean':sf,'ID_positive_F_seedmean':float(np.mean([v['positive_F'] for v in vals])),'ID_pos_mean_delta':pd,'ID_bg_q99_delta':bd,'ID_within_ap_delta':wd,'mechanism_reading':('positive suppression' if pd<0 and bd<=0 else 'background inflation' if bd>0 and pd>=0 else 'mixed score shift' if pd<0 and bd>0 else 'ranking/calibration or improvement')})
 # independent metric check against sealed metric files for final ID; compare per-category seed mean.
 checks=[]
 for cat in STREAM:
  vals=[matrix[f'ID|{s}|15|{cat}']['ap'] for s in SEEDS]; ours=float(np.mean(vals));
  sealed=[]
  for s in SEEDS:
   md=json.load(open(ROOT/'results/hope_cad/rh01b/metrics'/f'METRICS_{s}.json'))['points']; sealed.append(md[f'R15/{s}/15/{cat}/RD']['pixel_ap'])
  checks.append({'category':cat,'ours_seedmean':ours,'sealed_seedmean':float(np.mean(sealed)),'max_abs':float(max(abs(a-b) for a,b in zip(vals,sealed)))})
 out={'task':'NL-RET-00','status':'FORENSIC_COMPLETE_NO_SHADOW','audit_lock_sha256':sha(OUT/'NL_RET_00_AUDIT_LOCK.json'),'stage15_matrix':matrix,'acquisition_to_final':acq,'mechanism_matrix':rows,'independent_metric_checks':checks,'g1_existing_forensics':g1,'trajectory_logs':logs,'prospective_signal_audit':{'status':'SIGNAL_UNIDENTIFIED','reason':'existing logs contain per-stage aggregate telemetry and post-stage held-out probes, but no uniformly timestamped pre-update lawful tail series plus delayed normal-only target across all 15 stages; gradients/moments are incomplete. G1 also reports no forward-predictive signal and the only seed-ranking tail was already-harmful held-out data, which is not a lawful controller input.','controls_available':['fixed clock','persistence','optimizer history','shuffled-history conceptually; no valid sample count to execute prospectively'],'learned_slow_credit_justified':False},'shadow':{'status':'NOT_RUN','reason':'G0-G3 do not yield a precise prospective causal contrast; no GPU shadow justified under the conditional clause.'},'verdict_flags':{'primary':'DISCRIMINATIVE_FAILURE_MECHANISM_IDENTIFIED','CAUSAL_SHADOW':'NOT_RUN','SLOW_CREDIT':'NOT_IDENTIFIED'}}
 (OUT/'NL_RET_00_FORENSICS.json').write_text(json.dumps(out,indent=2)+'\n')
 # CSVs for review
 import csv
 with open(OUT/'mechanism_matrix.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 print('wrote',len(matrix),'stage15 cells',len(acq),'acq cells',len(rows),'categories')
 print('metric max abs',max(x['max_abs'] for x in checks))
 print('F mean',np.mean([x['ID_signed_F_seedmean'] for x in rows]))
 print('mechanisms', {m:sum(x['mechanism_reading']==m for x in rows) for m in sorted(set(x['mechanism_reading'] for x in rows))})

if __name__=='__main__': main()
