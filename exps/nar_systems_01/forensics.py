"""Descriptive read-only score suppression and reversible storage measurements."""
import hashlib
import json
from pathlib import Path
import resource
import time
import zlib
import numpy as np
import torch
from exps.calm02.native import native_hw,native_mask
from exps.task_p.p1_readout import native_map
from exps.nar_pilot_00 import full as R
from exps.nar_systems_01.audit import OUT


def lock():
    R.verify_source()
    R.write_json(OUT/'FORENSIC_MAPPING.json',{
      'source_sha256':R.sha(Path(__file__)),'result_sha256':R.sha(R.OUT/'RESULT.json'),
      'mapping':'All15 nested categories; compare acquisition prefix and final prefix on unchanged native bilinear grids. Summarize mean scores/suppression for positive versus background pixels; no thresholds or AP re-selection.',
      'exposure':'Posthoc descriptive analysis on exposed official test; no learning/model selection.',
      'features_context':'Historical NORM00 INC all-normal-support category-conditional results only; not a matched task-free baseline.',
      'storage':'Streaming zlib level1 roundtrip of encoder/base/final-state files; count compressed bytes without replacing files.',
      'caps':'CPU only, 300s wall, 8.5GB RSS; no training or GPU model evaluation.'})


def run():
    torch.set_num_threads(2)
    lk=json.loads((OUT/'FORENSIC_MAPPING.json').read_text()); assert R.sha(Path(__file__))==lk['source_sha256']
    assert R.sha(R.OUT/'RESULT.json')==lk['result_sha256']; R.verify_source()
    start=time.monotonic(); result=json.loads((R.OUT/'RESULT.json').read_text())
    norm=json.loads((R.ROOT/'results/hope_cad/norm_00/RESULT.json').read_text())
    data={}
    for i,c in enumerate(R.ORDER):
        score=torch.load(R.OUT/'nested'/f'scores_{c}.pt',weights_only=True)
        grids=score['prefix_grids']; early=grids[i]; late=grids[-1]
        sums={k:0. for k in ('early_pos','late_pos','early_background','late_background')}
        counts={'positive':0,'background':0}; occupied=[]
        for j,row in enumerate(score['rows']):
            hw=native_hw(R.DATA/row['relative_path']); mask=native_mask(R.DATA/row['mask_path'] if row['mask_path'] else None,hw)
            a=native_map(early[j].flatten(),hw); b=native_map(late[j].flatten(),hw)
            P=int(mask.sum()); N=mask.size-P; counts['positive']+=P; counts['background']+=N
            sums['early_pos']+=float(a[mask].sum(dtype=np.float64)); sums['late_pos']+=float(b[mask].sum(dtype=np.float64))
            sums['early_background']+=float(a[~mask].sum(dtype=np.float64)); sums['late_background']+=float(b[~mask].sum(dtype=np.float64))
            if P: occupied.append(P/mask.size*784)
        means={k:v/counts['positive' if k.endswith('pos') else 'background'] for k,v in sums.items()}
        metrics=result['arms']['nested']['per_category'][c]
        fit=json.loads((R.OUT/'nested'/f'stage{i+1:02}.json').read_text())
        data[c]={'metrics':metrics,'mean_native_scores':means,'pixel_counts':counts,
          'positive_suppression':means['early_pos']-means['late_pos'],
          'background_suppression':means['early_background']-means['late_background'],
          'changed_grid_fraction':float((early>late).float().mean()),
          'median_defect_area_in_28grid_cells':float(np.median(occupied)),
          'normal_eval_after':fit['after'],'norm00_INC_reference':norm['per_category'][c]['INC']}
        R.log('FORENSIC',c); assert time.monotonic()-start<300
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8.5e9
    compression={}
    for label,path in [('encoder',R.CHECKPOINT),('decoder',R.OUT/'BASE.pt'),('continual',R.OUT/'nested/stage15.pt')]:
        a=hashlib.sha256(); b=hashlib.sha256(); enc=zlib.compressobj(level=1); dec=zlib.decompressobj(); size=0; t=time.monotonic()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1<<20),b''):
                a.update(chunk); packed=enc.compress(chunk); size+=len(packed); b.update(dec.decompress(packed))
        packed=enc.flush(); size+=len(packed); b.update(dec.decompress(packed)); b.update(dec.flush())
        assert a.digest()==b.digest() and dec.eof
        compression[label]={'source_bytes':path.stat().st_size,'zlib1_bytes':size,'roundtrip_sha256':a.hexdigest(),'seconds':time.monotonic()-t}
        R.log('LOSSLESS',label,compression[label]); assert time.monotonic()-start<300
    R.write_json(OUT/'FORENSICS.json',{'lock_sha256':R.sha(OUT/'FORENSIC_MAPPING.json'),'categories':data,
       'compression':compression,'seconds':time.monotonic()-start,
       'limits':'Prefix minima cannot recover individual expert maps. Cross-method NORM00 data budget and task-ID assumptions differ. Means/suppression are descriptive, not anomaly objectives or proof of root cause.'})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('lock','run'));a=p.parse_args()
    lock() if a.phase=='lock' else run()
