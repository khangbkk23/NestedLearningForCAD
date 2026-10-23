"""Separate hash-bound normal feature delivery and read-only official evaluator data."""
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from hope_cad.data.prefix import HashedNormalDataset,NonzeroMaskDataset,safe_path,file_hash
from hope_cad.contract import CONTRACT,digest,ROOT
from hope_cad.models.primitives.state import tensor_digest,finite_tensors
from hope_cad.runtime.system import atomic_json
from hope_cad.runtime.measurement import timed

class NormalFeatures:
    """No unbounded raw cache; regenerate and verify receipts on interrupted replay."""
    def __init__(self,root,extractor,receipts,device):
        self.root,self.extractor,self.receipts,self.device=Path(root),extractor,Path(receipts),device
    @timed('normal_input_and_features')
    def get(self,row):
        if row.get('split')!='train' or '/train/good/' not in row['relative_path']:raise ValueError('Only permitted train/good rows enter learning')
        g=CONTRACT['geometry']
        image=HashedNormalDataset(self.root,[row],224,g['mean'],g['std'],True)[0]['images']
        with torch.no_grad():raw=self.extractor.extract_patch_features(image[None].to(self.device)).detach().contiguous()
        if raw.shape!=(1,784,768) or raw.dtype!=torch.float32:raise ValueError('Raw feature geometry changed')
        finite_tensors((raw,),'raw benchmark features')
        receipt=dict(row=row,backbone_sha256=CONTRACT['backbone_sha256'],tensor_sha256=tensor_digest(({'patches':raw},)))
        path=self.receipts/(digest(row)+'.json')
        if path.exists() and json.loads(path.read_text())!=receipt:raise ValueError('Regenerated normal features changed')
        atomic_json(receipt,path);return raw

def test_manifest(root,tasks):
    """Evaluator metadata only. Never passed to the learner or used for selection."""
    root=Path(root);rows=[];qa=[]
    for ti,task in enumerate(tasks):
        folder=root/task/'test'
        if not folder.is_dir():raise FileNotFoundError(folder)
        for defect in sorted(p for p in folder.iterdir() if p.is_dir()):
            for image in sorted(defect.glob('*.png')):
                relative=image.relative_to(root).as_posix();safe_path(root,relative)
                label=int(defect.name!='good');mask=f'{task}/ground_truth/{defect.name}/{image.stem}_mask.png' if label else None
                row=dict(relative_path=relative,label=label,mask_path=mask,task_id=ti,task_name=task,image_sha256=file_hash(image),mask_sha256=None)
                if label:
                    mp=safe_path(root,mask);row['mask_sha256']=file_hash(mp)
                    from hope_cad.data.masks import inspect_mask
                    qa.append(inspect_mask(root,row,224))
                rows.append(row)
        selected=[r for r in rows if r['task_name']==task]
        if {r['label'] for r in selected}!={0,1}:raise ValueError('Each task requires both official image labels')
    return dict(schema='support_v5_1_test_manifest_v1',task_order=tasks,rows=rows,mask_qa=qa,digest=digest(dict(task_order=tasks,rows=rows,mask_qa=qa)))

class EvaluationFeatures:
    def __init__(self,root,extractor,device):self.root,self.extractor,self.device=Path(root),extractor,device
    @timed('evaluation_input_and_features')
    def get(self,row):
        if '/test/' not in row['relative_path']:raise ValueError('Official evaluation row required')
        if file_hash(safe_path(self.root,row['relative_path']))!=row['image_sha256']:raise ValueError('Test image bytes changed')
        if row['label'] and file_hash(safe_path(self.root,row['mask_path']))!=row['mask_sha256']:raise ValueError('GT bytes changed')
        g=CONTRACT['geometry']
        sample=NonzeroMaskDataset(self.root,[row],224,g['mean'],g['std'],False)[0]
        with torch.no_grad():raw=self.extractor.extract_patch_features(sample['images'][None].to(self.device)).detach()
        if raw.shape!=(1,784,768) or raw.dtype!=torch.float32:raise ValueError('Evaluation raw geometry changed')
        finite_tensors((raw,),'evaluation features')
        return raw,sample['masks'].cpu().numpy().astype(bool)
