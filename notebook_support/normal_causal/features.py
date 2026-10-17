"""Capability-scoped frozen RGB features. Task identity never enters the predictor."""
from pathlib import Path
import torch
from dataset.benchmark_protocol_v2 import HashedNormalDataset
from dataset.hope_mask_views import patch_masks,mask_normalized_rgb
from models.hope_cad_v2.teacher_head import frozen_teacher
from models.hope_cad_v2.state import tensor_digest,finite_tensors
from training.hope_runtime_v4 import atomic_torch_save
from .contract import digest,atomic_json

class NormalFeatures:
    def __init__(self,root,extractor,cache,receipts,device):
        self.root,self.extractor,self.cache,self.receipts,self.device=Path(root),extractor,Path(cache),Path(receipts),device
        self.masks=patch_masks(device=device)
    def row(self,row,allowed):
        if row!=allowed.get(row['relative_path']) or row['split']!='train' or '/train/good/' not in row['relative_path']:raise ValueError('forbidden normal feature read')
        # Hash current image even when a detached cache exists.
        image=HashedNormalDataset(self.root,[row],224,[.485,.456,.406],[.229,.224,.225],True)[0]['images'].unsqueeze(0).to(self.device)
        key=digest(row);path=self.cache/(key+'.pt');receipt=self.receipts/(key+'.json')
        if path.exists():
            tensors=torch.load(path,map_location='cpu',weights_only=True)
            old=__import__('json').loads(receipt.read_text()) if receipt.exists() else None
            if old is None or old.get('row')!=row or old['tensor_sha256']!=tensor_digest((tensors,)):raise ValueError('normal cache identity changed')
        else:
            with torch.no_grad():
                tensors={'teacher':frozen_teacher(self.extractor.extract_patch_features(image)).cpu()}
                for view in range(4):tensors[f'masked_{view}']=self.extractor.extract_patch_features(mask_normalized_rgb(image,view)).cpu()
            finite_tensors(tensors.values(),'normal features')
            payload=dict(row=row,tensor_sha256=tensor_digest((tensors,)),source='frozen ViT-B/8 block8; no task ID/model feedback')
            if receipt.exists() and __import__('json').loads(receipt.read_text())!=payload:raise ValueError('regenerated normal features differ')
            atomic_json(payload,receipt)
            atomic_torch_save(tensors,path)
        return [(tensors[f'masked_{v}'].to(self.device),tensors['teacher'].to(self.device),self.masks[v]) for v in range(4)]

@torch.no_grad()
def encode_rows(model,provider,rows,allowed):
    return [[(model.encode(f).detach(),target,mask) for f,target,mask in provider.row(row,allowed)] for row in rows]
