"""Standalone v5 RGB adapter; old frozen registry is deliberately not edited."""
from pathlib import Path
import torch
from torch.nn import functional as F
from models.benchmark_adapter_base_v1 import BenchmarkMethodAdapter
from models.feature_extractors.cadic_vit_v1 import CADICViTConfig,CADICViTFeatureExtractor
from .session import SupportSession
from .contract import CONTRACT,DESIGN_ID,file_sha,design_digest

class SupportBenchmarkAdapter(BenchmarkMethodAdapter):
    def __init__(self,checkpoint,*,device='cpu',seed=0,mode='NESTED',extractor=None,session=None):
        self.device=torch.device(device);self.session=session or SupportSession(mode,device=device,seed=seed)
        if extractor is None:
            if file_sha(checkpoint)!=CONTRACT['backbone_sha256']:raise ValueError('wrong frozen ViT checkpoint')
            extractor=CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=str(checkpoint),checkpoint_identity='timm/vit_base_patch8_224.augreg_in21k'),self.device)
        self.extractor=extractor.eval().requires_grad_(False)
    def fit_task(self,task_id,task_name,train_loader):
        records=[]
        for batch in train_loader:
            if 'labels' not in batch or bool((batch['labels']!=0).any()):raise ValueError('normal-only training required')
            if 'relative_path' not in batch:raise ValueError('stable training identities required')
            for image,identity in zip(batch['images'],batch['relative_path']):
                raw=self.extractor.extract_patch_features(image.unsqueeze(0).to(self.device)).detach()
                records.append(self.session.observe(raw,str(identity)))
        # Task labels are returned as evaluator bookkeeping only; never forwarded to session.
        return dict(task_id=int(task_id),task_name=str(task_name),samples=len(records),event_cursor=self.session.cursor,completed_labels=self.session.counters['labels_completed'])
    @torch.no_grad()
    def score_batch(self,batch):
        images=batch['images'].to(self.device);raw=self.extractor.extract_patch_features(images).detach();scores=[];maps=[]
        grid=CONTRACT['geometry']['grid']
        for feature in raw:
            pixel,_,image=self.session.score(feature.unsqueeze(0));scores.append(image[0])
            maps.append(F.interpolate(pixel.reshape(1,1,*grid),size=tuple(images.shape[-2:]),mode='bilinear',align_corners=False)[0,0])
        return dict(image_scores=torch.stack(scores).cpu(),anomaly_maps=torch.stack(maps).cpu())
    def state_dict(self):return dict(schema='hope_support_rgb_adapter_v5',design_digest=design_digest(),session=self.session.state_dict())
    def load_state_dict(self,state):
        if state['schema']!='hope_support_rgb_adapter_v5' or state['design_digest']!=design_digest():raise ValueError('RGB adapter identity changed')
        self.session.load_state_dict(state['session'])
    def memory_stats(self):return self.session.memory_profile()
    def method_metadata(self):return dict(design_id=DESIGN_ID,adapter='hope_support_rgb_adapter_v5',raw_geometry='ViT-B/8 block8 FP32 no extra normalization',update_unit='image',score='CADIC Eq8/full Eq9',policy=self.session.mode,reportable=False)
    def close(self):
        close=getattr(self.extractor,'close',None)
        if callable(close):close()
