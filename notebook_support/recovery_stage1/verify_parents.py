"""Read-only compatibility of reviewed W6 weights and all 15 completed M checkpoints.
No dataset, ViT inference, optimizer, RNG restore, or VisA access is performed.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import zipfile
import torch
from models.hope_cad_v2.predictor import HOPEFeaturePredictorV4
from models.hope_cad_v2.visual_block import VisualBlockConfig
from training.benchmark_engine_v1 import _state_equal
from training.hope_runtime_v4 import configure_runtime
from training.hope_w7_contract import W6_MANIFEST_SHA256,MVTEC_COUNTS
from .common import W72_MANIFEST_SHA,M_RUN_DIGEST,atomic_json,source_identity


class Parent:
    def __init__(self,path,expected):
        self.z=zipfile.ZipFile(path);raw=self.z.read('artifact_manifest.json')
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('unreviewed parent manifest')
        self.files=json.loads(raw)['files']
    def read(self,name):
        raw=self.z.read(name)
        if hashlib.sha256(raw).hexdigest()!=self.files[name]:raise ValueError('parent entry changed')
        return raw
    def load(self,name):return torch.load(io.BytesIO(self.read(name)),map_location='cpu',weights_only=False)
    def close(self):self.z.close()


def verify(w6,w72,output):
    source=source_identity();a=Parent(w6,W6_MANIFEST_SHA256);b=Parent(w72,W72_MANIFEST_SHA)
    try:
        initials={};checks=[]
        for name in ('M','C0','C1','C2','C3','C4','A1'):
            initial=a.load(f'selection/{name}/deployment_initialization.pt');meta=initial['metadata']
            beta=meta['dynamics']['rates'][0]
            model=HOPEFeaturePredictorV4(VisualBlockConfig(**meta['visual_config']),method=name,beta=beta,
                selected_m_beta=beta if name=='C4' else None)
            if model.metadata()!=meta:raise ValueError('initial metadata mismatch: '+name)
            model.load_state_dict(initial['slow_and_initial_weights'],strict=True)
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise ValueError('nonfinite init')
            initials[name]=initial;checks.append(dict(method=name,strict_weights_and_metadata_load=True))
        m=initials['M'];beta=m['metadata']['dynamics']['rates'][0]
        model=HOPEFeaturePredictorV4(VisualBlockConfig(**m['metadata']['visual_config']),beta=beta)
        checkpoints=[];cursor=0
        for i,(name,counts) in enumerate(MVTEC_COUNTS.items()):
            cursor+=counts[0]-(30 if i==0 else 0)
            p=b.load(f'runs/M_seed0/states/task_{i:02d}.pt')
            if p['run_digest']!=M_RUN_DIGEST or p['task_id']!=i or p['event_cursor']!=cursor:raise ValueError('checkpoint cursor/run mismatch')
            session=p['adapter']['session'];model.load_state_dict(session['model'],strict=True)
            if not _state_equal(session['model'],m['slow_and_initial_weights']):raise ValueError('frozen outer weights differ')
            state=model.cms.load_state(session['cms'])
            if state.clock!=cursor or state.commits!=(cursor,cursor//8) or state.pending!=(0,cursor%8):raise ValueError('CMS clock mismatch')
            checkpoints.append(dict(task=name,clock=state.clock,commits=list(state.commits),pending=list(state.pending),
                frozen_outer_weights_equal_W6=True,current_CMS_strict_load=True))
        report=dict(schema='recovery_real_parent_compatibility_v1',success=True,initializations=checks,checkpoints=checkpoints,
            frozen_dependencies_verified=len(source['frozen']),new_model_inference=False,optimizer_steps=0,
            RNG_restored=False,canonical_T4_execution=False,scope='strict actual weight/metadata/state compatibility only')
        atomic_json(report,output);print(json.dumps({'success':True,'initializations':len(checks),'checkpoints':len(checkpoints),'output':str(output)},indent=2))
        return report
    finally:a.close();b.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--w6',type=Path,required=True);p.add_argument('--w72',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();configure_runtime();verify(args.w6,args.w72,args.output)

if __name__=='__main__':main()