"""User-authorized causal normal-bank fit; no real-data scorer/evaluator.

Uses existing immutable clean caches. No encoder weights are trained or replaced.
Run outside the sandbox where the session's CUDA device is accessible.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .contcore import CausalContCore,farthest_indices
from .distance import Euclidean,configure
from . import oracle

ROOT=Path(__file__).resolve().parents[3]
FEATURES=ROOT/'results/hope_cad/frozen_features_v1'
CATEGORIES=('bottle','carpet','hazelnut')
CP_SHA='755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415'
SOURCE_SHA='060579457d344992eef159b1d4f1a86c34dd9283b97b70d3c3a386006b0f4dcc'
CACHE_SHA={
 'bottle':'ef4f3180d2381b64b4cdafea447b9df254b7522b9338f24992f4a857114ad6b3',
 'carpet':'f3b1957d7b9d12add45c4ff9c33a5f9f5e237ff7ce6dbdb7693c81527c89b179',
 'hazelnut':'7bd97accb4a927285881d8c8b61cbc5db8106e57e2ceee1fe623f5903f62d180'}


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def tensor_sha(x):
    x=x.detach().contiguous().cpu()
    return hashlib.sha256(str((tuple(x.shape),x.dtype)).encode()+x.numpy().tobytes()).hexdigest()


def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True)


def cpu_state(model):
    return {k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in model.state_dict().items()}


def cuda_gate():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in this executor')
    device='cuda'
    xx=torch.tensor([[10.],[0.]],device=device)
    choice=farthest_indices(xx,torch.arange(2,device=device),1,torch.tensor([[9.]],device=device))
    assert choice.cpu().tolist()==[1]
    for q in (0,.75):
        model=CausalContCore(7,3,8,3,q=q,device=device)
        bank,ids=[],[]
        g=torch.Generator().manual_seed(3311)
        for t in range(6):
            x=torch.randint(-20,20,(8,3),generator=g).float()
            model.update(x.to(device),event=t)
            bank,ids=oracle.update(bank,ids,x.tolist(),t,7,3,q)
            assert model.serials.cpu().tolist()==ids
            assert model.bank.cpu().tolist()==bank
        state=cpu_state(model);resumed=CausalContCore.from_state_dict(state,device=device)
        torch.testing.assert_close(model.bank,resumed.bank,rtol=0,atol=0)
        before=tensor_sha(model.bank);model.score(torch.zeros(8,3,device=device))
        assert tensor_sha(model.bank)==before
    g=torch.Generator().manual_seed(3312)
    x=torch.randn(19,768,generator=g);y=torch.cat((x[:5],x[:5]+1e-4,torch.randn(13,768,generator=g)))
    reference=(x[:,None].double()-y[None].double()).square().sum(2).sqrt().float()
    for chunk in (7,256):
        actual=Euclidean(chunk).matrix(x.to(device),y.to(device)).cpu()
        torch.testing.assert_close(actual,reference,rtol=2e-6,atol=2e-6)
    torch.cuda.synchronize()
    return {'status':'PASS','device':torch.cuda.get_device_name(0),
            'oracle_selection_q':[0,.75],'distance_dimensions':768,
            'distance_chunks':[7,256],'scoring_inputs':'synthetic only'}


def writer_plan(source):
    assert all(Path(r['relative_path']).parts[1:3]==('train','good') for r in source['entries'])
    roles={};by_path={r['relative_path']:r for r in source['entries']}
    for c in CATEGORIES:
        rows=[r for r in source['entries'] if r['category']==c]
        rows.sort(key=lambda r:hashlib.sha256(('ecad-r1a-split-11021:'+r['raw_png_sha256']).encode()).hexdigest())
        assert len(rows)>=206
        roles[c]={'first':rows[32:132],'return':rows[132:182]}
    updates=[]
    for c,role in [('bottle','first'),('carpet','first'),('hazelnut','first'),('bottle','return')]:
        rows=roles[c][role]
        rng=np.random.default_rng(np.random.SeedSequence([8011,CATEGORIES.index(c),0 if role=='first' else 1]))
        for i in rng.permutation(len(rows)):
            row=rows[int(i)]
            updates.append({'category':c,'relative_path':row['relative_path'],
                            'raw_png_sha256':row['raw_png_sha256'],'event':len(updates),'role':role})
    assert len(updates)==350
    assert len({r['raw_png_sha256'] for r in updates})==350
    return updates,by_path


class ArrivalCache:
    """Private input service: materialize only the next admitted image's grid."""
    def __init__(self,source,updates):
        self.source,self.updates,self.next_event=source,updates,0
        self.category=None;self.obj=None;self.checked={}

    def next(self,event):
        if event!=self.next_event:raise ValueError('out-of-order cache request')
        row=self.updates[event];c=row['category']
        if c!=self.category:
            self.obj=None
            path=FEATURES/f'clean/{c}_train.pt'
            if c not in self.checked:
                digest=file_sha(path)
                if digest!=CACHE_SHA[c]:raise RuntimeError(f'cache digest changed: {c}')
                self.checked[c]=digest
            obj=torch.load(path,mmap=True,weights_only=True,map_location='cpu')
            entries=[r for r in self.source['entries'] if r['category']==c]
            assert obj['view']=='clean' and obj['class_name']==c
            assert obj['relative_paths']==[r['relative_path'] for r in entries]
            assert obj['per_image_source_sha256']==[r['raw_png_sha256'] for r in entries]
            assert obj['patches'].shape==(len(entries),784,768) and obj['patches'].dtype==torch.float32
            assert obj['source_manifest_digest']==SOURCE_SHA
            meta=obj['metadata']
            assert meta['checkpoint_sha256']==CP_SHA and meta['block_index']==8
            assert meta['feature_normalization']=='none' and meta['cls_removed']
            self.obj=obj;self.category=c
            self.index={p:i for i,p in enumerate(obj['relative_paths'])}
        i=self.index[row['relative_path']]
        assert self.obj['per_image_source_sha256'][i]==row['raw_png_sha256']
        grid=self.obj['patches'][i].clone()
        if tensor_sha(grid)!=self.obj['per_image_feature_sha256'][i]:raise RuntimeError('arrival feature hash mismatch')
        if not bool(torch.isfinite(grid).all()):raise RuntimeError('nonfinite arrival')
        self.next_event+=1
        return grid


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    configure();torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();deadline=start+7200
    write(args.out/'cuda_gate.json',cuda_gate())
    source=json.loads((FEATURES/'source_manifest.json').read_text())
    digest=hashlib.sha256(json.dumps({k:v for k,v in source.items() if k!='manifest_digest'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert digest==source['manifest_digest']==SOURCE_SHA
    updates,_=writer_plan(source)
    pin=json.loads((Path(__file__).parent/'IMPLEMENTATION_PIN.json').read_text())
    for n,h in pin['files'].items():assert file_sha(ROOT/n)==h,(n,'source pin mismatch')
    write(args.out/'fit_contract.json',{'kind':'causal normal memory-bank fit, no efficacy evaluation',
          'cost_profile':'waived by user','capacity':2500,'candidates':78,'q':.75,
          'checkpoint_lineage':CP_SHA,'source_manifest_digest':SOURCE_SHA,
          'writer_seed':8011,'updates':updates,'torch':torch.__version__,
          'core_source_pin':pin,'runner_sha256':file_sha(__file__)})
    reader=ArrivalCache(source,updates);model=CausalContCore(device='cuda')
    checkpoints=[]
    with (args.out/'updates.jsonl').open('x') as log:
        for event in range(350):
            if time.monotonic()>deadline:raise TimeoutError('120-minute fit hard stop')
            grid=reader.next(event).to('cuda')
            stats=model.update(grid,event=event)
            record={k:int(v) if isinstance(v,torch.Tensor) else v for k,v in stats.items()}
            record['completed']=event+1
            log.write(json.dumps(record)+'\n');log.flush()
            if (event+1)%25==0:
                print(json.dumps({'completed':event+1,'occupied':len(model.bank)}),flush=True)
            if event+1 in (100,200,300,350):
                path=args.out/f'contcore_{event+1:03}.pt'
                with path.open('xb') as f:torch.save(cpu_state(model),f)
                checkpoints.append({'path':str(path),'sha256':file_sha(path),'bytes':path.stat().st_size})
    torch.cuda.synchronize()
    saved=torch.load(args.out/'contcore_350.pt',weights_only=True,map_location='cpu')
    restored=CausalContCore.from_state_dict(saved,device='cuda')
    torch.testing.assert_close(model.bank,restored.bank,rtol=0,atol=0)
    assert torch.equal(model.serials,restored.serials) and model.next_event==restored.next_event==350
    # Compare identical synthetic continuations on two clones; the real fit and
    # final checkpoint remain unchanged, and no real writer is replayed/scored.
    control=CausalContCore.from_state_dict(cpu_state(model),device='cuda')
    probe=torch.randn(784,768,generator=torch.Generator().manual_seed(8811)).to('cuda')
    restored.update(probe,event=350);control.update(probe,event=350)
    torch.testing.assert_close(restored.bank,control.bank,rtol=0,atol=0)
    assert torch.equal(restored.serials,control.serials)
    write(args.out/'summary.json',{'status':'DONE','kind':'normal memory-bank fit',
          'normal_images':350,'observed_patches':350*784,'candidate_opportunities':350*78,
          'retained_descriptors':len(model.bank),'dim':768,'checkpoint_resume':'PASS',
          'synthetic_continuation_resume':'PASS','real_data_scoring_performed':False,
          'neural_parameters_trained':0,'cost_profile_performed':False,
          'cache_sha256_verified':reader.checked,'checkpoints':checkpoints,
          'final_bank_sha256':tensor_sha(model.bank),'final_serials_sha256':tensor_sha(model.serials),
          'elapsed_seconds':time.monotonic()-start,'device':torch.cuda.get_device_name(0)})
    print('FIT_DONE '+str(args.out/'summary.json'),flush=True)


if __name__=='__main__':main()
