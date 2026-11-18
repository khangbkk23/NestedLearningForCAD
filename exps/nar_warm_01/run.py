"""One registered warm-start arm. Historical source and outputs stay read-only."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import resource
import shutil
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

from exps.nar_pilot_00 import core as K, full as R
from exps.nar_pilot_00.cost import quantiles
from exps.nar_pilot_00.detector import Detector, prefix_grids, fm_eq8

OUT=R.ROOT/'results/hope_cad/nar_warm_01'
HIST=R.OUT
CAPS={'wall_seconds':2700,'gpu_seconds':1800,'cuda_bytes':5*2**30,
      'rss_bytes':8_500_000_000,'artifact_bytes':700_000_000}
SOURCES=('exps/nar_warm_01/run.py','exps/nar_warm_01/test_run.py',
         'exps/nar_pilot_00/cost.py','exps/nl_sgm_01a/am03_eval.py',
         'agents/tasks/task_nar_warm_01.md','agents/reports/nar_improvement_audit_01.md')


def disk_bytes():
    return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())


def digest(state):
    h=hashlib.sha256()
    for name,pair in state.items():
        h.update(name.encode())
        for x in pair:
            h.update(str((tuple(x.shape),str(x.dtype))).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initial_state(fixed,archive):
    return archive[-1] if archive else fixed


def gate(nested,warm,ci):
    guards={'pixel_gain':ci['point']>=.01,'category_ci':ci['lo']>0,
      'image':nested['final_image_auroc']>=warm['final_image_auroc']-.01,
      'acquisition':nested['acquisition_pixel_ap']>=warm['acquisition_pixel_ap']-.01,
      'forgetting':nested['pixel_fm_eq8']['macro']<=warm['pixel_fm_eq8']['macro']+.01}
    return {'passed':all(guards.values()),'guards':guards}


def lock():
    K.require(not OUT.exists(),'preserve any existing experiment directory')
    R.verify_training()
    old=R.read(HIST/'FINAL_AUDIT.json')
    for p,h in old['artifact_sha256'].items(): K.require(R.sha(HIST/p)==h,'historical seal')
    K.require(R.sha(R.CHECKPOINT)==R.CHECKPOINT_SHA,'encoder hash')
    inputs={}
    for c in R.ORDER:
        inputs[c]={}
        for split in ('train','test'):
            p=HIST/'features'/f'{c}_{split}.pt'; meta=p.with_suffix('.json')
            m=R.read(meta)
            K.require(p.exists() and m['source_lock_sha256']==R.sha(HIST/'SOURCE_LOCK.json'),'cache provenance')
            # Opaque hash verification does not deserialize test tensors/labels.
            K.require(R.sha(p)==m['sha256'],'input byte hash')
            inputs[c][split]={'path':str(p.relative_to(R.ROOT)),'sha256':m['sha256'],
                                'metadata_sha256':R.sha(meta),'n':m['n']}
    K.require(shutil.disk_usage(R.ROOT).free>CAPS['artifact_bytes']+1e9,'disk headroom')
    R.write_json(OUT/'LOCK.json',{'sources':{p:R.sha(R.ROOT/p) for p in SOURCES},
      'historical_sources':R.read(HIST/'SOURCE_LOCK.json')['sources'],
      'historical_artifacts':old['artifact_sha256'],'historical_audit_sha256':R.sha(HIST/'FINAL_AUDIT.json'),
      'inputs':inputs,'base_sha256':R.sha(HIST/'BASE.pt'),'encoder_sha256':R.CHECKPOINT_SHA,
      'cfg':R.CFG.dict(),'seed':R.SEED,'order':R.ORDER,'arm':'warm','caps':CAPS,
      'policy':'previous own endpoint; lambda=0; no Reptile; matched discarded credit',
      'gate':{'ap_delta':.01,'ci_lower':0,'image_tolerance':.01,'acquisition_tolerance':.01,
              'fm_tolerance':.01,'bootstrap_seed':20261117,'bootstrap_B':2000},
      'cost':'canonical first20 bottle train/good; batch1; three warmups; no optimized scorer',
      'torch':torch.__version__,'cuda_build':torch.version.cuda,
      'projected_wall_seconds':2200,'projected_gpu_seconds':1500,
      'projected_artifact_bytes':550_000_000})
    R.log('LOCK',R.sha(OUT/'LOCK.json'))


def verify():
    lk=R.read(OUT/'LOCK.json')
    for key in ('sources','historical_sources'):
        for p,h in lk[key].items(): K.require(R.sha(R.ROOT/p)==h,'locked source changed: '+p)
    for p,h in lk['historical_artifacts'].items(): K.require(R.sha(HIST/p)==h,'historical artifact changed')
    K.require(R.sha(HIST/'BASE.pt')==lk['base_sha256'],'base hash')
    K.require(R.sha(R.CHECKPOINT)==lk['encoder_sha256'],'encoder hash')
    K.require(lk['torch']==torch.__version__ and lk['cuda_build']==torch.version.cuda,'environment mismatch')
    K.require(lk['caps']==CAPS and lk['seed']==R.SEED and lk['order']==list(R.ORDER),'config mismatch')
    K.require(lk['cfg']==json.loads(json.dumps(R.CFG.dict())),'CFG mismatch')
    return lk


class Budget:
    def __init__(self):
        self.start=time.monotonic(); self.gpu_seconds=0.; self.phase_start=None
        self.phases={}; self.peak_cuda=0
    def check(self,*args,**kwargs):
        K.require(time.monotonic()-self.start<CAPS['wall_seconds'],'wall cap')
        gpu=self.gpu_seconds+(time.monotonic()-self.phase_start if self.phase_start is not None else 0)
        K.require(gpu<CAPS['gpu_seconds'],'GPU elapsed phase cap')
        K.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<CAPS['rss_bytes'],'RSS cap')
        if torch.cuda.is_initialized():
            self.peak_cuda=max(self.peak_cuda,torch.cuda.max_memory_allocated())
            K.require(self.peak_cuda<=CAPS['cuda_bytes'],'CUDA cap')
        K.require(disk_bytes()<CAPS['artifact_bytes'],'artifact cap')
    def alarm(self,*args):
        raise RuntimeError('registered elapsed-time deadline reached')
    @contextmanager
    def phase(self,name,gpu=False):
        self.check(); start=time.monotonic()
        if gpu: self.phase_start=start
        remaining=CAPS['wall_seconds']-(start-self.start)
        if gpu: remaining=min(remaining,CAPS['gpu_seconds']-self.gpu_seconds)
        signal.signal(signal.SIGALRM,self.alarm); signal.setitimer(signal.ITIMER_REAL,max(.001,remaining))
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            elapsed=time.monotonic()-start
            self.phases[name]=elapsed
            if gpu: self.gpu_seconds+=elapsed; self.phase_start=None
        self.check()
    def json(self):
        return {'wall_seconds':time.monotonic()-self.start,'gpu_phase_seconds':self.gpu_seconds,
          'phase_seconds':self.phases,'peak_cuda_allocated_bytes':self.peak_cuda,
          'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          'artifact_bytes':disk_bytes()}


def load_cache(lk,c,split):
    if split=='test': verify_training()
    d=lk['inputs'][c][split]; p=R.ROOT/d['path']
    K.require(R.sha(p)==d['sha256'],'input changed')
    return torch.load(p,weights_only=True,mmap=True)


def bounded_save(path,payload,estimated_bytes,budget):
    budget.check()
    K.require(disk_bytes()+estimated_bytes+1_000_000<CAPS['artifact_bytes'],'save would exceed disk cap')
    R.save(path,payload); budget.check()


def train(lk,budget):
    base=torch.load(HIST/'BASE.pt',weights_only=True)
    model,loras,_=R.model_new(base)
    fixed=K.init_state(loras,R.SEED); lam=torch.zeros(len(loras),dtype=torch.float64); archive=[]
    records=[]
    for stage,c in enumerate(R.ORDER):
        data=load_cache(lk,c,'train'); roles=R.assert_cache(data,c)
        init=initial_state(fixed,archive); K.set_state(loras,init)
        K.require(R.state_equal(K.get_state(loras),init),'warm initialization mismatch')
        init_digest=digest(init)
        before=R.normal_eval(model,data['features'],roles['EVAL'])
        lam,trace,wall=R.fit_task(model,loras,data['features'],roles,stage,lam,False)
        endpoint=K.get_state(loras)
        after=R.normal_eval(model,data['features'],roles['EVAL'])
        K.require(torch.equal(lam,torch.zeros_like(lam)),'nonzero warm lambda')
        K.require(len(trace['fit'])==300 and len(trace['credit'])==30,'gradient budget')
        if stage==0:
            reference=torch.load(HIST/'fixed/stage01.pt',weights_only=True)
            same=R.state_equal(endpoint,reference['archive'][0])
            reference_trace=R.read(HIST/'fixed/stage01.json')
            R.write_json(OUT/'TASK1_PARITY.json',{'adapter_bitwise':same,
              'trace_equal':trace==reference_trace['trace'],
              'normal_before_equal':before==reference_trace['before'],
              'normal_after_equal':after==reference_trace['after'],
              'reference_sha256':R.sha(HIST/'fixed/stage01.pt'),
              'endpoint_digest':digest(endpoint)})
            K.require(same,'task1 fixed parity failed; stop before task2')
            del reference
            R.log('TASK1_BITWISE_PASS')
        archive.append(endpoint)
        target=OUT/f'stage{stage+1:02}.pt'
        # phi is an inert, distinct fixed-init slot for capacity matching, never the warm initializer.
        bounded_save(target,{'archive':archive,'phi':fixed,'lam':lam,'cfg':R.CFG.dict(),
            'stage':stage+1,'arm':'warm','base_sha256':lk['base_sha256']},
            sum(K.tensor_state_bytes(a) for a in archive)+K.slow_bytes(lam,fixed),budget)
        record={'stage':stage+1,'category':c,'before':before,'after':after,'seconds':wall,
          'trace':trace,'roles':roles,'feature_sha256':lk['inputs'][c]['train']['sha256'],
          'initial_digest':init_digest,'endpoint_digest':digest(endpoint),'state_sha256':R.sha(target),
          'adapter_bytes':K.tensor_state_bytes(endpoint),'archive_bytes':sum(K.tensor_state_bytes(a) for a in archive),
          'policy_bytes':K.slow_bytes(lam,fixed)}
        R.write_json(target.with_suffix('.json'),record); records.append(record)
        R.log('TRAIN',stage+1,c,before,after,'seconds',wall)
        del data; budget.check()
    R.write_json(OUT/'TRAIN_COMPLETE.json',{'lock_sha256':R.sha(OUT/'LOCK.json'),
      'base_sha256':lk['base_sha256'],'checkpoints':{f'stage{i:02}.pt':R.sha(OUT/f'stage{i:02}.pt') for i in range(1,16)},
      'records':{f'stage{i:02}.json':R.sha(OUT/f'stage{i:02}.json') for i in range(1,16)},
      'task1_parity_sha256':R.sha(OUT/'TASK1_PARITY.json'),
      'update_seconds':sum(r['seconds'] for r in records),'resource_at_seal':budget.json()})
    del model,loras,base,archive; gc.collect(); torch.cuda.empty_cache()


def verify_training():
    m=R.read(OUT/'TRAIN_COMPLETE.json')
    K.require(m['lock_sha256']==R.sha(OUT/'LOCK.json') and len(m['checkpoints'])==15,'training seal incomplete')
    for key in ('checkpoints','records'):
        for p,h in m[key].items(): K.require(R.sha(OUT/p)==h,'training seal mismatch')
    K.require(R.sha(OUT/'TASK1_PARITY.json')==m['task1_parity_sha256'],'parity seal')
    K.require(R.read(OUT/'TASK1_PARITY.json')['adapter_bitwise'],'parity failed')
    return m


def score(lk,budget):
    complete=verify_training()
    final=torch.load(OUT/'stage15.pt',weights_only=True)
    for t in range(15):
        stage=torch.load(OUT/f'stage{t+1:02}.pt',weights_only=True)
        K.require(R.state_equal(stage['archive'][t],final['archive'][t]),'archive changed')
    del stage
    model,loras,_=R.model_new(torch.load(HIST/'BASE.pt',weights_only=True)); model.eval()
    start=time.monotonic()
    for c in R.ORDER:
        data=load_cache(lk,c,'test'); features=data['features']; n=len(features)
        all_scores=torch.empty(15,n,28,28)
        # Identical expert-outer/batch-four arithmetic and ordering to sealed full.score.
        for t,state in enumerate(final['archive']):
            K.set_state(loras,state)
            with torch.no_grad():
                for j in range(0,n,4):
                    en,de=model(features[j:j+4].cuda())
                    all_scores[t,j:j+4]=(1-F.cosine_similarity(en[0].float(),de[0].float(),dim=1)).clamp_min(0).cpu()
                    budget.check()
        prefixes=prefix_grids(all_scores)
        bounded_save(OUT/f'scores_{c}.pt',{'prefix_grids':prefixes,'rows':data['rows'],
          'base_sha256':complete['base_sha256']},prefixes.numel()*prefixes.element_size(),budget)
        R.log('SCORE',c,n); del data,features,all_scores,prefixes
    R.write_json(OUT/'SCORES_COMPLETE.json',{'train_seal_sha256':R.sha(OUT/'TRAIN_COMPLETE.json'),
      'seconds':time.monotonic()-start,'files':{p.name:R.sha(p) for p in OUT.glob('scores_*.pt')}})
    del model,loras,final; gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def cost(lk,budget):
    from PIL import Image
    from dataset.benchmark_protocol_v1 import _ImageDataset
    data=load_cache(lk,'bottle','train'); rows=data['rows'][:20]
    ds=_ImageDataset(R.DATA,rows,224,(.5,)*3,(.5,)*3,training=True)
    encoder=R.make_encoder(); base=torch.load(HIST/'BASE.pt',weights_only=True)
    state=torch.load(OUT/'stage15.pt',weights_only=True)
    model,loras,_=R.model_new(base); detector=Detector(model,loras,state['archive'])
    pre,enc,det=[],[],[]; torch.cuda.reset_peak_memory_stats()
    for i in range(23):
        j=max(0,i-3); start=time.perf_counter(); x=ds[j]['images'][None]
        with Image.open(R.DATA/rows[j]['relative_path']) as image: hw=(image.height,image.width)
        prep=time.perf_counter()-start
        torch.cuda.synchronize(); start=time.perf_counter(); feature=encoder.extract_patch_features(x.cuda()).float()
        torch.cuda.synchronize(); enc_time=time.perf_counter()-start
        start=time.perf_counter(); detector.predict(feature,hw); torch.cuda.synchronize(); dt=time.perf_counter()-start
        if i>=3: pre.append(prep); enc.append(enc_time); det.append(dt)
        budget.check()
    parts={'shared_encoder':sum(x.numel()*x.element_size() for x in encoder.state_dict().values()),
           'shared_decoder':sum(x.numel()*x.element_size() for x in base.values()),
           'archive':sum(K.tensor_state_bytes(a) for a in state['archive']),
           'policy':K.slow_bytes(state['lam'],state['phi'])}
    R.write_json(OUT/'COST.json',{'preprocess':quantiles(pre),'encoder':quantiles(enc),'detector':quantiles(det),
      'end_to_end':quantiles(np.array(pre)+enc+det),'cuda_peak_bytes':torch.cuda.max_memory_allocated(),
      'persistent_bytes':parts,'total_persistent_bytes':sum(parts.values()),
      'serialized_deployment_bytes':R.CHECKPOINT.stat().st_size+(HIST/'BASE.pt').stat().st_size+(OUT/'stage15.pt').stat().st_size,
      'query_source':'first20 bottle/train/good; canonical 15-expert detector; batch1; three warmups',
      'device':torch.cuda.get_device_name(0)})
    del encoder,model,loras,detector,data; gc.collect(); torch.cuda.empty_cache()


def metrics(budget):
    from exps.calm02.native import native_ap,native_mask,native_hw
    from sklearn.metrics import roc_auc_score
    from exps.nl_sgm_01a.am03_eval import bootstrap_delta
    verify_training(); scored=R.read(OUT/'SCORES_COMPLETE.json'); cells={}
    for i,c in enumerate(R.ORDER):
        path=OUT/f'scores_{c}.pt'; K.require(R.sha(path)==scored['files'][path.name],'score hash')
        data=torch.load(path,weights_only=True); rows=data['rows']; grids=data['prefix_grids']
        hws=[native_hw(R.DATA/r['relative_path']) for r in rows]
        masks=[native_mask(R.DATA/r['mask_path'] if r['mask_path'] else None,hw) for r,hw in zip(rows,hws)]
        labels=[r['label'] for r in rows]
        for t in range(i,15):
            grid=grids[t].reshape(len(rows),784).numpy()
            cell={'stage':t+1,'category':c,'pixel_ap':native_ap(grid,hws,masks)['ap'],
                  'image_auroc':float(roc_auc_score(labels,grid.max(axis=1))),
                  'scores_sha256':scored['files'][path.name]}
            R.write_json(OUT/f'metric_{t+1:02}_{c}.json',cell); cells[t,c]=cell
            R.log('METRIC',t+1,c,cell['pixel_ap'],cell['image_auroc']); budget.check()
        del data,grids,masks
    per={c:{'acquisition':cells[i,c],'final':cells[14,c]} for i,c in enumerate(R.ORDER)}
    warm={'per_category':per,
      'final_pixel_ap':float(np.mean([per[c]['final']['pixel_ap'] for c in R.ORDER])),
      'final_image_auroc':float(np.mean([per[c]['final']['image_auroc'] for c in R.ORDER])),
      'acquisition_pixel_ap':float(np.mean([per[c]['acquisition']['pixel_ap'] for c in R.ORDER])),
      'pixel_fm_eq8':fm_eq8(cells,R.ORDER,'pixel_ap'),'image_fm_eq8':fm_eq8(cells,R.ORDER,'image_auroc')}
    historic=R.read(HIST/'RESULT.json')['arms']; nested=historic['nested']
    ci=bootstrap_delta([nested['per_category'][c]['final']['pixel_ap'] for c in R.ORDER],
      [per[c]['final']['pixel_ap'] for c in R.ORDER],seed=20261117,B=2000)
    return {'arms':{**historic,'warm':warm},'nested_minus_warm':ci,
      'NL_ADDED_VALUE_VS_WARM':gate(nested,warm,ci),
      'exposure':'One seed2711, alphabetical15, historically exposed MVTec; exploratory only'}


def execute():
    K.require(not (OUT/'STARTED.json').exists(),'no implicit restart or resume')
    budget=Budget(); lk=verify(); R.configure()
    # Only the guard callback is redirected; no numerical function or file changes.
    original_cap=R.cap; R.cap=budget.check
    R.write_json(OUT/'STARTED.json',{'lock_sha256':R.sha(OUT/'LOCK.json'),'started_unix':time.time(),
      'device':torch.cuda.get_device_name(0),'free_disk':shutil.disk_usage(R.ROOT).free})
    try:
        with budget.phase('training',gpu=True): train(lk,budget)
        with budget.phase('scoring',gpu=True): score(lk,budget)
        with budget.phase('inference_cost',gpu=True): cost(lk,budget)
        with budget.phase('metrics'): result=metrics(budget)
        verify(); verify_training(); budget.check()
        result['resources']=budget.json(); result['cost']=R.read(OUT/'COST.json')
        R.write_json(OUT/'RESULT.json',result)
        R.write_json(OUT/'FINAL_SEAL.json',{'lock_sha256':R.sha(OUT/'LOCK.json'),
          'files':{p.name:R.sha(p) for p in OUT.iterdir() if p.is_file()},'resources':budget.json()})
        budget.check(); R.log('COMPLETE',result['nested_minus_warm'],result['NL_ADDED_VALUE_VS_WARM'],budget.json())
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0)
        R.write_json(OUT/'STOPPED.json',{'reason':repr(exc),'resources':budget.json()})
        raise
    finally:
        R.cap=original_cap


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('phase',choices=('lock','execute'))
    parser.add_argument('--owner-authorized',action='store_true'); args=parser.parse_args()
    if args.phase=='lock': lock()
    else:
        K.require(args.owner_authorized,'explicit owner execution flag required'); execute()
