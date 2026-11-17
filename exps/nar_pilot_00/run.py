"""NAR-PILOT-00 train-normal pilot; explicit owner flag required for GPU work.

prepare -> immutable INC train cache; lock -> hash inputs/source; execute -> normal-only pilot.
No test images, masks, VisA, or historical learned decoder are used. The frozen decoder
base is the existing SingleStreamDecoder deterministic initialization, a capacity-limited
mechanism testbed rather than an established anomaly detector. No execution is automatic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn

from exps.dinomaly import official as O
from exps.nar_pilot_00 import core as K
from exps.vitb8_q2.common import SingleStreamDecoder

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/hope_cad/nar_pilot_00'
CATEGORIES = ('bottle', 'carpet', 'hazelnut')
ARMS = ('fixed', 'warm', 'nested', 'frozen', 'init_only', 'rate_only')
SEED = 2711
CFG = K.Cfg()
SOURCES = ('exps/nar_pilot_00/core.py', 'exps/nar_pilot_00/run.py',
           'exps/nar_pilot_00/preflight.py', 'exps/nar_pilot_00/test_core.py',
           'exps/dinomaly/official.py', 'exps/vitb8_q2/common.py',
           'exps/calm02/data.py', 'models/feature_extractors/cadic_vit_v1.py',
           'dataset/benchmark_protocol_v1.py', 'agents/plans/nar_pilot_00_prereg.md',
           'agents/plans/nl_cad_final_paradigm_spec.md')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def write_json(p, data):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with Path(p).open('x') as f:
        json.dump(data, f, indent=2)


def save(p, data):
    with Path(p).open('xb') as f:
        torch.save(data, f)


def verify_norm():
    from exps.nar_pilot_00.preflight import norm_audit
    a = norm_audit(ROOT)
    K.require(a['verified_final_verdict'] == 'NORMALIZATION_MATTERS_GO', 'NORM prerequisite')
    return a


def assert_cache(cache, category):
    rows, feat = cache['rows'], cache['features']
    roles = K.normal_partition(rows)
    K.require(all(r['relative_path'].split('/')[0] == category for r in rows), 'wrong/future task')
    K.require(feat.dtype == torch.float32 and tuple(feat.shape) == (len(rows), 784, 768), 'INC cache geometry')
    K.require(bool(torch.isfinite(feat).all()), 'nonfinite cache')
    return roles


def prepare(owner_authorized):
    K.require(owner_authorized, 'GPU preparation requires separate owner execution decision')
    norm = verify_norm()
    K.require(torch.cuda.is_available(), 'CUDA unavailable')
    from exps.calm02.data import CHECKPOINT, CHECKPOINT_SHA
    from models.feature_extractors.cadic_vit_v1 import CADICViTConfig, CADICViTFeatureExtractor
    from dataset.benchmark_protocol_v1 import _ImageDataset
    K.require(sha(CHECKPOINT) == CHECKPOINT_SHA, 'backbone checkpoint hash')
    K.require(not (OUT/'INPUTS.json').exists(), 'input manifest exists; preserve completed preparation')
    (OUT/'inc_train').mkdir(parents=True, exist_ok=True)
    files = {}
    start = time.monotonic()
    model = CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=str(CHECKPOINT),
                checkpoint_identity='sha256:'+CHECKPOINT_SHA, mean=(.5,)*3, std=(.5,)*3), 'cuda')
    for c in CATEGORIES:
        target = OUT/'inc_train'/f'{c}_train.pt'
        K.require(not target.exists(), 'partial preparation exists; preserve and audit before resume')
        rows = [{'relative_path': p.relative_to(ROOT/'data/mvtec').as_posix()}
                for p in sorted((ROOT/'data/mvtec'/c/'train/good').glob('*.png'))]
        roles = K.normal_partition(rows)
        ds = _ImageDataset(ROOT/'data/mvtec', rows, 224, (.5,)*3, (.5,)*3, training=True)
        features = []
        with torch.no_grad():
            for i in range(0, len(rows), 8):
                x = torch.stack([ds[j]['images'] for j in range(i, min(i+8,len(rows)))]).cuda()
                features.append(model.extract_patch_features(x).float().cpu())
        save(target, {'rows': rows, 'features': torch.cat(features), 'roles': roles})
        files[c] = {'path': str(target.relative_to(ROOT)), 'sha256': sha(target), 'n': len(rows)}
        K.require(time.monotonic()-start < 1800, 'feature preparation exceeds 0.5 GPU-hour cap')
        K.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 < 8.5e9, 'preparation RSS cap')
    write_json(OUT/'INPUTS.json', {'schema':'nar_inc_train_v1', 'files':files,
               'normalization':'INC', 'checkpoint_sha256':CHECKPOINT_SHA,
               'norm_result_sha256':norm['result_sha256'], 'wall_s':time.monotonic()-start})


def lock():
    verify_norm()
    inputs = json.loads((OUT/'INPUTS.json').read_text())
    K.require(inputs['normalization'] == 'INC' and set(inputs['files']) == set(CATEGORIES), 'input contract')
    for c, row in inputs['files'].items():
        p = (ROOT/row['path']).resolve()
        K.require(p == (OUT/'inc_train'/f'{c}_train.pt').resolve(), 'unexpected input path')
        K.require(sha(p) == row['sha256'], 'input hash mismatch')
    write_json(OUT/'LOCK.json', {'sources':{p:sha(ROOT/p) for p in SOURCES},
               'inputs_sha256':sha(OUT/'INPUTS.json'), 'cfg':CFG.dict(), 'arms':ARMS,
               'categories':CATEGORIES, 'seed':SEED, 'base':'deterministic_initialized_frozen',
               'normalization':'INC', 'prepared_only':True,
               'environment':{'torch':torch.__version__,'cuda_build':torch.version.cuda}})


def verify_lock():
    lk = json.loads((OUT/'LOCK.json').read_text())
    K.require(all(sha(ROOT/p) == h for p,h in lk['sources'].items()), 'source changed after pilot lock')
    K.require(sha(OUT/'INPUTS.json') == lk['inputs_sha256'], 'input manifest changed')
    K.require(lk['seed']==SEED and lk['categories']==list(CATEGORIES) and lk['arms']==list(ARMS), 'pilot configuration')
    # The source hashes freeze CFG; JSON encodes tuples as lists.
    return lk


def _loss(model, x):
    en, de = model(x)
    return O.global_cosine(en, de)


@torch.no_grad()
def _eval(model, features, indices, device):
    was_training = model.training
    model.eval()
    values = [float(_loss(model, features[j:j+1].to(device))) for j in indices]
    model.train(was_training)
    return sum(values)/len(values)


def execute(owner_authorized):
    K.require(owner_authorized, 'GPU pilot requires separate owner execution decision')
    verify_lock()
    K.require(torch.cuda.is_available(), 'CUDA unavailable')
    gpu = json.loads((OUT/'CUDA_PREFLIGHT.json').read_text())
    K.require(gpu['passed'] and gpu['lock_sha256']==sha(OUT/'LOCK.json'), 'CUDA preflight prerequisite')
    K.require(not (OUT/'STARTED.json').exists(), 'no implicit resume/restart; preserve pilot')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs = json.loads((OUT/'INPUTS.json').read_text())
    write_json(OUT/'STARTED.json', {'lock_sha256':sha(OUT/'LOCK.json'), 'started':time.time()})
    start = time.monotonic()
    for arm in ARMS:
        torch.manual_seed(SEED)
        model = SingleStreamDecoder(checkpoint=True).cuda()
        O.init_trainable(model)
        base_state = model.state_dict()
        if arm == ARMS[0]:
            save(OUT/'BASE.pt', {k:v.detach().cpu() for k,v in base_state.items()})
        else:
            expected = torch.load(OUT/'BASE.pt',weights_only=True,mmap=True)
            K.require(all(torch.equal(base_state[k].cpu(),v) for k,v in expected.items()), 'shared base parity')
            del expected
        base_bytes = sum(x.numel()*x.element_size() for x in base_state.values())
        names = [n for n,m in model.named_modules() if isinstance(m, nn.Linear)]
        loras = K.attach(model, names, CFG.rank)
        fixed = K.init_state(loras, SEED)
        phi, lam, last, archive = fixed, torch.zeros(len(names), dtype=torch.float64), None, []
        for stage,c in enumerate(CATEGORIES):
            # Only current task tensor is opened; old tensors/probes are never reopened as training data.
            inp = inputs['files'][c]
            K.require(sha(ROOT/inp['path']) == inp['sha256'], 'current feature hash')
            cache = torch.load(ROOT/inp['path'], weights_only=True, mmap=True)
            roles = assert_cache(cache, c)
            features = cache['features']
            init = last if arm=='warm' and last is not None else fixed if arm in ('fixed','rate_only') else phi
            K.set_state(loras, init)
            learn = arm in ('nested','rate_only') or (arm=='frozen' and stage==0)
            used_lam = lam if arm in ('nested','rate_only','frozen') else torch.zeros_like(lam)
            gen = torch.Generator().manual_seed(SEED+100*stage)
            torch.manual_seed(SEED+100*stage)  # matched dropout/RNG across arms
            def fit(i):
                idx = torch.randperm(len(roles['FIT']),generator=gen)[:1]
                j = roles['FIT'][int(idx[0])]
                return _loss(model, features[j:j+1].cuda())
            def credit():
                # Held out of fast training, but used by slow learning; EVAL stays separate.
                j = roles['CREDIT'][(credit.calls % len(roles['CREDIT']))]
                credit.calls += 1
                was = model.training
                model.eval()
                value = _loss(model, features[j:j+1].cuda())
                model.train(was)
                return value
            credit.calls = 0
            model.train()
            opt = lambda groups: O.StableAdamW(groups, lr=CFG.lr, betas=CFG.betas,
                       weight_decay=CFG.weight_decay, eps=CFG.eps, amsgrad=True)
            curve = [{'step':0,'loss':_eval(model,features,roles['EVAL'],'cuda')}]
            def observe(k):
                if k % 50 == 0:
                    curve.append({'step':k,'loss':_eval(model,features,roles['EVAL'],'cuda')})
                K.require(time.monotonic()-start < 9000, '2.5 GPU-hour pilot cap')
                K.require(torch.cuda.max_memory_allocated() <= 5*2**30, '5 GiB VRAM cap')
                K.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 < 8.5e9, '8.5 GB RSS cap')
            lam_new, log = K.acquire(loras,fit,credit,CFG,used_lam,learn,opt,on_step=observe,observe_credit=True)
            endpoint = K.get_state(loras)
            archive.append(endpoint)
            if arm in ('nested','init_only') or (arm=='frozen' and stage==0):
                phi = K.reptile(phi,endpoint,CFG.beta)
            if arm in ('nested','rate_only','frozen'):
                lam = lam_new
            last = endpoint
            target = OUT/f'{arm}_stage{stage+1}.pt'
            save(target, {'cfg':CFG.dict(),'archive':archive,'phi':phi,'lam':lam,'category':c})
            write_json(OUT/f'{arm}_stage{stage+1}.json', {'curve':curve,'log':log,
                'state_sha256':sha(target),'base_bytes':base_bytes,'adapter_bytes':K.tensor_state_bytes(endpoint),
                'policy_bytes':K.slow_bytes(lam,phi), 'archive_bytes':sum(K.tensor_state_bytes(a) for a in archive),
                'cuda_peak_allocated':torch.cuda.max_memory_allocated(), 'wall_s':time.monotonic()-start})
            del cache, features
        del model, loras, archive, phi, last, endpoint
        torch.cuda.empty_cache()
    write_json(OUT/'COMPLETED.json', {'lock_sha256':sha(OUT/'LOCK.json'),'wall_s':time.monotonic()-start,
               'scientific_verdict':'ANALYSIS_REQUIRED_NO_ANOMALY_CLAIM'})


def gpu_preflight(owner_authorized):
    K.require(owner_authorized, 'GPU preflight requires separate owner execution decision')
    verify_lock()
    K.require(torch.cuda.is_available(), 'CUDA unavailable')
    K.require(not (OUT/'CUDA_PREFLIGHT.json').exists(), 'preserve CUDA preflight')
    torch.manual_seed(SEED)
    model = SingleStreamDecoder(checkpoint=True).cuda()
    O.init_trainable(model)
    names = [n for n,m in model.named_modules() if isinstance(m,nn.Linear)]
    loras = K.attach(model,names,CFG.rank)
    K.set_state(loras,K.init_state(loras,SEED))
    inp = json.loads((OUT/'INPUTS.json').read_text())['files'][CATEGORIES[0]]
    K.require(sha(ROOT/inp['path'])==inp['sha256'],'preflight current feature hash')
    cache = torch.load(ROOT/inp['path'],weights_only=True,mmap=True)
    roles = assert_cache(cache,CATEGORIES[0])
    fit_x = cache['features'][roles['FIT'][0]:roles['FIT'][0]+1].cuda()
    credit_x = cache['features'][roles['CREDIT'][0]:roles['CREDIT'][0]+1].cuda()
    cfg = replace(CFG,steps=3,m=1)
    opt = lambda groups: O.StableAdamW(groups,lr=cfg.lr,betas=cfg.betas,
                   weight_decay=cfg.weight_decay,eps=cfg.eps,amsgrad=True)
    def credit():
        model.eval()
        value = _loss(model,credit_x)
        model.train()
        return value
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.monotonic()
    _, log = K.acquire(loras,lambda i:_loss(model,fit_x),credit,cfg,
                      torch.zeros(len(names),dtype=torch.float64),True,opt)
    torch.cuda.synchronize()
    wall = time.monotonic()-start
    # Conservative upper estimate: charge fit+credit cost on EVERY pilot step.
    projection = (wall/3)*len(ARMS)*len(CATEGORIES)*CFG.steps
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    peak = torch.cuda.max_memory_allocated()
    passed = wall < 120 and projection < 9000 and peak <= 5*2**30 and rss < 8.5e9
    write_json(OUT/'CUDA_PREFLIGHT.json',{'passed':passed,'wall_s':wall,'conservative_projected_pilot_s':projection,
        'peak_cuda_allocated':peak,'peak_rss':rss,'fit':log['fit'],'credit':log['credit'],
        'lock_sha256':sha(OUT/'LOCK.json'),'current_task_only':CATEGORIES[0],
        'torch':torch.__version__,'device':torch.cuda.get_device_name(0)})
    K.require(passed,'GPU resource gate failed; do not run full pilot')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','lock','gpu-preflight','execute'))
    parser.add_argument('--owner-authorized',action='store_true')
    args = parser.parse_args()
    if args.command=='lock':
        lock()
    elif args.command=='prepare':
        prepare(args.owner_authorized)
    elif args.command=='gpu-preflight':
        gpu_preflight(args.owner_authorized)
    else:
        execute(args.owner_authorized)


if __name__=='__main__':
    main()
