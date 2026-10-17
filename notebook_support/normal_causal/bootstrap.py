"""Transport only reviewed normal-data state; reject CPU/mismatched runtime early."""
from pathlib import Path,PurePosixPath
import json,os,tarfile,zipfile
from .contract import *
from notebook_support.recovery_stage1.common import BACKBONE_SHA
from training.hope_w7_contract import W6_MANIFEST_SHA256
from training.hope_w7_io import restore_archive,destination

def read_entry(source,name):
    if Path(source).is_dir():return (Path(source)/name).read_bytes()
    with zipfile.ZipFile(source) as z:return z.read(name)

def verified_import(source,root,pinned,required,schema):
    m,h=read_manifest(source)
    if h!=pinned or not set(required)<=set(m['files']):raise ValueError('wrong/incomplete reviewed parent')
    print('[IMPORT]',len(required),'verified files ->',root,flush=True)
    restore_archive(source,root,schema=schema,select=lambda n:n in required)
    return {n:m['files'][n] for n in required}

def discover_resume(inputs,override=None):
    candidates=[Path(override)] if override else [p.parent for p in Path(inputs).rglob('artifact_manifest.json')]
    if not override and not any(read_manifest(p)[0].get('schema')==SCHEMA for p in candidates):candidates+=list(Path(inputs).rglob('*.zip'))
    found=[]
    for p in candidates:
        try:
            if read_manifest(p)[0].get('schema')==SCHEMA:found.append(p)
        except (ValueError,KeyError,zipfile.BadZipFile):pass
    if override and not found:raise ValueError('RESUME_OVERRIDE is not diagnostic evidence')
    if len({read_manifest(p)[1] for p in found})>1:raise ValueError('attach only LATEST diagnostic output or set RESUME_OVERRIDE')
    return sorted(found)[0] if found else None

def normal_members(stream,root):
    # Even an archive carrying official tests extracts only train/good PNGs.
    chosen=[]
    for member in stream.getmembers():
        rel=PurePosixPath(member.name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in member.name or ':' in member.name:raise ValueError('unsafe dataset path')
        if member.issym() or member.islnk():raise ValueError('dataset links forbidden')
        if member.isfile() and len(rel.parts)>=4 and rel.parts[-3:-1]==('train','good') and rel.suffix=='.png':
            destination(root,rel.as_posix());chosen.append(member)
    if not chosen:raise ValueError('no MVTec train/good images in archive')
    return chosen

def prepare(config):
    import torch
    from training.hope_runtime_v4 import configure_runtime,runtime_identity
    # Fail in seconds, before restoring gigabytes or extracting data.
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable. Enable GPU T4 x2 in Kaggle Settings and create a NEW GPU version; do not pip-upgrade torch.')
    configure_runtime();runtime=runtime_identity()
    inputs,work,w6,assets=map(Path,(config['inputs'],config['work'],config['w6'],config['assets']))
    work.mkdir(parents=True,exist_ok=True);w6.mkdir(parents=True,exist_ok=True)
    parent=find_reviewed(inputs,RECOVERY_MANIFEST,config.get('RECOVERY_OVERRIDE'))
    w6parent=find_reviewed(inputs,W6_MANIFEST_SHA256,config.get('W6_OVERRIDE'))
    expected=json.loads(read_entry(w6parent,'pipeline_lock.json'))['runtime']
    print('[BOOTSTRAP] runtime',json.dumps(runtime),flush=True)
    if canon(runtime)!=canon(expected):raise ValueError('runtime differs from reviewed W6; preserve failure and request parity review, do not upgrade packages blindly')
    previous=discover_resume(inputs,config.get('RESUME_OVERRIDE'))
    if previous:
        old=json.loads(read_entry(previous,'diagnostic_lock.json'))
        if old['identity']['source_digest']!=source_identity() or old['identity']['protocol_digest']!=digest(PROTOCOL):raise ValueError('resume source/protocol changed')
        restore_archive(previous,work,schema=SCHEMA)
    keep={'recovery_lock.json','assets_receipt.json','handoff.json','reference_w7/suite_lock.json','reference_w7/reference_import.json','reference_w7/runs/M_seed0/run_lock.json','reference_w7/runs/M_seed0/manifest_train.json','reference_w7/runs/M_seed0/completion.json','diagnostics/M_seed0/sample_manifest.json'}
    keep.update(f'reference_w7/runs/M_seed0/states/task_{t:02d}.pt' for t in PROTOCOL['checkpoint_indices'])
    imported=verified_import(parent,work/'parents/recovery',RECOVERY_MANIFEST,keep,'hope_recovery_stage1_evidence_v1')
    keep6={'pipeline_lock.json','prefix_features.json',FIT_ENTRY,'baseline_lock/manifest_train.json','baseline_lock/baseline_lock.json'}
    imported6=verified_import(w6parent,w6,W6_MANIFEST_SHA256,keep6,'hope_normal_evidence_v4')
    checkpoint=Path(config['CHECKPOINT_OVERRIDE']) if config.get('CHECKPOINT_OVERRIDE') else unique(inputs.rglob('vit_base_patch8_224.augreg_in21k.safetensors'),'backbone')
    if sha(checkpoint)!=BACKBONE_SHA:raise ValueError('wrong backbone bytes')
    roots=[p.parent for p in inputs.rglob('bottle') if (p/'train/good').is_dir()]
    if config.get('MVTEC_OVERRIDE'):dataset=Path(config['MVTEC_OVERRIDE'])
    elif roots:dataset=unique(roots,'MVTec root')
    else:
        archive=unique([*inputs.rglob('*mvtec*.tar.xz'),*inputs.rglob('*mvtec*.tar.gz')],'MVTec archive')
        print('[BOOTSTRAP] extracting ONLY train/good PNGs',flush=True);assets.mkdir(parents=True,exist_ok=True)
        with tarfile.open(archive) as stream:stream.extractall(assets,members=normal_members(stream,assets),filter='data')
        dataset=unique([p.parent for p in assets.rglob('bottle') if (p/'train/good').is_dir()],'extracted MVTec')
    if not (dataset/'bottle/train/good').is_dir():raise ValueError('invalid MVTec normal root')
    os.environ.update(MVTEC_ROOT=str(dataset),VIT_B8_CHECKPOINT=str(checkpoint),VIT_B8_SHA256=BACKBONE_SHA)
    # Reuse the EXACT reviewed detached 85-image cache, not a newly generated approximation.
    cache=assets/'prefix';cache.mkdir(parents=True,exist_ok=True)
    verified_import(w6parent,cache,W6_MANIFEST_SHA256,{'prefix_features.pt'},'hope_normal_evidence_v4')
    identity=dict(schema='normal_causal_identity_v1',protocol_digest=digest(PROTOCOL),source_digest=source_identity(),runtime=runtime,
        recovery_manifest_sha256=RECOVERY_MANIFEST,w6_manifest_sha256=W6_MANIFEST_SHA256,backbone_sha256=BACKBONE_SHA,
        imported_recovery=imported,imported_w6=imported6,prefix_cache_sha256=sha(cache/'prefix_features.pt'))
    locked_json(work/'diagnostic_lock.json',identity)
    atomic_json(dict(protocol=PROTOCOL,official_test_accessed=False,visa_accessed=False,reused_normal_samples_are_not_independent=True),work/'scope.json')
    return dict(dataset=str(dataset),checkpoint=str(checkpoint),prefix_cache=str(cache/'prefix_features.pt'),identity=identity)

def unique(paths,label):
    paths=sorted(set(map(Path,paths)))
    if len(paths)!=1:raise ValueError(f'{label}: expected one input, got {paths}; set override')
    return paths[0]
