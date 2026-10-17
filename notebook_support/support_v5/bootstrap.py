"""Three-input v5 bootstrap; require CUDA before data extraction."""
from pathlib import Path,PurePosixPath
import json,tarfile,zipfile,os
from .common import SCHEMA,ROOT,configure_runtime,runtime_identity,source_identity
from models.hope_cad_v5.contract import CONTRACT,file_sha,canonical,digest
from notebook_support.common.evidence import restore_archive,destination
from training.runtime import atomic_json
from notebook_support.common.evidence import read_manifest

def one(values,label):
    values=sorted(set(map(Path,values)))
    if len(values)!=1:raise ValueError(f'{label}: expected ONE input; found {values}; set its override')
    return values[0]

def normal_members(archive,root):
    chosen=[]
    for m in archive.getmembers():
        p=PurePosixPath(m.name)
        if p.is_absolute() or '..' in p.parts or '\\' in m.name or ':' in m.name or m.issym() or m.islnk():raise ValueError('unsafe normal dataset archive')
        if m.isfile() and len(p.parts)>=4 and p.parts[-3:-1]==('train','good') and p.suffix=='.png':destination(root,p.as_posix());chosen.append(m)
    if not chosen:raise ValueError('no normal train/good PNGs in archive')
    return chosen

def resume_source(inputs,override=None):
    if override:
        p=Path(override)
        if read_manifest(p)[0]['schema']!=SCHEMA:raise ValueError('wrong v5 resume schema')
        return p
    candidates=[]
    for p in Path(inputs).rglob('artifact_manifest.json'):
        if json.loads(p.read_text())['schema']==SCHEMA:candidates.append(p.parent)
    if not candidates:
        for p in Path(inputs).rglob('*.zip'):
            try:
                if read_manifest(p)[0]['schema']==SCHEMA:candidates.append(p)
            except (KeyError,ValueError,zipfile.BadZipFile):pass
    if len({read_manifest(p)[1] for p in candidates})>1:raise ValueError('attach only latest v5 pilot output or set RESUME_OVERRIDE')
    return sorted(candidates)[0] if candidates else None

def prepare(config):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Enable GPU T4 x2 in Kaggle Settings, create a GPU version; do not pip-upgrade torch')
    configure_runtime();runtime=runtime_identity();expected=json.loads((ROOT/'notebook_support/support_v5/runtime_lock.json').read_text())
    if {k:runtime[k] for k in expected}!=expected:raise ValueError('Kaggle runtime differs from pinned T4 environment; preserve evidence and request parity review')
    print('[V5 RUNTIME]',json.dumps(runtime),flush=True)
    inputs,work,assets=map(Path,(config['inputs'],config['work'],config['assets']));work.mkdir(parents=True,exist_ok=True)
    previous=resume_source(inputs,config.get('RESUME_OVERRIDE'))
    if previous:
        m,_=read_manifest(previous)
        if 'pipeline_lock.json' not in m['files']:raise ValueError('failed bootstrap output has no resumable state; detach it and restart with three inputs')
        if previous.is_dir():raw=(previous/'pipeline_lock.json').read_bytes()
        else:
            with zipfile.ZipFile(previous) as archive:raw=archive.read('pipeline_lock.json')
        old=json.loads(raw)['identity']
        if old['source_digest']!=source_identity() or old['design_digest']!=digest(CONTRACT):raise ValueError('v5 resume source/design identity changed')
        restore_archive(previous,work,schema=SCHEMA)
    checkpoint=Path(config['CHECKPOINT_OVERRIDE']) if config.get('CHECKPOINT_OVERRIDE') else one(inputs.rglob('vit_base_patch8_224.augreg_in21k.safetensors'),'ViT-B/8')
    if file_sha(checkpoint)!=CONTRACT['backbone_sha256']:raise ValueError('wrong backbone SHA256')
    roots=[p.parent for p in inputs.rglob('bottle') if (p/'train/good').is_dir()]
    if config.get('MVTEC_OVERRIDE'):dataset=Path(config['MVTEC_OVERRIDE'])
    elif roots:dataset=one(roots,'MVTec root')
    else:
        archive=one([*inputs.rglob('*mvtec*.tar.xz'),*inputs.rglob('*mvtec*.tar.gz')],'MVTec archive')
        assets.mkdir(parents=True,exist_ok=True);print('[V5 BOOTSTRAP] extracting train/good PNGs only',flush=True)
        with tarfile.open(archive) as z:z.extractall(assets,members=normal_members(z,assets),filter='data')
        dataset=one([p.parent for p in assets.rglob('bottle') if (p/'train/good').is_dir()],'extracted MVTec')
    if not (dataset/'bottle/train/good').is_dir():raise ValueError('invalid MVTec normal root')
    os.environ.update(MVTEC_ROOT=str(dataset),VIT_B8_CHECKPOINT=str(checkpoint),VIT_B8_SHA256=CONTRACT['backbone_sha256'])
    prepared=dict(dataset=str(dataset),checkpoint=str(checkpoint),runtime=runtime,source_digest=source_identity())
    atomic_json(prepared,work/'assets_receipt.json');print('[V5 INPUTS READY]',dataset,flush=True);return prepared
