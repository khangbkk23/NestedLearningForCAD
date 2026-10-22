import json,os,tarfile,zipfile
from pathlib import Path,PurePosixPath
from models.hope_cad_v5_1.contract import ROOT,CONTRACT,file_sha,design_digest
from notebook_support.support_v5_1.common import configure_runtime,runtime_identity
from notebook_support.support_v5_1.bootstrap import one
from notebook_support.common.evidence import read_manifest,restore_archive,destination
from training.runtime import atomic_json
from training.support_benchmark_v5_1.contract import EVIDENCE_SCHEMA,source_digest,study_digest,job

def sources(inputs,override=None):
    if override:return Path(override)
    choices=[]
    for p in Path(inputs).rglob('artifact_manifest.json'):
        if json.loads(p.read_text()).get('schema')==EVIDENCE_SCHEMA:choices.append(p.parent)
    if not choices:
        for p in Path(inputs).rglob('*.zip'):
            try:
                if read_manifest(p)[0].get('schema')==EVIDENCE_SCHEMA:choices.append(p)
            except (ValueError,KeyError,zipfile.BadZipFile):pass
    if len({read_manifest(p)[1] for p in choices})>1:raise ValueError('Attach only latest benchmark output, remove older/preflight parent or set RESUME_OVERRIDE')
    return sorted(choices)[0] if choices else None

def entry(source,name):
    if Path(source).is_dir():return (Path(source)/name).read_bytes()
    with zipfile.ZipFile(source) as z:return z.read(name)

def selected_payload(name,selected_job):
    job(selected_job)
    return name.startswith('runs/'+selected_job+'/') or name in ('suite_lock.json','benchmark_preflight.json','preflight.json','manifest_train.json','test_manifest.json','data_rights.json')

def prepare(c):
    import torch
    if not torch.cuda.is_available():raise ValueError('GPU T4 required; no pip upgrades')
    configure_runtime();runtime=runtime_identity();expected=json.loads((ROOT/'notebook_support/support_v5_1/runtime_lock.json').read_text())
    if {k:runtime[k] for k in expected}!=expected:raise ValueError('Pinned T4 runtime mismatch; stop for parity review')
    inputs,work,assets=map(Path,(c['inputs'],c['work'],c['assets']));work.mkdir(parents=True,exist_ok=True)
    previous=sources(inputs,c.get('RESUME_OVERRIDE'))
    if previous:
        manifest,_=read_manifest(previous)
        if 'suite_lock.json' not in manifest['files']:raise ValueError('Failed bootstrap is not resumable')
        old=json.loads(entry(previous,'suite_lock.json'))['identity']
        if old['source_digest']!=source_digest() or old['study_digest']!=study_digest() or old['runtime']!=runtime:raise ValueError('Resume source/study/runtime changed')
        # A job's learned tensors never migrate into a different registered job.
        restore_archive(previous,work,schema=EVIDENCE_SCHEMA,select=lambda n:selected_payload(n,c['job']))
    checkpoint=Path(c['CHECKPOINT_OVERRIDE']) if c.get('CHECKPOINT_OVERRIDE') else one(inputs.rglob('vit_base_patch8_224.augreg_in21k.safetensors'),'ViT-B/8')
    if file_sha(checkpoint)!=CONTRACT['backbone_sha256']:raise ValueError('Wrong backbone bytes')
    roots=[p.parent for p in inputs.rglob('bottle') if (p/'train/good').is_dir()]
    if c.get('MVTEC_OVERRIDE'):dataset=Path(c['MVTEC_OVERRIDE'])
    elif roots:dataset=one(roots,'MVTec root')
    else:
        archive=one([*inputs.rglob('*mvtec*.tar.xz'),*inputs.rglob('*mvtec*.tar.gz')],'MVTec archive')
        assets.mkdir(parents=True,exist_ok=True);chosen=[]
        print('[BENCH BOOTSTRAP] extracting MVTec images/masks',flush=True)
        with tarfile.open(archive) as z:
            for m in z.getmembers():
                p=PurePosixPath(m.name)
                if p.is_absolute() or '..' in p.parts or '\\' in m.name or ':' in m.name or m.issym() or m.islnk():raise ValueError('Unsafe data tar')
                allowed=(len(p.parts)>=4 and p.parts[-3:-1]==('train','good'))
                # Preflight never needs official test/GT bytes.
                if c['phase']!='preflight':allowed=allowed or any(x in ('test','ground_truth') for x in p.parts)
                if m.isfile() and allowed and p.suffix=='.png':destination(assets,p.as_posix());chosen.append(m)
            z.extractall(assets,members=chosen,filter='data')
        dataset=one([p.parent for p in assets.rglob('bottle') if (p/'train/good').is_dir()],'extracted MVTec')
    os.environ.update(MVTEC_ROOT=str(dataset),VIT_B8_CHECKPOINT=str(checkpoint),VIT_B8_SHA256=CONTRACT['backbone_sha256'])
    value=dict(dataset=str(dataset),checkpoint=str(checkpoint),runtime=runtime,source_digest=source_digest())
    atomic_json(value,work/'assets_receipt.json');print('[BENCH INPUTS READY]',json.dumps(value),flush=True);return value
