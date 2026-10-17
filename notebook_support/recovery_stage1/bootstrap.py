"""Kaggle input preparation: pinned parents, unchanged W6 runtime, safe extraction, explicit resume."""
from pathlib import Path
import json
import os
import tarfile
from .common import (read_manifest,find_reviewed,import_reference,verify_reference,restore_recovery,SCHEMA,
                     W72_MANIFEST_SHA,BACKBONE_SHA,sha,atomic_json)


def unique(paths,label):
    paths=sorted(set(map(Path,paths)))
    if len(paths)!=1:raise ValueError(f'{label}: expected one input, found {paths}; set its override')
    return paths[0]


def resume_input(inputs,override=None):
    if override:
        source=Path(override)
        if read_manifest(source)[0].get('schema')!=SCHEMA:raise ValueError('not recovery evidence')
        return source
    found=[]
    for p in Path(inputs).rglob('artifact_manifest.json'):
        try:
            if json.loads(p.read_text())['schema']==SCHEMA:found.append(p.parent)
        except (ValueError,KeyError):pass
    if not found:
        for p in Path(inputs).rglob('*.zip'):
            try:
                if read_manifest(p)[0].get('schema')==SCHEMA:found.append(p)
            except (ValueError,KeyError,tarfile.ReadError):pass
            except Exception as error:
                import zipfile
                if not isinstance(error,zipfile.BadZipFile):raise
    if not found:return None
    groups={read_manifest(p)[1] for p in found}
    if len(groups)!=1:raise ValueError('multiple DIFFERENT recovery outputs attached; keep only latest or set RESUME_OVERRIDE')
    return sorted(found)[0]


def prepare(inputs,work,w6,assets,*,phase,w6_override=None,w72_override=None,resume_override=None,
            checkpoint_override=None,mvtec_override=None):
    from training.hope_runtime_v4 import configure_runtime,runtime_identity
    from training.hope_w7_contract import W6_MANIFEST_SHA256
    from training.hope_w7_io import import_w6,destination
    inputs,work,w6,assets=map(Path,(inputs,work,w6,assets))
    work.mkdir(parents=True,exist_ok=True)
    previous=resume_input(inputs,resume_override)
    if previous:
        manifest,_=read_manifest(previous)
        # Verify execution source before restoring any old lock/state.
        import zipfile
        if Path(previous).is_dir():raw=(Path(previous)/'recovery_lock.json').read_bytes()
        else:
            with zipfile.ZipFile(previous) as archive:raw=archive.read('recovery_lock.json')
        from .common import source_identity,digest
        prior_source=json.loads(raw)['identity']['source']
        if digest(prior_source)!=digest(source_identity()):
            raise ValueError('Recovery output belongs to another source delivery. Keep it as evidence; fresh preflight needs original W6 + completed W7.2, not failed r1/r2/r3 outputs.')
        old=prior_source['protocol'].get('cadic_execution',{})
        if old.get('backend')!='cadic_exact_pair_tile_cache_v2':
            raise ValueError('Old failed recovery output is evidence only. Detach it; r4 preflight requires fresh original W6 + completed W7.2 inputs.')
    if previous and not (work/'recovery_lock.json').exists():
        print('[BOOTSTRAP] restoring recovery output',previous,flush=True);restore_recovery(previous,work)
    if phase=='controls' and not (work/'recovery_preflight.json').exists():
        raise ValueError('controls require PASS recovery preflight output; attach that output instead of W7 preflight')
    parent=find_reviewed(inputs,W6_MANIFEST_SHA256,w6_override)
    print('[BOOTSTRAP] importing selected W6 weights/receipts (no feature-cache refit)',parent,flush=True)
    import_w6(parent,w6,include_features=False)
    configure_runtime()
    runtime=runtime_identity();expected=json.loads((w6/'pipeline_lock.json').read_text())['runtime']
    print('[BOOTSTRAP] runtime',json.dumps(runtime),flush=True)
    if runtime!=expected:raise ValueError('Kaggle runtime differs from W6. Stop for parity review; do not upgrade packages blindly.')
    reference=work/'reference_w7'
    if (reference/'reference_import.json').exists():verify_reference(reference)
    else:
        p=find_reviewed(inputs,W72_MANIFEST_SHA,w72_override)
        print('[BOOTSTRAP] importing reviewed M checkpoint/metrics subset from completed 7.2',p,flush=True)
        import_reference(p,reference)
    candidates=list(inputs.rglob('vit_base_patch8_224.augreg_in21k.safetensors'))
    checkpoint=Path(checkpoint_override) if checkpoint_override else unique(candidates,'ViT checkpoint')
    if sha(checkpoint)!=BACKBONE_SHA:raise ValueError('wrong backbone checkpoint bytes')
    roots=[p.parent for p in inputs.rglob('bottle') if (p/'train/good').is_dir()]
    if mvtec_override:dataset=Path(mvtec_override)
    elif roots:dataset=unique(roots,'MVTec root')
    else:
        archive=unique(list(inputs.rglob('*mvtec*.tar.xz'))+list(inputs.rglob('*mvtec*.tar.gz')),'MVTec archive')
        assets.mkdir(parents=True,exist_ok=True)
        print('[BOOTSTRAP] extracting MVTec',archive,flush=True)
        with tarfile.open(archive) as stream:
            for member in stream.getmembers():
                from pathlib import PurePosixPath
                relative=PurePosixPath(member.name)
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe dataset path')
                destination(assets,relative.as_posix())
                if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):raise ValueError('unsafe dataset member')
            stream.extractall(assets,filter='data')
        dataset=unique([p.parent for p in assets.rglob('bottle') if (p/'train/good').is_dir()],'extracted MVTec root')
    if not (dataset/'bottle/train/good').is_dir():raise ValueError('invalid MVTec root')
    os.environ.update(MVTEC_ROOT=str(dataset),VIT_B8_CHECKPOINT=str(checkpoint),VIT_B8_SHA256=BACKBONE_SHA)
    atomic_json(dict(schema='recovery_assets_receipt_v1',W6_manifest_sha256=W6_MANIFEST_SHA256,
        W72_manifest_sha256=W72_MANIFEST_SHA,backbone_sha256=BACKBONE_SHA,
        runtime=runtime,learned_state_reselected=False),work/'assets_receipt.json')
    print('[BOOTSTRAP] inputs ready; dataset:',dataset,'resume:',previous,flush=True)
    return dataset,checkpoint