"""Verify exact original parent and runtime before loading any new normal image."""
from pathlib import Path
from training.runtime import atomic_json
from training.support_throughput_v5_1.contract import select_evidence,Evidence,verify_parent,sha,contract as profile_contract
from training.support_migration_v5_1.contract import verify_source,contract,SCHEMA
from notebook_support.support_throughput_v5_1.bootstrap import one

def prepare(c):
    import torch
    from notebook_support.support_v5_1.common import configure_runtime,runtime_identity
    source=verify_source();cfg=contract();pc=profile_contract();configure_runtime();runtime=runtime_identity()
    if not torch.cuda.is_available():raise ValueError('Enable GPU T4; no package upgrades or CPU bridge')
    inputs=Path(c['inputs']);work=Path(c['work'])
    if select_evidence(inputs,schema=SCHEMA) is not None:raise ValueError('Bridge output is for review, not automatic retry/resume. Attach original376parent only.')
    parent=select_evidence(inputs,schema='hope_support_v5_1_benchmark_evidence_v1',manifest_sha=pc['parent']['artifact_manifest_sha256'],override=c.get('PARENT_OVERRIDE'))
    with Evidence(parent) as e:
        if runtime!=e.json('suite_lock.json')['identity']['runtime']:raise ValueError('Runtime differs from reviewed parent; stop for review')
        info=verify_parent(e,pc)
    print('[BRIDGE RUNTIME]',runtime,flush=True)
    checkpoint=Path(c['CHECKPOINT_OVERRIDE']) if c.get('CHECKPOINT_OVERRIDE') else one(inputs.rglob('vit_base_patch8_224.augreg_in21k.safetensors'),'ViT-B/8')
    if sha(checkpoint)!=pc['frozen_reference']['backbone_sha256']:raise ValueError('Backbone SHA changed')
    if c.get('MVTEC_OVERRIDE'):dataset=Path(c['MVTEC_OVERRIDE']);archive=None
    else:
        roots=[p.parent for p in inputs.rglob('cable') if (p/'train/good').is_dir()]
        if roots:dataset=one(roots,'MVTec root');archive=None
        else:
            archive=one([*inputs.rglob('*mvtec*.tar.xz'),*inputs.rglob('*mvtec*.tar.gz')],'MVTec tar')
            dataset=Path(c['assets']);dataset.mkdir(parents=True,exist_ok=True)
    if any((work/n).exists() for n in ['dispatch.json','shadow','benchmark','.publication_pending']):raise ValueError('Previous bridge attempt exists; preserve for review, never silently retry')
    result=dict(parent=str(parent),dataset=str(dataset),dataset_archive=str(archive) if archive else None,checkpoint=str(checkpoint),source=source,runtime=runtime)
    atomic_json(dict(**result,raw_images_pre_extracted=0,original_parent_written=False),work/'bootstrap.json')
    print('[BRIDGE READY] parent376 verified; no new images read; fixed8arrivals',flush=True)
    return result
