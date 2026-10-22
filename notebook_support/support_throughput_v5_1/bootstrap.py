"""Select exact reviewed inputs; extract only the sixteen permitted normals."""
import json, os, tarfile
from pathlib import Path, PurePosixPath
from training.support_throughput_v5_1.contract import (contract, verify_source, verify_parent,
    select_evidence, Evidence, sha, safe, SCHEMA)

def one(values,label):
    values=sorted(set(map(Path,values)))
    if len(values)!=1:raise ValueError(label+': expected ONE, got '+str(values)+'; set override')
    return values[0]

def extract_window(archive,assets,rows):
    """No test/GT or other train image extraction, even from a full dataset tar."""
    assets=Path(assets); assets.mkdir(parents=True,exist_ok=True)
    wanted={r['relative_path']:r for r in rows}; found=set()
    with tarfile.open(archive) as z:
        for m in z:
            p=PurePosixPath(m.name)
            if p.is_absolute() or '..' in p.parts or '\\' in m.name or ':' in m.name or m.issym() or m.islnk():
                raise ValueError('Unsafe dataset tar')
            if not m.isfile():continue
            matches=[n for n in wanted if p.as_posix()==n or p.as_posix().endswith('/'+n)]
            if not matches:continue
            if len(matches)!=1 or matches[0] in found:raise ValueError('Duplicate permitted image in tar')
            name=matches[0];dest=safe(assets,name);dest.parent.mkdir(parents=True,exist_ok=True)
            with z.extractfile(m) as reader,dest.open('wb') as out:
                import shutil
                shutil.copyfileobj(reader,out)
            found.add(name)
            if found==set(wanted):break  # No reason to scan/decompress later unobserved images or GT.
    if found!=set(wanted):raise ValueError('Tar lacks registered normal window')
    return assets

def verify_window(dataset,rows):
    from dataset.benchmark_protocol_v2 import file_hash
    for row in rows:
        if row.get('split')!='train' or '/train/good/' not in row['relative_path']:raise ValueError('Only registered normals allowed')
        if file_hash(safe(dataset,row['relative_path']))!=row['sha256']:raise ValueError('Normal raw bytes changed')

def prepare(c):
    import torch
    from notebook_support.support_v5_1.common import configure_runtime,runtime_identity
    from notebook_support.common.evidence import restore_archive
    from training.runtime import atomic_json
    cfg=contract();source=verify_source();inputs=Path(c['inputs']);work=Path(c['work'])
    configure_runtime();runtime=runtime_identity()
    if not torch.cuda.is_available():raise ValueError('GPU T4 required; enable accelerator; never pip-upgrade to bypass')
    parent=select_evidence(inputs,schema='hope_support_v5_1_benchmark_evidence_v1',
        manifest_sha=cfg['parent']['artifact_manifest_sha256'],override=c.get('PARENT_OVERRIDE'))
    with Evidence(parent) as e:
        # Runtime comparison precedes large archive hashing, extraction and actor creation.
        expected=e.json('suite_lock.json')['identity']['runtime']
        print('[PROFILE RUNTIME]',json.dumps(runtime),flush=True)
        if runtime!=expected:raise ValueError('Runtime differs from exact reviewed parent; stop for review')
        verified=verify_parent(e,cfg)
    previous=select_evidence(inputs,schema=SCHEMA,override=c.get('RESUME_OVERRIDE'))
    if previous:
        with Evidence(previous) as e:
            e.verify_all();old=e.json('profile_lock.json')
            identity=dict(source=source,parent_manifest_sha256=cfg['parent']['artifact_manifest_sha256'],runtime=runtime)
            if old!=identity:raise ValueError('Profiler resume identity changed')
        restore_archive(previous,work,schema=SCHEMA)
    checkpoint=Path(c['CHECKPOINT_OVERRIDE']) if c.get('CHECKPOINT_OVERRIDE') else one(inputs.rglob('vit_base_patch8_224.augreg_in21k.safetensors'),'ViT-B/8')
    if sha(checkpoint)!=cfg['frozen_reference']['backbone_sha256']:raise ValueError('Wrong checkpoint SHA')
    f=cfg['profile'];rows=verified['rows'][f['start_cursor']:f['end_cursor_exclusive']]
    if c.get('MVTEC_OVERRIDE'):dataset=Path(c['MVTEC_OVERRIDE'])
    else:
        roots=[p.parent for p in inputs.rglob('cable') if (p/'train/good/000.png').is_file()]
        if roots:dataset=one(roots,'MVTec root')
        else:
            archive=one([*inputs.rglob('*mvtec*.tar.xz'),*inputs.rglob('*mvtec*.tar.gz')],'MVTec tar')
            print('[PROFILE ASSETS] extracting only16 previously observed train/good images',flush=True)
            dataset=extract_window(archive,Path(c['assets']),rows)
    verify_window(dataset,rows)
    work.mkdir(parents=True,exist_ok=True)
    lock=dict(source=source,parent_manifest_sha256=cfg['parent']['artifact_manifest_sha256'],runtime=runtime)
    path=work/'profile_lock.json'
    if path.exists() and json.loads(path.read_text())!=lock:raise ValueError('Local profiler lock changed')
    atomic_json(lock,path)
    atomic_json(dict(rows=rows,start_cursor=179,end_cursor_exclusive=195,window_sha256=cfg['profile']['window_rows_canonical_sha256']),work/'window.json')
    atomic_json(dict(parent=str(parent),representation=verified['representation'],verified_payloads=verified['verified_payloads'],
        archive_SHA_required_for_ZIP=True,directory_identity='exact reviewed manifest and all payload hashes',original_parent_written=False),work/'parent_receipt.json')
    return dict(parent=str(parent),dataset=str(dataset),checkpoint=str(checkpoint),runtime=runtime,source=source)
