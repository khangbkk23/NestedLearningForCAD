"""Pinned parents and a finite diagnostic budget, independent of anomaly metrics."""
from pathlib import Path
import json
from notebook_support.recovery_stage1.common import sha, canon, digest, read_manifest, find_reviewed, locked_json
from training.hope_runtime_v4 import atomic_json, atomic_torch_save
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=json.loads((ROOT/'docs/main_spec/normal_causal_diagnostic_v1.json').read_text(encoding='utf-8'))
SCHEMA='hope_normal_causal_evidence_v1'
SOURCE_SCHEMA='hope_normal_causal_source_v1'
RECOVERY_MANIFEST='087f167eb0fbcb935e36148ceaf64d373aa196894e928ac9b17aad49b46db6cc'
R4_ZIP_SHA='2ad2a4a29c6aa99f1ad5fbbb337ca7e3f1462d722c83bc19f69fafa27a68923d'
FIT_ENTRY='selection/M/M_b0.01_lr0.0001_fit_resume.pt'

def source_identity():
    # All executable dependencies remain bound, including the immutable r4 base.
    manifest=json.loads((ROOT/'bundle_manifest.json').read_text(encoding='utf-8')) if (ROOT/'bundle_manifest.json').exists() else None
    if manifest and manifest.get('schema')==SOURCE_SCHEMA:
        files=manifest['files']
        if any(sha(ROOT/n)!=h for n,h in files.items()):raise ValueError('packaged source changed')
        return digest(files)
    import zipfile
    base=ROOT/'ultils/recovery_stage1/hope_cad_recovery_stage1_source_r4.zip'
    if sha(base)!=R4_ZIP_SHA:raise ValueError('r4 dependency archive changed')
    with zipfile.ZipFile(base) as z: files=json.loads(z.read('bundle_manifest.json'))['files']
    if any(sha(ROOT/n)!=h for n,h in files.items()):raise ValueError('live frozen r4 dependencies changed')
    paths=list((ROOT/'notebook_support/normal_causal').glob('*.py'))+list((ROOT/'tests/normal_causal').glob('*.py'))
    paths+=list((ROOT/'docs/main_spec').glob('normal_causal_diagnostic_v1.*'))+list((ROOT/'notebooks').glob('hope_cad_normal_causal*.ipynb'))
    files.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    return digest(files)

def legal_rows(train_manifest,sample_manifest,checkpoint):
    if checkpoint not in PROTOCOL['checkpoint_indices']:raise ValueError('unregistered checkpoint')
    online={r['relative_path']:r for r in train_manifest['online']}
    tasks=train_manifest['task_order']
    current=sample_manifest['samples'][tasks[checkpoint]]
    past=sorted(set([0,checkpoint//2]))
    selected={t:sample_manifest['samples'][tasks[t]] for t in [checkpoint,*past]}
    import hashlib
    for t,rows in selected.items():
        expected=sorted([r for r in online.values() if r['task_id']==t],key=lambda r:hashlib.sha256(('recovery-normal-v1-20261016|'+r['relative_path']).encode()).hexdigest())[:8]
        if len(rows)!=8 or rows!=expected:raise ValueError('normal sample selection differs from pinned hash rule')
        for row in rows:
            if row!=online.get(row['relative_path']) or row['task_id']>checkpoint or row['split']!='train' or '/train/good/' not in row['relative_path']:raise ValueError('forbidden normal identity')
    return dict(fit=current[:4],query=current[4:],past=[r for t in past for r in selected[t][4:]],past_tasks=past)

def event_order(cursor,steps=96):
    return [((s+s//4)%4,(cursor+s)%4) for s in range(steps)]
