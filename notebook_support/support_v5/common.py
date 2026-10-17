from pathlib import Path
import json,os
from models.hope_cad_v5.contract import CONTRACT,digest,file_sha,canonical,ROOT
SCHEMA='hope_support_v5_normal_pilot_evidence_v1'
SOURCE_SCHEMA='hope_support_v5_source_v1'
R4_SHA='2ad2a4a29c6aa99f1ad5fbbb337ca7e3f1462d722c83bc19f69fafa27a68923d'

def source_identity():
    path=ROOT/'bundle_manifest.json'
    if path.exists():
        m=json.loads(path.read_text(encoding='utf-8'))
        if m['schema']!=SOURCE_SCHEMA:raise ValueError('wrong packaged v5 source schema')
        for n,h in m['files'].items():
            if file_sha(ROOT/n)!=h:raise ValueError('packaged source drift '+n)
        return digest(m['files'])
    paths=list((ROOT/'models/hope_cad_v5').glob('*.py'))+list((ROOT/'notebook_support/support_v5').glob('*.py'))+list((ROOT/'tests/hope_cad_v5').glob('*.py'))
    paths+=[ROOT/'training/hope_support_v5.py',ROOT/'docs/main_spec/v5_implementation.md']
    paths+=list((ROOT/'docs/main_spec').glob('*_v5.*'))+list((ROOT/'notebooks').glob('hope_cad_support_v5*.ipynb'))
    paths+=list((ROOT/'notebook_support/support_v5').glob('*.json'))
    import zipfile
    base=ROOT/'ultils/recovery_stage1/hope_cad_recovery_stage1_source_r4.zip'
    if file_sha(base)!=R4_SHA:raise ValueError('r4 source template drift')
    with zipfile.ZipFile(base) as z:files=json.loads(z.read('bundle_manifest.json'))['files']
    if any(file_sha(ROOT/n)!=h for n,h in files.items()):raise ValueError('frozen base dependency drift')
    files.update({p.relative_to(ROOT).as_posix():file_sha(p) for p in paths});return digest(files)

def configure_runtime():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from training.hope_runtime_v4 import configure_runtime as base
    base();torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)

def runtime_identity():
    import torch
    from training.hope_runtime_v4 import runtime_identity as base
    return dict(base(),v5_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),v5_cudnn_deterministic=torch.backends.cudnn.deterministic,v5_cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
