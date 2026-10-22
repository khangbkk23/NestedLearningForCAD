"""Current source identity and canonical numerical runtime; no historical checkout dependency."""
from pathlib import Path
import json,os
from models.hope_cad_v5_1.contract import CONTRACT,digest,file_sha,canonical,ROOT
SCHEMA='hope_support_v5_1_normal_pilot_evidence_v1'
SOURCE_SCHEMA='hope_support_v5_1_source_v1'

def source_files():
    policy=json.loads((ROOT/'conf/delivery/support_v5_1_source.json').read_text(encoding='utf-8'))
    if policy['schema']!='hope_support_source_inventory_v5_1':raise ValueError('wrong source inventory schema')
    files=policy['files']
    if files!=sorted(set(files)):raise ValueError('source inventory must be unique and sorted')
    for name in files:
        path=ROOT/name
        if not path.resolve().is_relative_to(ROOT.resolve()) or not path.is_file() or path.is_symlink():raise ValueError('unsafe/missing source inventory entry '+name)
    return {name:file_sha(ROOT/name) for name in files}

def source_identity():
    actual=source_files();path=ROOT/'bundle_manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text(encoding='utf-8'))
        if manifest['schema']!=SOURCE_SCHEMA or manifest['files']!=actual:raise ValueError('packaged source identity drift')
    return digest(actual)

def artifact_root():
    return Path(os.environ.get('HOPE_ARTIFACTS_ROOT',str(ROOT.parent/'.agents/artifacts'))).resolve()

def delivery_directory():return artifact_root()/'deliveries/hope_support_v5_1_r1'

def configure_runtime():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from training.runtime import configure_runtime as base
    base();torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)

def runtime_identity():
    import torch
    from training.runtime import runtime_identity as base
    return dict(base(),v5_1_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),v5_1_cudnn_deterministic=torch.backends.cudnn.deterministic,v5_1_cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def source_revision():
    path=ROOT/'bundle_manifest.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))['origin'].get('source_commit')
    import subprocess
    try:return subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True,timeout=10).strip()
    except (OSError,subprocess.SubprocessError):return None
