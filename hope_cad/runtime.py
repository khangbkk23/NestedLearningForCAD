"""Explicit deterministic runtime and source identities, no Kaggle/sibling dependency."""
import json,os,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def output_directory(path,*,data_root=None,checkpoint=None):
    work=Path(path).resolve()
    if ROOT.is_relative_to(work):raise ValueError('Run directory cannot contain the source checkout')
    if work.is_relative_to(ROOT) and work.relative_to(ROOT).parts[0] not in ('runs','results','logs'):
        raise ValueError('Use runs/results/logs or an external run directory, not source folders')
    if data_root:
        data=Path(data_root).resolve()
        if work.is_relative_to(data) or data.is_relative_to(work):raise ValueError('Run directory must be separate from dataset')
    if checkpoint and Path(checkpoint).resolve().is_relative_to(work):
        raise ValueError('Run directory must not contain checkpoint input')
    return work

def source_files():
    from models.hope_cad_v5_1.contract import file_sha
    policy=json.loads((ROOT/'conf/runtime/source_inventory_v7.json').read_text(encoding='utf-8'))
    names=policy['files']
    if policy['schema']!='hope_cad_portable_source_v1' or names!=sorted(set(names)):
        raise ValueError('Invalid portable source inventory')
    result={}
    for name in names:
        path=ROOT/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError('Missing/unsafe source: '+name)
        result[name]=file_sha(path)
    manifest=ROOT/'bundle_manifest.json'
    if manifest.exists():
        shipped=json.loads(manifest.read_text(encoding='utf-8'))
        if shipped.get('schema')!='hope_cad_portable_source_bundle_v1' or shipped.get('files')!=result:
            raise ValueError('Packaged source bytes differ from verified source bundle')
    return result

def source_digest():
    from models.hope_cad_v5_1.contract import digest
    return digest(source_files())

def configure_runtime():
    # Set before CUDA context creation; preserve the established CPU thread count.
    existing=os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if existing!=':4096:8':raise ValueError('Unexpected CUBLAS_WORKSPACE_CONFIG; use a fresh process')
    import torch
    from training.runtime import configure_runtime as base
    base();torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)

def runtime_identity(device='cpu'):
    import torch,platform
    from training.runtime import runtime_identity as base
    value=base()
    value.update(platform=platform.system(),machine=platform.machine(),device=str(torch.device(device)),
        threads=torch.get_num_threads(),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    if str(device).startswith('cuda'):
        prop=torch.cuda.get_device_properties(device)
        value.update(device_name=prop.name,compute_capability=list(torch.cuda.get_device_capability(device)),
            device_total_memory=prop.total_memory,compiled_architectures=torch.cuda.get_arch_list())
        import subprocess
        try:
            value['nvidia_driver']=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version',
                '--format=csv,noheader'],text=True,timeout=10).strip().splitlines()
        except (OSError,subprocess.SubprocessError):value['nvidia_driver']=None
    return value

def check_remote_runtime(device,expected_gpu):
    import torch
    from importlib.metadata import version
    if sys.platform!='linux':raise ValueError('Remote acceptance requires Linux; local smoke is a different gate')
    if torch.device(device).type!='cuda' or not torch.cuda.is_available():raise ValueError('CUDA GPU required')
    if sys.version_info[:2]!=(3,13):raise ValueError('Candidate remote runtime requires Python3.13')
    pins=json.loads((ROOT/'conf/runtime/linux_cuda128.json').read_text(encoding='utf-8'))
    for name,expected in pins['packages'].items():
        if version(name)!=expected:raise ValueError('Runtime package mismatch: '+name)
    if torch.version.cuda!=pins['cuda']:raise ValueError('CUDA wheel runtime differs from candidate lock')
    actual=torch.cuda.get_device_name(device)
    if actual!=expected_gpu:raise ValueError('Actual GPU differs from explicit expected GPU: '+actual)
    capability=torch.cuda.get_device_capability(device)
    if capability[0]>=12 and ('sm_'+str(capability[0])+str(capability[1])) not in torch.cuda.get_arch_list():
        raise ValueError('Wheel lacks native architecture for selected Blackwell GPU')
    return runtime_identity(device)
