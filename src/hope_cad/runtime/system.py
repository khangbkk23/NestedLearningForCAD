"""Shared precision, crash-safe writes and resource/RNG receipts.
Bodies retained from verified v3/v4 utilities; no historical source ZIP dependency.
"""
import json,os,platform,sys,random
import numpy as np
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
import torch
from hope_cad.paths import ROOT

def configure_runtime():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

def runtime_identity():
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'timm', 'safetensors', 'scikit-learn', 'PyYAML'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(python=platform.python_version(), packages=packages,
                device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                matmul_precision=torch.get_float32_matmul_precision(),
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark)

def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    # Same-filesystem replacement never removes the preceding good checkpoint.
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def resource_usage(device='cpu'):
    if sys.platform == 'win32':
        try:
            import psutil
            peak_host = psutil.Process().memory_info().peak_wset
        except (ImportError, AttributeError):
            peak_host = None
    else:
        import resource
        peak_host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    result = dict(host_process_peak_bytes=peak_host)
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
        result.update(gpu=torch.cuda.get_device_name(device),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result

def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state().clone(),
                cuda=[v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])

def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state['cuda']])

from hope_cad.data.manifest import canonical_digest

def lock_json(path, identity):
    path = Path(path)
    record = dict(identity=identity, digest=canonical_digest(identity))
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != record:
            raise ValueError(f'locked identity changed: {path}')
    else:
        atomic_json(record, path)
    return record

