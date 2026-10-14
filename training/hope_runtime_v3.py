"""Shared precision, content identities and crash-safe artifacts for v3 runners."""
import json
import os
import platform
import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import torch
from dataset.benchmark_protocol_v2 import file_hash

ROOT = Path(__file__).resolve().parents[1]
SPECS = tuple('docs/main_spec/' + name for name in (
    'architecture_decision_v3.md', 'instruction_CAD_v3.md', 'planning_CAD_v3.md'))


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


def source_identity():
    paths = set()
    for folder in ('models/hope_cad_v2', 'scripts/hope_cad_v2', 'tests/hope_cad_v2'):
        paths.update((ROOT / folder).glob('*.py'))
    paths.update((ROOT / 'training').glob('hope_*.py'))
    paths.update(ROOT / p for p in SPECS)
    paths.update(ROOT / p for p in (
        'dataset/hope_mask_views.py', 'dataset/benchmark_protocol_v2.py',
        'dataset/benchmark_protocol_v1.py', 'dataset/benchmark_manifest_v1.py',
        'models/feature_extractors/cadic_vit_v1.py',
        'training/benchmark_config_v1.py', 'training/benchmark_config_v2.py',
        'training/benchmark_engine_v1.py', 'training/benchmark_metrics_v1.py',
        'training/benchmark_artifacts_v1.py', 'models/benchmark_registry_v2.py'))
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(paths)}


def learning_source_identity():
    """Dependencies of the learned recipe; adding a reporting CLI need not refit it."""
    files = set(SPECS)
    files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'models/hope_cad_v2').glob('*.py')
                 if p.name != 'benchmark_adapter.py')
    files.update('training/'+name for name in (
        'hope_outer_learning_v2.py','hope_selection_v3.py','hope_cms_stress_v3.py',
        'hope_search_journal_v3.py','hope_feature_contract_v3.py','hope_initialization_v3.py',
        'hope_runtime_v3.py'))
    files.update(('scripts/hope_cad_v2/calibrate_features.py', 'scripts/hope_cad_v2/diagnostics.py'))
    return {p: file_hash(ROOT/p) for p in sorted(files)}


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
