"""Portable evidence for uncommitted implementation: SHA256 source/spec plus Git status."""
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[2]


def provenance():
    paths = set()
    for folder in ('models/hope_cad_v2', 'scripts/hope_cad_v2', 'tests/hope_cad_v2', 'docs/main_spec'):
        paths.update(p for p in (ROOT/folder).rglob('*') if p.suffix in ('.py', '.md', '.json'))
    paths.update((ROOT/'training').glob('hope_*.py'))
    paths.update((ROOT/'dataset').glob('*v2.py'))
    paths.update(ROOT/p for p in ('dataset/hope_mask_views.py','training/benchmark_config_v2.py','models/benchmark_registry_v2.py'))
    paths.update(ROOT/p for p in ('models/cadic_benchmark_adapter_v1.py','models/cadic_patch_coreset_v1.py',
                  'models/cadic_paperfaithful/cadic_benchmark_adapter.py','models/cadic_paperfaithful/cadic_patch_coreset.py',
                  'models/feature_extractors/cadic_vit_v1.py','dataset/benchmark_protocol_v1.py',
                  'dataset/benchmark_manifest_v1.py','training/benchmark_metrics_v1.py','training/benchmark_engine_v1.py'))
    paths.update((ROOT/'conf/benchmarks').rglob('*.yaml'))
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    try:
        origin = dict(source_commit=git('rev-parse','HEAD'), branch=git('branch','--show-current'),
                      git_dirty=bool(git('status','--porcelain')))
        try:
            diff = subprocess.check_output(['git','diff','HEAD'],cwd=ROOT,stderr=subprocess.DEVNULL)
            origin['tracked_diff_sha256'] = hashlib.sha256(diff).hexdigest()
        except (OSError,subprocess.CalledProcessError):
            # Per-file hashes remain authoritative even in restricted Git sandboxes.
            origin['tracked_diff_sha256'] = None
            origin['git_diff_status'] = 'unavailable'
    except (OSError, subprocess.CalledProcessError):
        archive = ROOT/'bundle_manifest.json'
        recorded = json.loads(archive.read_text(encoding='utf-8')).get('origin', {}) if archive.exists() else {}
        origin = {k:recorded[k] for k in ('source_commit','branch','git_dirty','tracked_diff_sha256') if k in recorded}
        origin = dict(**origin, execution_checkout='archive_without_git')
    return dict(**origin, source_sha256=hashes,
                python=sys.version, platform=platform.platform(), torch=torch.__version__,
                cuda_runtime=torch.version.cuda, cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)
