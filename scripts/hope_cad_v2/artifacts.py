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
    from hope_cad.runtime import source_files
    hashes=source_files();manifest_path=ROOT/'bundle_manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('files')!=hashes:raise ValueError('Packaged baseline source drift')
        recorded=manifest.get('origin',{})
        origin={k:recorded[k] for k in ('source_commit','branch','git_dirty','tracked_diff_sha256','merge_parent') if k in recorded}
        origin['execution_checkout']='verified_source_archive'
    else:
        def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True,stderr=subprocess.DEVNULL).strip()
        try:
            origin=dict(source_commit=git('rev-parse','HEAD'),branch=git('branch','--show-current'),git_dirty=bool(git('status','--porcelain')))
            diff=subprocess.check_output(['git','-C',str(ROOT),'diff','HEAD'],stderr=subprocess.DEVNULL)
            origin['tracked_diff_sha256']=hashlib.sha256(diff).hexdigest()
        except (OSError,subprocess.CalledProcessError):origin=dict(source_commit=None,execution_checkout='unversioned_live_source')
    return dict(**origin,source_sha256=hashes,python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda_runtime=torch.version.cuda,
                cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)
