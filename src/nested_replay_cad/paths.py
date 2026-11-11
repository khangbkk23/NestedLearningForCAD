"""Local artifact contract, created on demand under NRCAD_LOCAL_ROOT or repo/.local.

inputs/                    Shared verified dataset and frozen weights.
runs/b319/<id>/            One experiment: package, logs, outputs, receipts, audit.
cache/                     Reproducible downloads and test/library caches.
tmp/                       Incomplete staging; never the sole retained run record.

Pixi environments live in repo/.pixi, outside this artifact tree. Old receipts
remain byte-identical even when migration makes their embedded paths historical.
"""
from datetime import datetime, timezone
import os,re,uuid
from pathlib import Path

REPOSITORY=Path(__file__).resolve().parents[2]

def local_root(root=None):
    value=root if root is not None else os.environ.get('NRCAD_LOCAL_ROOT')
    path=Path(value).expanduser() if value else REPOSITORY/'.local'
    if not path.is_absolute():path=REPOSITORY/path
    path=path.resolve()
    if path==REPOSITORY.resolve() or path==Path(path.anchor):
        raise ValueError('Local artifact root must be a dedicated directory')
    return path

def run_id():
    return 'b319-s0-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]

def run_root(identifier=None,root=None):
    identifier=identifier or run_id()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',identifier) or '..' in identifier:
        raise ValueError('Unsafe run ID')
    return local_root(root)/'runs'/'b319'/identifier

def inputs(root=None):return local_root(root)/'inputs'
def weights(root=None):return inputs(root)/'weights'/'dinov2_vitb14_reg4_pretrain.pth'
def dataset(root=None):return inputs(root)/'mvtec'
def cache(root=None):return local_root(root)/'cache'
def temporary(root=None):return local_root(root)/'tmp'

def process_environment(root=None):
    base=local_root(root);temp=temporary(base);temp.mkdir(parents=True,exist_ok=True)
    os.environ.update(NRCAD_LOCAL_ROOT=str(base),TEMP=str(temp),TMP=str(temp),TMPDIR=str(temp),PYTHONDONTWRITEBYTECODE='1',
                      TORCH_HOME=str(cache(base)/'torch'),
                      TORCHINDUCTOR_CACHE_DIR=str(cache(base)/'torch/inductor'),
                      TORCH_EXTENSIONS_DIR=str(cache(base)/'torch/extensions'),
                      TRITON_CACHE_DIR=str(cache(base)/'triton'),HF_HOME=str(cache(base)/'huggingface'),
                      XDG_CACHE_HOME=str(cache(base)))
