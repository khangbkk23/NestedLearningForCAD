# scripts/benchmarks/common/config.py
import os
from pathlib import Path
import yaml

def _expand(value):
    if isinstance(value, dict): return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list): return [_expand(v) for v in value]
    if isinstance(value, str):
        old = None
        while old != value:
            old = value
            value = os.path.expandvars(value)
        return value
    return value

def load_yaml(path: str | Path, *, expand=True) -> dict:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return _expand(data) if expand else data

def resolve_root(value: str | Path, base: str | Path | None = None) -> Path:
    p = Path(value).expanduser()
    return (Path(base) / p).resolve() if base and not p.is_absolute() else p.resolve()

def load_protocol(path): return load_yaml(path)
def load_method(path): return load_yaml(path)
