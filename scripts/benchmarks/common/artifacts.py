# scripts/benchmarks/common/artifacts.py
import json, os, tempfile
from pathlib import Path

def atomic_json(path: str | Path, value) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f: json.dump(value, f, indent=2, sort_keys=True, default=str); f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return path

def read_json(path, default=None):
    p = Path(path)
    return default if not p.is_file() else json.loads(p.read_text())

def ensure_run_dir(path: str | Path, *, overwrite=False) -> Path:
    p = Path(path).expanduser().resolve()
    if p.exists() and any(p.iterdir()) and not overwrite:
        raise FileExistsError(f"non-empty run directory: {p}")
    p.mkdir(parents=True, exist_ok=True)
    for name in ("metrics", "normalized_metrics", "timing", "resource", "states", "logs", "author_outputs", "replay"):
        (p / name).mkdir(exist_ok=True)
    return p

def stage_marker(run_dir, stage, status, **details):
    return atomic_json(Path(run_dir) / "timing" / f"{stage}.json", {"stage": stage, "status": status, **details})

def command_file(run_dir, commands):
    p = Path(run_dir) / "commands.json"
    return atomic_json(p, commands)
