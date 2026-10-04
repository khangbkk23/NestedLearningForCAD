# scripts/benchmarks/common/resources.py
import os, time
from pathlib import Path

def directory_bytes(path):
    total = 0; count = 0
    for p in Path(path).rglob("*") if Path(path).exists() else ():
        if p.is_file() and not p.is_symlink(): total += p.stat().st_size; count += 1
    return {"bytes": total, "files": count}

def measure_stage(start, **extra): return {"wall_seconds": max(0.0, time.monotonic() - start), **extra}

def resource_snapshot(run_dir):
    p = Path(run_dir)
    return {"generated_replay": directory_bytes(p / "author_outputs" / "generated"), "conditions": directory_bytes(p / "author_outputs" / "conditions"), "checkpoints": directory_bytes(p / "author_outputs" / "invad")}
