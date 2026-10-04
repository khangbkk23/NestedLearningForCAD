# scripts/benchmarks/common/cli.py
import re
from pathlib import Path

TASK_ORDER = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]

def tasks_arg(value: str | None) -> list[str]:
    if not value:
        return list(TASK_ORDER)
    values = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in values if x not in TASK_ORDER]
    if unknown or len(set(values)) != len(values):
        raise ValueError(f"invalid or duplicate MVTec tasks: {unknown or values}")
    return [x for x in TASK_ORDER if x in values]

def safe_run_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise ValueError(f"unsafe run name: {value}")
    return value

def path_arg(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()
