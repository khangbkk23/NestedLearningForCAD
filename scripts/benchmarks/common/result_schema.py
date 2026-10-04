# scripts/benchmarks/common/result_schema.py
import math
from pathlib import Path
from .artifacts import read_json

def metric(name, value=None, available=None, reason=None, source="reproduced"):
    if available is None: available = value is not None
    return {"metric": name, "value": value, "availability": bool(available), "reason": reason if not available else None, "source": source}

def unavailable(name, reason, source="reproduced"): return metric(name, None, False, reason, source)

def validate_metric(value):
    if not isinstance(value, dict) or "metric" not in value or "availability" not in value: raise ValueError("invalid metric record")
    if value["availability"] and value.get("value") is None: raise ValueError("available metric has null value")
    return value

def normalize_scalar(name, value, *, source="reproduced", reason=None):
    if value is None or (isinstance(value, float) and math.isnan(value)): return unavailable(name, reason or "not_reported", source)
    return metric(name, float(value), True, None, source)

def load_normalized(run_dir: str | Path) -> dict:
    p = Path(run_dir)
    for candidate in (p / "normalized_metrics" / "final_macro.json", p / "metrics" / "final_macro.json"):
        if candidate.is_file(): return read_json(candidate, {})
    summary = read_json(p / "summary.json", {})
    return {"metrics": {k: normalize_scalar(k, summary.get(k), source="reproduced") for k in ("i_auroc", "p_aupr", "i_ap", "p_auroc")}, "legacy": True}

def metric_value(record, name):
    x = record.get(name) if isinstance(record, dict) else None
    if isinstance(x, dict): return x.get("value") if x.get("availability") else None
    return x
