# training/replaycad_normalizer.py
"""Parse native InvAD metric.txt and create a lossless normalized view."""
import json, math, re
from pathlib import Path
from scripts.benchmarks.common.artifacts import atomic_json
from scripts.benchmarks.common.result_schema import metric, unavailable

def _number(value):
    x = float(value)
    return x / 100.0 if abs(x) > 1.0 and abs(x) <= 100.0 else x

def parse_metric_lines(path, classes, final_epoch=None):
    """ReplayCAD writes rows as epoch followed by class metric columns and Avg columns.

    The parser accepts both a labelled CSV/TSV fixture and the author's whitespace
    rows. Class identity is taken from the supplied class list, never row position.
    """
    rows = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.lower().startswith(("epoch", "iter")): continue
        vals = re.split(r"[\s,]+", line)
        try: epoch = int(float(vals[0]))
        except (ValueError, IndexError): continue
        nums = []
        for v in vals[1:]:
            try: nums.append(_number(v))
            except ValueError: pass
        rows.append((epoch, nums, line))
    if not rows: return {"rows": [], "final_epoch": None, "classes": list(classes)}
    chosen = max(r[0] for r in rows) if final_epoch is None else final_epoch
    selected = next((r for r in rows if r[0] == chosen), None)
    if selected is None: raise ValueError(f"configured final epoch {chosen} missing from metric.txt")
    nums = selected[1]
    out = {c: {} for c in classes}
    # Native order is interleaved i_<class>, p_<class>, then Avg columns.
    for i, cls in enumerate(classes):
        if len(nums) >= 2*i+2:
            out[cls] = {"i_auroc": nums[2*i], "p_aupr": nums[2*i+1]}
    return {"rows": [{"epoch": e, "raw": line} for e, _, line in rows], "final_epoch": chosen, "classes": list(classes), "per_task": out, "raw_path": str(path)}

def normalize_native(native, *, source="reproduced", semantics_match=True):
    per = native.get("per_task", {})
    def avg(name):
        vals = [v.get(name) for v in per.values() if v.get(name) is not None]
        return sum(vals) / len(vals) if vals else None
    image, pixel = avg("i_auroc"), avg("p_aupr")
    metrics = {"i_auroc": metric("i_auroc", image, source=source) if semantics_match and image is not None else unavailable("i_auroc", "not_reported_by_author_output" if image is None else "evaluator_semantics_not_confirmed", source),
               "p_aupr": metric("p_aupr", pixel, source=source) if semantics_match and pixel is not None else unavailable("p_aupr", "not_reported_by_author_output" if pixel is None else "evaluator_semantics_not_confirmed", source),
               "i_ap": unavailable("i_ap", "not_reported_by_author_config", source), "p_auroc": unavailable("p_auroc", "not_reported_by_author_config", source)}
    return {"schema": "normalized_metrics_v1", "source": source, "final_epoch": native.get("final_epoch"), "metrics": metrics, "per_task": per}

def normalize_run(run_dir, metric_path=None, classes=None, final_epoch=None, semantics_match=True, source="reproduced"):
    run = Path(run_dir); classes = classes or json.loads((run / "run.json").read_text()).get("task_order", [])
    metric_path = Path(metric_path or run / "author_outputs" / "invad" / "metric.txt")
    native = parse_metric_lines(metric_path, classes, final_epoch) if metric_path.is_file() else {"rows": [], "per_task": {}, "final_epoch": None}
    atomic_json(run / "author_outputs" / "native_metrics.json", native)
    normalized = normalize_native(native, source=source, semantics_match=semantics_match)
    atomic_json(run / "normalized_metrics" / "final_macro.json", normalized)
    atomic_json(run / "normalized_metrics" / "final_per_task.json", {"source": source, "per_task": native.get("per_task", {})})
    atomic_json(run / "normalized_metrics" / "continual_matrix.json", {"matrix_available": False, "reason": "author_code_does_not_emit_task_boundary_states", "matrix": None})
    atomic_json(run / "normalized_metrics" / "forgetting.json", {"availability": False, "reason": "author_code_does_not_emit_task_boundary_states", "fm_i": None, "fm_p": None})
    return normalized
