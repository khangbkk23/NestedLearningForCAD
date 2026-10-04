# scripts/benchmarks/compare.py
"""Compare normalized CADIC and ReplayCAD artifacts without inventing values."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.benchmarks.common.result_schema import load_normalized, metric_value

def _metric(run, name, source="reproduced"):
    if source == "reported":
        ref=json.loads((Path(run)/"reported_reference.json").read_text()) if (Path(run)/"reported_reference.json").is_file() else {}
        key={"i_auroc":"image_auroc","p_aupr":"pixel_aupr"}.get(name)
        return f"{float(ref[key]):.6f}" if key in ref else "NA (reported value unavailable)"
    data=load_normalized(run); metrics=data.get("metrics",data); record=metrics.get(name) if isinstance(metrics,dict) else None
    value=metric_value(metrics,name)
    if value is None: return "NA ("+str(record.get("reason","unavailable"))+")" if isinstance(record,dict) else "NA (unavailable)"
    return f"{float(value):.6f}"

def _resource(run):
    p=Path(run); disk={}
    f=p/"resource"/"disk.json"
    if f.is_file(): disk=json.loads(f.read_text())
    return disk

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--run",dest="run",action="append"); p.add_argument("--runs",nargs="+"); p.add_argument("--source",choices=["reproduced","reported"],default="reproduced"); p.add_argument("--output",required=True); p.add_argument("--allow-protocol-mismatch",action="store_true"); a=p.parse_args(argv)
    runs=a.run or a.runs or []; records=[]; protocols=set()
    if not runs: raise SystemExit("at least one --run is required")
    for raw in runs:
        run=Path(raw); meta=json.loads((run/"run.json").read_text())
        if a.source=="reported" and meta.get("method_id")!="replaycad": raise SystemExit("reported references are only defined for ReplayCAD")
        protocols.add((meta.get("protocol_id"),tuple(meta.get("task_order",[])))); records.append((run,meta))
    if len(protocols)>1 and not a.allow_protocol_mismatch: raise SystemExit("protocol/task-order mismatch; pass --allow-protocol-mismatch")
    lines=["# Benchmark comparison", "", f"Source: **{a.source}**", "", "| Run | Method | Image AUROC | Pixel AUPR | Image FM | Pixel FM | Generated replay bytes | Checkpoint bytes |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for run,meta in records:
        fm=json.loads((run/"normalized_metrics"/"forgetting.json").read_text()) if (run/"normalized_metrics"/"forgetting.json").is_file() else {}
        if a.source=="reported":
            ref=json.loads((run/"reported_reference.json").read_text()) if (run/"reported_reference.json").is_file() else {}
            fm={"fm_i":ref.get("image_fm"),"fm_p":ref.get("pixel_fm")}
        disk=_resource(run); gen=disk.get("generated_replay",{}).get("bytes",0); ck=disk.get("checkpoints",{}).get("bytes",0)
        fmi=fm.get('fm_i'); fmp=fm.get('fm_p'); fmreason=fm.get('reason')
        fmi="NA ("+str(fmreason)+")" if fmi is None and fmreason else ("NA" if fmi is None else f"{float(fmi):.6f}")
        fmp="NA ("+str(fmreason)+")" if fmp is None and fmreason else ("NA" if fmp is None else f"{float(fmp):.6f}")
        lines.append(f"| `{run.name}` | {meta.get('method_id')} | {_metric(run,'i_auroc',a.source)} | {_metric(run,'p_aupr',a.source)} | {fmi} | {fmp} | {gen} | {ck} |")
    Path(a.output).write_text("\n".join(lines)+"\n")
if __name__=="__main__": main()
