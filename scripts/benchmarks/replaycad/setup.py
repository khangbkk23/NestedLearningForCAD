# scripts/benchmarks/replaycad/setup.py
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from scripts.benchmarks.common.config import load_protocol, load_method
from scripts.benchmarks.common.cli import safe_run_name, tasks_arg
from scripts.benchmarks.common.artifacts import ensure_run_dir, atomic_json, command_file
from scripts.benchmarks.common.environment import write_environment, git_info
from scripts.benchmarks.replaycad.pipeline import preflight, command_plan

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--protocol",required=True); p.add_argument("--method",required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--run-name"); p.add_argument("--output-root",default=str(ROOT/"results/benchmarks")); p.add_argument("--tasks"); p.add_argument("--smoke",action="store_true"); p.add_argument("--dry-run",action="store_true"); a=p.parse_args(argv)
    protocol=load_protocol(a.protocol); method=load_method(a.method); selected=tasks_arg(a.tasks); name=safe_run_name(a.run_name or f"setup_seed{a.seed}")
    run=Path(a.output_root)/protocol["id"]/"replaycad"/name; ensure_run_dir(run)
    reportable=not (a.smoke or protocol.get("id")=="mvtec_smoke_v1")
    atomic_json(run/"run.json", {"schema":"benchmark_run_v1","created_utc":__import__("datetime").datetime.utcnow().isoformat()+"Z","seed":a.seed,"reportable":reportable,"smoke":not reportable,"method_id":"replaycad","protocol_id":protocol["id"],"task_order":selected,"execution_mode":method.get("execution_mode","native")})
    (run/"protocol_resolved.yaml").write_text(__import__("yaml").safe_dump(protocol,sort_keys=False)); (run/"method_resolved.yaml").write_text(__import__("yaml").safe_dump(method,sort_keys=False))
    if method.get("reported_reference"): atomic_json(run/"reported_reference.json",method["reported_reference"])
    statuses, paths=preflight(method,protocol,project_root=ROOT,run_dir=run,selected=selected); commands=command_plan(method,selected,run,smoke=not reportable); command_file(run,commands); atomic_json(run/"preflight.json", {"statuses":statuses,"paths":paths,"dry_run":a.dry_run}); write_environment(run,ROOT,{"upstream":git_info(paths["root"])})
    print(json.dumps({"run_dir":str(run),"statuses":statuses,"commands":commands if a.dry_run else "saved to commands.json"},indent=2,default=str)); return 0 if not any(x.get("status")=="FAIL" for x in statuses) else 2
if __name__=="__main__": raise SystemExit(main())
