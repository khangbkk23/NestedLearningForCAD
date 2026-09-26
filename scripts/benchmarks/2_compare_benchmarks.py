# scripts/benchmarks/2_compare_benchmarks.py
"""Read-only comparison table; never ranks or selects a run."""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-protocol-mismatch", action="store_true")
    a = p.parse_args()
    records = []
    protocols = set()
    for raw in a.runs:
        run = Path(raw)
        r = json.loads((run/"run.json").read_text())
        s = json.loads((run/"summary.json").read_text())
        protocols.add(r["protocol_id"])
        records.append((r, s))
    if len(protocols) > 1 and not a.allow_protocol_mismatch:
        raise SystemExit(
            "protocol mismatch; pass --allow-protocol-mismatch to compare with warning")
    headers = ["protocol_id", "method_id", "seed", "final_macro_i_auroc", "final_macro_p_aupr", "FM-I", "FM-P",
               "persistent_method_memory_MiB", "train_update_wall_time", "final_eval_wall_time", "inference_FPS", "reportable_smoke"]
    lines = []
    if len(protocols) > 1:
        lines.append("WARNING: protocol mismatch explicitly allowed.\n")
    lines += ["|"+"|".join(headers)+"|", "|" +
              "|".join(["---"]*len(headers))+"|"]
    for r, s in records:
        mem = s.get("memory", {}).get("persistent_bytes", 0)/1048576
        rt = s.get("runtime", {})
        vals = [r["protocol_id"], r["method_id"], r.get("seed", ""), s.get("macro_final_i_auroc"), s.get("macro_final_p_aupr"), s.get("fm_i"), s.get(
            "fm_p"), f"{mem:.6f}", rt.get("train_wall_seconds"), rt.get("final_eval_wall_seconds"), s.get("inference_fps"), f"{s.get('reportable', False)}/{s.get('smoke', False)}"]
        lines.append("|"+"|".join(map(str, vals))+"|")
    Path(a.output).write_text("\n".join(lines)+"\n")

if __name__ == "__main__":
    main()