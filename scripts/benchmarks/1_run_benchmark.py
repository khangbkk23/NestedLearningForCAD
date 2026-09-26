# scripts/benchmarks/1_run_benchmark.py
"""Run prepared benchmark phases with a hard train/evaluation barrier."""
from models.fake_benchmark_adapter_v1 import FakeBenchmarkAdapter
from training.benchmark_config_v1 import validate_configs
from training.benchmark_engine_v1 import BenchmarkEngineV1
from training.benchmark_artifacts_v1 import BenchmarkArtifacts
from dataset.benchmark_protocol_v1 import MVTecContinualProtocol
from dataset.benchmark_manifest_v1 import validate_manifest
import argparse
import json
import sys
import time
from pathlib import Path
import yaml
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument(
        "--phase", choices=["all", "train", "eval", "fm", "summarize"], default="all")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-tasks", type=int)
    p.add_argument("--max-train-images", type=int)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--fail-on-state-mutation", action="store_true")
    return p.parse_args()

def main():
    a = args()
    run = Path(a.run_dir)
    protocol = yaml.safe_load((run/"protocol_resolved.yaml").read_text())
    method = yaml.safe_load((run/"method_resolved.yaml").read_text())
    manifest = json.loads((run/"manifest_train.json").read_text())
    runmeta = json.loads((run/"run.json").read_text())
    if (a.max_tasks or a.max_train_images) and runmeta.get("reportable", False):
        raise SystemExit("smoke caps are forbidden for reportable runs")
    validate_manifest(manifest, protocol, int(runmeta["seed"]))
    validate_configs(protocol, method, smoke=bool(runmeta.get("smoke")))
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = "cuda" if a.device == "cuda" or (
        a.device == "auto" and torch.cuda.is_available()) else "cpu"
    dataset = MVTecContinualProtocol(
        protocol, manifest, method, int(runmeta["seed"]), a.num_workers)
    if method["adapter"] == "fake":
        adapter = FakeBenchmarkAdapter(device)
    elif method["adapter"] == "cadic":
        from models.cadic_benchmark_adapter_v1 import CADICBenchmarkAdapterV1
        adapter = CADICBenchmarkAdapterV1(method, device)
    else:
        from models.metanath_benchmark_adapter_v1 import MetaNATHLegacyAdapterV1
        adapter = MetaNATHLegacyAdapterV1(method, device)
    art = BenchmarkArtifacts(run)
    engine = BenchmarkEngineV1(protocol, adapter, art,device, a.fail_on_state_mutation)
    if a.phase in ("all", "train"):
        engine.train(dataset, a.max_tasks, a.max_train_images, a.resume)
    if a.phase in ("all", "eval"):
        if not (art.states/"final.pt").is_file():
            raise SystemExit("final state missing; run train first")
        adapter.load_state_dict(torch.load(
            art.states/"final.pt", map_location=device, weights_only=False))
        engine.evaluate_final(dataset, a.max_tasks)
    if a.phase in ("all", "fm"):
        if not list(art.states.glob("task_*.pt")):
            raise SystemExit("task states missing; run train first")
        engine.evaluate_forgetting(dataset, a.max_tasks)
    if a.phase in ("all", "summarize"):
        macro = json.loads((art.metrics/"final_macro.json").read_text()) if (art.metrics/"final_macro.json").is_file() else {}
        fm = json.loads((art.metrics/"forgetting_summary.json").read_text()
                        ) if (art.metrics/"forgetting_summary.json").is_file() else {}
        summary = {**macro, "fm_i": fm.get("fm"), "fm_p": None, "method": method["id"], "protocol": protocol["id"], "memory": adapter.memory_stats(
        ), "runtime": engine.times, "reportable": runmeta.get("reportable", False), "smoke": runmeta.get("smoke", False)}
        art.write_json("profile/runtime.json", engine.times)
        art.write_json("profile/memory.json", adapter.memory_stats())
        art.summarize(summary)

if __name__ == "__main__":
    main()