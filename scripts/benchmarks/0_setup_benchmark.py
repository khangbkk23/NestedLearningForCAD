# scripts/benchmarks/0_setup_benchmark.py
"""Resolve configs and create an immutable, train-only manifest."""
from training.benchmark_artifacts_v1 import BenchmarkArtifacts, git_metadata
from dataset.benchmark_manifest_v1 import build_training_manifest
from training.benchmark_config_v1 import load_configs, validate_configs
import argparse
import json
import os
import subprocess
import sys
import time
import hashlib
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--output-root", default=str(ROOT/"results/benchmarks"))
    p.add_argument("--run-name")
    p.add_argument("--require-clean-git", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()

def main():
    a = args()
    protocol, method = load_configs(a.protocol, a.method, a.overrides)
    validate_configs(protocol, method, a.smoke)
    git = git_metadata(ROOT)
    if a.require_clean_git and git["dirty"]:
        raise SystemExit("working tree is dirty")
    manifest = build_training_manifest(protocol, a.seed, git["commit"])
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = a.run_name or f"{timestamp}_seed{a.seed}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
        raise SystemExit("unsafe run name")
    run = Path(a.output_root)/protocol["id"]/method["id"]/name
    if run.exists() and any(run.iterdir()):
        raise SystemExit(f"run directory already exists: {run}")
    art = BenchmarkArtifacts(run)
    art.initialize(protocol, method, manifest, git, a.smoke, not a.smoke)
    protocol_hash = hashlib.sha256(json.dumps(
        protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    method_hash = hashlib.sha256(json.dumps(
        method, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    art.write_json("run.json", {"created_utc": timestamp, "seed": a.seed, "smoke": a.smoke, "reportable": not a.smoke,
                   "protocol_id": protocol["id"], "method_id": method["id"], "protocol_sha256": protocol_hash, "method_sha256": method_hash, "manifest_sha256": manifest["digest"], "overrides": a.overrides})
    print(run)

if __name__ == "__main__":
    main()