"""Immutable benchmark artifacts; tensors remain torch files."""
from __future__ import annotations
import json, os, platform, subprocess, sys, time
from pathlib import Path
import yaml

def jsonable(value):
    if hasattr(value, "item"): return value.item()
    if isinstance(value, float) and (value != value or abs(value) == float("inf")): return None
    if isinstance(value, dict): return {str(k): jsonable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    return value

class BenchmarkArtifacts:
    def __init__(self, run_dir):
        self.root = Path(run_dir); self.metrics = self.root / "metrics"; self.profile = self.root / "profile"
        self.states = self.root / "states"; self.logs = self.root / "logs"
        for p in (self.metrics, self.profile, self.states, self.logs): p.mkdir(parents=True, exist_ok=True)
    def write_json(self, relative, value):
        path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    def write_configs(self, protocol, method):
        (self.root / "protocol_resolved.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
        (self.root / "method_resolved.yaml").write_text(yaml.safe_dump(method, sort_keys=False))
    def initialize(self, protocol, method, manifest, git, smoke, reportable):
        self.write_configs(protocol, method); self.write_json("manifest_train.json", manifest)
        self.write_json("git.json", git); self.write_json("environment.json", {"python": sys.version, "platform": platform.platform()})
        self.write_json("run.json", {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "smoke": smoke, "reportable": reportable})
    def save_state(self, name, state):
        import torch
        torch.save(state, self.states / name)
    def summarize(self, summary):
        self.write_json("summary.json", summary)
        lines = ["# Benchmark summary", "", "```json", json.dumps(jsonable(summary), indent=2, allow_nan=False), "```", ""]
        (self.root / "summary.md").write_text("\n".join(lines))

def git_metadata(root):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except Exception: commit, dirty = "unavailable", None
    return {"commit": commit, "dirty": dirty}
