# training/benchmark_core.py
import time
from pathlib import Path
from scripts.benchmarks.common.artifacts import atomic_json, stage_marker
from scripts.benchmarks.common.environment import write_environment
from scripts.benchmarks.common.resources import resource_snapshot

class BenchmarkRun:
    def __init__(self, run_dir, project_root=None):
        self.run_dir = Path(run_dir); self.project_root = project_root; self.times = {}
    def stage(self, name, fn, **details):
        start = time.monotonic()
        try:
            result = fn() or {}
            self.times[name] = {"wall_seconds": time.monotonic()-start, **result}
            stage_marker(self.run_dir, name, "PASS", **self.times[name])
            atomic_json(self.run_dir / "timing" / "stages.json", self.times)
            return result
        except Exception as exc:
            self.times[name] = {"wall_seconds": time.monotonic()-start, "error": str(exc)}
            stage_marker(self.run_dir, name, "FAIL", **self.times[name]); raise
    def finish_environment(self, extra=None): return write_environment(self.run_dir, self.project_root, extra)
    def resource_snapshot(self):
        result = resource_snapshot(self.run_dir); atomic_json(self.run_dir / "resource" / "disk.json", result); return result