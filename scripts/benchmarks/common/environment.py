# scripts/benchmarks/common/environment.py
import json, os, platform, subprocess, sys
from pathlib import Path
from .artifacts import atomic_json

def git_info(root):
    out = {"root": str(root), "commit": None, "dirty": None}
    try:
        out["commit"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        out["dirty"] = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except Exception: pass
    return out

def environment_record(project_root=None, extra=None):
    result = {"python": sys.executable, "python_version": platform.python_version(), "platform": platform.platform(), "env": {k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "MVTEC_ROOT", "REPLAYCAD_ROOT", "REPLAYCAD_CACHE_ROOT") if k in os.environ}}
    try:
        import torch
        result.update({"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available()), "gpu_count": torch.cuda.device_count(), "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    except Exception as exc: result["torch_error"] = str(exc)
    if project_root: result["git"] = git_info(project_root)
    if extra: result.update(extra)
    return result

def write_environment(run_dir, project_root=None, extra=None): return atomic_json(Path(run_dir) / "environment.json", environment_record(project_root, extra))

def probe_python(executable, imports=(), cwd=None, env=None):
    if not executable or not Path(executable).is_file(): return {"status": "MISSING", "reason": "python_executable_missing", "executable": str(executable)}
    script = "import importlib.util, json; names=%r; print(json.dumps({n: bool(importlib.util.find_spec(n)) for n in names}))" % list(imports)
    try:
        p = subprocess.run([str(executable), "-c", script], cwd=cwd, env={**os.environ, **(env or {}), "PYTHONDONTWRITEBYTECODE": "1"}, text=True, capture_output=True, timeout=30)
        return {"status": "PASS" if p.returncode == 0 else "FAIL", "imports": json.loads(p.stdout.strip() or "{}"), "stderr": p.stderr[-4000:]}
    except Exception as exc: return {"status": "FAIL", "reason": str(exc)}
