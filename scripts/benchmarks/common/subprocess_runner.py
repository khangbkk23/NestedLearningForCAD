# scripts/benchmarks/common/subprocess_runner.py
import os, shlex, subprocess, time
from pathlib import Path

class CommandError(RuntimeError): pass

def run_command(argv, *, cwd=None, env=None, log_path=None, dry_run=False, timeout=None):
    command = [str(x) for x in argv]
    record = {"argv": command, "command": shlex.join(command), "cwd": str(cwd) if cwd else None}
    if dry_run: record.update(status="DRY_RUN", wall_seconds=0.0); return record
    start = time.monotonic(); log = None
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True); log = open(log_path, "w")
    try:
        p = subprocess.run(command, cwd=cwd, env={**os.environ, **(env or {}), "PYTHONDONTWRITEBYTECODE": "1"}, stdout=log or subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        record.update(status="PASS" if p.returncode == 0 else "FAIL", returncode=p.returncode, wall_seconds=time.monotonic()-start)
        if log is None: record["output"] = p.stdout[-4000:]
        if p.returncode: raise CommandError(f"command failed ({p.returncode}): {shlex.join(command)}")
        return record
    finally:
        if log: log.close()
