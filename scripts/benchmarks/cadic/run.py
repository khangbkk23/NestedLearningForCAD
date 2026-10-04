# scripts/benchmarks/cadic/run.py
"""CADIC's method-specific entry point; numerical engine remains v1."""
from __future__ import annotations
import runpy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def main():
    argv=list(sys.argv[1:])
    if "--stage" in argv:
        i=argv.index("--stage"); argv[i]="--phase"
    old=sys.argv; sys.argv=[old[0]]+argv
    try: return runpy.run_path(str(ROOT/"scripts/benchmarks/1_run_benchmark.py"), run_name="__main__")
    finally: sys.argv=old
if __name__=="__main__": main()
