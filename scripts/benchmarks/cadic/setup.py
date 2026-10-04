# scripts/benchmarks/cadic/setup.py
"""CADIC's method-specific entry point; delegates to the established v1 setup."""
from __future__ import annotations
import runpy, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def main():
    return runpy.run_path(str(ROOT/"scripts/benchmarks/0_setup_benchmark.py"), run_name="__main__")
if __name__=="__main__": main()
