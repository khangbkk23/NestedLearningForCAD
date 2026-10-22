# src/hope_cad/integrations/cadic/run.py
"""CADIC's method-specific entry point; numerical engine remains v1."""
from __future__ import annotations
import sys
from hope_cad.benchmark.run import main as execute
def main():
    argv=list(sys.argv[1:])
    if "--stage" in argv:
        i=argv.index("--stage"); argv[i]="--phase"
    old=sys.argv; sys.argv=[old[0]]+argv
    try: return execute()
    finally: sys.argv=old
if __name__=="__main__": main()
