"""Complete legacy regression suite plus independent v4 revision contracts."""
import argparse
from pathlib import Path
import sys
import time
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from training.hope_runtime_v4 import configure_runtime, source_identity, runtime_identity, atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pattern',default='test_*.py',help='focused debugging only; default runs the complete suite')
    args=p.parse_args(); configure_runtime(); started=time.perf_counter()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/hope_cad_v2'),pattern=args.pattern)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    atomic_json(dict(design_id='hope_cad_v4_warm_joint_cms_row_retention',
                    success=result.wasSuccessful(),tests_run=result.testsRun,reportable=False,
                    pattern=args.pattern,complete_suite=args.pattern=='test_*.py',
                    scope='numerical/gradient/data/identity/resume fixtures; not scientific W6 or CAD utility',
                    failures=[(str(t),e) for t,e in result.failures],errors=[(str(t),e) for t,e in result.errors],
                    skipped=[(str(t),e) for t,e in result.skipped],source_sha256=source_identity(),
                    runtime=runtime_identity(),elapsed_seconds=time.perf_counter()-started),args.output)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__': raise SystemExit(main())
