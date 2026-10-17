"""Machine-readable local/Kaggle engineering acceptance for recovery-only contracts."""
from pathlib import Path
import argparse
import io
import json
import platform
import sys
import time
import unittest
from .common import ROOT,atomic_json,source_identity,digest
from training.hope_runtime_v4 import configure_runtime


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args(argv)
    configure_runtime();start=time.perf_counter();source=source_identity()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/recovery_stage1'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(schema='recovery_stage1_test_verification_v1',success=result.wasSuccessful(),tests_run=result.testsRun,
        failures=[dict(test=str(t),traceback=trace) for t,trace in result.failures],
        errors=[dict(test=str(t),traceback=trace) for t,trace in result.errors],
        skipped=[dict(test=str(t),reason=why) for t,why in result.skipped],
        elapsed_seconds=time.perf_counter()-start,python=platform.python_version(),
        frozen_dependencies_verified=len(source['frozen']),operations_source_digest=digest(source),
        scope='engineering fixtures only; no AD benchmark, canonical ViT/T4 or VisA validation')
    atomic_json(report,args.output);print(json.dumps({k:v for k,v in report.items() if k not in ('failures','errors','skipped')},indent=2))
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())