"""W7 + shared/v4 regression suite; historical v3 notebook UI is outside delivery."""
import argparse
from pathlib import Path
import sys
import time
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from training.hope_runtime_v4 import configure_runtime,runtime_identity,atomic_json
from training.hope_w7_contract import source_hashes

EXCLUDED={
 'test_notebook_handoff_v3.NotebookHandoffContracts.test_bootstrap_accepts_zip_and_extracted_source_and_rejects_hash_drift',
 'test_notebook_handoff_v3.NotebookHandoffContracts.test_every_delivered_code_cell_compiles_and_has_a_unique_id',
}

def cases(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):yield from cases(item)
        else:yield item

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();configure_runtime();started=time.perf_counter()
    all_cases=list(cases(unittest.defaultTestLoader.discover(str(ROOT/'tests/hope_cad_v2'),pattern='test_*.py')))
    excluded=[t.id() for t in all_cases if t.id() in EXCLUDED]
    if set(excluded)!=EXCLUDED:raise RuntimeError('unexpected historical-test inventory')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(t for t in all_cases if t.id() not in EXCLUDED))
    atomic_json(dict(schema='hope_w7_local_verification_v1',success=result.wasSuccessful(),
        tests_run=result.testsRun,complete_w7_and_shared_v4_scope=True,reportable=False,
        scope='W7 fixtures and shared/v4 algorithm/data/identity/gradient/resume regression; no AD benchmark',
        historical_v3_notebook_tests_excluded=excluded,
        exclusion_reason='historical v3 notebook UI is not a W7 input/deliverable; user-modified local file preserved',
        failures=[(str(t),e) for t,e in result.failures],errors=[(str(t),e) for t,e in result.errors],
        skipped=[(str(t),e) for t,e in result.skipped],source_sha256=source_hashes(),runtime=runtime_identity(),
        elapsed_seconds=time.perf_counter()-started),args.output)
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())
