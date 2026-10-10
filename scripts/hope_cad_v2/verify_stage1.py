"""Source-hashed W1/W3/W4 verification; no dataset/checkpoint downloads."""
import argparse
from pathlib import Path
import sys
import time
import unittest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from scripts.hope_cad_v2.artifacts import provenance, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests/hope_cad_v2'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(args.output, dict(**provenance(), scope='W1/W3/W4 synthetic correctness, not real CAD validation',
               tests_run=result.testsRun, success=result.wasSuccessful(), reportable=False,
               failures=[(str(test), error) for test,error in result.failures],
               errors=[(str(test), error) for test,error in result.errors],
               skipped=[(str(test), reason) for test,reason in result.skipped],
               elapsed_seconds=time.perf_counter()-started))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__': raise SystemExit(main())
