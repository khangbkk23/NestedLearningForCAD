"""Run W3 correctness tests and save a source-hashed verification artifact.

Example: pixi run python scripts/hope_cad_v2/verify_visual.py
No external assets, dataset or pytest are required. CUDA tests run if available.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/hope_cad_v2/w3_verification.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests/hope_cad_v2'), pattern='test_visual_w3.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    paths = sorted((ROOT / 'models/hope_cad_v2').glob('*.py')) + sorted((ROOT / 'tests/hope_cad_v2').glob('*.py')) + [Path(__file__)]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    report = {
        'scope': 'W3 visual operator correctness on synthetic tensors; not CAD validation or T4 profiling',
        'source_commit': commit, 'source_sha256': hashes,
        'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'tests_run': result.testsRun, 'success': result.wasSuccessful(),
        'failures': [(str(test), error) for test, error in result.failures],
        'errors': [(str(test), error) for test, error in result.errors],
        'skipped': [(str(test), reason) for test, reason in result.skipped],
        'numerical_measurements': getattr(sys.modules.get('test_visual_w3'), 'MEASUREMENTS', {}),
        'elapsed_seconds': time.perf_counter() - started,
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'W3 verification artifact: {args.output}')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
