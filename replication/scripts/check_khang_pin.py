"""Check replication/khang_source.json against the current checkout (stdlib only, no torch, no data).

Recomputes the static import closure of the Khang modules that the B325/B326 drivers import (plus Khang's four gate
test files) and compares it with the pin, file by file:

    python -I -B replication/scripts/check_khang_pin.py

Exit code 0 means every pinned file is unchanged and the closure has not grown, i.e. the checkout still runs the
exact code of the published results (commit 5217cb1). Anything else is reported and gives exit code 1; the pin is
never rewritten by this script.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

REPLICATION = Path(__file__).resolve().parents[1]
ROOT = REPLICATION.parent
PIN = REPLICATION / 'khang_source.json'
SEEDS = ('exps.dinomaly.encoder', 'exps.hope_s_01.run', 'exps.nar_b14_01.run', 'exps.nar_lite_01.run',
         'exps.nar_patch_01.run', 'exps.nar_pilot_00.core', 'exps.nar_pilot_00.full', 'exps.nar_pilot_00.detector',
         'exps.nar_warm_01.run', 'exps.vitb8_q2.common')
GATE_TESTS = ('exps/nar_pilot_00/test_core.py', 'exps/nar_pilot_00/test_detector.py', 'exps/hope_s_01/test_run.py',
              'exps/nar_lite_01/test_run.py')
LOCAL_PACKAGES = ('exps', 'models', 'dataset', 'training', 'utils', 'conf')


def text_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def module_file(name):
    p = ROOT.joinpath(*name.split('.'))
    if p.with_suffix('.py').is_file():
        return p.with_suffix('.py')
    if (p / '__init__.py').is_file():
        return p / '__init__.py'
    return None


def closure():
    seen, stack = set(), [module_file(m) for m in SEEDS] + [ROOT / t for t in GATE_TESTS]
    while stack:
        f = stack.pop()
        if f is None or f in seen or not f.is_file():
            continue
        seen.add(f)
        parts = f.relative_to(ROOT).parts
        for i in range(1, len(parts)):                       # package __init__ files on the way
            stack.append(ROOT.joinpath(*parts[:i], '__init__.py'))
        package = list(parts[:-1])
        for node in ast.walk(ast.parse(f.read_text(encoding='utf8'))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[:len(package) - (node.level - 1)]
                    mod = '.'.join(base + ([node.module] if node.module else []))
                else:
                    mod = node.module or ''
                names = [mod] + [f'{mod}.{a.name}' for a in node.names]
            for n in names:
                if n.split('.')[0] in LOCAL_PACKAGES:
                    stack.append(module_file(n))
    return {f.relative_to(ROOT).as_posix(): text_sha(f) for f in seen}


def main():
    pin = json.loads(PIN.read_text(encoding='utf8'))
    now = closure()
    changed = sorted(p for p, h in pin['files'].items() if not (ROOT / p).is_file() or text_sha(ROOT / p) != h)
    added = sorted(set(now) - set(pin['files']))
    for p in changed:
        print('CHANGED_OR_MISSING', p)
    for p in added:
        print('NOT_PINNED', p)
    ok = not changed and not added
    print(f"{'OK' if ok else 'DIFFERS'}: {len(pin['files'])} pinned files, commit {pin['commit'][:7]}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
