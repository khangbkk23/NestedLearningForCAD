"""Explicit source inventory for a clean checkout; never scans data or environments."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AREAS = ('src', 'scripts', 'tests', 'docs', 'refs')
ROOT_FILES = ('.gitattributes', '.gitignore', 'AGENTS.md', 'README.md', 'QUICK_UPDATE.md', 'LICENSE',
              'pyproject.toml', 'pixi.toml', 'pixi.lock')
SKIP = {'.git', '.pixi', '.venv', '__pycache__', '.pytest_cache', '.cache',
        'workspace', 'results', 'logs', 'runs', 'dist', 'build'}
SUFFIXES = {'.py', '.yaml', '.yml', '.json', '.md', '.toml', '.txt', '.sh', '.pdf'}


def paths():
    found = {n for n in ROOT_FILES if (ROOT / n).is_file()}
    for area in AREAS:
        for folder, dirs, files in os.walk(ROOT / area, followlinks=False):
            dirs[:] = [n for n in dirs if n not in SKIP and not n.endswith('.egg-info')
                       and not (Path(folder) / n).is_symlink()]
            for name in files:
                p = Path(folder) / name
                if p.suffix.lower() in SUFFIXES:
                    if p.is_symlink():
                        raise ValueError('Source symlink: ' + str(p))
                    found.add(p.relative_to(ROOT).as_posix())
    found.add('src/hope_cad/resources/source_inventory.json')
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    target = ROOT / 'src/hope_cad/resources/source_inventory.json'
    value = dict(schema='hope_cad_portable_source_v1', files=paths(),
                 scope='Complete functional package, configuration, docs, tests and scripts; no environments, data or artifacts')
    if args.write:
        target.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')
    elif json.loads(target.read_text(encoding='utf-8')) != value:
        raise ValueError('Review source changes, then regenerate the inventory')
    print(json.dumps(dict(files=len(value['files']), written=args.write)))


if __name__ == '__main__':
    main()
