"""Source-only W1-W6 handoff, including uncommitted files and verified hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hope_cad_v2.artifacts import provenance
from dataset.benchmark_protocol_v2 import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = set()
    for folder in ('models', 'dataset', 'training', 'tests', 'scripts/hope_cad_v2',
                   'conf/benchmarks', 'docs/main_spec'):
        paths.update(p for p in (ROOT/folder).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.json', '.yaml', '.md'))
    paths.update(ROOT/p for p in ('pixi.toml', 'pixi.lock', 'README.md', 'requirements.txt',
                                  'notebooks/hope_cad_v2_2_kaggle.ipynb',
                                  'notebooks/hope_cad_v2_2_preflight_kaggle.ipynb'))
    paths = sorted(p for p in paths if p.exists())
    origin = provenance()
    manifest = dict(schema='hope_w1_w6_source_bundle_v3',
                    origin={k: v for k, v in origin.items() if k != 'source_sha256'},
                    files={p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix+'.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr('bundle_manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('source archive CRC failed')
        for name, expected in manifest['files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise RuntimeError(f'source archive digest failed: {name}')
    temporary.replace(args.output)
    print(args.output)
    print('Verified files:', len(paths))
    print('SHA256:', file_hash(args.output))


if __name__ == '__main__':
    main()
