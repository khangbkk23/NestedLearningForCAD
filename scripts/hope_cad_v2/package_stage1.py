"""Build a source-only Kaggle handoff: no data, checkpoint, secrets, legacy outputs or .git."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.hope_cad_v2.artifacts import provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    paths = set()
    # Explicit source roots and extensions; never include arbitrary user files.
    for root in ('models','dataset','training','tests','scripts/hope_cad_v2','conf/benchmarks','docs/main_spec'):
        paths.update(f for f in (ROOT/root).rglob('*') if f.is_file() and f.suffix in ('.py','.json','.yaml','.md'))
    paths.update(ROOT/p for p in ('pixi.toml','pixi.lock','README.md','requirements.txt'))
    paths = sorted(p for p in paths if p.exists())
    origin = provenance()
    manifest = dict(schema='stage1_source_bundle_v1',origin={k:v for k,v in origin.items() if k != 'source_sha256'},
                    files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in paths: archive.write(path,path.relative_to(ROOT).as_posix())
        archive.writestr('bundle_manifest.json',json.dumps(manifest,indent=2))
    print(args.output)
    print('SHA256',hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == '__main__': main()
