"""Package the concrete W7 source, including untracked implementation files."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.hope_cad_v2.artifacts import provenance
from dataset.benchmark_protocol_v2 import file_hash

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'ultils/hope_cad_w7_source_bundle.zip')
    args=p.parse_args();paths=set()
    for folder in ('models','dataset','training','tests','scripts/hope_cad_v2','conf/benchmarks','docs/main_spec'):
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.yaml'))
    paths.update((ROOT/'notebooks').glob('hope_cad_w7_*.ipynb'))
    paths.update(ROOT/'notebooks'/name for name in ('hope_cad_v4_normal_kaggle.ipynb','hope_cad_v4_preflight_kaggle.ipynb'))
    paths.update(ROOT/p for p in ('pixi.lock','pixi.toml','requirements.txt','README.md'))
    paths=sorted(paths);origin=provenance()
    manifest=dict(schema='hope_w7_source_v1',origin={k:v for k,v in origin.items() if k!='source_sha256'},
                  files={p.relative_to(ROOT).as_posix():file_hash(p) for p in paths})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_name(args.output.name+'.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in paths:archive.write(path,path.relative_to(ROOT).as_posix())
        archive.writestr('bundle_manifest.json',json.dumps(manifest,sort_keys=True))
    with zipfile.ZipFile(temp) as archive:
        assert archive.testzip() is None
        for n,h in manifest['files'].items():
            if hashlib.sha256(archive.read(n)).hexdigest()!=h:raise RuntimeError('source bundle verification failed')
    temp.replace(args.output)
    print(json.dumps(dict(path=str(args.output),files=len(paths),sha256=file_hash(args.output)),indent=2))

if __name__=='__main__':main()
