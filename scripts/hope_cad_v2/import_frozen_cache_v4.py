"""Copy ONLY frozen calibration features from a verified parent evidence bundle.

Never imports learned weights, selection, probe, checkpoint, source or runtime locks.
The destination runner still validates the tensor contract against current RGB/data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dataset.benchmark_protocol_v2 import file_hash
from training.hope_runtime_v4 import atomic_json


def import_cache(source, output, receipt):
    source,output=Path(source),Path(output)
    if output.exists():
        raise ValueError('refuse overwrite of existing frozen cache')
    name='prefix_features.pt'
    if source.is_dir():
        manifest=json.loads((source/'artifact_manifest.json').read_text())
        expected=manifest.get('files',{}).get(name)
        parent=source/name
        actual=file_hash(parent)
        parent_hash=file_hash(source/'artifact_manifest.json')
        stream=parent.open('rb')
        archive=None
    else:
        parent_hash=file_hash(source)
        archive=zipfile.ZipFile(source)
        names=archive.namelist()
        if len(names)!=len(set(names)):
            archive.close(); raise ValueError('duplicate parent archive entries')
        manifest=json.loads(archive.read('artifact_manifest.json'))
        expected=manifest.get('files',{}).get(name)
        stream=archive.open(name)
        actual=None
    try:
        if manifest.get('schema') not in ('hope_w1_w6_evidence_v3','hope_normal_evidence_v4') or not expected:
            raise ValueError('parent has no registered frozen calibration cache')
        output.parent.mkdir(parents=True,exist_ok=True)
        temporary=output.with_name(output.name+'.tmp')
        digest=hashlib.sha256()
        with stream,temporary.open('wb') as target:
            while block:=stream.read(8*1024**2):
                digest.update(block); target.write(block)
        if digest.hexdigest()!=expected or (actual is not None and actual!=expected):
            raise ValueError('parent frozen cache hash/CRC mismatch')
        temporary.replace(output)
        record=dict(parent=str(source),parent_sha256=parent_hash,parent_manifest_schema=manifest['schema'],
                    cache_entry=name,cache_sha256=expected,copied_entries=[name],
                    learned_weights_imported=False,probe_imported=False,
                    status='bytes_verified_context_and_RGB_parity_still_required')
        atomic_json(record,receipt)
        return record
    finally:
        if archive: archive.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(import_cache(args.parent,args.output,args.receipt),indent=2))


if __name__=='__main__': main()
