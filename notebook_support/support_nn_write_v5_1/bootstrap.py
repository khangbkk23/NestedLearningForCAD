"""Reuse verified16-image assets, never reinterpret profiler output as candidate resume."""
import json
from pathlib import Path
from training.runtime import atomic_json
from training.support_throughput_v5_1.contract import select_evidence,Evidence,SCHEMA as PROFILE_SCHEMA,sha
from notebook_support.common.evidence import restore_archive
from notebook_support.support_throughput_v5_1.bootstrap import prepare as base_prepare
from training.support_nn_write_v5_1.contract import verify_source,contract,SCHEMA,CONTRACT_PATH

def prepare(c):
    source=verify_source();inputs=Path(c['inputs']);work=Path(c['work'])
    if select_evidence(inputs,schema=PROFILE_SCHEMA) is not None:
        raise ValueError('Remove completed profiler output; G1 is already reviewed/pinned. Attach original376parent plus newNNsource/assets only.')
    previous=select_evidence(inputs,schema=SCHEMA,override=c.get('RESUME_OVERRIDE'))
    if previous:
        with Evidence(previous) as e:
            e.verify_all();old=e.json('nn_write_lock.json')
            if old['source']!=source:raise ValueError('Validation resume source changed')
        restore_archive(previous,work,schema=SCHEMA)
    base_config=dict(c,RESUME_OVERRIDE=None)
    prepared=base_prepare(base_config)
    lock=dict(source=source,runtime=prepared['runtime'],original_parent_manifest=json.loads((work/'profile_lock.json').read_text())['parent_manifest_sha256'],contract_sha256=sha(CONTRACT_PATH))
    path=work/'nn_write_lock.json'
    if path.exists() and json.loads(path.read_text())!=lock:raise ValueError('Validation identity changed')
    atomic_json(lock,path)
    return dict(prepared,nn_write_source=source)
