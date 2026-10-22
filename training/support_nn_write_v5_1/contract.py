"""Additive identity on the immutable183 profiler source map."""
import json
from pathlib import Path
from training.support_throughput_v5_1.contract import ROOT,sha,digest,safe,verify_source as verify_base

SCHEMA='hope_support_v5_1_nn_write_evidence_v1'
SOURCE_SCHEMA='hope_support_v5_1_nn_write_source_v1'
PASSES=('R1','O1','O2','R2')
CONTRACT_PATH=ROOT/'docs/main_spec/throughput_nn_write_v5_1_contract.json'

def contract():
    c=json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))
    if c['candidate_id']!='nearest_fixed_shape_writes_v1' or c['passes']!=list(PASSES) or c['max_attempted_transitions']!=64 or c['capture_events']!=[181,189]:
        raise ValueError('Unknown NN candidate contract')
    return c

def verify_source():
    base=verify_base();c=contract()
    if base['source_digest']!=c['base_profile_source_digest']:raise ValueError('Wrong profiler reference source')
    manifest=json.loads((ROOT/'nn_write_manifest.json').read_text(encoding='utf-8'))
    if manifest['schema']!=SOURCE_SCHEMA or manifest['contract_sha256']!=sha(CONTRACT_PATH):raise ValueError('NN source/contract identity changed')
    for n,h in manifest['files'].items():
        if sha(safe(ROOT,n))!=h:raise ValueError('NN source drift: '+n)
    original=json.loads((ROOT/'throughput_manifest.json').read_text(encoding='utf-8'))
    if any(manifest['files'].get(n)!=h for n,h in original['files'].items()):raise ValueError('NN source overwrites original profiler/base')
    return dict(source_digest=digest(manifest['files']),files=len(manifest['files']),base_profile_source_digest=base['source_digest'],contract_sha256=sha(CONTRACT_PATH))
