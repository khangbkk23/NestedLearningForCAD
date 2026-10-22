"""Benchmark-only identities; never modifies the accepted learner or pilot source."""
import json,re
from pathlib import Path
from models.hope_cad_v5_1.contract import ROOT,CONTRACT,design_digest,digest,file_sha,canonical
from models.hope_cad_v5_1.session import SupportSession
from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetConfig
from training.runtime import lock_json

STUDY_PATH=ROOT/'conf/benchmarks/support_v5_1_study.json'
METHODS=('NESTED','CADIC','FIRST_ORDER','GEOMETRIC','FAST_ONLY','NO_SELF_MODIFICATION','NO_ANCHOR_OBJECTIVE','NO_FAST_ADAPTATION','CADIC_TOTAL_BYTES')
PHASES=('preflight','7.1','7.2','7.3')
SOURCE_SCHEMA='hope_support_v5_1_benchmark_source_v1'
EVIDENCE_SCHEMA='hope_support_v5_1_benchmark_evidence_v1'

def study():
    value=json.loads(STUDY_PATH.read_text(encoding='utf-8'))
    if value['method_design_digest']!=design_digest() or value['methods']!=list(METHODS) or value['seeds']!=[0,1,2]:raise ValueError('Frozen study/method drift')
    return value

def study_digest():
    return digest(dict(study=study(),spec_sha256=file_sha(ROOT/'docs/main_spec/benchmark_v5_1.md')))

def job(value):
    m=re.fullmatch(r'([A-Z_]+)_seed([012])',value)
    if not m or m[1] not in METHODS:raise ValueError('Unregistered job: '+str(value))
    return m[1],int(m[2])

def session_for(name,device='cpu'):
    mode,seed=job(name)
    if mode=='CADIC_TOTAL_BYTES':return SupportSession('CADIC',device=device,seed=seed,bank_config=CADICPatchCoresetConfig(budget=24576,chunk_size=256,query_chunk_size=256,pair_chunk_size=256))
    return SupportSession(mode,device=device,seed=seed)

def lock(path,value):return lock_json(path,canonical(value))

def source_files():
    from hope_cad.runtime import source_files as portable_files
    return portable_files()

def source_digest():
    return digest(source_files())
