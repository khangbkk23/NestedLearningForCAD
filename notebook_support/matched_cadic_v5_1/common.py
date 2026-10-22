import json,zipfile
from pathlib import Path
from training.support_throughput_v5_1.contract import ROOT,sha,digest,safe,Evidence
from continuation_runtime_v1.common import deadline,stage_tar
from notebook_support.common.evidence import restore_archive

SOURCE_SCHEMA='hope_matched_cadic_seed0_source_v1'
SCHEMA='hope_matched_cadic_seed0_evidence_v1'
BASE_DIGEST='827deb7ceb763b125494ebc2c4116fb4fce8d9965e05e30541b3215bc82e6022'
RECEIPTS=ROOT/'notebook_support/matched_cadic_v5_1/receipts'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def verify_source():
    from continuation_runtime_v1.common import verify_delivery
    from training.support_benchmark_v5_1.contract import source_digest,study_digest
    from models.hope_cad_v5_1.contract import design_digest
    core,base=verify_delivery();m=read(ROOT/'matched_cadic_manifest.json')
    if m.get('schema')!=SOURCE_SCHEMA or base['source_digest']!=BASE_DIGEST:raise ValueError('Matched CADIC base/source differs')
    for n,h in m['files'].items():
        if sha(safe(ROOT,n))!=h:raise ValueError('Matched CADIC source drift: '+n)
    old=read(ROOT/'continuation_manifest.json')
    if any(m['files'].get(n)!=h for n,h in old['files'].items()) or m['files'].get('continuation_manifest.json')!=sha(ROOT/'continuation_manifest.json'):raise ValueError('Matched delivery replaced base source')
    ref=read(RECEIPTS/'reference.json');manifest=read(RECEIPTS/'manifest_train.json')
    if sha(RECEIPTS/'manifest_train.json')!=ref['train_manifest_sha256'] or digest(manifest['online'])!=ref['normal_rows_digest']:raise ValueError('Reference train manifest changed')
    hashes=read(RECEIPTS/'feature_hashes.json')
    if sha(RECEIPTS/'feature_hashes.json')!=ref['feature_hashes_sha256'] or set(hashes)!={digest(r) for r in manifest['online']}:raise ValueError('Reference feature hash coverage changed')
    if source_digest()!=ref['original_scientific_source_digest'] or study_digest()!=ref['study_digest'] or design_digest()!=ref['design_digest']:raise ValueError('Scientific study/method changed')
    proof=read(RECEIPTS/'cadic_cost_review.json')
    if sha(RECEIPTS/'cadic_cost_review.json')!=ref['cadic_cost_review_sha256'] or proof['scientific_status']!='canonical_T4_scoped_engineering_pass_no_AD_claim' or proof['status']!='finalized':raise ValueError('Missing reviewed optimizedCADIC acceptance')
    op=dict(schema='matched_cadic_orchestration_v1',source_digest=digest(m['files']),core_r2_source_digest=core['source_digest'],phase='7.1',job='CADIC_seed0',backend='nearest_fixed_shape_writes_v1',reference_train_sha256=ref['train_manifest_sha256'],reference_test_sha256=ref['test_manifest_sha256'],study_digest=ref['study_digest'],design_digest=ref['design_digest'])
    return op,ref,manifest

def identity(op,ref,runtime):
    if runtime!=ref['runtime']:raise ValueError('Runtime differs from locked NESTED/T4')
    si=dict(schema='matched_cadic_suite_v1',orchestration=op,study_digest=ref['study_digest'],design_digest=ref['design_digest'],normal_rows_digest=ref['normal_rows_digest'],runtime=runtime,backbone_sha256=ref['backbone_sha256'])
    suite=dict(identity=si,digest=digest(si))
    ji=dict(job='CADIC_seed0',mode='CADIC',seed=0,suite_digest=suite['digest'],rows_digest=ref['normal_rows_digest'],fresh_initialization=True,learned_state_imported=False)
    return suite,ji

def check_manifest(actual,expected):
    if actual!=expected:raise ValueError('Matched prefix/bytes/roles differ from NESTED reference')
    if len(actual['online'])!=3599 or len(actual['entries'])!=3629 or len(actual['task_order'])!=15:raise ValueError('Wrong canonical train scope')
    excluded={r['relative_path'] for k in ('normal_validation','normal_probe') for r in actual['roles'][k]}
    if len(excluded)!=30 or excluded&{r['relative_path'] for r in actual['online']}:raise ValueError('Excludednormal permissions changed')

def select_resume(inputs,override=None):
    paths=[Path(override)] if override else [p.parent for p in Path(inputs).rglob('artifact_manifest.json')]+list(Path(inputs).rglob('*.zip'))
    choices=[]
    for p in paths:
        try:
            with Evidence(p) as e:
                if e.manifest.get('schema')!=SCHEMA:
                    if override:raise ValueError('Wrong resume schema/job; no NESTED/pilot/cost states')
                    continue
                if e.json('handoff.json')['status'] not in ('paused_at_atomic_boundary','7.1_complete_pending_review'):raise ValueError('Failed CADIC evidence needs review; no automatic retry')
                choices.append((e.manifest_sha256,p))
        except (OSError,KeyError,zipfile.BadZipFile) as exc:
            if override:raise ValueError('Invalid resume evidence override') from exc
    if len({h for h,p in choices})>1:raise ValueError('Attach latest CADIC output only; remove older versions or set RESUME_OVERRIDE')
    return sorted(choices,key=lambda x:str(x[1]))[0][1] if choices else None

def restore(parent,work,op):
    work=Path(work)
    if any(work.iterdir()):raise ValueError('Fresh work directory required')
    if parent is None:return 0
    with Evidence(parent) as e:
        count=e.verify_all()
        if e.manifest['schema']!=SCHEMA or e.json('operation_lock.json')!=op:raise ValueError('Resume operational source/job identity changed')
        # Only own run schema; all payloads stay hash-bound and no selective learned import.
        restore_archive(parent,work,schema=SCHEMA)
    return count
