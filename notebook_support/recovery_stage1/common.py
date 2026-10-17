"""Frozen-v4 recovery experiment contracts and verified evidence transport."""
from pathlib import Path
import hashlib
import json
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'hope_recovery_stage1_evidence_v1'
SOURCE_SCHEMA = 'hope_recovery_stage1_source_v1'
W72_SHA = 'a4cba0f4c04c1c8e6677eb79e1f51b419efb8d88a02838fc91cd5f1086469563'
W72_MANIFEST_SHA = '4873b641b86fe665af6696a09ba79315677f6094bb90796132f35f08da911f3c'
SUITE_DIGEST = '17cd635b35e62484c5a000b4be4af503284e7d00da9aec79d64576c8d5055240'
M_RUN_DIGEST = '7bc6e394bd9849e45dc357559bcd2fb369f6a5535f4dddfca76a25fe3c312227'
JOBS = ('C1_seed0', 'A1_seed0', 'CADIC10000_seed0')
BACKBONE_SHA = '2c4839589adbbf701573dd155e1814f8adb145cd4d235bb86a93269369e111ec'
PROTOCOL = dict(schema='hope_recovery_stage1_protocol_v1',
    algorithm='unchanged frozen HOPE-CAD v4 and matched CADIC', jobs=list(JOBS),
    cadic_execution=dict(backend='cadic_exact_pair_tile_cache_v2',bank_budget=10000,
        reference_kernel_and_tile_shapes_unchanged=True,auxiliary_cache_disclosed=True,reference_parity_required=True),
    selection_or_refitting=False, official_evaluation='final row only; no full suite claim',
    diagnostic=dict(samples_per_task=8, salt='recovery-normal-v1-20261016',
        histories=['diagonal','final'], data='manifest.online train normals only; reused/in-sample',
        gradients='discarded single-event next-view proposal; never applied',
        selection_feedback=False, future_category_model_reads=False),
    claim='development diagnosis only; no superiority, no VisA access')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def canon(value):
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def digest(value):
    from dataset.benchmark_manifest_v1 import canonical_digest
    return canonical_digest(value)


def atomic_json(value,path):
    from training.hope_runtime_v4 import atomic_json as write
    write(value,path)


def source_identity():
    from notebook_support.w7_resume_fix.launcher import verify_frozen_source
    frozen=verify_frozen_source()
    paths=list((ROOT/'notebook_support/recovery_stage1').glob('*.py'))
    paths+=list((ROOT/'notebook_support/w7_resume_fix').glob('*.py'))
    paths+=list((ROOT/'notebook_support/w7_resume_fix').glob('*.json'))
    paths+=list((ROOT/'conf/benchmarks').rglob('*.yaml'))
    paths+=[ROOT/'models/benchmark_adapter_base_v1.py']
    paths+=list((ROOT/'tests/recovery_stage1').glob('*.py'))
    paths+=list((ROOT/'docs/main_spec').glob('recovery_stage1*'))
    paths+=list((ROOT/'notebooks').glob('hope_cad_recovery_stage1*.ipynb'))
    extras={p.relative_to(ROOT).as_posix():sha(p) for p in paths if p.is_file()}
    return dict(frozen=frozen,extras=dict(sorted(extras.items())),protocol=PROTOCOL)


def read_manifest(source):
    source=Path(source)
    if source.is_dir():
        raw=(source/'artifact_manifest.json').read_bytes()
    else:
        with zipfile.ZipFile(source) as z:
            names=z.namelist()
            if len(names)!=len(set(names)):raise ValueError('duplicate archive paths')
            raw=z.read('artifact_manifest.json')
            if set(names)!=set(json.loads(raw)['files'])|{'artifact_manifest.json'}:
                raise ValueError('evidence inventory differs from manifest')
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def find_reviewed(inputs,manifest_sha,override=None):
    """Allow duplicate extracted copies of precisely the SAME pinned parent."""
    if override:
        p=Path(override)
        if read_manifest(p)[1]!=manifest_sha:raise ValueError('override is not reviewed evidence')
        return p
    choices=[]
    for p in Path(inputs).rglob('artifact_manifest.json'):
        if sha(p)==manifest_sha:choices.append(p.parent)
    if not choices:
        for p in Path(inputs).rglob('*.zip'):
            try:
                if read_manifest(p)[1]==manifest_sha:choices.append(p)
            except (KeyError,ValueError,zipfile.BadZipFile):pass
    if not choices:raise ValueError('reviewed evidence not found; attach the completed parent or set override')
    return sorted(set(choices))[0]


def import_reference(source,root):
    from training.hope_w7_io import restore_archive
    manifest,manifest_sha=read_manifest(source)
    if manifest_sha!=W72_MANIFEST_SHA:raise ValueError('requires reviewed completed W7.2, not preflight/7.1')
    keep=lambda n:(n in ('suite_lock.json','preflight.json','verification_w7.json','handoff.json') or
        (n.startswith('runs/M_seed0/') and any('/'+part+'/' in n for part in ('states','metrics','test_manifests','transport_fixes'))) or
        n in ('runs/M_seed0/run_lock.json','runs/M_seed0/job_lock.json','runs/M_seed0/completion.json',
              'runs/M_seed0/manifest_train.json','runs/M_seed0/train_summary.json'))
    restore_archive(source,root,schema='hope_w7_evidence_v1',select=keep)
    h=json.loads((Path(root)/'handoff.json').read_text())
    c=json.loads((Path(root)/'runs/M_seed0/completion.json').read_text())
    if h['status']!='7.2_complete_pending_review' or not c['matrix_complete'] or c['run_digest']!=M_RUN_DIGEST:
        raise ValueError('parent is not completed reviewed M seed0 matrix')
    atomic_json(dict(schema='recovery_reference_import_v1',manifest_sha256=manifest_sha,
        reviewed_archive_sha256=W72_SHA,verified_subset={n:v for n,v in manifest['files'].items() if keep(n)},
        full_archive_not_copied=True),Path(root)/'reference_import.json')
    return manifest


def verify_reference(root):
    root=Path(root)
    receipt=json.loads((root/'reference_import.json').read_text())
    if receipt['manifest_sha256']!=W72_MANIFEST_SHA or receipt['reviewed_archive_sha256']!=W72_SHA:
        raise ValueError('unreviewed reference identity')
    if any(sha(root/n)!=h for n,h in receipt['verified_subset'].items()):raise ValueError('reference bytes changed')
    return receipt


def restore_recovery(source,root):
    from training.hope_w7_io import restore_archive
    return restore_archive(source,root,schema=SCHEMA)


def locked_json(path,identity):
    from notebook_support.w7_resume_fix.launcher import normalize_json_identity
    from training.hope_w7_contract import lock_json
    return lock_json(path,normalize_json_identity(identity))


def check_deadline(deadline):
    from training.hope_w7_engine import SessionLimit
    if deadline is not None and time.monotonic()>=deadline:raise SessionLimit('saved boundary before next operation')


def export(root,output):
    from training.hope_w7_io import export_archive
    if (Path(root)/'artifact_manifest.json').exists():raise ValueError('do not nest an old root manifest during export')
    return export_archive(root,output,schema=SCHEMA)


def sampled_rows(dataset,task,count=8):
    rows=dataset.training_paths(task)
    if len(rows)<count:raise ValueError('not enough legal normal images for fixed diagnostic sample')
    salt=PROTOCOL['diagnostic']['salt']
    key=lambda r:hashlib.sha256((salt+'|'+r['relative_path']).encode()).hexdigest()
    return sorted(rows,key=key)[:count]


def assert_legal_normal(row,allowed,checkpoint):
    if row.get('split')!='train' or row.get('task_id',checkpoint+1)>checkpoint or allowed.get(row['relative_path'])!=row:
        raise ValueError('diagnostic rejects future/test/non-online normal identity')


class ObservedAdapter:
    """Capture existing fit_task return values; adapter state/update/scoring stay exact."""
    def __init__(self,adapter,folder):
        self.adapter,self.folder=adapter,Path(folder)
    def __getattr__(self,name):return getattr(self.adapter,name)
    def fit_task(self,task,name,batches):
        result=None
        for batch in batches:
            paths=list(batch['relative_path'])
            if any('/train/good/' not in p for p in paths) or bool((batch['labels']!=0).any()):
                raise ValueError('training observer accepts normal train images only')
            result=self.adapter.fit_task(task,name,[batch])
            payload=dict(schema='recovery_train_observation_v1',task_id=task,task_name=name,
                identities=paths,samples=len(paths),observed_fit_return={k:v for k,v in (result or {}).items()
                    if k in ('samples','cms_clock','event_cursor','cms_commits','cms_pending','mean_level_losses','coreset','update_unit')})
            target=self.folder/(hashlib.sha256('|'.join(paths).encode()).hexdigest()+'.json')
            if target.exists():
                if json.loads(target.read_text())!=canon(payload):raise ValueError('replayed training observation differs')
            else:atomic_json(payload,target)
        return result