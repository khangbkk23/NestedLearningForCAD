"""Pinned completed parents, byte-preserved transport and strict readonly owners."""
import hashlib,json,zipfile
from pathlib import Path
from training.support_throughput_v5_1.contract import ROOT,Evidence,sha,safe,digest
from notebook_support.common.evidence import restore_archive
from continuation_runtime_v1.common import deadline,stage_tar

SCHEMA='hope_support_v5_1_matrix_7_2_evidence_v1'
SOURCE_SCHEMA='hope_support_v5_1_matrix_7_2_source_v1'
JOBS=('NESTED_seed0','CADIC_seed0')
RECEIPTS=ROOT/'notebook_support/matrix_v5_1/receipts'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def job_name(job):
    if job not in JOBS:raise ValueError('Only NESTED_seed0/CADIC_seed0 readonly matrices')
    return job

def verify_source():
    from notebook_support.matched_cadic_v5_1.common import verify_source as base_verify
    from continuation_runtime_v1.common import verify_delivery
    cadic_op,ref,train=base_verify();core,nested_op=verify_delivery()
    m=read(ROOT/'matrix_manifest.json')
    if m.get('schema')!=SOURCE_SCHEMA:raise ValueError('Wrong matrix source schema')
    for n,h in m['files'].items():
        if sha(safe(ROOT,n))!=h:raise ValueError('Matrix source drift: '+n)
    base=read(ROOT/'matched_cadic_manifest.json')
    if any(m['files'].get(n)!=h for n,h in base['files'].items()) or m['files'].get('matched_cadic_manifest.json')!=sha(ROOT/'matched_cadic_manifest.json'):raise ValueError('Matrix source replaced frozen base')
    refs=read(RECEIPTS/'parents.json')
    if refs['runtime']!=ref['runtime'] or refs['train_manifest_sha256']!=ref['train_manifest_sha256'] or refs['test_manifest_sha256']!=ref['test_manifest_sha256']:raise ValueError('Parent study/runtime changed')
    for job in JOBS:
        raw=(RECEIPTS/(job+'_manifest.json')).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=refs['jobs'][job]['manifest_sha256']:raise ValueError('Reviewed parent manifest changed')
    op=dict(schema='matrix_7_2_orchestration_v1',source_digest=digest(m['files']),jobs=list(JOBS),phase='7.2',initial_parents={j:refs['jobs'][j]['manifest_sha256'] for j in JOBS},backend='nearest_fixed_shape_writes_v1',study_digest=ref['study_digest'],design_digest=ref['design_digest'])
    return op,refs,core

def operation(op,job):return dict(orchestration=op,job=job_name(job),phase='7.2')

def parent_manifest(job):return read(RECEIPTS/(job_name(job)+'_manifest.json'))

def select_parent(inputs,job,refs,override=None):
    job_name(job);paths=[Path(override)] if override else [p.parent for p in Path(inputs).rglob('artifact_manifest.json')]+list(Path(inputs).rglob('*.zip'))
    originals=[];resumes=[];pin=refs['jobs'][job]
    for p in paths:
        try:
            with Evidence(p) as e:
                schema=e.manifest.get('schema')
                if schema==SCHEMA:
                    h=e.json('handoff.json')
                    if h.get('job')!=job:
                        if override:raise ValueError('Wrong matrix job')
                        continue
                    if h.get('status') not in ('paused_at_atomic_boundary','7.2_complete_pending_review'):raise ValueError('Failed matrix evidence; review before retry')
                    resumes.append((e.manifest_sha256,p,SCHEMA))
                elif schema==pin['schema']:
                    if e.manifest_sha256!=pin['manifest_sha256']:raise ValueError('Wrong/unreviewed 7.1 parent version')
                    if not p.is_dir() and sha(p)!=pin['archive_sha256']:raise ValueError('Initial ZIP bytes differ from reviewed parent')
                    h=e.json('handoff.json')
                    if h.get('job')!=job or h.get('status')!='7.1_complete_pending_review':raise ValueError('Completed reviewed own7.1 required')
                    originals.append((e.manifest_sha256,p,schema))
                elif override:raise ValueError('Wrong parent job/schema; no cost/pilot/partial states')
        except (OSError,KeyError,zipfile.BadZipFile) as exc:
            if override:raise ValueError('Invalid explicit parent') from exc
    choices=resumes or originals
    if not choices:raise ValueError('Attach completed own7.1 or latest own7.2 output; set PARENT_OVERRIDE')
    if len({h for h,_,_ in choices})!=1:raise ValueError('Multiple matrix versions; keep latest only or set override')
    return sorted(choices,key=lambda x:str(x[1]))[0]

def verify_frozen(work,job,refs):
    """All original payloads except root handoff are immutable, including old logs."""
    work=Path(work);pin=refs['jobs'][job_name(job)];raw=(work/'matrix_origin/artifact_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pin['manifest_sha256'] or json.loads(raw)!=parent_manifest(job):raise ValueError('Initial matrix origin changed')
    m=json.loads(raw)
    if sha(work/'matrix_origin/handoff.json')!=m['files']['handoff.json']:raise ValueError('Initial handoff changed')
    for n,h in m['files'].items():
        if n=='handoff.json':continue
        if sha(safe(work,n))!=h:raise ValueError('Frozen 7.1 payload changed: '+n)
    return len(m['files'])-1

def restore(parent,kind,work,job,refs,op):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    if any(work.iterdir()):raise ValueError('Fresh kernel/clean work directory required')
    with Evidence(parent) as e:
        count=e.verify_all()
        if kind==SCHEMA:
            if e.json('matrix_lock.json')!=operation(op,job):raise ValueError('Matrix source/job identity changed')
        elif e.manifest_sha256!=refs['jobs'][job]['manifest_sha256']:raise ValueError('Unreviewed initial parent')
        restore_archive(parent,work,schema=kind)
        if kind!=SCHEMA:
            origin=work/'matrix_origin';origin.mkdir()
            raw=e.archive.read('artifact_manifest.json') if e.archive else (Path(parent)/'artifact_manifest.json').read_bytes()
            (origin/'artifact_manifest.json').write_bytes(raw);(origin/'handoff.json').write_bytes(e.read('handoff.json'))
    from training.support_benchmark_v5_1.engine import save_or_verify
    save_or_verify(work/'matrix_lock.json',operation(op,job));verify_frozen(work,job,refs)
    return count

class CADICOwner:
    def __init__(self,root,refs,base_op,runtime):
        from notebook_support.matched_cadic_v5_1.common import identity
        root=Path(root);suite,ji=identity(base_op,refs,runtime);self.folder=root/'runs/CADIC_seed0'
        train=read(root/'manifest_train.json');self.rows,self.tasks=train['online'],train['task_order']
        if read(root/'suite_lock.json')!=suite:raise ValueError('CADIC suite changed')
        self.identities=dict(job=read(self.folder/'job_lock.json'))
        jl=self.identities['job']
        if jl['digest']!=digest(jl['identity']) or jl['identity']!=dict(**ji,rows=self.rows,task_order=self.tasks):raise ValueError('CADIC job changed')
    def task_state(self,i,actor):
        from training.support_benchmark_v5_1.engine import task_state
        return task_state(self.folder,i,actor,self.identities['job']['digest'])
    def evaluate(self,rows,factory,identity,*,deadline=None):
        from training.support_benchmark_v5_1.engine import evaluate
        from training.support_nn_write_v5_1.candidate import activation
        with activation(True):return evaluate(self.folder,self.tasks,rows,factory,identity,matrix=True,deadline=deadline)

def open_owner(work,job,runtime,refs):
    from notebook_support.matched_cadic_v5_1.common import verify_source as cadic_verify
    if runtime!=refs['runtime']:raise ValueError('Runtime differs from canonical T4 parent')
    work=Path(work);root=work/'benchmark'
    if sha(root/'manifest_train.json')!=refs['train_manifest_sha256'] or sha(root/'test_manifest.json')!=refs['test_manifest_sha256']:raise ValueError('Training/test manifest changed')
    if job_name(job)=='NESTED_seed0':
        from continuation_runtime_v1.common import verify_delivery,verify_publication
        from training.support_migration_v5_1.contract import contract
        from training.support_migration_v5_1.lineage import Execution
        from training.support_throughput_v5_1.contract import contract as profile_contract
        core,_=verify_delivery();verify_publication(work)
        owner=Execution(root,core,runtime,contract(),profile_contract()['parent'])
    else:
        base_op,base_ref,_=cadic_verify();owner=CADICOwner(root,base_ref,base_op,runtime)
    folder=owner.folder;jd=owner.identities['job']['digest'];t=read(folder/'train_complete.json');f=read(folder/'final_complete.json')
    if len(owner.rows)!=3599 or len(owner.tasks)!=15 or t['job_digest']!=jd or t['images']!=3599 or not t['finalized'] or t['official_test_used_for_learning'] or f!=dict(status='final_complete',job_digest=jd,cell_count=15,train_repeated=False):raise ValueError('Completed full7.1 required; 7.2 never trains')
    # The active checkpoint is hash-bound before weights-only deserialization.
    import torch
    if read(folder/'active.json')!=dict(cursor=3599,sha256=sha(folder/'active.pt')):raise ValueError('Active receipt changed')
    active=torch.load(folder/'active.pt',map_location='cpu',weights_only=True)
    if active['cursor']!=3599 or active['job_digest']!=jd or not active['session']['finalized'] or active['session']['pending'] is not None:raise ValueError('Active endpoint changed')
    chain=digest('empty event stream')
    for i,r in enumerate(owner.rows):
        event=read(folder/'events'/f'{i:05d}.json')
        if event['cursor']!=i or event['identity']!=r['relative_path']:raise ValueError('Event order changed')
        chain=digest([chain,event])
    if chain!=active['trace_digest'] or event['bank_sha256']!=active['session']['bank_sha256']:raise ValueError('Active event trace changed')
    for i in range(15):
        own=root/'legacy/runs/NESTED_seed0' if job=='NESTED_seed0' and i==0 else folder
        p=own/'states'/f'task_{i:02d}.pt';meta=read(p.with_suffix('.json'))
        expected_jd=owner.lineage['original_job_digest'] if job=='NESTED_seed0' and i==0 else jd
        if sha(p)!=meta['sha256'] or meta['task_id']!=i or meta['cursor']!=sum(r['task_id']<=i for r in owner.rows) or meta['job_digest']!=expected_jd:raise ValueError('Task checkpoint metadata changed')
    return owner

def load_task(owner,i,actor):
    if isinstance(owner,CADICOwner):return owner.task_state(i,actor)
    return owner.task_state(owner.folder,i,actor,owner.identities['job']['digest'])

def complete(folder,tasks):
    folder=Path(folder);jd=read(folder/'job_lock.json')['digest'];n=len(tasks)
    if read(folder/'matrix_complete.json')!=dict(status='matrix_complete',job_digest=jd,cell_count=n*(n+1)//2,train_repeated=False):raise ValueError('Wrong/incomplete matrix completion')
    actual={p.parent.name for p in (folder/'cells').glob('*/completion.json')};expected={f'{i:02d}_{j:02d}' for i in range(n) for j in range(i+1)}
    if actual!=expected:raise ValueError('Missing/extra matrix cells')
    import math
    result=read(folder/'metrics/task_matrix.json')
    if set(result)!=set(('i_auroc','p_ap')):raise ValueError('Wrong matrix metric keys')
    for key,value in result.items():
        grid=value['matrix']
        if len(grid)!=n or any(len(row)!=n for row in grid):raise ValueError('Wrong matrix shape')
        for i in range(n):
            for j in range(n):
                v=grid[i][j]
                if j>i:
                    if v is not None:raise ValueError('Future-task cell must remain unevaluated')
                elif v is None or not math.isfinite(v) or v!=read(folder/'cells'/f'{i:02d}_{j:02d}'/'completion.json')['metrics'][key]:raise ValueError('Matrix/cell metric mismatch')
        forgetting=[max(grid[i][j] for i in range(j,n-1))-grid[-1][j] for j in range(n-1)]
        bwt=sum(grid[-1][j]-grid[j][j] for j in range(n-1))/(n-1) if n>1 else None
        if value['per_task_forgetting']!=forgetting or value['denominator']!=n-1 or value['clamped_at_zero'] is not False:raise ValueError('Forgetting definition changed')
        if n>1 and (abs(value['fm']-sum(forgetting)/(n-1))>1e-12 or abs(value['bwt']-bwt)>1e-12):raise ValueError('Forgetting/BWT aggregation changed')
    return result
