"""New execution owner and narrowly scoped inherited-task resolution; frozen engine stays intact."""
from contextlib import contextmanager
from pathlib import Path
import io,json,os,shutil
import torch
from training.runtime import atomic_json,atomic_torch_save
from training.support_throughput_v5_1.contract import safe,sha,digest,ROOT
from training.support_throughput_v5_1.replay import assert_equal,state_digest
from training.support_nn_write_v5_1.candidate import activation
from training.support_benchmark_v5_1 import engine
from .contract import make_identity

ORIGINAL_TASK_STATE=engine.task_state

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def inherited_names(info,start):
    names={'suite_lock.json','manifest_train.json','data_rights.json','runs/NESTED_seed0/job_lock.json','runs/NESTED_seed0/initialization.json',
           'runs/NESTED_seed0/states/task_00.pt','runs/NESTED_seed0/states/task_00.json'}
    for i in range(start):
        names.add(f'runs/NESTED_seed0/events/{i:05d}.json')
        names.add('runs/NESTED_seed0/feature_receipts/'+digest(info['rows'][i])+'.json')
    return sorted(names)

def publish(work,evidence,info,identities,result,source,policy,parent_active,*,source_map,base_source_map,pins):
    """Canonical output appears by directory rename only after exact complete shadow bridge."""
    work=Path(work);target=work/'benchmark';stage=work/'.publication_pending'
    if target.exists() or stage.exists():raise ValueError('Publication destination already exists; no overwrite/retry')
    start,end=policy['start_cursor'],policy['end_cursor_exclusive']
    if result['ledger']['status']!='complete' or result['ledger']['completed_pairs']!=list(range(start,end)) or result['state']['cursor']!=end or result['state']['finalized']:raise ValueError('Incomplete bridge cannot publish')
    if not (work/'shadow/attempts.json').exists():raise ValueError('Missing durable bridge ledger')
    stage.mkdir();job=stage/'runs/NESTED_seed0';job.mkdir(parents=True)
    legacy=stage/'legacy';legacy.mkdir()
    # Preserve the raw manifest bytes, whose SHA is pinned by the contract.
    rawmanifest=evidence.archive.read('artifact_manifest.json') if evidence.archive else (evidence.path/'artifact_manifest.json').read_bytes()
    (legacy/'artifact_manifest.json').write_bytes(rawmanifest)
    inherited={}
    for name in inherited_names(info,start):
        raw=evidence.read(name);path=safe(legacy,name);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        inherited[name]=evidence.manifest['files'][name]
        if name in ('manifest_train.json','data_rights.json'):(stage/name).write_bytes(raw)
        prefix='runs/NESTED_seed0/'
        if name.startswith(prefix) and ('/events/' in name or '/feature_receipts/' in name or name.endswith('/initialization.json')):
            dst=safe(job,name[len(prefix):]);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
    for name,lock in [('suite_lock.json',identities['suite']),('execution_lock.json',identities['execution'])]:atomic_json(lock,stage/name)
    atomic_json(identities['job'],job/'job_lock.json')
    for p in (work/'shadow/events').glob('*.json'):
        dst=job/'events'/p.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    for p in (work/'feature_receipts').glob('*.json'):
        dst=job/'feature_receipts'/p.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    candidate_seconds=result['ledger']['times']['candidate']
    deployment_components=candidate_seconds+result['ledger']['times']['provider']
    state=dict(schema='support_benchmark_active_v1',job_digest=identities['job']['digest'],cursor=end,session=result['state'],
        trace_digest=result['trace_digest'],peak_persistent_tensor_bytes=max(parent_active['peak_persistent_tensor_bytes'],result['peak_persistent_tensor_bytes']),
        elapsed_seconds=parent_active['elapsed_seconds']+deployment_components)
    atomic_torch_save(state,stage/'bridge_checkpoint.pt');shutil.copy2(stage/'bridge_checkpoint.pt',job/'active.pt')
    atomic_json(dict(cursor=end,sha256=sha(job/'active.pt')),job/'active.json')
    atomic_json(source_map,stage/'execution_source_map.json');atomic_json(base_source_map,stage/'original_source_map.json')
    lineage=dict(schema='nn_write_owner_lineage_v1',inherited=inherited,original_job_digest=info['job']['digest'],
        original_suite_digest=info['suite']['digest'],parent_manifest_sha256=info['manifest_sha256'],inherited_task_ids=[0],
        new_job_digest=identities['job']['digest'],new_execution_digest=identities['execution']['digest'],source=source,
        original_task_owner='legacy/runs/NESTED_seed0',new_task_owner='runs/NESTED_seed0',bridge_start=start,bridge_end=end)
    atomic_json(lineage,stage/'lineage.json')
    mutable={'runs/NESTED_seed0/active.pt','runs/NESTED_seed0/active.json'}
    immutable={p.relative_to(stage).as_posix():sha(p) for p in stage.rglob('*') if p.is_file() and p.relative_to(stage).as_posix() not in mutable}
    receipt=dict(schema='nn_write_migration_receipt_v1',status='accepted_for_exact_parent_and_source_pair',engineering_id=policy['id'],
        parent=dict(manifest_sha256=info['manifest_sha256'],archive_sha256=policy['parent_archive_sha256'],suite_digest=info['suite']['digest'],job_digest=info['job']['digest'],active_sha256=policy['parent_active_sha256'],task00_sha256=inherited['runs/NESTED_seed0/states/task_00.pt']),
        execution_digest=identities['execution']['digest'],new_job_digest=identities['job']['digest'],new_source=source,
        old_source_digest=info['suite']['identity']['source_digest'],new_source_map_sha256=sha(stage/'execution_source_map.json'),old_source_map_sha256=sha(stage/'original_source_map.json'),
        changed_files=dict(modified_existing_payloads=[],added_payloads=sorted(set(source_map['files'])-set(base_source_map['files'])),
            operator_override='CADICPatchCoresetV1._nearest -> nearest_fixed_shape_writes_v1; scoped only; scientific arithmetic unchanged'),
        reviews=policy['review_receipts_sha256'],pending_check=result['pending_check'],bridge=dict(start=start,end=end,ids=[r['relative_path'] for r in info['rows'][start:end]],trace_digest=result['trace_digest'],scientific_state_digest=state_digest(result['state']),checkpoint_sha256=sha(stage/'bridge_checkpoint.pt')),
        compute=dict(parent_elapsed_seconds=parent_active['elapsed_seconds'],candidate_seconds=candidate_seconds,shared_provider_seconds=result['ledger']['times']['provider'],
            deployment_components_seconds=deployment_components,shadow_ledger=result['ledger'],active_elapsed_includes_shadow=False,
            accounting='active carries parent+provider+candidate observe; comparison/bridge serialization/bootstrap/export are separate overhead; not full notebook or cold-start runtime'),
        immutable_files=immutable,human_review_required=True,full_benchmark_authorized=False,AD_superiority=False,
        equality='exact scientific events/full state each arrival; no timing/path/derived-cache equality claim')
    atomic_json(receipt,stage/'migration_receipt.json')
    atomic_json(dict(schema='nn_write_migration_completion_v1',receipt_sha256=sha(stage/'migration_receipt.json')),stage/'migration_complete.json')
    Execution(stage,source,identities['execution']['identity']['runtime'],policy,pins)
    os.rename(stage,target)
    return receipt

class Execution:
    def __init__(self,root,source,runtime,policy,pins):
        self.root=Path(root);self.folder=self.root/'runs/NESTED_seed0';self.source=source;self.runtime=runtime
        marker=read(self.root/'migration_complete.json');receipt=read(self.root/'migration_receipt.json')
        if marker.get('schema')!='nn_write_migration_completion_v1' or sha(self.root/'migration_receipt.json')!=marker.get('receipt_sha256') or receipt['status']!='accepted_for_exact_parent_and_source_pair':raise ValueError('Missing/corrupt accepted migration publication')
        for name,h in receipt['immutable_files'].items():
            if sha(safe(self.root,name))!=h:raise ValueError('Immutable migrated payload changed: '+name)
        self.lineage=read(self.root/'lineage.json');lineage=self.lineage
        raw=(self.root/'legacy/artifact_manifest.json').read_bytes()
        if __import__('hashlib').sha256(raw).hexdigest()!=pins['artifact_manifest_sha256']:raise ValueError('Original manifest changed')
        manifest=json.loads(raw)
        for name,h in lineage['inherited'].items():
            if manifest['files'].get(name)!=h or sha(safe(self.root/'legacy',name))!=h:raise ValueError('Inherited owner/hash changed')
        suite=read(self.root/'legacy/suite_lock.json');oldjob=read(self.root/'legacy/runs/NESTED_seed0/job_lock.json')
        if suite['digest']!=pins['suite_digest'] or oldjob['digest']!=pins['job_digest']:raise ValueError('Unknown original owner')
        info=dict(suite=suite,job=oldjob,rows=oldjob['identity']['rows'],manifest_sha256=pins['artifact_manifest_sha256'])
        expected=make_identity(info,source,runtime,policy)
        if read(self.root/'execution_lock.json')!=expected['execution'] or read(self.root/'suite_lock.json')!=expected['suite'] or read(self.folder/'job_lock.json')!=expected['job']:raise ValueError('New execution/source/runtime identity changed')
        if receipt['new_source']!=source or receipt['reviews']!=policy['review_receipts_sha256'] or receipt['parent']['active_sha256']!=pins['active_sha256'] or receipt['parent']['archive_sha256']!=pins['archive_sha256']:raise ValueError('Receipt prerequisite/parent/source changed')
        if digest(read(self.root/'execution_source_map.json')['files'])!=source['source_digest'] or digest(read(self.root/'original_source_map.json')['files'])!=suite['identity']['source_digest']:raise ValueError('Source map digest changed')
        if lineage['inherited_task_ids']!=[0] or lineage['bridge_start']!=policy['start_cursor'] or lineage['bridge_end']!=policy['end_cursor_exclusive'] or lineage['new_job_digest']!=expected['job']['digest'] or lineage['original_job_digest']!=oldjob['digest'] or lineage['new_execution_digest']!=expected['execution']['digest']:raise ValueError('Task lineage changed')
        if set(lineage['inherited'])!=set(inherited_names(info,policy['start_cursor'])):raise ValueError('Inherited inventory changed')
        if sha(self.root/'bridge_checkpoint.pt')!=receipt['bridge']['checkpoint_sha256']:raise ValueError('Immutable bridge checkpoint changed')
        if receipt['execution_digest']!=expected['execution']['digest'] or receipt['new_job_digest']!=expected['job']['digest'] or receipt['engineering_id']!=policy['id']:raise ValueError('Receipt owner changed')
        if receipt['parent']['manifest_sha256']!=pins['artifact_manifest_sha256'] or receipt['parent']['suite_digest']!=suite['digest'] or receipt['parent']['job_digest']!=oldjob['digest']:raise ValueError('Receipt parent changed')
        if receipt['bridge']['start']!=policy['start_cursor'] or receipt['bridge']['end']!=policy['end_cursor_exclusive'] or receipt['bridge']['ids']!=[r['relative_path'] for r in info['rows'][policy['start_cursor']:policy['end_cursor_exclusive']]]:raise ValueError('Bridge scope changed')
        checkpoint=torch.load(self.root/'bridge_checkpoint.pt',map_location='cpu',weights_only=True)
        if checkpoint['schema']!='support_benchmark_active_v1' or checkpoint['job_digest']!=expected['job']['digest'] or checkpoint['cursor']!=policy['end_cursor_exclusive'] or checkpoint['session']['seen_ids']!=[r['relative_path'] for r in info['rows'][:checkpoint['cursor']]] or checkpoint['session']['finalized'] or state_digest(checkpoint['session'])!=receipt['bridge']['scientific_state_digest']:raise ValueError('Bridge checkpoint scientific owner/state changed')
        chain=digest('empty event stream')
        for i in range(checkpoint['cursor']):
            event=read(self.folder/'events'/f'{i:05d}.json')
            if event['cursor']!=i or event['identity']!=info['rows'][i]['relative_path']:raise ValueError('Migrated event prefix changed')
            chain=digest([chain,event])
        if chain!=checkpoint['trace_digest'] or chain!=receipt['bridge']['trace_digest'] or event['bank_sha256']!=checkpoint['session']['bank_sha256']:raise ValueError('Migrated trace/bank differs')
        self.receipt=receipt;self.identities=expected;self.rows=info['rows'];self.tasks=oldjob['identity']['task_order']
    def task_state(self,folder,i,actor,job_digest):
        if Path(folder).resolve()!=self.folder.resolve() or job_digest!=self.identities['job']['digest'] or type(i) is not int or not 0<=i<len(self.tasks):raise ValueError('Wrong execution owner/task')
        if i==0:
            legacy=self.root/'legacy/runs/NESTED_seed0'
            for suffix in ('.pt','.json'):
                n='runs/NESTED_seed0/states/task_00'+suffix
                if sha(self.root/'legacy'/n)!=self.lineage['inherited'][n]:raise ValueError('Inherited task changed after execution opened')
            return ORIGINAL_TASK_STATE(legacy,i,actor,self.lineage['original_job_digest'])
        return ORIGINAL_TASK_STATE(folder,i,actor,job_digest)
    @contextmanager
    def installed(self):
        if engine.task_state is not ORIGINAL_TASK_STATE:raise ValueError('Unknown task resolver already installed')
        engine.task_state=self.task_state
        try:yield
        finally:engine.task_state=ORIGINAL_TASK_STATE
    def train(self,provider,factory,deadline=None):
        identity={k:v for k,v in self.identities['job']['identity'].items() if k not in ('rows','task_order')}
        with activation(True):return engine.train(self.folder,self.rows,self.tasks,identity,provider,factory,deadline=deadline)
    def evaluate(self,test_rows,factory,identity,*,matrix=False,deadline=None):
        with self.installed(),activation(True):return engine.evaluate(self.folder,self.tasks,test_rows,factory,identity,matrix=matrix,deadline=deadline)
