"""B321 complete update-boundary state; atomic, bounded, hash-verified checkpoints.

Only load explicitly selected, trusted task artifacts. weights_only=True limits
unpickling; hashes prove byte identity, not provenance/authenticity.
"""
import copy,dataclasses,hashlib,json,os,random,shutil,sys,time,zipfile
from pathlib import Path
INTERVAL=500
KEEP=2

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1024**2):h.update(block)
    return h.hexdigest()

def due(update):return update>0 and (update%INTERVAL==0 or update==10005)

def selection_digest(rows):
    h=hashlib.sha256()
    for row in rows:h.update((json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode('utf8'))
    return h

def cpu_tree(t,value):
    if isinstance(value,t.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(t,v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(cpu_tree(t,v) for v in value)
    return copy.deepcopy(value)

def runtime_identity(t):
    import numpy as np
    return dict(torch=str(t.__version__),numpy=np.__version__,python=list(sys.version_info[:3]),cuda=t.version.cuda,devices=[{'name':t.cuda.get_device_name(i),'capability':list(t.cuda.get_device_capability(i))} for i in range(t.cuda.device_count())] if t.cuda.is_available() else [],deterministic=t.are_deterministic_algorithms_enabled(),matmul_tf32=t.backends.cuda.matmul.allow_tf32,cudnn_tf32=t.backends.cudnn.allow_tf32,cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))

def capture(t,model,opt,learner,schedule,batches,store,study,selection_rows,stage,iteration,identity):
    import numpy as np
    ctrl=learner.ctrl
    controller=None if ctrl is None else dict(phi=cpu_tree(t,dict(ctrl.phi.parameters)),m=cpu_tree(t,dict(ctrl.adam.m)),v=cpu_tree(t,dict(ctrl.adam.v)),step=ctrl.adam.step,eligible=ctrl.eligible_commits,slow=ctrl.slow_steps,objective=ctrl.objective,normalize_context=ctrl.normalize_context)
    np_rng=np.random.get_state()
    return dict(schema='b321_checkpoint_v1',identity=identity,stage=stage,iteration=iteration,update=learner.counts['optimizer'],model=cpu_tree(t,model.state_dict()),optimizer=cpu_tree(t,opt.state_dict()),controller=controller,counts=copy.deepcopy(learner.counts),schedule=dict(admission=dataclasses.asdict(schedule.admission),reservoir=list(schedule.reservoir),fifo=list(schedule.fifo),seen=sorted(schedule.seen),metas=copy.deepcopy(schedule.metas),serial=schedule.serial,rng={n:g.getstate() for n,g in schedule.rng.items()}),batches=dict(ids=list(batches.ids),size=batches.size,queue=list(batches.queue),rng=batches.rng.getstate()),store=cpu_tree(t,store),study=copy.deepcopy(study),selection_rows=copy.deepcopy(selection_rows),rng=dict(torch=t.get_rng_state().clone(),cuda=[x.cpu().clone() for x in t.cuda.get_rng_state_all()] if t.cuda.is_available() else [],python=random.getstate(),numpy=[np_rng[0],t.from_numpy(np_rng[1].copy()),int(np_rng[2]),int(np_rng[3]),float(np_rng[4])]),runtime=runtime_identity(t))

def restore(t,state,model,opt,learner,schedule,batches,expected):
    import numpy as np
    from nested_replay_cad.core.contracts import PcgState,PhiState,AdamState
    from nested_replay_cad.replay import NestedReplay
    if state.get('schema')!='b321_checkpoint_v1' or state.get('identity')!=expected:raise ValueError('Checkpoint identity/schema mismatch')
    if state['runtime']!=runtime_identity(t):raise ValueError('Checkpoint runtime mismatch; cross-runtime resume not qualified')
    update=state['update'];stage=state['stage'];iteration=state['iteration']
    if update!=(stage-1)*667+iteration or not 1<=stage<=15 or not 0<=iteration<=667 or state['counts']['optimizer']!=update:raise ValueError('Checkpoint cursor/count mismatch')
    if len(state['selection_rows'])!=update or selection_digest(state['selection_rows']).hexdigest()!=state['study']['data_selection_sha256']:raise ValueError('Checkpoint selection transcript mismatch')
    if set(state['store'])!=set(state['schedule']['reservoir'])|set(state['schedule']['fifo']):raise ValueError('Checkpoint memory membership mismatch')
    if state['schedule']['serial']!=len(state['schedule']['seen']):raise ValueError('Checkpoint admission serial mismatch')
    b=state['batches']
    if list(batches.ids)!=b['ids'] or batches.size!=b['size']:raise ValueError('Checkpoint current task batch identities mismatch')
    if set(state['schedule']['rng'])!=set(schedule.rng):raise ValueError('Checkpoint sampling RNG set mismatch')
    model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
    if any(opt.state[p].get('step',0)!=update for p in model.parameters()):raise ValueError('Checkpoint optimizer clock mismatch')
    learner.counts=copy.deepcopy(state['counts']);c=state['controller'];learner.ctrl=None
    if c is not None:
        ctrl=NestedReplay(learner.seed,c['objective'],device=str(next(model.parameters()).device),normalize_context=c['normalize_context'])
        ctrl.phi=PhiState({n:v.clone().requires_grad_() for n,v in c['phi'].items()});ctrl.adam=AdamState(c['m'],c['v'],c['step']);ctrl.eligible_commits=c['eligible'];ctrl.slow_steps=c['slow'];learner.ctrl=ctrl
    s=state['schedule'];schedule.admission=PcgState(**s['admission']);schedule.reservoir=list(s['reservoir']);schedule.fifo=list(s['fifo']);schedule.seen=set(s['seen']);schedule.metas=copy.deepcopy(s['metas']);schedule.serial=s['serial']
    for n,rng in s['rng'].items():schedule.rng[n].setstate(rng)
    b=state['batches']
    if list(batches.ids)!=b['ids'] or batches.size!=b['size']:raise ValueError('Checkpoint current task batch identities mismatch')
    batches.queue=list(b['queue']);batches.rng.setstate(b['rng'])
    r=state['rng'];t.set_rng_state(r['torch']);cuda=r['cuda']
    if len(cuda)!=(t.cuda.device_count() if t.cuda.is_available() else 0):raise ValueError('Checkpoint visible CUDA topology mismatch')
    if cuda:t.cuda.set_rng_state_all(cuda)
    random.setstate(r['python']);n=r['numpy'];np.random.set_state((n[0],n[1].numpy(),n[2],n[3],n[4]))
    return copy.deepcopy(state['store'])

def save(t,root,state,guard=lambda:None,fault=lambda point:None):
    """Publish directory commit marker last; keep previous two complete states.

    Only this run's completed checkpoints enter retention. Partial writes are
    left for audit; never replace or delete a checkpoint belonging to another run.
    """
    from nested_replay_cad.storage import DiskGuard,tensor_bytes
    root=Path(root);root.mkdir(parents=True,exist_ok=True);name='g'+str(state['update']).zfill(5)
    target=root/name;partial=root/(name+'.partial')
    if target.exists() or partial.exists():raise FileExistsError(target)
    # A conservative uncompressed tensor estimate, plus metadata/ZIP overhead.
    def tensors(v):
        if isinstance(v,t.Tensor):yield v
        elif isinstance(v,dict):
            for x in v.values():yield from tensors(x)
        elif isinstance(v,(tuple,list)):
            for x in v:yield from tensors(x)
    DiskGuard(root.parent.parent).check(force=True,additional_bytes=int(tensor_bytes(tensors(state))*1.02)+16*1024**2)
    guard();partial.mkdir();payload=partial/'state.pt';start=time.monotonic()
    with payload.open('xb') as stream:t.save(state,stream);stream.flush();os.fsync(stream.fileno())
    fault('payload_written')
    with zipfile.ZipFile(payload) as archive:
        if archive.testzip() is not None:raise ValueError('Checkpoint ZIP CRC mismatch')
    record=dict(schema='b321_checkpoint_commit_v1',update=state['update'],stage=state['stage'],iteration=state['iteration'],sha256=digest(payload),bytes=payload.stat().st_size,identity=state['identity'],seconds=time.monotonic()-start)
    marker=partial/'COMMIT.json'
    with marker.open('x',encoding='utf8',newline='\n') as f:f.write(json.dumps(record,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    guard();fault('before_publish');partial.rename(target)
    # Ensure directory publication is durable on the supported Linux runtime.
    if os.name=='posix':
        fd=os.open(root,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    with (root/'ledger.jsonl').open('a',encoding='utf8',newline='\n') as f:f.write(json.dumps(dict(record,path=name))+'\n');f.flush();os.fsync(f.fileno())
    complete=sorted(p for p in root.glob('g[0-9][0-9][0-9][0-9][0-9]') if p.is_dir())
    for old in complete[:-KEEP]:
        rec=json.loads((old/'COMMIT.json').read_text(encoding='utf8'))
        if rec['identity']!=state['identity'] or old.is_symlink() or old.resolve().parent!=root.resolve():raise ValueError('Unsafe checkpoint retention scope')
        if digest(old/'state.pt')!=rec['sha256']:raise ValueError('Refuse retention of corrupted checkpoint')
        shutil.rmtree(old)
    return dict(record,path=str(target))

def load(t,checkpoint,expected):
    checkpoint=Path(checkpoint)
    if checkpoint.name.endswith('.partial') or checkpoint.is_symlink():raise ValueError('Uncommitted/unsafe checkpoint')
    rec=json.loads((checkpoint/'COMMIT.json').read_text(encoding='utf8'))
    payload=checkpoint/'state.pt'
    if rec.get('schema')!='b321_checkpoint_commit_v1' or rec.get('identity')!=expected or payload.stat().st_size!=rec['bytes'] or digest(payload)!=rec['sha256']:raise ValueError('Checkpoint commit/identity/hash mismatch')
    state=t.load(payload,map_location='cpu',weights_only=True)
    if state['update']!=rec['update'] or state['identity']!=expected:raise ValueError('Checkpoint marker/payload mismatch')
    return state


def copy_evidence(source_arm,destination,study):
    """Copy only hash-verified completed boundary evidence into a fresh run.

    Never copy a partial boundary or write into the source run. Archived scores
    live in the shared parent, retaining the same producer-relative locators.
    """
    source_arm=Path(source_arm).resolve();destination=Path(destination).resolve()
    if source_arm==destination or source_arm.name!=study['arm'] or destination.name!=study['arm']:raise ValueError('Resume arm/output scope mismatch')
    if len(list(destination.iterdir())):raise FileExistsError('Resume output must be empty')
    for snapshot in study['snapshots']:
        for record in snapshot['score_archives']:
            name=record['path']
            if Path(name).name!=name or not name.startswith(study['arm']+'_g'):raise ValueError('Unsafe parent score archive')
            old=source_arm.parent/name;new=destination.parent/name
            if old.is_symlink() or new.exists():raise FileExistsError('Unsafe/existing continuation archive')
            if old.stat().st_size!=record['bytes'] or digest(old)!=record['sha256']:raise ValueError('Parent score archive mismatch')
            with old.open('rb') as src,new.open('xb') as dst:shutil.copyfileobj(src,dst,1024**2)
            if digest(new)!=record['sha256']:raise ValueError('Copied archive mismatch')
    (destination/'resume-lineage.json').write_text(json.dumps({'source_arm':str(source_arm),'completed_snapshots':len(study['snapshots']),'scope':'Only hash-verified complete score archives; parent run unchanged'},indent=2)+'\n',encoding='utf8')
