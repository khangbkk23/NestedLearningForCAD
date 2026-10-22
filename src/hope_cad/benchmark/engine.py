"""Atomic image training, immutable task states, resumable read-only triangular evaluation."""
import io,json,time,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from hope_cad.contract import digest,file_sha,canonical
from hope_cad.models.validation import tree_finite
from hope_cad.benchmark.stream import _state_equal
from hope_cad.benchmark.metrics import compute_metrics,macro_task_metrics,forgetting_matrix
from hope_cad.runtime.system import atomic_json,atomic_torch_save
from hope_cad.benchmark.study import lock

class Pause(Exception):pass
def boundary(deadline):
    if deadline is not None and time.monotonic()>=deadline:raise Pause('saved atomic boundary; resume latest benchmark output')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save_or_verify(path,value):
    if Path(path).exists():
        if read(path)!=canonical(value):raise ValueError('Immutable record changed: '+str(path))
    else:atomic_json(value,path)

def train(folder,rows,tasks,identity,provider,factory,deadline=None,checkpoint_every=8):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    locked=lock(folder/'job_lock.json',dict(**identity,rows=rows,task_order=tasks));active=folder/'active.pt';receipt=folder/'active.json'
    actor=factory();lock(folder/'initialization.json',actor.initialization_hashes())
    elapsed=0.;start=time.monotonic();chain=digest('empty event stream');peak=0
    if active.exists():
        state=torch.load(active,map_location='cpu',weights_only=True)
        expected={'schema','job_digest','cursor','session','trace_digest','peak_persistent_tensor_bytes','elapsed_seconds'}
        if set(state)!=expected or state['schema']!='support_benchmark_active_v1' or type(state['cursor']) is not int or type(state['peak_persistent_tensor_bytes']) is not int or state['peak_persistent_tensor_bytes']<0 or not math.isfinite(state['elapsed_seconds']) or state['elapsed_seconds']<0:raise ValueError('Malformed active checkpoint')
        if state['job_digest']!=locked['digest']:raise ValueError('Active state identity changed')
        if receipt.exists() and read(receipt)['sha256']!=file_sha(active):
            # PT->JSON publication may have been interrupted; check structural state and event prefix below.
            if state['cursor']<=read(receipt)['cursor']:raise ValueError('Active checkpoint changed without forward atomic publication')
        actor.load_state_dict(state['session']);elapsed=state['elapsed_seconds']
        peak=state['peak_persistent_tensor_bytes']
        if state['cursor']!=actor.cursor or actor.cursor>len(rows):raise ValueError('Corrupt cursor')
        if actor.seen_ids!=[r['relative_path'] for r in rows[:actor.cursor]]:raise ValueError('Active checkpoint observed identities changed')
        for i in range(actor.cursor):
            event=read(folder/'events'/f'{i:05d}.json')
            if event['cursor']!=i or event['identity']!=rows[i]['relative_path']:raise ValueError('Checkpoint event prefix mismatch')
            chain=digest([chain,event])
        if chain!=state['trace_digest']:raise ValueError('Checkpoint event chain changed')
        if actor.cursor and event['bank_sha256']!=actor.state_dict()['bank_sha256']:raise ValueError('Checkpoint bank differs from last event')
    def save():
        state=dict(schema='support_benchmark_active_v1',job_digest=locked['digest'],cursor=actor.cursor,session=actor.state_dict(),trace_digest=chain,peak_persistent_tensor_bytes=peak,elapsed_seconds=elapsed+time.monotonic()-start)
        atomic_torch_save(state,active);atomic_json(dict(cursor=actor.cursor,sha256=file_sha(active)),receipt)
    if not active.exists():save()
    # Recover the task-state publication if killed after active save at a task boundary.
    endpoints={sum(r['task_id']<=i for r in rows):i for i in range(len(tasks))}
    def task_save(i):
        path=folder/'states'/f'task_{i:02d}.pt';meta=path.with_suffix('.json')
        if path.exists():
            value=torch.load(path,map_location='cpu',weights_only=True)
            if value.get('schema')!='support_benchmark_task_v1' or value.get('job_digest')!=locked['digest'] or value.get('task_id')!=i or value.get('cursor')!=actor.cursor or not _state_equal(value['session'],actor.state_dict()):raise ValueError('Task state changed')
        else:atomic_torch_save(dict(schema='support_benchmark_task_v1',job_digest=locked['digest'],task_id=i,cursor=actor.cursor,session=actor.state_dict()),path)
        save_or_verify(meta,dict(sha256=file_sha(path),task_id=i,cursor=actor.cursor,job_digest=locked['digest']))
    if actor.cursor in endpoints:
        if actor.cursor==len(rows):actor.finalize_stream()
        task_save(endpoints[actor.cursor])
    if actor.finalized and actor.cursor!=len(rows):raise ValueError('Premature stream finalize')
    if (folder/'train_complete.json').exists():
        if not actor.finalized or actor.cursor!=len(rows) or read(folder/'train_complete.json')['job_digest']!=locked['digest']:raise ValueError('Invalid training completion')
        return actor
    for i in range(actor.cursor,len(rows)):
        try:boundary(deadline)
        except Pause:save();raise
        event=actor.observe(provider.get(rows[i]),rows[i]['relative_path'])
        save_or_verify(folder/'events'/f'{i:05d}.json',event)
        chain=digest([chain,event])
        peak=max(peak,actor.memory_profile()['persistent_tensor_bytes'])
        if actor.cursor==len(rows):actor.finalize_stream()
        if actor.cursor%checkpoint_every==0 or actor.cursor in endpoints:save()
        if actor.cursor in endpoints:task_save(endpoints[actor.cursor])
        if actor.cursor%8==0:print('[BENCH TRAIN]',identity['job'],actor.cursor,'/',len(rows),'labels',actor.counters['labels_completed'],flush=True)
    save()
    out=dict(status='train_complete',job_digest=locked['digest'],images=actor.cursor,task_count=len(tasks),counters=actor.counters,finalized=actor.finalized,memory=actor.memory_profile(),peak_persistent_tensor_bytes=peak,elapsed_seconds=elapsed+time.monotonic()-start,official_test_used_for_learning=False)
    # Time is execution metadata and can differ on repeated no-op invocation.
    if not (folder/'train_complete.json').exists():atomic_json(out,folder/'train_complete.json')
    return actor

def task_state(folder,i,actor,job_digest):
    path=Path(folder)/'states'/f'task_{i:02d}.pt';meta=read(path.with_suffix('.json'))
    if file_sha(path)!=meta['sha256'] or meta['task_id']!=i or meta['job_digest']!=job_digest:raise ValueError('Task checkpoint identity/hash changed')
    value=torch.load(path,map_location='cpu',weights_only=True)
    if set(value)!={'schema','job_digest','task_id','cursor','session'} or value['schema']!='support_benchmark_task_v1' or value['job_digest']!=job_digest or value['task_id']!=i or value['cursor']!=meta['cursor']:raise ValueError('Task checkpoint metadata changed')
    normal_rows=read(Path(folder)/'job_lock.json')['identity']['rows']
    if value['cursor']!=sum(r['task_id']<=i for r in normal_rows):raise ValueError('Task boundary cursor differs from registered training rows')
    actor.load_state_dict(value['session'])
    if actor.seen_ids!=[r['relative_path'] for r in normal_rows[:actor.cursor]]:raise ValueError('Task checkpoint observed identities changed')
    return meta['sha256']

def atomic_npz(path,**value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as out:np.savez_compressed(out,**value)
    tmp.replace(path)

def maps_from_patches(patches,size=224):
    n,d=patches.shape;side=int(d**.5)
    if side*side!=d:raise ValueError('Square raw patch geometry required')
    return F.interpolate(torch.from_numpy(patches).reshape(n,1,side,side),size=(size,size),mode='bilinear',align_corners=False)[:,0].numpy()

def cell(folder,checkpoint_id,task_id,actor,rows,provider,identity,deadline=None):
    folder=Path(folder)/'cells'/f'{checkpoint_id:02d}_{task_id:02d}';folder.mkdir(parents=True,exist_ok=True)
    cl=lock(folder/'cell_lock.json',dict(**identity,checkpoint_id=checkpoint_id,task_id=task_id,rows=rows))
    complete=folder/'completion.json'
    if complete.exists():
        value=read(complete)
        if value['cell_digest']!=cl['digest'] or value['samples']!=len(rows):raise ValueError('Cell completion identity changed')
        expected={f'chunks/{i:04d}.npz' for i in range(0,len(rows),8)}
        if set(value['chunks'])!=expected:raise ValueError('Missing completed chunks')
        for n,h in value['chunks'].items():
            if file_sha(folder/n)!=h:raise ValueError('Completed score chunk changed')
        if not all(np.isfinite(v) for v in value['metrics'].values()):raise ValueError('Undefined completed metric')
        return value['metrics']
    before=actor.state_dict();chunks=[]
    for start in range(0,len(rows),8):
        path=folder/'chunks'/f'{start:04d}.npz';meta=path.with_suffix('.json')
        expected=[r['relative_path'] for r in rows[start:start+8]]
        if path.exists() and meta.exists():
            receipt=read(meta)
            if receipt['sha256']!=file_sha(path) or receipt['identities']!=expected or receipt['cell_digest']!=cl['digest']:raise ValueError('Evaluation chunk changed')
        else:
            boundary(deadline);scores=[];patches=[];labels=[];masks=[]
            for row in rows[start:start+8]:
                raw,mask=provider.get(row)
                # Only raw features are passed to score; labels and masks never enter learner/session.
                with torch.no_grad():pixel,_,score=actor.score(raw)
                if not bool(torch.isfinite(pixel).all()) or not bool(torch.isfinite(score).all()):raise ValueError('Nonfinite score')
                scores.append(float(score[0]));patches.append(pixel[0].cpu().numpy());labels.append(row['label']);masks.append(mask)
            if not _state_equal(before,actor.state_dict()):raise RuntimeError('Scoring mutated scientific state')
            atomic_npz(path,image_scores=np.asarray(scores,dtype=np.float32),patches=np.stack(patches).astype(np.float32),labels=np.asarray(labels,dtype=np.int64),masks=np.stack(masks).astype(bool))
            save_or_verify(meta,dict(sha256=file_sha(path),identities=expected,cell_digest=cl['digest']))
        chunks.append(path)
    if not _state_equal(before,actor.state_dict()):raise RuntimeError('Evaluation changed state')
    arrays={}
    for p in chunks:
        with np.load(p,allow_pickle=False) as v:
            if set(v.files)!={'image_scores','patches','labels','masks'}:raise ValueError('Unknown score payload')
            for k in v.files:arrays.setdefault(k,[]).append(v[k])
    arrays={k:np.concatenate(v) for k,v in arrays.items()}
    if arrays['patches'].shape[0]!=len(rows) or arrays['labels'].tolist()!=[r['label'] for r in rows]:raise ValueError('Score/manifest length mismatch')
    for v in arrays.values():
        if not np.isfinite(v).all():raise ValueError('Nonfinite stored metric input')
    metrics=compute_metrics(arrays['image_scores'],arrays['labels'],maps_from_patches(arrays['patches'],arrays['masks'].shape[-1]),arrays['masks'])
    if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Undefined official metric; preserve evidence, never drop category')
    metrics.update(p_ap=metrics['p_aupr'])
    out=dict(schema='support_benchmark_cell_v1',cell_digest=cl['digest'],metrics=metrics,samples=len(rows),chunks={p.relative_to(folder).as_posix():file_sha(p) for p in chunks},scientific_state_unchanged=True)
    save_or_verify(folder/'completion.json',out)
    print('[BENCH CELL]',checkpoint_id,task_id,'complete',flush=True);return metrics

def evaluate(folder,tasks,test_rows,factory,identity,*,matrix=False,deadline=None):
    folder=Path(folder);jd=read(folder/'job_lock.json')['digest'];actor=factory();n=len(tasks)
    if not (folder/'train_complete.json').exists():raise ValueError('Evaluation requires completed training; 7.2 cannot train')
    checkpoints=range(n) if matrix else [n-1];all_metrics={}
    for i in checkpoints:
        boundary(deadline);checkpoint_sha=task_state(folder,i,actor,jd)
        for j in range(i+1):
            rows=[r for r in test_rows if r['task_id']==j]
            all_metrics[i,j]=cell(folder,i,j,actor,rows,identity['provider'],dict(job_digest=jd,checkpoint_sha256=checkpoint_sha,test_manifest_digest=identity['test_digest']),deadline)
    final={tasks[j]:all_metrics[n-1,j] for j in range(n)}
    macro=macro_task_metrics(final);macro['macro_final_p_ap']=macro['p_aupr'];macro['p_ap']=macro['p_aupr']
    save_or_verify(folder/'metrics/final_per_task.json',final);save_or_verify(folder/'metrics/final_macro.json',macro)
    if matrix:
        result={}
        for key in ('i_auroc','p_ap'):
            values=np.full((n,n),np.nan)
            for (i,j),v in all_metrics.items():values[i,j]=v[key]
            f=forgetting_matrix(values)
            f['bwt']=float(np.mean([values[-1,j]-values[j,j] for j in range(n-1)])) if n>1 else None
            result[key]=f
        save_or_verify(folder/'metrics/task_matrix.json',result)
    save_or_verify(folder/('matrix_complete.json' if matrix else 'final_complete.json'),dict(status='matrix_complete' if matrix else 'final_complete',job_digest=jd,cell_count=n*(n+1)//2 if matrix else n,train_repeated=False))
    return macro
