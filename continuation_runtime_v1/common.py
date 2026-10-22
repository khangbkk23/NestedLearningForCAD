"""Hash-bound operational policy and parent transport, separate from r2 source identity."""
import hashlib,json,math,shutil,tarfile,zipfile
from pathlib import Path,PurePosixPath
from training.support_throughput_v5_1.contract import ROOT,sha,digest,safe,Evidence
from notebook_support.common.evidence import restore_archive

SCHEMA='hope_support_v5_1_continuation_7_1_evidence_v1'
SOURCE_SCHEMA='hope_support_v5_1_continuation_7_1_source_v1'
BRIDGE_SCHEMA='hope_support_v5_1_nn_write_bridge_evidence_v1'
BRIDGE_MANIFEST='05350f6f35cc2cbfa4e79f3fd906889229898f9d451e792564621569fa7a0993'
BRIDGE_ZIP='8d173d666f82fd03ab7c2a10cef6d63a1243a9b9c5a8ca117c08b94297e0b3ad'
CHECKPOINT='337aaf57025cf74f60a8f669053e24ced6d47125dcad614eb3b2678a608221df'
CORE_DIGEST='07b2b53d2927332feac56c6b86ddea2f69f1cbdeb23a5d0153896684450a06ee'
EXECUTION='3538438ba058c3abb6c10522b6bb22d2af1ae958be0c99a84978e5dc09b01b03'
RESERVE=600

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def verify_delivery():
    from training.support_migration_v5_1.contract import verify_source
    m=read(ROOT/'continuation_manifest.json')
    if m.get('schema')!=SOURCE_SCHEMA:raise ValueError('Unknown continuation source schema')
    for n,h in m['files'].items():
        if sha(safe(ROOT,n))!=h:raise ValueError('Continuation source drift: '+n)
    core=read(ROOT/'bridge_manifest.json')
    if len(core['files'])!=221 or digest(core['files'])!=CORE_DIGEST:raise ValueError('Wrong r2 core identity')
    if any(m['files'].get(n)!=h for n,h in core['files'].items()) or m['files'].get('bridge_manifest.json')!=sha(ROOT/'bridge_manifest.json'):raise ValueError('Continuation replaced r2 payloads')
    review=read(ROOT/'continuation_runtime_v1/receipts/G4_review.json')
    audit=read(ROOT/'continuation_runtime_v1/receipts/G4_audit.json')
    if review.get('status')!='finalized' or review.get('scientific_status')!='G4_scoped_pass_checkpoint384_ready_for_continuation_interface' or not audit.get('passed'):raise ValueError('Missing reviewed G4 acceptance')
    refs={v['path'].split('/')[-1]:v['sha256'] for v in review['files']}
    if refs.get('audit.json')!=sha(ROOT/'continuation_runtime_v1/receipts/G4_audit.json'):raise ValueError('G4 audit differs from sealed review')
    if sha(ROOT/'continuation_runtime_v1/receipts/G4_manifest.json')!=BRIDGE_MANIFEST:raise ValueError('Reviewed bridge manifest bytes changed')
    return verify_source(),dict(schema='continuation_orchestration_v1',source_digest=digest(m['files']),core_source_digest=CORE_DIGEST,execution_digest=EXECUTION,job='NESTED_seed0',phase='7.1')

def verify_publication(work):
    """Rebind each inherited G4 payload to the actual reviewed manifest on every resume."""
    accepted=read(ROOT/'continuation_runtime_v1/receipts/G4_manifest.json')
    mutable={'benchmark/runs/NESTED_seed0/active.pt','benchmark/runs/NESTED_seed0/active.json'}
    for n,h in accepted['files'].items():
        if n.startswith('benchmark/') and n not in mutable and sha(safe(work,n))!=h:raise ValueError('Reviewed G4 publication changed: '+n)
    if sha(Path(work)/'origin/artifact_manifest.json')!=BRIDGE_MANIFEST:raise ValueError('Original bridge manifest lineage changed')
    if sha(Path(work)/'origin/handoff.json')!=accepted['files']['handoff.json']:raise ValueError('Original bridge handoff changed')

def deadline(started,hours):
    if isinstance(hours,bool) or not isinstance(hours,(int,float)) or not math.isfinite(hours) or not .1<=hours<=11 or hours*3600<=RESERVE:raise ValueError('SESSION_HOURS must be finite, > export reserve600s and <=11h')
    return started+hours*3600-RESERVE

def select_parent(inputs,override=None):
    paths=[Path(override)] if override else [p.parent for p in Path(inputs).rglob('artifact_manifest.json')]+list(Path(inputs).rglob('*.zip'))
    choices=[]
    for p in paths:
        try:
            with Evidence(p) as e:
                s=e.manifest.get('schema')
                if s not in (SCHEMA,BRIDGE_SCHEMA):
                    if override:raise ValueError('Explicit parent has wrong schema')
                    continue
                if s==BRIDGE_SCHEMA:
                    if e.manifest_sha256!=BRIDGE_MANIFEST or (not p.is_dir() and sha(p)!=BRIDGE_ZIP):raise ValueError('Wrong reviewed bridge evidence')
                    if e.json('handoff.json')['status']!='bridge_complete_pending_review':raise ValueError('Bridge not complete')
                else:
                    if e.json('handoff.json')['status'] not in ('paused_at_atomic_boundary','7.1_complete_pending_review'):raise ValueError('Failed/incomplete bootstrap evidence needs review; no automatic retry')
                choices.append((e.manifest_sha256,p,s))
        except (OSError,KeyError,zipfile.BadZipFile) as exc:
            if override:raise ValueError('Invalid explicit parent') from exc
    if not choices:raise ValueError('Attach reviewed bridge384 or latest continuation output; set PARENT_OVERRIDE if needed')
    if len({h for h,_,_ in choices})!=1:raise ValueError('Multiple parent versions; remove older outputs or set PARENT_OVERRIDE')
    return sorted(choices,key=lambda x:str(x[1]))[0]

def restore_parent(parent,kind,work,orchestration):
    """Require a clean output; verify every payload, including files not copied forward."""
    from training.runtime import atomic_json
    work=Path(work)
    if any(work.iterdir()):raise ValueError('Use a fresh kernel/clean work directory; no in-place retry')
    with Evidence(parent) as e:
        count=e.verify_all()
        if kind==SCHEMA:
            if e.json('continuation_lock.json')!=orchestration:raise ValueError('Continuation orchestration identity changed')
            restore_archive(parent,work,schema=SCHEMA)
        else:
            if e.manifest['files'].get('benchmark/bridge_checkpoint.pt')!=CHECKPOINT:raise ValueError('Bridge checkpoint384 differs')
            restore_archive(parent,work,schema=BRIDGE_SCHEMA,select=lambda n:n.startswith('benchmark/'))
            origin=work/'origin';origin.mkdir()
            for n in ['artifact_manifest.json','handoff.json']:
                raw=json.dumps(e.manifest,sort_keys=True).encode() if n=='artifact_manifest.json' else e.read(n)
                # Keep original manifest bytes, not a reserialized version.
                if n=='artifact_manifest.json':raw=(parent/'artifact_manifest.json').read_bytes() if Path(parent).is_dir() else e.archive.read(n)
                (origin/n).write_bytes(raw)
            atomic_json(orchestration,work/'continuation_lock.json')
    return count

def stage_tar(archive,root,*,normal_rows=None,evaluation=False):
    """One sequential raw-byte staging pass; no future feature computation."""
    root=Path(root);targets={r['relative_path'] for r in normal_rows} if normal_rows is not None else None
    if not evaluation and targets is None:raise ValueError('Training staging needs registered normal rows')
    if targets is not None and (len(targets)!=len(normal_rows) or any(r.get('split')!='train' or '/train/good/' not in r['relative_path'] for r in normal_rows)):raise ValueError('Forbidden/duplicate normal staging rows')
    seen=set()
    tasks=set(r['task_name'] for r in normal_rows) if normal_rows else set(__import__('training.support_benchmark_v5_1.contract',fromlist=['study']).study()['task_order'])
    with tarfile.open(archive) as z:
        for m in z:
            p=PurePosixPath(m.name)
            if p.is_absolute() or '..' in p.parts or '\\' in m.name or ':' in m.name or m.issym() or m.islnk():raise ValueError('Unsafe dataset tar')
            if not m.isfile():continue
            # Resolve optional archive wrapper by exact category component.
            matches=[i for i,v in enumerate(p.parts) if v in tasks]
            if len(matches)>1:raise ValueError('Ambiguous dataset member')
            if not matches:continue
            name=PurePosixPath(*p.parts[matches[0]:]).as_posix();parts=PurePosixPath(name).parts
            allowed=name in targets if targets is not None and not evaluation else len(parts)>=4 and parts[1] in ('test','ground_truth') and name.endswith('.png')
            if not allowed:continue
            if name in seen:raise ValueError('Duplicate dataset member: '+name)
            seen.add(name);dest=safe(root,name);dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.is_symlink():raise ValueError('Symlink staging destination')
            with z.extractfile(m) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
    if targets is not None and not evaluation and seen!=targets:raise ValueError('Dataset missing registered normal bytes')
    return len(seen)
