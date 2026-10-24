"""Verified, exact-identity research handoff. Never trains or grants acceptance."""
import gc
import json
import math
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from hope_cad.contract import digest, file_sha
from hope_cad.paths import ROOT, WORKSPACE
from hope_cad.research import contract, job, method_identity, session_for, study_digest
from hope_cad.runtime.identity import source_digest, source_files
from hope_cad.runtime.system import atomic_json

SCHEMA = 'hope_stage_handoff_v1'
HEADER = 'handoff_manifest.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=bad_constant,
                      object_pairs_hook=unique_object)


def bad_constant(value):
    raise ValueError('Nonfinite JSON: '+value)


def unique_object(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError('Duplicate JSON field: '+key)
        value[key] = item
    return value


def safe_name(name):
    p = PurePosixPath(name)
    if (not isinstance(name, str) or not name or p.is_absolute() or p.as_posix() != name
            or any(part in ('..', '.') or part.endswith((' ', '.')) for part in p.parts)
            or any(c in name for c in '\\:\x00')
            or any(ord(c) < 32 for c in name)):
        raise ValueError('Unsafe handoff path: '+repr(name))
    if name != HEADER and p.parts[0] not in ('experiment','records','assets','source'):
        raise ValueError('Unknown payload prefix')
    return name


def inspect_archive(source, *, extract=None):
    """Hash every member before using any payload; extract only into empty staging."""
    source = Path(source)
    with zipfile.ZipFile(source) as z:
        entries = z.infolist()
        names = [safe_name(e.filename) for e in entries]
        if (len(names) != len(set(n.casefold() for n in names)) or len(names) > 1000000
                or sum(e.file_size for e in entries) > 512*1024**3):
            raise ValueError('Duplicate/case-colliding or excessive handoff inventory')
        for entry in entries:
            mode = entry.external_attr >> 16
            if entry.is_dir() or (stat.S_IFMT(mode) not in (0,stat.S_IFREG)) or entry.flag_bits & 1:
                raise ValueError('No directories, links, devices or encrypted members')
        if HEADER not in names or z.getinfo(HEADER).file_size > 64*1024**2:
            raise ValueError('Missing/oversized handoff manifest')
        manifest = json.loads(z.read(HEADER),parse_constant=bad_constant,object_pairs_hook=unique_object)
        if (manifest.get('schema') != SCHEMA or set(manifest['files']) != set(names)-{HEADER}
                or not re.fullmatch('[0-9a-f]{40}',manifest.get('source_commit',''))
                or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}',manifest.get('experiment_id',''))):
            raise ValueError('Handoff schema/inventory/origin mismatch')
        for name, receipt in manifest['files'].items():
            if set(receipt) != {'sha256','bytes'} or receipt['bytes'] != z.getinfo(name).file_size:
                raise ValueError('Payload size mismatch: '+name)
            import hashlib
            with z.open(name) as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest() != receipt['sha256']:
                    raise ValueError('Payload checksum mismatch: '+name)
        sf = manifest['source_file_hashes']
        if (digest(sf) != manifest['source_digest'] or
                {n[7:]:r['sha256'] for n,r in manifest['files'].items() if n.startswith('source/')} != sf):
            raise ValueError('Source snapshot/identity mismatch')
        if extract is not None:
            root = Path(extract)
            if any(root.iterdir()):
                raise ValueError('Extract only into empty staging')
            for name in names:
                dest = root/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                with z.open(name) as inp, dest.open('xb') as out:
                    shutil.copyfileobj(inp,out,8*1024**2)
                if name != HEADER and file_sha(dest) != manifest['files'][name]['sha256']:
                    raise ValueError('Extracted payload changed')
        return manifest


def registered():
    cfg = contract()['study']
    main = cfg['C_full_seed0_jobs']
    extension = [v+'_seed0' for v in cfg['D_conditional_remaining_seed0_controls']]+cfg['D_conditional_replicates']
    return cfg, main, extension


def validate_job(folder, rows, tasks, expected, *, test=None, device='cpu'):
    """Load actual state on its numerical device, plus all event/task/cell bindings."""
    import torch
    from hope_cad.benchmark.engine import task_state
    from hope_cad.contract import CONTRACT
    from hope_cad.runtime.system import capture_rng, restore_rng
    folder = Path(folder)
    if not folder.exists():
        return dict(started=False,train=False,matrix=False,cursor=0)
    locked = read(folder/'job_lock.json')
    if locked['digest'] != digest(locked['identity']) or locked['identity'] != dict(**expected,rows=rows,task_order=tasks):
        raise ValueError('Job identity/rows changed: '+folder.name)
    pointer = read(folder/'active.json')
    if file_sha(folder/'active.pt') != pointer['sha256']:
        raise ValueError('Uncommitted/corrupt active checkpoint: '+folder.name)
    state = torch.load(folder/'active.pt',map_location='cpu',weights_only=True)
    cursor = state['cursor']
    if (state['schema'] != 'support_benchmark_active_v1' or state['job_digest'] != locked['digest']
            or type(cursor) is not int or not 0 <= cursor <= len(rows) or pointer['cursor'] != cursor):
        raise ValueError('Active checkpoint identity/cursor changed')
    chain = digest('empty event stream')
    for i,row in enumerate(rows[:cursor]):
        event = read(folder/'events'/f'{i:05d}.json')
        if event['cursor'] != i or event['identity'] != row['relative_path']:
            raise ValueError('Event chronology changed')
        chain = digest([chain,event])
    if chain != state['trace_digest']:
        raise ValueError('Event trace changed')
    rng = capture_rng()
    actor = None
    try:
        actor = session_for(folder.name,device)
        init = read(folder/'initialization.json')
        if init['digest'] != digest(init['identity']) or init['identity'] != actor.initialization_hashes():
            raise ValueError('Initialization changed')
        actor.load_state_dict(state['session'])
        if actor.cursor != cursor or actor.seen_ids != [r['relative_path'] for r in rows[:cursor]]:
            raise ValueError('Scientific state prefix differs')
        if cursor and event['bank_sha256'] != actor.state_dict()['bank_sha256']:
            raise ValueError('Scientific bank/event mismatch')
        for row in rows[:cursor]:
            receipt = read(folder/'feature_receipts'/(digest(row)+'.json'))
            if (receipt['row'] != row or receipt['backbone_sha256'] != CONTRACT['backbone_sha256']
                    or not re.fullmatch('[0-9a-f]{64}',receipt.get('tensor_sha256',''))):
                raise ValueError('Feature receipt row changed')
        completed = (folder/'train_complete.json').exists()
        if completed:
            done = read(folder/'train_complete.json')
            if (done.get('status') != 'train_complete' or done['job_digest'] != locked['digest']
                    or done['images'] != len(rows) or done['task_count'] != len(tasks)
                    or done.get('finalized') is not True or not actor.finalized or cursor != len(rows)):
                raise ValueError('Fabricated training completion')
        elif actor.finalized:
            # A hard crash after final active publication can recover completion in engine.
            if cursor != len(rows):
                raise ValueError('Premature finalized state')
        checkpoints = {}
        for i in range(len(tasks)):
            p = folder/'states'/f'task_{i:02d}.pt'
            if p.exists():
                if read(p.with_suffix('.json'))['cursor'] > cursor:
                    raise ValueError('Task checkpoint ahead of committed active state')
                checkpoints[i] = task_state(folder,i,actor,locked['digest'])
            elif completed:
                raise ValueError('Completed job missing task checkpoint')
        for cell_folder in sorted((folder/'cells').glob('*')):
            if test is None or not cell_folder.is_dir() or not re.fullmatch(r'\d{2}_\d{2}',cell_folder.name):
                raise ValueError('Unknown cell or missing test manifest')
            i,j = map(int,cell_folder.name.split('_'))
            ci = read(cell_folder/'cell_lock.json')
            selected = [r for r in test['rows'] if r['task_id'] == j]
            expected_cell = dict(job_digest=locked['digest'],checkpoint_sha256=checkpoints.get(i),
                test_manifest_digest=test['digest'],checkpoint_id=i,task_id=j,rows=selected)
            if not 0 <= j <= i < len(tasks) or ci['identity'] != expected_cell or ci['digest'] != digest(expected_cell):
                raise ValueError('Cell identity/test/checkpoint changed')
            completion = read(cell_folder/'completion.json') if (cell_folder/'completion.json').exists() else None
            actual = {}
            for chunk in (cell_folder/'chunks').glob('*.npz'):
                meta = read(chunk.with_suffix('.json')); offset=int(chunk.stem)
                if offset % 8 or offset >= len(selected) or chunk.stem != f'{offset:04d}':
                    raise ValueError('Unknown score chunk')
                if (meta['cell_digest'] != ci['digest'] or meta['sha256'] != file_sha(chunk)
                        or meta['identities'] != [r['relative_path'] for r in selected[offset:offset+8]]):
                    raise ValueError('Score chunk changed')
                actual['chunks/'+chunk.name] = file_sha(chunk)
            if completion is not None and (completion['cell_digest'] != ci['digest']
                    or completion.get('scientific_state_unchanged') is not True
                    or completion['samples'] != len(selected) or completion['chunks'] != actual
                    or set(actual) != {f'chunks/{o:04d}.npz' for o in range(0,len(selected),8)}
                    or not all(type(v) in (int,float) and math.isfinite(v) for v in completion['metrics'].values())):
                raise ValueError('Incomplete/corrupt cell completion')
        for kind in ('final','matrix'):
            p = folder/(kind+'_complete.json')
            if not p.exists():
                continue
            marker = read(p); n=len(tasks)
            pairs = [(n-1,j) for j in range(n)] if kind == 'final' else [(i,j) for i in range(n) for j in range(i+1)]
            if not completed or marker != dict(status=kind+'_complete',job_digest=locked['digest'],cell_count=len(pairs),train_repeated=False):
                raise ValueError('Invalid evaluation completion')
            if not all((folder/'cells'/f'{i:02d}_{j:02d}'/'completion.json').is_file() for i,j in pairs):
                raise ValueError('Missing completed matrix/final cells')
            import numpy as np
            from hope_cad.benchmark.metrics import macro_task_metrics,forgetting_matrix
            values={(i,j):read(folder/'cells'/f'{i:02d}_{j:02d}'/'completion.json')['metrics'] for i,j in pairs}
            final={tasks[j]:values[len(tasks)-1,j] for j in range(len(tasks))}
            macro=macro_task_metrics(final);macro['macro_final_p_ap']=macro['p_aupr'];macro['p_ap']=macro['p_aupr']
            if read(folder/'metrics/final_macro.json') != macro or read(folder/'metrics/final_per_task.json') != final:
                raise ValueError('Final metrics differ from completed cells')
            if kind == 'matrix':
                matrices={}
                for key in ('i_auroc','p_ap'):
                    array=np.full((n,n),np.nan)
                    for (i,j),v in values.items():
                        array[i,j]=v[key]
                    f=forgetting_matrix(array)
                    f['bwt']=float(np.mean([array[-1,j]-array[j,j] for j in range(n-1)])) if n>1 else None
                    matrices[key]=f
                if read(folder/'metrics/task_matrix.json') != matrices:
                    raise ValueError('Task matrix differs from completed cells')
        return dict(started=True,train=completed,matrix=(folder/'matrix_complete.json').is_file(),cursor=cursor)
    finally:
        del actor
        gc.collect()
        restore_rng(rng)


def accepted(run, phase, identity):
    p = run/(phase+'.json'); rp=run/(phase+'_review.json')
    if not p.exists() or not rp.exists():
        return False
    v=read(p); r=read(rp)
    if (r.get('accepted') is not True or r.get('phase') != phase
            or r.get('source_digest') != identity['source_digest'] or r.get('receipt_sha256') != file_sha(p)):
        raise ValueError('Review/receipt binding changed: '+phase)
    if phase == 'preflight':
        if (v.get('status') != 'research_preflight_passed_pending_review'
                or v.get('binding') != identity or v.get('binding_digest') != digest(identity)):
            raise ValueError('Preflight identity changed')
    elif v.get('passed') is not True or r.get('pilot_receipt_sha256') != file_sha(p):
        raise ValueError('Pilot review/gate failed')
    return True


def plan(coverage, preflight, pilot, screen, *, preflight_exists=False, pilot_exists=False):
    """Explicit scheduler decision; never grants reviewer acceptance."""
    cfg,main,extension = registered()
    if not preflight:
        return dict(status='review_required' if preflight_exists else 'ready_phase1',phase='preflight',stage='B')
    if not pilot:
        return dict(status='review_required' if pilot_exists else 'ready_phase1',phase='pilot',stage='B')
    for phase,names,stage in (('main',main,'C'),('extension',extension,'D')):
        if phase == 'extension' and screen is not True:
            return dict(status='blocked_screen',phase=None,stage=None)
        for name in names:
            if not coverage.get(name,{}).get('matrix'):
                return dict(status='ready_phase2' if phase == 'main' else 'ready_phase3',
                            phase=phase,stage=stage,job=name,completed_jobs=[n for n,v in coverage.items() if v['matrix']])
    return dict(status='complete',phase=None,stage=None)


def inspect_run(run, *, own_lock=False, device='cpu'):
    """Strict metadata/state audit. Can preserve partial or negative experiments."""
    from hope_cad.data.protocol import CANONICAL_MVTEC_TASKS
    from hope_cad.contract import CONTRACT
    run=Path(run)
    if (run/'.writer.lock').exists() and not own_lock:
        raise ValueError('Active writer: do not pack/check a changing run')
    suite=read(run/'suite_lock.json'); identity=suite['identity']
    if suite['digest'] != digest(identity) or identity['research_study_digest'] != study_digest():
        raise ValueError('Suite/study changed')
    manifest=read(run/'manifest_train.json'); rows=manifest['online']; tasks=manifest['task_order']
    if (manifest['digest'] != digest({k:v for k,v in manifest.items() if k != 'digest'})
            or manifest['source_identity'] != identity['source_digest'] or digest(rows) != identity['normal_rows_digest']
            or tasks != list(CANONICAL_MVTEC_TASKS) or len(rows) != 3629
            or sum(r['task_id'] == 0 for r in rows) != 209 or manifest['excluded'] != []
            or len({r['relative_path'] for r in rows}) != 3629):
        raise ValueError('Full-train manifest changed')
    cfg,main,extension=registered(); coverage={}; pilot_coverage={}
    test=read(run/'test_manifest.json') if (run/'test_manifest.json').exists() else None
    if test is not None and (len(test['rows']) != 1725 or test['task_order'] != tasks
            or test['digest'] != digest({k:test[k] for k in ('task_order','rows','mask_qa')})):
        raise ValueError('Official test manifest changed')
    for category,names in (('pilot',[v+'_seed0' for v in cfg['B_pilot_job_order']]),('runs',main+extension)):
        parent=run/category
        if parent.exists() and {p.name for p in parent.iterdir()}-set(names):
            raise ValueError('Unregistered job directory')
        for name in names:
            variant,seed=job(name)
            pilot=category == 'pilot'
            selected=rows[:256] if pilot else rows
            expected=dict(job=name,mode=variant,seed=seed,suite_digest=digest(identity),
                method_digest=digest(method_identity(variant)),scope='discarded_pilot_256' if pilot else 'fresh_fulltrain_3629')
            value=validate_job(parent/name,selected,tasks[:2] if pilot else tasks,expected,test=test,device=device)
            (pilot_coverage if pilot else coverage)[name]=value
    pf=accepted(run,'preflight',identity); pg=accepted(run,'pilot',identity)
    if test is not None and not any(v['train'] for v in coverage.values()):
        raise ValueError('Official test metadata without a completed own training')
    if any(v['started'] for v in coverage.values()) and not pg:
        raise ValueError('Main/extension state without accepted pilot')
    if any(v['started'] for v in pilot_coverage.values()) and not (run/'preflight.json').exists():
        raise ValueError('Pilot state without preflight')
    if pg:
        p=read(run/'pilot.json')
        if (not pf or not all(v['train'] for v in pilot_coverage.values())
                or set(p['arms']) != set(cfg['B_pilot_job_order']) or not p['gates'] or not all(v is True for v in p['gates'].values())):
            raise ValueError('Accepted pilot lacks completed arms/gates')
    screen=None
    if (run/'development_screen.json').exists():
        if not pg or not all(coverage[n]['matrix'] for n in main):
            raise ValueError('Screen without completed accepted predecessors')
        data={n:read(run/'runs'/n/'metrics/final_macro.json') for n in main}
        h,r,b=[data[n] for n in ('H32_EXACT_seed0','R_EXACT_seed0','CADIC_seed0')]
        gates=dict(history_pixel=h['macro_final_p_ap']-r['macro_final_p_ap']>0,
            history_image=h['macro_final_i_auroc']-r['macro_final_i_auroc']>=-.02,
            cadic_image=h['macro_final_i_auroc']-b['macro_final_i_auroc']>=-.02,
            cadic_pixel=h['macro_final_p_ap']-b['macro_final_p_ap']>=-.02)
        value=read(run/'development_screen.json')
        if value != dict(passed=all(gates.values()),gates=gates,metrics=data,scope='spend_screen_only_not_confirmatory_NI'):
            raise ValueError('Development screen changed')
        screen=value['passed']
    result=plan(coverage,pf,pg,screen,preflight_exists=(run/'preflight.json').exists(),pilot_exists=(run/'pilot.json').exists())
    if (run/'pilot.json').exists() and read(run/'pilot.json').get('passed') is False:
        result.update(status='blocked_pilot',phase=None,stage=None)
    if (run/'preflight.json').exists() and read(run/'preflight.json').get('status') != 'research_preflight_passed_pending_review':
        result.update(status='blocked_preflight',phase=None,stage=None)
    completed_phases=[]
    if pg:
        completed_phases.append(1)
        if all(coverage[n]['matrix'] for n in main):
            completed_phases.append(2)
            if all(coverage[n]['matrix'] for n in extension):
                completed_phases.append(3)
    result.update(identity=identity,coverage=coverage,pilot_coverage=pilot_coverage,
                  completed_phases=completed_phases,accepted_preflight=pf,accepted_pilot=pg)
    return result


def ledger_check(run, experiment):
    """Verify closed live reservations and immutable export-sidecar bindings."""
    ids=set()
    ledgers=list(Path(run).glob('budget_*.json'))
    if not ledgers:
        raise ValueError('budget_unresolved: missing origin ledger')
    for ledger in ledgers:
        value=read(ledger); binding=value['binding']
        if ledger.name != 'budget_'+binding['stage']+'.json' or binding['stage'] not in ('B','C','D'):
            raise ValueError('budget_unresolved: ledger stage changed')
        for attempt in value['attempts']:
            if attempt['id'] in ids or attempt['status'] != 'closed_after_verified_export':
                raise ValueError('budget_unresolved: open/duplicate reservation')
            ids.add(attempt['id'])
            matches=[]
            for sidecar in (Path(experiment)/'exports').glob('session_*/accounting.json'):
                receipt=read(sidecar)
                if receipt.get('attempt',{}).get('id') == attempt['id']:
                    matches.append((sidecar,receipt))
            if len(matches) != 1:
                raise ValueError('budget_unresolved: missing/conflicting accounting sidecar')
            sidecar,receipt=matches[0]; archive=sidecar.parent/'evidence.zip'
            if (receipt.get('schema') != 'hope_research_accounting_v1' or receipt['binding'] != binding
                    or receipt['attempt'] != attempt or receipt['archive']['sha256'] != file_sha(archive)):
                raise ValueError('budget_unresolved: accounting/archive mismatch')
            from hope_cad.runtime.evidence import read_manifest
            old,_=read_manifest(archive)
            if old['schema'] != 'hope_hq_fulltrain_evidence_v1':
                raise ValueError('budget_unresolved: wrong origin archive schema')
            with zipfile.ZipFile(archive) as z:
                old_ledger=json.loads(z.read(ledger.name))
            if old_ledger['binding'] != binding or not any(a['id'] == attempt['id'] and a['status'] == 'export_incomplete'
                    and a['reserved_invocation_seconds'] == attempt['reserved_invocation_seconds'] for a in old_ledger['attempts']):
                raise ValueError('budget_unresolved: no corresponding export reservation')
            elapsed=attempt['elapsed_seconds']; external=attempt['external_billed_seconds']
            if (not all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (elapsed,external,attempt['charged_seconds']))
                    or abs(attempt['charged_seconds']-(elapsed+180+external))>1e-6):
                raise ValueError('budget_unresolved: invalid charge')


def pack(run_dir, output=None):
    import subprocess
    from hope_cad.runtime.execution import exclusive_work
    run=Path(run_dir).resolve(); experiment=run.parent
    if run.name != 'run' or experiment.parent != WORKSPACE/'experiments':
        raise ValueError('Use canonical workspace/experiments/<id>/run')
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}',experiment.name):
        raise ValueError('Invalid experiment ID')
    output=Path(output).resolve() if output else experiment/'handoffs'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex+'.zip')
    if output.exists() or output.with_suffix(output.suffix+'.tmp').exists():
        raise ValueError('Do not overwrite a handoff archive')
    if not output.is_relative_to(experiment/'handoffs'):
        raise ValueError('Store outer ZIP under experiment/handoffs, outside its payload')
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
        raise ValueError('Portable handoff requires committed clean source; commit/push before cohort')
    with exclusive_work(run):
        origin=read(run/'suite_lock.json')['identity']
        state_device=origin['runtime'].get('device','cpu')
        if str(state_device).startswith('cuda'):
            from hope_cad.runtime.identity import configure_runtime,check_remote_runtime
            configure_runtime()
            if check_remote_runtime(state_device,origin['runtime']['device_name']) != origin['runtime']:
                raise ValueError('Pack state qualification requires the recorded runtime; preserve raw experiment if unavailable')
        summary=inspect_run(run,own_lock=True,device=state_device)
        paths={}
        for prefix,base in (('experiment',experiment),('records',WORKSPACE/'records')):
            for path in sorted(base.rglob('*')):
                if path.is_symlink():
                    raise ValueError('No symlinks in handoff input')
                if not path.is_file():
                    continue
                rel=path.relative_to(base)
                if (prefix == 'experiment' and rel.parts[0] == 'handoffs') or (prefix == 'records' and rel.parts[0] in ('transfers','handoff_checks')):
                    continue
                if path.name == '.writer.lock' or path.name.endswith(('.tmp','.import_tmp')):
                    continue
                paths[safe_name(prefix+'/'+rel.as_posix())]=path
        for name,path in (('mvtec.asset.json',WORKSPACE/'data/mvtec.asset.json'),
                          ('vit.asset.json',WORKSPACE/'weights/vit_base_patch8_224.augreg_in21k.asset.json')):
            if not path.is_file() or path.is_symlink():
                raise ValueError('Missing asset receipt: '+str(path))
            paths['assets/'+name]=path
        sf=source_files()
        if summary['identity']['source_digest'] != digest(sf):
            raise ValueError('Current source differs from research origin; legacy import unsupported')
        paths.update({'source/'+n:ROOT/n for n in sf})
        manifest=dict(schema=SCHEMA,experiment_id=experiment.name,source_commit=subprocess.check_output(
            ['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),source_digest=digest(sf),source_file_hashes=sf,
            suite_identity=summary['identity'],files={n:dict(sha256=file_sha(p),bytes=p.stat().st_size) for n,p in paths.items()})
        output.parent.mkdir(parents=True,exist_ok=True)
        tmp=output.with_suffix(output.suffix+'.tmp')
        try:
            with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
                for name,path in paths.items():
                    z.write(path,name)
                z.writestr(HEADER,json.dumps(manifest,sort_keys=True,allow_nan=False))
            inspect_archive(tmp)
            if source_files() != sf:
                raise ValueError('Source changed during pack')
            tmp.replace(output)
        finally:
            tmp.unlink(missing_ok=True)
    return dict(path=str(output),sha256=file_sha(output),bytes=output.stat().st_size,
                stage_summary={k:summary.get(k) for k in ('status','phase','job','completed_phases')},
                scope='Download this one ZIP and retain SHA separately; not a new scientific acceptance')


def target_probe(run, data_root, checkpoint, device):
    """Fixed normals, exact source feature receipts and disposable read-only states."""
    import torch
    from hope_cad.contract import CONTRACT
    from hope_cad.models.features import CADICViTConfig, CADICViTFeatureExtractor
    from hope_cad.runtime.cadic_diagnostic import Inputs
    from hope_cad.models.primitives.state import tensor_digest
    from hope_cad.benchmark.stream import _state_equal
    from hope_cad.runtime.system import capture_rng,restore_rng
    rows=read(run/'manifest_train.json')['online']; checked=[]; rng=capture_rng()
    extractor=CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=str(checkpoint),
        checkpoint_identity='timm/vit_base_patch8_224.augreg_in21k'),device)
    try:
        inputs=Inputs(data_root,extractor,device); raw_cache={}
        with torch.no_grad():
            # Even a preflight-only handoff has genuine source feature references.
            for receipt_path in sorted((run/'preflight_features').glob('*.json')):
                receipt=read(receipt_path)
                if receipt.get('row') == rows[0]:
                    raw=inputs.get(rows[0])[0]
                    if (receipt['backbone_sha256'] != CONTRACT['backbone_sha256']
                            or receipt['tensor_sha256'] != tensor_digest(({'patches':raw},))):
                        raise ValueError('Target preflight feature parity failed')
                    raw_cache[0]=raw
                    checked.append(dict(job='discarded_preflight',index=0))
                    break
            for parent in ('pilot','runs'):
                for folder in sorted((run/parent).glob('*')):
                    if not (folder/'active.pt').is_file():
                        continue
                    state=torch.load(folder/'active.pt',map_location='cpu',weights_only=True)
                    actor=session_for(folder.name,device)
                    actor.load_state_dict(state['session']); before=actor.state_dict()
                    for index in (0,128,255):
                        if index >= state['cursor']:
                            continue
                        if index not in raw_cache:
                            raw_cache[index]=inputs.get(rows[index])[0]
                        raw=raw_cache[index]
                        receipt=read(folder/'feature_receipts'/(digest(rows[index])+'.json'))
                        if (receipt['backbone_sha256'] != CONTRACT['backbone_sha256']
                                or receipt['tensor_sha256'] != tensor_digest(({'patches':raw},))):
                            raise ValueError('Target raw feature parity failed: '+folder.name)
                        out=actor.score(raw)
                        if not isinstance(out,tuple) or len(out) != 3 or not all(bool(torch.isfinite(v).all()) for v in out):
                            raise ValueError('Target scoring nonfinite')
                        if not _state_equal(before,actor.state_dict()):
                            raise ValueError('Target scoring mutated scientific state')
                        checked.append(dict(job=folder.name,index=index))
                    del actor,state,before
                    gc.collect(); torch.cuda.empty_cache()
        if (run/'preflight.json').exists() and not checked:
            raise ValueError('No genuine observed feature reference for target qualification')
        # No trained jobs means only a new discarded kernel smoke; cannot open main.
        x=torch.ones((4,4),device=device)
        if not torch.equal(x@x,torch.full_like(x,4.)):
            raise ValueError('Target CUDA smoke failed')
        torch.cuda.synchronize(device)
        return dict(passed=True,checks=checked,scope='Finite fixed normals/read-only clones, not whole-trajectory proof')
    finally:
        extractor.close(); restore_rng(rng)


def prepare_allocation(run, stage, rate, cap):
    if not all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in (rate,cap)):
        raise ValueError('budget_unresolved: supply actual positive --rate and --cap for next stage')
    value=dict(stage=stage,authorized=True,USD_per_hour=rate,USD_cap=cap)
    if stage == 'D':
        value['gpu_wall_seconds_cap']=cap/rate*3600
    ledger=Path(run)/('budget_'+stage+'.json')
    if ledger.exists():
        old=read(ledger)
        if old['binding']['allocation'] != value:
            raise ValueError('budget_unresolved: current stage rate/cap differs; v1 never rewrites or resets ledger')
        if old['binding']['cap_seconds']-sum(a['charged_seconds'] for a in old['attempts'])<=180:
            raise ValueError('budget_unresolved: stage allocation exhausted')
    return value


def check(archive=None, *, restore=False, device='cuda:0', expected_gpu=None, data_root=None,
          checkpoint=None, rate=None, cap=None):
    """Always return a concrete readiness report; blocked never means rerun automatically."""
    from hope_cad.paths import DATA, WEIGHTS
    from hope_cad.runtime.identity import configure_runtime, check_remote_runtime
    from hope_cad.runtime.execution import verify_checkpoint
    from hope_cad.data.fulltrain import build_manifest
    from hope_cad.runtime.research import binding
    report=dict(schema='hope_stage_handoff_check_v1',status='fresh',next_phase='preflight',
        allowed=False,automatic_training=False,scientific_acceptance=False)
    scratch=None
    try:
        if archive is None:
            report.update(status='fresh',reason='No handoff supplied; begin fresh Phase1 after CADIC diagnostic')
            return report
        manifest=inspect_archive(archive)
        report.update(archive_sha256=file_sha(archive),origin_commit=manifest['source_commit'])
        if manifest['source_file_hashes'] != source_files():
            report.update(status='blocked_source',reason='Source snapshot differs; no legacy/source-mapping support in v1')
            return report
        # Pending predictions are exact-device computations in the existing
        # loader. Never try to validate GPU-produced pending state on CPU.
        configure_runtime()
        import torch
        actual=expected_gpu or (torch.cuda.get_device_name(device) if torch.cuda.is_available() else '')
        rt=check_remote_runtime(device,actual)
        data_root=data_root or DATA/'mvtec'
        checkpoint=checkpoint or WEIGHTS/'vit_base_patch8_224.augreg_in21k.safetensors'
        _,rd=build_manifest(data_root,source_digest())
        current=binding(rt,rd,verify_checkpoint(checkpoint))
        if current != manifest['suite_identity']:
            report.update(status='blocked_identity',reason='Source/runtime/data/backbone differs; no compatibility override',
                different_fields=sorted(k for k in current if current[k] != manifest['suite_identity'].get(k)))
            return report
        staging_root=WORKSPACE/'checks/handoff_staging'
        staging_root.mkdir(parents=True,exist_ok=True)
        scratch=Path(tempfile.mkdtemp(dir=staging_root))
        inspect_archive(archive,extract=scratch)
        experiment=scratch/'experiment'; run=experiment/'run'
        summary=inspect_run(run,device=device)
        if summary['identity'] != manifest['suite_identity']:
            raise ValueError('Outer/inner suite binding differs')
        report.update(status=summary['status'],next_phase=summary['phase'],job=summary.get('job'),
            completed_phases=summary['completed_phases'],accepted_preflight=summary['accepted_preflight'],
            accepted_pilot=summary['accepted_pilot'])
        ledger_check(run,experiment)
        if summary['status'] in ('review_required','blocked_preflight','blocked_pilot','blocked_screen','complete'):
            report['reason']='Evidence retained; reviewer/gates determine next work, never rerun automatically'
            return report
        if (run/'test_manifest.json').exists():
            # Only after a completed own training already permitted test access.
            from hope_cad.benchmark.data import test_manifest
            if test_manifest(data_root,read(run/'manifest_train.json')['task_order']) != read(run/'test_manifest.json'):
                report.update(status='blocked_identity',reason='Official test/GT bytes differ',different_fields=['official_test'])
                return report
        allocation=prepare_allocation(run,summary['stage'],rate,cap)
        if not restore:
            report['restore_required']=True
        dest=WORKSPACE/'experiments'/manifest['experiment_id']
        if restore and dest.exists():
            raise ValueError('Target experiment already exists; never overwrite live work')
        if any(p.is_symlink() for p in (WORKSPACE,WORKSPACE/'experiments',dest)):
            raise ValueError('Target workspace/experiment cannot be a symlink')
        report['probe']=target_probe(run,data_root,checkpoint,device)
        report.update(allocation=allocation,run_dir=str(dest/'run'),allowed=bool(restore),compatible=True,restore_required=not restore,
            reason='Exact identity, evidence, fixed probe and budget checked; user launches next command')
        if restore:
            origin=experiment/'origin'/report['archive_sha256'];origin.mkdir(parents=True)
            for name in ('records','assets','source'):
                if (scratch/name).exists():
                    shutil.move(str(scratch/name),str(origin/name))
            shutil.copyfile(scratch/HEADER,origin/HEADER)
            atomic_json(dict(archive_sha256=report['archive_sha256'],probe=report['probe'],source_commit=manifest['source_commit']),origin/'restore_receipt.json')
            allocation_path=origin/(summary['stage']+'_target_allocation.json')
            atomic_json(allocation,allocation_path)
            dest.parent.mkdir(parents=True,exist_ok=True)
            experiment.rename(dest)
            report['restored']=True
        report['allocation_path']=str(dest/'origin'/report['archive_sha256']/(summary['stage']+'_target_allocation.json')) if restore else None
        report['unrecorded_billing_note']='Record target setup/idle and packaging/upload time once with --external-billed-seconds; gate does not spend/reset a ledger.'
        phase=summary['phase']; name=summary.get('job')
        report['command']=['bash','scripts/remote/run.sh','research','--phase',phase,'--run-dir',str(dest/'run'),
            '--device',str(device),'--expected-gpu',actual,'--session-hours','1']
        if restore:
            report['command']+=['--allocation',report['allocation_path']]
        if name:
            report['command']+=['--job',name]
        if not restore:
            report['command']=None
        return report
    except Exception as exc:
        report.update(status='blocked_budget' if 'budget_unresolved' in str(exc) else 'blocked_evidence',
                      allowed=False,reason=str(exc),next_phase=None)
        return report
    finally:
        if scratch is not None:
            # Only our newly created staging directory, never a user-supplied path.
            if scratch.parent.resolve() == (WORKSPACE/'checks/handoff_staging').resolve():
                try:
                    shutil.rmtree(scratch)
                except OSError as exc:
                    report['staging_retained']=str(scratch)
                    report['cleanup_warning']=str(exc)
        folder=WORKSPACE/'records/handoff_checks';folder.mkdir(parents=True,exist_ok=True)
        report['report_path']=str(folder/(uuid.uuid4().hex+'.json'))
        atomic_json(report,Path(report['report_path']))
