"""Fresh portable execution cohort, reusing the unchanged scientific engine."""
import gc,json,math,os,time,uuid
from pathlib import Path
from contextlib import contextmanager

from hope_cad.runtime.identity import ROOT,configure_runtime,runtime_identity,source_digest,output_directory

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

@contextmanager
def exclusive_work(work):
    work=Path(work);work.mkdir(parents=True,exist_ok=True);path=work/'.writer.lock'
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
        yield
    finally:path.unlink(missing_ok=True)

def normal_manifest(data_root):
    import yaml
    from hope_cad.data.prefix import build_prefix_manifest
    from hope_cad.contract import digest
    from hope_cad.benchmark.study import study
    cfg=yaml.safe_load((ROOT/'src/hope_cad/resources/protocols/mvtec_matched.yaml').read_text(encoding='utf-8'))
    cfg['dataset']['root']=str(Path(data_root).resolve())
    m=build_prefix_manifest(cfg,seed=0,source_commit='portable-source:'+source_digest())
    s=study()
    if len(m['online'])!=s['online_images'] or m['task_order']!=s['task_order']:
        raise ValueError('Wrong registered train manifest/order; no reduced official benchmark')
    excluded={r['relative_path'] for k in ('normal_validation','normal_probe') for r in m['roles'][k]}
    if len(excluded)!=30 or excluded&{r['relative_path'] for r in m['online']}:
        raise ValueError('Matched excluded-normal rights changed')
    return m,digest(m['online'])

def verify_checkpoint(path):
    from hope_cad.contract import file_sha,CONTRACT
    actual=file_sha(path)
    if actual!=CONTRACT['backbone_sha256']:raise ValueError('Locked ViT checkpoint SHA256 differs')
    return actual

def require_preflight(folder,runtime,normal_rows_digest,checkpoint_sha):
    from hope_cad.contract import digest,canonical,design_digest
    value=read(Path(folder)/'preflight.json');binding=value.get('binding',{})
    expected=dict(source_digest=source_digest(),design_digest=design_digest(),runtime=runtime,
                  normal_rows_digest=normal_rows_digest,backbone_sha256=checkpoint_sha)
    if (value.get('status')!='remote_preflight_passed_pending_review' or value.get('remote_acceptance') is not True
        or binding!=canonical(expected) or value.get('binding_digest')!=digest(expected)):
        raise ValueError('Remote preflight/source/runtime/data/checkpoint identity mismatch; local smoke is insufficient')
    return value

def run(c):
    import torch
    from hope_cad.contract import CONTRACT,design_digest,digest,canonical
    from hope_cad.models.features import CADICViTConfig,CADICViTFeatureExtractor
    from hope_cad.runtime.system import atomic_json,resource_usage
    from hope_cad.benchmark.study import study,study_digest,job,session_for,lock
    from hope_cad.benchmark.data import NormalFeatures,EvaluationFeatures,test_manifest
    from hope_cad.benchmark.engine import train,evaluate,save_or_verify,Pause
    from hope_cad.runtime.backend import activation
    from hope_cad.runtime.evidence import export_run

    hours=c['session_hours']
    if not math.isfinite(hours) or hours<=0:raise ValueError('session-hours must be finite and positive')
    # Starts before manifest hashing, so setup consumes the declared budget too.
    started=time.monotonic();deadline=started+max(0.,hours*3600-180.);attempt_id=uuid.uuid4().hex
    work=output_directory(c['run_dir'],data_root=c['data_root'],checkpoint=c['checkpoint'])
    mode,seed=job(c['job']);configure_runtime();rt=runtime_identity(c['device'])
    if not str(c['device']).startswith('cuda') or not torch.cuda.is_available():raise ValueError('Real execution requires CUDA and accepted remote preflight')
    manifest,rows_digest=normal_manifest(c['data_root']);checkpoint_sha=verify_checkpoint(c['checkpoint'])
    require_preflight(c['preflight'],rt,rows_digest,checkpoint_sha)
    rows,tasks=manifest['online'],manifest['task_order'];cfg=study()
    identity=dict(schema='portable_support_suite_v1',source_digest=source_digest(),study_digest=study_digest(),
        design_digest=design_digest(),runtime=rt,normal_rows_digest=rows_digest,backbone_sha256=checkpoint_sha,
        execution='fresh_portable_fixed_shape_NN_v1',historical_T4_resume_supported=False)
    extractor=None;status='failed_preserve_evidence';failure=None
    with exclusive_work(work):
        try:
            suite=lock(work/'suite_lock.json',identity)
            save_or_verify(work/'manifest_train.json',manifest)
            ji=dict(job=c['job'],mode=mode,seed=seed,suite_digest=suite['digest'],rows_digest=rows_digest,
                fresh_initialization=True,learned_state_imported=False)
            folder=work/'runs'/c['job'];factory=lambda:session_for(c['job'],c['device'])
            if c['command']!='train':
                if not (folder/'train_complete.json').is_file():raise ValueError('Evaluation cannot train or construct missing checkpoints')
                if c['command']=='matrix' and not (folder/'final_complete.json').is_file():
                    raise ValueError('Complete final evaluation first; matrix adds missing cells only')
                lock(folder/'job_lock.json',dict(**ji,rows=rows,task_order=tasks))
            extractor=CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=c['checkpoint'],checkpoint_identity='timm/vit_base_patch8_224.augreg_in21k'),c['device'])
            with activation(True):
                if c['command']=='train':
                    provider=NormalFeatures(c['data_root'],extractor,folder/'feature_receipts',c['device'])
                    actor=train(folder,rows,tasks,ji,provider,factory,deadline=deadline)
                    del actor;gc.collect();torch.cuda.empty_cache()
                    status='train_complete_pending_review'
                else:
                    # Official test discovery exists only after completed own training.
                    tm=test_manifest(c['data_root'],tasks)
                    if len(tm['rows'])!=cfg['test_images']:raise ValueError('Official test scope changed')
                    save_or_verify(work/'test_manifest.json',tm)
                    provider=EvaluationFeatures(c['data_root'],extractor,c['device'])
                    evaluate(folder,tasks,tm['rows'],factory,dict(provider=provider,test_digest=tm['digest']),
                        matrix=c['command']=='matrix',deadline=deadline)
                    status=('matrix' if c['command']=='matrix' else 'final')+'_complete_pending_review'
        except Pause as exc:
            status='paused_at_atomic_boundary';atomic_json(dict(reason=str(exc)),work/'pause.json')
        except BaseException as exc:
            failure=exc;atomic_json(dict(error=repr(exc)),work/'failure.json')
        finally:
            if extractor is not None:
                try:extractor.close()
                except BaseException as exc:
                    status='failed_preserve_evidence';failure=failure or exc
            try:resources=resource_usage(c['device'])
            except BaseException as exc:
                resources=dict(unavailable=repr(exc));failure=failure or exc;status='failed_preserve_evidence'
            atomic_json(dict(status=status,job=c['job'],command=c['command'],reportable=False,
                official_test_for_learning=False,VisA_authorized=False),work/'handoff.json')
            atomic_json(dict(elapsed_seconds=time.monotonic()-started,resources=resources,
                scope='whole invocation; not a paired speedup estimate'),work/'resources.json')
            atomic_json(dict(status=status,job=c['job'],command=c['command'],attempt_id=attempt_id,
                session_hours=hours,elapsed_seconds=time.monotonic()-started,resources=resources,
                error=repr(failure) if failure is not None else None),work/'attempts'/(attempt_id+'.json'))
    receipt=export_run(work,work.parent/(work.name+'.zip'))
    if failure is not None:raise RuntimeError('Run failed; evidence exported: '+str(receipt['path'])) from failure
    return dict(status=status,evidence=receipt)
