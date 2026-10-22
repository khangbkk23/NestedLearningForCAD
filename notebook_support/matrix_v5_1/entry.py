import json,sys,time,traceback
from pathlib import Path
import torch
from training.runtime import atomic_json
from notebook_support.support_v5_1.common import configure_runtime,runtime_identity
from notebook_support.support_throughput_v5_1.bootstrap import one
from notebook_support.common.process import stream_process
from notebook_support.common.evidence import export_archive
from training.support_benchmark_v5_1.engine import boundary,Pause
from .common import ROOT,SCHEMA,read,verify_source,operation,job_name,select_parent,restore,verify_frozen,deadline,stage_tar,sha,complete

def prepare(c):
    job=job_name(c.get('job'))
    if c.get('phase')!='7.2':raise ValueError('Only readonly7.2 is exposed')
    dl=deadline(c['started'],c['session_hours']);op,refs,_=verify_source();configure_runtime();runtime=runtime_identity()
    print('[MATRIX RUNTIME]',json.dumps(runtime),flush=True)
    if runtime!=refs['runtime'] or not torch.cuda.is_available():raise ValueError('Canonical T4 runtime differs; no package upgrades/CPU fallback')
    inputs=Path(c['inputs']);work=Path(c['work']);work.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(c['CHECKPOINT_OVERRIDE']) if c.get('CHECKPOINT_OVERRIDE') else one(inputs.rglob('vit_base_patch8_224.augreg_in21k.safetensors'),'ViT-B/8')
    if sha(checkpoint)!=refs['backbone_sha256']:raise ValueError('Wrong checkpoint bytes')
    _,parent,kind=select_parent(inputs,job,refs,c.get('PARENT_OVERRIDE'))
    print('[MATRIX] verifying and importing own parent',job,str(parent),flush=True)
    count=restore(parent,kind,work,job,refs,op)
    history=work/'matrix_sessions';history.mkdir(exist_ok=True);nums=[int(p.name) for p in history.iterdir() if p.is_dir() and p.name.isdigit()]
    session=history/f'{max(nums,default=0)+1:04d}';session.mkdir()
    atomic_json(dict(parent=str(parent),representation=kind,job=job,verified_payloads=count,operation=operation(op,job),no_training=True,original_input_modified=False),session/'import_receipt.json')
    boundary(dl)
    if c.get('MVTEC_OVERRIDE'):dataset=Path(c['MVTEC_OVERRIDE']);archive=None
    else:
        roots=[p.parent for p in inputs.rglob('bottle') if (p/'test').is_dir()]
        if roots:dataset=one(roots,'MVTec test root');archive=None
        else:
            archive=one([*inputs.rglob('*mvtec*.tar.xz'),*inputs.rglob('*mvtec*.tar.gz')],'MVTec archive');dataset=Path(c['assets']);dataset.mkdir(parents=True,exist_ok=True)
    print('[MATRIX] staging only test/GT; train images will not be loaded',flush=True)
    staged=stage_tar(archive,dataset,evaluation=True) if archive else 0
    atomic_json(dict(dataset=str(dataset),checkpoint=str(checkpoint),staged_test_gt_files=staged,train_features_accessed=False,official_test_for_selection=False),session/'assets_receipt.json')
    return dict(work=str(work),session=str(session),phase='7.2',job=job,dataset=str(dataset),checkpoint=str(checkpoint),runtime=runtime,operation=operation(op,job),deadline=dl)

def run_notebook(c):
    job=job_name(c.get('job'));work=Path(c['work']);work.mkdir(parents=True,exist_ok=True)
    if any(work.iterdir()):raise ValueError('Use a fresh kernel/work directory; resume latest own7.2 evidence')
    status='failed_preserve_evidence';session=None
    try:
        cfg=prepare(c);session=Path(cfg['session']);atomic_json(cfg,session/'worker_config.json')
        code=stream_process([sys.executable,'-u','-B','-m','notebook_support.matrix_v5_1.worker','--config',str(session/'worker_config.json')],session/'worker.log',cwd=ROOT)
        status=read(session/'worker_status.json')['status']
        if code or status not in ('paused_at_atomic_boundary','7.2_complete_pending_review'):raise ValueError('Matrix worker failed/unknown status')
        if status=='7.2_complete_pending_review':
            tasks=read(work/'benchmark/manifest_train.json')['task_order'];complete(work/'benchmark/runs'/job,tasks)
    except Pause as exc:
        status='paused_at_atomic_boundary';atomic_json(dict(reason=str(exc)),work/'matrix_bootstrap_pause.json')
    except BaseException as exc:
        status='failed_preserve_evidence';traceback.print_exc();atomic_json(dict(error=repr(exc)),session/'failure.json' if session else work/'matrix_bootstrap_failure.json')
    finally:
        if (work/'matrix_origin/artifact_manifest.json').exists():
            try:
                _,refs,_=verify_source();count=verify_frozen(work,job,refs)
                atomic_json(dict(immutable_parent_payloads_checked=count,train_repeated=False),session/'preservation.json' if session else work/'matrix_preservation.json')
            except BaseException as exc:
                status='failed_preserve_evidence';traceback.print_exc();atomic_json(dict(error=repr(exc)),work/'matrix_preservation_failure.json')
        handoff=dict(status=status,phase='7.2',job=job,reportable=False,human_review_required=True,VisA_authorized=False,train_repeated=False)
        atomic_json(handoff,work/'handoff.json')
        if session:atomic_json(handoff,session/'handoff.json')
        atomic_json(dict(elapsed_before_export_seconds=time.monotonic()-c['started'],export_reserve_seconds=600),session/'timing.json' if session else work/'matrix_failure_timing.json')
        began=time.monotonic();receipt=export_archive(work,c['output'],schema=SCHEMA);seconds=time.monotonic()-began
        print('[MATRIX EVIDENCE EXPORTED]',json.dumps(dict(**receipt,export_seconds=seconds)),flush=True)
        try:
            from IPython.display import display,FileLink
            display(FileLink(receipt['path']))
        except ImportError:pass
    if status=='failed_preserve_evidence' or seconds>600:raise RuntimeError('Matrix failed/export reserve exceeded; preserve ZIP/log: '+receipt['path'])
    print('[MATRIX HANDOFF]',job,status,flush=True);return receipt
