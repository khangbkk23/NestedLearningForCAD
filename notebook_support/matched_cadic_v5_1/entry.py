import json,os,sys,time,traceback
from pathlib import Path
import torch
from training.runtime import atomic_json
from training.benchmark_config_v1 import load_configs
from dataset.benchmark_protocol_v2 import build_prefix_manifest
from notebook_support.support_v5_1.common import configure_runtime,runtime_identity
from notebook_support.support_throughput_v5_1.bootstrap import one
from notebook_support.common.process import stream_process
from notebook_support.common.evidence import export_archive
from training.support_benchmark_v5_1.contract import lock
from training.support_benchmark_v5_1.engine import save_or_verify
from .common import ROOT,SCHEMA,read,verify_source,identity,check_manifest,deadline,stage_tar,select_resume,restore,sha

def prepare(c):
    if c.get('phase')!='7.1' or c.get('job')!='CADIC_seed0':raise ValueError('Only7.1/CADIC_seed0 is exposed')
    dl=deadline(c['started'],c['session_hours']);op,ref,expected=verify_source();configure_runtime();runtime=runtime_identity()
    suite,ji=identity(op,ref,runtime)
    if not torch.cuda.is_available():raise ValueError('Enable canonical GPU T4; no package upgrades or CPU fallback')
    print('[MATCHED CADIC RUNTIME]',json.dumps(runtime),flush=True)
    inputs=Path(c['inputs']);work=Path(c['work']);work.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(c['CHECKPOINT_OVERRIDE']) if c.get('CHECKPOINT_OVERRIDE') else one(inputs.rglob('vit_base_patch8_224.augreg_in21k.safetensors'),'ViT-B/8 checkpoint')
    if sha(checkpoint)!=ref['backbone_sha256']:raise ValueError('Wrong backbone bytes')
    parent=select_resume(inputs,c.get('RESUME_OVERRIDE'))
    print('[MATCHED CADIC] verifying resume' if parent else '[MATCHED CADIC] fresh bank, no learned-state input',flush=True)
    restored=restore(parent,work,op)
    save_or_verify(work/'operation_lock.json',op)
    history=work/'session_history';history.mkdir(exist_ok=True)
    nums=[int(p.name) for p in history.iterdir() if p.is_dir() and p.name.isdigit()]
    session=history/f'{max(nums,default=0)+1:04d}';session.mkdir()
    atomic_json(dict(resume=parent is not None,parent=str(parent) if parent else None,verified_payloads=restored,operation=op,learned_state_imported_from_other_job=False),session/'import_receipt.json')
    if c.get('MVTEC_OVERRIDE'):dataset=Path(c['MVTEC_OVERRIDE']);archive=None
    else:
        roots=[p.parent for p in inputs.rglob('bottle') if (p/'train/good').is_dir()]
        if roots:dataset=one(roots,'MVTec root');archive=None
        else:
            archive=one([*inputs.rglob('*mvtec*.tar.xz'),*inputs.rglob('*mvtec*.tar.gz')],'MVTec archive')
            dataset=Path(c['assets']);dataset.mkdir(parents=True,exist_ok=True)
    if time.monotonic()>=dl:raise ValueError('Bootstrap exhausted soft budget before staging; no learning dispatched')
    print('[MATCHED CADIC] staging registered train bytes; verifying full prefix manifest',flush=True)
    staged=stage_tar(archive,dataset,normal_rows=expected['entries']) if archive else 0
    os.environ.update(MVTEC_ROOT=str(dataset),VIT_B8_CHECKPOINT=str(checkpoint),VIT_B8_SHA256=ref['backbone_sha256'])
    p,_=load_configs(ROOT/'conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml',ROOT/'conf/benchmarks/methods/cadic_matched_prefix_v2.yaml')
    actual=build_prefix_manifest(p,seed=0,source_commit=expected['source_commit']);check_manifest(actual,expected)
    atomic_json(dict(dataset=str(dataset),checkpoint=str(checkpoint),staged_train_files=staged,online_images=3599,excluded_images=30,feature_access_before_arrival=False,official_test_staged=False,normal_rows_digest=ref['normal_rows_digest']),session/'assets_receipt.json')
    return dict(work=str(work),session=str(session),phase=c['phase'],job=c['job'],dataset=str(dataset),dataset_archive=str(archive) if archive else None,checkpoint=str(checkpoint),runtime=runtime,operation=op,deadline=dl)

def run_notebook(c):
    work=Path(c['work']);work.mkdir(parents=True,exist_ok=True)
    if any(work.iterdir()):raise ValueError('Use a new kernel/work directory; resume latest CADIC evidence')
    status='failed_preserve_evidence';session=None
    try:
        cfg=prepare(c);session=Path(cfg['session']);atomic_json(cfg,session/'worker_config.json')
        print('[MATCHED CADIC] worker starting; integrated discarded-normal preflight then fresh benchmark',flush=True)
        code=stream_process([sys.executable,'-u','-B','-m','notebook_support.matched_cadic_v5_1.worker','--config',str(session/'worker_config.json')],session/'worker.log',cwd=ROOT)
        status=read(session/'worker_status.json')['status']
        if code or status not in ('paused_at_atomic_boundary','7.1_complete_pending_review'):raise ValueError('Worker failed or returned unknown status')
        if status=='7.1_complete_pending_review':
            b=work/'benchmark/runs/CADIC_seed0';t=read(b/'train_complete.json');f=read(b/'final_complete.json');jd=read(b/'job_lock.json')['digest']
            if t['job_digest']!=jd or f['job_digest']!=jd or t['images']!=3599 or not t['finalized'] or f['cell_count']!=15:raise ValueError('Incomplete final benchmark')
    except BaseException as exc:
        status='failed_preserve_evidence';traceback.print_exc()
        atomic_json(dict(error=repr(exc)),session/'failure.json' if session else work/'bootstrap_failure.json')
    finally:
        handoff=dict(status=status,phase='7.1',job='CADIC_seed0',reportable=False,human_review_required=True,VisA_authorized=False)
        atomic_json(handoff,work/'handoff.json')
        if session:atomic_json(handoff,session/'handoff.json')
        atomic_json(dict(elapsed_before_export_seconds=time.monotonic()-c['started'],export_reserve_seconds=600),session/'timing.json' if session else work/'failure_timing.json')
        began=time.monotonic();receipt=export_archive(work,c['output'],schema=SCHEMA);seconds=time.monotonic()-began
        print('[MATCHED CADIC EVIDENCE EXPORTED]',json.dumps(dict(**receipt,export_seconds=seconds)),flush=True)
        try:
            from IPython.display import display,FileLink
            display(FileLink(receipt['path']))
        except ImportError:pass
    if status=='failed_preserve_evidence' or seconds>600:raise RuntimeError('CADIC failed/export reserve exceeded; keep ZIP/log: '+receipt['path'])
    print('[MATCHED CADIC HANDOFF]',status,flush=True);return receipt
