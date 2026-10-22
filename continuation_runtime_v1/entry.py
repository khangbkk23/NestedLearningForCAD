"""Fail closed before training; always export failure/pause/completion evidence."""
import json,time,traceback,sys
from pathlib import Path
from training.runtime import atomic_json
from notebook_support.common.process import stream_process
from notebook_support.common.evidence import export_archive
from notebook_support.support_throughput_v5_1.bootstrap import one
from notebook_support.support_v5_1.common import configure_runtime,runtime_identity
from training.support_migration_v5_1.lineage import Execution
from training.support_migration_v5_1.contract import contract
from training.support_throughput_v5_1.contract import contract as profile_contract,ROOT,sha
from .common import SCHEMA,EXECUTION,RESERVE,read,verify_delivery,verify_publication,select_parent,restore_parent,deadline,stage_tar

def prepare(c):
    import torch
    if c.get('phase')!='7.1' or c.get('job')!='NESTED_seed0':raise ValueError('This delivery exposes only7.1/NESTED_seed0')
    dl=deadline(c['started'],c['session_hours']);source,op=verify_delivery();configure_runtime();runtime=runtime_identity()
    if not torch.cuda.is_available():raise ValueError('Enable canonical GPU T4; no CPU fallback')
    print('[CONTINUATION RUNTIME]',json.dumps(runtime),flush=True)
    inputs=Path(c['inputs']);work=Path(c['work']);work.mkdir(parents=True,exist_ok=True)
    mh,parent,kind=select_parent(inputs,c.get('PARENT_OVERRIDE'))
    from training.support_throughput_v5_1.contract import Evidence
    with Evidence(parent) as ev:
        expected=ev.json('benchmark/execution_lock.json')['identity']['runtime']
        if runtime!=expected:raise ValueError('Runtime differs from reviewed G4/T4; do not upgrade packages')
    checkpoint=Path(c['CHECKPOINT_OVERRIDE']) if c.get('CHECKPOINT_OVERRIDE') else one(inputs.rglob('vit_base_patch8_224.augreg_in21k.safetensors'),'ViT-B/8')
    if sha(checkpoint)!=profile_contract()['frozen_reference']['backbone_sha256']:raise ValueError('Wrong backbone bytes')
    print('[CONTINUATION] verifying/restoring all parent payloads; no learning yet',flush=True)
    restored=restore_parent(parent,kind,work,op)
    verify_publication(work)
    e=Execution(work/'benchmark',source,runtime,contract(),profile_contract()['parent'])
    if e.identities['execution']['digest']!=EXECUTION:raise ValueError('Unexpected G4 execution owner')
    active=__import__('torch').load(e.folder/'active.pt',map_location='cpu',weights_only=True)
    if not 384<=active['cursor']<=3599 or active['job_digest']!=e.identities['job']['digest']:raise ValueError('Wrong active owner/cursor')
    # Frozen train validates active receipt, entire event prefix and session state on actual device.
    history=work/'session_history';history.mkdir(exist_ok=True)
    nums=[int(p.name) for p in history.iterdir() if p.is_dir() and p.name.isdigit()]
    session=history/f'{max(nums,default=0)+1:04d}';session.mkdir()
    atomic_json(dict(parent_manifest_sha256=mh,representation='directory' if parent.is_dir() else 'ZIP',verified_payloads=restored,restored_cursor=active['cursor'],orchestration=op,core_source=source),session/'import_receipt.json')
    if c.get('MVTEC_OVERRIDE'):dataset=Path(c['MVTEC_OVERRIDE']);archive=None
    else:
        roots=[p.parent for p in inputs.rglob('bottle') if (p/'train/good').is_dir()]
        if roots:dataset=one(roots,'MVTec root');archive=None
        else:
            archive=one([*inputs.rglob('*mvtec*.tar.xz'),*inputs.rglob('*mvtec*.tar.gz')],'MVTec archive')
            dataset=Path(c['assets']);dataset.mkdir(parents=True,exist_ok=True)
    if time.monotonic()>=dl:raise ValueError('Bootstrap consumed session budget; no train dispatched')
    print('[CONTINUATION] restored cursor',active['cursor'],'/3599; staging permitted train bytes',flush=True)
    staged=stage_tar(archive,dataset,normal_rows=e.rows) if archive else 0
    atomic_json(dict(dataset=str(dataset),archive=str(archive) if archive else None,staged_normal_files=staged,staging_is_feature_access=False,future_features_to_learner=False,test_staged=False),session/'assets_receipt.json')
    return dict(work=str(work),session=str(session),deadline=dl,dataset=str(dataset),dataset_archive=str(archive) if archive else None,checkpoint=str(checkpoint),source=source,orchestration=op,runtime=runtime)

def run_notebook(c):
    work=Path(c['work']);work.mkdir(parents=True,exist_ok=True)
    status='failed_preserve_evidence';session=None;code=1
    # Refuse dirty local destination without overwriting/exporting somebody else's handoff.
    if any(work.iterdir()):raise ValueError('Use a new kernel/work directory; resume by attaching latest output')
    try:
        prepared=prepare(c);session=Path(prepared['session'])
        atomic_json(prepared,session/'worker_config.json')
        print('[CONTINUATION] worker starting; heartbeat30s; no bridge rerun',flush=True)
        code=stream_process([sys.executable,'-u','-B','-m','continuation_runtime_v1.worker','--config',str(session/'worker_config.json')],session/'worker.log',cwd=ROOT)
        status=read(session/'worker_status.json')['status']
        if code or status not in ('paused_at_atomic_boundary','7.1_complete_pending_review'):raise ValueError('Worker failed or incomplete status')
        if status=='7.1_complete_pending_review':
            final=read(work/'benchmark/runs/NESTED_seed0/final_complete.json')
            if final['cell_count']!=15 or not (work/'benchmark/runs/NESTED_seed0/train_complete.json').exists():raise ValueError('Missing completed training/final15')
    except BaseException as exc:
        status='failed_preserve_evidence';traceback.print_exc()
        failure=session/'failure.json' if session else work/'bootstrap_failure.json'
        atomic_json(dict(error=repr(exc)),failure)
    finally:
        handoff=dict(status=status,phase='7.1',job='NESTED_seed0',reportable=False,human_review_required=True,VisA_authorized=False)
        atomic_json(handoff,work/'handoff.json')
        if session:atomic_json(handoff,session/'handoff.json')
        atomic_json(dict(elapsed_before_export_seconds=time.monotonic()-c['started'],export_reserve_seconds=RESERVE),session/'timing.json' if session else work/'failure_timing.json')
        began=time.monotonic();receipt=export_archive(work,c['output'],schema=SCHEMA);seconds=time.monotonic()-began
        print('[CONTINUATION EVIDENCE EXPORTED]',json.dumps(dict(**receipt,export_seconds=seconds)),flush=True)
        try:
            from IPython.display import display,FileLink
            display(FileLink(receipt['path']))
        except ImportError:pass
    if status=='failed_preserve_evidence' or seconds>RESERVE:raise RuntimeError('Continuation failed/export reserve exceeded; keep ZIP/log: '+receipt['path'])
    print('[CONTINUATION HANDOFF]',status,flush=True);return receipt
