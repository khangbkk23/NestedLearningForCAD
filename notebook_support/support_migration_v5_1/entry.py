"""Always export failure/incomplete evidence; never auto-run the remaining benchmark."""
from pathlib import Path
import json,sys,time,traceback
from training.runtime import atomic_json
from notebook_support.common.process import stream_process
from notebook_support.common.evidence import export_archive
from training.support_migration_v5_1.contract import contract,SCHEMA,ROOT
from .bootstrap import prepare

def run_notebook(c):
    work=Path(c['work']);work.mkdir(parents=True,exist_ok=True);status='bridge_failed_preserve_parent'
    try:
        if c.get('phase')!='bridge':raise ValueError('Only bridge is exposed; review output before full benchmark')
        cfg=contract();prepared=prepare(c);deadline=c['started']+cfg['budget_seconds']-cfg['export_reserve_seconds']
        config=dict(**prepared,work=str(work),deadline=deadline)
        atomic_json(config,work/'worker_config.json')
        code=stream_process([sys.executable,'-B','-m','training.support_migration_v5_1.worker','--config',str(work/'worker_config.json')],work/'logs/bridge.log',cwd=ROOT)
        if code:raise RuntimeError('Bridge worker failed; preserve evidence')
        status=json.loads((work/'worker_status.json').read_text())['status']
        if status not in ('bridge_complete_pending_review','bridge_incomplete_preserve_parent'):raise ValueError('Unknown bridge status')
    except BaseException as exc:
        traceback.print_exc();atomic_json(dict(error=repr(exc)),work/'entry_failure.json')
    finally:
        atomic_json(dict(status=status,reportable=False,human_review_required=True,full_benchmark_started=False,VisA_authorized=False),work/'handoff.json')
        atomic_json(dict(elapsed_before_export_seconds=time.monotonic()-c['started'],export_reserve_seconds=180),work/'execution.json')
        began=time.monotonic();receipt=export_archive(work,c['output'],schema=SCHEMA);seconds=time.monotonic()-began
        print('[BRIDGE EVIDENCE EXPORTED]',json.dumps(dict(**receipt,export_seconds=seconds)),flush=True)
        try:
            from IPython.display import display,FileLink
            display(FileLink(receipt['path']))
        except ImportError:pass
    if status=='bridge_failed_preserve_parent' or seconds>180:raise RuntimeError('Bridge failed/export reserve exceeded; keep ZIP/log: '+receipt['path'])
    print('[BRIDGE HANDOFF]',status,flush=True);return receipt
