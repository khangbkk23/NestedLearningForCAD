"""Export complete, inconclusive and failed validation artifacts before surfacing status."""
import json,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from training.runtime import atomic_json
from training.support_nn_write_v5_1.contract import SCHEMA
from training.support_nn_write_v5_1.runner import run
from notebook_support.common.evidence import export_archive
from .bootstrap import prepare

def run_notebook(c):
    work=Path(c['work']);work.mkdir(parents=True,exist_ok=True);status='validation_failed_preserve_evidence'
    try:
        if c.get('phase')!='validate':raise ValueError('Only paired validation is implemented')
        c['session_utc']=datetime.now(timezone.utc).isoformat();c['prepared']=prepare(c)
        atomic_json(c,work/'session_config.json');status=run(c)
    except BaseException as e:
        traceback.print_exc();atomic_json(dict(error=repr(e)),work/'failure.json')
    finally:
        atomic_json(dict(status=status,reportable=False,cloned_state_adopted=False,migration_implemented=False,VisA_authorized=False),work/'handoff.json')
        atomic_json(dict(elapsed_before_export_seconds=time.monotonic()-c['started']),work/'execution.json')
        t=time.monotonic();receipt=export_archive(work,c['output'],schema=SCHEMA);seconds=time.monotonic()-t
        print('[NN EVIDENCE EXPORTED]',json.dumps(dict(**receipt,export_seconds=seconds)),flush=True)
        try:
            from IPython.display import display,FileLink
            display(FileLink(receipt['path']))
        except ImportError:pass
    if status=='validation_failed_preserve_evidence' or seconds>180:raise RuntimeError('Validation failed/export reserve exceeded; keep ZIP/log: '+receipt['path'])
    print('[NN STATUS]',status,flush=True)
    return receipt
