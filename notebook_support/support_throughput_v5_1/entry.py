"""Always export evidence before exposing failure; never dispatch benchmark jobs."""
import json,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from training.runtime import atomic_json
from notebook_support.common.evidence import export_archive
from training.support_throughput_v5_1.contract import SCHEMA
from training.support_throughput_v5_1.runner import run
from .bootstrap import prepare

def run_notebook(c):
    work=Path(c['work']);work.mkdir(parents=True,exist_ok=True);status='profile_failed_preserve_evidence'
    try:
        if c.get('phase','profile')!='profile':raise ValueError('Only bounded profile is implemented')
        c['session_utc']=datetime.now(timezone.utc).isoformat()
        c['prepared']=prepare(c)
        atomic_json(c,work/'session_config.json')
        status=run(c)
    except BaseException as exc:
        traceback.print_exc();atomic_json(dict(error=repr(exc)),work/'failure.json')
    finally:
        atomic_json(dict(status=status,reportable=False,profiler_only=True,optimization_implemented=False,
            benchmark_continuation=False,VisA_authorized=False),work/'handoff.json')
        atomic_json(dict(elapsed_before_export_seconds=time.monotonic()-c['started']),work/'execution.json')
        t=time.monotonic();receipt=export_archive(work,c['output'],schema=SCHEMA);export_seconds=time.monotonic()-t
        print('[PROFILE EVIDENCE EXPORTED]',json.dumps(dict(**receipt,export_seconds=export_seconds)),flush=True)
        try:
            from IPython.display import display,FileLink
            display(FileLink(receipt['path']))
        except ImportError:pass
    if status=='profile_failed_preserve_evidence' or export_seconds>180:
        raise RuntimeError('Profiler failed or export exceeded charged reserve; keep ZIP/log for review: '+receipt['path'])
    print('[PROFILE STATUS]',status,flush=True)
    return receipt
