"""Always export failure/pause evidence; preserve platform hard-kill limitation."""
from pathlib import Path
import json,sys,time,traceback
from .contract import SCHEMA,ROOT,atomic_json
from .bootstrap import prepare
from .experiments import BudgetPause
from training.hope_w7_io import export_archive
from notebook_support.recovery_stage1.entry import stream_process

def run_notebook(config):
    work=Path(config['work']);work.mkdir(parents=True,exist_ok=True);code=1
    try:
        if config['phase'] not in ('run','preflight') or not .1<=float(config['session_hours'])<=11:raise ValueError('phase must be run/preflight, session hours .1..11')
        prepared=prepare(config);config['prepared']=prepared
        atomic_json(config,work/'execution_config.json')
        code=stream_process([sys.executable,'-B','-m','notebook_support.normal_causal.runner','--config',str(work/'execution_config.json')],work/'logs'/f"run_{int(time.time())}.log",cwd=ROOT)
        if code:raise RuntimeError(f'worker exited {code}')
    except BaseException as error:
        traceback.print_exc();atomic_json(dict(status='failed_preserve_evidence',error=repr(error),reportable=False),work/'handoff.json')
    finally:
        atomic_json(dict(elapsed_seconds=time.monotonic()-config['started'],runner_exit_code=code,hard_platform_kill_cannot_guarantee_export=True),work/'execution.json')
        receipt=export_archive(work,config['output'],schema=SCHEMA)
        print('[EVIDENCE EXPORTED]',json.dumps(receipt),flush=True)
        try:
            from IPython.display import display,FileLink
            display(FileLink(receipt['path']))
        except ImportError:pass
    status=json.loads((work/'handoff.json').read_text())['status']
    if code or status not in ('preflight_passed','complete_pending_review','paused_at_atomic_boundary'):raise RuntimeError('Diagnostic failed; keep exported ZIP for review. '+str(receipt))
    print('[STATUS]',status,flush=True);return receipt
