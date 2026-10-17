"""Notebook launch with heartbeat, failure/pause export and truthful handoff."""
from pathlib import Path
import json,sys,time,traceback
from .bootstrap import prepare
from .common import SCHEMA,ROOT
from training.hope_runtime_v4 import atomic_json
from training.hope_w7_io import export_archive
from notebook_support.recovery_stage1.entry import stream_process

def run_notebook(config):
    work=Path(config['work']);work.mkdir(parents=True,exist_ok=True);code=1
    try:
        if config['phase'] not in ('pilot','preflight') or not .1<=config['session_hours']<=11:raise ValueError('phase pilot/preflight; soft budget .1..11 hours')
        config['prepared']=prepare(config);atomic_json(config,work/'execution_config.json')
        code=stream_process([sys.executable,'-B','-m','notebook_support.support_v5.runner','--config',str(work/'execution_config.json')],work/'logs'/f'run_{int(time.time())}.log',cwd=ROOT)
        if code:raise RuntimeError('v5 worker failed, preserve checkpoint/evidence')
    except BaseException as e:
        traceback.print_exc();atomic_json(dict(status='failed_preserve_evidence',error=repr(e),reportable=False),work/'handoff.json')
    finally:
        atomic_json(dict(runner_exit_code=code,elapsed_seconds=time.monotonic()-config['started'],hard_platform_kill_can_prevent_export=True),work/'execution.json')
        receipt=export_archive(work,config['output'],schema=SCHEMA);print('[V5 EVIDENCE EXPORTED]',json.dumps(receipt),flush=True)
        try:
            from IPython.display import display,FileLink
            display(FileLink(receipt['path']))
        except ImportError:pass
    status=json.loads((work/'handoff.json').read_text())['status']
    if code or status not in ('preflight_passed','paused_at_image_boundary','pilot_complete_pending_review','pilot_complete_scientific_gate_failed'):raise RuntimeError('v5 failed; keep exported ZIP: '+receipt['path'])
    print('[V5 STATUS]',status,flush=True)
    if status=='pilot_complete_scientific_gate_failed':print('Finite pilot completed, scientific gate did NOT pass. Stop for review; do not retune or launch anomaly benchmark.',flush=True)
    return receipt
