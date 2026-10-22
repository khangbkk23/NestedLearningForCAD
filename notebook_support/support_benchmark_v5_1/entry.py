import json,sys,time,traceback
from pathlib import Path
from .bootstrap import prepare
from training.support_benchmark_v5_1.contract import PHASES,EVIDENCE_SCHEMA,ROOT,job
from training.runtime import atomic_json
from notebook_support.common.evidence import export_archive
from notebook_support.common.process import stream_process

def run_notebook(c):
    work=Path(c['work']);work.mkdir(parents=True,exist_ok=True);code=1
    try:
        if c['phase'] not in PHASES or not .1<=c['session_hours']<=11:raise ValueError('Invalid phase/session soft budget')
        job(c['job']);c['prepared']=prepare(c);atomic_json(c,work/'execution_config.json')
        code=stream_process([sys.executable,'-B','-m','notebook_support.support_benchmark_v5_1.runner','--config',str(work/'execution_config.json')],work/'logs'/f'run_{int(time.time())}.log',cwd=ROOT)
        if code:raise RuntimeError('Benchmark worker failed')
    except BaseException as e:traceback.print_exc();atomic_json(dict(status='failed_preserve_evidence',job=c.get('job'),error=repr(e),reportable=False),work/'handoff.json')
    finally:
        atomic_json(dict(runner_exit_code=code,elapsed_seconds=time.monotonic()-c['started']),work/'execution.json')
        receipt=export_archive(work,c['output'],schema=EVIDENCE_SCHEMA);print('[BENCH EVIDENCE EXPORTED]',json.dumps(receipt),flush=True)
        try:
            from IPython.display import display,FileLink
            display(FileLink(receipt['path']))
        except ImportError:pass
    status=json.loads((work/'handoff.json').read_text())['status']
    if code or status not in ('preflight_passed','paused_at_atomic_boundary','7.1_complete_pending_review','7.2_complete_pending_review','7.3_complete_pending_review'):raise RuntimeError('Benchmark failed; keep ZIP/log '+receipt['path'])
    print('[BENCH STATUS]',status,flush=True);return receipt
