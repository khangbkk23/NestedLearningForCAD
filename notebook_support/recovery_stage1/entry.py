"""Notebook execution with streamed logs, periodic heartbeat, and failure export in finally."""
from pathlib import Path
import json
import os
import signal
import queue
import subprocess
import sys
import threading
import time
import traceback
from .common import atomic_json,export,ROOT
from .bootstrap import prepare


def stream_process(command,logfile,*,cwd=ROOT,heartbeat_seconds=30):
    logpath=Path(logfile);logpath.parent.mkdir(parents=True,exist_ok=True)
    process=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,encoding='utf8',errors='replace',bufsize=1,start_new_session=(os.name=='posix'))
    q=queue.Queue()
    def reader():
        try:
            for line in process.stdout:q.put(line)
        finally:q.put(None)
    thread=threading.Thread(target=reader,daemon=True);thread.start();started=time.monotonic()
    try:
        with logpath.open('w',encoding='utf8') as log:
            while True:
                try:line=q.get(timeout=heartbeat_seconds)
                except queue.Empty:
                    line=f'[RUNNING] runner elapsed {time.monotonic()-started:.0f}s; waiting for next atomic operation/log line\n'
                if line is None:break
                print(line,end='',flush=True);log.write(line);log.flush()
        return process.wait()
    finally:
        if process.poll() is None:
            if os.name=='posix':os.killpg(process.pid,signal.SIGTERM)
            else:process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
                else:process.kill()
                process.wait()
        thread.join(timeout=2)


def run_notebook(config):
    """All bootstrap/runner failures export evidence BEFORE the notebook raises."""
    work=Path(config['work']);w6=Path(config['w6']);phase=config['phase']
    work.mkdir(parents=True,exist_ok=True);code=1;result=None
    try:
        prepare(config['inputs'],work,w6,config['assets'],phase=phase,
            w6_override=config.get('W6_OVERRIDE'),w72_override=config.get('W72_OVERRIDE'),
            resume_override=config.get('RESUME_OVERRIDE'),checkpoint_override=config.get('CHECKPOINT_OVERRIDE'),
            mvtec_override=config.get('MVTEC_OVERRIDE'))
        command=[sys.executable,'-B','-m','notebook_support.recovery_stage1.runner',
            '--phase',phase,'--work-dir',str(work),'--w6-dir',str(w6),'--device',config.get('device','cuda'),
            '--session-hours',str(config['session_hours']),'--session-start-monotonic',str(config['started'])]
        if phase=='controls':command+=['--jobs',*config['jobs']]
        print('[RECOVERY COMMAND]',' '.join(command),flush=True)
        code=stream_process(command,work/'logs'/f'{phase}_{int(time.time())}.log')
        if code and (not (work/'handoff.json').exists() or json.loads((work/'handoff.json').read_text()).get('status')!='failed_preserve_evidence'):
            atomic_json(dict(status='failed_preserve_evidence',runner_exit_code=code,
                reason='runner exited without a fresh successful handoff; preserve checkpoints',reportable=False),work/'handoff.json')
    except BaseException as error:
        traceback.print_exc()
        atomic_json(dict(status='failed_preserve_evidence',phase=phase,error=repr(error),
            location='notebook bootstrap/launch',reportable=False),work/'handoff.json')
    finally:
        atomic_json(dict(phase=phase,runner_exit_code=code,elapsed_seconds=time.monotonic()-config['started'],
            explicit_exports_are_not_guaranteed_on_hard_platform_kill=True),work/'notebook_execution.json')
        result=export(work,Path(config['output']))
        print('[EVIDENCE EXPORTED]',json.dumps(result),flush=True)
        try:
            from IPython.display import display,FileLink
            display(FileLink(result['path']))
        except ImportError:pass
    handoff=json.loads((work/'handoff.json').read_text()) if (work/'handoff.json').exists() else {}
    accepted={'paused_at_atomic_boundary','recovery_preflight_passed'} if phase=='preflight' else {
        'paused_at_atomic_boundary','recovery_stage1_complete_pending_review','requested_jobs_complete_remaining_pending'}
    if code or handoff.get('status') not in accepted:
        raise RuntimeError(f"Recovery did NOT pass: exit={code}, status={handoff.get('status')}. ZIP exported at {result['path']}; keep it for review.")
    print('[VERIFIED STATUS]',handoff['status'],flush=True)
    if handoff['status']=='paused_at_atomic_boundary':print('Resume LATEST recovery output; this is not completion.',flush=True)
    return result