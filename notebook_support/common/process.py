"""Stream a subprocess and emit periodic heartbeats; terminate owned child on failure."""
from pathlib import Path
import os,signal,queue,subprocess,threading,time
ROOT=Path(__file__).resolve().parents[2]

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
