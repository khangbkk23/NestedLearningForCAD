"""Follow Kaggle logs; distinguish provider completion from workload admission."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path


def terminal_payload(line):
    for marker in ('B324_TERMINAL ', 'B323_TERMINAL ', 'B322_TERMINAL ', 'B321_TERMINAL ','B320_TERMINAL ', 'B319_TERMINAL '):
        if marker in line:
            try:
                value=json.loads(line.split(marker,1)[1].strip())
            except (ValueError,TypeError):
                return None
            if isinstance(value,dict) and isinstance(value.get('status'),str):return value
    return None


def completion_notice(payload):
    status=(payload or {}).get('status','UNKNOWN')
    reason=(payload or {}).get('worker_status',status)
    if status=='BLOCKED_RESOURCE_PREFLIGHT' or status.startswith('PREFLIGHT_INSUFFICIENT_'):
        return ('[BLOCKED] RESOURCE PREFLIGHT - FULL TRAINING NOT STARTED',
                'Reason: '+reason+'\nPreserve outputs; resource review is required before a new attempt.',2)
    if status in ('FAILED','ERROR','CANCELLED','CANCELED'):
        return ('[FAILED] WORKLOAD REPORTED FAILURE','Preserve logs and outputs for the audit.',1)
    if status=='PAUSED_RESUMABLE':
        return ('[PAUSED] CHECKPOINT SAVED - RESUME REQUIRED','Session ended normally; experiment is NOT complete. Verify/export checkpoints before continuation.',0)
    if status=='COMPLETE_REVIEW_PENDING':
        return ('[ENDED] WORKLOAD COMPLETED - AUDIT PENDING',
                'Execution ended; results are NOT confirmed until the outputs are audited.',0)
    return ('[ENDED] KAGGLE EXECUTION ENDED - AUDIT REQUIRED',
            'Training completion is NOT verified. Inspect the workload receipt.',0)


def follow_logs(command):
    # Stream without retaining the full log or downloading tensor/score outputs.
    payload=None
    proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,encoding='utf-8',errors='replace')
    try:
        for line in proc.stdout:
            print(line,end='',flush=True)
            found=terminal_payload(line)
            if found is not None:payload=found
        proc.wait()
        return payload
    finally:
        # Only the local CLI viewer is terminated, never a remote Kaggle job.
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        proc.stdout.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--kernel',required=True);p.add_argument('--token-file',type=Path);a=p.parse_args()
    if a.token_file:os.environ['KAGGLE_API_TOKEN']=str(a.token_file.resolve(strict=True))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();user=api.config_values.get(api.CONFIG_NAME_USER,'')
    if user.casefold()!=a.kernel.split('/')[0].casefold():raise RuntimeError('Kaggle account does not own this run')
    cli=shutil.which('kaggle') or str(Path(sys.prefix)/'Scripts/kaggle.exe')
    print('Live logs:',a.kernel,'| Ctrl+C disconnects only this viewer',flush=True)
    unknown=0
    while True:
        # A new stream may follow a new version; never reuse a prior stream's verdict.
        payload=follow_logs([cli,'kernels','logs',a.kernel,'--follow'])
        status=subprocess.run([cli,'kernels','status',a.kernel],capture_output=True,text=True,encoding='utf-8',errors='replace')
        text=status.stdout+' '+status.stderr
        if status.returncode==0 and 'KernelWorkerStatus.COMPLETE' in text:
            title,message,code=completion_notice(payload)
            print('\n'+'='*72+'\n  '+title+'\n  '+a.kernel+'\n  '+message.replace('\n','\n  ')+'\n'+'='*72,flush=True);return code
        if status.returncode==0 and any('KernelWorkerStatus.'+x in text for x in ('ERROR','CANCELLED','FAILED','CANCELED','CANCEL_ACKNOWLEDGED')):
            print('\n'+'='*72+'\n  [STOPPED] KAGGLE RUN FAILED OR CANCELLED\n  Preserve outputs for the audit.\n'+'='*72,flush=True);return 1
        active=any('KernelWorkerStatus.'+x in text for x in ('RUNNING','QUEUED','PENDING','CANCEL_REQUESTED','NEW_SCRIPT'))
        unknown=0 if status.returncode==0 and active else unknown+1
        if unknown>3:raise RuntimeError('Viewer disconnected; remote completion NOT confirmed. Check Kaggle page.')
        print('Log stream disconnected; reconnecting in5s. Remote job was not stopped.',flush=True);time.sleep(5)


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:print('Viewer disconnected. Kaggle job keeps running.')
