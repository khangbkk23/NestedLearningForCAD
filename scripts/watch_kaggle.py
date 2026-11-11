"""Follow Kaggle live logs; disconnecting this viewer does not stop the remote job."""
import argparse,os,shutil,subprocess,sys,time
from pathlib import Path

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
        subprocess.run([cli,'kernels','logs',a.kernel,'--follow'])
        status=subprocess.run([cli,'kernels','status',a.kernel],capture_output=True,text=True)
        text=status.stdout+' '+status.stderr
        if status.returncode==0 and 'KernelWorkerStatus.COMPLETE' in text:
            print('\n'+'='*72+'\n  [DONE] KAGGLE RUN COMPLETED\n  Codex can now download and audit the outputs.\n'+'='*72,flush=True);return 0
        if status.returncode==0 and any('KernelWorkerStatus.'+x in text for x in ('ERROR','CANCELLED','FAILED','CANCELED','CANCEL_ACKNOWLEDGED')):
            print('\n'+'='*72+'\n  [STOPPED] KAGGLE RUN FAILED OR CANCELLED\n  Preserve outputs; report to Codex.\n'+'='*72,flush=True);return 1
        active=any('KernelWorkerStatus.'+x in text for x in ('RUNNING','QUEUED','PENDING','CANCEL_REQUESTED','NEW_SCRIPT'))
        unknown=0 if status.returncode==0 and active else unknown+1
        if unknown>3:raise RuntimeError('Viewer disconnected; remote completion NOT confirmed. Check Kaggle page.')
        print('Log stream disconnected; reconnecting in5s. Remote job was not stopped.',flush=True);time.sleep(5)

if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:print('Viewer disconnected. Kaggle job keeps running.')
