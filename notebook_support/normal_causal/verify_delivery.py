"""Read-only parent compatibility and independent byte verification for handoff."""
import argparse,hashlib,json,tempfile,zipfile,io,subprocess,sys,os
from pathlib import Path
import torch
from .contract import ROOT,sha,digest,RECOVERY_MANIFEST,R4_ZIP_SHA,PROTOCOL
from .bootstrap import verified_import
from .runner import parent_model

def verify(source,recovery,output):
    from training.hope_runtime_v4 import configure_runtime
    configure_runtime()
    with zipfile.ZipFile(source) as z:
        m=json.loads(z.read('bundle_manifest.json'))
        if len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=set(m['files'])|{'bundle_manifest.json'}:raise ValueError('bundle inventory invalid')
        for n,h in m['files'].items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('bundle hash mismatch '+n)
        notebook=json.loads(z.read('notebooks/hope_cad_normal_causal_kaggle.ipynb'))
        for c in notebook['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'notebook_cell','exec')
    with zipfile.ZipFile(ROOT/'ultils/recovery_stage1/hope_cad_recovery_stage1_source_r4.zip') as z:
        old=json.loads(z.read('bundle_manifest.json'))
        if any(m['files'].get(n)!=h for n,h in old['files'].items()):raise ValueError('frozen r4 dependency changed')
    checks=[]
    with tempfile.TemporaryDirectory(prefix='normal_causal_delivery_') as tmp:
        tmp=Path(tmp);required={f'reference_w7/runs/M_seed0/states/task_{t:02d}.pt' for t in PROTOCOL['checkpoint_indices']}
        verified_import(recovery,tmp/'parent',RECOVERY_MANIFEST,required,'hope_recovery_stage1_evidence_v1')
        for t in PROTOCOL['checkpoint_indices']:
            model,state,cursor=parent_model(tmp/f'parent/reference_w7/runs/M_seed0/states/task_{t:02d}.pt','cpu')
            checks.append(dict(task=t,clock=cursor,pending=list(state.pending),commits=list(state.commits),metadata=model.metadata()))
        # Execute the actual cell-1 bootstrap in independent processes for ZIP and unpacked inputs.
        for style in ('zip','directory'):
            inputs=tmp/style;inputs.mkdir()
            if style=='zip':
                import shutil
                shutil.copyfile(source,inputs/'source.zip')
            else:
                with zipfile.ZipFile(source) as z:z.extractall(inputs/'source')
            root=tmp/(style+'_runtime')
            code=''.join(notebook['cells'][1]['source']).replace("Path('/kaggle/input')",f'Path({str(inputs)!r})').replace("Path('/tmp/hope_normal_causal_source_v1')",f'Path({str(root)!r})')
            code+="\nfrom notebook_support.normal_causal.contract import source_identity\nassert source_identity()=="+repr(digest(m['files']))+"\nprint('BOOTSTRAP_PARITY_PASS')\n"
            done=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',env={**os.environ,'PYTHONIOENCODING':'utf-8'})
            if done.returncode:raise RuntimeError(style+' bootstrap failed '+done.stdout+done.stderr)
            if 'BOOTSTRAP_PARITY_PASS' not in done.stdout:raise RuntimeError('missing bootstrap marker')
    result=dict(status='passed',source_sha256=sha(source),source_digest=digest(m['files']),files=len(m['files']),frozen_r4_unchanged=len(old['files']),actual_parent_checkpoint_compatibility=checks,packaged_bootstrap_styles=['zip','kaggle_extracted_directory'],notebook_cells_compiled=True,canonical_gpu_profile_pending=True,scientific_result_pending=True)
    Path(output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--recovery',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();verify(a.source,a.recovery,a.output)
