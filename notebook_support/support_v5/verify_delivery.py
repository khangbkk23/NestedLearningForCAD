"""Independent packaged-bootstrap and test execution; no real-data fit."""
from pathlib import Path
import json,hashlib,zipfile,tempfile,subprocess,sys,os,shutil
from .common import ROOT,R4_SHA,SOURCE_SCHEMA
from models.hope_cad_v5.contract import file_sha,digest

def verify():
    source=ROOT/'ultils/support_v5/hope_cad_support_v5_source_v1.zip'
    with zipfile.ZipFile(source) as z:
        m=json.loads(z.read('bundle_manifest.json'));files=m['files']
        if m['schema']!=SOURCE_SCHEMA or len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=set(files)|{'bundle_manifest.json'}:raise ValueError('v5 bundle inventory/schema invalid')
        for n,h in files.items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h or file_sha(ROOT/n)!=h:raise ValueError('source byte mismatch '+n)
        notebook=json.loads(z.read('notebooks/hope_cad_support_v5_kaggle.ipynb'))
        for c in notebook['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'notebook_cell','exec')
    base=ROOT/'ultils/recovery_stage1/hope_cad_recovery_stage1_source_r4.zip'
    with zipfile.ZipFile(base) as z:
        old=json.loads(z.read('bundle_manifest.json'))['files']
        if file_sha(base)!=R4_SHA or any(files.get(n)!=h for n,h in old.items()):raise ValueError('frozen base changed')
    tests=None
    with tempfile.TemporaryDirectory(prefix='support_v5_delivery_') as tmp:
        tmp=Path(tmp)
        for style in ('zip','directory'):
            inputs=tmp/style;inputs.mkdir();runtime=tmp/(style+'_source')
            if style=='zip':shutil.copyfile(source,inputs/'source.zip')
            else:
                with zipfile.ZipFile(source) as z:z.extractall(inputs/'source')
            code=''.join(notebook['cells'][1]['source']).replace("Path('/kaggle/input')",f'Path({str(inputs)!r})').replace("Path('/tmp/hope_support_v5_source')",f'Path({str(runtime)!r})')
            code+="\nfrom notebook_support.support_v5.common import source_identity\nassert source_identity()=="+repr(digest(files))+"\nprint('V5_BOOTSTRAP_PASS')\n"
            completed=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env={**os.environ,'PYTHONIOENCODING':'utf-8'},capture_output=True,text=True,encoding='utf-8')
            if completed.returncode or 'V5_BOOTSTRAP_PASS' not in completed.stdout:raise RuntimeError(style+' bootstrap failed '+completed.stdout+completed.stderr)
            if style=='directory':
                tested=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests/hope_cad_v5','-q'],cwd=runtime,env={**os.environ,'PYTHONIOENCODING':'utf-8','CUBLAS_WORKSPACE_CONFIG':':4096:8'},capture_output=True,text=True,encoding='utf-8')
                (ROOT/'ultils/support_v5/packaged_tests.log').write_text(tested.stdout+tested.stderr,encoding='utf-8')
                if tested.returncode:raise RuntimeError('packaged tests failed '+tested.stdout+tested.stderr)
                tests='passed; see packaged_tests.log'
    result=dict(status='passed',source_sha256=file_sha(source),source_digest=digest(files),files=len(files),frozen_r4_files_unchanged=len(old),bootstrap_formats=['zip','Kaggle-extracted-directory'],notebook_cells_compiled=True,packaged_test_suite=tests,canonical_T4_real_normal_pilot_pending=True)
    (ROOT/'ultils/support_v5/delivery_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':verify()
