"""Verify hashes, source bootstrap forms and packaged benchmark tests, without GPU work."""
import ast,json,os,shutil,subprocess,sys,tempfile,zipfile
from pathlib import Path
from models.hope_cad_v5_1.contract import ROOT,file_sha,digest
from training.support_benchmark_v5_1.contract import source_files,SOURCE_SCHEMA
from .build_delivery import delivery_directory

def verify():
    out=delivery_directory();source=out/'hope_cad_support_v5_1_benchmark_source_r1.zip';files=source_files()
    with zipfile.ZipFile(source) as z:
        manifest=json.loads(z.read('bundle_manifest.json'))
        if manifest['schema']!=SOURCE_SCHEMA or manifest['files']!=files or len(set(z.namelist()))!=len(z.namelist()) or set(z.namelist())!=set(files)|{'bundle_manifest.json'}:raise ValueError('Source inventory mismatch')
        import hashlib
        for n,h in files.items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('Source checksum failed')
        notebooks=[json.loads(z.read(n)) for n in files if n.startswith('notebooks/hope_cad_v5_1_benchmark_')]
        for nb in notebooks:
            for cell in nb['cells']:
                if cell['cell_type']=='code':compile(''.join(cell['source']),'notebook','exec')
    with tempfile.TemporaryDirectory(prefix='benchmark_delivery_') as tmp:
        tmp=Path(tmp)
        for style in ('zip','directory'):
            inputs=tmp/style;inputs.mkdir();runtime=tmp/(style+'_runtime')
            if style=='zip':shutil.copyfile(source,inputs/'source.zip')
            else:
                with zipfile.ZipFile(source) as z:z.extractall(inputs/'source')
            code=''.join(notebooks[0]['cells'][1]['source']).replace("Path('/kaggle/input')",f'Path({str(inputs)!r})').replace("Path('/tmp/hope_support_v5_1_benchmark_source')",f'Path({str(runtime)!r})')
            code+='\nfrom training.support_benchmark_v5_1.contract import source_digest\nassert source_digest()=='+repr(digest(files))+"\nprint('BOOTSTRAP_PASS')\n"
            done=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',env={**os.environ,'PYTHONIOENCODING':'utf-8'})
            if done.returncode or 'BOOTSTRAP_PASS' not in done.stdout:raise RuntimeError(style+' bootstrap: '+done.stdout+done.stderr)
            if style=='directory':
                tests=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests/support_benchmark_v5_1','-q'],cwd=runtime,capture_output=True,text=True,encoding='utf-8',env={**os.environ,'PYTHONIOENCODING':'utf-8'})
                (out/'packaged_tests.log').write_text(tests.stdout+tests.stderr,encoding='utf-8')
                if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
    result=dict(status='passed',source_sha256=file_sha(source),files=len(files),bootstrap_formats=['zip','Kaggle-extracted-directory'],notebooks_compiled=len(notebooks),packaged_tests='passed',canonical_T4_benchmark='not_run',scientific_superiority=False)
    (out/'delivery_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result
if __name__=='__main__':verify()
