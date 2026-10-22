"""Verify shipping bytes, algorithm invariants, both bootstrap formats and packaged tests."""
from pathlib import Path
import ast,json,hashlib,zipfile,tempfile,subprocess,sys,os,shutil
from .common import ROOT,SOURCE_SCHEMA,source_files,delivery_directory
from models.hope_cad_v5_1.contract import file_sha,digest

def functions(path):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    return {n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}

def verify():
    out=delivery_directory();source=out/'hope_cad_support_v5_1_source_r1.zip'
    with zipfile.ZipFile(source) as z:
        m=json.loads(z.read('bundle_manifest.json'));files=m['files']
        if m['schema']!=SOURCE_SCHEMA or len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=set(files)|{'bundle_manifest.json'}:raise ValueError('source inventory invalid')
        if files!=source_files():raise ValueError('live source differs from delivery')
        for n,h in files.items():
            with z.open(n) as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=h:raise ValueError('source checksum '+n)
        notebook=json.loads(z.read('notebooks/hope_cad_support_v5_1_kaggle.ipynb'))
        for c in notebook['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'notebook_cell','exec')
    invariant=json.loads((ROOT/'conf/provenance/algorithm_invariance.json').read_text())
    for rel,expected in invariant['definition_fingerprints'].items():
        if functions(ROOT/rel)!=expected:raise ValueError('algorithm definition drift '+rel)
    for rel,expected in invariant['byte_exact_files'].items():
        if file_sha(ROOT/rel)!=expected:raise ValueError('frozen numerical dependency changed '+rel)
    with tempfile.TemporaryDirectory(prefix='support_v6_delivery_') as folder:
        folder=Path(folder)
        for style in ('zip','directory'):
            inputs=folder/style;inputs.mkdir();runtime=folder/(style+'_source')
            if style=='zip':shutil.copyfile(source,inputs/'source.zip')
            else:
                with zipfile.ZipFile(source) as z:z.extractall(inputs/'source')
            code=''.join(notebook['cells'][1]['source']).replace("Path('/kaggle/input')",f'Path({str(inputs)!r})').replace("Path('/tmp/hope_support_v5_1_source')",f'Path({str(runtime)!r})')
            code+="\nfrom notebook_support.support_v5_1.common import source_identity\nassert source_identity()=="+repr(digest(files))+"\nprint('BOOTSTRAP_PASS')\n"
            completed=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env={**os.environ,'PYTHONIOENCODING':'utf-8'},capture_output=True,text=True,encoding='utf-8')
            if completed.returncode or 'BOOTSTRAP_PASS' not in completed.stdout:raise RuntimeError(style+' bootstrap '+completed.stdout+completed.stderr)
            if style=='directory':
                tested=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests/hope_cad_v5_1','-q'],cwd=runtime,env={**os.environ,'PYTHONIOENCODING':'utf-8','CUBLAS_WORKSPACE_CONFIG':':4096:8'},capture_output=True,text=True,encoding='utf-8')
                (out/'packaged_tests.log').write_text(tested.stdout+tested.stderr,encoding='utf-8')
                if tested.returncode:raise RuntimeError('packaged tests '+tested.stdout+tested.stderr)
    result=dict(status='passed',source_sha256=file_sha(source),source_digest=digest(files),files=len(files),algorithm_definitions_unchanged=True,byte_exact_dependencies=len(invariant['byte_exact_files']),bootstrap_formats=['zip','Kaggle-extracted-directory'],notebook_cells_compiled=True,packaged_v5_1_test_suite='passed; see packaged_tests.log',canonical_T4_real_normal_pilot_pending=True)
    (out/'delivery_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2));return result
if __name__=='__main__':verify()
