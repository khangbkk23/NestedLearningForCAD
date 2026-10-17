"""Execute source-bootstrap cells against ZIP and Kaggle-extracted fixtures, including adversarial paths."""
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
import os
import sys
import tempfile
import unittest
import zipfile
from notebook_support.recovery_stage1.build_delivery import notebook
from notebook_support.recovery_stage1.common import SOURCE_SCHEMA

ROOT=Path(__file__).resolve().parents[2]

class NotebookTests(unittest.TestCase):
    def setUp(self):
        base=ROOT/'results/recovery_stage1/tmp';base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.root=Path(self.temp.name)
        self.inputs=self.root/'inputs';self.inputs.mkdir()
    def tearDown(self):self.temp.cleanup()
    def source(self):
        s=notebook('preflight')['cells'][1]['source']
        for old,new in [("Path('/kaggle/input')",repr(self.inputs.as_posix())),
            ("Path('/tmp/hope_recovery_stage1_source')",repr((self.root/'repo').as_posix())),
            ("Path('/kaggle/working/hope_recovery_stage1')",repr((self.root/'work').as_posix()))]:
            s=s.replace(old,'Path('+new+')')
        return s
    def run_bootstrap(self):
        cwd=Path.cwd();paths=list(sys.path)
        try:
            with redirect_stdout(StringIO()):exec(compile(self.source(),'fixture_bootstrap','exec'),{})
        finally:os.chdir(cwd);sys.path[:]=paths
    def manifest(self):
        return dict(schema=SOURCE_SCHEMA,origin={'fixture':True,'delivery_revision':'lifecycle-r4'},files={'fixture.py':hashlib.sha256(b'pass\n').hexdigest()})
    def test_extracted_source_directory_supported(self):
        p=self.inputs/'source';p.mkdir();(p/'fixture.py').write_bytes(b'pass\n')
        (p/'bundle_manifest.json').write_text(json.dumps(self.manifest()))
        self.run_bootstrap();self.assertEqual((self.root/'repo/fixture.py').read_bytes(),b'pass\n')
    def test_zip_source_supported_and_cell_compile(self):
        with zipfile.ZipFile(self.inputs/'hope_cad_recovery_stage1_source.zip','w') as z:
            z.writestr('fixture.py',b'pass\n');z.writestr('bundle_manifest.json',json.dumps(self.manifest()))
        self.run_bootstrap();self.assertTrue((self.root/'repo/fixture.py').exists())
        for phase in ('preflight','controls'):
            for cell in notebook(phase)['cells']:
                if cell['cell_type']=='code':compile(cell['source'],'handoff','exec')
    def test_source_hash_mismatch_fails_before_runner(self):
        p=self.inputs/'source';p.mkdir();(p/'fixture.py').write_bytes(b'corrupt')
        (p/'bundle_manifest.json').write_text(json.dumps(self.manifest()))
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):self.run_bootstrap()
    def test_source_traversal_rejected_before_write(self):
        m=self.manifest();m['files']={'../escape.py':hashlib.sha256(b'pass\n').hexdigest()}
        with zipfile.ZipFile(self.inputs/'hope_cad_recovery_stage1_source.zip','w') as z:
            z.writestr('../escape.py',b'pass\n');z.writestr('bundle_manifest.json',json.dumps(m))
        with self.assertRaisesRegex(ValueError,'Unsafe'):self.run_bootstrap()
        self.assertFalse((self.root/'escape.py').exists())
    def test_old_timeout_source_revision_is_rejected(self):
        p=self.inputs/'source';p.mkdir();m=self.manifest();m['origin'].pop('delivery_revision')
        (p/'bundle_manifest.json').write_text(json.dumps(m));(p/'fixture.py').write_bytes(b'pass\n')
        with self.assertRaisesRegex(ValueError,'NEW r4 source'):self.run_bootstrap()
    def test_different_source_versions_not_silently_selected(self):
        for name in ('a','b'):
            p=self.inputs/name;p.mkdir();m=self.manifest();m['origin']['version']=name
            (p/'bundle_manifest.json').write_text(json.dumps(m));(p/'fixture.py').write_bytes(b'pass\n')
        with self.assertRaisesRegex(ValueError,'Multiple different'):self.run_bootstrap()

if __name__=='__main__':unittest.main()