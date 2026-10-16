import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

from scripts.hope_cad_v2.build_w7_notebooks import BOOTSTRAP,notebook
from training.hope_w7_io import export_archive
from training.hope_w7_contract import job_by_id
from scripts.hope_cad_v2.report_w7 import summarize
from dataset.benchmark_manifest_v1 import canonical_digest


class W7HandoffContracts(unittest.TestCase):
    def test_source_bootstrap_accepts_zip_and_extracted_inputs_and_rejects_changed_bytes(self):
        for zipped in (False,True):
            with self.subTest(zipped=zipped),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);inputs=root/'input';inputs.mkdir();src=inputs/'source';src.mkdir()
                raw=b'# harmless source fixture\n';(src/'fixture.py').write_bytes(raw)
                manifest=dict(schema='hope_w7_source_v1',origin={},files={'fixture.py':hashlib.sha256(raw).hexdigest()})
                (src/'bundle_manifest.json').write_text(json.dumps(manifest))
                if zipped:
                    with zipfile.ZipFile(inputs/'hope_cad_w7_source_bundle.zip','w') as z:
                        z.write(src/'fixture.py','fixture.py');z.writestr('bundle_manifest.json',json.dumps(manifest))
                    (src/'fixture.py').unlink();(src/'bundle_manifest.json').unlink();src.rmdir()
                code=BOOTSTRAP.split('# Keep the W6-tested runtime.')[0]
                code=code.replace("INPUT=Path('/kaggle/input')",f'INPUT=Path({inputs.as_posix()!r})')
                code=code.replace("REPO=Path('/tmp/hope_w7_source')",f'REPO=Path({(root/"repo").as_posix()!r})')
                oldcwd=Path.cwd();oldpath=list(sys.path)
                try:
                    exec(compile(code,'w7_source_bootstrap','exec'),{})
                    self.assertEqual((root/'repo/fixture.py').read_bytes(),raw)
                    if not zipped:
                        (src/'fixture.py').write_bytes(b'wrong')
                        with self.assertRaisesRegex(RuntimeError,'checksum'):
                            exec(compile(code,'w7_source_bootstrap','exec'),{})
                finally:
                    os.chdir(oldcwd);sys.path[:]=oldpath
    def test_generated_phase_cells_compile_and_do_not_chain_other_phases(self):
        for phase in ('preflight','7.1','7.2','7.3'):
            nb=notebook(phase,'M_seed1' if phase=='7.3' else 'M_seed0')
            for cell in nb['cells']:
                if cell['cell_type']=='code':compile(cell['source'],f'w7_{phase}','exec')
            self.assertIn("--phase",nb['cells'][2]['source'])
            self.assertNotIn('run_normal_v4.py',str(nb))
    def test_partial_suite_inventory_does_not_claim_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);e=root/'evidence';(e/'runs/M_seed0').mkdir(parents=True)
            identity={'fixture':True};suite=dict(identity=identity,digest=canonical_digest(identity))
            (e/'suite_lock.json').write_text(json.dumps(suite))
            completion=dict(status='complete',job=job_by_id('M_seed0'),suite_digest=suite['digest'],matrix_complete=False)
            (e/'runs/M_seed0/completion.json').write_text(json.dumps(completion))
            export_archive(e,root/'evidence.zip')
            result=summarize([root/'evidence.zip'],root/'report',inventory_only=True)
            self.assertEqual(result['present_jobs'],1)
            self.assertEqual(len(result['missing']),34)
            self.assertEqual(result['incomplete_matrices'],['M_seed0'])
            self.assertFalse(result['reportable'])


if __name__=='__main__':unittest.main()
