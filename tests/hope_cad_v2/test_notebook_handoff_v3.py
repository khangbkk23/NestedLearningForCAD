"""Exercise the actual notebook bootstrap for zipped and Kaggle-extracted inputs."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]


class NotebookHandoffContracts(unittest.TestCase):
    def test_bootstrap_accepts_zip_and_extracted_source_and_rejects_hash_drift(self):
        notebook = json.loads((ROOT/'notebooks/hope_cad_v2_2_kaggle.ipynb').read_text(encoding='utf-8'))
        source = ''.join(notebook['cells'][1]['source'])
        files = {'scripts/hope_cad_v2/run_w1_w6.py': b'# bootstrap fixture\n'}
        manifest = dict(files={name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
                        origin={'fixture': True})
        for zipped in (True, False):
            with self.subTest(zipped=zipped), tempfile.TemporaryDirectory() as root:
                root=Path(root)
                inputs=root/'inputs'; inputs.mkdir()
                if zipped:
                    with zipfile.ZipFile(inputs/'hope_cad_v2_2_source_bundle.zip','w') as archive:
                        archive.writestr('bundle_manifest.json',json.dumps(manifest))
                        for name,data in files.items(): archive.writestr(name,data)
                else:
                    (inputs/'bundle_manifest.json').write_text(json.dumps(manifest))
                    for name,data in files.items():
                        target=inputs/name; target.parent.mkdir(parents=True); target.write_bytes(data)
                code=source.replace("Path('/kaggle/input')",f'Path({str(inputs)!r})')
                code=code.replace("Path('/tmp/hope_cad_source_v3')",f'Path({str(root/"repo")!r})')
                original_cwd, original_path = Path.cwd(), list(sys.path)
                try:
                    namespace={}
                    exec(compile(code,'notebook_source_bootstrap','exec'),namespace)
                    self.assertEqual(namespace['SESSION_HOURS'],10.5)
                    started=namespace['NOTEBOOK_STARTED_MONOTONIC']
                    self.assertEqual((root/'repo'/next(iter(files))).read_bytes(),next(iter(files.values())))
                    # A second run may reuse precisely the same source, never a modified one.
                    exec(compile(code,'notebook_source_bootstrap','exec'),namespace)
                    self.assertEqual(namespace['NOTEBOOK_STARTED_MONOTONIC'],started)
                    (root/'repo'/next(iter(files))).write_bytes(b'changed')
                    with self.assertRaisesRegex(RuntimeError,'Working source changed'):
                        exec(compile(code,'notebook_source_bootstrap','exec'),{})
                    with self.assertRaisesRegex(RuntimeError,'Unsafe archive path'):
                        namespace['safe_relative'](root/'repo','../escape.py')
                finally:
                    os.chdir(original_cwd)
                    sys.path[:]=original_path

    def test_every_delivered_code_cell_compiles_and_has_a_unique_id(self):
        for name in ('hope_cad_v2_2_kaggle.ipynb','hope_cad_v2_2_preflight_kaggle.ipynb'):
            notebook=json.loads((ROOT/'notebooks'/name).read_text(encoding='utf-8'))
            self.assertEqual(len({c['id'] for c in notebook['cells']}),len(notebook['cells']))
            for index,cell in enumerate(notebook['cells']):
                if cell['cell_type']=='code': compile(''.join(cell['source']),f'{name}_cell_{index}','exec')
            command_code=''.join(notebook['cells'][5]['source'])
            self.assertIn("'--session-start-monotonic',str(NOTEBOOK_STARTED_MONOTONIC)",command_code)

    def test_planned_pause_is_normal_completion_but_never_scientific_acceptance(self):
        from scripts.hope_cad_v2.run_w1_w6 import notebook_outcome
        self.assertIn('W6 is incomplete',notebook_outcome(0,{'status':'paused_for_session_limit'}))
        self.assertIn('PREFLIGHT PASSED',notebook_outcome(0,
            {'status':'preflight_passed','engineering_gates_passed':True},preflight_only=True))
        for code,summary in ((1,{'status':'paused_for_session_limit'}),(1,{'status':'stopped'}),
                             (0,{'status':'preflight_passed','engineering_gates_passed':True}),
                             (0,{'status':'ready_for_artifact_review','engineering_gates_passed':False})):
            with self.assertRaises(RuntimeError): notebook_outcome(code,summary)


if __name__=='__main__': unittest.main()
