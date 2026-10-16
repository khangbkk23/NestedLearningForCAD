"""Transport-only regressions kept outside the historical experiment source map."""
from pathlib import Path
from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from notebook_support.w7_resume_fix import launcher
from notebook_support.w7_resume_fix.build_delivery import patched_notebook,W6_SELECTOR
from training import hope_w7_contract as contract
from dataset.benchmark_manifest_v1 import canonical_digest
from models.hope_cad_v2.predictor import HOPEFeaturePredictorV4
from models.hope_cad_v2.visual_block import VisualBlockConfig

ROOT=Path(__file__).resolve().parents[2]
FROZEN_LOCK=contract.lock_json


class JSONLockFixContracts(unittest.TestCase):
    def model_identity(self):
        model=HOPEFeaturePredictorV4(VisualBlockConfig(8,8,8,4,4,4),beta=.01)
        model.requires_grad_(False)
        return dict(checkpoint='a'*64,runtime={'torch':'fixture'},manifest='b'*64,
                    effective_method={'predictor':model.metadata()})

    def test_original_bug_reproduces_with_real_predictor_metadata_and_fix_preserves_lock_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'lock.json';identity=self.model_identity()
            old=FROZEN_LOCK(p,identity);before=p.read_bytes()
            with self.assertRaisesRegex(ValueError,'locked identity changed'):FROZEN_LOCK(p,identity)
            with patch.object(contract,'lock_json',FROZEN_LOCK):
                fixed=launcher.install_json_lock_fix();restored=fixed(p,identity)
            self.assertEqual(restored,json.loads(before))
            self.assertEqual(restored['digest'],old['digest'])
            self.assertEqual(p.read_bytes(),before)

    def test_changed_beta_geometry_checkpoint_manifest_or_runtime_still_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'lock.json';identity=self.model_identity();FROZEN_LOCK(p,identity)
            proposals=[]
            for key in ('checkpoint','manifest','runtime'):
                altered=deepcopy(identity);altered[key]='changed';proposals.append(altered)
            altered=deepcopy(identity);altered['effective_method']['predictor']['dynamics']['rates']=(.001,.001);proposals.append(altered)
            altered=deepcopy(identity);altered['effective_method']['predictor']['visual_config']['width']=5;proposals.append(altered)
            with patch.object(contract,'lock_json',FROZEN_LOCK):
                fixed=launcher.install_json_lock_fix()
                for value in proposals:
                    with self.assertRaisesRegex(ValueError,'locked identity changed'):fixed(p,value)

    def test_corrupt_stored_digest_and_nonfinite_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'lock.json';identity=self.model_identity();FROZEN_LOCK(p,identity)
            stored=json.loads(p.read_text());stored['digest']='0'*64;p.write_text(json.dumps(stored))
            with patch.object(contract,'lock_json',FROZEN_LOCK):
                fixed=launcher.install_json_lock_fix()
                with self.assertRaisesRegex(ValueError,'locked identity changed'):fixed(p,identity)
                with self.assertRaises(ValueError):fixed(p,{'bad':float('nan')})

    def test_serialized_7_1_to_7_2_reuses_final_scores_and_never_trains(self):
        import sys
        from unittest.mock import patch
        sys.path.insert(0,str(ROOT/'tests/hope_cad_v2'))
        from test_w7_engine import FixtureData,FixtureAdapter
        from training.hope_w7_engine import W7Engine
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=FixtureData(root);folder=root/'run'
            identity=self.model_identity();path=folder/'run_lock.json'
            first=FROZEN_LOCK(path,identity)
            engine=W7Engine(FixtureAdapter(),data,folder,first['digest'],checkpoint_every=2)
            engine.train();engine.final()
            before={str(p.relative_to(folder)):p.read_bytes() for p in folder.rglob('*')
                    if p.is_file() and ('states/' in p.as_posix() or '/scores/02_' in p.as_posix() or '/metrics/' in p.as_posix())}
            with patch.object(contract,'lock_json',FROZEN_LOCK):
                second=launcher.install_json_lock_fix()(path,identity)
            self.assertEqual(first['digest'],second['digest'])
            adapter=FixtureAdapter()
            with patch.object(adapter,'fit_task',side_effect=AssertionError('7.2 must not train')):
                W7Engine(adapter,data,folder,second['digest'],checkpoint_every=2).matrix()
            self.assertEqual(adapter.scores,12)  # only 3 missing cells x 4 fixture images
            self.assertEqual(len(list((folder/'scores').glob('*/result.json'))),6)
            for name,raw in before.items():self.assertEqual((folder/name).read_bytes(),raw)

    def test_live_frozen_source_matches_all_119_original_files(self):
        self.assertEqual(len(launcher.verify_frozen_source()),119)

    def test_source_and_reference_drift_are_rejected(self):
        reference=launcher.verify_frozen_source();changed=dict(reference);changed['models/hope_cad_v2/predictor.py']='0'*64
        with patch.object(contract,'source_hashes',return_value=changed):
            with self.assertRaisesRegex(ValueError,'frozen source drift'):launcher.verify_frozen_source()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'frozen_reference.json').write_text('{}')
            with patch.object(launcher,'__file__',str(root/'launcher.py')):
                with self.assertRaisesRegex(ValueError,'reference was changed'):launcher.verify_frozen_source()

    def test_launcher_receipt_preserves_exit_code_and_archives_exact_helper(self):
        for code in (0,1):
            with self.subTest(code=code),tempfile.TemporaryDirectory() as temp:
                with patch.object(contract,'lock_json',FROZEN_LOCK),patch.object(launcher.runpy,'run_path',side_effect=SystemExit(code)):
                    self.assertEqual(launcher.main(['--phase','7.2','--job','M_seed0','--work-dir',temp]),code)
                receipts=list(Path(temp).rglob('7.2_*.json'));self.assertEqual(len(receipts),1)
                receipt=json.loads(receipts[0].read_text());self.assertEqual(receipt['runner_exit_code'],code)
                source=receipts[0].parent/(receipt['launcher_sha256']+'.py')
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),receipt['launcher_sha256'])
                self.assertFalse(receipt['existing_locks_rewritten'])


class NotebookFixContracts(unittest.TestCase):
    def notebooks(self):
        parent=ROOT/'ultils/hope_cad_w7_source_bundle.zip'
        if parent.exists():
            with zipfile.ZipFile(parent) as z:
                return [patched_notebook(json.loads(z.read(n))) for n in z.namelist()
                        if n.startswith('notebooks/hope_cad_w7_') and n.endswith('.ipynb') and 'report_cpu' not in n]
        # The delivered bundle carries patched notebooks, not a nested legacy ZIP.
        return [json.loads(p.read_text(encoding='utf8')) for p in sorted((ROOT/'notebooks').glob('hope_cad_w7_*.ipynb'))
                if 'report_cpu' not in p.name]

    def test_all_four_phase_notebooks_compile_and_use_verified_launcher(self):
        nbs=self.notebooks();self.assertEqual(len(nbs),4)
        for nb in nbs:
            for c in nb['cells']:
                if c['cell_type']=='code':compile(c['source'],'phase_cell','exec')
            self.assertIn('notebook_support/w7_resume_fix/launcher.py',nb['cells'][2]['source'])
            self.assertIn('EXIT_CODE!=0',nb['cells'][3]['source'])
            self.assertIn("PHASE=='7.2'",nb['cells'][1]['source'])

    def test_duplicate_w6_directories_select_only_matching_pinned_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);raw=b'{"fixture":"reviewed"}'
            for name in ('reviewed_a','reviewed_b','stale'):
                folder=root/name;folder.mkdir();(folder/'artifact_manifest.json').write_bytes(raw if name!='stale' else b'{}')
            scope=dict(INPUT=root,W6_OVERRIDE=None,Path=Path,sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest())
            with patch.object(contract,'W6_MANIFEST_SHA256',hashlib.sha256(raw).hexdigest()):exec(W6_SELECTOR,scope)
            self.assertEqual(scope['w6source'],root/'reviewed_a')
            with patch.object(contract,'W6_MANIFEST_SHA256','0'*64):
                with self.assertRaisesRegex(RuntimeError,'No reviewed completed W6'):exec(W6_SELECTOR,scope)

    def test_final_cell_exports_before_failure_and_accepts_pause(self):
        source=[nb for nb in self.notebooks() if "PHASE='7.2'" in nb['cells'][1]['source']][0]['cells'][3]['source']
        for code,status,should_fail in ((1,'failed_preserve_evidence',True),(0,'7.1_complete_pending_review',True),
                                        (0,'7.2_complete_pending_review',False),(0,'paused_at_atomic_boundary',False)):
            with self.subTest(code=code,status=status),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);(root/'handoff.json').write_text(json.dumps({'status':status}))
                output=root/'evidence.zip'
                def export(work,path):
                    output.write_bytes(b'evidence was exported');return dict(path=str(output))
                display=SimpleNamespace(display=lambda x:None,FileLink=lambda p:p)
                scope=dict(WORK=root,Path=Path,json=json,EXIT_CODE=code,PHASE='7.2')
                with patch('training.hope_w7_io.export_archive',side_effect=export),patch.dict('sys.modules',{'IPython.display':display}),redirect_stdout(StringIO()):
                    if should_fail:
                        with self.assertRaisesRegex(RuntimeError,'Evidence ZIP was exported'):exec(source,scope)
                    else:exec(source,scope)
                self.assertTrue(output.exists())


if __name__=='__main__':unittest.main(verbosity=2)
