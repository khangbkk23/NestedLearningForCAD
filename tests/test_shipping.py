"""Operational shipping contracts independent of learned-method quality."""
from pathlib import Path
import json,time,tempfile,unittest,zipfile
from unittest.mock import patch
from notebook_support.common.evidence import export_archive,restore_archive,destination
from notebook_support.support_v5.common import source_files,source_identity,SOURCE_SCHEMA
from notebook_support.support_v5.entry import run_notebook
from models.hope_cad_v5.contract import ROOT,design_digest
from notebook_support.support_v5.verify_delivery import functions

class ShippingContracts(unittest.TestCase):
    def test_inventory_is_self_contained_and_excludes_retired_workflows(self):
        files=source_files();self.assertIn('conf/delivery/support_source.json',files)
        self.assertFalse(any(p.startswith(('legacy/','exps/','ultils/','results/','notebook_support/recovery_stage1/')) for p in files))
        self.assertNotIn('bundle_manifest.json',files);self.assertEqual(len(source_identity()),64)
    def test_numerical_definitions_and_frozen_bytes_match_independent_v5_baseline(self):
        from models.hope_cad_v5.contract import file_sha
        inv=json.loads((ROOT/'conf/provenance/algorithm_invariance.json').read_text())
        for p,expected in inv['definition_fingerprints'].items():self.assertEqual(functions(ROOT/p),expected,p)
        for p,expected in inv['byte_exact_files'].items():self.assertEqual(file_sha(ROOT/p),expected,p)
        self.assertEqual(design_digest(),'40a51b1e58ebd3444d447c3fafa7692ee7e47aaf14a8df55aa4a766e256528e8')
    def test_export_roundtrip_and_tampering_are_explicit(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);(t/'payload').mkdir();(t/'payload/state.json').write_text('{"cursor":13}')
            export_archive(t/'payload',t/'e.zip',schema='fixture');restore_archive(t/'e.zip',t/'restored',schema='fixture')
            self.assertEqual((t/'restored/state.json').read_text(),'{"cursor":13}')
            (t/'restored/state.json').write_text('changed')
            with self.assertRaises(ValueError):restore_archive(t/'e.zip',t/'restored',schema='fixture')
    def test_matched_cadic_prepare_does_not_depend_on_retired_files_or_test_data(self):
        import os,subprocess,sys,hashlib
        from PIL import Image
        from dataset.benchmark_protocol_v1 import CANONICAL_MVTEC_TASKS
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);data=t/'data'
            for task in CANONICAL_MVTEC_TASKS:
                folder=data/task/'train/good';folder.mkdir(parents=True)
                for i in range(60 if task=='bottle' else 1):Image.new('RGB',(8,8),(i,0,0)).save(folder/f'{i:03d}.png')
            checkpoint=t/'fixture.safetensors';checkpoint.write_bytes(b'structural smoke only, not real weights')
            env={**os.environ,'MVTEC_ROOT':str(data),'VIT_B8_CHECKPOINT':str(checkpoint),'VIT_B8_SHA256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            result=subprocess.run([sys.executable,'-B','-m','scripts.hope_cad_v2.run_matched_cadic','--phase','prepare','--run-dir',str(t/'run'),'--smoke','--max-tasks','1','--max-train-images','1'],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertTrue((t/'run/baseline_lock.json').exists());self.assertFalse((data/'bottle/test').exists())
    def test_archive_provenance_wins_over_an_unrelated_parent_git_repository(self):
        from scripts.hope_cad_v2 import artifacts
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);manifest=dict(files=source_files(),origin=dict(source_commit='recorded_source',branch='recorded_branch',created_utc='packing-time'))
            (t/'bundle_manifest.json').write_text(json.dumps(manifest))
            with patch.object(artifacts,'ROOT',t),patch.object(artifacts.subprocess,'check_output') as git:
                value=artifacts.provenance()
            git.assert_not_called();self.assertEqual(value['source_commit'],'recorded_source');self.assertEqual(value['execution_checkout'],'verified_source_archive')
    def test_repeated_build_preserves_an_existing_delivery(self):
        from notebook_support.support_v5 import build_delivery
        with tempfile.TemporaryDirectory() as t:
            with patch.object(build_delivery,'delivery_directory',return_value=Path(t)):
                first=build_delivery.build();second=build_delivery.build()
            self.assertEqual(first['sha256'],second['sha256'])
    def test_archive_paths_reject_traversal_and_windows_absolute(self):
        with tempfile.TemporaryDirectory() as t:
            for name in ('../escape','/absolute','C:/escape','nested\\escape','x/../y'):
                with self.assertRaises(ValueError):destination(t,name)
    def test_bootstrap_failure_preserves_a_valid_failure_zip(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);cfg=dict(work=str(t/'work'),phase='pilot',session_hours=3.,started=time.monotonic(),output=str(t/'failure.zip'))
            with patch('notebook_support.support_v5.entry.prepare',side_effect=ValueError('fixture missing input')):
                with self.assertRaises(RuntimeError):run_notebook(cfg)
            with zipfile.ZipFile(t/'failure.zip') as z:
                m=json.loads(z.read('artifact_manifest.json'));h=json.loads(z.read('handoff.json'))
                self.assertEqual(h['status'],'failed_preserve_evidence');self.assertIn('handoff.json',m['files'])
if __name__=='__main__':unittest.main()
