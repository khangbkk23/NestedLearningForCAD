"""Contracts for artifact placement and portable Kaggle packaging; no ML run."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nested_replay_cad import paths, provenance
spec=importlib.util.spec_from_file_location('kaggle_run',ROOT/'scripts/kaggle_run.py')
kaggle_run=importlib.util.module_from_spec(spec);spec.loader.exec_module(kaggle_run)

class LocalLayout(unittest.TestCase):
    def setUp(self):
        scratch=paths.cache()/'tests';scratch.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        identity=patch.object(provenance,'source_identity',return_value={'git':{'commit':'fixture','dirty':True},'design_commit':'fixture','design_hashes':{}})
        identity.start();self.addCleanup(identity.stop)
        self.local=Path(self.temp.name)/'artifacts'

    def test_read_only_lookup(self):
        self.assertEqual(paths.dataset(self.local),self.local/'inputs/mvtec')
        self.assertEqual(paths.weights(self.local).parent,self.local/'inputs/weights')
        self.assertFalse(self.local.exists())

    def test_environment_override(self):
        with patch.dict(os.environ,{'NRCAD_LOCAL_ROOT':str(self.local)}):
            self.assertEqual(paths.run_root('b319-test'),self.local/'runs/b319/b319-test')
        self.assertEqual(paths.local_root('.local'),ROOT/'.local')

    def test_reject_unsafe_root(self):
        for root in (ROOT,Path(ROOT.anchor)):
            with self.subTest(root=root),self.assertRaises(ValueError):paths.local_root(root)

    def test_reject_unsafe_id(self):
        for name in ('../run','a/b','a\\b','..','/run','a:b','x'*129):
            with self.subTest(name=name),self.assertRaises(ValueError):paths.run_root(name,self.local)

    def test_environment_writers(self):
        with patch.dict(os.environ,clear=False):
            paths.process_environment(self.local)
            self.assertEqual(Path(os.environ['TMP']),self.local/'tmp')
            self.assertEqual(paths.local_root(),self.local)
            self.assertEqual(Path(os.environ['TORCH_HOME']),self.local/'cache/torch')
            self.assertEqual(Path(os.environ['TORCHINDUCTOR_CACHE_DIR']),self.local/'cache/torch/inductor')
            self.assertEqual(Path(os.environ['HF_HOME']),self.local/'cache/huggingface')
        self.assertEqual({p.name for p in self.local.iterdir()},{'tmp'})

    def test_provider_legacy_layout(self):
        run=self.local/'runs/b319/test'
        self.assertEqual(kaggle_run.destination(run,'v5','B319/summary.json'),run/'outputs/v5/B319/summary.json')
        self.assertEqual(kaggle_run.destination(run,'v5','setup/train.log'),run/'logs/v5/setup/train.log')
        self.assertEqual(kaggle_run.destination(run,'v5','RUN_STATUS.json'),run/'receipts/v5/RUN_STATUS.json')

    def test_provider_new_layout(self):
        run=self.local/'runs/b319/test'
        self.assertEqual(kaggle_run.destination(run,'v6','runs/b319/test/outputs/summary.json'),run/'outputs/v6/summary.json')
        self.assertEqual(kaggle_run.destination(run,'v6','runs/b319/test/receipts/RUN_STATUS.json'),run/'receipts/v6/RUN_STATUS.json')

    def test_provider_rejects_traversal(self):
        for name in ('../secret','/absolute','runs/b319/test/../../bad','C:/secret','a\\b','.git/config'):
            with self.subTest(name=name),self.assertRaises(ValueError):kaggle_run.destination(self.local,'v5',name)

    def test_package_is_self_contained(self):
        args=type('Args',(),{'run_id':'b319-pack-test','owner':'test-owner'})()
        with patch.dict(os.environ,{'NRCAD_LOCAL_ROOT':str(self.local)}),contextlib.redirect_stdout(io.StringIO()):
            run=kaggle_run.pack(args)
            with self.assertRaises(FileExistsError):kaggle_run.pack(args)
        record=json.loads((run/'run.json').read_text())
        source=run/'package/dataset/source.zip'
        self.assertEqual(kaggle_run.digest(source),record['package_sha256'])
        with zipfile.ZipFile(source) as archive:
            self.assertIsNone(archive.testzip())
            manifest=json.loads(archive.read('BUNDLE.json'))
            self.assertEqual(record['files'],manifest)
            for name,sha in manifest.items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(),sha)
                self.assertNotIn('.local',Path(name).parts)
                self.assertNotIn('.pixi',Path(name).parts)
        notebook=json.loads((run/'package/kernel/b319.ipynb').read_text())
        code=''.join(notebook['cells'][0]['source']);compile(code,'notebook','exec')
        self.assertIn(record['package_sha256'],code)
        self.assertIn(record['manifest_sha256'],code)
        self.assertIn("RUN_ID='b319-pack-test'",code)
        self.assertNotIn('__RUN_ID__',code)
        self.assertIn("WORK/'outputs'",code)
        self.assertIn("RECEIPTS/'gpu_preflight.json'",code)
        kernel=json.loads((run/'package/kernel/kernel-metadata.json').read_text())
        self.assertTrue(kernel['is_private'])
        self.assertEqual(kernel['dataset_sources'],[record['dataset']])

class TrainingLayout(unittest.TestCase):
    setUp=LocalLayout.setUp
    def test_three_arm_logs_and_final_state(self):
        self._orchestrate(0)

    def test_failure_stays_in_its_run(self):
        self._orchestrate(3)

    def _orchestrate(self,exit_code):
        spec=importlib.util.spec_from_file_location('train_layout',ROOT/'scripts/train.py')
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        cfg=json.loads((ROOT/'configs/b319.json').read_text())
        cfg.update(dataset_path=str(self.local/'inputs/mvtec'),encoder={'path':str(self.local/'inputs/weights/w')})
        calls=[]
        class Process:
            def __init__(self,command,**kwargs):
                calls.append(command);self.stdout=iter(['event log\n'])
            def wait(self):return exit_code
        with patch.dict(os.environ,{'NRCAD_LOCAL_ROOT':str(self.local)}),patch.object(cli,'load_inputs',return_value=(cfg,{}, {},ROOT/'configs/b319.json')),patch.object(cli.platform,'system',return_value='Linux'),patch.object(cli.subprocess,'Popen',Process),patch.object(cli.subprocess,'run') as summary,patch.object(sys,'argv',['train.py','--run-id','orchestration-test']),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(),exit_code)
        run=self.local/'runs/b319/orchestration-test'
        record=json.loads((run/'run.json').read_text())
        expected_arms=cfg['arms'] if exit_code==0 else cfg['arms'][:1]
        self.assertEqual([c[c.index('--arm')+1] for c in calls],expected_arms)
        for arm in expected_arms:
            self.assertEqual((run/'logs'/f'{arm}.log').read_text(),'event log\n')
        self.assertEqual(record['state'],'complete_review_pending' if exit_code==0 else 'failed')
        self.assertEqual(summary.call_count,1 if exit_code==0 else 0)
        self.assertEqual({p.name for p in self.local.iterdir()},{'runs','tmp'})

class ReportingContract(unittest.TestCase):
    setUp=LocalLayout.setUp
    def test_original_signed_forgetting_excludes_final_and_last_task(self):
        spec=importlib.util.spec_from_file_location('summarize_layout',ROOT/'scripts/summarize.py')
        summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)
        for arm in ('er_uniform','a0_tail','a0_tail_ctx'):
            snapshots=[]
            for stage in range(15):
                metrics={str(i):{'pixel':{'AP':.5},'image':{'AUROC':.75}} for i in range(stage+1)}
                metrics['0']['pixel']['AP']=.8 if stage==0 else .9 if stage==1 else .6
                # Category 1 improves at final: signed forgetting must remain negative.
                if stage>=1:metrics['1']['pixel']['AP']=.7 if stage==14 else .5
                snapshots.append({'metrics':metrics})
            folder=self.local/arm/'arms'/arm;folder.mkdir(parents=True)
            (folder/'study.json').write_text(json.dumps({'status':'COMPLETE_REVIEW_PENDING','committed':3629,'snapshots':snapshots}))
        result=summary.summarize(self.local)
        for row in result['arms'].values():
            self.assertAlmostEqual(row['macro_Fmax'],(.3-.2)/14)
            self.assertAlmostEqual(row['macro_Facq'],(.2-.2)/14)
            self.assertIsNone(row['per_category']['14']['Fmax'])
            self.assertAlmostEqual(row['legacy_macro_final_inclusive_Fmax_15'],.3/15)

class AssetPlacement(unittest.TestCase):
    setUp=LocalLayout.setUp
    def _download(self,valid):
        spec=importlib.util.spec_from_file_location('assets_layout',ROOT/'scripts/prepare_assets.py')
        assets=importlib.util.module_from_spec(spec);spec.loader.exec_module(assets)
        data=b'verified test bytes';expected=hashlib.sha256(data if valid else b'other bytes').hexdigest()
        target=paths.weights(self.local)
        with patch.dict(os.environ,{'NRCAD_LOCAL_ROOT':str(self.local)}),patch.object(assets.urllib.request,'urlopen',return_value=io.BytesIO(data)):
            if valid:assets.fetch('https://fixture.invalid/weight',target,expected,len(data))
            else:
                with self.assertRaises(ValueError):assets.fetch('https://fixture.invalid/weight',target,expected,len(data))
        return target,list(paths.temporary(self.local).glob('*.part'))

    def test_verified_download_enters_inputs(self):
        target,partials=self._download(True)
        self.assertTrue(target.is_file());self.assertEqual(partials,[])

    def test_failed_download_stays_in_tmp(self):
        target,partials=self._download(False)
        self.assertFalse(target.exists());self.assertEqual(len(partials),1)

if __name__=='__main__':unittest.main()
