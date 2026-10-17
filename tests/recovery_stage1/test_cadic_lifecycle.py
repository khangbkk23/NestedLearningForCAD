"""Complete registry/worker/control lifecycles. Only ViT/data identities are synthetic fixtures."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
import sys
import unittest
from unittest.mock import patch
import torch
from PIL import Image
import test_registry_cache as registry_fixture
import test_operations as operations_fixture
from notebook_support.recovery_stage1.cadic_accel import accelerate
from notebook_support.recovery_stage1.cadic_profile import bounded_probe
from notebook_support.recovery_stage1 import runner
from training.hope_w7_adapter import CompactCADIC
from training.benchmark_engine_v1 import _state_equal

ROOT=Path(__file__).resolve().parents[2]

def fixture_worker(config_path):
    from notebook_support.recovery_stage1 import cadic_profile
    torch.set_num_threads(2)
    d=json.loads(Path(config_path).read_text())
    argv=['fixture','--w6-dir',d['root'],'--output',d['output'],'--device','cpu']
    with patch('models.cadic_paperfaithful.cadic_benchmark_adapter.CADICViTFeatureExtractor',registry_fixture.FixtureExtractor),\
         patch.object(cadic_profile,'reviewed_context',return_value=d['context']),\
         patch.object(cadic_profile,'baseline_config',return_value=d['config']),patch.object(sys,'argv',argv):
        return cadic_profile.main()

class CADICLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def setUp(self):
        self.fixture=registry_fixture.RegistryCacheTests();self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown);self.root=self.fixture.root
    def test_close_delegates_to_extractor_once_without_mutating_bank(self):
        compact=CompactCADIC(accelerate(self.fixture.build()),'fixture')
        with redirect_stdout(StringIO()):compact.fit_task(0,'normal',[self.fixture.batch])
        before=deepcopy(compact.state_dict())
        with patch.object(registry_fixture.FixtureExtractor,'close',autospec=True) as close:
            compact.close();compact.close()
            self.assertEqual(close.call_count,1)
        self.assertTrue(_state_equal(before,compact.state_dict()))
    def test_cleanup_failure_is_not_silently_accepted_and_can_retry(self):
        compact=CompactCADIC(accelerate(self.fixture.build()),'fixture')
        with patch.object(registry_fixture.FixtureExtractor,'close',side_effect=RuntimeError('cleanup fixture')):
            with self.assertRaisesRegex(RuntimeError,'cleanup fixture'):compact.close()
        with patch.object(registry_fixture.FixtureExtractor,'close',autospec=True) as close:
            compact.close();compact.close();self.assertEqual(close.call_count,1)
    def test_real_registered_control_trains_evaluates_exports_completion_and_closes(self):
        data=operations_fixture.PathsFixture(self.root)
        compact=CompactCADIC(accelerate(self.fixture.build()),'fixture')
        args=SimpleNamespace(work_dir=self.root/'work',w6_dir=self.root/'w6',device='cpu')
        protocol={'fixture':True};manifest={'digest':'fixture-manifest','online':[]}
        with patch.object(runner,'job_protocol',return_value=(protocol,manifest)),\
             patch.object(runner,'make_adapter',return_value=(compact,data,{'fixture':True},self.fixture.config,protocol,manifest)),\
             patch.object(registry_fixture.FixtureExtractor,'close',autospec=True) as close,redirect_stdout(StringIO()):
            runner.run_control('CADIC10000_seed0',{}, {'digest':'fixture-suite'}, {'digest':'fixture-study'},args,None)
            self.assertEqual(close.call_count,1)
        folder=args.work_dir/'runs/CADIC10000_seed0'
        record=json.loads((folder/'recovery_completion.json').read_text())
        self.assertTrue(record['final_complete']);self.assertFalse(record['matrix_complete'])
        self.assertTrue(json.loads((folder/'lifecycle_completion.json').read_text())['cleanup_complete'])
        self.assertEqual(len(list((folder/'states').glob('task_*.pt'))),3)
        self.assertEqual(compact.coreset.update_batches,9)
        self.assertEqual(len(list((folder/'scores').glob('*/result.json'))),3)
    def test_full_resource_worker_subprocess_returns_zero_and_preserves_passed_receipt(self):
        rows=[]
        for i in range(16):
            p=self.root/'fixture/train/good'/f'{i:03d}.png';p.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(16,16),(20+i,30+i,40+i)).save(p)
            rows.append(dict(relative_path=p.relative_to(self.root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),split='train',task_id=0))
        config=deepcopy(self.fixture.config);config['memory']['budget']=10000
        context={'roles':{'roles':{'meta_support':rows}},'protocol':{'dataset':{'root':str(self.root)}},
            'method':{'preprocessing':{'image_size':224,'mean':[0,0,0],'std':[1,1,1]}}}
        output=self.root/'resource.json';setup=self.root/'worker_config.json'
        setup.write_text(json.dumps(dict(root=str(self.root),output=str(output),context=context,config=config)))
        code='import sys; sys.path.insert(0,'+repr(str(ROOT/'tests/recovery_stage1'))+'); from test_cadic_lifecycle import fixture_worker; raise SystemExit(fixture_worker('+repr(str(setup))+'))'
        report=bounded_probe([sys.executable,'-B','-c',code],output,90)
        self.assertEqual(report['status'],'passed');self.assertEqual(report['normal_train_images_completed'],16)
        self.assertEqual(report['coreset']['count'],10000);self.assertTrue(report['reference_parity']['passed'])
        self.assertTrue(report['binding']['fit_task_and_snapshot_shared'])
        self.assertGreater(report['cache_progress']['closest_pair_calls'],0)
        self.assertNotIn('error',report);self.assertTrue(report['cleanup_complete'])

if __name__=='__main__':unittest.main()
