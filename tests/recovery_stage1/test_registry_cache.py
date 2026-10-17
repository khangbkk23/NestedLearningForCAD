"""Exercise the actual registry, adapter methods and compact snapshots; only ViT is a fixture."""
from pathlib import Path
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
import hashlib
import tempfile
import unittest
from unittest.mock import patch
import torch
from models.benchmark_registry_v2 import build_adapter,RegisteredAdapter
from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetV1
from training.hope_w7_adapter import CompactCADIC
from training.benchmark_engine_v1 import _state_equal
from notebook_support.recovery_stage1.cadic_accel import accelerate,CachedPairCoreset

ROOT=Path(__file__).resolve().parents[2]
class FixtureExtractor(torch.nn.Module):
    def __init__(self,config,device):
        super().__init__();self.device=torch.device(device);self.register_buffer('anchor',torch.zeros(1,device=device))
    def extract_patch_features(self,images):
        t=torch.arange(784,device=images.device).remainder(8).float()[None,:,None].expand(len(images),-1,-1)
        mean=images.mean((1,2,3))[:,None,None].expand(-1,784,-1)
        return torch.cat((t,t.square(),mean,t+mean),dim=-1)
    def protocol_metadata(self):return dict(fixture=True,canonical_ViT=False)
    def close(self):pass

class RegistryCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def setUp(self):
        base=ROOT/'results/recovery_stage1/tmp';base.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base);self.root=Path(self.tmp.name)
        p=self.root/'fixture.pt';p.write_bytes(b'fixture checkpoint')
        self.config=dict(id='fixture',adapter='cadic_paperfaithful_image_v1',
            extractor=dict(checkpoint_path=str(p),checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),feature_dim=4),
            preprocessing=dict(image_size=224,mean=[0,0,0],std=[1,1,1]),
            memory=dict(budget=5,update_unit='image',distance='euclidean',chunk_size=256,query_chunk_size=256,pair_chunk_size=256),
            scoring=dict(image_neighbors_b=3),runtime=dict(batch_size=8))
        self.batch={'images':torch.stack((torch.zeros(3,16,16),torch.ones(3,16,16)))}
    def tearDown(self):self.tmp.cleanup()
    def build(self):
        with patch('models.cadic_paperfaithful.cadic_benchmark_adapter.CADICViTFeatureExtractor',FixtureExtractor):
            return build_adapter(deepcopy(self.config),'cpu')
    def test_real_registry_binds_train_score_and_snapshot_to_same_cached_bank(self):
        registered=self.build();self.assertIsInstance(registered,RegisteredAdapter)
        fast=accelerate(registered)
        self.assertIs(registered.adapter.coreset,fast.coreset)
        self.assertIsInstance(registered.adapter.coreset,CachedPairCoreset)
        self.assertNotIn('coreset',vars(registered))
        with redirect_stdout(StringIO()):result=fast.fit_task(0,'normal',[self.batch])
        self.assertEqual(result['coreset'],fast.coreset.stats())
        self.assertEqual(fast.coreset.count,5)
        self.assertGreater(fast.coreset.cache_profile()['closest_pair_calls'],0)
        self.assertTrue(fast.binding_profile()['fit_task_and_snapshot_shared'])
    def test_registered_compact_resume_has_exact_reference_features_counters_and_scores(self):
        reference=self.build();fast=CompactCADIC(accelerate(self.build()),'fixture')
        with redirect_stdout(StringIO()):
            expected=reference.fit_task(0,'normal',[self.batch]);actual=fast.fit_task(0,'normal',[self.batch])
        self.assertEqual(expected,actual)
        self.assertTrue(_state_equal(reference.coreset.state_dict(),fast.state_dict()['coreset']))
        self.assertTrue(_state_equal(reference.score_batch(self.batch),fast.score_batch(self.batch)))
        state=deepcopy(fast.state_dict());resumed=CompactCADIC(accelerate(self.build()),'fixture');resumed.load_state_dict(state)
        self.assertTrue(_state_equal(state,resumed.state_dict()))
        self.assertTrue(_state_equal(reference.score_batch(self.batch),resumed.score_batch(self.batch)))
        with redirect_stdout(StringIO()):
            reference.fit_task(1,'next',[self.batch]);resumed.fit_task(1,'next',[self.batch])
        self.assertTrue(_state_equal(reference.coreset.state_dict(),resumed.state_dict()['coreset']))
    def test_registry_shadow_bank_is_rejected_before_any_training(self):
        fast=accelerate(self.build());before=deepcopy(fast.coreset.state_dict())
        fast.adapter.coreset=CADICPatchCoresetV1(fast.coreset.config)
        with self.assertRaisesRegex(RuntimeError,'binding'):fast.fit_task(0,'normal',[self.batch])
        self.assertTrue(_state_equal(before,fast._owner.coreset.state_dict()))

if __name__=='__main__':unittest.main()
