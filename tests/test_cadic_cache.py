"""Reference-equivalence and invalidation tests for the CADIC execution cache."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import torch
from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetV1,CADICPatchCoresetConfig
from training.benchmark_engine_v1 import _state_equal
from models.cadic_paperfaithful.cache import CachedPairCoreset,accelerate,BACKEND

class CADICCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def pair(self,budget=11,dim=7,chunk=4,device='cpu'):
        config=CADICPatchCoresetConfig(budget=budget,dim=dim,chunk_size=chunk,
            query_chunk_size=3,pair_chunk_size=chunk,image_neighbors=3)
        return CADICPatchCoresetV1(config,device=device),CachedPairCoreset(config,device=device)
    def same_pair(self,a,b):
        av,ai=a._closest_pair();bv,bi=b._closest_pair()
        self.assertTrue(torch.equal(av,bv),(av,bv));self.assertEqual(ai,bi)
    def test_random_stream_exact_decisions_counters_features_and_scores(self):
        for seed in range(5):
            a,b=self.pair();g=torch.Generator().manual_seed(seed)
            stream=[torch.randn(n,7,generator=g) for n in (3,12,1,8,0,17,2)]
            stream.append(torch.randn(2,6,7,generator=g))
            for x in stream:
                self.assertEqual(a.update(x),b.update(x));self.assertTrue(_state_equal(a.state_dict(),b.state_dict()))
                self.same_pair(a,b)
            q=torch.randn(2,9,7,generator=g)
            self.assertTrue(_state_equal(a.pixel_scores(q),b.pixel_scores(q)))
    def test_ties_duplicates_and_fp32_cancellation_match_lexicographic_rule(self):
        for x in (torch.zeros(12,7),torch.arange(84).reshape(12,7).float(),torch.full((12,7),10000.)):
            a,b=self.pair();a.update(x);b.update(x);self.same_pair(a,b)
            self.assertEqual(a.update(x.flip(0)),b.update(x.flip(0)))
            self.assertTrue(_state_equal(a.state_dict(),b.state_dict()));self.same_pair(a,b)
    def test_only_tiles_touching_replaced_row_are_recomputed(self):
        a,b=self.pair(budget=19,chunk=4);x=torch.randn(19,7);a.update(x);b.update(x);self.same_pair(a,b)
        old=b.cache_profile()['tile_evaluations'];row=9;expected=len(b._adjacent[row//4])
        replacement=torch.randn(7);a.features[row]=replacement;b.features[row]=replacement;b._changed(row)
        self.same_pair(a,b)
        self.assertEqual(b.cache_profile()['tile_evaluations']-old,expected)
        self.assertLess(expected,b.cache_profile()['tile_count'])
    def test_external_write_and_tensor_replacement_force_full_rebuild(self):
        a,b=self.pair();a.update(torch.randn(11,7));b.load_state_dict(a.state_dict());self.same_pair(a,b)
        n=b.cache_profile()['full_rebuilds'];a.features[0]=0;b.features[0]=0;self.same_pair(a,b)
        self.assertEqual(b.cache_profile()['full_rebuilds'],n+1)
        a.features=a.features.flip(0);b.features=b.features.flip(0);self.same_pair(a,b)
        self.assertEqual(b.cache_profile()['full_rebuilds'],n+2)
    def test_checkpoint_excludes_cache_and_resume_matches_uninterrupted(self):
        a,b=self.pair();a.update(torch.randn(20,7));b.load_state_dict(a.state_dict());self.same_pair(a,b)
        payload=deepcopy(b.state_dict());self.assertEqual(set(payload),set(a.state_dict()))
        resumed=CachedPairCoreset(b.config);resumed.load_state_dict(payload)
        self.assertIsNone(resumed._tile_values)
        for _ in range(5):
            x=torch.randn(6,7);self.assertEqual(a.update(x),resumed.update(x))
            self.assertTrue(_state_equal(a.state_dict(),resumed.state_dict()))
    def test_small_bank_empty_invalid_geometry_and_singleton(self):
        for budget in (1,2,3):
            a,b=self.pair(budget=budget)
            for x in (torch.empty(0,7),torch.ones(1,7),torch.randn(5,7)):
                self.assertEqual(a.update(x),b.update(x));self.same_pair(a,b)
            before=b.state_dict()
            with self.assertRaises(ValueError):b.update(torch.ones(3,8))
            self.assertTrue(_state_equal(before,b.state_dict()))
    def test_runtime_cache_is_disclosed_and_scientific_state_unchanged(self):
        a,b=self.pair();a.update(torch.randn(11,7))
        base=SimpleNamespace(coreset=a,method_metadata=lambda:dict(adapter='cadic'),
            memory_stats=lambda:dict(runtime_cache_bytes=0,current_state_bytes=100,total_deployment_bytes=100,accounting_basis='fixture'))
        wrapped=accelerate(base);self.assertEqual(wrapped.method_metadata()['execution_optimization'],BACKEND)
        self.assertTrue(_state_equal(wrapped.coreset.state_dict(),a.state_dict()))
        wrapped.coreset._closest_pair();stats=wrapped.memory_stats()
        self.assertGreater(stats['runtime_cache_bytes'],0)
        self.assertEqual(stats['total_deployment_bytes'],100+stats['runtime_cache_bytes'])
    def test_reference_operator_gate_is_readonly_and_detects_corrupted_cache(self):
        from models.cadic_paperfaithful.cache import reference_parity
        _,b=self.pair();b.update(torch.randn(16,7));before=deepcopy(b.state_dict())
        report=reference_parity(b,torch.randn(4,7),torch.randn(1,8,7))
        self.assertTrue(report['passed']);self.assertTrue(_state_equal(before,b.state_dict()))
        self.assertTrue(report['main_state_rng_unchanged'])
        with self.assertRaises(ValueError):reference_parity(b,torch.randn(3,7),torch.randn(1,8,7))
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available')
    def test_cuda_full_and_edge_tiles_keep_exact_fp32_decisions(self):
        a,b=self.pair(budget=513,dim=768,chunk=256,device='cuda')
        g=torch.Generator().manual_seed(408)
        a.update(torch.randn(513,768,generator=g));b.load_state_dict(a.state_dict());self.same_pair(a,b)
        for n in (1,5,3):
            x=torch.randn(n,768,generator=g)
            self.assertEqual(a.update(x),b.update(x));self.assertTrue(_state_equal(a.state_dict(),b.state_dict()))
        q=torch.randn(1,7,768,generator=g);self.assertTrue(_state_equal(a.pixel_scores(q),b.pixel_scores(q)))

if __name__=='__main__':unittest.main()
