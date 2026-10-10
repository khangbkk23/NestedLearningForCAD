import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image
from dataset.benchmark_protocol_v2 import (build_prefix_manifest, calibration_roles,
    validate_prefix_manifest, PrefixContinualProtocol, file_hash)
from dataset.benchmark_manifest_v1 import manifest_digest
from models.benchmark_registry_v2 import route_descriptor, build_adapter
from training.hope_outer_learning_v2 import PrefixFeatures, make_schedule, validate_schedule


class PrefixContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = dict(id='fixture_prefix', dataset={'name': 'MVTec AD', 'root': str(self.root)}, task_order=['bottle', 'cable'])
        for task, count in [('bottle', 100), ('cable', 2)]:
            folder = self.root/task/'train/good'
            folder.mkdir(parents=True)
            for n in range(count):
                Image.new('RGB', (8, 8), (n, 0, 0)).save(folder/f'{n:03d}.png')
        self.manifest = build_prefix_manifest(self.config)

    def tearDown(self): self.temp.cleanup()

    def test_partition_data_permissions_and_matched_loader(self):
        m = self.manifest
        self.assertEqual([len(v) for v in m['roles'].values()], [50, 20, 15, 15])
        self.assertEqual(len(m['online']), 72)
        roles = calibration_roles(m)
        self.assertNotIn('entries', roles)
        self.assertNotIn('cable', str(roles))
        data = PrefixFeatures(roles, lambda *args: None)
        for role in ('normal_probe', 'official_test'):
            with self.assertRaises(ValueError): data.ids(role)
        with self.assertRaises(ValueError): data.get('cable/train/good/000.png', 0, 'meta_support')
        cfg = dict(runtime={'batch_size': 8}, preprocessing={'image_size': 224, 'mean': [.485,.456,.406], 'std': [.229,.224,.225]})
        protocol = PrefixContinualProtocol(self.config, m, cfg, seed=0)
        loader = protocol.build_train_loader(0)
        seen = [path for batch in loader for path in batch['relative_path']]
        self.assertEqual(seen, [r['relative_path'] for r in m['online'] if r['task_id'] == 0])

    def test_role_tampering_and_changed_bytes_fail(self):
        bad = copy.deepcopy(self.manifest)
        bad['roles']['meta_support'][0] = bad['entries'][-1]
        bad['digest'] = manifest_digest(bad)
        with self.assertRaises(ValueError): validate_prefix_manifest(bad, self.config)
        (self.root/'bottle/train/good/000.png').write_bytes(b'changed')
        with self.assertRaises(ValueError): validate_prefix_manifest(self.manifest, self.config)

    def test_schedule_common_random_numbers_and_16_unique(self):
        data = PrefixFeatures(calibration_roles(self.manifest), lambda *args: None)
        a, b = make_schedule(data, 0), make_schedule(data, 0)
        self.assertEqual(a, b)
        self.assertEqual(len(a['episodes']), 200)
        self.assertTrue(all(len(set(e['support'])) == 16 for e in a['episodes']))
        validate_schedule(a, data)
        a['episodes'][0]['support'][0] = a['episodes'][0]['support'][1]
        with self.assertRaises(ValueError): validate_schedule(a, data)

    def test_minimum_prefix_and_floor_splits(self):
        from dataset.benchmark_protocol_v2 import split_prefix
        for p, expected in ((32, [16,6,4,6]), (60, [30,12,9,9])):
            roles, _ = split_prefix(self.manifest['entries'][:p], ['bottle'])
            self.assertEqual([len(v) for v in roles.values()], expected)
        with self.assertRaises(ValueError): split_prefix(self.manifest['entries'][:31], ['bottle'])


class RegistryContracts(unittest.TestCase):
    def test_full_eq9_image_score_includes_distance_multiplier(self):
        from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetV1, CADICPatchCoresetConfig
        import math
        bank = CADICPatchCoresetV1(CADICPatchCoresetConfig(budget=3,dim=2))
        bank.update(torch.tensor([[0.,0.],[4.,0.],[0.,5.]]))
        image,pixels = bank.score(torch.tensor([[[1.,1.]]]),b=2)
        expected = math.sqrt(2)/(1+math.exp(math.sqrt(2)-math.sqrt(10)))
        self.assertAlmostEqual(image.item(),expected,places=6)
        self.assertAlmostEqual(pixels.item(),math.sqrt(2),places=6)

    def test_unknown_route_never_falls_back(self):
        for name in ('cadic', 'hope', 'unknown'):
            with self.assertRaises(ValueError): route_descriptor(name)
        self.assertEqual(route_descriptor('cadic_loader_batch_v1')['effective_update_unit'], 'loader_batch')
        self.assertEqual(route_descriptor('cadic_paperfaithful_image_v1')['effective_update_unit'], 'image')

    def test_registry_rejects_mislabeled_update_before_backbone_load(self):
        cfg = dict(adapter='cadic_loader_batch_v1', memory={'update_unit': 'image'})
        with self.assertRaises(ValueError): build_adapter(cfg)

    def test_effective_paperfaithful_image_update_counts(self):
        from models.cadic_paperfaithful.cadic_benchmark_adapter import CADICBenchmarkAdapterV1
        from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetV1, CADICPatchCoresetConfig
        # Real adapter fit_task and coreset; only frozen extractor replaced by tiny deterministic fixture.
        adapter = object.__new__(CADICBenchmarkAdapterV1)
        adapter.device = 'cpu'
        adapter.update_unit = 'image'
        adapter.timings = dict(feature_seconds=0., update_seconds=0., score_seconds=0.)
        adapter.extractor = type('Fixture', (), {'extract_patch_features': lambda _, x: x})()
        adapter.coreset = CADICPatchCoresetV1(CADICPatchCoresetConfig(budget=4, dim=2))
        adapter.fit_task(0, 'fixture', [{'images': torch.arange(12.).reshape(3,2,2)}])
        self.assertEqual(adapter.coreset.update_batches, 3)


class MaskQAContracts(unittest.TestCase):
    def test_tiny_defect_erasure_keeps_image_label_and_palette_nonzero(self):
        import numpy as np
        from dataset.visa_mask_qa_v2 import inspect_mask
        from dataset.benchmark_protocol_v2 import NonzeroMaskDataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB',(1024,800)).save(root/'image.png')
            raw = np.zeros((800,1024),dtype=np.uint8)
            raw[0,0] = 1
            Image.fromarray(raw).save(root/'mask.png')
            row = dict(relative_path='image.png',label=1,mask_path='mask.png')
            qa = inspect_mask(root,row)
            self.assertEqual(qa['original_positive_pixels'],1)
            self.assertEqual(qa['image_label'],1)
            self.assertTrue(qa['emptied_after_resize'])
            item = NonzeroMaskDataset(root,[row],224,[.485,.456,.406],[.229,.224,.225],False)[0]
            self.assertEqual(item['labels'],1)
            self.assertEqual(item['masks'].sum().item(),0)
            raw[200:600,200:700] = 1
            palette = Image.fromarray(raw).convert('P')
            palette.putpalette([0]*768)
            palette.save(root/'mask.png')
            item = NonzeroMaskDataset(root,[row],224,[.485,.456,.406],[.229,.224,.225],False)[0]
            self.assertGreater(item['masks'].sum().item(),0)

    def test_empty_or_disagreeing_raw_mask_fails(self):
        from dataset.visa_mask_qa_v2 import inspect_mask
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB',(16,16)).save(root/'image.png')
            row = dict(relative_path='image.png',label=1,mask_path='mask.png')
            for colour in ((0,0,0),(255,0,0)):
                Image.new('RGB',(16,16),colour).save(root/'mask.png')
                with self.assertRaises(ValueError): inspect_mask(root,row)


class ValidatorContracts(unittest.TestCase):
    def test_frozen_configs_fail_unknown_routes_and_leakage_and_allow_locked_orders(self):
        import yaml, random
        from training.benchmark_config_v2 import validate_configs
        root = Path(__file__).resolve().parents[2]
        protocol = yaml.safe_load((root/'conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml').read_text())
        method = yaml.safe_load((root/'conf/benchmarks/methods/cadic_matched_prefix_v2.yaml').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'fixture.pt'
            path.write_bytes(b'validator-only; not a usable backbone')
            protocol['dataset']['root'] = temporary
            method['extractor']['checkpoint_path'] = str(path)
            method['extractor']['checkpoint_sha256'] = file_hash(path)
            result = validate_configs(protocol,method,smoke=True)
            self.assertFalse(result['reportable'])
            broken = copy.deepcopy(method)
            broken['adapter'] = 'cadic'
            with self.assertRaises(ValueError): validate_configs(protocol,broken,smoke=True)
            bad = copy.deepcopy(protocol)
            bad['leakage']['future_task_calibration'] = 'allowed'
            with self.assertRaises(ValueError): validate_configs(bad,method,smoke=True)
            order = copy.deepcopy(protocol)
            order['order_seed'] = 100
            order['id'] += '_order100'
            tail = order['task_order'][1:]
            random.Random(100).shuffle(tail)
            order['task_order'] = [order['task_order'][0],*tail]
            validate_configs(order,method,smoke=True)

    def test_resume_lock_ignores_mount_paths_but_not_checkpoint_or_adapter_hash(self):
        from scripts.hope_cad_v2.run_matched_cadic import baseline_identity
        def identity(root,checkpoint,sha='abc'):
            return baseline_identity({'dataset':{'root':root}}, {'extractor':{'checkpoint_path':checkpoint,'checkpoint_sha256':sha}},
                                     {'route':'fixed'},'datahash',seed=0,smoke=True,max_tasks=1,max_train_images=8)
        self.assertEqual(identity('D:/data','D:/model'),identity('/kaggle/input/data','/kaggle/input/model'))
        self.assertNotEqual(identity('D:/data','D:/model'),identity('D:/data','D:/model','different'))


if __name__ == '__main__': unittest.main()
