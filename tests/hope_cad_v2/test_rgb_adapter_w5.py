import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from dataset.hope_mask_views import mask_normalized_rgb, patch_masks
from models.hope_cad_v2.benchmark_adapter import HOPEBenchmarkAdapterV3
from models.hope_cad_v2.predictor import HOPEFeaturePredictor
from models.hope_cad_v2.visual_block import VisualBlockConfig
from training.benchmark_engine_v1 import _state_equal
from training.hope_outer_learning_v2 import export_initialization


class TinyExtractor:
    def __init__(self):
        self.device = torch.device("cpu")
        self.seen = []

    def extract_patch_features(self, images):
        self.seen.append(images.detach().clone())
        pooled = torch.nn.functional.avg_pool2d(images, kernel_size=8, stride=8)
        patches = pooled.flatten(2).transpose(1, 2)
        return torch.cat((patches, patches, patches[:, :, :2]), dim=-1)

    def named_parameters(self):
        return iter(())

    def close(self):
        pass


class OneImageDataset(Dataset):
    def __init__(self):
        self.image = torch.linspace(-1, 1, 3 * 32 * 32).reshape(3, 32, 32)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"images": self.image, "labels": 0, "masks": torch.zeros(32, 32, dtype=torch.bool)}


class RGBAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        config = VisualBlockConfig(8, 8, 8, 4, 4, 4)
        model = HOPEFeaturePredictor(config, method="M")
        identity = {
            "roles_digest": "roles",
            "schedule_digest": "schedule",
            "source_sha256": {"source.py": "source"},
            "spec_sha256": {"spec.md": "spec"},
            "feature_identity": {"backbone_sha256": "a" * 64},
        }
        initialization = export_initialization(model, identity)
        self.initialization = initialization
        self.extractor = TinyExtractor()
        self.adapter = HOPEBenchmarkAdapterV3(
            initialization,
            checkpoint_path="test-only.safetensors",
            checkpoint_sha256="a" * 64,
            checkpoint_identity="test-backbone",
            extractor=self.extractor,
            protocol_identity={"protocol_id": "test-only"},
            allow_test_components=True,
        )

    def tearDown(self):
        self.adapter.close()

    def test_rgb_mask_is_applied_before_feature_extraction(self):
        image = torch.ones(1, 3, 32, 32)
        masked = mask_normalized_rgb(image, 0, height=4, width=4)
        patch_mask = patch_masks(4, 4)[0].reshape(4, 4).repeat_interleave(8, 0).repeat_interleave(8, 1)
        self.assertTrue(torch.equal(masked[0, :, patch_mask], torch.zeros_like(masked[0, :, patch_mask])))
        self.assertTrue(torch.equal(masked[0, :, ~patch_mask], torch.ones_like(masked[0, :, ~patch_mask])))

    def test_training_updates_one_image_and_scoring_is_read_only(self):
        batch = next(iter(DataLoader(OneImageDataset(), batch_size=1)))
        before = self.adapter.state_dict()
        scored_before = self.adapter.score_batch(batch)
        self.assertEqual(tuple(scored_before["image_scores"].shape), (1,))
        self.assertEqual(tuple(scored_before["anomaly_maps"].shape), (1, 32, 32))
        self.assertTrue(bool(torch.isfinite(scored_before["anomaly_maps"]).all()))
        self.assertTrue(_state_equal(before, self.adapter.state_dict()))

        result = self.adapter.fit_task(0, "fixture", [batch])
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["cms_clock"], 1)
        self.assertEqual(result["cms_commits"], [1, 0])
        self.assertEqual(result["cms_pending"], [0, 1])
        self.assertTrue(bool((patch_masks(4, 4).sum(0) == 1).all()))
        scored_after = self.adapter.score_batch(batch)
        self.assertFalse(torch.equal(scored_before["anomaly_maps"], scored_after["anomaly_maps"]))

    def test_batch_order_does_not_change_per_image_scores(self):
        batch = next(iter(DataLoader(OneImageDataset(), batch_size=1)))
        images = torch.cat((batch["images"], batch["images"].flip(-1)), dim=0)
        forward = self.adapter.score_batch({"images": images})
        reverse = self.adapter.score_batch({"images": images.flip(0)})
        self.assertTrue(torch.equal(forward["image_scores"], reverse["image_scores"].flip(0)))
        self.assertTrue(torch.equal(forward["anomaly_maps"], reverse["anomaly_maps"].flip(0)))

    def test_training_rejects_anomaly_labeled_batch_before_state_change(self):
        batch = next(iter(DataLoader(OneImageDataset(), batch_size=1)))
        batch["labels"] = torch.ones_like(batch["labels"])
        before = self.adapter.state_dict()
        with self.assertRaisesRegex(ValueError, "normal-only"):
            self.adapter.fit_task(0, "fixture", [batch])
        self.assertTrue(_state_equal(before, self.adapter.state_dict()))

    def test_checkpoint_roundtrip_preserves_cms_clock_and_readout(self):
        batch = next(iter(DataLoader(OneImageDataset(), batch_size=1)))
        self.adapter.fit_task(0, "fixture", [batch])
        state = self.adapter.state_dict()
        expected = self.adapter.score_batch(batch)
        restored = HOPEBenchmarkAdapterV3(
            self.initialization,
            checkpoint_path="test-only.safetensors",
            checkpoint_sha256="a" * 64,
            checkpoint_identity="test-backbone",
            extractor=TinyExtractor(),
            protocol_identity={"protocol_id": "test-only"},
            allow_test_components=True,
        )
        restored.load_state_dict(state)
        actual = restored.score_batch(batch)
        self.assertEqual(restored.session.cursor, 1)
        self.assertTrue(torch.equal(expected["anomaly_maps"], actual["anomaly_maps"]))
        restored.close()


if __name__ == "__main__":
    unittest.main()