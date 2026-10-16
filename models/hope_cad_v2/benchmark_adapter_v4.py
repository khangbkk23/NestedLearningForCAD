"""RGB-level HOPE-CAD adapter for the shared continual benchmark engine."""
from copy import deepcopy
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset.benchmark_protocol_v2 import file_hash
from dataset.hope_mask_views import mask_normalized_rgb, patch_masks
from models.benchmark_adapter_base_v1 import BenchmarkMethodAdapter
from models.feature_extractors.cadic_vit_v1 import CADICViTConfig, CADICViTFeatureExtractor
from .predictor import HOPEFeaturePredictorV4 as HOPEFeaturePredictor
from .teacher_head import frozen_teacher
from .visual_block import VisualBlockConfig
from training.hope_cms_session_v2 import CMSSession
from training.hope_runtime_v4 import source_identity, runtime_identity
from training.hope_initialization_v4 import validate_initialization


class HOPEBenchmarkAdapterV4(BenchmarkMethodAdapter):
    def __init__(self, initialization, *, checkpoint_path, checkpoint_sha256,
                 checkpoint_identity, device="cpu", extractor=None,
                 protocol_identity=None, allow_test_components=False, context=None):
        if initialization.get("schema") != "hope_deployment_initialization_v4":
            raise ValueError("expected a selected W4 deployment initialization")
        metadata = initialization.get("metadata", {})
        if metadata.get("design_id") != "hope_cad_v4_warm_joint_cms_row_retention":
            raise ValueError("initialization design identity mismatch")
        if not metadata.get("canonical") and not allow_test_components:
            raise ValueError("noncanonical predictor config is only allowed in explicit tests")
        if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
            raise ValueError("backbone SHA256 is required")

        feature_identity = initialization.get("provenance", {}).get("feature_identity", {})
        if feature_identity.get("backbone_sha256") != checkpoint_sha256:
            raise ValueError("initialization and runtime backbone hashes differ")
        if extractor is not None and not allow_test_components:
            raise ValueError("extractor injection is restricted to explicit tests")
        if not allow_test_components:
            if context is None:
                raise ValueError('scientific adapter requires a byte-verified protocol context')
            if 'w7_refit_receipt' in initialization:
                from training.hope_w7_refit import validate_refit
                validate_refit(initialization, context)
            else:
                validate_initialization(initialization, context)

        visual_config = VisualBlockConfig(**metadata["visual_config"])
        dynamics = metadata["dynamics"]
        method = dynamics["method"]
        beta = dynamics["rates"][0]
        model = HOPEFeaturePredictor(
            visual_config,
            method=method,
            beta=beta,
            selected_m_beta=beta if method == "C4" else None,
        )
        if model.metadata() != metadata:
            raise ValueError("deployment initialization model metadata mismatch")
        model.load_state_dict(initialization["slow_and_initial_weights"], strict=True)
        model.to(device).eval()

        self.device = torch.device(device)
        self.config = visual_config
        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_identity = checkpoint_identity
        self.protocol_identity = deepcopy(protocol_identity)
        self.image_size = (visual_config.height * 8, visual_config.width * 8)
        if extractor is None:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.is_file() or file_hash(checkpoint_path) != checkpoint_sha256:
                raise ValueError("frozen backbone checkpoint is missing or has the wrong SHA256")
            extractor_config = CADICViTConfig(
                model_name="vit_base_patch8_224",
                checkpoint_path=str(checkpoint_path),
                checkpoint_identity=checkpoint_identity,
                pretraining="ImageNet-21k",
                image_size=224,
                patch_size=8,
                embed_dim=768,
                layer_number=9,
                layer_indexing="one_based",
            )
            extractor = CADICViTFeatureExtractor(extractor_config, self.device)
        self.extractor = extractor
        identity = dict(
            schema="hope_rgb_session_v4",
            design_id=metadata["design_id"],
            method_metadata=metadata,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_identity=checkpoint_identity,
            protocol_identity=deepcopy(protocol_identity),
            initialization_sha256=initialization["initial_cms_sha256"],
            provenance=deepcopy(initialization["provenance"]),
            runtime_source_sha256=source_identity(),
            runtime=runtime_identity(),
        )
        self.session = CMSSession(model, identity)
        self.timings = {"feature_seconds": 0.0, "update_seconds": 0.0, "score_seconds": 0.0}

    def _validate_images(self, images):
        if (images.ndim != 4 or images.shape[1] != 3
                or tuple(images.shape[-2:]) != self.image_size
                or images.dtype != torch.float32
                or not bool(torch.isfinite(images).all())):
            raise ValueError(f"expected [B,3,{self.image_size[0]},{self.image_size[1]}] normalized RGB")

    def _extract(self, images):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        features = self.extractor.extract_patch_features(images)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.timings["feature_seconds"] += time.perf_counter() - started
        expected = (images.shape[0], self.config.height * self.config.width, self.config.input_dim)
        if features.shape != expected or not bool(torch.isfinite(features).all()):
            raise ValueError(f"frozen backbone feature contract mismatch: {tuple(features.shape)}")
        return features

    def fit_task(self, task_id, task_name, train_loader):
        samples = 0
        losses = []
        for batch in train_loader:
            images = batch["images"]
            self._validate_images(images)
            if "labels" not in batch or bool((batch["labels"] != 0).any()):
                raise ValueError("HOPE continual updates accept normal-only training images")
            for image in images.to(self.device):
                image = image.unsqueeze(0)

                def feature_factory(view):
                    with torch.no_grad():
                        target = frozen_teacher(self._extract(image))
                        masked = mask_normalized_rgb(
                            image, view, height=self.config.height, width=self.config.width
                        )
                        features = self._extract(masked)
                    mask = patch_masks(self.config.height, self.config.width,
                                       device=features.device)[view]
                    return features, target, mask

                started = time.perf_counter()
                result = self.session.update(feature_factory)
                self.timings["update_seconds"] += time.perf_counter() - started
                losses.append(tuple(float(value) for value in result.losses))
                samples += 1
        return {
            "task_id": int(task_id),
            "task_name": str(task_name),
            "samples": samples,
            "cms_clock": self.session.state.clock,
            "event_cursor": self.session.cursor,
            "cms_commits": list(self.session.state.commits),
            "cms_pending": list(self.session.state.pending),
            "mean_level_losses": [
                sum(event[level] for event in losses) / len(losses)
                for level in range(len(self.session.state.weights))
            ] if losses else [],
        }

    @torch.no_grad()
    def score_batch(self, batch):
        images = batch["images"]
        self._validate_images(images)
        started = time.perf_counter()
        image_scores, anomaly_maps = [], []
        masks = patch_masks(self.config.height, self.config.width, device=self.device)
        for image in images.to(self.device):
            image = image.unsqueeze(0)
            target = frozen_teacher(self._extract(image))
            patch_scores = torch.empty(
                self.config.height * self.config.width,
                dtype=target.dtype,
                device=target.device,
            )
            covered = torch.zeros_like(patch_scores, dtype=torch.int32)
            for view in range(4):
                masked = mask_normalized_rgb(
                    image, view, height=self.config.height, width=self.config.width
                )
                features = self._extract(masked)
                prediction = self.session.predict(features)
                scores = (prediction - target).square().sum(dim=-1)[0]
                if not bool(torch.isfinite(scores).all()):
                    raise FloatingPointError("non-finite HOPE anomaly score")
                patch_scores[masks[view]] = scores[masks[view]]
                covered[masks[view]] += 1
            if not bool((covered == 1).all()):
                raise RuntimeError("masked views must score every patch exactly once")
            image_scores.append(patch_scores.max())
            patch_map = patch_scores.reshape(1, 1, self.config.height, self.config.width)
            anomaly_maps.append(F.interpolate(
                patch_map,
                size=tuple(images.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )[0, 0])
        if not bool(torch.isfinite(torch.stack(anomaly_maps)).all()):
            raise FloatingPointError("non-finite HOPE anomaly map")
        self.timings["score_seconds"] += time.perf_counter() - started
        return {
            "image_scores": torch.stack(image_scores).cpu(),
            "anomaly_maps": torch.stack(anomaly_maps).cpu(),
        }

    def state_dict(self):
        versions = []
        for name, parameter in self.extractor.named_parameters():
            versions.append((name, int(parameter._version)))
        return {
            "schema": "hope_rgb_benchmark_adapter_v4",
            "identity": deepcopy(self.session.identity),
            "session": self.session.checkpoint(),
            "extractor_parameter_versions": versions,
        }

    def load_state_dict(self, state):
        if (state.get("schema") != "hope_rgb_benchmark_adapter_v4"
                or state.get("identity") != self.session.identity):
            raise ValueError("HOPE adapter checkpoint identity mismatch")
        self.session.restore(state["session"])

    def memory_stats(self):
        def bytes_in(groups):
            return sum(int(t.numel() * t.element_size())
                       for group in groups for t in group.values())

        backbone_bytes = sum(int(p.numel() * p.element_size())
                             for p in self.extractor.parameters())
        model_bytes = sum(int(p.numel() * p.element_size())
                          for p in self.session.model.parameters())
        cms_bytes = bytes_in(self.session.state.weights) + bytes_in(self.session.state.accumulators)
        return {
            "continual_memory_bytes": cms_bytes,
            "model_parameter_bytes": model_bytes,
            "backbone_parameter_bytes": backbone_bytes,
            "current_state_bytes": model_bytes + cms_bytes + backbone_bytes,
            "total_deployment_bytes": model_bytes + cms_bytes + backbone_bytes,
            "persistent_bytes": cms_bytes,
            "cms_clock": self.session.state.clock,
            "cms_commits": list(self.session.state.commits),
            "accounting_basis": "tensor payload bytes; excludes allocator/object overhead",
        }

    def profile_metadata(self):
        return dict(self.timings)

    def method_metadata(self):
        return {
            "adapter": "hope_cad_v4_rgb",
            "design_id": self.session.identity["design_id"],
            "full_rgb_adapter": True,
            "checkpoint_identity": self.checkpoint_identity,
            "checkpoint_sha256": self.checkpoint_sha256,
            "protocol_identity": deepcopy(self.protocol_identity),
            "score": "four-view masked fixed-teacher squared patch residual; image=max; map=bilinear",
            "update": "one normal image event, score-before-update, CMS persistent across tasks",
            "cms_updates": self.session.model.dynamics.cms_updates,
            "predictor": self.session.model.metadata(),
        }

    def close(self):
        close = getattr(self.extractor, "close", None)
        if close is not None:
            close()
