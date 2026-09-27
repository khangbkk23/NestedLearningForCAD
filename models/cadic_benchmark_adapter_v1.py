# models/cadic_benchmark_adapter_v1.py
"""Generic harness adapter composed from the isolated CADIC components."""

import sys
import time

import torch
import torch.nn.functional as F

from models.benchmark_adapter_base_v1 import BenchmarkMethodAdapter
from models.cadic_patch_coreset_v1 import CADICPatchCoresetV1, CADICPatchCoresetConfig
from models.feature_extractors.cadic_vit_v1 import CADICViTFeatureExtractor, CADICViTConfig


class CADICBenchmarkAdapterV1(BenchmarkMethodAdapter):
    def __init__(self, config, device="cpu"):
        self.device = torch.device(device)
        self.config = config

        ec = dict(config["extractor"])
        ec["checkpoint_path"] = str(ec.get("checkpoint_path", ""))
        ec.update(
            image_size=config["preprocessing"]["image_size"],
            mean=tuple(config["preprocessing"]["mean"]),
            std=tuple(config["preprocessing"]["std"]),
        )
        self.extractor = CADICViTFeatureExtractor(
            CADICViTConfig(
                **{k: v for k, v in ec.items() if k in CADICViTConfig.__dataclass_fields__}
            ),
            self.device,
        )

        mc = config["memory"]
        self.coreset = CADICPatchCoresetV1(
            CADICPatchCoresetConfig(
                budget=int(mc["budget"]),
                dim=int(ec.get("feature_dim", 768)),
                dtype="float32",
                distance=mc["distance"],
                chunk_size=int(mc["chunk_size"]),
                image_neighbors=int(config["scoring"]["image_neighbors_b"]),
                query_chunk_size=int(mc.get("query_chunk_size", 256)),
                pair_chunk_size=int(mc.get("pair_chunk_size", 256)),
            ),
            device=self.device,
        )
        self.timings = {"feature_seconds": 0.0, "update_seconds": 0.0, "score_seconds": 0.0}

    def fit_task(self, task_id, task_name, train_loader):
        n = 0
        total_batches = len(train_loader)
        total_images = len(train_loader.dataset)

        for batch_idx, batch in enumerate(train_loader, start=1):
            batch_images = int(batch["images"].shape[0])
            print(
                f"  [TRAIN:{task_name}] batch {batch_idx}/{total_batches} | "
                f"images={n + batch_images}/{total_images} | "
                f"coreset={self.coreset.count}/{self.coreset.config.budget}",
                end="",
                flush=True,
            )

            start = time.perf_counter()
            patches = self.extractor.extract_patch_features(batch["images"].to(self.device))
            feature_seconds = time.perf_counter() - start
            self.timings["feature_seconds"] += feature_seconds

            start = time.perf_counter()
            result = self.coreset.update(patches)
            update_seconds = time.perf_counter() - start
            self.timings["update_seconds"] += update_seconds
            n += batch_images

            stats = self.coreset.stats()
            print(
                f" | feature={feature_seconds:.2f}s | update={update_seconds:.2f}s | "
                f"coreset={stats['count']}/{stats['budget']} | "
                f"+accepted={result['accepted']} | +replaced={result['replaced']} | "
                f"+rejected={result['rejected']}",
                flush=True,
            )

        return {"samples": n, "coreset": self.coreset.stats()}

    def score_batch(self, batch):
        start = time.perf_counter()
        patches = self.extractor.extract_patch_features(batch["images"].to(self.device))
        self.timings["feature_seconds"] += time.perf_counter() - start

        start = time.perf_counter()
        image, pix = self.coreset.score(
            patches, b=self.config["scoring"]["image_neighbors_b"]
        )
        self.timings["score_seconds"] += time.perf_counter() - start

        maps = F.interpolate(
            pix.reshape(-1, 1, 28, 28),
            size=batch["images"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        return {"image_scores": image.cpu(), "anomaly_maps": maps.cpu()}

    def state_dict(self):
        # Include the frozen backbone in the scientific snapshot: a scoring
        # bug modifying its parameters must be detected by the evaluator.
        return {
            "coreset": self.coreset.state_dict(),
            "extractor_state": self.extractor.state_dict(),
        }

    def load_state_dict(self, state):
        self.coreset.load_state_dict(state["coreset"])
        if "extractor_state" in state:
            self.extractor.load_state_dict(state["extractor_state"], strict=True)
        # Legacy coreset-only states remain loadable using the required local
        # extractor checkpoint. Timers are profiler data rather than state.

    def profile_metadata(self):
        return {"distance_blocks": self.coreset.distance_profile(), **self.timings}

    def memory_stats(self):
        extractor_bytes = sum(
            int(p.numel() * p.element_size()) for p in self.extractor.parameters()
        )
        buffer_bytes = sum(
            int(b.numel() * b.element_size()) for b in self.extractor.buffers()
        )
        counters = (
            self.coreset.seen_features,
            self.coreset.accepted_features,
            self.coreset.replaced_features,
            self.coreset.rejected_features,
            self.coreset.update_batches,
        )
        counter_bytes = sum(sys.getsizeof(value) for value in counters)
        continual_bytes = self.coreset.memory_bytes + counter_bytes

        return {
            "continual_memory_bytes": continual_bytes,
            "model_parameter_bytes": extractor_bytes,
            "model_buffer_bytes": buffer_bytes,
            "total_deployment_bytes": continual_bytes + extractor_bytes + buffer_bytes,
            "persistent_bytes": continual_bytes,
            "coreset_feature_bytes": self.coreset.memory_bytes,
            "counter_metadata_bytes": counter_bytes,
            "accounting_basis": (
                "tensor payload bytes plus shallow Python int counter bytes; "
                "excludes allocator/object overhead and profiler"
            ),
            "extractor_parameter_bytes": extractor_bytes,
            "coreset_count": self.coreset.count,
            "checkpoint_bytes": None,
        }

    def method_metadata(self):
        return {
            "adapter": "cadic",
            "exact_parity_claim": False,
            "cadic": self.extractor.protocol_metadata(),
            "coreset": self.coreset.stats(),
            "public_compatibility_choice": self.config["extractor"].get(
                "public_compatibility_choice"
            ),
            "unresolved_assumptions": list(self.config.get("unresolved_assumptions", [])),
            "memory_scope": "single unified patch bank across tasks",
            "replacement_tie_policy": "lexicographic unordered (i,j), replace i",
            "pixel_aupr_integration": (
                "sklearn average_precision_score (declared compatibility assumption)"
            ),
        }
