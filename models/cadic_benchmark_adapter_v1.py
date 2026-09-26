# models/cadic_benchmark_adapter_v1.py
"""Generic harness adapter composed from the isolated CADIC components."""
import time
import torch
import torch.nn.functional as F
from models.benchmark_adapter_base_v1 import BenchmarkMethodAdapter
from models.cadic_patch_coreset_v1 import CADICPatchCoresetV1, CADICPatchCoresetConfig
from models.feature_extractors.cadic_vit_v1 import CADICViTFeatureExtractor, CADICViTConfig

class CADICBenchmarkAdapterV1(BenchmarkMethodAdapter):
    def __init__(self, config, device="cpu"):
        self.device = torch.device(device); self.config = config
        ec = dict(config["extractor"]); ec["checkpoint_path"] = str(ec.get("checkpoint_path", ""))
        self.extractor = CADICViTFeatureExtractor(CADICViTConfig(**{k:v for k,v in ec.items() if k in CADICViTConfig.__dataclass_fields__}), self.device)
        mc = config["memory"]
        self.coreset = CADICPatchCoresetV1(CADICPatchCoresetConfig(budget=int(mc["budget"]), dim=int(ec.get("feature_dim",768)), dtype="float32", distance=mc["distance"], chunk_size=int(mc["chunk_size"]), image_neighbors=int(config["scoring"]["image_neighbors_b"])), device=self.device)
        self.timings = {"feature_seconds": 0., "update_seconds": 0., "score_seconds": 0.}
    def fit_task(self, task_id, task_name, train_loader):
        n = 0
        for batch in train_loader:
            start=time.perf_counter(); patches=self.extractor.extract_patch_features(batch["images"].to(self.device)); self.timings["feature_seconds"] += time.perf_counter()-start
            start=time.perf_counter(); result=self.coreset.update(patches); self.timings["update_seconds"] += time.perf_counter()-start; n += int(batch["images"].shape[0])
        return {"samples": n, "coreset": self.coreset.stats()}
    def score_batch(self, batch):
        start=time.perf_counter(); patches=self.extractor.extract_patch_features(batch["images"].to(self.device)); self.timings["feature_seconds"] += time.perf_counter()-start
        start=time.perf_counter(); image, pix = self.coreset.score(patches, b=self.config["scoring"]["image_neighbors_b"]); self.timings["score_seconds"] += time.perf_counter()-start
        maps=F.interpolate(pix.reshape(-1,1,28,28), size=batch["images"].shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        return {"image_scores": image.cpu(), "anomaly_maps": maps.cpu()}
    def state_dict(self): return {"coreset": self.coreset.state_dict(), "timings": self.timings}
    def load_state_dict(self, state): self.coreset.load_state_dict(state["coreset"]); self.timings.update(state.get("timings",{}))
    def memory_stats(self):
        extractor_bytes = sum(int(p.numel() * p.element_size()) for p in self.extractor.parameters())
        return {"persistent_bytes": self.coreset.memory_bytes + extractor_bytes,
                "coreset_feature_bytes": self.coreset.memory_bytes,
                "extractor_parameter_bytes": extractor_bytes,
                "coreset_count": self.coreset.count, "checkpoint_bytes": 0, **self.timings}
    def method_metadata(self): return {"adapter":"cadic", "cadic": self.extractor.protocol_metadata(), "coreset": self.coreset.stats()}
