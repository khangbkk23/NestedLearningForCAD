# models/metanath_benchmark_adapter_v1.py
"""Historical Meta-NATH checkpoint adapter; never introduces test selection."""
from __future__ import annotations
import torch
from models.benchmark_adapter_base_v1 import BenchmarkMethodAdapter

class MetaNATHLegacyAdapterV1(BenchmarkMethodAdapter):
    def __init__(self, config, device="cpu"):
        self.config=config; self.device=device; self.model=None; self.loaded=False
        path=config.get("checkpoint",{}).get("path","")
        if not path: raise ValueError("Meta-NATH legacy adapter requires an existing final checkpoint path")
        from models.meta_nath_core import MetaNATHCore
        self.model=MetaNATHCore(device=device)
        payload=torch.load(path,map_location=device,weights_only=False)
        self.model.load_full_state_dict(payload.get("model_state_dict",payload)); self.loaded=True
    def fit_task(self, task_id, task_name, train_loader): raise RuntimeError("Historical Meta-NATH adapter is evaluation-only")
    def score_batch(self,batch):
        result=self.model.score_image(batch["images"].to(self.device)); rows=result.get("batch",[result])
        scores=torch.tensor([r["s_img"] for r in rows]); maps=torch.stack([r["anomaly_map"] for r in rows])
        return {"image_scores":scores,"anomaly_maps":maps}
    def state_dict(self): return self.model.full_state_dict(include_backbone=False,include_images=False)
    def load_state_dict(self,state): self.model.load_full_state_dict(state)
    def memory_stats(self): return {"persistent_bytes":0,"historical_selection_provenance":self.config["checkpoint"].get("selection_provenance")}
    def method_metadata(self): return {"adapter":"metanath_legacy","reportable":False,"selection_provenance":self.config["checkpoint"].get("selection_provenance")}
