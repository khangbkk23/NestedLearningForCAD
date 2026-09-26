# models/fake_benchmark_adapter_v1.py
import torch
from models.benchmark_adapter_base_v1 import BenchmarkMethodAdapter

class FakeBenchmarkAdapter(BenchmarkMethodAdapter):
    def __init__(self, device="cpu"):
        self.device = torch.device(device); self.offset = torch.tensor(0., device=self.device); self.updates = 0
    def fit_task(self, task_id, task_name, train_loader):
        total = 0.
        for batch in train_loader: total += float(batch["images"].mean())
        self.offset += total; self.updates += 1
        return {"samples": len(train_loader.dataset), "updates": 1}
    def score_batch(self, batch):
        x = batch["images"].to(self.device); score = x.mean(dim=(1,2,3)) + self.offset
        maps = score[:, None, None].expand(-1, x.shape[-2], x.shape[-1])
        return {"image_scores": score.detach().cpu(), "anomaly_maps": maps.detach().cpu()}
    def state_dict(self): return {"offset": self.offset.detach().cpu(), "updates": self.updates}
    def load_state_dict(self, state): self.offset = state["offset"].to(self.device).clone(); self.updates = int(state["updates"])
    def memory_stats(self): return {"persistent_bytes": int(self.offset.numel()*self.offset.element_size()), "updates": self.updates}
    def method_metadata(self): return {"adapter": "fake", "reportable": False}
