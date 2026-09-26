"""Leakage-barriered benchmark phases and state mutation guard."""
from __future__ import annotations
import hashlib, json, time
import torch
from training.benchmark_metrics_v1 import compute_metrics, macro_task_metrics, forgetting_matrix

def _fingerprint(obj):
    def norm(v):
        if isinstance(v, torch.Tensor):
            return (str(v.dtype), tuple(v.shape), hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest())
        if isinstance(v, dict): return {str(k): norm(x) for k, x in sorted(v.items()) if k not in {"timings"}}
        if isinstance(v, (list, tuple)): return [norm(x) for x in v]
        if isinstance(v, (str, int, float, bool, type(None))): return v
        return repr(v)
    return hashlib.sha256(json.dumps(norm(obj), sort_keys=True, default=str).encode()).hexdigest()

class BenchmarkEngineV1:
    def __init__(self, protocol, adapter, artifacts, device="cpu", fail_on_state_mutation=False):
        self.protocol, self.adapter, self.artifacts = protocol, adapter, artifacts
        self.device, self.fail = device, fail_on_state_mutation
        self.times = {}

    def _load(self, fn, *args):
        started = time.perf_counter(); result = fn(*args)
        self.times["data_loading_seconds"] = self.times.get("data_loading_seconds", 0.) + time.perf_counter() - started
        return result

    def train(self, dataset, max_tasks=None, max_train_images=None, resume=False):
        limit = min(len(dataset.task_names), max_tasks or len(dataset.task_names)); rows = []; started = time.perf_counter()
        existing = sorted(self.artifacts.states.glob("task_*.pt")) if resume else []
        start_task = len(existing)
        if existing:
            self.adapter.load_state_dict(torch.load(existing[-1], map_location=self.device, weights_only=False))
        for task_id in range(start_task, limit):
            loader = self._load(dataset.build_train_loader, task_id, max_train_images)
            task_started = time.perf_counter(); result = self.adapter.fit_task(task_id, dataset.task_names[task_id], loader)
            self.times[f"train_task_{task_id}"] = time.perf_counter() - task_started
            self.artifacts.save_state(f"task_{task_id:02d}.pt", self.adapter.state_dict())
            rows.append({"task_id": task_id, "task_name": dataset.task_names[task_id], "train_samples": len(loader.dataset), "result": result})
        self.artifacts.save_state("final.pt", self.adapter.state_dict())
        self.times["train_wall_seconds"] = time.perf_counter() - started
        self.artifacts.write_json("logs/train.json", rows)
        return rows

    def evaluate_final(self, dataset, max_tasks=None):
        before = _fingerprint(self.adapter.state_dict()); per = {}; started = time.perf_counter()
        names = dataset.task_names[:max_tasks] if max_tasks else dataset.task_names
        for task_id, name in enumerate(names):
            loader = self._load(dataset.build_test_loader, task_id); scores, labels, maps, masks = [], [], [], []
            for batch in loader:
                out = self.adapter.score_batch(batch); scores.extend(out["image_scores"].tolist()); labels.extend(batch["labels"].tolist())
                maps.append(out["anomaly_maps"]); masks.append(batch["masks"])
            per[name] = compute_metrics(scores, labels, torch.cat(maps).numpy(), torch.cat(masks).numpy())
        if before != _fingerprint(self.adapter.state_dict()):
            raise RuntimeError("adapter state mutated during evaluation")
        self.times["final_eval_wall_seconds"] = time.perf_counter() - started
        total = sum(value["n_images"] for value in per.values())
        self.times["inference_fps"] = total / self.times["final_eval_wall_seconds"] if self.times["final_eval_wall_seconds"] else None
        macro = macro_task_metrics(per)
        self.artifacts.write_json("metrics/final_per_task.json", per); self.artifacts.write_json("metrics/final_macro.json", macro)
        return per, macro

    def evaluate_forgetting(self, dataset, max_tasks=None):
        states = sorted(self.artifacts.states.glob("task_*.pt")); n = len(states); matrix_i, matrix_p = [], []
        for state_path in states:
            self.adapter.load_state_dict(torch.load(state_path, map_location=self.device, weights_only=False)); row_i, row_p = [], []
            for task_id in range(n):
                loader = self._load(dataset.build_test_loader, task_id); scores, labels, maps, masks = [], [], [], []
                for batch in loader:
                    out = self.adapter.score_batch(batch); scores.extend(out["image_scores"].tolist()); labels.extend(batch["labels"].tolist())
                    maps.append(out["anomaly_maps"]); masks.append(batch["masks"])
                from training.benchmark_metrics_v1 import image_auroc, pixel_aupr
                row_i.append(image_auroc(scores, labels)); row_p.append(pixel_aupr(torch.cat(maps).numpy(), torch.cat(masks).numpy()))
            matrix_i.append(row_i); matrix_p.append(row_p)
        image = forgetting_matrix(matrix_i); pixel = forgetting_matrix(matrix_p)
        result = {"matrix": matrix_i, "image": image, "pixel": pixel, "fm": image["fm"], "fm_p": pixel["fm"]}
        self.artifacts.write_json("metrics/forgetting_matrix.json", {"image": matrix_i, "pixel": matrix_p})
        self.artifacts.write_json("metrics/forgetting_summary.json", result)
        return result
