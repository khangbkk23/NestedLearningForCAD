# training/benchmark_engine_v1.py
"""Benchmark engine with triangular continual-forgetting evaluation."""

from __future__ import annotations

import copy
import time

import torch

from training.benchmark_metrics_v1 import compute_metrics, forgetting_matrix, macro_task_metrics


def _clone_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {str(k): _clone_state(v) for k, v in value.items() if k not in {"timings"}}
    if isinstance(value, list):
        return [_clone_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(v) for v in value)
    return copy.deepcopy(value)


def _state_equal(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_state_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_state_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_state_equal(x, y) for x, y in zip(a, b))
    return a == b


class BenchmarkEngineV1:
    def __init__(self, protocol, adapter, artifacts, device="cpu", fail_on_state_mutation=False):
        self.protocol = protocol
        self.adapter = adapter
        self.artifacts = artifacts
        self.device = device
        self.fail = bool(fail_on_state_mutation)
        self.times = {}

    def _load(self, fn, *args):
        started = time.perf_counter()
        result = fn(*args)
        self.times["data_loading_seconds"] = (
            self.times.get("data_loading_seconds", 0.0) + time.perf_counter() - started
        )
        return result

    def _snapshot_if_needed(self):
        return _clone_state(self.adapter.state_dict()) if self.fail else None

    def _assert_unchanged(self, before, message):
        if self.fail:
            after = _clone_state(self.adapter.state_dict())
            if not _state_equal(before, after):
                raise RuntimeError(message)

    def train(self, dataset, max_tasks=None, max_train_images=None, resume=False):
        limit = min(len(dataset.task_names), max_tasks or len(dataset.task_names))
        rows = []
        started = time.perf_counter()

        existing = sorted(self.artifacts.states.glob("task_*.pt")) if resume else []
        start_task = len(existing)

        if existing:
            self.adapter.load_state_dict(
                torch.load(existing[-1], map_location=self.device, weights_only=False)
            )

        for task_id in range(start_task, limit):
            task_name = dataset.task_names[task_id]
            loader = self._load(dataset.build_train_loader, task_id, max_train_images)
            print(
                f"\n[TRAIN] Task {task_id + 1}/{limit}: {task_name} | "
                f"samples={len(loader.dataset)} | batches={len(loader)}",
                flush=True,
            )

            task_started = time.perf_counter()
            result = self.adapter.fit_task(task_id, task_name, loader)
            task_seconds = time.perf_counter() - task_started
            self.times[f"train_task_{task_id}"] = task_seconds

            self.artifacts.save_state(f"task_{task_id:02d}.pt", self.adapter.state_dict())
            rows.append({
                "task_id": task_id,
                "task_name": task_name,
                "train_samples": len(loader.dataset),
                "result": result,
            })

            coreset = result.get("coreset", {})
            print(
                f"[TRAIN] Done {task_name} | {task_seconds:.1f}s | "
                f"coreset={coreset.get('count', '?')}/{coreset.get('budget', '?')} | "
                f"seen={coreset.get('seen_features', '?')}",
                flush=True,
            )

        self.artifacts.save_state("final.pt", self.adapter.state_dict())
        self.times["train_wall_seconds"] = time.perf_counter() - started
        self.artifacts.write_json("logs/train.json", rows)
        return rows

    def evaluate_final(self, dataset, max_tasks=None):
        before = self._snapshot_if_needed()
        per = {}
        started = time.perf_counter()
        names = dataset.task_names[:max_tasks] if max_tasks else dataset.task_names

        for task_id, name in enumerate(names):
            loader = self._load(dataset.build_test_loader, task_id)
            print(
                f"[EVAL] Task {task_id + 1}/{len(names)}: {name} | "
                f"samples={len(loader.dataset)} | batches={len(loader)}",
                flush=True,
            )
            scores, labels, maps, masks = [], [], [], []
            task_started = time.perf_counter()

            for batch in loader:
                out = self.adapter.score_batch(batch)
                scores.extend(out["image_scores"].tolist())
                labels.extend(batch["labels"].tolist())
                maps.append(out["anomaly_maps"])
                masks.append(batch["masks"])

            per[name] = compute_metrics(
                scores, labels, torch.cat(maps).numpy(), torch.cat(masks).numpy()
            )
            print(
                f"[EVAL] Done {name} | {time.perf_counter() - task_started:.1f}s | "
                f"I-AUROC={per[name]['i_auroc']:.4f} | P-AUPR={per[name]['p_aupr']:.4f}",
                flush=True,
            )

        self._assert_unchanged(before, "adapter state mutated during evaluation")

        self.times["final_eval_wall_seconds"] = time.perf_counter() - started
        total = sum(v["n_images"] for v in per.values())
        elapsed = self.times["final_eval_wall_seconds"]
        self.times["inference_fps"] = total / elapsed if elapsed else None

        macro = macro_task_metrics(per)
        self.artifacts.write_json("metrics/final_per_task.json", per)
        self.artifacts.write_json("metrics/final_macro.json", macro)

        profile = getattr(self.adapter, "profile_metadata", lambda: {})()
        self.artifacts.write_json("profile/adapter.json", profile)
        return per, macro

    def evaluate_forgetting(self, dataset, max_tasks=None):
        """Evaluate each checkpoint only on tasks learned so far."""
        states = sorted(self.artifacts.states.glob("task_*.pt"))
        if max_tasks is not None:
            states = states[:max_tasks]

        n = len(states)
        if n < 1:
            raise ValueError("FM requires at least one saved task state")

        from training.benchmark_metrics_v1 import image_auroc, pixel_aupr

        matrix_i, matrix_p = [], []

        for state_idx, state_path in enumerate(states):
            self.adapter.load_state_dict(
                torch.load(state_path, map_location=self.device, weights_only=False)
            )

            learned_name = dataset.task_names[state_idx]
            print(
                f"\n[FM] State {state_idx + 1}/{n}: after {learned_name} | "
                f"evaluating {state_idx + 1} learned task(s)",
                flush=True,
            )

            row_i = [None] * n
            row_p = [None] * n
            before = self._snapshot_if_needed()

            # State after task i is evaluated only on tasks 0..i.
            # Future tasks are deliberately not evaluated.
            for task_id in range(state_idx + 1):
                task_name = dataset.task_names[task_id]
                loader = self._load(dataset.build_test_loader, task_id)
                sample_count = (
                    len(loader.dataset)
                    if hasattr(loader, "dataset")
                    else "?"
                )
                print(
                    f"  [FM] Task {task_id + 1}/{state_idx + 1}: {task_name} | "
                    f"samples={sample_count}",
                    end="",
                    flush=True,
                )
                scores, labels, maps, masks = [], [], [], []
                task_started = time.perf_counter()

                for batch in loader:
                    out = self.adapter.score_batch(batch)
                    scores.extend(out["image_scores"].tolist())
                    labels.extend(batch["labels"].tolist())
                    maps.append(out["anomaly_maps"])
                    masks.append(batch["masks"])

                row_i[task_id] = image_auroc(scores, labels)
                row_p[task_id] = pixel_aupr(
                    torch.cat(maps).numpy(), torch.cat(masks).numpy()
                )
                print(
                    f" | {time.perf_counter() - task_started:.1f}s | "
                    f"I-AUROC={row_i[task_id]:.4f} | P-AUPR={row_p[task_id]:.4f}",
                    flush=True,
                )

            self._assert_unchanged(
                before, "adapter state mutated during forgetting evaluation"
            )

            matrix_i.append(row_i)
            matrix_p.append(row_p)

        image = forgetting_matrix(matrix_i)
        pixel = forgetting_matrix(matrix_p)

        result = {
            "matrix": matrix_i,
            "image": image,
            "pixel": pixel,
            "fm": image["fm"],
            "fm_p": pixel["fm"],
            "final_state_evaluated_after_training": True,
            "future_tasks_evaluated_before_learning": False,
        }

        self.artifacts.write_json(
            "metrics/forgetting_matrix.json", {"image": matrix_i, "pixel": matrix_p}
        )
        self.artifacts.write_json("metrics/forgetting_summary.json", result)
        return result
