from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch


VALID_CHECKPOINT_POLICIES = {"all", "last_only", "best_and_last", "none"}


def resolve_checkpoint_policy(policy: str | None) -> str:
    resolved = str(policy or "last_only").lower()
    if resolved not in VALID_CHECKPOINT_POLICIES:
        raise ValueError(
            f"logging.checkpoint_policy must be one of {sorted(VALID_CHECKPOINT_POLICIES)}, "
            f"got {resolved!r}."
        )
    return resolved


@dataclass
class CheckpointManager:
    run_dir: str
    checkpoint_mode: str = "phase12_light"
    checkpoint_policy: str = "last_only"
    save_models: bool = True

    def __post_init__(self) -> None:
        self.checkpoint_mode = str(self.checkpoint_mode or "phase12_light").lower()
        self.checkpoint_policy = resolve_checkpoint_policy(self.checkpoint_policy)
        self.best_image_auroc = float("-inf")
        self.best_pixel_aupr = float("-inf")

    @property
    def include_full_state(self) -> bool:
        return self.checkpoint_mode == "phase3_full"

    def save_task(
        self,
        *,
        model: Any,
        config: Dict[str, Any],
        task_id: int,
        category: str,
        eval_metrics: Dict[str, Any],
    ) -> List[str]:
        if not self.save_models or self.checkpoint_policy == "none":
            return []

        payload = {
            "task_id": task_id,
            "category": category,
            "checkpoint_mode": self.checkpoint_mode,
            "checkpoint_policy": self.checkpoint_policy,
            "model_state_dict": model.full_state_dict(
                include_backbone=self.include_full_state,
                include_images=self.include_full_state,
            ),
            "config": config,
        }

        saved_paths: List[str] = []
        if self.checkpoint_policy == "all":
            path = os.path.join(self.run_dir, f"task_{task_id:02d}_checkpoint.pt")
            torch.save(payload, path)
            return [path]

        last_path = os.path.join(self.run_dir, "last_checkpoint.pt")
        torch.save(payload, last_path)
        saved_paths.append(last_path)

        if self.checkpoint_policy == "best_and_last":
            image_auroc = float(eval_metrics.get("image_auroc", 0.0))
            pixel_aupr = float(eval_metrics.get("pixel_aupr", 0.0))
            if image_auroc >= self.best_image_auroc:
                self.best_image_auroc = image_auroc
                path = os.path.join(self.run_dir, "best_image_auroc.pt")
                torch.save(payload, path)
                saved_paths.append(path)
            if pixel_aupr >= self.best_pixel_aupr:
                self.best_pixel_aupr = pixel_aupr
                path = os.path.join(self.run_dir, "best_pixel_aupr.pt")
                torch.save(payload, path)
                saved_paths.append(path)

        return saved_paths
