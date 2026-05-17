"""
consolidation_engine.py
-----------------------
Phase 3 N2B-NC consolidation for Meta-NATH CAD.

This module owns the algorithmic logic. Notebooks and CLI scripts should call
into it instead of carrying training code in cells.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cbp import CBPConfig, CBPMonitor
from models.meta_nath_core import MetaNATHCore
from models.null_space_proj import NSP2Config, NullSpaceProjector


@dataclass
class Phase3Config:
    top_k_anchors: int = 32
    unfreeze_last_blocks: int = 2
    drift_threshold: float = 0.05
    grad_clip: float = 1.0
    distill_weight: float = 1.0
    patch_distill_weight: float = 1.0
    lejepa_weight: float = 0.1
    lr: float = 1e-5
    weight_decay: float = 0.01
    steps: int = 1
    balanced_anchors: bool = True
    refresh_coreset: bool = True
    refresh_batch_size: int = 32


class NestedBackboneConsolidator:
    def __init__(
        self,
        core_model: MetaNATHCore,
        phase3_config: Phase3Config,
        nsp2_config: NSP2Config | None = None,
        cbp_config: CBPConfig | None = None,
        device: str | torch.device = "cuda",
    ):
        self.core = core_model
        self.config = phase3_config
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.projector = NullSpaceProjector(
            d=self.core.d,
            config=nsp2_config or NSP2Config(),
            device=self.device,
        )
        self.cbp = CBPMonitor(cbp_config or CBPConfig())
        self.predictor = nn.Linear(self.core.d, self.core.d, bias=False).to(self.device)

    def execute_global_consolidation(self) -> Dict[str, Any]:
        images, targets, anchor_stats = self.core.coreset.get_top_k_by_utility(
            self.config.top_k_anchors,
            balanced_by_task=bool(self.config.balanced_anchors),
            return_metadata=True,
        )
        if images is None:
            raise RuntimeError(
                "Phase 3 needs coreset anchor images. Run a warmup with "
                "model.store_images=true and logging.checkpoint_mode=phase3_full."
            )

        images = images.to(self.device)
        targets = targets.to(self.device).detach()
        self.core.backbone.to(self.device)

        original_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.core.backbone.state_dict().items()
        }

        with torch.no_grad():
            z_before, patch_targets = self._extract_tokens_with_grad(images)
            z_before = z_before.detach()
            patch_targets = patch_targets.detach()

        trainable_names = self._unfreeze_last_blocks()
        if not trainable_names:
            self._freeze_backbone()
            raise RuntimeError("No backbone parameters were selected for Phase 3 consolidation.")

        nsp2_stats = self.projector.fit(targets) if self.projector.config.enabled else self.projector.stats()
        optimizer = torch.optim.AdamW(
            list(self._trainable_backbone_parameters()) + list(self.predictor.parameters()),
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
        )

        losses: List[float] = []
        distill_losses: List[float] = []
        patch_distill_losses: List[float] = []
        lejepa_losses: List[float] = []
        cbp_stats: Dict[str, Any] = {}

        self.core.backbone.train()
        for _ in range(max(int(self.config.steps), 1)):
            optimizer.zero_grad(set_to_none=True)
            z_backbone, patch_backbone = self._extract_tokens_with_grad(images)
            distill_loss = F.mse_loss(z_backbone, targets)
            patch_distill_loss = F.mse_loss(patch_backbone, patch_targets)
            lejepa_loss = self._lejepa_surrogate(z_backbone, targets)
            loss = (
                float(self.config.distill_weight) * distill_loss
                + float(self.config.patch_distill_weight) * patch_distill_loss
                + float(self.config.lejepa_weight) * lejepa_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self._trainable_backbone_parameters()) + list(self.predictor.parameters()),
                max_norm=float(self.config.grad_clip),
            )

            if self.projector.config.enabled:
                self._apply_nsp2_to_gradients()

            optimizer.step()
            cbp_stats = self.cbp.scan_and_maybe_reset(self.core.backbone, projector=self.projector)

            losses.append(float(loss.detach().cpu()))
            distill_losses.append(float(distill_loss.detach().cpu()))
            patch_distill_losses.append(float(patch_distill_loss.detach().cpu()))
            lejepa_losses.append(float(lejepa_loss.detach().cpu()))

        self.core.backbone.eval()
        with torch.no_grad():
            z_after, patches_after = self._extract_tokens_with_grad(images)
            z_after = z_after.detach()
            patches_after = patches_after.detach()
            drift = 1.0 - F.cosine_similarity(z_before, z_after, dim=-1).mean()
            drift_value = float(drift.detach().cpu())
            patch_drift_value = float(F.mse_loss(patches_after, patch_targets).detach().cpu())

        rolled_back = drift_value > float(self.config.drift_threshold)
        refresh_stats = {"refreshed": False, "reason": "rolled_back" if rolled_back else "disabled"}
        if rolled_back:
            self.core.backbone.load_state_dict(original_state)
        elif self.config.refresh_coreset:
            refresh_stats = self.core.refresh_coreset_embeddings(
                batch_size=int(self.config.refresh_batch_size)
            )

        self._freeze_backbone()

        return {
            "success": not rolled_back,
            "rolled_back": rolled_back,
            "drift": drift_value,
            "patch_drift_mse": patch_drift_value,
            "drift_threshold": float(self.config.drift_threshold),
            "top_k_anchors": int(images.shape[0]),
            "anchor_selection": anchor_stats,
            "steps": max(int(self.config.steps), 1),
            "loss": losses[-1] if losses else None,
            "loss_history": losses,
            "distill_loss": distill_losses[-1] if distill_losses else None,
            "patch_distill_loss": patch_distill_losses[-1] if patch_distill_losses else None,
            "patch_distill_loss_history": patch_distill_losses,
            "lejepa_loss": lejepa_losses[-1] if lejepa_losses else None,
            "loss_weights": {
                "distill": float(self.config.distill_weight),
                "patch_distill": float(self.config.patch_distill_weight),
                "lejepa": float(self.config.lejepa_weight),
            },
            "trainable_param_count": len(trainable_names),
            "trainable_param_names": trainable_names,
            "coreset_refresh": refresh_stats,
            "nsp2": nsp2_stats,
            "cbp": cbp_stats,
        }

    def _extract_tokens_with_grad(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.core.backbone(images)
        if isinstance(out, dict) and "x_norm_clstoken" in out and "x_norm_patchtokens" in out:
            return out["x_norm_clstoken"], out["x_norm_patchtokens"]
        if hasattr(out, "last_hidden_state"):
            hidden_states = out.last_hidden_state
            return hidden_states[:, 0, :], hidden_states[:, 1:, :]
        if isinstance(out, dict) and "last_hidden_state" in out:
            hidden_states = out["last_hidden_state"]
            return hidden_states[:, 0, :], hidden_states[:, 1:, :]
        raise RuntimeError("Unexpected backbone output format during Phase 3 consolidation.")

    def _lejepa_surrogate(self, z_backbone: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if z_backbone.shape[0] < 2:
            return F.mse_loss(self.predictor(z_backbone), targets.detach())
        z_context = z_backbone[::2]
        z_target = targets[1::2]
        if z_target.numel() == 0:
            z_target = targets
        return F.mse_loss(
            self.predictor(z_context.mean(dim=0, keepdim=True)),
            z_target.mean(dim=0, keepdim=True).detach(),
        )

    def _unfreeze_last_blocks(self) -> List[str]:
        layer_ids = self._detect_layer_ids()
        selected_layers = set(layer_ids[-int(self.config.unfreeze_last_blocks):]) if layer_ids else set()
        trainable_names: List[str] = []

        for name, param in self.core.backbone.named_parameters():
            layer_id = self._parameter_layer_id(name)
            should_train = (
                layer_id in selected_layers
                or (layer_id is None and self._is_norm_parameter(name))
            )
            param.requires_grad_(bool(should_train))
            if should_train:
                trainable_names.append(name)

        return trainable_names

    def _freeze_backbone(self) -> None:
        self.core.backbone.eval()
        for param in self.core.backbone.parameters():
            param.requires_grad_(False)

    def _trainable_backbone_parameters(self):
        for param in self.core.backbone.parameters():
            if param.requires_grad:
                yield param

    def _apply_nsp2_to_gradients(self) -> None:
        for param in self.core.backbone.parameters():
            if param.grad is not None:
                param.grad = self.projector.project(param.grad)

    def _detect_layer_ids(self) -> List[int]:
        ids = set()
        for name, _ in self.core.backbone.named_parameters():
            layer_id = self._parameter_layer_id(name)
            if layer_id is not None:
                ids.add(layer_id)
        return sorted(ids)

    @staticmethod
    def _parameter_layer_id(name: str) -> int | None:
        for pattern in (r"encoder\.layer\.(\d+)", r"blocks\.(\d+)", r"layer\.(\d+)"):
            match = re.search(pattern, name)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _is_norm_parameter(name: str) -> bool:
        lowered = name.lower()
        return "norm" in lowered or "layernorm" in lowered
