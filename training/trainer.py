from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.titans_memory import TITANSMemory
from .memory_buffer import SlowMemory
from .cms_optim import CMSOptimizerWrapper
from .nsp2_optim import NSP2Optimizer


class Trainer:
    """
    Continual anomaly trainer with ACC gating, TITANS memory and NSP2 updates.

    Backward components:
      - NSP2 null-space projected prompt gradients
      - CBP dead-neuron re-initialization via optimizer wrapper
      - Slow memory utility buffer for high-diversity normal samples
      - LayerNorm prompt drift regularization
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        optimizer: Optional[optim.Optimizer] = None,
        learning_rate: float = 1e-4,
        use_replay: bool = False,
        replay_batch_size: int = 32,
        task_type: str = "anomaly",
        pixel_loss_weight: float = 0.2,
        weight_decay: float = 1e-5,
        acc_loss_weight: float = 0.5,
        proxy_loss_weight: float = 0.5,
        ln_loss_weight: float = 0.1,
        gradient_clip_norm: float = 1.0,
        titans_bank_size: int = 8192,
        titans_k_neighbors: int = 8,
        slow_memory_size: int = 2048,
        nsp2_svd_tol: float = 1e-6,
        nsp2_svd_rel_tol: float = 1e-4,
        cbp_patience: int = 50,
        cbp_activation_threshold: float = 1e-5,
        covariance_dir: Optional[str] = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device

        # Legacy args kept for compatibility with old entry points.
        self.use_replay = use_replay
        self.replay_batch_size = replay_batch_size
        self.task_type = task_type

        self.pixel_loss_weight = float(pixel_loss_weight)
        self.acc_loss_weight = float(acc_loss_weight)
        self.proxy_loss_weight = float(proxy_loss_weight)
        self.ln_loss_weight = float(ln_loss_weight)
        self.gradient_clip_norm = float(gradient_clip_norm)

        self.image_criterion = nn.BCELoss()
        self.proxy_criterion = nn.BCEWithLogitsLoss()
        self.acc_criterion = nn.CrossEntropyLoss()

        embedding_dim = int(getattr(self.model, "embed_dim", 768))

        self.titans_memory = TITANSMemory(
            embedding_dim=embedding_dim,
            bank_size=titans_bank_size,
            k_neighbors=titans_k_neighbors,
        ).to(device)

        if hasattr(self.model, "set_titans_memory"):
            self.model.set_titans_memory(self.titans_memory)
        else:
            setattr(self.model, "_titans_memory", self.titans_memory)

        self.slow_memory = SlowMemory(
            capacity=slow_memory_size,
            embedding_dim=embedding_dim,
        )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        base_optimizer = optimizer
        if base_optimizer is None:
            base_optimizer = optim.AdamW(
                trainable_params,
                lr=float(learning_rate),
                weight_decay=float(weight_decay),
            )

        cms_optimizer = CMSOptimizerWrapper(
            optimizer=base_optimizer,
            model=self.model,
        )

        self.optimizer = NSP2Optimizer(
            optimizer=cms_optimizer,
            model=self.model,
            prompt_param_names=("prompt_embeddings",),
            svd_tol=nsp2_svd_tol,
            svd_rel_tol=nsp2_svd_rel_tol,
            cbp_patience=cbp_patience,
            cbp_activation_threshold=cbp_activation_threshold,
            cbp_module_filters=("acc_gating", "proxy_head"),
        )

        self.prev_prompt_mu: Optional[torch.Tensor] = None
        self.prev_prompt_sigma: Optional[torch.Tensor] = None

        self.covariance_dir = Path(covariance_dir) if covariance_dir else None

    def _unpack_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(batch, dict):
            images = batch["img"]
            labels = batch["anomaly"]
            masks = batch.get("img_mask")
            return images, labels, masks

        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            images, labels = batch[0], batch[1]
            return images, labels, None

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _acc_targets(self, labels: torch.Tensor) -> torch.Tensor:
        # ACC class 1: shift/normal accepted as new normal.
        # ACC class 0: suspected defect.
        labels = labels.long().view(-1)
        return torch.where(labels == 0, torch.ones_like(labels), torch.zeros_like(labels))

    def _ln_drift_loss(self, prompt_mu: torch.Tensor, prompt_sigma: torch.Tensor) -> torch.Tensor:
        if self.prev_prompt_mu is None or self.prev_prompt_sigma is None:
            return torch.zeros((), device=self.device)

        return (
            torch.abs(prompt_mu - self.prev_prompt_mu).mean()
            + torch.abs(prompt_sigma - self.prev_prompt_sigma).mean()
        )

    @torch.no_grad()
    def _update_prompt_reference(self, prompt_mu: torch.Tensor, prompt_sigma: torch.Tensor) -> None:
        self.prev_prompt_mu = prompt_mu.detach()
        self.prev_prompt_sigma = prompt_sigma.detach()

    @torch.no_grad()
    def _update_memories(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        outputs: Dict[str, Any],
    ) -> None:
        labels = labels.long().view(-1)
        normal_mask = labels == 0

        if not torch.any(normal_mask):
            return

        shift_probability = outputs.get("shift_probability", None)
        if not torch.is_tensor(shift_probability):
            accept_mask = normal_mask
        else:
            threshold = float(getattr(self.model, "gating_threshold", 0.55))
            accept_mask = normal_mask & (shift_probability.detach() > threshold)

        if not torch.any(accept_mask):
            return

        embeddings = outputs["features"][accept_mask]
        selected_images = images[accept_mask]

        self.titans_memory.update(embeddings)
        self.slow_memory.add_batch(
            images=selected_images,
            embeddings=embeddings.detach().cpu(),
        )

    def train_task(
        self,
        train_loader: DataLoader,
        task_id: int,
        epochs: int = 10,
        verbose: bool = True,
    ) -> Dict[str, float]:
        self.model.train()

        history = {
            "loss": [],
            "accuracy": [],
            "image_loss": [],
            "pixel_loss": [],
            "acc_loss": [],
            "proxy_loss": [],
            "ln_loss": [],
        }

        if verbose:
            print(f"Training task {task_id} for {epochs} epochs...")

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            epoch_image_loss = 0.0
            epoch_pixel_loss = 0.0
            epoch_acc_loss = 0.0
            epoch_proxy_loss = 0.0
            epoch_ln_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Task {task_id} Epoch {epoch + 1}/{epochs}") if verbose else train_loader

            for batch in pbar:
                images, labels, masks = self._unpack_batch(batch)
                images = images.to(self.device)
                labels = labels.to(self.device)
                if torch.is_tensor(masks):
                    masks = masks.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)

                self.optimizer.accumulate_covariances(
                    qxwk_mats=outputs.get("qxwk_mats", []),
                    sp_mats=outputs.get("sp_mats", []),
                )

                labels_float = labels.float().view(-1)
                image_score = outputs["image_score"].view(-1).clamp(1e-6, 1 - 1e-6)
                proxy_logit = outputs["proxy_logit"].view(-1)
                acc_logits = outputs["acc_logits"]
                prompt_mu = outputs["prompt_mu"]
                prompt_sigma = outputs["prompt_sigma"]

                image_loss = self.image_criterion(image_score, labels_float)
                proxy_loss = self.proxy_criterion(proxy_logit, labels_float)
                acc_loss = self.acc_criterion(acc_logits, self._acc_targets(labels))
                ln_loss = self._ln_drift_loss(prompt_mu, prompt_sigma)

                total_loss = (
                    image_loss
                    + self.proxy_loss_weight * proxy_loss
                    + self.acc_loss_weight * acc_loss
                    + self.ln_loss_weight * ln_loss
                )

                pixel_loss_value = torch.zeros((), device=self.device)
                anomaly_map = outputs.get("anomaly_map", None)
                if torch.is_tensor(masks) and torch.is_tensor(anomaly_map):
                    target_masks = masks.float().clamp(0.0, 1.0)
                    if target_masks.ndim == 3:
                        target_masks = target_masks[:, None, :, :]
                    if target_masks.shape[-2:] != anomaly_map.shape[-2:]:
                        target_masks = F.interpolate(
                            target_masks,
                            size=anomaly_map.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    pixel_loss_value = F.binary_cross_entropy(
                        anomaly_map.clamp(1e-6, 1 - 1e-6),
                        target_masks,
                    )
                    total_loss = total_loss + self.pixel_loss_weight * pixel_loss_value

                total_loss.backward()

                trainable_params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                if trainable_params:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.gradient_clip_norm)

                cbp_refreshed = self.optimizer.step()
                _ = cbp_refreshed

                self._update_prompt_reference(prompt_mu, prompt_sigma)
                self._update_memories(images=images, labels=labels, outputs=outputs)

                pred = (image_score >= 0.5).long()
                epoch_correct += pred.eq(labels.long().view(-1)).sum().item()
                epoch_total += labels.shape[0]

                epoch_loss += float(total_loss.item())
                epoch_image_loss += float(image_loss.item())
                epoch_pixel_loss += float(pixel_loss_value.item())
                epoch_acc_loss += float(acc_loss.item())
                epoch_proxy_loss += float(proxy_loss.item())
                epoch_ln_loss += float(ln_loss.item())

                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix(
                        {
                            "loss": f"{total_loss.item():.4f}",
                            "acc": f"{100.0 * epoch_correct / max(epoch_total, 1):.2f}%",
                        }
                    )

            batches = max(len(train_loader), 1)
            avg_loss = epoch_loss / batches
            avg_acc = 100.0 * epoch_correct / max(epoch_total, 1)

            history["loss"].append(avg_loss)
            history["accuracy"].append(avg_acc)
            history["image_loss"].append(epoch_image_loss / batches)
            history["pixel_loss"].append(epoch_pixel_loss / batches)
            history["acc_loss"].append(epoch_acc_loss / batches)
            history["proxy_loss"].append(epoch_proxy_loss / batches)
            history["ln_loss"].append(epoch_ln_loss / batches)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {avg_loss:.4f} - "
                    f"Acc: {avg_acc:.2f}%"
                )

        self.optimizer.finalize_task(
            task_id=task_id,
            save_dir=str(self.covariance_dir) if self.covariance_dir is not None else None,
        )

        return {
            "loss": sum(history["loss"]) / max(len(history["loss"]), 1),
            "accuracy": sum(history["accuracy"]) / max(len(history["accuracy"]), 1),
            "image_loss": sum(history["image_loss"]) / max(len(history["image_loss"]), 1),
            "pixel_loss": sum(history["pixel_loss"]) / max(len(history["pixel_loss"]), 1),
            "acc_loss": sum(history["acc_loss"]) / max(len(history["acc_loss"]), 1),
            "proxy_loss": sum(history["proxy_loss"]) / max(len(history["proxy_loss"]), 1),
            "ln_loss": sum(history["ln_loss"]) / max(len(history["ln_loss"]), 1),
            "history": history,
        }

    def set_learning_rate(self, lr: float) -> None:
        base_optimizer = self._resolve_base_optimizer()
        for param_group in base_optimizer.param_groups:
            param_group["lr"] = float(lr)

    def _resolve_base_optimizer(self) -> optim.Optimizer:
        optimizer: object = self.optimizer
        while hasattr(optimizer, "optimizer"):
            optimizer = getattr(optimizer, "optimizer")
        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError("Resolved optimizer is not a torch Optimizer.")
        return optimizer