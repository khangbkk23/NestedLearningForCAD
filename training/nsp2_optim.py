from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer


class ContinualBackpropMonitor:
    """
    Continual Backpropagation (CBP) helper.

    Tracks near-zero activations and re-initializes persistently dead neurons.
    """

    def __init__(
        self,
        model: nn.Module,
        module_name_filters: Optional[Sequence[str]] = None,
        patience: int = 50,
        activation_threshold: float = 1e-5,
    ) -> None:
        self.model = model
        self.module_name_filters = tuple(module_name_filters or ())
        self.patience = int(patience)
        self.activation_threshold = float(activation_threshold)

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._tracked_modules: Dict[str, nn.Linear] = {}
        self._dead_counters: Dict[str, torch.Tensor] = {}

        self._register_hooks()

    def _should_track(self, module_name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if not self.module_name_filters:
            return True
        lowered = module_name.lower()
        return any(token.lower() in lowered for token in self.module_name_filters)

    def _register_hooks(self) -> None:
        for module_name, module in self.model.named_modules():
            if not self._should_track(module_name, module):
                continue

            self._tracked_modules[module_name] = module
            self._dead_counters[module_name] = torch.zeros(
                module.out_features,
                dtype=torch.long,
                device=module.weight.device,
            )
            handle = module.register_forward_hook(self._make_hook(module_name))
            self._handles.append(handle)

    def _make_hook(self, module_name: str):
        def hook(module: nn.Module, _inputs, output: torch.Tensor) -> None:
            if not torch.is_tensor(output):
                return
            if output.ndim < 2:
                return

            reduce_dims = tuple(range(output.ndim - 1))
            activity = output.detach().abs().mean(dim=reduce_dims)
            dead_mask = activity <= self.activation_threshold

            counters = self._dead_counters[module_name]
            counters[dead_mask] += 1
            counters[~dead_mask] = 0

        return hook

    @torch.no_grad()
    def reinitialize_dead_neurons(self) -> Dict[str, int]:
        """Re-initialize dead neuron rows and reset their counters."""
        refreshed: Dict[str, int] = {}

        for module_name, module in self._tracked_modules.items():
            counters = self._dead_counters[module_name]
            dead_indices = torch.nonzero(counters >= self.patience, as_tuple=False).flatten()
            if dead_indices.numel() == 0:
                continue

            fan_in = max(module.in_features, 1)
            bound = 1.0 / math.sqrt(fan_in)
            module.weight[dead_indices].uniform_(-bound, bound)
            if module.bias is not None:
                module.bias[dead_indices].zero_()

            counters[dead_indices] = 0
            refreshed[module_name] = int(dead_indices.numel())

        return refreshed

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            name: counter.detach().cpu()
            for name, counter in self._dead_counters.items()
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        for name, counter in state_dict.items():
            if name not in self._dead_counters:
                continue
            self._dead_counters[name] = counter.to(self._dead_counters[name].device)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class NSP2Optimizer:
    """
    Optimizer wrapper implementing NSP2 gradient projection and CBP.

    Mathematical mapping used here for prompt gradient ``P_G``:
      - Prompt parameter is stored as ``(P, C)``
      - We project on transposed gradient ``G^T`` with shape ``(C, P)``
      - ``B1`` is derived from affinity covariance ``C1`` over prompt axis ``P``
      - ``B2`` is derived from aggregation covariance ``C2`` over channel axis ``C``
      - Update becomes ``Delta P = (B2 @ G^T @ B1)^T``
    """

    def __init__(
        self,
        optimizer: Optimizer,
        model: nn.Module,
        prompt_param_names: Sequence[str] = ("prompt_embeddings",),
        svd_tol: float = 1e-6,
        svd_rel_tol: float = 1e-4,
        cbp_patience: int = 50,
        cbp_activation_threshold: float = 1e-5,
        cbp_module_filters: Optional[Sequence[str]] = None,
    ) -> None:
        self.optimizer = optimizer
        self.model = model
        self.prompt_param_names = tuple(prompt_param_names)
        self.svd_tol = float(svd_tol)
        self.svd_rel_tol = float(svd_rel_tol)

        self.prompt_params: List[nn.Parameter] = self._collect_prompt_params()
        if not self.prompt_params:
            raise RuntimeError(
                "No prompt parameters found. Check prompt_param_names and model structure."
            )

        sample_prompt = self.prompt_params[0]
        if sample_prompt.ndim < 2:
            raise RuntimeError("Prompt parameter must be at least 2D.")

        self.prompt_len = int(sample_prompt.shape[-2])
        self.embed_dim = int(sample_prompt.shape[-1])
        self.device = sample_prompt.device

        self.cov_affinity_global = torch.zeros(self.prompt_len, self.prompt_len, device=self.device)
        self.cov_aggregation_global = torch.zeros(self.embed_dim, self.embed_dim, device=self.device)
        self.cov_affinity_task = torch.zeros_like(self.cov_affinity_global)
        self.cov_aggregation_task = torch.zeros_like(self.cov_aggregation_global)

        self.B1 = torch.eye(self.prompt_len, device=self.device)
        self.B2 = torch.eye(self.embed_dim, device=self.device)
        self._projection_dirty = True

        self.cbp = ContinualBackpropMonitor(
            model=model,
            module_name_filters=cbp_module_filters,
            patience=cbp_patience,
            activation_threshold=cbp_activation_threshold,
        )

    def _collect_prompt_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(token in name for token in self.prompt_param_names):
                params.append(param)
        return params

    @torch.no_grad()
    def accumulate_covariances(
        self,
        qxwk_mats: Iterable[torch.Tensor],
        sp_mats: Iterable[torch.Tensor],
    ) -> None:
        """
        Accumulate uncentered covariance matrices from model forward tensors.

        Args:
            qxwk_mats: Sequence of ``Q_X W_k^T`` tensors with trailing dim ``P``.
            sp_mats: Sequence of ``S_P`` tensors with trailing dim ``C``.
        """
        for qxwk in qxwk_mats:
            if qxwk.numel() == 0:
                continue
            q_flat = qxwk.detach().to(self.device).reshape(-1, qxwk.shape[-1])
            if q_flat.shape[1] != self.prompt_len:
                continue
            self.cov_affinity_task += q_flat.t() @ q_flat

        for sp in sp_mats:
            if sp.numel() == 0:
                continue
            s_flat = sp.detach().to(self.device).reshape(-1, sp.shape[-1])
            if s_flat.shape[1] != self.embed_dim:
                continue
            self.cov_aggregation_task += s_flat.t() @ s_flat

        self._projection_dirty = True

    def _null_projection(self, covariance: torch.Tensor) -> torch.Tensor:
        if torch.count_nonzero(covariance).item() == 0:
            return torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)

        _, singular_values, vh = torch.linalg.svd(covariance, full_matrices=False)
        right_vectors = vh.transpose(-2, -1)

        max_sv = float(singular_values.max().item())
        threshold = max(self.svd_tol, self.svd_rel_tol * max_sv)
        null_mask = singular_values <= threshold

        if not bool(null_mask.any()):
            min_index = int(torch.argmin(singular_values).item())
            null_mask[min_index] = True

        basis = right_vectors[:, null_mask]
        projection = basis @ basis.t()
        projection = 0.5 * (projection + projection.t())
        return projection

    @torch.no_grad()
    def refresh_projections(self) -> None:
        """Update NSP2 projection matrices from global task covariances."""
        self.B1 = self._null_projection(self.cov_affinity_global)
        self.B2 = self._null_projection(self.cov_aggregation_global)
        self._projection_dirty = False

    @torch.no_grad()
    def project_prompt_gradients(self) -> None:
        """Project prompt gradients: ``Delta P = B2 * P_G * B1``."""
        if self._projection_dirty:
            self.refresh_projections()

        for param in self.prompt_params:
            if param.grad is None:
                continue

            grad = param.grad
            if grad.ndim == 2:
                projected_t = self.B2 @ grad.t() @ self.B1
                grad.copy_(projected_t.t())
                continue

            if grad.ndim >= 3:
                grad_view = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
                for idx in range(grad_view.shape[0]):
                    projected_t = self.B2 @ grad_view[idx].t() @ self.B1
                    grad_view[idx].copy_(projected_t.t())

    @torch.no_grad()
    def step(self) -> Dict[str, int]:
        """Apply NSP2 projection, optimizer update, then CBP re-initialization."""
        self.project_prompt_gradients()
        self.optimizer.step()
        return self.cbp.reinitialize_dead_neurons()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def finalize_task(self, task_id: int, save_dir: Optional[str] = None) -> None:
        """
        Merge task covariances into the stability memory and persist artifacts.
        """
        self.cov_affinity_global += self.cov_affinity_task
        self.cov_aggregation_global += self.cov_aggregation_task

        self.refresh_projections()

        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "task_id": int(task_id),
                    "cov_affinity": self.cov_affinity_task.detach().cpu(),
                    "cov_aggregation": self.cov_aggregation_task.detach().cpu(),
                    "B1": self.B1.detach().cpu(),
                    "B2": self.B2.detach().cpu(),
                },
                save_path / f"cov_task_{int(task_id):02d}.pt",
            )

        self.cov_affinity_task.zero_()
        self.cov_aggregation_task.zero_()

    def state_dict(self) -> Dict[str, object]:
        return {
            "base_optimizer": self.optimizer.state_dict(),
            "cov_affinity_global": self.cov_affinity_global.detach().cpu(),
            "cov_aggregation_global": self.cov_aggregation_global.detach().cpu(),
            "cov_affinity_task": self.cov_affinity_task.detach().cpu(),
            "cov_aggregation_task": self.cov_aggregation_task.detach().cpu(),
            "B1": self.B1.detach().cpu(),
            "B2": self.B2.detach().cpu(),
            "cbp": self.cbp.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:
        self.optimizer.load_state_dict(state_dict["base_optimizer"])  # type: ignore[index]

        self.cov_affinity_global = state_dict["cov_affinity_global"].to(self.device)  # type: ignore[index]
        self.cov_aggregation_global = state_dict["cov_aggregation_global"].to(self.device)  # type: ignore[index]
        self.cov_affinity_task = state_dict["cov_affinity_task"].to(self.device)  # type: ignore[index]
        self.cov_aggregation_task = state_dict["cov_aggregation_task"].to(self.device)  # type: ignore[index]
        self.B1 = state_dict["B1"].to(self.device)  # type: ignore[index]
        self.B2 = state_dict["B2"].to(self.device)  # type: ignore[index]

        cbp_state = state_dict.get("cbp", {})
        if isinstance(cbp_state, dict):
            self.cbp.load_state_dict(cbp_state)

        self._projection_dirty = False

    def close(self) -> None:
        self.cbp.close()
