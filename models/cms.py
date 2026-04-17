# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MlpBlock(nn.Module):
#     def __init__(self, in_features: int, hidden_features: int,
#                  out_features: int, drop: float = 0.):
#         super().__init__()
#         self.fc1  = nn.Linear(in_features, hidden_features)
#         self.act  = nn.GELU()
#         self.drop1 = nn.Dropout(drop)
#         self.fc2  = nn.Linear(hidden_features, out_features)
#         self.drop2 = nn.Dropout(drop)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop1(x)
#         x = self.fc2(x)
#         x = self.drop2(x)
#         return x


# class SpatialGatingUnit(nn.Module):
    
#     def __init__(self, dim: int):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.proj = nn.Linear(dim, 1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         gate = torch.sigmoid(self.proj(self.norm(x)))  # (B, N, 1)
#         return x * gate


# class CMS(nn.Module):
#     """
#     CV/CAD variant.

#     Args:
#         in_features:      token embedding dimension (e.g. 768 for ViT-B)
#         hidden_features:  total hidden dim (split evenly across levels)
#         out_features:     output dimension (defaults to in_features)
#         drop:             dropout rate
#         num_levels:       number of nested memory levels (paper default: 3)
#         k:                update ratio base (paper default: 2)
#         vit_layer_idx:    which ViT block this CMS lives in (set by
#                           replace_mlp_with_cms); determines active levels
#         use_spatial_gate: add SpatialGatingUnit at level 0 (recommended for CAD)
#     """

#     def __init__(
#         self,
#         in_features: int,
#         hidden_features: int = None,
#         out_features: int = None,
#         drop: float = 0.,
#         num_levels: int = 3,
#         k: int = 2,
#         vit_layer_idx: int = 0,
#         use_spatial_gate: bool = True,
#     ):
#         super().__init__()
#         out_features     = out_features or in_features
#         hidden_features  = hidden_features or in_features

#         self.num_levels     = num_levels
#         self.in_features    = in_features
#         self.out_features   = out_features
#         self.k              = k
#         self.vit_layer_idx  = vit_layer_idx

#         level_hidden = max(hidden_features // num_levels, 1)

#         self.levels = nn.ModuleList([
#             MlpBlock(in_features, level_hidden, in_features, drop)
#             for _ in range(num_levels)
#         ])

#         self.spatial_gate = SpatialGatingUnit(in_features) if use_spatial_gate else None

#         self.output_proj = (
#             nn.Linear(in_features, out_features)
#             if out_features != in_features else None
#         )

#         self._active_levels = [
#             i for i in range(num_levels)
#             if vit_layer_idx % (k ** i) == 0
#         ]
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         for i in self._active_levels:
#             delta = self.levels[i](x)
#             if i == 0 and self.spatial_gate is not None:
#                 delta = self.spatial_gate(delta)
#             x = x + delta

#         if self.output_proj is not None:
#             x = self.output_proj(x)
#         return x

#     def extra_repr(self) -> str:
#         active = self._active_levels
#         return (
#             f"in={self.in_features}, out={self.out_features}, "
#             f"levels={self.num_levels}, k={self.k}, "
#             f"vit_layer={self.vit_layer_idx}, active_levels={active}"
#         )

import torch
import torch.nn as nn

from .titans.memory import TitanMemory, TitanMemoryConfig
from .titans.self_mod import SelfModifier

class SpatialGatingUnit(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.proj(self.norm(x)))
        return x * gate

class CMS(nn.Module):
    """
    NeoViT-TITANS: Continual Anomaly Memory System
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        drop: float = 0.,
        num_levels: int = 3,  
        k: int = 2,
        vit_layer_idx: int = 0,
        use_spatial_gate: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.vit_layer_idx = vit_layer_idx
        
        # SWITCH: Only set True for NORMAL images loading
        self.update_allowed = False

        # TITANS
        titan_cfg = TitanMemoryConfig(
            dim=in_features,
            hidden_multiplier=4,
            layers=2,
            activation="gelu",
        )
        self.fast_memory = TitanMemory(titan_cfg)
        self.self_modifier = SelfModifier(dim=in_features, hidden_multiplier=4)
        self.spatial_gate = SpatialGatingUnit(in_features) if use_spatial_gate else None

    # Don't add teach_signal here to keep timm ViT integration safe
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Phase 1: Retrieval
        mem_out = self.fast_memory(x)

        if self.spatial_gate is not None:
            mem_out = self.spatial_gate(mem_out)

        out = x + mem_out

        # Phase 2 & 3: Measure and evolve (Delta Rule)
        # ONLY update when training and the switch is enabled (normal images)
        if self.training and self.update_allowed:
            target = x.detach()  # TITANS learns to memorize this exact feature representation
            
            # Compute surprise (if x is anomalous, surprise tends to spike)
            residual = target - mem_out
            surprise_scalar = self.fast_memory.surprise(residual).mean().item()

            # Dynamic LR: larger surprise leads to more conservative updates
            lr_dynamic = 0.01 / (1.0 + surprise_scalar)

            # Update memory weights directly without loss.backward()
            self.fast_memory.update(
                key=x,
                value=target,
                error_signal=None, 
                lr=lr_dynamic,
            )

        return out

    def extra_repr(self) -> str:
        return f"NeoViT-TITANS, in={self.in_features}, layer={self.vit_layer_idx}"