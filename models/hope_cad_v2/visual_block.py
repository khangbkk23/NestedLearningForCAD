"""W3 masked-patch visual front end; CMS/decoder/AD scoring belong to W4/W5."""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .scan_geometry import scan_routes
from .visual_memory import FourScanVisualMemory


@dataclass(frozen=True)
class VisualBlockConfig:
    input_dim: int = 768
    dim: int = 256
    memory_dim: int = 128
    head_dim: int = 16
    height: int = 28
    width: int = 28

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in asdict(self).values()):
            raise ValueError("all visual dimensions must be positive integers")
        if self.memory_dim % self.head_dim:
            raise ValueError("memory_dim must be divisible by head_dim")
        scan_routes(self.height, self.width)

    @property
    def canonical(self) -> bool:
        return self == VisualBlockConfig()


class HOPECADVisualBlock(nn.Module):
    """P + minimal spatial block of architecture_decision_v2.md §3.1.

    Input: already extracted *masked RGB* frozen-ViT patches [B, 784, 768].
    Output: h0 [B, 784, 256], to be consumed by CMS. This is not an anomaly
    detector by itself. The module never accepts teacher targets or task IDs.
    Reduced configs are explicitly noncanonical and intended for test fixtures.
    """

    def __init__(self, config: VisualBlockConfig | None = None):
        super().__init__()
        self.config = config or VisualBlockConfig()
        c = self.config
        self.projection = nn.Linear(c.input_dim, c.dim)
        self.input_norm = nn.LayerNorm(c.dim, eps=1e-6)
        self.in_projection = nn.Linear(c.dim, c.memory_dim)
        self.depthwise = nn.Conv2d(c.memory_dim, c.memory_dim, 3, padding=1, groups=c.memory_dim, bias=True)
        self.memory = FourScanVisualMemory(c.memory_dim, c.head_dim, (c.height, c.width))
        self.out_projection = nn.Linear(c.memory_dim, c.dim)
        self.ffn_norm = nn.LayerNorm(c.dim, eps=1e-6)
        self.ffn = nn.Sequential(nn.Linear(c.dim, 4 * c.dim), nn.GELU(), nn.Linear(4 * c.dim, c.dim))
        for layer in (self.projection, self.in_projection, self.depthwise, self.out_projection, self.ffn[0], self.ffn[2]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def get_extra_state(self):
        return {"schema": "hope_cad_v2_visual_block_w3_v1", "config": asdict(self.config)}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("visual block checkpoint configuration/schema mismatch")

    def forward(self, masked_patch_features: torch.Tensor) -> torch.Tensor:
        c = self.config
        x = masked_patch_features
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1:] != (c.height * c.width, c.input_dim):
            raise ValueError(f"expected masked patch features [B,{c.height * c.width},{c.input_dim}]")
        if x.dtype not in (torch.float32, torch.float64) or x.dtype != self.projection.weight.dtype:
            raise ValueError("input and parameters must share float32 or float64 dtype")
        if x.device != self.projection.weight.device:
            raise ValueError("input and parameters must share device")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("non-finite masked patch features")
        with torch.autocast(device_type=x.device.type, enabled=False):
            projected = self.projection(x)
            spatial = self.in_projection(self.input_norm(projected))
            spatial = spatial.transpose(1, 2).reshape(x.shape[0], c.memory_dim, c.height, c.width)
            z = self.depthwise(spatial).flatten(2).transpose(1, 2)
            residual = projected + self.out_projection(self.memory(z))
            output = residual + self.ffn(self.ffn_norm(residual))
            if not bool(torch.isfinite(output).all()):
                raise ValueError("non-finite visual block output")
            return output

    def method_metadata(self) -> dict:
        return {
            "design_id": "hope_cad_v2_masked_teacher_cms",
            "component": "W3_visual_front_end",
            "full_method_implemented": False,
            "canonical_visual_config": self.config.canonical,
            "config": asdict(self.config),
            "heads_per_direction": self.config.memory_dim // self.config.head_dim,
            "directions": 4,
            "working_memory_lifecycle": "reset_per_image_view",
            "recurrence": "chunk_boundary_read_shared_gain_rank_one",
            "guards": "injection_1e-3_spectral_1e-6_nextafter_toward_zero",
            "nextafter_backward": "continuous_pre_rounding_limit",
            "cms_and_anomaly_head": "not_implemented_in_W3",
        }
