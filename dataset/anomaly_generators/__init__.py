"""
Anomaly Generator Registry
==========================
Each generator implements the interface: AnomalyGeneratorBase
    .generate(img_np, category) -> (result_img_np, mask_np, has_anomaly: bool)

Usage in the active YAML config:
    dataset:
      anomaly_generator: "superpixel"   # or "perlin", "destseg", "realnet", "mixed"
"""

from .base import AnomalyGeneratorBase
from .superpixel import SuperpixelAnomalyGenerator
from .perlin import PerlinAnomalyGenerator
from .destseg import DeSTSegAnomalyGenerator
from .realnet import RealnetAnomalyGenerator
from .mixed import MixedAnomalyGenerator

_REGISTRY = {
    "superpixel": SuperpixelAnomalyGenerator,
    "perlin":     PerlinAnomalyGenerator,
    "destseg":    DeSTSegAnomalyGenerator,
    "realnet":    RealnetAnomalyGenerator,
    "mixed":      MixedAnomalyGenerator,
}


def build_anomaly_generator(cfg: dict) -> AnomalyGeneratorBase:
    name = cfg.get("anomaly_generator", "superpixel").lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown anomaly_generator '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](cfg)


__all__ = [
    "AnomalyGeneratorBase",
    "SuperpixelAnomalyGenerator",
    "PerlinAnomalyGenerator",
    "DeSTSegAnomalyGenerator",
    "RealnetAnomalyGenerator",
    "MixedAnomalyGenerator",
    "build_anomaly_generator",
]
