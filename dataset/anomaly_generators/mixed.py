"""
MixedAnomalyGenerator
======================
Meta-generator: randomly selects one of the registered generators
each time generate() is called.

Useful for training with maximum anomaly diversity — the model sees
Superpixel, Perlin (DRAEM), DeSTSeg, and Realnet anomalies within the
same epoch.

Config keys:
    mixed_generators : list of generator names + optional weights
                       (default: all four equally weighted)
    e.g.
        mixed_generators:
          - [superpixel, 1]
          - [perlin,     2]
          - [destseg,    1]
          - [realnet,    2]

    All other generator-specific keys (dtd_dir, perlin_scale, ...) are
    passed through to the sub-generators as usual.
"""

import numpy as np

from .base import AnomalyGeneratorBase


class MixedAnomalyGenerator(AnomalyGeneratorBase):
    """Randomly picks one sub-generator per sample."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)

        # Import here to avoid circular import
        from .superpixel import SuperpixelAnomalyGenerator
        from .perlin     import PerlinAnomalyGenerator
        from .destseg    import DeSTSegAnomalyGenerator
        from .realnet    import RealnetAnomalyGenerator

        _cls_map = {
            "superpixel": SuperpixelAnomalyGenerator,
            "perlin":     PerlinAnomalyGenerator,
            "destseg":    DeSTSegAnomalyGenerator,
            "realnet":    RealnetAnomalyGenerator,
        }

        spec = cfg.get("mixed_generators", [
            ["superpixel", 1],
            ["perlin",     1],
            ["destseg",    1],
            ["realnet",    1],
        ])

        self._generators = []
        weights = []
        for entry in spec:
            name, w = entry[0], entry[1] if len(entry) > 1 else 1
            if name not in _cls_map:
                raise ValueError(f"[Mixed] Unknown generator '{name}'.")
            self._generators.append(_cls_map[name](cfg))
            weights.append(float(w))

        total = sum(weights)
        self._probs = [w / total for w in weights]

    def generate(self, img_np: np.ndarray, category: str):
        idx = np.random.choice(len(self._generators), p=self._probs)
        return self._generators[idx].generate(img_np, category)
