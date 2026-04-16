"""
Every subclass must implement:
    generate(img_np, category) -> (result_img_np, mask_np, has_anomaly)

Parameters
----------
img_np      : np.ndarray  shape (H, W, 3), dtype float32, range [0, 255]
category    : str         e.g. "bottle", "capsule"

Returns
-------
result_img_np : np.ndarray  same shape as img_np, dtype float32, range [0, 255]
mask_np       : np.ndarray  shape (H, W), dtype float32, values in {0, 1}
has_anomaly   : bool        True if a real anomaly was placed
"""

from abc import ABC, abstractmethod
import numpy as np


class AnomalyGeneratorBase(ABC):

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def generate(
        self,
        img_np: np.ndarray,       # (H, W, 3), float32, [0-255]
        category: str,
    ):
        """
        Returns
        -------
        result_img_np : np.ndarray  (H, W, 3), float32
        mask_np       : np.ndarray  (H, W),    float32 ∈ {0.0, 1.0}
        has_anomaly   : bool
        """
        ...
