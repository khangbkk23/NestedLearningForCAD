"""RIGEL-CAD configuration and fixed constants.

Every constant below is the frozen value used for the paper numbers (locked in the research code under exps/; see
rigel_cad/README.md, "Provenance"). Changing any of them changes the method.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

MiB = 2 ** 20

# ---------------------------------------------------------------- token geometry (frozen encoder contract)
TOKEN_DIM = 768                 # ViT-B feature width
GRID = 28                       # 28 x 28 patch tokens per image
TOKENS_PER_IMAGE = GRID * GRID  # 784
INPUT_SIZE = 392                # DINOv2 input (full field of view, square bilinear resize)

# ---------------------------------------------------------------- persistent-state accounting
KEY_BYTES = 2 * TOKEN_DIM                       # one fp16 key                         = 1,536 B
SLOW_BYTES = TOKEN_DIM * TOKEN_DIM * 4 + 8      # fp32 residual moment S + fp64 count  = 2,359,304 B
KEY4_BYTES = TOKEN_DIM // 2 + 2                 # one int4 key (packed codes + fp16 scale) = 386 B
SLOW_TRI_BYTES = TOKEN_DIM * (TOKEN_DIM + 1) // 2 * 4 + 8   # fp32 upper triangle of S + fp64 count = 1,181,192 B
STORAGES = ('fp16', 'int4_triangular')          # persistent-state formats (rigel_cad/storage.py); fp16 = v1
ENCODER_BYTES = 144_806_400                     # frozen shared encoder, reported separately (not CL state)

# ---------------------------------------------------------------- fast level (episodic keys)
CORESET_RATIO = 0.01            # per-task greedy k-centre keeps ceil(1 % of the task's tokens)
PROJ_DIM = 128                  # seeded Gaussian projection used by greedy k-centre
QUERY_CHUNK = 512               # scoring chunk (rows)
RESIDUAL_CHUNK = 2048           # residual-moment chunk (rows)

# ---------------------------------------------------------------- slow level (residual whitening)
SHRINK = 0.5                    # alpha: W = (shrink(S / n, alpha))^(-1/2)
EIG_FLOOR = 1e-12               # smallest admissible eigenvalue of the shrunk moment

# ---------------------------------------------------------------- A1 readout
SMOOTH_SIGMA = 4
SMOOTH_KSIZE = 2 * int(4 * 4 + .5) + 1          # 33
IMAGE_TOP = 0.01                # image score = mean of the top 1 % of the 392 x 392 map

# ---------------------------------------------------------------- evaluation
AP_BINS = 8192
BOOT_REPLICATES = 2000
BOOT_SEED = 20261120
DEFAULT_SEED = 2711
SEEDS = (2711, 2712, 2713)


@dataclass
class RigelConfig:
    """User-facing configuration of one RIGEL-CAD learner.

    budget_mib     persistent continual-learning state budget in MiB (keys + slow statistic). None = unbounded
                   (every per-task coreset is kept; reference arm "U").
    slow_level     True: two-level RIGEL-CAD (fast keys + slow residual whitening).
                   False: ablation "fast keys only" (Euclidean scoring, the whole budget spent on keys).
    selection      'kcenter' (RIGEL-CAD: task-boundary greedy k-centre consolidation of keys) or
                   'fifo' (ablation: evict the oldest keys first).
    seed           seed of every greedy k-centre call.
    backbone       'dinov2_vitb14_reg4' (default, 392 px) or 'vit_b8_in21k' (matched-backbone ablation, 224 px).
    device         'cuda' or 'cpu' (the frozen results were produced on CUDA).
    storage        'fp16' (default; RIGEL-CAD v1: fp16 keys, full residual moment) or 'int4_triangular' (compact memory:
                   int4 keys with per-key fp16 scale, fp32 upper triangle of the residual moment; more keys in the same
                   budget). 'int4_triangular' requires a finite budget, slow_level=True and selection='kcenter'.
    """
    budget_mib: Optional[float] = 8
    slow_level: bool = True
    selection: str = 'kcenter'
    seed: int = DEFAULT_SEED
    backbone: str = 'dinov2_vitb14_reg4'
    device: str = 'cuda'
    extra: dict = field(default_factory=dict)
    storage: str = 'fp16'

    def __post_init__(self):
        if self.selection not in ('kcenter', 'fifo'):
            raise ValueError(f"selection must be 'kcenter' or 'fifo', got {self.selection!r}")
        if self.budget_mib is not None and self.budget_mib <= 0:
            raise ValueError('budget_mib must be positive or None')
        if self.storage not in STORAGES:
            raise ValueError(f'storage must be one of {STORAGES}, got {self.storage!r}')
        if self.storage == 'int4_triangular' and (self.budget_mib is None or not self.slow_level
                                                  or self.selection != 'kcenter'):
            raise ValueError("storage='int4_triangular' is defined for a finite budget with slow_level=True and "
                             "selection='kcenter' only")

    @property
    def budget_bytes(self) -> Optional[int]:
        return None if self.budget_mib is None else int(round(self.budget_mib * MiB))

    @property
    def key_bytes(self) -> int:
        """Persistent bytes of one key in the configured storage format."""
        return KEY4_BYTES if self.storage == 'int4_triangular' else KEY_BYTES

    @property
    def slow_bytes(self) -> int:
        if not self.slow_level:
            return 0
        return SLOW_TRI_BYTES if self.storage == 'int4_triangular' else SLOW_BYTES

    @property
    def key_capacity(self) -> Optional[int]:
        """Largest number of keys (in the storage format) such that keys + slow statistic fit the budget."""
        b = self.budget_bytes
        if b is None:
            return None
        cap = (b - self.slow_bytes) // self.key_bytes
        if cap <= 0:
            raise ValueError(f'budget {b} B cannot hold the slow statistic ({self.slow_bytes} B) and one key')
        return cap

    def to_dict(self) -> dict:
        d = asdict(self)
        if d['storage'] == 'fp16':
            del d['storage']        # fp16 configs keep the RIGEL-CAD 1.0 layout (and stay readable by 1.0)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'RigelConfig':
        return cls(**d)
