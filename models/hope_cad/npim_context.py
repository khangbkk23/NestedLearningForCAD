# models/hope_cad/npim_context.py
"""Leakage-safe pre-attention context for NPIM-CAD (EXP-NPIM-00).

This module is deliberately independent of the existing HOPE-CAD, ECAD, CCNM
and ContCore implementations.  It adds:

* :class:`NPIMFeatureExtractor` -- a frozen ViT-B/8 pass that captures BOTH
  ``Z`` (pre-attention patch embeddings at the input of ``blocks[0]``) and
  ``Y`` (block-8 patch tokens used by the CADIC/P1 readout).  The extractor
  refuses to load unless the checkpoint SHA-256 matches the caller's pin and
  the architecture satisfies the patch-independence preconditions.
* Context/key construction from ``Z`` ONLY.  ``Y`` never enters the predictor
  input, and the target patch's own ``Z[p]`` is never included in ``key_p``.

Mathematical contract (handoff sections 3.2 and 8.1)
----------------------------------------------------
For target patch ``p`` on the 28x28 grid::

    zbar_N,p   = mean_{q in N4(p)} Z[q]                 (no padding copy of p)
    zbar_G,!p  = sum_{q != p} Z[q] / 783
    u_p        = 0.5 * zbar_N,p + 0.5 * zbar_G,!p + gamma * e(p)
    k_p        = normalize(P u_p, eps)                  P in R^{64x768}

``gamma`` defaults to ``0.0`` for EXP-NPIM-00 (coordinate-only ablation is a
separate, later experiment).  The global leave-one-out mean is computed with
prefix/suffix cumulative sums that never contain ``Z[p]``; the FP32
``sum(Z) - Z[p]`` shortcut is *not* used.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 28
PATCHES = GRID * GRID
IMAGE_SIZE = 224
PATCH_SIZE = 8
EMBED_DIM = 768
BLOCK_INDEX = 8  # zero-based: one-based layer 9, matching the CADIC P1 readout
MODEL_NAME = "vit_base_patch8_224"


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file, streamed (never load a 343 MB checkpoint into RAM twice)."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    """Shape/dtype-sensitive SHA-256 of a tensor's logical bytes (CPU, contiguous)."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("sha256_tensor expects a torch.Tensor")
    payload = tensor.detach().to("cpu", copy=True).contiguous()
    header = f"{tuple(payload.shape)}|{payload.dtype}|".encode()
    return hashlib.sha256(header + payload.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class NPIMContextConfig:
    """Frozen representation/context configuration for EXP-NPIM-00."""

    checkpoint_path: str
    checkpoint_sha256: str
    image_size: int = IMAGE_SIZE
    patch_size: int = PATCH_SIZE
    embed_dim: int = EMBED_DIM
    grid: int = GRID
    block_index: int = BLOCK_INDEX
    key_dim: int = 64
    projection_seed: int = 20261021
    gamma: float = 0.0
    eps: float = 1e-8
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def validate(self) -> None:
        if self.image_size != IMAGE_SIZE or self.patch_size != PATCH_SIZE:
            raise ValueError("NPIM-00 requires 224px / 8px patches")
        if self.embed_dim != EMBED_DIM or self.grid != GRID:
            raise ValueError("NPIM-00 requires a 768-d, 28x28 token grid")
        if not 0 <= self.block_index < 12:
            raise ValueError("block_index must be in 0..11")
        if self.key_dim <= 0:
            raise ValueError("key_dim must be positive")
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative")
        if not 0.0 < self.eps < 1.0:
            raise ValueError("eps must be a small positive clamp")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("three pinned channel means/stds are required")


# ---------------------------------------------------------------------------
# Context/key construction (Z only)
# ---------------------------------------------------------------------------


def neighbour_table(grid: int = GRID) -> List[List[int]]:
    """Valid up/down/left/right patch indices per position; never includes self.

    There is no padding rule that copies the centre patch at borders: corners
    have two valid neighbours and edges three.
    """

    table: List[List[int]] = []
    for row in range(grid):
        for col in range(grid):
            neighbours: List[int] = []
            if row > 0:
                neighbours.append((row - 1) * grid + col)
            if row + 1 < grid:
                neighbours.append((row + 1) * grid + col)
            if col > 0:
                neighbours.append(row * grid + (col - 1))
            if col + 1 < grid:
                neighbours.append(row * grid + (col + 1))
            if (row * grid + col) in neighbours:
                raise AssertionError("neighbour table accidentally contains self")
            table.append(neighbours)
    return table


def neighbour_mean(Z: torch.Tensor, grid: int = GRID) -> torch.Tensor:
    """Mean of the valid 4-neighbourhood of every patch, excluding the patch itself."""

    _check_Z(Z, grid)
    channels = Z.shape[1]
    Zg = Z.reshape(grid, grid, channels)
    padded = F.pad(Zg, (0, 0, 1, 1, 1, 1))  # zero padding, shifted slices below
    up = padded[0:grid, 1 : grid + 1]
    down = padded[2 : grid + 2, 1 : grid + 1]
    left = padded[1 : grid + 1, 0:grid]
    right = padded[1 : grid + 1, 2 : grid + 2]
    total = up + down + left + right
    rows = torch.arange(grid, device=Z.device).view(grid, 1)
    cols = torch.arange(grid, device=Z.device).view(1, grid)
    count = (
        (rows > 0).to(Z.dtype)
        + (rows + 1 < grid).to(Z.dtype)
        + (cols > 0).to(Z.dtype)
        + (cols + 1 < grid).to(Z.dtype)
    ).view(grid, grid, 1)
    if bool((count < 1).any()):
        raise AssertionError("every patch must have at least one valid neighbour")
    return (total / count).reshape(grid * grid, channels)


def global_leave_one_out(Z: torch.Tensor) -> torch.Tensor:
    """Mean of all patches except each target patch, with NO ``Z[p]`` term.

    Prefix sums cover ``q < p`` and suffix sums cover ``q > p``; neither
    accumulator ever includes ``Z[p]``.  This is the vectorised form of an
    explicit masked reduction and avoids the FP32 ``sum(Z) - Z[p]`` shortcut.
    Accumulation is FP64 and the result is cast back to ``Z.dtype``.
    """

    _check_Z(Z)
    patches = Z.shape[0]
    if patches != PATCHES:
        raise ValueError(f"expected {PATCHES} patches, got {patches}")
    Zd = Z.to(torch.float64)
    prefix = torch.zeros_like(Zd)
    if patches > 1:
        prefix[1:] = torch.cumsum(Zd[:-1], dim=0)
    reverse = torch.flip(Zd, dims=[0])
    reverse_cumsum = torch.cumsum(reverse, dim=0)
    # R[p] = sum_{q >= p} Z[q]; suffix[p] = R[p + 1] = sum_{q > p} Z[q].
    suffix = torch.zeros_like(Zd)
    suffix[:-1] = torch.flip(reverse_cumsum, dims=[0])[1:]
    return ((prefix + suffix) / (patches - 1)).to(Z.dtype)


def positional_coordinates(grid: int = GRID, embed_dim: int = EMBED_DIM) -> torch.Tensor:
    """Fixed coordinate function ``e(p)`` of length ``embed_dim``.

    Dimensions 0/1 hold centred, scaled row/column coordinates; the rest are
    zero.  ``gamma`` is 0 in EXP-NPIM-00, so this tensor is inert unless a
    future, separately-registered ablation enables it.
    """

    rows, cols = torch.meshgrid(
        torch.arange(grid, dtype=torch.float32),
        torch.arange(grid, dtype=torch.float32),
        indexing="ij",
    )
    e = torch.zeros(grid * grid, embed_dim, dtype=torch.float32)
    scale = float(max(grid - 1, 1))
    e[:, 0] = (rows.reshape(-1) - (grid - 1) / 2.0) / scale
    e[:, 1] = (cols.reshape(-1) - (grid - 1) / 2.0) / scale
    return e


def make_projection(key_dim: int, embed_dim: int, seed: int) -> torch.Tensor:
    """Deterministic, row-unit-norm FP32 projection ``P in R^{key_dim x embed_dim}``."""

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    projection = torch.randn(key_dim, embed_dim, generator=generator, dtype=torch.float32)
    norms = projection.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return projection / norms


def normalize_rows(x: torch.Tensor, eps: float) -> torch.Tensor:
    """L2-normalise each row with an epsilon clamp (zero rows stay exactly zero)."""

    if not 0.0 < eps < 1.0:
        raise ValueError("eps must be a small positive clamp")
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def build_context(
    Z: torch.Tensor,
    *,
    gamma: float = 0.0,
    coordinates: torch.Tensor | None = None,
    grid: int = GRID,
) -> torch.Tensor:
    """``u_p = 0.5*neighbour_mean + 0.5*global_leave_one_out + gamma*e(p)``."""

    _check_Z(Z, grid)
    local = neighbour_mean(Z, grid)
    global_loo = global_leave_one_out(Z)
    context = 0.5 * local + 0.5 * global_loo
    if gamma != 0.0:
        if coordinates is None:
            coordinates = positional_coordinates(grid, Z.shape[1])
        if coordinates.shape != Z.shape:
            raise ValueError("coordinate tensor geometry mismatch")
        context = context + gamma * coordinates.to(device=Z.device, dtype=Z.dtype)
    return context


def build_keys(
    Z: torch.Tensor,
    projection: torch.Tensor,
    *,
    gamma: float = 0.0,
    coordinates: torch.Tensor | None = None,
    eps: float = 1e-8,
    grid: int = GRID,
) -> torch.Tensor:
    """Vectorised context keys ``k_p = normalize(P u_p, eps)`` from ``Z`` only."""

    if projection.ndim != 2 or projection.shape[1] != Z.shape[1]:
        raise ValueError("projection geometry mismatch")
    context = build_context(Z, gamma=gamma, coordinates=coordinates, grid=grid)
    logits = context @ projection.T.to(device=Z.device, dtype=Z.dtype)
    return normalize_rows(logits, eps)


def build_keys_loop(
    Z: torch.Tensor,
    projection: torch.Tensor,
    *,
    gamma: float = 0.0,
    coordinates: torch.Tensor | None = None,
    eps: float = 1e-8,
    grid: int = GRID,
) -> torch.Tensor:
    """Independent loop oracle: explicit gather/sum of neighbours and ``q != p``.

    Uses no cumulative-sum identity.  Slow by design (784 Python iterations) and
    intended only for correctness tests on synthetic fixtures.
    """

    _check_Z(Z, grid)
    table = neighbour_table(grid)
    if gamma != 0.0 and coordinates is None:
        coordinates = positional_coordinates(grid, Z.shape[1])
    contexts: List[torch.Tensor] = []
    for p in range(grid * grid):
        neighbours = torch.tensor(table[p], dtype=torch.long, device=Z.device)
        local = Z.index_select(0, neighbours).sum(dim=0) / float(len(table[p]))
        mask = torch.arange(grid * grid, device=Z.device) != p
        loo = Z[mask].sum(dim=0) / float(grid * grid - 1)
        context = 0.5 * local + 0.5 * loo
        if gamma != 0.0:
            context = context + gamma * coordinates[p].to(device=Z.device, dtype=Z.dtype)
        contexts.append(context)
    logits = torch.stack(contexts, dim=0) @ projection.T.to(device=Z.device, dtype=Z.dtype)
    return normalize_rows(logits, eps)


def _check_Z(Z: torch.Tensor, grid: int = GRID) -> None:
    if not isinstance(Z, torch.Tensor):
        raise TypeError("Z must be a torch.Tensor")
    if Z.ndim != 2 or Z.shape[0] != grid * grid or Z.shape[1] != EMBED_DIM:
        raise ValueError(f"expected Z of shape [{grid * grid},{EMBED_DIM}], got {tuple(Z.shape)}")
    if not torch.isfinite(Z).all():
        raise ValueError("non-finite pre-attention context")


# ---------------------------------------------------------------------------
# Frozen extractor with verified hook identity
# ---------------------------------------------------------------------------


class NPIMFeatureExtractor(nn.Module):
    """Frozen ViT-B/8 encoder exposing ``Z`` (context-only) and ``Y`` (target-only).

    ``Z`` is captured at a forward-pre-hook on ``blocks[0]``: the tensor is
    ``[CLS, patch tokens]`` after non-overlapping 8x8 patch embedding, fixed
    per-token positional embedding and the (p=0) dropout, and BEFORE any
    cross-token attention.  ``Y`` is the output of block index 8, matching the
    existing CADIC adapter (one-based layer 9, final norm, CLS removed).
    """

    def __init__(self, config: NPIMContextConfig, device: str | torch.device = "cpu") -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.device = torch.device(device)

        checkpoint = Path(config.checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"frozen ViT-B/8 checkpoint not found: {checkpoint}")
        digest = sha256_file(checkpoint)
        if digest != config.checkpoint_sha256:
            raise ValueError(
                "checkpoint SHA-256 does not match the pinned identity: "
                f"expected {config.checkpoint_sha256}, got {digest}"
            )
        self.checkpoint_sha256 = digest

        try:
            import timm
        except Exception as exc:  # pragma: no cover - dependency-specific
            raise RuntimeError("NPIM requires timm to instantiate ViT-B/8") from exc

        self.model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0, img_size=config.image_size)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        if not isinstance(state, dict):
            raise ValueError("checkpoint must contain a state-dict mapping")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.architecture_manifest = self._verify_architecture()

        self._z: torch.Tensor | None = None
        self._y: torch.Tensor | None = None
        self._hook_z = self.model.blocks[0].register_forward_pre_hook(self._capture_z)
        self._hook_y = self.model.blocks[config.block_index].register_forward_hook(self._capture_y)

    # -- architecture preconditions -----------------------------------------
    def _verify_architecture(self) -> Dict[str, Any]:
        model = self.model
        patch_embed = model.patch_embed
        proj = patch_embed.proj
        checks = {
            "patch_kernel": tuple(proj.kernel_size) == (PATCH_SIZE, PATCH_SIZE),
            "patch_stride": tuple(proj.stride) == (PATCH_SIZE, PATCH_SIZE),
            "patch_padding": tuple(proj.padding) == (0, 0),
            "patch_dilation": tuple(proj.dilation) == (1, 1),
            "patch_groups": proj.groups == 1,
            "patch_norm_identity": isinstance(patch_embed.norm, nn.Identity),
            "pos_embed_fixed": model.pos_embed is not None
            and tuple(model.pos_embed.shape) == (1, PATCHES + 1, EMBED_DIM),
            "pos_drop_zero": float(getattr(model.pos_drop, "p", 1.0)) == 0.0,
            "patch_drop_identity": isinstance(getattr(model, "patch_drop", None), nn.Identity)
            or float(getattr(model.patch_drop, "p", 1.0)) == 0.0,
            "norm_pre_identity": isinstance(getattr(model, "norm_pre", None), nn.Identity),
            "twelve_blocks": len(model.blocks) == 12,
            "block_index_valid": 0 <= self.config.block_index < len(model.blocks),
            "no_global_norm_before_z": isinstance(getattr(model, "norm_pre", None), nn.Identity),
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise RuntimeError(
                "pre-attention patch independence is NOT verified; failing checks: "
                + ", ".join(failed)
            )
        return {
            "model_name": MODEL_NAME,
            "checkpoint_path": str(Path(self.config.checkpoint_path).resolve()),
            "checkpoint_sha256": self.checkpoint_sha256,
            "hook_z": "blocks[0].forward_pre_hook input tokens[:,1:,:] "
                      "(non-overlapping 8x8 patch embed + fixed per-token pos embed + p=0 dropout)",
            "hook_y": f"blocks[{self.config.block_index}].forward_hook output tokens[:,1:,:]",
            "patch_embed": "Conv2d(3,768,kernel=8,stride=8,pad=0,dilation=1,groups=1), norm=Identity",
            "pos_embed": "fixed per-token parameter, shape [1,785,768]",
            "pos_drop": "p=0",
            "patch_drop": "Identity" if isinstance(getattr(model, "patch_drop", None), nn.Identity) else "p=0",
            "norm_pre": "Identity",
            "checks": checks,
            "patch_grid": [GRID, GRID],
            "image_size": self.config.image_size,
            "preprocessing": {
                "resize": "PIL direct square bilinear to 224x224",
                "mean": list(self.config.mean),
                "std": list(self.config.std),
                "normalization": "fixed dataset constants (no per-image statistics)",
            },
            "block_index_zero_based": self.config.block_index,
            "layer_number_one_based": self.config.block_index + 1,
        }

    # -- hooks ---------------------------------------------------------------
    def _capture_z(self, _module: nn.Module, args: Sequence[Any]) -> None:
        self._z = args[0].detach()

    def _capture_y(self, _module: nn.Module, _args: Sequence[Any], output: Any) -> None:
        if isinstance(output, (tuple, list)):
            output = output[0]
        self._y = output.detach()

    # -- forward -------------------------------------------------------------
    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(Z, Y)`` each ``[B, 784, 768]`` for ``images`` ``[B, 3, 224, 224]``."""

        if images.ndim != 4 or images.shape[1] != 3 or tuple(images.shape[-2:]) != (
            self.config.image_size,
            self.config.image_size,
        ):
            raise ValueError(f"expected [B,3,224,224], got {tuple(images.shape)}")
        self._z = None
        self._y = None
        _ = self.model(images.to(self.device))
        z, y = self._z, self._y
        self._z = None
        self._y = None
        if z is None or y is None:
            raise RuntimeError("pre-attention or block-8 hook produced no tokens")
        if z.ndim != 3 or z.shape[1] != PATCHES + 1 or z.shape[2] != EMBED_DIM:
            raise RuntimeError(f"Z geometry assertion failed: {tuple(z.shape)}")
        if y.ndim != 3 or y.shape[1] != PATCHES + 1 or y.shape[2] != EMBED_DIM:
            raise RuntimeError(f"Y geometry assertion failed: {tuple(y.shape)}")
        z = z[:, 1:, :].detach()
        y = y[:, 1:, :].detach()
        if not torch.isfinite(z).all() or not torch.isfinite(y).all():
            raise RuntimeError("non-finite Z or Y")
        return z, y

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(images)

    def close(self) -> None:
        for name in ("_hook_z", "_hook_y"):
            hook = getattr(self, name, None)
            if hook is not None:
                hook.remove()
                setattr(self, name, None)

    def __del__(self):  # pragma: no cover - interpreter shutdown
        try:
            self.close()
        except Exception:
            pass
