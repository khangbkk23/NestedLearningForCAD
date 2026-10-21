# exps/npim_cad/test_context_isolation.py
"""G1/G2 oracles: target-patch leakage, geometry and block-8 parity.

Synthetic tests need no checkpoint.  Real-ViT tests are skipped with an explicit
reason when the frozen checkpoint is absent; they are never reported as PASS by
inspection.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from dataset.benchmark_protocol_v1 import _ImageDataset
from models.hope_cad.npim_context import (
    GRID,
    NPIMContextConfig,
    NPIMFeatureExtractor,
    build_keys,
    build_keys_loop,
    global_leave_one_out,
    make_projection,
    neighbour_mean,
    neighbour_table,
    positional_coordinates,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
CP = ROOT / "checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth"
CP_SHA = "755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415"
MVTEC = ROOT / "data/mvtec"
CACHE = ROOT / "results/hope_cad/frozen_features_v1/clean"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

requires_vit = pytest.mark.skipif(
    not CP.is_file() or sha256_file(CP) != CP_SHA,
    reason="frozen ViT-B/8 checkpoint absent or not the pinned identity",
)


def random_Z(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(784, 768, generator=generator)


def image_tensor(relative_path: str = "bottle/train/good/000.png") -> torch.Tensor:
    row = {"relative_path": relative_path, "label": 0}
    dataset = _ImageDataset(MVTEC.resolve(), [row], 224, MEAN, STD, training=True)
    return dataset[0]["images"][None]


@pytest.fixture(scope="module")
def extractor() -> NPIMFeatureExtractor:
    cfg = NPIMContextConfig(checkpoint_path=str(CP), checkpoint_sha256=CP_SHA)
    instance = NPIMFeatureExtractor(cfg, device="cpu")
    yield instance
    instance.close()


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def test_neighbour_table_borders_never_reuse_self():
    table = neighbour_table(GRID)
    assert len(table) == 784
    assert len(table[0]) == 2  # corner
    assert len(table[13]) == 3  # top edge
    assert len(table[13 * 28]) == 3  # left edge
    assert len(table[14 * 28 + 14]) == 4  # interior
    for p, neighbours in enumerate(table):
        assert p not in neighbours
        assert len(neighbours) == len(set(neighbours))
        for q in neighbours:
            assert abs(q // 28 - p // 28) + abs(q % 28 - p % 28) == 1


def test_neighbour_mean_uses_other_patches_only():
    Z = random_Z(1)
    local = neighbour_mean(Z)
    p = 14 * 28 + 14
    neighbours = neighbour_table()[p]
    expected = Z[neighbours].mean(dim=0)
    assert torch.allclose(local[p], expected, atol=1e-6)
    modified = Z.clone()
    modified[p] = 1.0e6
    assert torch.equal(neighbour_mean(modified)[p], local[p])


def test_global_leave_one_out_excludes_target_exactly():
    Z = random_Z(2)
    p = 200
    loo = global_leave_one_out(Z)
    modified = Z.clone()
    modified[p] = 1.0e6
    assert torch.equal(global_leave_one_out(modified)[p], loo[p])
    # The target patch is legitimate context for the *other* patches.
    changed = (global_leave_one_out(modified) - loo).abs().max(dim=1).values
    assert changed[p].item() == 0.0
    assert changed[torch.arange(784) != p].max().item() > 0.0


def test_target_key_independent_of_target_patch_context():
    torch.manual_seed(0)
    projection = make_projection(64, 768, 20261021)
    Z = random_Z(3)
    keys = build_keys(Z, projection)
    modified = Z.clone()
    p = 14 * 28 + 14
    modified[p] = Z[p] + 3.0
    modified_keys = build_keys(modified, projection)
    assert torch.equal(modified_keys[p], keys[p])
    # Neighbour keys legitimately change because p is their context.
    assert (modified_keys[neighbour_table()[p]] - keys[neighbour_table()[p]]).abs().max().item() > 0.0


def test_vectorised_context_matches_loop_oracle():
    projection = make_projection(64, 768, 20261021)
    Z = random_Z(4)
    vectorised = build_keys(Z, projection)
    loop = build_keys_loop(Z, projection)
    assert (vectorised - loop).abs().max().item() <= 1e-5


def test_zero_context_gives_finite_zero_keys():
    projection = make_projection(64, 768, 20261021)
    keys = build_keys(torch.zeros(784, 768), projection)
    assert torch.isfinite(keys).all()
    assert torch.equal(keys, torch.zeros_like(keys))


def test_projection_is_deterministic_and_row_normalised():
    left = make_projection(64, 768, 20261021)
    right = make_projection(64, 768, 20261021)
    assert torch.equal(left, right)
    assert torch.allclose(left.norm(dim=1), torch.ones(64), atol=1e-6)


def test_positional_coordinates_are_inert_at_gamma_zero():
    projection = make_projection(64, 768, 20261021)
    Z = random_Z(5)
    coordinates = positional_coordinates()
    without = build_keys(Z, projection)
    with_inert = build_keys(Z, projection, gamma=0.0, coordinates=coordinates)
    assert torch.equal(without, with_inert)
    enabled = build_keys(Z, projection, gamma=0.25, coordinates=coordinates)
    assert (enabled - without).abs().max().item() > 0.0


def test_synthetic_nonmixing_patchifier_is_target_isolated():
    """A stride-8 non-overlapping patchifier cannot leak across patch borders."""

    generator = torch.Generator().manual_seed(7)
    patch_matrix = torch.randn(768, 8 * 8 * 3, generator=generator)
    image = torch.randn(3, 224, 224, generator=generator)
    patches = image.reshape(3, 28, 8, 28, 8).permute(1, 3, 0, 2, 4).reshape(784, 3 * 64)
    Z = patches @ patch_matrix.T
    projection = make_projection(64, 768, 20261021)
    keys = build_keys(Z, projection)
    p = 14 * 28 + 14
    row, col = divmod(p, 28)
    perturbed = image.clone()
    perturbed[:, row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8] += 0.5
    patches_2 = perturbed.reshape(3, 28, 8, 28, 8).permute(1, 3, 0, 2, 4).reshape(784, 3 * 64)
    Z2 = patches_2 @ patch_matrix.T
    assert torch.equal(Z2[torch.arange(784) != p], Z[torch.arange(784) != p])
    assert not torch.equal(Z2[p], Z[p])
    assert torch.equal(build_keys(Z2, projection)[p], keys[p])


# ---------------------------------------------------------------------------
# Real frozen ViT
# ---------------------------------------------------------------------------


@requires_vit
def test_real_architecture_preconditions(extractor: NPIMFeatureExtractor):
    manifest = extractor.architecture_manifest
    assert all(manifest["checks"].values())
    assert manifest["checkpoint_sha256"] == CP_SHA
    assert manifest["patch_embed"].startswith("Conv2d(3,768,kernel=8,stride=8,pad=0")
    image = image_tensor()
    z, y = extractor.encode(image)
    assert tuple(z.shape) == (1, 784, 768)
    assert tuple(y.shape) == (1, 784, 768)
    assert torch.isfinite(z).all() and torch.isfinite(y).all()


@requires_vit
def test_real_block8_Y_parity_to_frozen_cache(extractor: NPIMFeatureExtractor):
    cache = torch.load(CACHE / "bottle_train.pt", mmap=True, weights_only=True)
    index = 0
    relative = cache["relative_paths"][index]
    assert sha256_file(MVTEC / relative) == cache["per_image_source_sha256"][index]
    _, y = extractor.encode(image_tensor(relative))
    reference = cache["patches"][index].to(torch.float32)
    assert tuple(y[0].shape) == tuple(reference.shape)
    difference = (y[0] - reference).abs().max().item()
    relative_l2 = ((y[0] - reference).norm() / reference.norm()).item()
    scaled_max = difference / reference.abs().max().item()
    # The immutable cache marker pins relative-L2 <= 3.0517578125e-05 and
    # scaled max-abs <= 1.220703125e-04.  The cache was written with timm
    # 1.0.26; this run uses 1.0.30, so bitwise equality is not required.
    assert relative_l2 <= 3.0517578125e-05, f"Y parity rel_l2={relative_l2}"
    assert scaled_max <= 1.220703125e-04, f"Y parity scaled_max={scaled_max}"


@requires_vit
@pytest.mark.parametrize("target_patch", [14 * 28 + 14, 0, 13 * 28])
def test_real_target_patch_perturbation_leaves_key_unchanged(
    extractor: NPIMFeatureExtractor, target_patch: int
):
    projection = make_projection(64, 768, 20261021)
    image = image_tensor()
    row, col = divmod(target_patch, 28)
    perturbed = image.clone()
    perturbed[0, :, row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8] += 0.5

    z0, y0 = extractor.encode(image)
    z1, y1 = extractor.encode(perturbed)
    keys0 = build_keys(z0[0], projection)
    keys1 = build_keys(z1[0], projection)

    other = torch.arange(784) != target_patch
    assert torch.equal(keys1[target_patch], keys0[target_patch]), "target key moved"
    assert torch.equal(z1[0][other], z0[0][other]), "a non-target pre-attention token moved"
    # Diagnostic: deep block-8 tokens ARE globally contaminated.
    assert (y1[0][other] - y0[0][other]).abs().max().item() > 0.0


@requires_vit
def test_real_autograd_target_patch_gradient_is_zero(extractor: NPIMFeatureExtractor):
    projection = make_projection(64, 768, 20261021)
    image = image_tensor()
    target_patch = 14 * 28 + 14
    model = extractor.model
    captured = {}

    def capture_pre(_module, args):
        captured["z"] = args[0]

    hook = model.blocks[0].register_forward_pre_hook(capture_pre)
    differentiable = image.clone().detach().requires_grad_(True)
    try:
        with torch.enable_grad():
            _ = model(differentiable)
            keys = build_keys(captured["z"][:, 1:, :][0], projection)
            loss = keys[target_patch].sum()
            gradient, = torch.autograd.grad(loss, differentiable)
    finally:
        hook.remove()
    row, col = divmod(target_patch, 28)
    target = gradient[0, :, row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8]
    assert target.abs().max().item() == 0.0
    outside = gradient.clone()
    outside[:, row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8] = 0.0
    assert outside.abs().max().item() > 0.0


@requires_vit
def test_real_deep_features_are_not_leakage_free_context(extractor: NPIMFeatureExtractor):
    """Negative diagnostic: block-8 neighbour/CLS tokens carry target-patch information."""

    image = image_tensor()
    perturbed = image.clone()
    perturbed[0, :, 14 * 8 : 15 * 8, 14 * 8 : 15 * 8] += 0.5
    z0, y0 = extractor.encode(image)
    z1, y1 = extractor.encode(perturbed)
    assert not torch.equal(z1[0], z0[0])  # only the target Z moved
    other = torch.arange(784) != 14 * 28 + 14
    contamination = (y1[0][other] - y0[0][other]).abs().max().item()
    assert contamination > 0.0
    # A predictor fed those deep tokens would see target information.
    assert (z1[0][other] - z0[0][other]).abs().max().item() == 0.0
