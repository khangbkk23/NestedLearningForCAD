"""Frozen four-colour 2x2-patch mask family; mask RGB before the ViT."""
import torch


def patch_masks(height=28, width=28, *, device=None):
    rows = torch.arange(height, device=device)[:, None]
    cols = torch.arange(width, device=device)[None, :]
    colours = (2 * ((rows // 2) % 2) + (cols // 2) % 2).flatten()
    return torch.stack([colours == k for k in range(4)])


def mask_normalized_rgb(image, view):
    if image.shape[-3:] != (3, 224, 224) or type(view) is not int or not 0 <= view < 4:
        raise ValueError("expected normalized RGB224 and view in [0,3]")
    pixels = patch_masks(device=image.device)[view].reshape(28, 28).repeat_interleave(8, 0).repeat_interleave(8, 1)
    # Mean fill before normalization is exactly zero in normalized RGB.
    return image.masked_fill(pixels, 0)
