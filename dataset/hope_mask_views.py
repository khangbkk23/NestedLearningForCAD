"""Frozen four-colour 2x2-patch mask family; mask RGB before the ViT."""
import torch


def patch_masks(height=28, width=28, *, device=None):
    rows = torch.arange(height, device=device)[:, None]
    cols = torch.arange(width, device=device)[None, :]
    colours = (2 * ((rows // 2) % 2) + (cols // 2) % 2).flatten()
    return torch.stack([colours == k for k in range(4)])


def mask_normalized_rgb(image, view, *, height=28, width=28, patch_size=8):
    if (image.ndim < 3 or image.shape[-3] != 3
            or tuple(image.shape[-2:]) != (height * patch_size, width * patch_size)
            or type(view) is not int or not 0 <= view < 4):
        raise ValueError("image geometry must match the configured patch grid")
    pixels = patch_masks(height, width, device=image.device)[view].reshape(height, width)
    pixels = pixels.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)
    # Mean fill before normalization is exactly zero in normalized RGB.
    return image.masked_fill(pixels, 0)
