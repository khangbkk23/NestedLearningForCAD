"""Normal-only teacher objective; RGB feature extraction belongs to W5."""
import torch
from torch import nn
from .cms import init_affine


class TeacherDecoder(nn.Sequential):
    def __init__(self, dim=256, target_dim=768):
        super().__init__(nn.Linear(dim, target_dim), nn.GELU(), nn.Linear(target_dim, target_dim))
        init_affine(self)


def frozen_teacher(raw_clean_features):
    with torch.no_grad():
        return raw_clean_features.detach() / raw_clean_features.detach().norm(dim=-1, keepdim=True).clamp_min(1e-8)
