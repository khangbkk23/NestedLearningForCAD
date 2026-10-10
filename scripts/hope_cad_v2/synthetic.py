"""Clearly marked resource/correctness fixtures. Never a real-feature fallback."""
import torch
from dataset.benchmark_manifest_v1 import canonical_digest
from dataset.benchmark_protocol_v2 import ROLES
from dataset.hope_mask_views import patch_masks
from models.hope_cad_v2.teacher_head import frozen_teacher
from training.hope_outer_learning_v2 import PrefixFeatures


def feature_fixture(config, device='cpu', seed=103):
    roles, offset = {}, 0
    for role, count in zip(ROLES, (16,6,4,6)):
        roles[role] = [dict(relative_path=f'synthetic/train/good/{i:03d}.png', task_name='synthetic',
                           task_id=0, split='train', sha256=f'synthetic:{i}') for i in range(offset, offset+count)]
        offset += count
    manifest = dict(schema='hope_prefix_roles_v3', first_task='synthetic', roles=roles)
    manifest['digest'] = canonical_digest(manifest)
    cache = {}
    masks = patch_masks(config.height, config.width, device=device)
    def provider(identity, view):
        key = (identity, view)
        if key not in cache:
            index = int(identity.rsplit('/',1)[-1].split('.')[0])
            gen = torch.Generator().manual_seed(seed + 8*index + view)
            shape = (1, config.height*config.width, config.input_dim)
            features = torch.randn(shape, generator=gen).to(device)
            teacher_gen = torch.Generator().manual_seed(seed+10000+index)
            target = frozen_teacher(torch.randn(shape, generator=teacher_gen).to(device))
            cache[key] = features, target, masks[view]
        return cache[key]
    return PrefixFeatures(manifest, provider)
