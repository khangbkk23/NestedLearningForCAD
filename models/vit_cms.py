"""
vit_cms.py
----------
Chỉ chứa các module hỗ trợ (Decoder) và Baseline Model.
Tuyệt đối KHÔNG chứa logic của TITANS Memory (đã chuyển giao cho meta_nath_core.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Dict

class AnomalyDecoder(nn.Module):
    """
    Fuses multi-scale patch-token features into a single spatial anomaly map.
    Được sử dụng trong Phase 2 (CADIC Coreset) để tính Pixel-level Anomaly.
    """
    def __init__(
        self,
        embed_dim: int,
        num_scales: int,
        patch_size: int = 16,
        img_size: int = 256,
        reduced_dim: int = 128,
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.img_size    = img_size
        self.num_patches = img_size // patch_size

        self.scale_projs = nn.ModuleList([
            nn.Conv2d(embed_dim, reduced_dim, kernel_size=1)
            for _ in range(num_scales)
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(reduced_dim * num_scales, reduced_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(reduced_dim, 1, kernel_size=1),
        )

    def forward(self, scale_features: List[torch.Tensor]) -> torch.Tensor:
        B = scale_features[0].shape[0]
        maps = []
        for feat, proj in zip(scale_features, self.scale_projs):
            feat_2d = feat.reshape(B, self.num_patches, self.num_patches, -1)
            feat_2d = feat_2d.permute(0, 3, 1, 2).contiguous()
            feat_2d = proj(feat_2d)
            feat_2d = F.interpolate(
                feat_2d, size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False
            )
            maps.append(feat_2d)

        fused = torch.cat(maps, dim=1)
        return self.fusion(fused)


class ViT_Simple(nn.Module):
    """
    Baseline ViT thuần túy (Không có TITANS, Không có Continual Learning).
    Sử dụng để chạy đối chứng (Ablation Study) và test DataLoader của team.
    """
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        img_size: int = 256,
        extract_layers: List[int] = (3, 6, 9),
        reduced_dim: int = 128,
    ):
        super().__init__()
        self.extract_layers = list(extract_layers)
        self.img_size = img_size

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, img_size=img_size
        )
        self.embed_dim  = self.backbone.num_features
        self.patch_size = self.backbone.patch_embed.patch_size
        if isinstance(self.patch_size, (tuple, list)):
            self.patch_size = self.patch_size[0]

        self._hook_outputs: Dict[int, torch.Tensor] = {}
        self._hooks = []
        for idx in self.extract_layers:
            h = self.backbone.blocks[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self._hooks.append(h)

        self.decoder = AnomalyDecoder(
            embed_dim=self.embed_dim,
            num_scales=len(self.extract_layers),
            patch_size=self.patch_size,
            img_size=img_size,
            reduced_dim=reduced_dim,
        )

    def _make_hook(self, idx: int):
        def hook(module, input, output):
            self._hook_outputs[idx] = output[:, 1:, :]
        return hook

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._hook_outputs.clear()
        cls_features = self.backbone(x)
        scale_feats = [self._hook_outputs[i] for i in self.extract_layers
                       if i in self._hook_outputs]
        anomaly_map = torch.sigmoid(self.decoder(scale_feats))
        return {
            'anomaly_map':    anomaly_map,
            'image_score':    anomaly_map.mean(dim=(1, 2, 3)),
            'features':       cls_features,
            'patch_features': scale_feats,
        }

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)['features']