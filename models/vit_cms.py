import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Optional, Tuple, Dict
from .cms import CMS

def replace_mlp_with_cms(
    model: nn.Module,
    num_levels: int = 3,
    k: int = 2,
    use_spatial_gate: bool = True,
    verbose: bool = False,
) -> nn.Module:
  
    if hasattr(model, 'blocks'):
        for layer_idx, block in enumerate(model.blocks):
            for attr_name in ['mlp', 'ffn']:
                child = getattr(block, attr_name, None)
                if child is None:
                    continue
                if not hasattr(child, 'fc1'):
                    continue

                in_feat     = child.fc1.in_features
                hidden_feat = child.fc1.out_features
                out_feat    = child.fc2.out_features
                drop_rate   = 0.0
                for attr in ['drop', 'drop1']:
                    d = getattr(child, attr, None)
                    if d is not None and hasattr(d, 'p'):
                        drop_rate = d.p
                        break

                cms = CMS(
                    in_features=in_feat,
                    hidden_features=hidden_feat,
                    out_features=out_feat,
                    drop=drop_rate,
                    num_levels=num_levels,
                    k=k,
                    vit_layer_idx=layer_idx,
                    use_spatial_gate=use_spatial_gate,
                )
                setattr(block, attr_name, cms)
                if verbose:
                    print(f"  Replaced block[{layer_idx}].{attr_name} → {cms}")
        return model

    def _replace_recursive(module: nn.Module, depth: int = 0):
        for name, child in module.named_children():
            if child.__class__.__name__ in ('Mlp', 'MlpBlock', 'FeedForward'):
                if not hasattr(child, 'fc1'):
                    continue
                in_feat     = child.fc1.in_features
                hidden_feat = child.fc1.out_features
                out_feat    = child.fc2.out_features
                drop_rate   = 0.0
                for attr in ['drop', 'drop1']:
                    d = getattr(child, attr, None)
                    if d is not None and hasattr(d, 'p'):
                        drop_rate = d.p
                        break
                cms = CMS(
                    in_features=in_feat,
                    hidden_features=hidden_feat,
                    out_features=out_feat,
                    drop=drop_rate,
                    num_levels=num_levels,
                    k=k,
                    vit_layer_idx=depth,
                    use_spatial_gate=use_spatial_gate,
                )
                setattr(module, name, cms)
                if verbose:
                    print(f"  Replaced {name} at depth {depth} → {cms}")
            else:
                _replace_recursive(child, depth + 1)

    _replace_recursive(model)
    return model

class AnomalyDecoder(nn.Module):
    """
    Fuses multi-scale patch-token features into a single spatial anomaly map.

    Input:  list of (B, N_patches, C) tensors from different ViT layers
    Output: (B, 1, H_out, W_out) anomaly score map

    Architecture:
      Per-scale: 1×1 conv to reduce channels → bilinear upsample to target_size
      Fusion:    concat → 3×3 conv → sigmoid
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
        self.num_patches = img_size // patch_size   # spatial side length

        # Per-scale channel reduction
        self.scale_projs = nn.ModuleList([
            nn.Conv2d(embed_dim, reduced_dim, kernel_size=1)
            for _ in range(num_scales)
        ])

        # Fusion convolution
        self.fusion = nn.Sequential(
            nn.Conv2d(reduced_dim * num_scales, reduced_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(reduced_dim, 1, kernel_size=1),
        )

    def forward(self, scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        scale_features: list of (B, N, C) where N = num_patches²
        returns: (B, 1, img_size, img_size)
        """
        B = scale_features[0].shape[0]
        maps = []
        for feat, proj in zip(scale_features, self.scale_projs):
            # (B, N, C) → (B, C, H_p, W_p)
            feat_2d = feat.reshape(B, self.num_patches, self.num_patches, -1)
            feat_2d = feat_2d.permute(0, 3, 1, 2).contiguous()
            feat_2d = proj(feat_2d)
            # upsample to full image resolution
            feat_2d = F.interpolate(
                feat_2d, size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False
            )
            maps.append(feat_2d)

        fused = torch.cat(maps, dim=1)   # (B, reduced_dim*num_scales, H, W)
        return self.fusion(fused)        # (B, 1, H, W)


class ViT_CMS(nn.Module):
    """

    Args:
        model_name:       timm model identifier
        pretrained:       load ImageNet weights
        cms_levels:       number of CMS levels (default 3)
        k:                CMS update ratio base (default 2)
        extract_layers:   which ViT block indices to tap for multi-scale features
        img_size:         input image resolution (must match backbone config)
        use_spatial_gate: enable SpatialGatingUnit in CMS Level 0
        freeze_backbone:  freeze ALL backbone weights (only train decoder)
        freeze_patch_embed: freeze patch embedding only
    """

    # Standard ViT-B/16 settings as defaults; override via config
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_256',
        pretrained: bool = True,
        cms_levels: int = 3,
        k: int = 2,
        extract_layers: List[int] = (3, 6, 9),
        img_size: int = 256,
        use_spatial_gate: bool = True,
        freeze_backbone: bool = False,
        freeze_patch_embed: bool = False,
        reduced_dim: int = 128,
    ):
        super().__init__()
        self.extract_layers = list(extract_layers)
        self.img_size = img_size

        print(f"[ViT_CMS] Loading {model_name} (pretrained={pretrained})...")
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, img_size=img_size
        )
        self.embed_dim  = self.backbone.num_features
        self.patch_size = self.backbone.patch_embed.patch_size
        if isinstance(self.patch_size, (tuple, list)):
            self.patch_size = self.patch_size[0]

        print(f"[ViT_CMS] Replacing MLP → CMS (levels={cms_levels}, k={k}, "
              f"spatial_gate={use_spatial_gate})...")
        self.backbone = replace_mlp_with_cms(
            self.backbone,
            num_levels=cms_levels,
            k=k,
            use_spatial_gate=use_spatial_gate,
            verbose=False,
        )

        self._hook_outputs: Dict[int, torch.Tensor] = {}
        self._hooks = []
        if hasattr(self.backbone, 'blocks'):
            for idx in self.extract_layers:
                hook = self.backbone.blocks[idx].register_forward_hook(
                    self._make_hook(idx)
                )
                self._hooks.append(hook)
        else:
            print("[ViT_CMS] WARNING: backbone has no .blocks — "
                  "multi-scale extraction disabled.")

        self.decoder = AnomalyDecoder(
            embed_dim=self.embed_dim,
            num_scales=len(self.extract_layers),
            patch_size=self.patch_size,
            img_size=img_size,
            reduced_dim=reduced_dim,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            print("[ViT_CMS] Backbone frozen.")
        elif freeze_patch_embed:
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad_(False)
            print("[ViT_CMS] Patch embedding frozen.")

    def _make_hook(self, idx: int):
        def hook(module, input, output):
            self._hook_outputs[idx] = output[:, 1:, :]
        return hook

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._hook_outputs.clear()

        cls_features = self.backbone(x)

        scale_feats = [
            self._hook_outputs[idx]
            for idx in self.extract_layers
            if idx in self._hook_outputs
        ]

        if scale_feats:
            anomaly_map = self.decoder(scale_feats)          # (B, 1, H, W)
            anomaly_map = torch.sigmoid(anomaly_map)
        else:
            score = cls_features.mean(dim=-1, keepdim=True)  # (B, 1)
            anomaly_map = score[:, :, None, None].expand(
                -1, 1, self.img_size, self.img_size
            )

        image_score = anomaly_map.mean(dim=(1, 2, 3))        # (B,)

        return {
            'anomaly_map':    anomaly_map,    # (B, 1, H, W) — for Pixel-AP
            'image_score':    image_score,    # (B,)          — for AUROC
            'features':       cls_features,   # (B, C)        — for KD loss
            'patch_features': scale_feats,    # list[(B,N,C)] — for memory bank
        }

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)['features']

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()


class ViT_Simple(nn.Module):
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_256',
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