import torch
import torch.nn as nn
import timm
from typing import Optional
from .cms import CMS


def replace_mlp_with_cms(model, num_levels=3, k=2, verbose=False):
    def _replace_in_module(module, parent_name=""):
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name

            if child.__class__.__name__ in ['Mlp', 'MlpBlock', 'FeedForward']:
                if verbose:
                    print(f"Replacing MLP at: {full_name}")
                
                in_features = child.fc1.in_features
                hidden_features = child.fc1.out_features
                out_features = child.fc2.out_features
                drop_rate = 0.0
                if hasattr(child, 'drop'):
                    drop_rate = child.drop.p if hasattr(child.drop, 'p') else 0.0
                elif hasattr(child, 'drop1'):
                    drop_rate = child.drop1.p if hasattr(child.drop1, 'p') else 0.0
                
                cms_layer = CMS(
                    in_features=in_features,
                    hidden_features=hidden_features,
                    out_features=out_features,
                    drop=drop_rate,
                    num_levels=num_levels,
                    k=k
                )

                setattr(module, name, cms_layer)
            else:
                _replace_in_module(child, full_name)
    
    _replace_in_module(model)
    return model

class ViT_CMS(nn.Module):
    def __init__(
        self, 
        model_name='vit_base_patch16_224',
        pretrained=True,
        num_classes=None,
        cms_levels=3,
        k=5
    ):
        super().__init__()
        
        print(f"Loading {model_name} (pretrained={pretrained})...")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        self.feature_dim = self.backbone.num_features
        
        print(f"Replacing MLP blocks with CMS (levels={cms_levels}, k={k})...")
        self.backbone = replace_mlp_with_cms(self.backbone, num_levels=cms_levels, k=k, verbose=False)

        self.head = None
        if num_classes is not None:
            self.head = nn.Linear(self.feature_dim, num_classes)
        
        self.num_classes = num_classes
        
    def forward(self, x):
        features = self.backbone(x)
        
        if self.head is not None:
            return self.head(features)
        return features
    
    def set_head(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.feature_dim, num_classes)
        
    def get_features(self, x):
        return self.backbone(x)


class ViT_Simple(nn.Module):
    def __init__(
        self,
        model_name='vit_base_patch16_224',
        pretrained=True,
        num_classes=10,
        head_layers=3,
        hidden_dim=512
    ):
        super().__init__()
        
        print(f"Loading {model_name} (pretrained={pretrained})...")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features
        
        # Create head
        if head_layers == 2:
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        elif head_layers == 3:
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            raise ValueError(f"head_layers must be 2 or 3, got {head_layers}")
        
        self.num_classes = num_classes
        
    def forward(self, x):
        """Forward pass through backbone and head."""
        features = self.backbone(x)
        return self.head(features)
    
    def get_features(self, x):
        """Extract features without classification head."""
        return self.backbone(x)
    
    def set_head(self, num_classes, head_layers=2, hidden_dim=512):
        """Set or replace the classification head."""
        self.num_classes = num_classes
        
        if head_layers == 2:
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        elif head_layers == 3:
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_classes)
            )


class ViT_Replay(nn.Module):
    def __init__(
        self,
        model_name='vit_base_patch16_224',
        pretrained=True,
        num_classes=10,
        head_layers=3,
        hidden_dim=512,
        buffer_size=1000
    ):
        super().__init__()
        
        # Load pretrained ViT
        print(f"Loading {model_name} (pretrained={pretrained})...")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        if head_layers == 2:
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        elif head_layers == 3:
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            raise ValueError(f"head_layers must be 2 or 3, got {head_layers}")
        
        self.num_classes = num_classes

        from .cnn_baseline import ReplayBuffer
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)
        print(f"Initialized replay buffer with size {buffer_size}")
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
    
    def get_features(self, x):
        return self.backbone(x)
    
    def set_head(self, num_classes, head_layers=3, hidden_dim=512):
        self.num_classes = num_classes
        
        if head_layers == 2:
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        elif head_layers == 3:
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_classes)
            )
    
    def add_to_buffer(self, images, labels, task_id):
        self.replay_buffer.add_samples(images, labels, task_id)
    
    def sample_from_buffer(self, batch_size):
        return self.replay_buffer.sample(batch_size)
    
    def get_buffer_size(self):
        return len(self.replay_buffer)