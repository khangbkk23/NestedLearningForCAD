"""
Models for Continual Learning Experiments
"""

from .dino_nsp2 import ACCGatingHead, DinoNSP2
from .titans_memory import TITANSMemory
from .cms import CMS, MlpBlock
from .vit_cms import ViT_CMS, ViT_Simple, ViT_Replay
from .cnn_baseline import SimpleCNN, CNN_Replay, ReplayBuffer

__all__ = [
    'ACCGatingHead',
    'DinoNSP2',
    'TITANSMemory',
    'CMS',
    'MlpBlock',
    'ViT_CMS',
    'ViT_Simple',
    'ViT_Replay',
    'SimpleCNN',
    'CNN_Replay',
    'ReplayBuffer'
]