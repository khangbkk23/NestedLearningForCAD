"""
Simple CNN Baseline with Replay Buffer for Continual Learning
FIXED: ReplayBuffer logic to avoid Tensor comparison error
"""

import torch
import torch.nn as nn
import random
from typing import List, Tuple


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, hidden_dim=64, input_size=32):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate feature dimension dynamically
        feature_map_size = input_size // 8
        self.feature_dim = hidden_dim * 4 * feature_map_size * feature_map_size
        
        self.flatten = nn.Flatten()
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.fc = nn.Linear(512, num_classes)
        self.num_classes = num_classes
        
    def forward(self, x): 
        x = self.features(x)
        x = self.flatten(x)
        x = self.projection(x)
        x = self.fc(x)
        return x
    
    def get_features(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ReplayBuffer:
    def __init__(self, buffer_size=1000, sampling_strategy='balanced'):
        self.buffer_size = buffer_size
        self.sampling_strategy = sampling_strategy
        self.buffer = []
        
    def add_samples(self, images: torch.Tensor, labels: torch.Tensor, task_id: int):
        batch_size = images.shape[0]
        for i in range(batch_size):
            if len(self.buffer) < self.buffer_size:
                self.buffer.append((
                    images[i].cpu().clone(),
                    labels[i].cpu().clone(),
                    task_id
                ))
            else:
                idx = random.randint(0, self.buffer_size - 1)
                self.buffer[idx] = (
                    images[i].cpu().clone(),
                    labels[i].cpu().clone(),
                    task_id
                )
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        if len(self.buffer) == 0:
            return None, None, None
        
        sample_size = min(batch_size, len(self.buffer))
        selected_indices = []

        if self.sampling_strategy == 'random':
            selected_indices = random.sample(range(len(self.buffer)), sample_size)
            
        elif self.sampling_strategy == 'balanced':
            # 1. Group indices by task_id
            indices_by_task = {}
            for i, (_, _, tid) in enumerate(self.buffer):
                if tid not in indices_by_task:
                    indices_by_task[tid] = []
                indices_by_task[tid].append(i)
            
            task_ids = list(indices_by_task.keys())
            
            # 2. Calculate how many samples per task
            per_task = sample_size // len(task_ids)
            remainder = sample_size % len(task_ids)
            
            # 3. Select indices for each task
            for tid in task_ids:
                available_indices = indices_by_task[tid]
                n_pick = per_task + (1 if remainder > 0 else 0)
                remainder -= 1
                
                # Pick random indices for this task
                if available_indices:
                    picked = random.sample(available_indices, min(n_pick, len(available_indices)))
                    selected_indices.extend(picked)
            
            # 4. Fill remaining spots if any (due to small task buffers)
            if len(selected_indices) < sample_size:
                all_indices = set(range(len(self.buffer)))
                used_indices = set(selected_indices)
                remaining_pool = list(all_indices - used_indices)
                
                needed = sample_size - len(selected_indices)
                if len(remaining_pool) >= needed:
                    extra_picks = random.sample(remaining_pool, needed)
                    selected_indices.extend(extra_picks)
        
        else:
            # Fallback to random
            selected_indices = random.sample(range(len(self.buffer)), sample_size)
        
        # Retrieve actual data using selected indices
        samples = [self.buffer[i] for i in selected_indices]
        
        # Unpack
        images = torch.stack([s[0] for s in samples])
        labels = torch.stack([s[1] for s in samples])
        task_ids = [s[2] for s in samples]
        
        return images, labels, task_ids
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer = []


class CNN_Replay(nn.Module):
    def __init__(self, num_classes=10, buffer_size=1000, hidden_dim=64, input_size=32):
        super().__init__()
        
        self.cnn = SimpleCNN(
            num_classes=num_classes, 
            hidden_dim=hidden_dim,
            input_size=input_size
        )
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)
        self.num_classes = num_classes
        self.fc = self.cnn.fc
        
    def forward(self, x):
        return self.cnn(x)
    
    def get_features(self, x):
        return self.cnn.get_features(x)
    
    def add_to_buffer(self, images, labels, task_id):
        self.replay_buffer.add_samples(images, labels, task_id)
    
    def sample_from_buffer(self, batch_size):
        return self.replay_buffer.sample(batch_size)
        
    def get_buffer_size(self):
        return len(self.replay_buffer)