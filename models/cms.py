import torch
import torch.nn as nn


class MlpBlock(nn.Module):
    """
    Architecture: Linear -> GELU -> Dropout -> Linear -> Dropout
    """
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class CMS(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., num_levels=3, k=2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.num_levels = num_levels
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.step_counter = 0
        level_hidden = hidden_features // num_levels

        self.levels = nn.ModuleList()
        for i in range(num_levels):
            self.levels.append(
                MlpBlock(in_features, level_hidden, in_features, drop)
            )
        if out_features != in_features:
            self.output_proj = nn.Linear(in_features, out_features)
        else:
            self.output_proj = None

    def forward(self, x):
        self.step_counter += 1

        for i, level in enumerate(self.levels):
            if self.step_counter % (self.k ** i) == 0:
                x = x + level(x)
        
        if self.output_proj is not None:
            x = self.output_proj(x)
            
        return x

    def __repr__(self):
        return (f"CMS(in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"num_levels={self.num_levels}, "
                f"k={self.k})")