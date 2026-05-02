import torch
from torch.optim import Optimizer

class CMSOptimizerWrapper:
    def __init__(self, optimizer: Optimizer, model: torch.nn.Module, k_factor: int = 5):
        self.optimizer = optimizer
        self.model = model
        self.k_factor = k_factor
        self.global_step = 0
        self.param_levels = self._map_params_to_levels()

    def _map_params_to_levels(self):
        level_map = {}
        for name, param in self.model.named_parameters():
            level = 0 
            parts = name.split('.')
            for part in parts:
                if part.startswith('level_'):
                    try:
                        level = int(part.split('_')[1])
                    except ValueError:
                        pass
            
            level_map[param] = level
        return level_map

    def step(self):
        self.global_step += 1
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                level = self.param_levels.get(p, 0)
                update_freq = self.k_factor ** level
                if self.global_step % update_freq != 0:
                    p.grad = None

        return self.optimizer.step()

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)