"""Fixed-boundary visual readout used by the no-self-modification control."""
import torch
from torch.nn import functional as F
from .scan_geometry import scan_routes

def encode(self, features):
    if self.dynamics.visual_updates:
        return self.visual(features)
    v, c = self.visual, self.visual.config
    if features.shape != (1, c.height * c.width, c.input_dim):
        raise ValueError('invalid one-image masked features')
    x = v.projection(features)
    z = v.depthwise(v.in_projection(v.input_norm(x)).transpose(1, 2).reshape(1, c.memory_dim, c.height, c.width))
    z = z.flatten(2).transpose(1, 2)
    memory = v.memory
    q = F.normalize(memory.query(z).reshape(1, c.height*c.width, memory.heads, memory.head_dim), dim=-1, eps=1e-6)
    outputs = []
    for index, route in enumerate(scan_routes(c.height, c.width, device=z.device)):
        query = route.scan(q.flatten(-2)).reshape(1, c.height*c.width, memory.heads, memory.head_dim).transpose(1, 2)
        y = (query @ memory.initial_content[index].transpose(-1, -2)).transpose(1, 2).flatten(-2)
        outputs.append(route.restore(y))
    readout = (torch.stack(outputs, 1) * memory.fusion[None, :, None]).sum(1)
    residual = x + v.out_projection(readout)
    return residual + v.ffn(v.ffn_norm(residual))

