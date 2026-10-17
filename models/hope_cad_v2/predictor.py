"""W4 feature-level predictor. Full RGB benchmark adapter is a separate W5 task."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .visual_block import HOPECADVisualBlock, VisualBlockConfig
from .cms import ContinuumMemory
from .teacher_head import TeacherDecoder
from .scan_geometry import scan_routes


@dataclass(frozen=True)
class Dynamics:
    method: str
    periods: tuple
    rates: tuple
    hidden: tuple
    visual_updates: bool
    cms_updates: bool


def dynamics(method='M', beta=1e-3, *, selected_m_beta=None, dim=256):
    if method not in ('M', 'C0', 'C1', 'C2', 'C3', 'C4', 'A1'):
        raise ValueError('unregistered control')
    if method == 'C4':
        if selected_m_beta not in (1e-4, 1e-3, 1e-2):
            raise ValueError('C4 requires selected M beta, not an independent beta search')
        beta = selected_m_beta
    periods = (1,) if method == 'C3' else ((1, 1) if method == 'C4' else (1, 8))
    rates = (beta,) if method == 'C3' else (beta, beta / 8 if method == 'C4' else beta)
    hidden = (4 * dim,) if method == 'C3' else (2 * dim, 2 * dim)
    return Dynamics(method, periods, rates, hidden, method not in ('C0', 'C2'), method not in ('C0', 'C1'))


class HOPEFeaturePredictor(nn.Module):
    def __init__(self, config=None, method='M', beta=1e-3, selected_m_beta=None, *, revision='v3'):
        super().__init__()
        config = config or VisualBlockConfig()
        if revision not in ('v3', 'v4'):
            raise ValueError('unregistered architecture revision')
        self.revision = revision
        self.visual = HOPECADVisualBlock(config, row_retention=0.9 if revision == 'v4' else None)
        self.dynamics = dynamics(method, beta, selected_m_beta=selected_m_beta, dim=config.dim)
        self.cms = ContinuumMemory(config.dim, self.dynamics.hidden, self.dynamics.periods, self.dynamics.rates,
                                   objective='joint_final' if revision == 'v4' else 'intermediate')
        self.decoder = TeacherDecoder(config.dim, config.input_dim)
        if not self.dynamics.visual_updates:
            # These controls cannot affect a static boundary readout. Disclose inactive capacity.
            for name in ('initial_key', 'initial_value', 'initial_eta', 'initial_alpha'):
                getattr(self.visual.memory, name).requires_grad_(False)

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

    def predict(self, features, state):
        return self.decoder(self.cms(self.encode(features), state)[-1])

    def metadata(self):
        from dataclasses import asdict
        if self.cms.periods != self.dynamics.periods or self.cms.rates != self.dynamics.rates:
            raise ValueError('CMS dynamics changed without a method configuration revision')
        metadata = dict(design_id='hope_cad_v3_masked_teacher_cms', component='W4_feature_predictor',
                    full_rgb_adapter=False, canonical=self.visual.config.canonical,
                    visual_config=asdict(self.visual.config), dynamics=asdict(self.dynamics),
                    parameter_count=sum(p.numel() for p in self.parameters()),
                    active_parameter_count=sum(p.numel() for p in self.parameters() if p.requires_grad))
        if self.revision == 'v4':
            if self.cms.objective != 'joint_final' or self.visual.memory.row_retention != 0.9:
                raise ValueError('v4 dynamics changed without a revision')
            metadata.update(design_id='hope_cad_v4_warm_joint_cms_row_retention',
                            cms_objective='joint_final', row_retention=0.9,
                            outer_objective='mean_query_at_0_and_16')
        return metadata


class HOPEFeaturePredictorV4(HOPEFeaturePredictor):
    def __init__(self, config=None, method='M', beta=1e-3, selected_m_beta=None):
        super().__init__(config, method, beta, selected_m_beta, revision='v4')
