"""NAR detector with the existing task-agnostic min-expert scorer."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from exps.nar_pilot_00 import core as K


class Detector:
    def __init__(self, model, loras, archive):
        K.require(len(archive)>0, 'no learned task states')
        self.model, self.loras, self.archive = model.eval(), loras, archive

    @torch.no_grad()
    def expert_grids(self, features):
        """All archived experts, [T,B,H,W]. No category/task-ID argument."""
        self.model.eval()
        grids = []
        for state in self.archive:
            K.set_state(self.loras,state)
            en,de = self.model(features)
            grid = (1-F.cosine_similarity(en[0].float(),de[0].float(),dim=1)).clamp_min(0)
            K.require(bool(torch.isfinite(grid).all()),'nonfinite expert score')
            grids.append(grid)
        return torch.stack(grids)

    @torch.no_grad()
    def predict(self,features,hw=None):
        experts = self.expert_grids(features)
        grid,image = K.task_agnostic_grid(list(experts))
        native = F.interpolate(grid[:,None],size=hw,mode='bilinear',align_corners=False)[:,0] if hw else grid
        return native,image


def prefix_grids(experts):
    """Exactly equivalent to separate min over every arrived expert prefix."""
    K.require(experts.ndim==4 and bool(torch.isfinite(experts).all()),'expert stack')
    return experts.clamp_min(0).cummin(dim=0).values


def fm_eq8(cells,order,metric):
    """Signed ReplayCAD Eq.8, including intermediate boundaries, last task excluded."""
    n=len(order)
    per={c:max(cells[(t,c)][metric] for t in range(i,n-1))-cells[(n-1,c)][metric]
         for i,c in enumerate(order[:-1])}
    return {'per_category':per,'macro':sum(per.values())/len(per),
            'positive_part_macro':sum(max(0,v) for v in per.values())/len(per)}
