"""Inference-only cache of immutable NAR factors; canonical decoder arithmetic unchanged."""
import torch
import torch.nn.functional as F
from exps.nar_pilot_00 import core as K
from exps.nar_pilot_00.detector import Detector


class ResidentDetector(Detector):
    """Own a separate eval-only model; do not share it with a trainer or oracle.

    Only parameter storage references change between experts. Original CPU archive
    and checkpoint are untouched. GPU factor copies are explicit runtime memory.
    deferred_checks preserves finite-input score arithmetic, but reports nonfinite
    expert scores at the end of the scan instead of the first offending expert.
    """
    def __init__(self, model, loras, archive, deferred_checks=False):
        super().__init__(model,loras,archive)
        self.deferred_checks=deferred_checks
        for p in model.parameters(): p.requires_grad_(False)
        self.resident=[]
        for state in archive:
            self.resident.append({n:tuple(x.detach().to(device=w.A.device,dtype=w.A.dtype,copy=True)
                for x in state[n]) for n,w in loras.items()})

    @torch.no_grad()
    def bind(self,state):
        K.require(not self.model.training,'resident model is inference-only')
        for n,w in self.loras.items():
            w.A.set_(state[n][0]); w.B.set_(state[n][1])

    @torch.no_grad()
    def expert_grids(self,features):
        self.model.eval(); grids=[]
        for state in self.resident:
            self.bind(state)
            en,de=self.model(features)
            grid=(1-F.cosine_similarity(en[0].float(),de[0].float(),dim=1)).clamp_min(0)
            if not self.deferred_checks: K.require(bool(torch.isfinite(grid).all()),'nonfinite expert score')
            grids.append(grid)
        experts=torch.stack(grids)
        if self.deferred_checks: K.require(bool(torch.isfinite(experts).all()),'nonfinite expert score')
        return experts

    @torch.no_grad()
    def predict(self,features,hw=None):
        if not self.deferred_checks: return super().predict(features,hw)
        experts=self.expert_grids(features)
        grid=experts.clamp_min(0).amin(0)
        image=grid.flatten(1).amax(1)
        native=F.interpolate(grid[:,None],size=hw,mode='bilinear',align_corners=False)[:,0] if hw else grid
        return native,image

    def runtime_factor_bytes(self):
        return sum(K.tensor_state_bytes(s) for s in self.resident)
