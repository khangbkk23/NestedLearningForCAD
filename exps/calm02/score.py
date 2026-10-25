# exps/calm02/score.py
"""CALM-02 scorer, identical for every arm (the memory is the only difference).

Pixel score: 1-NN Euclidean distance to the stored rows. The neighbour is found by an FP32 matmul
(TF32 off) and its distance is recomputed exactly in FP64. Image score: CADIC Eq. (9) as in
``models/cadic_patch_coreset_v1.py`` (b nearest bank rows to c*), plus the max patch score.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from exps.task_p.p0_normal_support import require

B_NEIGHBOURS = 9


def decode(state):
    """Rows exactly as stored: FP32, or UINT8 decoded by the CSG-01 rule."""
    if "x" in state:
        return state["x"].float()
    return (state["xmin"].double()[:, None] + state["step"].double()[:, None] * state["q"].double()).float()


class Scorer:
    def __init__(self, bank, device="cuda", b=B_NEIGHBOURS):
        require(bank.ndim == 2 and len(bank) >= b and bool(torch.isfinite(bank).all()), "bad bank")
        self.b, self.device = b, device
        self.B64 = bank.double()
        self.B = bank.float().to(device)
        self.bn = self.B.square().sum(1)

    @torch.no_grad()
    def patch(self, q):
        """q [N,768] -> exact FP64 1-NN distances [N] and neighbour rows [N]."""
        qd = q.float().to(self.device)
        d2 = qd.square().sum(1)[:, None] + self.bn[None] - 2 * (qd @ self.B.T)
        idx = d2.argmin(1).cpu()
        d = (q.double() - self.B64[idx]).square().sum(1).sqrt()
        return d, idx

    @torch.no_grad()
    def image_cadic(self, q, d, idx):
        """Eq. (9): w * s*, w = 1 - exp(s*) / sum_{m in N_b(c*)} exp(||m - z*||)."""
        star = int(d.argmax())
        c_star = self.B64[int(idx[star])]
        to_c = (self.B64 - c_star).square().sum(1).sqrt()
        support = torch.argsort(to_c, stable=True)[:self.b]
        dist = (self.B64[support] - q[star].double()).square().sum(1).sqrt()
        s = d[star]
        return float((1.0 - torch.exp(s - torch.logsumexp(dist, 0))) * s)


def pixel_map(d, size=224):
    """[784] -> [size,size] FP32 bilinear (align_corners=False), as the CADIC harness adapter."""
    return F.interpolate(d.float().reshape(1, 1, 28, 28), size=(size, size), mode="bilinear",
                         align_corners=False)[0, 0]
