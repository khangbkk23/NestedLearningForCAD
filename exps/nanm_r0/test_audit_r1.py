"""r1: the prefix transform equals an explicit union-memory kNN, and argmin recovers the nearest-neighbour task."""
import torch

from exps.nanm_r0 import audit as A
from exps.nanm_r0.audit_r1 import prefix


def test_prefix_equals_union_memory_knn():
    g = torch.Generator().manual_seed(3)
    mems = [A.unit(torch.randn(50, 16, generator=g)) for _ in range(4)]
    q = A.unit(torch.randn(30, 16, generator=g))
    per = torch.stack([A.min_cos_dist(q, m) for m in mems])          # [T, Nq] as stored by audit.test
    pre = prefix({'full': per})['full']
    for s in range(4):
        union = torch.cat(mems[:s + 1])
        assert torch.allclose(pre[s], A.min_cos_dist(q, union), atol=1e-6)
    nn_task = pre.argmin(0)
    owner = torch.cat([torch.full((50,), t) for t in range(4)])
    sims = q @ torch.cat(mems).T
    assert torch.equal(nn_task, owner[sims.argmax(1)])
