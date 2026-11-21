"""Synthetic checks for NANM-R2b (no dataset access)."""
import math

import torch

from exps.nanm_r0 import audit as A
from exps.nanm_r2b import run as B


def test_control_sizes_base():
    n_fit = [99568, 106624, 28224]
    r, s = B.control_sizes(n_fit, 2_362_376, 0.02)
    assert sum(s) >= sum(math.ceil(0.02 * n) for n in n_fit) + math.ceil(2_362_376 / 1536)
    assert all(x >= math.ceil(0.02 * n) for x, n in zip(s, n_fit))


def test_residual_stats_members_excluded_matches_bruteforce():
    g = torch.Generator().manual_seed(0)
    x = A.unit(torch.randn(60, 8, generator=g)).to(torch.float32)
    x = torch.nn.functional.pad(x, (0, 760))                    # embed in 768-d
    mem = torch.tensor([0, 5, 9, 20])
    S, n = B.residual_stats(x, mem, chunk=7)
    keep = [i for i in range(60) if i not in mem.tolist()]
    r = torch.stack([x[i] - x[mem][torch.cdist(x[i:i + 1], x[mem]).argmin()] for i in keep])
    assert n == 56 and torch.allclose(S, (r.T @ r).double(), atol=1e-5)


def test_residual_stats_leave_one_image_out():
    g = torch.Generator().manual_seed(1)
    x = torch.nn.functional.pad(A.unit(torch.randn(40, 8, generator=g)), (0, 760))
    img = torch.arange(40) // 4
    S, n = B.residual_stats(x, None, img, chunk=9)
    rs = []
    for i in range(40):
        d = torch.cdist(x[i:i + 1], x)[0]; d[img == img[i]] = float('inf')
        rs.append(x[i] - x[d.argmin()])
    r = torch.stack(rs)
    assert n == 40 and torch.allclose(S, (r.T @ r).double(), atol=1e-5)


def test_orders_are_permutations():
    o = B.orders()
    for seq in o.values():
        assert sorted(seq) == sorted(A.ORDER) and len(seq) == 15
    assert o['reverse'][0] == A.ORDER[-1]
