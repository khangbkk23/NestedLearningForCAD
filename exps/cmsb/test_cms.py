import torch
import torch.nn as nn

from exps.cmsb import cms as C


def test_zero_increment_and_active_levels():
    torch.manual_seed(0)
    base = nn.Linear(6, 5)
    w = C.CMSLinear(base, 2, 3)
    x = torch.randn(4, 6)
    assert torch.allclose(w(x), base(x))
    with torch.no_grad():
        w.A[1].normal_(); w.B[1].normal_()
    w.active = [0, 2]
    assert torch.allclose(w(x), base(x))
    w.active = [0, 1, 2]
    assert torch.allclose(w(x), base(x) + x @ w.A[1].T @ w.B[1].T, atol=1e-6)


def test_reservoir_bytes_constant_in_number_of_tasks():
    cap = 16 * 2**20
    sizes = []
    g = torch.Generator().manual_seed(0)
    for tasks in (12, 24):
        r = C.Reservoir(cap, 1)
        for t in range(tasks):
            for _ in range(500):
                r.offer(b'x' * int(torch.randint(20_000, 90_000, (1,), generator=g)), t)
                assert r.bytes() <= cap
        sizes.append(r.bytes())
    assert all(cap - 100_000 < s <= cap for s in sizes)


def test_reservoir_keeps_old_tasks():
    r = C.Reservoir(16 * 2**20, 3)
    for t in range(12):
        for _ in range(500):
            r.offer(b'y' * 55_000, t)
    counts = [r.tags.count(t) for t in range(12)]
    assert min(counts) > 0 and max(counts) < 3 * min(counts)


def test_jpeg_item_cap_and_decode():
    from exps.cmsb import dev as D
    from exps.vbench_00 import run as V
    p = V.train_rows('cashew')[0]['relative_path']
    b = C.jpeg392(p)
    assert len(b) <= C.ITEM_BYTES
    x = C.decode(b)
    assert x.shape == (3, 392, 392) and float((x - D.normalise(D.load392(p))).abs().mean()) < .1


def test_adapter_state_constant_in_tasks():
    spec = C.Spec()
    st = {f's{i}': ([torch.zeros(8, 768)] * len(spec.levels), [torch.zeros(768, 8)] * len(spec.levels)) for i in range(34)}
    assert C.state_bytes(st) == 34 * len(spec.levels) * 2 * 8 * 768 * 4
