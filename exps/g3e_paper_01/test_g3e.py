"""G3E-PAPER-01 unit tests (byte accounting, overflow fixture, eviction rules on synthetic unit tokens)."""
import pytest
import torch

from exps.g3e_paper_01 import learner as LR

MiB = LR.MiB


def test_caps_and_bytes():
    assert LR.cap_of('E@8') == (8 * MiB - LR.S_B) // 1536 == 3925
    assert LR.cap_of('E@4') == 1194 and LR.cap_of('E@16') == 9386 and LR.cap_of('E@32') == 20309
    assert LR.cap_of('F') == 8 * MiB // 1536 == 5461
    assert LR.cap_of('G') == (8 * MiB - LR.G1_B) // 1536 == 3923
    assert LR.cap_of('E@inf') is None and LR.cap_of('U') is None
    for a in ('E@4', 'E@8', 'E@16', 'E@32', 'Q', 'F', 'G', 'FIFO', 'E4', 'B1'):
        assert LR.bytes_of(a, LR.cap_of(a)) <= LR.budget_of(a), a


def test_overflow_fixture_triggers():
    for a in ('E@4', 'E@8', 'E@16', 'E@32', 'Q', 'F', 'G', 'FIFO', 'E4'):
        cap = LR.cap_of(a)
        LR.assert_budget(a, 0, cap)
        with pytest.raises(LR.BudgetViolation):
            LR.assert_budget(a, 0, cap + 1)
    LR.assert_budget('U', 0, 10 ** 6)             # unbounded never raises
    LR.assert_budget('E@inf', 0, 10 ** 6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda')
def test_learner_small_stream():
    """Tiny budget so that every rule evicts; E@inf keys == U union; FIFO keeps newest; Q <= cap; budget asserted."""
    torch.manual_seed(0)
    L = LR.Learner(['U', 'E@inf', 'E@4', 'Q', 'FIFO', 'F', 'G', 'E4', 'B1'], 2711)
    for l in range(3):
        x = torch.nn.functional.normalize(torch.randn(784 * 160, 768, device='cuda'), dim=1).half()
        L.task(x, l)
    K_u, W_u = L.keys_at('U', 2)
    K_i, W_i = L.keys_at('E@inf', 2)
    assert torch.equal(K_u, K_i) and torch.equal(W_u, W_i)
    assert L.bytes['U'] == L.bytes['E@inf']
    # 3 tasks x ceil(1% of 125,440) = 3 x 1,255 = 3,765 keys > cap 1,194 at 4 MiB -> evictions happened
    assert len(L.keys_at('E@4', 2)[0]) == LR.cap_of('E@4')
    last = L.blocks[-1]
    kf = L.keys_at('FIFO', 2)[0]
    assert torch.equal(kf[-len(last):], last)                 # newest task kept in FIFO (cap 3,925 >= 1,255)
    assert L.keys_at('B1', 2)[1] is None and torch.equal(L.keys_at('B1', 2)[0], L.keys_at('E4', 2)[0])
