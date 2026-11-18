"""NAR-B14-01 tests (synthetic CPU; no dataset or checkpoint access)."""
import torch

from exps.nar_b14_01 import run as B


class FakeEnc:
    num_prefix = 5

    def groups(self, x):
        g = torch.arange(2 * 789 * 768, dtype=torch.float32).reshape(1, 2, 789, 768)
        return g.expand(x.shape[0], -1, -1, -1)


def test_tokens_take_group2_patch_tokens():
    t = B.tokens(FakeEnc(), torch.zeros(3, 3, 392, 392))
    assert t.shape == (3, 784, 768)
    assert t[0, 0, 0] == 789 * 768 + 5 * 768          # group index 1, first patch token after CLS + 4 registers


def test_caps_and_sources_bound():
    B.bind()
    assert B.W.OUT == B.OUT and B.P.OUT == B.OUT and B.W.CAPS == B.CAPS
    assert B.W.verify_training is B.P.verify_training
