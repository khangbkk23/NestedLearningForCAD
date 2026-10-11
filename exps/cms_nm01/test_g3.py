# exps/cms_nm01/test_g3.py
"""Small locked-score checks; no development payload is loaded here."""

import numpy as np
import torch

from exps.cms_nm01.g3 import TOP_K, forgetting


def test_locked_top_k_is_eight_and_native_score_shape():
    assert TOP_K == 8
    patch = torch.arange(20 * 784, dtype=torch.float32).reshape(20, 784)
    image = torch.topk(patch, TOP_K, dim=1).values.mean(1)
    assert image.shape == (20,)
    assert torch.equal(image, patch[:, -TOP_K:].mean(1))


def test_forgetting_uses_only_post_acquisition_rows():
    matrix = [[.9, .1, .2], [.8, .8, .3], [.7, .7, .7]]
    result = forgetting(matrix, ["bottle", "carpet", "hazelnut"])
    assert np.isclose(result["per_category"]["bottle"], .2)
    assert np.isclose(result["per_category"]["carpet"], .1)
    assert result["per_category"]["hazelnut"] == 0.0
    assert np.isclose(result["mean"], .1)
