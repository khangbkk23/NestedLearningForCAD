"""HOPE-S-02 tests (CPU; no dataset, checkpoint or test access)."""
import inspect

from exps.hope_s_01 import run as H
from exps.hope_s_02 import run as F


def test_fixed_policy_constants():
    assert F.GAMMA_FIXED == 2.1 and 'posthoc' in F.GAMMA_CLASS
    assert F.HOPE == H.OUT and F.LITE.name == 'nar_lite_01' and F.OUT.name == 'hope_s_02'


def test_same_acquisition_code_path_with_learning_disabled():
    src = inspect.getsource(F.train)
    assert 'H.acquire(' in src and 'learn=False' in src and 'GAMMA_FIXED' in src


def test_acquire_learn_false_never_updates_policy():
    src = inspect.getsource(H.acquire)
    # updates are guarded by `if learn:`; credit and gamma-direction gradients are always computed
    assert 'if learn:' in src and 'tail_loss(' in src and 'gamma_direction_loss(' in src


def test_bind_and_caps():
    F.bind()
    assert F.L.OUT == F.OUT and F.L.REF == F.OUT and F.W.OUT == F.OUT
    assert F.CAPS['wall_seconds'] == 2700 and F.CAPS['gpu_seconds'] == 1800
