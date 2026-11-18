"""VBENCH-00 tests (CPU; split CSV identities only)."""
import inspect

from exps.tcs_00 import run as T
from exps.vbench_00 import run as V


def test_frozen_configuration():
    assert V.GAMMA == 2.1 and V.SEED == 2711 and V.ORDER == T.ORDER


def test_test_rows_cover_full_visa_and_mark_dev():
    rows = V.test_rows()
    assert sum(len(v) for v in rows.values()) == 2162
    assert sum(r['dev'] for v in rows.values() for r in v) == 240


def test_train_rows_are_all_train_normals():
    assert sum(len(V.train_rows(c)) for c in V.ORDER) == 8659


def test_scoring_requires_seal_and_fixed_policy():
    assert 'verify_training()' in inspect.getsource(V.score)
    assert 'learn=False' in inspect.getsource(V.train)
