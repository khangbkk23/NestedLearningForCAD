"""Focused CPU tests for NANM-EVAL-01 amendment A1 (MVTec AD). No tensor, label or mask is read; GPU parity is
checked at run time (Stage-A bitwise grid parity vs the oracle's canonical MVTec grids, check_S1_bitwise)."""
import pickle

import pytest

from exps.nanm_eval_01 import common as U
from exps.nanm_eval_01 import mvtec as MV

COUNTS = {'bottle': (209, 83), 'cable': (224, 150), 'capsule': (219, 132), 'carpet': (280, 117), 'grid': (264, 78),
          'hazelnut': (391, 110), 'leather': (245, 124), 'metal_nut': (220, 115), 'pill': (267, 167),
          'screw': (320, 160), 'tile': (230, 117), 'toothbrush': (60, 42), 'transistor': (213, 100),
          'wood': (247, 79), 'zipper': (240, 151)}


@pytest.fixture(scope='module')
def G():
    return MV.mv_bind()


def test_dispatch_mvtec_and_original_untouched(G):
    from exps.calm02 import data as CD
    assert G.order('mvtec') == tuple(CD.MVTEC_ORDER)
    assert G.order('mpdd') == G.MPDD_ORDER and G.order('visa') == tuple(CD.VISA_ORDER)
    assert G.data_root('mvtec') == MV.MV_ROOT and G.data_root('mpdd') == G.MPDD
    assert len(G.train_rows('mpdd', 'tubes')) > 0


def test_counts_match_source_audit(G):
    for c, (ntr, nte) in COUNTS.items():
        assert len(G.train_rows('mvtec', c)) == ntr, c
        assert len(G.test_rows('mvtec', c)) == nte, c
    assert sum(v[0] for v in COUNTS.values()) == 3629 and sum(v[1] for v in COUNTS.values()) == 1725


def test_fit_partition_is_registered_rule(G):
    from exps.nar_pilot_00 import core as K
    for c in COUNTS:
        rows = G.train_rows('mvtec', c)
        r = K.normal_partition(rows)
        assert G.fit_index('mvtec', rows) == r['FIT']
        assert not set(r['FIT']) & set(r['EVAL']) and not set(r['FIT']) & set(r['CREDIT'])
        assert len(r['FIT']) == len(rows) - 2 * max(1, len(rows) // 5)


def test_sizes_and_bytes(G):
    sz = G.sizes_for('mvtec')
    bt = G.byte_table(sz, 'mvtec')
    assert len(sz['n_fit']) == 15 and all(n % 784 == 0 for n in sz['n_fit'])
    assert 0 < bt['A0b_minus_G3'] <= U.KEY_B
    assert bt['G3'] == U.bytes_of('g3', sum(sz['g3']))
    for cap in U.CAPS_MIB:                                     # every registered cap is feasible on MVTec
        assert U.cap_ratio(sz['n_fit'], cap * U.MiB, U.S_B)[0] is not None, cap


def test_test_manifest_paths_only():
    tm = MV.test_manifest()
    assert sum(len(v) for v in tm.values()) == 1725
    assert all(isinstance(p, str) and '/test/' in p for v in tm.values() for p in v)
    assert MV.k_text(tm) == MV.k_text(MV.test_manifest())


def test_rebind_and_worker_pickle():
    from exps.nanm_eval_01 import metrics as M
    from exps.nanm_eval_01 import run as RN
    MV.rebind()
    assert U.OUT == MV.MV_OUT and U.CACHE == MV.MV_CACHE and U.EXEC_DATASETS == ('mvtec',) and U.bind is MV.mv_bind
    assert RN.grid_path('mvtec', 'a|2711', 'N0', 'bottle').is_relative_to(MV.MV_CACHE)
    assert M.run_category is MV.run_category
    assert pickle.loads(pickle.dumps(M.run_category)) is MV.run_category
    assert MV.MV_CACHE.as_posix().startswith('/mnt/k/')         # K: only (never the D-backed WSL root)
