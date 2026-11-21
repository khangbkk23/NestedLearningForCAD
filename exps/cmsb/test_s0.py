import numpy as np
import torch

from exps.calm02 import data as CD
from exps.cmsb import dev as D
from exps.cmsb import s0 as S0
from exps.dinomaly import encoder as E
from exps.tcs_00 import run as T
from exps.vbench_00 import run as V

IDS = CD.identities('visa')


def test_hosts_are_held_out_and_deterministic():
    for c in D.ORDER:
        hs = D.hosts(c, IDS)
        assert len(hs) == len(set(hs)) == 40 and hs == D.hosts(c, IDS)
        rows = V.train_rows(c, IDS)
        roles = T.roles_for(rows)
        trained = {rows[i]['relative_path'] for i in roles['FIT'] + roles['CREDIT']}
        trs = T.train_rows(c, IDS)
        tr = T.roles_for(trs)
        trained |= {trs[i]['relative_path'] for i in tr['FIT'] + tr['CREDIT']}
        assert not set(hs) & trained


def test_normalise_matches_encoder_preprocess():
    p = D.hosts(D.ORDER[0], IDS)[0]
    assert torch.equal(D.normalise(D.load392(p)), E.preprocess(D.VISA / p))


def test_synthesis_deterministic_and_masked():
    from exps.ptag import synth as S
    gen = S.NativePerlin()
    c = D.ORDER[0]
    paths = D.hosts(c, IDS)
    a = D.synthesise(c, paths, gen)
    b = D.synthesise(c, paths, gen)
    assert a[2] == ['normal'] * 20 + ['Q2'] * 10 + ['Q3'] * 10
    for x, y, m, k in zip(a[0], b[0], a[1], a[2]):
        assert np.array_equal(x, y) and x.shape == (E.SIZE, E.SIZE, 3)
        assert (m.any() if k != 'normal' else not m.any())


def test_analysis_known_cases():
    dets = ('F_archive', 'F_last') + S0.TCS_CHAINS
    rng = np.random.default_rng(0)
    cells = {}
    for i, c in enumerate(D.ORDER):
        for j, d in enumerate(dets):
            v = .3 + .02 * i + .01 * j + (.05 if d == 'F_archive' else 0.)
            cells[d, c] = {'S': {'all': {'pixel_ap': v, 'image_auroc': .9}},
                           'R': {'pixel_ap': v + rng.normal(0, 1e-4), 'image_auroc': .9}}
    res = S0.analyse(cells)
    assert res['proxy']['valid'] and res['primary_set'] == 'Dev-S'
    assert abs(res['headroom']['Dev-S']['point'] - .04) < 1e-9 and res['verdict'] == 'BOUNDED_HEADROOM'
