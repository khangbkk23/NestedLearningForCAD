"""TCS-00 tests (CPU; identities from the split CSV only; no image, mask or anomaly metric access)."""
import numpy as np

from exps.tcs_00 import run as T


def _ap(fn):
    return {s: {g: {c: {h: fn(s, g, c, h) for h in ('full', 'S1', 'S2')} for c in T.ORDER} for g in T.GAMMAS}
            for s in T.SEEDS}


def test_identities_deterministic_and_disjoint():
    rows = T.train_rows('candle')
    assert len(rows) == T.N_TRAIN and rows == T.train_rows('candle')
    r = T.roles_for(rows)
    sets = [set(v) for v in r.values()]
    assert sum(map(len, sets)) == T.N_TRAIN and not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(r['EVAL']) == len(r['CREDIT']) == T.N_TRAIN // 5


def test_dev_identity_and_halves():
    dev = T.dev_rows()                                   # asserts the frozen VisA-DEV hash
    assert sum(len(v) for v in dev.values()) == 240
    for rows in dev.values():
        for lab in (0, 1):
            halves = [r['half'] for r in rows if r['label'] == lab]
            assert halves.count('S1') == halves.count('S2') == 5


def test_crossfit_never_evaluates_on_selection():
    _, _, pairs, picks = T.crossfit(_ap(lambda s, g, c, h: 0.5))
    assert len(pairs) == 4 and all(s != s2 and h != h2 for s, h, s2, h2 in pairs)
    assert all(p['select'][0] != p['eval'][0] and p['select'][1] != p['eval'][1] for p in picks)


def test_headroom_known_cases():
    # gamma-independent AP -> exactly zero headroom, NO-GO
    flat = T.headroom(_ap(lambda s, g, c, h: 0.4 + 0.01 * T.ORDER.index(c)))
    assert abs(flat['H']['point']) < 1e-12 and flat['verdict'] == 'NO_TASK_CONDITIONAL_HEADROOM'
    # stable class-specific optimum (half the classes prefer gamma 1, half gamma 3), noise-free -> known H
    pref = {c: (1.0 if i % 2 == 0 else 3.0) for i, c in enumerate(T.ORDER)}
    ap = _ap(lambda s, g, c, h: 0.5 - 0.02 * (g - pref[c]) ** 2 + (0.001 if s == 2712 else 0.))
    res = T.headroom(ap)
    # constant argmax is gamma 2 (macro 0.48 vs 0.46); every class loses 0.02 under it -> H = 0.02
    assert np.isclose(res['H']['point'], 0.02) and res['argmax_agreement'] == 12
    assert res['verdict'] == 'GO_TO_E1'


def test_gate_constants():
    assert T.GATE == {'headroom': .010, 'ci_lower': 0., 'argmax_agreement': 8, 'spearman': .5,
                      'effect_over_noise': 2.}
    assert T.GAMMAS == (1.0, 2.0, 3.0) and T.SEEDS == (2711, 2712)
