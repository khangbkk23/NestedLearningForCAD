"""CPU checks for NL-ALIGN-01 G1 (no GPU, no anomaly label)."""
import numpy as np
import pytest
import torch

from exps.nl_align_01 import g1
from exps.dinomaly import common as D
from exps.nlcad import bank as BK
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB


def _hosts(seed=2711):
    man = D.read(g1.OUT / f"BANK_seed{seed}.json")
    return {c: {k: [int(x) for x in v] for k, v in h.items()} for c, h in man["hosts"].items()}


def test_roles_partition_and_lawful_pools():
    hosts = _hosts()
    role = g1.roles(hosts)
    assert set(role.values()) == {"OPT", "DIAG"}
    n = R1.n_train(RB.verify_lock())
    for s in range(g1.START + 1, g1.LAST + 1):
        allowed = BK.allowed(hosts, 2711, n, s)
        opt = {k for k in allowed if role[k] == "OPT"}
        assert opt and opt <= allowed
        cur = C.STREAM[s - 1]
        assert any(k[0] == cur for k in opt) and any(k[0] != cur for k in opt)
        assert not ({k for k, r in role.items() if r == "DIAG"} & opt)


def _fake_items(hosts):
    items = {g: {} for g in g1.GENS}
    for c, h in hosts.items():
        for i in set(h["current"]) | set(h["buffer"]):
            for v in range(2):
                for g in g1.GENS:
                    items[g][(c, i, v)] = (torch.zeros(1), torch.zeros(1))
    return items


def test_aux_arms_draw_identical_hosts():
    hosts = _hosts()
    role = g1.roles(hosts)
    n = R1.n_train(RB.verify_lock())
    items = _fake_items(hosts)
    s = 8
    done = list(C.STREAM[:s - 1])
    idx = C.buffer_index(2711, n, done)
    rk = [(d, i) for d in done for i in idx[d]]
    current = torch.zeros(n[C.STREAM[s - 1]], 1)
    replay = torch.zeros(len(rk), 1)
    seqs = []
    for arm in ("FS_A", "FS_B", "CN"):
        a = g1.Aux(arm, items, hosts, role, 2711, n, s, current, replay, rk)
        seqs.append([a.draw() for _ in range(5)])
        for draw in seqs[-1]:
            assert len(draw) == g1.N_CUR + g1.N_REP
            assert all(role[(c, i)] == "OPT" for c, i, _ in draw)
    assert seqs[0] == seqs[1] == seqs[2]


def test_mix_formula():
    p = [torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(2))]
    gn = [torch.tensor([3., 0., 0.]), torch.tensor([0., 4.])]
    ga = [torch.tensor([0., 1., 0.]), torch.tensor([0., 0.])]
    for q, b in zip(p, ga):
        q.grad = b.clone()
    st = g1.mix_(p, [x.clone() for x in gn], rho=0.25)
    assert st["nN"] == pytest.approx(5.0) and st["nA"] == pytest.approx(1.0)
    assert st["scale"] == pytest.approx(1.25)
    assert torch.allclose(p[0].grad, torch.tensor([3., 1.25, 0.]))
    assert torch.allclose(p[1].grad, torch.tensor([0., 4.]))
    assert st["cos"] == pytest.approx(0.0)


def test_weighted_ap_matches_sklearn():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(0)
    n_items, per = 6, 50
    item = np.repeat(np.arange(n_items), per)
    y = rng.random(n_items * per) < 0.2
    s = rng.random(n_items * per).astype(np.float32) + y * 0.3
    assert g1.weighted_ap(s, y, item, np.ones((1, n_items)))[0] == pytest.approx(average_precision_score(y, s))
    w = np.array([[2., 0., 1., 1., 3., 0.]])
    ref = average_precision_score(y, s, sample_weight=w[0][item])
    assert g1.weighted_ap(s, y, item, w)[0] == pytest.approx(ref)
