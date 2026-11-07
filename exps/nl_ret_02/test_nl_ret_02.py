"""NL-RET-02 invariants. Mechanical correctness only (passing them proves nothing scientific)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from exps.nl_ret_02 import common as C

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def pn():
    return C.proto_and_n()


def test_boundary_order_and_coverage():
    r = C.rows("ID")
    assert len(r) == 105 and C.boundaries("ID") == list(range(1, 15))
    for x in r:
        assert C.a_of(x["category"]) <= x["s"] and x["incoming"] == C.STREAM[x["s"]]   # incoming = task s+1
    for c in C.STREAM[:-1]:                                                          # every acquired category covered
        assert [x["s"] for x in r if x["category"] == c] == list(range(C.a_of(c), 15))
    f = C.rows("FS_B")
    assert len(f) == 90 and sum(x["state_distinct_from_ID"] for x in f) == 84 and \
        all(x["outcome_distinct_from_ID"] for x in f)
    assert C.local_ckpt("FS_B", 2711, 6) == C.local_ckpt("ID", 2711, 6)               # shared state, not double-owned


def test_probe_identities_are_lawful(pn):
    proto, n = pn
    for seed in C.SEEDS:
        for s in range(1, 15):
            p = C.probe(seed, n, s, 4)
            assert p["incoming"] == C.STREAM[s] and len(set(p["incoming_idx"])) == 32
            assert all(0 <= i < n[p["incoming"]] for i in p["incoming_idx"])
            mem = C.members(seed, n, s)
            for c, i in p["first_batch"]:
                assert c == p["incoming"] or (c in C.STREAM[:s] and i in mem[c])       # only M_s, never nonmembers
                if c != p["incoming"]:
                    assert i not in C.nonmembers(seed, n, c)
            cur = [i for c, i in p["first_batch"] if c == p["incoming"]]
            assert sorted(cur) == sorted(p["incoming_idx"][:8])                        # batch 1 inside the window


@GPU
def test_sampler_replay_matches_training_loop(pn):
    """Run the real R15 training function on a dummy decoder whose inputs encode identities; the batches it draws
    must equal C.sampler_batches (same generator path)."""
    from exps.rh01 import run as R1

    class Dum(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1, device="cuda"))
            self.seen = []

        def forward(self, a, b):
            self.seen += a.flatten(1)[:, 0].round().long().tolist()
            B = len(a)
            e = torch.ones(B, 2, 2, 2, device="cuda") + 0 * a.view(B, 1, 1, 1).float()
            d = self.w * torch.arange(1, 9, device="cuda", dtype=torch.float).view(1, 2, 2, 2).expand(B, -1, -1, -1)
            return [e], [d]

    proto, n = pn
    for seed, s in ((2711, 1), (2712, 7), (2713, 13)):
        t = s + 1
        nc, nr = n[C.STREAM[s]], len(C.replay_rows(seed, n, s))
        cur = torch.arange(nc, dtype=torch.float).view(nc, 1, 1, 1).expand(nc, 2, 1, 1).contiguous()
        rep = (1000 + torch.arange(nr, dtype=torch.float)).view(nr, 1, 1, 1).expand(nr, 2, 1, 1).contiguous()
        dum = Dum()
        _, gen = C.stage_gens(seed, t)
        R1.train_stage_guarded(dum, cur, 667, gen, rep)          # real iteration count (ramp > 0)
        want = []
        for ci, ri, perm in C.sampler_batches(seed, t, nc, nr, 4):
            v = ci + [1000 + j for j in ri]
            want += [v[j] for j in perm]
        assert dum.seen[:len(want)] == want


def test_fsb_sampler_call_order_is_identical():
    src = Path("exps/nl_align_01/g1.py").read_text()
    body = src.split("def train_stage(")[1].split("\ndef ")[0]
    assert "cur = DR.Batches(len(current), batch // 2, gen)" in body and "rep = DR.Batches(len(replay), batch // 2, gen)" in body
    assert body.index("ci, ri = cur.next(), rep.next()") < body.index("torch.randperm(len(x), generator=gen)")
    assert 'RB.stage_seeds("R15", seed, s)' in src


@GPU
def test_gradient_reduction_and_no_update(pn):
    from exps.nl_ret_02 import probe as P
    P.setup()
    proto, n = pn
    fe = P.Feats()
    dec = P.decoder("ID", 2711, 5)
    h0 = P.param_hash(dec)
    params = [p for p in dec.parameters() if p.requires_grad]
    x = fe.get("bottle", list(range(16)))
    g, lv = P.grad(dec, params, x)
    g1, _ = P.grad(dec, params, x[:8])
    g2, _ = P.grad(dec, params, x[8:])
    assert torch.allclose(g, 0.5 * (g1 + g2), rtol=1e-4, atol=1e-9)            # micro-batch weighting len(xb)/len(x)
    g3, _ = P.grad(dec, params, x[:3])
    g4, _ = P.grad(dec, params, x[3:8])
    assert float(torch.nn.functional.cosine_similarity((3 * g3 + 5 * g4) / 8, g1, 0)) > 0.9999   # unequal parts
    assert P.param_hash(dec) == h0 and all(p.grad is None for p in params)        # no update, no stale grads
    assert lv.shape == (16,)


def test_small_replay_handling():
    from exps.nl_ret_02 import probe as P
    a = torch.nn.functional.normalize(torch.randn(32, 6), dim=1)
    b = torch.nn.functional.normalize(torch.randn(2, 6), dim=1)
    assert 0 <= P.principal_overlap(a, torch.cat([b, b[:1] + 0.1]), 2) <= 1 + 1e-6
    assert P.nn_dist(b, a).shape == (2,)


def test_probe_reads_no_future_or_labels():
    src = Path("exps/nl_ret_02/probe.py").read_text()
    assert not re.search(r"native_mask|mask_path|ground_truth|/test/|OUTCOME|Y1|Y4|STAGE_TRAJECTORY", src)
    assert "local_ckpt(arm, seed, s)" in src and "s + 1)" not in src.replace("C.probe(seed, n, s, K)", "") \
        .replace("aux = G.Aux(\"FS_B\", items, hosts, role, seed, n, s + 1", "")
    assert "nonmembers(" not in src


def test_prereg_information_sets_nested():
    from exps.nl_ret_02 import prereg as R
    assert set(R.PRIMARY) <= set(R.INFO_SETS)
    assert not any(f.startswith(("Y", "ORACLE")) for f in R.GN + R.GE + R.FN + R.I0)
    assert not set(R.FSQ) & set(R.GN + R.GE + R.FN)


def test_information_sets_are_nested_and_lawful():
    from exps.nl_ret_02 import validate as W
    S = W.sets()
    assert set(S["I0"]) < set(S["I1"]) < set(S["I4"]) < set(S["I5"])
    assert set(S["I2"]) | set(S["I3"]) == set(S["I4"]) and set(S["I0+GE+GN"]) < set(S["I4"])
    for cols in S.values():
        assert len(cols) == len(set(cols))
        assert not any(c.startswith(("Y", "ORACLE", "FSQ_")) or "nonmember" in c for c in cols)


def test_shuffle_incoming_touches_training_rows_only():
    from exps.nl_ret_02 import validate as W
    rng = np.random.default_rng(0)
    D = pd.DataFrame({"category": np.repeat(C.STREAM[:5], 12), "AGE_BUCKET": np.tile([0, 1, 2, 3], 15),
                      "x": rng.normal(size=60), "inc": rng.normal(size=60)})
    D["y"] = D.x + D.inc
    tr, te = D.index[D.category != "bottle"], D.index[D.category == "bottle"]
    p_true = W.fit_predict(D, tr, te, ("ridge", ["x", "inc"]), "y", rng)
    p_shuf = W.fit_predict(D, tr, te, ("shuf_inc", ["x", "inc"], ["inc"]), "y", rng)
    D2 = D.copy(); D2.loc[te, "y"] = 1e6
    assert np.allclose(W.fit_predict(D2, tr, te, ("shuf_inc", ["x", "inc"], ["inc"]), "y", np.random.default_rng(0)),
                       W.fit_predict(D, tr, te, ("shuf_inc", ["x", "inc"], ["inc"]), "y", np.random.default_rng(0)))
    assert np.abs(p_true - D.loc[te, "y"]).mean() < np.abs(p_shuf - D.loc[te, "y"]).mean()


def test_alignment_reads_y4_only_after_freeze():
    src = Path("exps/nl_ret_02/validate.py").read_text()
    before = src.split("def run_align")[0]
    assert "Y4_exposed\" not in O" in before and before.count("Y4_exposed") == 1   # only the guard forbidding it
    assert "Y4" not in Path("exps/nl_ret_02/tables.py").read_text().replace("Y4 is NOT copied", "") \
        .replace('"y4_in_outcome_table": False', "")
