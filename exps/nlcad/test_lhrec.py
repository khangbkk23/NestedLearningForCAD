# exps/nlcad/test_lhrec.py
"""NLCAD-03 Phase 1 checks: window schedule, credit arithmetic, and (GPU) that recording leaves the locked
train_stage_guarded trajectory bitwise unchanged while producing the expected records."""
from __future__ import annotations

import pytest
import torch

from exps.dinomaly import official as O
from exps.nlcad import lhrec as L
from exps.nlcad.controller import BLOCKS
from exps.nlcad.test_nlcad import feats, state, tiny


def test_window_schedule():
    ends = {i: L.window_ends(i) for i in range(256)}
    assert ends[7] == [8] and ends[31] == [8, 32] and ends[63] == [8, 32, 64] and ends[127] == [8, 32, 64, 128]
    assert sum(1 for i in range(667) if 8 in L.window_ends(i)) == 83
    assert sum(1 for i in range(667) if 32 in L.window_ends(i)) == 20
    assert L.window_starts(0) == [8, 32, 64, 128] and L.window_starts(8) == [8]


def test_credit_is_blockwise_dot_with_displacement():
    dec = tiny(3)
    rec = L.Recorder(dec, feats(16, 1), feats(8, 2), feats(4, 3), torch.Generator().manual_seed(0))
    anchor = rec._snapshot()
    with torch.no_grad():
        for _, p in dec.named_parameters():
            p.add_(0.01 * torch.randn_like(p))
    grads = [torch.randn_like(p) for _, p in dec.named_parameters()]
    c = rec._credit(grads, anchor)
    for b in BLOCKS:
        want = sum(float((g.double() * (p.detach().double() - o.double())).sum())
                   for (n, p), o, g in zip(dec.named_parameters(), anchor, grads) if n.startswith(b + "."))
        assert abs(c["dot"][b] - want) < 1e-9 * max(1.0, abs(want))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_recording_leaves_locked_trajectory_bitwise_unchanged():
    from exps.rh01 import numerics as NUM
    from exps.rh01 import run as R1
    torch.cuda.set_per_process_memory_fraction(2.0 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    NUM.use(True)
    cur, rep, held = feats(64, 1), feats(16, 2), feats(15, 3)
    outs = []
    for mode in ("locked", "recorded"):
        torch.manual_seed(0)
        dec = O.Decoder(32, 4, checkpoint=True).cuda()
        O.init_trainable(dec)
        torch.manual_seed(7)
        gen = torch.Generator().manual_seed(3)
        if mode == "locked":
            losses, skips = R1.train_stage_guarded(dec, cur, 64, gen, rep)
            records = None
        else:
            r = L.Recorder(dec, cur, rep, held, torch.Generator().manual_seed(11))
            losses, skips = L.recorded_stage(dec, cur, 64, gen, rep, r)
            records = r.records
            assert r.i == 64 and r.anchor_peak <= 5
        outs.append((state(dec), losses, skips, records))
    NUM.use(False)
    assert O.make_optimizer is not L.make_recording_optimizer                 # patch removed
    (s0, l0, k0, _), (s1, l1, k1, recs) = outs
    assert all(torch.equal(a, b) for a, b in zip(s0.values(), s1.values())) and l0 == l1 and k0 == k1 == []
    assert [x["iter"] for x in recs] == [7, 15, 23, 31, 39, 47, 55, 63]
    assert recs[3]["ends"] == [8, 32] and set(recs[3]["q"]) == {"K8", "K32", "onestep", "onestep_pre"}
    assert recs[7]["ends"] == [8, 32, 64] and "K64" in recs[7]["held"]
    for x in recs:
        assert len(x["query_current"]) == len(x["query_replay"]) == 8
        assert not set(x["query_current"]) & set(x["inner_current"]) and not set(x["query_replay"]) & set(x["inner_replay"])


def _fake_records(seed):
    rng = torch.Generator().manual_seed(seed)
    cred = lambda: {"dot": {b: float(torch.randn(1, generator=rng)) for b in BLOCKS}, "disp_sq": {}, "grad_sq": {}}
    recs = []
    for i in range(667):
        ends = L.window_ends(i)
        if not ends:
            continue
        r = {"iter": i, "lr": 1e-3, "ends": ends, "q": {f"K{k}": cred() for k in ends}}
        if (i + 1) % 32 == 0:
            r["q"]["onestep"], r["q"]["onestep_pre"] = cred(), cred()
            r["held"] = {f"K{k}": cred() for k in ends} | {"onestep": cred()}
        recs.append(r)
    return recs


def test_analysis_shapes_on_synthetic_records():
    from exps.nlcad import lhrec_analyze as A
    per = {s: A.stage_analysis(_fake_records(s)) for s in range(7, 10)}
    assert per[7]["periods"] == 20 and per[7]["post_warmup"]["n"] == 16
    assert len(per[7]["all"]["cos_tbptt_K64"]) == 10 and len(per[7]["all"]["cos_tbptt_K128"]) == 5
    cs = A.cross_stage(per, "tbptt_dirs", "post_warmup")
    assert -1 <= cs["mean"] <= 1 and len(cs["matrix"]) == 3
    # tbptt sum of a period equals the sum of its four K8 credits
    recs = _fake_records(1)
    P = A.periods(recs)
    by = {r["iter"]: r for r in recs}
    want = sum(A.vec(by[i]["q"]["K8"]) for i in (39, 47, 55, 63))
    assert all(abs(x - y) < 1e-12 for x, y in zip(P[1]["tbptt"], want))
