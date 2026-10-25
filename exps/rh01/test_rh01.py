# exps/rh01/test_rh01.py
"""CPU pre-run checks for RH-01 (no development image, no mask)."""
from __future__ import annotations

import os

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.rh01 import common as C
from exps.rh01 import run as RR


def counts():
    n = {c: len(os.listdir(D.DATA / c / "train/good")) for c in C.STREAM}
    n.update({"bottle": 150, "carpet": 100, "hazelnut": 100})
    return n


def test_allocation_full_balanced_nested_nonincreasing():
    n = counts()
    for seed in RR.SEEDS:
        prev = None
        for k in range(1, len(C.STREAM)):
            done = list(C.STREAM[:k])
            idx = C.buffer_index(seed, n, done)
            sizes = [len(v) for v in idx.values()]
            assert sum(sizes) == C.CAPACITY and max(sizes) - min(sizes) <= 1
            if prev is not None:
                for c in prev:                                        # nested prefixes, quota never grows
                    assert len(idx[c]) <= len(prev[c]) and idx[c] == prev[c][:len(idx[c])]
            prev = idx


def test_first_boundaries_match_pilot_rule_and_identities():
    n = counts()
    proto = D.identities()
    for seed in RR.SEEDS:
        assert [len(C.buffer_index(seed, n, list(C.STREAM[:k]))["bottle"]) for k in (1, 2, 3)] == [24, 12, 8]
        idx = C.buffer_index(seed, n, ["bottle"])
        ids = [proto["train"]["bottle"][i]["relative_path"] for i in idx["bottle"]]
        rec = D.read(C.DRF / f"runs/B_seed{seed}/TRAIN.json")
        assert ids == rec["stages"][0]["replay"]["identities"]       # pilot B stage-2 buffer, exactly


def test_pool_matches_concatenation_with_mmap(tmp_path):
    a, b = torch.randn(5, 2, 3, 4), torch.randn(7, 2, 3, 4)
    for name, t in (("a", a), ("b", b)):
        torch.save({"features": t}, tmp_path / f"{name}.pt")
    pool = C.Pool([torch.load(tmp_path / f"{x}.pt", weights_only=True, mmap=True)["features"] for x in "ab"])
    idx = [0, 11, 5, 4, 6, 3]
    assert len(pool) == 12 and torch.equal(pool[idx], torch.cat([a, b])[idx])


def test_sampler_exposure():
    gen = torch.Generator().manual_seed(0)
    cc, rc = RR.simulate(100, 667, gen, n_replay=24)
    assert cc.sum() == rc.sum() == 667 * 8 and rc.min() >= 222 and rc.max() <= 223
    gen = torch.Generator().manual_seed(0)
    cj, _ = RR.simulate(3099, RR.JOINT_ITERS, gen)
    # 193 full batches per epoch (11 of 3,099 dropped per epoch, as the official drop_last loader): 51.8 epochs.
    assert cj.sum() == RR.JOINT_ITERS * 16 and cj.max() <= 52 and cj.min() >= 45


def test_scheduler_and_ramp():
    for iters, ramp in ((RR.ITERS, 67), (RR.JOINT_ITERS, 1000)):
        p = torch.nn.Parameter(torch.zeros(1))
        opt = O.StableAdamW([p], lr=2e-3)
        sch = O.make_scheduler(opt, iters)
        lrs = []
        for _ in range(iters):
            lrs.append(opt.param_groups[0]["lr"])
            sch.step()
        assert lrs[0] == 0 and abs(lrs[100] - 2e-3) < 1e-12 and 2e-4 < lrs[-1] < 2.1e-4
        assert round(0.1 * iters) == ramp


def test_stage_seeds_distinct():
    for seed in RR.SEEDS:
        gens = [RR.stage_seeds("R15", seed, s)[1] for s in range(2, 16)] + [RR.stage_seeds("J15", seed, 1)[1]]
        drops = [RR.stage_seeds("R15", seed, s)[0] for s in range(2, 16)]
        assert len(set(gens)) == len(gens) and len(set(drops)) == len(drops)


# --- AMENDMENT_1: skip-step guard (GPU; tiny decoder, synthetic features only) ------------------------

import pytest

from exps.dinomaly import run as DR


def _tiny(seed=0):
    torch.manual_seed(seed)
    dec = O.Decoder(32, 4).cuda()
    O.init_trainable(dec)
    return dec


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_guarded_stage_equals_locked_stage_on_finite_data():
    g = torch.Generator().manual_seed(5)
    feats, rep = torch.randn(32, 2, 21, 32, generator=g), torch.randn(8, 2, 21, 32, generator=g)
    out = []
    for fn in (DR.train_stage, RR.train_stage_guarded):
        dec = _tiny()
        torch.manual_seed(1)
        res = fn(dec, feats, 6, torch.Generator().manual_seed(2), rep)
        out.append(([p.detach().clone() for p in dec.parameters()], res))
    losses, skips = out[1][1]
    assert skips == [] and np.allclose(out[0][1], losses, rtol=0, atol=1e-7)
    assert all(torch.allclose(a, b, rtol=0, atol=1e-7) for a, b in zip(out[0][0], out[1][0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_guard_skips_without_update_and_stops_after_consecutive_limit():
    feats = torch.full((32, 2, 21, 32), float("nan"))
    dec = _tiny()
    before = [p.detach().clone() for p in dec.parameters()]
    with pytest.raises(RuntimeError, match="exceed limits"):
        RR.train_stage_guarded(dec, feats, 6, torch.Generator().manual_seed(2))
    assert all(torch.equal(a, b) for a, b in zip(before, dec.parameters()))
