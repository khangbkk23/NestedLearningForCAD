"""NL-RET-02 shared definitions: boundary rows, causal probe identities (exact R15/FS_B sampler replay), lineage.

Boundary s -> s+1 (s = 1..14). Observation point: theta_s (verified end-of-stage-s checkpoint), M_s = Replay-24 members
held at the end of s (= the buffer replayed during s+1), and the incoming task t = s+1's normal training set, which is
loaded in full (`current = load_features(c)`) BEFORE the first optimizer step of stage t in both R15
(exps/rh01b/run.py::train) and FS_B (exps/nl_align_01/g1.py::train, exps/nl_align_02/m3.py). Row (c, s): old category c
with a(c) <= s, outcome over s -> s+1 (the NL-RET-01A rows and targets, unchanged).

Probe identities: the incoming-task halves of the first K training minibatches of stage t, regenerated with the very
same code path (DR.Batches and the stage generator `RB.stage_seeds('R15', seed, t)`; FS_B uses the identical call),
so they are exactly the images the historical run drew first. All of them belong to the arrival dataset available
before update 1; with K <= floor(n_t / 8) batches they are distinct (first epoch, no replacement).
"""
from __future__ import annotations

import torch

from exps.dinomaly import run as DR
from exps.dinomaly.common import require
from exps.nl_ret_01a import common as C1

ROOT = C1.ROOT
OUT = ROOT / "results/hope_cad/nl_ret_02"
EXT = OUT / "extract"
OUT01 = C1.OUT
STREAM, SEEDS, CAPACITY = C1.STREAM, C1.SEEDS, C1.CAPACITY
a_of, members, nonmembers, ever_members, proto_and_n = C1.a_of, C1.members, C1.nonmembers, C1.ever_members, C1.proto_and_n
checkpoint, local_ckpt, owner = C1.checkpoint, C1.local_ckpt, C1.owner
HALF = 8                      # batch_size 16 // 2 (replay present from stage 2 on)


def boundaries(arm):
    """Observation states s whose theta_s / outcome are owned by `arm` for boundary s -> s+1.
    ID: s = 1..14. FS_B: s = 6..14 (outcome s -> s+1 distinct from ID from s = 6; theta_s distinct from s = 7)."""
    return list(range(1, 15)) if arm == "ID" else list(range(6, 15))


def rows(arm):
    out = []
    for s in boundaries(arm):
        for c in STREAM[:s]:
            out.append({"arm": arm, "category": c, "s": s, "incoming": STREAM[s],
                        "state_owner": owner(arm, s), "state_distinct_from_ID": arm == "FS_B" and s >= 7,
                        "outcome_distinct_from_ID": arm == "FS_B" and s >= 6})
    return out


def stage_gens(seed, t):
    from exps.rh01b import run as RB
    dseed, gseed = RB.stage_seeds("R15", seed, t)
    return dseed, torch.Generator().manual_seed(gseed)


def sampler_batches(seed, t, n_cur, n_rep, K):
    """First K iterations of stage t's sampler, replayed exactly: [(current idx (8), replay row idx (8), perm (16))]."""
    require(t >= 2 and n_rep >= HALF, "replay present from stage 2")
    _, gen = stage_gens(seed, t)
    cur, rep = DR.Batches(n_cur, HALF, gen), DR.Batches(n_rep, HALF, gen)
    out = []
    for _ in range(K):
        ci = cur.next()
        ri = rep.next()
        perm = torch.randperm(2 * HALF, generator=gen).tolist()
        out.append((ci, ri, perm))
    return out


def replay_rows(seed, n, s):
    """Replay tensor row order used by the training code at stage s+1: [(category, train index)]."""
    m = members(seed, n, s)
    return [(d, i) for d in STREAM[:s] for i in m[d]]


def probe(seed, n, s, K):
    """Probe record for boundary s -> s+1: incoming identities (first K current halves, deduplicated in order) and
    the exact first training batch (8 incoming + 8 replay members, permuted as in training)."""
    t = s + 1
    c_in = STREAM[t - 1]
    rr = replay_rows(seed, n, s)
    bs = sampler_batches(seed, t, n[c_in], len(rr), K)
    inc = []
    for ci, _, _ in bs:
        inc += [i for i in ci if i not in inc]
    ci, ri, perm = bs[0]
    first = [("cur", c_in, ci[j]) if j < HALF else ("rep",) + rr[ri[j - HALF]] for j in perm]
    first = [(x[1], x[2]) for x in first]
    return {"boundary": f"{s}->{t}", "incoming": c_in, "incoming_idx": inc, "first_batch": first,
            "first_batch_replay": [rr[j] for j in ri]}
