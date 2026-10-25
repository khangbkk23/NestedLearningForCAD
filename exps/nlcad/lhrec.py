# exps/nlcad/lhrec.py
"""NLCAD-03 Phase 1, LH-REC: record-only long-horizon credits on the identity (Replay-24) trajectory.

The fast loop is the locked `exps.rh01.run.train_stage_guarded`, called unchanged. Recording is attached through a
StableAdamW subclass (step() wrapped by a pre and a post hook) and a Batches subclass (remembers the last batch). Every
outer pass uses torch.autograd.grad (never touches .grad), dropout p = 0, a forked CPU/CUDA RNG and its own query
generator, so the trajectory must equal RH-01b R15 bitwise; that is checked against every stored R15 stage checkpoint
(hard stop on any difference). Nothing is learned and nothing feeds back.

Records (iteration i = 0-based optimizer step inside a stage; theta_i before step i, theta_{i+1} after it):
  every 8th step   J_q gradient at theta_{i+1}; credits <g_b, theta_{i+1} - anchor>_b for the 8-step window
                   (TBPTT-8) and, at their window ends, the K = 32/64/128 windows (anchors at multiples of K)
  every 32nd step  also: one-step credit (anchor theta_i), the pre-step J_q gradient at theta_i on the same queries
                   (first-order part <g_pre, Delta>), and the held-out-normal diagnostic credits (logged only)
J_q = global cosine on 8 current + 8 replay identities disjoint from the step's inner batch (BF16 autocast, FP32-core,
micro-batches of 8), as the NLCAD-01 normal look-ahead. Transient memory: CPU FP32 anchors for 8/32/64/128 plus the
one-step copy (at most 5 x 245.7 MB); GPU: the pre-step gradient is kept until the post hook (245.7 MB).

  python -m exps.nlcad.lhrec prereg
  python -m exps.nlcad.lhrec train
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nlcad.controller import BLOCKS, block_of
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

OUT = D.ROOT / "results/hope_cad/nlcad"
SEED, START = 2711, 6
TAG = f"LHREC_seed{SEED}_from{START}"
RUN = OUT / f"runs/{TAG}"
WINDOWS = (8, 32, 64, 128)
PRE_EVERY = 32
HELD_PER_CATEGORY = 5
TASK = "agents/tasks/task_nlcad_03_long_horizon_slow_signal.md"
SOURCES = ["exps/nlcad/lhrec.py", "exps/nlcad/lhrec_analyze.py", "exps/nlcad/test_lhrec.py", "exps/nlcad/chain_03.sh",
           "exps/nlcad/controller.py", "exps/rh01/run.py", "exps/rh01b/run.py", "exps/dinomaly/official.py",
           "exps/dinomaly/run.py"]
PREREG = OUT / "PREREG_03_LHREC.json"


def window_ends(i):
    """Window lengths that end after step i (theta_{i+1} is the end state)."""
    return [k for k in WINDOWS if (i + 1) % k == 0]


def window_starts(i):
    return [k for k in WINDOWS if i % k == 0]


class Recorder:
    def __init__(self, dec, current, replay, held, gen):
        self.dec, self.current, self.replay, self.held, self.gen = dec, current, replay, held, gen
        self.named = list(dec.named_parameters())
        self.blocks = [block_of(n) for n, _ in self.named]
        self.drops = [m for m in dec.modules() if isinstance(m, torch.nn.Dropout)]
        self.i, self.anchors, self.old, self.pre_grad, self.queries = 0, {}, None, None, None
        self.samplers, self.records, self.anchor_peak = [], [], 0

    # -- helpers -----------------------------------------------------------------------------------------------
    def _snapshot(self):
        return [p.detach().to("cpu", copy=True) for _, p in self.named]

    def _pick(self, n, exclude, k):
        pool = [j for j in range(n) if j not in set(exclude)]
        order = torch.randperm(len(pool), generator=self.gen).tolist()
        return [pool[j] for j in order[:k]]

    def _draw(self):
        cur, rep = self.samplers[0].last, self.samplers[1].last
        qc, qr = self._pick(len(self.current), cur, 8), self._pick(len(self.replay), rep, 8)
        return qc, qr, torch.cat([self.current[qc], self.replay[qr]])

    def _grad(self, x, micro=8):
        params = [p for _, p in self.named]
        saved = [m.p for m in self.drops]
        for m in self.drops:
            m.p = 0.0
        total, grads = 0.0, None
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                for xb in x.cuda().split(micro):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        en, de = self.dec(xb[:, 0], xb[:, 1])
                    loss = O.global_cosine([e.float() for e in en], [d.float() for d in de]) * len(xb) / len(x)
                    g = [torch.zeros_like(q) if u is None else u
                         for q, u in zip(params, torch.autograd.grad(loss, params, allow_unused=True))]
                    grads = g if grads is None else [u + v for u, v in zip(grads, g)]
                    total += float(loss)
        finally:
            for m, v in zip(self.drops, saved):
                m.p = v
        return total, grads

    def _credit(self, grads, anchor):
        """Per block: <g, theta_now - anchor>, ||theta_now - anchor||^2, ||g||^2 (FP64 accumulation)."""
        dot, dd, gg = ({b: 0.0 for b in BLOCKS} for _ in range(3))
        for (_, p), o, g, b in zip(self.named, anchor, grads, self.blocks):
            d = p.detach().double() - o.to(p.device, non_blocking=True).double()
            dot[b] += float((g.double() * d).sum())
            dd[b] += float((d * d).sum())
            gg[b] += float((g.double() ** 2).sum())
        return {"dot": dot, "disp_sq": dd, "grad_sq": gg}

    # -- hooks (called inside opt.step) --------------------------------------------------------------------------
    def pre(self, opt):
        i = self.i
        starts = window_starts(i)
        if starts:
            snap = self._snapshot()
            for k in starts:
                self.anchors[k] = snap                               # shared object across nested windows
        if (i + 1) % PRE_EVERY == 0:
            self.old = self._snapshot()
            self.queries = self._draw()
            self.pre_loss, self.pre_grad = self._grad(self.queries[2])
        live = {id(a) for a in self.anchors.values()} | ({id(self.old)} if self.old is not None else set())
        self.anchor_peak = max(self.anchor_peak, len(live))

    def post(self, opt):
        i = self.i
        ends = window_ends(i)
        if ends:
            rec = {"iter": i, "lr": float(opt.param_groups[0]["lr"]), "ends": ends}
            if (i + 1) % PRE_EVERY == 0:
                qc, qr, x = self.queries
            else:
                qc, qr, x = self._draw()
            rec["query_current"], rec["query_replay"] = qc, qr
            rec["inner_current"], rec["inner_replay"] = list(self.samplers[0].last), list(self.samplers[1].last)
            rec["J_q"], g = self._grad(x)
            rec["q"] = {f"K{k}": self._credit(g, self.anchors[k]) for k in ends}
            if (i + 1) % PRE_EVERY == 0:
                rec["q"]["onestep"] = self._credit(g, self.old)
                rec["q"]["onestep_pre"] = self._credit(self.pre_grad, self.old)
                rec["J_q_pre"] = self.pre_loss
                del g
                self.pre_grad = None
                rec["J_held"], gh = self._grad(self.held)
                rec["held"] = {f"K{k}": self._credit(gh, self.anchors[k]) for k in ends}
                rec["held"]["onestep"] = self._credit(gh, self.old)
                del gh
                self.old = None
            self.records.append(rec)
        for k in ends:
            self.anchors.pop(k, None)
        self.i += 1


class RecordingStableAdamW(O.StableAdamW):
    recorder = None

    def step(self, closure=None):
        if self.recorder is not None:
            self.recorder.pre(self)
        out = super().step(closure)
        if self.recorder is not None:
            self.recorder.post(self)
        return out


class RecordingBatches(DR.Batches):
    registry = None

    def __init__(self, n, size, gen):
        super().__init__(n, size, gen)
        self.last = []
        if RecordingBatches.registry is not None:
            RecordingBatches.registry.append(self)

    def next(self):
        self.last = super().next()
        return self.last


def make_recording_optimizer(trainable):
    """O.make_optimizer, argument for argument, with the recording subclass."""
    r = O.RECIPE
    return RecordingStableAdamW([{'params': trainable.parameters()}], lr=r["lr"], betas=r["betas"],
                                weight_decay=r["weight_decay"], amsgrad=r["amsgrad"], eps=r["eps"])


def recorded_stage(dec, current, iters, gen, replay, recorder):
    """Run the locked train_stage_guarded with the recording optimizer and sampler patched in for this call only."""
    saved = (O.make_optimizer, DR.Batches)
    RecordingStableAdamW.recorder, RecordingBatches.registry = recorder, recorder.samplers
    O.make_optimizer, DR.Batches = make_recording_optimizer, RecordingBatches
    try:
        return R1.train_stage_guarded(dec, current, iters, gen, replay)
    finally:
        O.make_optimizer, DR.Batches = saved
        RecordingStableAdamW.recorder, RecordingBatches.registry = None, None


def prereg():
    require(not PREREG.exists(), "already preregistered")
    D.write_json(PREREG, {
        "utc": DR.now(), "task": {TASK: sha(D.ROOT / TASK)}, "sources": {s: sha(D.ROOT / s) for s in SOURCES},
        "trajectory": "identity: locked train_stage_guarded, RH-01b R15 seed 2711 stages 7-15 from its task-6 checkpoint",
        "integrity": "every stage state bitwise equal to the stored RH-01b R15 checkpoint (hard stop)",
        "windows": WINDOWS, "pre_every": PRE_EVERY, "held_per_category": HELD_PER_CATEGORY,
        "query_generator_seed": "seed*1000 + 800 + stage", "outer": "global cosine, BF16 autocast, FP32-core, micro 8, dropout 0",
        "analysis": {
            "a_agreement": "per 32-period: centred cosine of sum of 4 TBPTT-8 window credits vs K32 credit and vs one-step "
                           "credit at the period end; K64/K128 vs the matching TBPTT-8 sums",
            "b_decomposition": "one-step = <g_post, D>; first order = <g_pre, D>; curvature = difference; shares per LR phase",
            "c_objective": "centred cosine and global sign agreement of J_q vs held-out credits, per credit type",
            "d_task_dependence": "per-stage mean centred TBPTT-8 direction (period sums); cross-stage cosine; same for one-step",
            "e_overhead": "stage seconds vs RH-01b R15 seed 2712 stage seconds (GPU alone)"},
        "go_no_go": {
            "stable": "median centred cosine between TBPTT-8 credits of consecutive 32-periods within a stage > 0",
            "differs_a": "median centred cosine(TBPTT-8 period sum, one-step at period end) < 0.5",
            "differs_b": "median curvature share of the one-step global signal > 0.5",
            "differs_d": "mean cross-stage cosine of TBPTT-8 < mean cross-stage cosine of one-step - 0.1",
            "go": "stable and (differs_a or differs_b or differs_d)",
            "computed_on": "32-periods starting at iteration >= 100 (16 of 20 per stage; the Phase 2 design skips the "
                           "warm-up periods); all-period values are reported as secondary"},
        "lambda_LH": "1e-5 * median|TBPTT-8 period-sum credit| / median|one-step credit| (block level, a = 1)",
        "phase2": "not launched automatically; requires the user's go after this analysis"})
    print("preregistered", sha(PREREG), flush=True)


def verify_prereg():
    pr = D.read(PREREG)
    for s, h in pr["sources"].items():
        require(sha(D.ROOT / s) == h, f"source changed after preregistration: {s}")
    return pr


def train():
    verify_prereg()
    DR.configure()
    cap = float(os.environ.get("LHREC_CAP_GIB", "4.0"))
    torch.cuda.set_per_process_memory_fraction(cap * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    proto = RB.verify_lock()
    RB.numerics_d4prime()
    RUN.mkdir(parents=True, exist_ok=True)
    n = R1.n_train(proto)
    held_stats = {c: DR.load_split("heldout", c) for c in C.EVAL}
    held_q = torch.cat([DR.load_split("heldout", c)[:HELD_PER_CATEGORY] for c in C.EVAL])
    rec0 = D.read(C.OUT / f"runs/R15_seed{SEED}/STAGE{START}.json")
    require(sha(D.ROOT / rec0["checkpoint"]) == rec0["checkpoint_sha256"], "start checkpoint changed")
    dec = DR.new_decoder(SEED)
    dec.load_state_dict(torch.load(D.ROOT / rec0["checkpoint"], weights_only=True)["state"])
    torch.cuda.reset_peak_memory_stats()
    for s in range(START + 1, len(C.STREAM) + 1):
        c = C.STREAM[s - 1]
        js = RUN / f"STAGE{s}.json"
        ref = D.read(C.OUT / f"runs/R15_seed{SEED}/STAGE{s}.json")
        ref_path = D.ROOT / ref["checkpoint"]
        require(sha(ref_path) == ref["checkpoint_sha256"], f"R15 checkpoint changed: stage {s}")
        if js.exists():                                                  # resume from the (bitwise-equal) R15 state
            require(D.read(js)["bitwise_equal_to_R15"], f"stage {s} recorded as not equal")
            dec.load_state_dict(torch.load(ref_path, weights_only=True)["state"])
            continue
        done = list(C.STREAM[:s - 1])
        idx = C.buffer_index(SEED, n, done)
        replay = torch.stack([C.load_features(d)[i] for d in done for i in idx[d]])
        current = C.load_features(c)
        dseed, gseed = RB.stage_seeds("R15", SEED, s)
        torch.manual_seed(dseed)
        gen = torch.Generator().manual_seed(gseed)
        recorder = Recorder(dec, current, replay, held_q, torch.Generator().manual_seed(SEED * 1000 + 800 + s))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        losses, skips = recorded_stage(dec, current, R1.ITERS, gen, replay, recorder)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        ref_state = torch.load(ref_path, weights_only=True)["state"]
        mine = {k: v.detach().cpu() for k, v in dec.state_dict().items()}
        diff = max(float((mine[k].double() - ref_state[k].double()).abs().max()) for k in ref_state)
        equal = set(mine) == set(ref_state) and all(torch.equal(mine[k], ref_state[k]) for k in ref_state)
        D.write_json(RUN / f"RECORDS{s}.json", recorder.records)
        D.write_json(js, {"index": s, "name": c, "seconds": seconds, "nonfinite_skips": skips,
                          "optimizer_steps": recorder.i, "records": len(recorder.records),
                          "anchor_peak_live": recorder.anchor_peak,
                          "anchor_bytes_each": int(sum(p.numel() * 4 for p in dec.parameters())),
                          "bitwise_equal_to_R15": equal, "max_abs_diff_to_R15": diff,
                          "R15_checkpoint_sha256": ref["checkpoint_sha256"],
                          "loss_curve_every50": [float(np.nanmean(losses[i:i + 50])) for i in range(0, len(losses), 50)],
                          "heldout_normal": DR.normal_stats(dec, held_stats),
                          "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "utc": DR.now()})
        print("LHREC", s, c, f"{seconds:.0f}s", "records", len(recorder.records), "steps", recorder.i,
              "equal", equal, f"maxdiff {diff:.3e}", "anchors", recorder.anchor_peak, flush=True)
        require(equal, f"stage {s}: trajectory differs from RH-01b R15 (max |diff| {diff:.3e}); hard stop")
        require(recorder.i == R1.ITERS and not skips, f"stage {s}: step accounting {recorder.i}, skips {len(skips)}")
        del replay, current


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prereg", "train"))
    a = ap.parse_args()
    {"prereg": prereg, "train": train}[a.stage]()


if __name__ == "__main__":
    main()
