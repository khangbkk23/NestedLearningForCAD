"""NL-SGM-01A-AM02/v2 runner. One stage per process (per-arm peak RSS and wall are then exact).

    python -m exps.nl_sgm_01a.am02_run lock            # create-exclusive LOCK.json (no outcome)
    python -m exps.nl_sgm_01a.am02_run a0              # frozen references W_c + per-category oracle minima
    python -m exps.nl_sgm_01a.am02_run arm C1_FPS_FP32 | C3_HASH_PQ | C2_FPS_PQ
    python -m exps.nl_sgm_01a.am02_run c4              # same-identity PQ from C1 checkpoints + codec diagnostics

Every stage: verifies LOCK, enforces caps (wall per stage, peak RSS, disk), writes boundary checkpoints and
per-boundary outputs (partial outputs survive a stop), and raises on any integrity/exactness failure.
Only EVAL probes are scored against arms; SEL probes are scored only against W_c (scales).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time

import numpy as np
import torch

from exps.csg01.select import random_priorities
from exps.nl_sgm_01a import am02 as A
from exps.nl_sgm_01a import am02_eval as E
from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a import exact as X
from exps.nl_sgm_01a import lock as L
from exps.nl_sgm_01a import rules as R
from exps.task_p.p0_normal_support import require

OUT = C.ROOT / "results/hope_cad/nl_sgm_01a_am02"
LOCKP = OUT / "LOCK_r2.json"          # r2: implementation fix (rules.CodecArm._prune); LOCK.json = r1, preserved
CAPS = {"per_arm_h": 4.0, "total_wall_h": 16.0, "rss_gb": 12.0, "disk_gb": 5.0}
AM02_SOURCES = L.SOURCES + [
    "agents/plans/nl_sgm_01a_am02_plan.md", "agents/plans/nl_sgm_01a_am02_plan_v2.md",
    "exps/nl_sgm_01a/am02.py", "exps/nl_sgm_01a/am02_eval.py", "exps/nl_sgm_01a/am02_run.py",
    "exps/nl_sgm_01a/am02_analyze.py", "exps/nl_sgm_01a/am02_bench.py", "exps/nl_sgm_01a/test_am02.py",
    "results/hope_cad/nl_sgm_01a_am02/BENCH.json", "results/hope_cad/nl_sgm_01a_am02/RESOURCE_PROJECTION.json",
]
SPOT_T = (0, 7, 14)


class Stage:
    def __init__(self, name):
        self.name, self.t0 = name, time.time()
        self.dir = OUT / "stages" / name
        self.dir.mkdir(parents=True, exist_ok=False)          # a stage never silently re-runs
        self.logf = (OUT / "logs" / f"{name}.log").open("x")
        self.events = []

    def log(self, *a):
        s = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
        print(s, flush=True); self.logf.write(s + "\n"); self.logf.flush()

    def check(self, where):
        wall_h = (time.time() - self.t0) / 3600
        rss = C.peak_rss_gb()
        disk = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e9
        rec = {"where": where, "wall_h": wall_h, "peak_rss_gb": rss, "disk_gb": disk}
        self.events.append(rec)
        require(wall_h <= CAPS["per_arm_h"], f"CAP: {self.name} wall {wall_h:.2f} h > {CAPS['per_arm_h']} h")
        require(rss <= CAPS["rss_gb"], f"CAP: peak RSS {rss:.2f} GB > {CAPS['rss_gb']} GB")
        require(disk <= CAPS["disk_gb"], f"CAP: disk {disk:.2f} GB > {CAPS['disk_gb']} GB")
        return rec

    def finish(self, extra):
        rec = dict(stage=self.name, wall_s=time.time() - self.t0, peak_rss_gb=C.peak_rss_gb(),
                   device="cpu", threads=torch.get_num_threads(), events=self.events, **extra)
        C.write_json(self.dir / "STAGE_DONE.json", rec)
        self.log("DONE", json.dumps({k: v for k, v in rec.items() if k != "events"}, default=str))


def setup():
    C.set_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    require(not torch.cuda.is_initialized(), "CUDA must not be used (CPU lock)")
    lock = L.verify_lock(LOCKP)
    co = C.cohorts()
    require(C.r15_digest()[:8] == lock["cohort_r15_digest"][:8], "cohort digest mismatch")
    return lock, co, {c: len(co[c]["writers"]) for c in C.ORDER}


def gid_base(n, c):
    i = C.ORDER.index(c)
    return sum(n[C.ORDER[j]] for j in range(i)) * C.PATCHES


def writer_gids(n, c):
    b = gid_base(n, c)
    return torch.arange(b, b + n[c] * C.PATCHES)


# ------------------------------------------------------------------ LOCK

def do_lock():
    require(not LOCKP.exists(), "LOCK exists")
    payload = {
        "task": "NL-SGM-01A-AM02", "version": A.VERSION.replace("v1", "v2"), "plan": "agents/plans/nl_sgm_01a_am02_plan_v2.md",
        "arms": {"C1_FPS_FP32": A.C1_CAPACITY, "C2_FPS_PQ": A.PQ_CAPACITY, "C3_HASH_PQ": A.PQ_CAPACITY,
                 "C4_SAMEID_PQ": A.C1_CAPACITY}, "arm_bytes": A.arm_bytes(), "budget_B1": C.B1,
        "codec": {"M": K.M, "K": K.K, "codebook_dtype": "float16", "C0": "fit at end of task 0 on hash-first <=100k "
                  "bottle writer rows, seed 20261106, 25 Lloyd iters; frozen"},
        "thresholds": {"delta": E.DELTA, "eps": E.EPS, "tol": E.TOL, "degenerate": E.DEGENERATE,
                       "note": "operational normal-support margins, not anomaly-AP thresholds"},
        "bootstrap": {"B": E.BOOT, "seeds": E.SEEDS, "unit": "EVAL image", "clusters": "old categories"},
        "verdict_order": ["AM02_BLOCKED", "NO_RETENTION_PROBLEM", "CONVENTIONAL_PQ_SUFFICIENT",
                          "RESIDUAL_GAP_CONSOLIDATION_CANDIDATE", "HEADROOM_ESTABLISHED_RESIDUAL_INCONCLUSIVE",
                          "NO_COMPRESSION_HEADROOM", "HEADROOM_NOT_ESTABLISHED"],
        "metrics": "plan v2 §4 + §13 (exps/nl_sgm_01a/am02_eval.py)",
        "stage_order": ["a0", "C1_FPS_FP32", "c4", "C3_HASH_PQ", "C2_FPS_PQ", "analyze"],
        "caps": CAPS, "device": "cpu", "tf32": False,
        "cohort_r15_digest": C.r15_digest(), "seed": C.SEED,
        "preflight": "results/hope_cad/nl_sgm_01a_am02/RESOURCE_PROJECTION.json (fits_caps true, 4.87 h)",
        "tests": "results/hope_cad/nl_sgm_01a_am02/tests/pytest_prelock_r2.txt",
        "revision": {"r": 2, "supersedes": "LOCK.json (r1, sha256 " + C.sha(OUT / "LOCK.json") + ")",
                     "reason": "run r1 stopped at C3_HASH_PQ t=9 (screw): KeyError in rules.CodecArm.persistent_state; "
                               "_prune dropped PQ codes of rows buffered by boundary rules (FPSRule/QuotaHashRule) when a "
                               "task exceeded 4*capacity+10,000 rows. Fix: buffered rows are live. Regression test AM6.",
                     "design_change": "none (arms, codec, cohorts, metrics, thresholds, verdict rules unchanged)",
                     "outcomes_inspected_before_fix": "none",
                     "r1_outputs": "preserved in run_r1_failed/; all stages re-run from scratch under r2"},
    }
    L.write_lock(LOCKP, payload, sources=AM02_SOURCES)
    print("LOCK written", C.sha(LOCKP))


# ------------------------------------------------------------------ A0 / W_c

def do_a0():
    lock, co, n = setup()
    st = Stage("a0")
    F = C.Features()
    evalQ = {c: F.images(co[c]["eval"]).clone() for c in C.ORDER}
    allQ = torch.cat([evalQ[c] for c in C.ORDER])                  # [90, 784, 768]
    for k, c in enumerate(C.ORDER):
        W = F.images(co[c]["writers"]).reshape(-1, 768).clone()
        rs = X.RowSet(W, writer_gids(n, c))
        selQ = F.images(co[c]["sel"]).clone()
        ds, _ = E.probe_nn(selQ, rs)
        de, ie = E.probe_nn(allQ, rs)                              # every EVAL probe vs W_c (moving oracle)
        ok = E.spot_check(allQ[6 * k].contiguous(), rs)
        require(ok, f"EXACTNESS: A0 spot check failed on {c}")
        ratio = X.verify_screen_bound(allQ[6 * k][:128].contiguous(), W[:20000].contiguous())
        require(ratio <= 1.0, f"EXACTNESS: screen bound ratio {ratio}")
        torch.save({"category": c, "sel_d": ds, "eval_d_all": de, "eval_id_all": ie, "screen_ratio": ratio},
                   st.dir / f"W_{k:02d}_{c}.pt")
        del W, rs
        F.drop(c)
        st.log("A0", c, st.check(c), "screen_ratio", round(ratio, 4))
    st.finish({"eval_categories_order": list(C.ORDER)})


# ------------------------------------------------------------------ arms

def eval_boundary(x, g, o, evalQ, seen):
    cats = E.by_owner(x, g, o)
    glob = E.rowset(x, g)
    owner_of = dict(zip(g.tolist(), o.tolist()))
    res = {}
    for k in seen:
        dc, ic = E.probe_nn(evalQ[k], cats[k])
        dg, ig = E.probe_nn(evalQ[k], glob)
        og = torch.tensor([owner_of[int(v)] for v in ig.reshape(-1)]).reshape(ig.shape)
        res[k] = {"cat_d": dc, "cat_id": ic, "glob_d": dg, "glob_owner": og}
    return res, glob


def check_subset(prev, cur, k_seen):
    for k in range(k_seen - 1):
        require(cur[k] <= prev[k], f"INTEGRITY: category {k} support is not a subset of its previous support")


def do_arm(name):
    lock, co, n = setup()
    st = Stage(name)
    F = C.Features()
    arm = A.build(name)
    prev, evalQ = {}, {}
    for t, c in enumerate(C.ORDER):
        arm.begin_task(t)
        base = gid_base(n, c)
        for j, p in enumerate(co[c]["writers"]):
            arm.image(F.image(p).clone(), torch.arange(base + j * C.PATCHES, base + (j + 1) * C.PATCHES), t)
        arm.end_task(t)
        F.drop(c)
        x, g, o = arm.support()
        cur = {k: set(g[o == k].tolist()) for k in range(t + 1)}
        check_subset(prev, cur, t + 1)
        require(sum(len(v) for v in cur.values()) <= arm.inner.capacity, "INTEGRITY: over capacity")
        state = arm.persistent_state()
        bytes_ = R.state_bytes(state)
        require(bytes_ <= C.B1 and bytes_ == A.arm_bytes()[name], f"LEDGER: {bytes_} B != formula")
        ck = {"gids": g.clone(), **state}                          # state["owner"]: int8, padded with -1
        torch.save(ck, st.dir / f"ckpt_t{t:02d}.pt")
        evalQ[t] = F.images(co[c]["eval"]).clone()                 # category c probes become scorable at a(c)=t
        F.drop(c)
        res, glob = eval_boundary(x, g, o, evalQ, range(t + 1))
        if t in SPOT_T:
            require(E.spot_check(evalQ[t][0].contiguous(), glob), f"EXACTNESS: spot check failed at t={t}")
        torch.save(res, st.dir / f"eval_t{t:02d}.pt")
        prev = cur
        st.log(name, t, c, "rows", len(g), "bytes", bytes_, st.check(f"t{t}"))
    st.finish({"arm": name, "bytes": bytes_, "transient_peak_bytes": int(max(arm.transient_peak,
                                                                             getattr(arm.inner, "transient_peak", 0)))})


# ------------------------------------------------------------------ C4 (same identity, offline) + codec diagnostics

def fit_c0(F, co, n, seed=C.SEED):
    c = C.ORDER[0]
    x = F.images(co[c]["writers"]).reshape(-1, 768).clone()
    g = writer_gids(n, c)
    sample = c0_sample(x, g, seed)
    return K.fit(sample, seed=seed), sample


def c0_sample(x, g, seed=C.SEED):
    """The exact C0 fit sample of rules.CodecArm: task-0 rows ordered by (hash priority, gid), first FIT_ROWS."""
    p = random_priorities(seed, g.numpy().astype(np.int64))
    order = torch.as_tensor(np.lexsort((g.numpy(), p))[:R.CodecArm.FIT_ROWS])
    return x[order].contiguous()


def do_c4():
    lock, co, n = setup()
    st = Stage("c4")
    src = OUT / "stages" / "C1_FPS_FP32"
    require((src / "STAGE_DONE.json").exists(), "C1 must complete before C4")
    F = C.Features()
    cb, sample = fit_c0(F, co, n)
    F.drop(C.ORDER[0])
    torch.save({"codebook": cb}, st.dir / "C0.pt")
    # bottle fit residual p99 per subspace (domain-shift reference)
    res_fit = (sample - K.decode(K.encode(sample, cb), cb)).reshape(len(sample), K.M, K.SUB).square().sum(-1)
    p99 = torch.quantile(res_fit[::4].double(), 0.99, dim=0)
    del sample, res_fit
    evalQ, diag = {}, {}
    for t, c in enumerate(C.ORDER):
        ck = torch.load(src / f"ckpt_t{t:02d}.pt")
        x1, g, o = ck["x"][:len(ck["gids"])], ck["gids"], ck["owner"][:len(ck["gids"])].long()
        codes = K.encode(x1, cb)
        x4 = K.decode(codes, cb)
        torch.save({"gids": g, "owner": o, "codes": codes}, st.dir / f"ckpt_t{t:02d}.pt")
        sel = o == t                                                    # category t at its acquisition boundary
        r = (x1[sel] - x4[sel]).reshape(-1, K.M, K.SUB).square().sum(-1)        # [rows, M] squared residual
        cnt = torch.stack([torch.bincount(codes[sel][:, m].long(), minlength=K.K) for m in range(K.M)]).double()
        pr = cnt / cnt.sum(1, keepdim=True)
        ent = -(pr * torch.log(pr.clamp_min(1e-300))).sum(1)
        diag[t] = {"recon_norm_median": float(r.sum(1).sqrt().median()), "subspace_mse": r.mean(0).double(),
                   "codeword_entropy": ent, "frac_resid_gt_bottle_p99": float((r.double() > p99[None]).double().mean()),
                   "rows": int(sel.sum())}
        evalQ[t] = F.images(co[c]["eval"]).clone()
        F.drop(c)
        res, glob = eval_boundary(x4, g, o, evalQ, range(t + 1))
        if t in SPOT_T:
            require(E.spot_check(evalQ[t][0].contiguous(), glob), f"EXACTNESS: spot check failed at t={t}")
        torch.save(res, st.dir / f"eval_t{t:02d}.pt")
        st.log("C4", t, c, st.check(f"t{t}"))
    torch.save(diag, st.dir / "codec_diagnostics.pt")
    st.finish({"arm": "C4_SAMEID_PQ", "bytes": A.arm_bytes()["C4_SAMEID_PQ"],
               "C0_sha": hashlib.sha256(cb.numpy().tobytes()).hexdigest()})


if __name__ == "__main__":
    cmd = sys.argv[1]
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    if cmd == "lock":
        do_lock()
    elif cmd == "a0":
        do_a0()
    elif cmd == "arm":
        require(sys.argv[2] in ("C1_FPS_FP32", "C2_FPS_PQ", "C3_HASH_PQ"), "unknown arm")
        do_arm(sys.argv[2])
    elif cmd == "c4":
        do_c4()
    else:
        raise SystemExit(f"unknown command {cmd}")
