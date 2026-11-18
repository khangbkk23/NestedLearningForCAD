"""NAR-ACQ-01: normal-only diagnostic of acquisition vs expert aggregation in frozen NAR checkpoints.

    python -B -m exps.nar_acq_01.run lock
    python -B -m exps.nar_acq_01.run execute --owner-authorized

Reads only: frozen BASE, the sealed nested (nar_full_15) and warm (nar_warm_01) stage-15 archives, and the INC
train/good feature caches. Test features, test images, masks, score and metric files are refused by an audit hook.
No parameter is trained; no optimizer exists. Gradients are computed on frozen weights and discarded.

Per arm, per category c (arrival index i), on the first N_EVAL hash-ordered EVAL normals of c:
  own expert i, arrived minimum over experts <= i (acquisition), minimum over all 15 (final), base (zero increment);
  winner frequencies, cross-expert residual matrix, normal tails, global-flattened vs patchwise cosine, patch-norm
  strata, spatial uniformity, output/input norms, collapse checks.
On the first CREDIT normal of c with expert i loaded: full-weight gradients of three losses (global cosine = the
training loss; mean patch 1-cos; mean top-1% patch 1-cos), their alignment, tangent-space share of the rank-8
adapter and gradient norm relative to the base state.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/hope_cad/nar_acq_01"
FULL = ROOT / "results/hope_cad/nar_full_15"
WARM = ROOT / "results/hope_cad/nar_warm_01"
ARMS = {"nested": FULL / "nested/stage15.pt", "warm": WARM / "stage15.pt"}
MANIFEST = {"nested": (FULL / "TRAIN_COMPLETE.json", "nested/stage15.pt"),
            "warm": (WARM / "TRAIN_COMPLETE.json", "stage15.pt")}
N_EVAL = 8
CAPS = {"gpu_s": 900.0, "cuda_bytes": 5 * 2**30, "rss_bytes": 8.5e9}
SOURCES = ("exps/nar_acq_01/run.py", "exps/nar_acq_01/test_run.py", "exps/nar_pilot_00/core.py",
           "exps/nar_pilot_00/detector.py", "exps/nar_pilot_00/full.py", "exps/nar_pilot_00/run.py",
           "exps/vitb8_q2/common.py", "exps/dinomaly/official.py")
# Decision rule, fixed before execution. A category "holds" a criterion in an arm; a candidate is supported when it
# holds in >= MIN_CATS categories in BOTH arms. Exactly one supported -> that verdict. More than one -> the one that
# holds in more of the acquisition-weak categories {screw, transistor, zipper} (both arms; identified by historical,
# already-exposed AP, used for retrospective focus only); tie or none -> ROOT_CAUSE_UNRESOLVED.
RULE = {
    "MIN_CATS": 8,
    "AGG": "final own-win < 0.5 and median foreign/own normal residual ratio < 1.25",
    "AGG_win": 0.5, "AGG_ratio": 1.25,
    "OBJ": "cos(grad global, grad patch-mean) < 0.5 or cos(grad global, grad top-1%) < 0.3; and low/high input-norm "
           "quintile residual ratio >= 1.5",
    "OBJ_cos_mean": 0.5, "OBJ_cos_tail": 0.3, "OBJ_norm_ratio": 1.5,
    "CAP": "endpoint/base global-gradient norm >= 0.25 and (tangent share / random-subspace share >= 5 [unfinished]"
           " or <= 2 [rank-restricted])",
    "CAP_gnorm": 0.25, "CAP_tan_hi": 5.0, "CAP_tan_lo": 2.0,
    "WEAK": ["screw", "transistor", "zipper"],
}


def require(c, msg):
    if not c:
        raise RuntimeError(msg)


def sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p, obj, mode="x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, mode) as f:
        json.dump(obj, f, indent=1)


# ------------------------------------------------------------------ custody

def allowed(path):
    s = str(Path(os.fsdecode(path)).resolve())
    bad = ("_test.pt", "_test.json", "/test/", "ground_truth", "/scores_", "/metric_", "data/visa")
    return not any(b in s for b in bad)


def _hook(ev, a):
    if ev == "open" and a and isinstance(a[0], (str, bytes, os.PathLike)) and not allowed(a[0]):
        raise PermissionError(f"NAR-ACQ-01 custody: refused {os.fsdecode(a[0])}")


# ------------------------------------------------------------------ math helpers

def patch_res(en, de):
    """[B,28,28] patch 1-cos clamped at 0 (canonical expert grid)."""
    return (1 - F.cosine_similarity(en[0].float(), de[0].float(), dim=1)).clamp_min(0)


def tangent_split(H, A, B):
    """Energy of H inside the tangent space {dB A + B dA} of W = W0 + B A, and the random-subspace expectation."""
    H = H.double()
    Qb, _ = torch.linalg.qr(B.double())
    Qa, _ = torch.linalg.qr(A.double().T)
    HQ = H - Qb @ (Qb.T @ H)
    orth = HQ - (HQ @ Qa) @ Qa.T
    tot = float((H * H).sum())
    out_e = float((orth * orth).sum())
    rb, ra = Qb.shape[1], Qa.shape[1]
    rand = 1 - (1 - rb / H.shape[0]) * (1 - ra / H.shape[1])
    return tot, tot - out_e, rand * tot


def stats_category(model, loras, archive, i, X, K):
    """Forward diagnostics on EVAL normals X [n,784,768] for the category that arrived at index i."""
    with torch.no_grad():
        n = len(X)
        grids, glob, outn = [], [], []
        states = [None] + list(archive)                         # None = base (zero increment)
        for st in states:
            if st is None:
                K.set_state(loras, {k: (v[0], torch.zeros_like(v[1])) for k, v in archive[0].items()})
            else:
                K.set_state(loras, st)
            g, gl, on = [], [], []
            for j in range(0, n, 4):
                x = X[j:j + 4].cuda()
                en, de = model(x)
                g.append(patch_res(en, de).cpu())
                e = en[0].float().flatten(1)
                d = de[0].float().flatten(1)
                gl.append((1 - F.cosine_similarity(e, d, dim=1)).cpu())
                on.append((de[0].float().norm(dim=1) / en[0].float().norm(dim=1).clamp_min(1e-12)).cpu())
            grids.append(torch.cat(g))
            glob.append(torch.cat(gl))
            outn.append(torch.cat(on))
        base, ex = grids[0], torch.stack(grids[1:])               # ex [15,n,28,28]
        own = ex[i]
        acq = ex[:i + 1].amin(0)
        fin = ex.amin(0)
        win_acq = (ex[:i + 1].argmin(0) == i).double().mean()
        win_fin_idx = ex.argmin(0)
        win_fin = (win_fin_idx == i).double().mean()
        winners = torch.bincount(win_fin_idx.flatten(), minlength=15).double() / win_fin_idx.numel()
        cross = ex.mean(dim=(1, 2, 3))                           # mean residual of each expert on c normals
        foreign = [float(cross[e] / cross[i]) for e in range(15) if e != i]
        xn = X.norm(dim=2).reshape(n, 28, 28)                    # input patch norms
        q = torch.quantile(xn.flatten(), torch.tensor([0.2, 0.8]))
        lo, hi = xn <= q[0], xn >= q[1]
        own_map, base_map = own.mean(0), base.mean(0)
        ring = torch.zeros(28, 28, dtype=torch.bool)
        ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
        imax = own.flatten(1).amax(1)
        p99 = torch.quantile(own.flatten(1), 0.99, dim=1)
        corr = float(np.corrcoef(xn.flatten().numpy(), own.flatten().numpy())[0, 1])
        return {
            "n": n,
            "global_loss_own": float(glob[i + 1].mean()), "global_loss_base": float(glob[0].mean()),
            "patch_mean_own": float(own.mean()), "patch_mean_base": float(base.mean()),
            "patch_mean_acq": float(acq.mean()), "patch_mean_fin": float(fin.mean()),
            "own_p99_mean": float(p99.mean()), "own_imax_mean": float(imax.mean()),
            "own_imax_cv": float(imax.std() / imax.mean().clamp_min(1e-12)),
            "acq_imax_mean": float(acq.flatten(1).amax(1).mean()), "fin_imax_mean": float(fin.flatten(1).amax(1).mean()),
            "own_minus_acq_mean": float((own - acq).mean()), "own_minus_fin_mean": float((own - fin).mean()),
            "suppression_acq_rel": float((own - acq).mean() / own.mean()),
            "suppression_fin_rel": float((own - fin).mean() / own.mean()),
            "tail_suppression_fin_rel": float(1 - fin.flatten(1).amax(1).mean() / imax.mean()),
            "win_own_acq": float(win_acq), "win_own_fin": float(win_fin),
            "winner_share_fin": winners.tolist(),
            "cross_residual": cross.tolist(),
            "foreign_over_own_median": float(np.median(foreign)), "foreign_over_own_min": float(np.min(foreign)),
            "norm_low_res": float(own[lo].mean()), "norm_high_res": float(own[hi].mean()),
            "norm_low_high_ratio": float(own[lo].mean() / own[hi].mean().clamp_min(1e-12)),
            "norm_residual_corr": corr,
            "base_low_high_ratio": float(base[lo].mean() / base[hi].mean().clamp_min(1e-12)),
            "improve_low": float(1 - own[lo].mean() / base[lo].mean()),
            "improve_high": float(1 - own[hi].mean() / base[hi].mean()),
            "spatial_cv_own": float(own_map.std() / own_map.mean()),
            "border_interior_ratio": float(own_map[ring].mean() / own_map[~ring].mean()),
            "improve_spatial_cv": float((1 - own_map / base_map).std() / (1 - own_map / base_map).mean().abs()),
            "out_in_norm_median": float(outn[i + 1].median()), "out_in_norm_p05": float(torch.quantile(outn[i + 1].flatten(), .05)),
            "frac_patch_lt_1e-4": float((own < 1e-4).double().mean()), "patch_std_own": float(own.std()),
            "finite": bool(torch.isfinite(ex).all() and torch.isfinite(base).all()),
        }


def grad_category(model, loras, state, x, K, O):
    """Full-weight gradients of three losses on one CREDIT normal x [1,784,768] at ``state``; nothing is updated."""
    K.set_state(loras, state)
    bases = {n: w.base.weight for n, w in loras.items()}
    for w in bases.values():
        w.requires_grad_(True)
    out = {}
    losses = {
        "global": lambda en, de: O.global_cosine(en, de),
        "patch_mean": lambda en, de: (1 - F.cosine_similarity(en[0].detach().float(), de[0].float(), dim=1)).mean(),
        "patch_top1": lambda en, de: torch.topk((1 - F.cosine_similarity(en[0].detach().float(), de[0].float(), dim=1)
                                                 ).flatten(), k=8).values.mean(),
    }
    grads = {}
    try:
        for name, fn in losses.items():
            for w in bases.values():
                w.grad = None
            en, de = model(x)
            val = fn(en, de)
            val.backward()
            grads[name] = {n: w.grad.detach().clone() for n, w in bases.items()}
            out[f"{name}_value"] = float(val.detach())
    finally:
        for w in bases.values():
            w.grad = None
            w.requires_grad_(False)
        for w in loras.values():
            w.A.grad = None
            w.B.grad = None
    g = grads["global"]
    dot = lambda a, b: sum(float((a[n].double() * b[n].double()).sum()) for n in a)
    nrm = lambda a: dot(a, a) ** 0.5
    out["gnorm_global"] = nrm(g)
    out["cos_global_patchmean"] = dot(g, grads["patch_mean"]) / (nrm(g) * nrm(grads["patch_mean"]))
    out["cos_global_top1"] = dot(g, grads["patch_top1"]) / (nrm(g) * nrm(grads["patch_top1"]))
    tot = tan = rnd = 0.0
    tan_pm = tot_pm = 0.0
    for n, w in loras.items():
        a, b, r = tangent_split(g[n], w.A.detach(), w.B.detach())
        tot += a; tan += b; rnd += r
        a2, b2, _ = tangent_split(grads["patch_mean"][n], w.A.detach(), w.B.detach())
        tot_pm += a2; tan_pm += b2
    out["tangent_share_global"] = tan / tot
    out["tangent_share_random"] = rnd / tot
    out["tangent_over_random"] = (tan / tot) / (rnd / tot)
    out["tangent_share_patchmean"] = tan_pm / tot_pm
    return out


def gnorm_base(model, loras, archive, x, K, O):
    zero = {k: (v[0], torch.zeros_like(v[1])) for k, v in archive[0].items()}
    K.set_state(loras, zero)
    bases = [w.base.weight for w in loras.values()]
    for w in bases:
        w.requires_grad_(True)
        w.grad = None
    try:
        en, de = model(x)
        O.global_cosine(en, de).backward()
        v = sum(float((w.grad.double() ** 2).sum()) for w in bases) ** 0.5
    finally:
        for w in bases:
            w.grad = None
            w.requires_grad_(False)
        for w in loras.values():
            w.A.grad = None
            w.B.grad = None
    return v


# ------------------------------------------------------------------ decision

def holds(c, g):
    agg = c["win_own_fin"] < RULE["AGG_win"] and c["foreign_over_own_median"] < RULE["AGG_ratio"]
    obj = ((g["cos_global_patchmean"] < RULE["OBJ_cos_mean"] or g["cos_global_top1"] < RULE["OBJ_cos_tail"])
           and c["norm_low_high_ratio"] >= RULE["OBJ_norm_ratio"])
    cap = g["gnorm_rel_base"] >= RULE["CAP_gnorm"] and (g["tangent_over_random"] >= RULE["CAP_tan_hi"]
                                                        or g["tangent_over_random"] <= RULE["CAP_tan_lo"])
    return {"AGG": bool(agg), "OBJ": bool(obj), "CAP": bool(cap)}


def decide(per):
    """per[arm][category] = {'holds': {...}}. Returns (verdict, counts)."""
    names = {"AGG": "EXPERT_AGGREGATION_BOTTLENECK_CANDIDATE", "OBJ": "DECODER_OBJECTIVE_MISMATCH_CANDIDATE",
             "CAP": "CAPACITY_OR_OPTIMIZATION_BOTTLENECK_CANDIDATE"}
    counts = {k: {a: sum(per[a][c]["holds"][k] for c in per[a]) for a in per} for k in names}
    weak = {k: sum(all(per[a][c]["holds"][k] for a in per) for c in RULE["WEAK"]) for k in names}
    sup = [k for k in names if all(v >= RULE["MIN_CATS"] for v in counts[k].values())]
    if len(sup) == 1:
        return names[sup[0]], counts, weak
    if len(sup) > 1:
        best = sorted(sup, key=lambda k: -weak[k])
        if weak[best[0]] > weak[best[1]]:
            return names[best[0]], counts, weak
    return "ROOT_CAUSE_UNRESOLVED", counts, weak


# ------------------------------------------------------------------ phases

def lock():
    require(not (OUT / "LOCK.json").exists(), "LOCK exists")
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in ARMS.values()}
    inputs["results/hope_cad/nar_full_15/BASE.pt"] = sha(FULL / "BASE.pt")
    for arm, (man, key) in MANIFEST.items():
        require(json.loads(man.read_text())["checkpoints"][key] == sha(ARMS[arm]), f"{arm} checkpoint manifest")
    require(json.loads((FULL / "TRAIN_COMPLETE.json").read_text())["base_sha256"] == inputs["results/hope_cad/nar_full_15/BASE.pt"], "BASE")
    from exps.calm02.data import MVTEC_ORDER
    feats = {}
    for c in MVTEC_ORDER:
        p = FULL / "features" / f"{c}_train.pt"
        feats[str(p.relative_to(ROOT))] = json.loads(p.with_suffix(".json").read_text())["sha256"]
    write_json(OUT / "LOCK.json", {"schema": "nar_acq_01_v1", "sources": {s: sha(ROOT / s) for s in SOURCES},
               "inputs": inputs, "train_feature_manifest_sha256": feats, "rule": RULE, "caps": CAPS,
               "n_eval": N_EVAL, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print("LOCK", sha(OUT / "LOCK.json"))


def execute():
    lk = json.loads((OUT / "LOCK.json").read_text())
    require(all(sha(ROOT / s) == h for s, h in lk["sources"].items()), "sources changed since LOCK")
    require(not (OUT / "STARTED.json").exists(), "already started (one-shot)")
    write_json(OUT / "STARTED.json", {"lock_sha256": sha(OUT / "LOCK.json"), "utc": time.time()})
    sys.addaudithook(_hook)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    from exps.calm02.data import MVTEC_ORDER
    from exps.dinomaly import official as O
    from exps.nar_pilot_00 import core as K
    from exps.nar_pilot_00 import full as FU
    ORDER = tuple(MVTEC_ORDER)
    for p, h in lk["inputs"].items():
        require(sha(ROOT / p) == h, f"input changed {p}")
    t0 = time.monotonic()

    def cap():
        require(time.monotonic() - t0 < CAPS["gpu_s"], "CAP gpu wall")
        require(torch.cuda.max_memory_allocated() <= CAPS["cuda_bytes"], "CAP cuda")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < CAPS["rss_bytes"], "CAP rss")

    base = torch.load(FULL / "BASE.pt", weights_only=True)
    model, loras, _ = FU.model_new(base)
    model.eval()
    require(not any(q.requires_grad for n, q in model.named_parameters() if not n.endswith((".A", ".B"))),
            "base must be frozen")
    archives = {a: torch.load(p, weights_only=True)["archive"] for a, p in ARMS.items()}
    per = {a: {} for a in ARMS}
    feats = {}
    for i, c in enumerate(ORDER):
        p = FULL / "features" / f"{c}_train.pt"
        require(sha(p) == lk["train_feature_manifest_sha256"][str(p.relative_to(ROOT))], f"feature hash {c}")
        data = torch.load(p, weights_only=True, mmap=True)
        roles = K.normal_partition(data["rows"])
        X = data["features"][roles["EVAL"][:N_EVAL]].clone()
        xc = data["features"][roles["CREDIT"][0]:roles["CREDIT"][0] + 1].clone().cuda()
        feats[c] = {"eval_idx": roles["EVAL"][:N_EVAL], "credit_idx": roles["CREDIT"][0]}
        del data
        for a in ARMS:
            arc = archives[a]
            st = stats_category(model, loras, arc, i, X, K)
            g = grad_category(model, loras, arc[i], xc, K, O)
            g["gnorm_base"] = gnorm_base(model, loras, arc, xc, K, O)
            g["gnorm_rel_base"] = g["gnorm_global"] / g["gnorm_base"]
            require(st["finite"] and all(np.isfinite(v) for v in g.values()), f"non-finite {a} {c}")
            per[a][c] = {**st, **{f"grad_{k}": v for k, v in g.items()}}
            per[a][c]["holds"] = holds(st, {**g})
            cap()
        print("ACQ", c, json.dumps({a: {k: round(per[a][c][k], 4) for k in (
            "win_own_fin", "foreign_over_own_median", "norm_low_high_ratio", "grad_cos_global_patchmean",
            "grad_cos_global_top1", "grad_tangent_over_random", "grad_gnorm_rel_base")} for a in ARMS}), flush=True)
    # invariance check: grads computed on frozen weights must not have changed any adapter
    for a in ARMS:
        ref = torch.load(ARMS[a], weights_only=True)["archive"]
        require(all(torch.equal(ref[t][n][k], archives[a][t][n][k]) for t in range(15) for n in ref[t] for k in (0, 1)),
                "archive mutated")
    verdict, counts, weak = decide(per)
    res = {"verdict": verdict, "criterion_counts": counts, "weak_category_both_arms": weak, "per_category": per,
           "identities": feats, "gpu_wall_s": time.monotonic() - t0,
           "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
           "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
           "lock_sha256": sha(OUT / "LOCK.json"), "exposure": "normal train/good features only; no test/mask access"}
    write_json(OUT / "RESULT.json", res)
    print("VERDICT", verdict, json.dumps(counts), json.dumps(weak), round(res["gpu_wall_s"], 1), "s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("lock", "execute"))
    p.add_argument("--owner-authorized", action="store_true")
    a = p.parse_args()
    if a.phase == "execute":
        require(a.owner_authorized, "owner authorization required")
    {"lock": lock, "execute": execute}[a.phase]()


if __name__ == "__main__":
    main()
