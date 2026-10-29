# exps/nlcad/forensic_06a.py
"""NLCAD-06A: can a compact old-normal memory identify the attn.proj update components that damage old-normal
functions?  Retrospective mechanism forensics on the RH-01b R15 seed-2711 identity checkpoints (CPU only; normal
features only; no anomaly label, no development AP, no GPU, nothing trained).

Layers: the eight decoder.{i}.attn.proj Linear layers, y = x W^T + b, W in R^{768x768}. An input basis U (768 x r)
acts on the input side:  Delta W_protected = Delta W - alpha * (Delta W U) U^T.

Update quantities (kept distinct):
  - effective stage change  Delta W_s = W_s - W_{s-1}: exact, from the SHA-verified checkpoints. It already contains
    every StableAdamW step of the stage, gradient clipping and decoupled weight decay (p *= 1 - lr*wd).
  - raw gradient G at W_{s-1} on the stage's first four slow-event batches (actual sampler indices), dropout off.
  - AdamW-preconditioned per-step updates: NOT available (no optimizer moments were saved and none are rebuilt).
    The pre-/post-optimizer projection difference is shown only for the analytic step-1 normaliser of a fresh Adam
    (g / |g|), on the actual raw gradient; it is labelled as such.
  - decoupled weight decay: exact per-stage multiplier prod_i (1 - lr_i * wd) from the recipe schedule.

Memories (all at tau-energy bases of token-level uncentred input second moments, trace-normalised per category):
  ACQ    causal: written once per task at its end, C_t(theta_t) on 24 of task t's training normals (data available
         then); the stage-s memory is the mean over tasks 1..s-1.  Stale by construction.
  REP    causal: refreshed at the start of stage s from the Replay-24 members at theta_{s-1}.
  ORACLE NON-CAUSAL forensic bound: theta_{s-1} activations of 8 non-retained training normals per old task.

Functional probe (open-loop, post hoc): the stage-end decoder with only the eight attn.proj weights edited. It ignores
the co-adaptation that per-step protection would induce in the other parameters; `revert_proj` (U = I, alpha = 1)
measures how much that matters.  Old-normal evaluation inputs: DRF-01 held-out normals of bottle/carpet/hazelnut and
6 non-retained, non-memory, non-oracle training normals of every other old task.  Current acquisition: 16 training
normals of task s not used for any covariance.

  python -m exps.nlcad.forensic_06a plan      # writes the predeclared plan (states, configs, metrics, gate)
  python -m exps.nlcad.forensic_06a run       # CPU, ~1.5 h, writes FORENSIC_06A_R15_2711.json
  python -m exps.nlcad.forensic_06a decide    # applies the predeclared gate, writes FORENSIC_06A_DECISION.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import readout as RO
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nlcad.analyze_05a import inner_batches
from exps.nlcad.forensic_blocks import checkpoint
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

OUT = D.ROOT / "results/hope_cad/nlcad"
PLAN = OUT / "FORENSIC_06A_PLAN.json"
RESULT = OUT / "FORENSIC_06A_R15_2711.json"
DECISION = OUT / "FORENSIC_06A_DECISION.json"
AMEND = OUT / "FORENSIC_06A_PLAN_AMENDMENT_1.json"
SRC = "exps/nlcad/forensic_06a.py"
SEED, SUBSET_SEED = 2711, 6061
STATES = list(range(7, 16))                       # every transition theta_{s-1} -> theta_s of the NLCAD segment
LAYERS = [f"decoder.{i}.attn.proj" for i in range(8)]
TAUS = (0.9, 0.97, 0.99)
TAU = 0.97                                        # functional probe
N_MEM, N_ORACLE, N_EVAL, N_CUR = 24, 8, 6, 16
N_GRAD_BATCHES, RAND_DRAWS = 4, 3
CONFIGS = ["prev", "full", "revert_proj", "ACQ_a0.5", "ACQ_a1", "REP_a1", "ORACLE_a1",
           "RAND_a1_0", "RAND_a1_1", "RAND_a1_2", "NORM_a1"]
GATE = {
    "informative_stage": "D_full > 0 with paired image-bootstrap 95% CI excluding 0 (old_eval global-cosine loss), "
                         "AND revert_proj protection >= 0 (else the post-hoc edit is dominated by co-adaptation)",
    "relevance": "median over stages x layers of ACQ(tau .97) update-energy share ||dW U||^2/||dW||^2 "
                 ">= 1.5 x its rank-matched random expectation r/768",
    "protective_stage": "ACQ_a1: protection >= 0.25 AND acquisition retained >= 0.75 AND protection - mean RAND "
                        ">= 0.15 AND protection - NORM >= 0.10 AND acquisition retained >= NORM's - 0.05",
    "FEASIBLE": "relevance AND >= 6 informative stages AND protective in >= 2/3 of informative stages",
    "MECHANISM_REJECTED": "relevance fails at <= 1.2x random, OR in >= 2/3 of informative stages "
                          "ACQ_a1 protection - max(mean RAND, NORM) < 0.05, OR in >= 2/3 of informative stages "
                          "with ACQ_a1 protection >= 0.25 the acquisition retained < 0.5",
    "INCONCLUSIVE": "otherwise, including fewer than 6 informative stages",
    "definitions": {"D_cfg": "mean over old_eval images of loss_cfg - loss_prev",
                    "protection": "1 - D_cfg / D_full",
                    "acquisition_retained": "(L_prev - L_cfg) / (L_prev - L_full) on the current-task set"},
}


def subsets(n):
    """Fixed, label-free image subsets per category (seeded permutation; Replay members ever held are excluded from
    the oracle and evaluation sets)."""
    ever = {c: set() for c in C.STREAM}
    for s in range(2, len(C.STREAM) + 1):
        for c, idx in C.buffer_index(SEED, n, list(C.STREAM[:s - 1])).items():
            ever[c] |= set(idx)
    out = {}
    for k, c in enumerate(C.STREAM):
        perm = torch.randperm(n[c], generator=torch.Generator().manual_seed(SUBSET_SEED + k)).tolist()
        mem = perm[:N_MEM]
        rest = [i for i in perm if i not in ever[c] and i not in set(mem)]
        cur = [i for i in perm[N_MEM:] if i not in set(rest[:N_ORACLE + N_EVAL])][:N_CUR]
        out[c] = {"mem": mem, "oracle": rest[:N_ORACLE], "eval": rest[N_ORACLE:N_ORACLE + N_EVAL], "cur": cur}
        require(len(out[c]["eval"]) == N_EVAL and len(out[c]["cur"]) == N_CUR, f"subset too small: {c}")
    return out


def rows(c, idx):
    f = C.load_features(c)
    return torch.stack([f[i] for i in idx])


@torch.no_grad()
def covs(dec, x, micro=4):
    """Token-level uncentred second moment of each attn.proj input."""
    acc, cnt, hooks = {}, {}, []
    mods = dict(dec.named_modules())

    def hook(name):
        def fn(_, inp):
            a = inp[0].reshape(-1, inp[0].shape[-1]).double()
            acc[name] = acc.get(name, 0) + a.T @ a
            cnt[name] = cnt.get(name, 0) + len(a)
        return fn

    for l in LAYERS:
        hooks.append(mods[l].register_forward_pre_hook(hook(l)))
    for xb in x.split(micro):
        dec(xb[:, 0].float(), xb[:, 1].float())
    for h in hooks:
        h.remove()
    return {l: acc[l] / cnt[l] for l in LAYERS}


def tnorm(cv):
    return {l: v / torch.trace(v) for l, v in cv.items()}


def mean_cov(lst):
    return {l: sum(c[l] for c in lst) / len(lst) for l in LAYERS}


def eig(cv):
    w, v = torch.linalg.eigh(cv)
    return w.flip(0).clamp(min=0), v.flip(1)


def basis(cv, tau):
    w, v = eig(cv) if torch.is_tensor(cv) else cv
    r = int(torch.searchsorted(torch.cumsum(w, 0) / w.sum(), torch.tensor(tau, dtype=w.dtype)).item()) + 1
    return v[:, :r], w[:r]


def frac_in(cv, U):
    return float(torch.trace(U.T @ cv @ U) / torch.trace(cv))


def rand_basis(r, gen):
    q, _ = torch.linalg.qr(torch.randn(768, r, generator=gen, dtype=torch.float64))
    return q


def wd_multiplier():
    """Exact decoupled-weight-decay multiplier of one stage: prod_i (1 - lr_i * wd), identity controller (a = 1)."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=O.RECIPE["lr"])
    sch = O.make_scheduler(opt, DR.ITERS_PER_CLASS)
    lrs = []
    for _ in range(DR.ITERS_PER_CLASS):
        lrs.append(sch.get_last_lr()[0])
        opt.step()
        sch.step()
    lrs = np.array(lrs)
    return float(np.prod(1 - lrs * O.RECIPE["weight_decay"])), float(lrs.sum())


def grad_at(dec, sd, stage, n, sub_cur_feats, replay):
    """Raw gradient of the normal objective (p = hm_p_final, dropout off) at theta_{s-1} on the first slow-event
    batches of the stage (actual sampler indices)."""
    dec.load_state_dict(sd)
    dec.eval()
    mods = dict(dec.named_modules())
    G = {l: torch.zeros(768, 768, dtype=torch.float64) for l in LAYERS}
    for k, (_, ci, ri) in enumerate(inner_batches(stage, n)):
        if k == N_GRAD_BATCHES:
            break
        x = torch.cat([sub_cur_feats[ci], replay[ri]]).float()
        dec.zero_grad(set_to_none=True)
        for xb in x.split(4):
            en, de = dec(xb[:, 0], xb[:, 1])
            loss = O.global_cosine_hm_percent(en, de, p=O.RECIPE["hm_p_final"],
                                              factor=O.RECIPE["hm_factor"]) * len(xb) / len(x)
            loss.backward()
        for l in LAYERS:
            G[l] += mods[l].weight.grad.double()
    dec.zero_grad(set_to_none=True)
    return {l: g / N_GRAD_BATCHES for l, g in G.items()}


@torch.no_grad()
def evaluate(dec, sd, x, ref=None):
    """Per image: global-cosine loss, Dinomaly image score (R_D) and output drift vs `ref` (pooled 14x14 maps)."""
    dec.load_state_dict(sd)
    dec.eval()
    loss, score, pooled = [], [], []
    for xb in x.split(8):
        en, de = dec(xb[:, 0].float(), xb[:, 1].float())
        loss += torch.stack([1 - torch.nn.functional.cosine_similarity(e.flatten(1), d.flatten(1))
                             for e, d in zip(en, de)], 1).mean(1).tolist()
        score += RO.dinomaly(en, de)[1].tolist()
        pooled.append(torch.stack([torch.nn.functional.adaptive_avg_pool2d(d, 14).flatten(1) for d in de], 1))
    pooled = torch.cat(pooled)
    out = {"loss": loss, "score": score}
    if ref is not None:
        out["drift"] = (1 - torch.nn.functional.cosine_similarity(pooled, ref, dim=2)).mean(1).tolist()
    return out, pooled


def edited(full, prev, P):
    """State dict of theta_s with attn.proj weights W_{s-1} + dW - (dW @ P) for per-layer input-side matrices P."""
    sd = dict(full)
    for l, M in P.items():
        w0, w1 = prev[l + ".weight"].double(), full[l + ".weight"].double()
        d = w1 - w0
        sd[l + ".weight"] = (w0 + d - d @ M).to(full[l + ".weight"].dtype)
    return sd


def write_partial(path, value):
    """Overwritable progress checkpoint (D.write_json is create-exclusive); the final result still uses D.write_json."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True))
    os.replace(tmp, path)


def source_ok(pl, src, amend):
    ok = {pl["source_sha256"]}
    if amend.exists():
        ok.add(D.read(amend)["amended_source_sha256"])
    return sha(D.ROOT / src) in ok


def amend(plan_path, src, amend_path, extra):
    require(not amend_path.exists(), f"amendment exists: {amend_path}")
    D.write_json(amend_path, {"plan": str(plan_path.name), "plan_sha256": sha(plan_path),
                              "locked_source_sha256": D.read(plan_path)["source_sha256"],
                              "amended_source_sha256": sha(D.ROOT / src), **extra,
                              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print("amendment written", sha(amend_path))


def plan():
    require(not PLAN.exists(), f"plan already written: {PLAN}")
    D.write_json(PLAN, {
        "task": "NLCAD-06A slow old-normal memory mechanism forensics (retrospective; CPU; no training)",
        "source": SRC, "source_sha256": sha(D.ROOT / SRC), "seed": SEED, "subset_seed": SUBSET_SEED,
        "states": STATES, "layers": LAYERS, "taus": TAUS, "functional_tau": TAU, "configs": CONFIGS,
        "sizes": {"mem_per_task": N_MEM, "oracle_per_old_task": N_ORACLE, "eval_per_old_nonEVAL_task": N_EVAL,
                  "eval_EVAL_heldout": "all DRF-01 held-out normals (20 per category)", "current": N_CUR,
                  "grad_batches": N_GRAD_BATCHES, "random_draws": RAND_DRAWS},
        "labels_used": "none: normal images only; no anomaly mask, no Stage D output, no development AP",
        "selection_note": "states = all nine NLCAD transitions (no selection); alpha in {0.5, 1} are fixed probes",
        "gate": GATE, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print("plan written", sha(PLAN))


def run(threads):
    pl = D.read(PLAN)
    require(source_ok(pl, SRC, AMEND), "source changed after the plan was written (no matching amendment)")
    require(not RESULT.exists(), f"result exists: {RESULT}")
    torch.set_num_threads(threads)
    t_all = time.perf_counter()
    proto = RB.verify_lock()
    n = R1.n_train(proto)
    sub = subsets(n)
    dec = O.Decoder(768, 12).eval()
    hashes = {}

    def ckpt(k):
        sd, hashes[k] = checkpoint(SEED, k)
        return sd
    held = {c: DR.load_split("heldout", c) for c in C.EVAL}
    wd_mult, lr_sum = wd_multiplier()
    timing = {}

    # ACQ memory: written once per task at its end (causal), 24 of the task's own training normals
    t0 = time.perf_counter()
    acq, acq_b, acq_self = {}, {}, {}
    for t in range(1, len(C.STREAM)):
        c = C.STREAM[t - 1]
        dec.load_state_dict(ckpt(t))
        acq[c] = tnorm(covs(dec, rows(c, sub[c]["mem"])))
        acq_b[c] = {l: basis(acq[c][l], TAU)[0] for l in LAYERS}
        # acquisition-time capture of the task's own memory on its evaluation images (staleness reference)
        cv = covs(dec, held[c] if c in C.EVAL else rows(c, sub[c]["eval"]))
        acq_self[c] = {l: frac_in(cv[l], acq_b[c][l]) for l in LAYERS}
        print("ACQ memory written", t, c, flush=True)
    timing["acq_memory_and_self_capture_s"] = time.perf_counter() - t0

    res = {"plan_sha256": sha(PLAN), "checkpoints": {str(s): h for s, h in hashes.items()},
           "weight_decay": {"stage_multiplier": wd_mult, "sum_lr": lr_sum, "wd": O.RECIPE["weight_decay"]},
           "subsets": {c: {k: list(map(int, v)) for k, v in d.items()} for c, d in sub.items()},
           "acq_self_capture_tau97": acq_self, "stages": {}}
    for s in STATES:
        ts = time.perf_counter()
        cur = C.STREAM[s - 1]
        old = list(C.STREAM[:s - 1])
        prev, full = ckpt(s - 1), ckpt(s)
        idx = C.buffer_index(SEED, n, old)
        replay = torch.stack([C.load_features(d)[i] for d in old for i in idx[d]])
        dec.load_state_dict(prev)
        # theta_{s-1} statistics
        eval_x = {c: (held[c] if c in C.EVAL else rows(c, sub[c]["eval"])) for c in old}
        cov_eval = {c: tnorm(covs(dec, x)) for c, x in eval_x.items()}
        cov_old = mean_cov(list(cov_eval.values()))
        cov_new = tnorm(covs(dec, rows(cur, sub[cur]["mem"])))
        cov_rep = mean_cov([tnorm(covs(dec, C.load_features(d)[idx[d]])) for d in old])
        cov_orc = mean_cov([tnorm(covs(dec, rows(c, sub[c]["oracle"]))) for c in old])
        cov_acq = mean_cov([acq[c] for c in old])
        mems = {"ACQ": cov_acq, "REP": cov_rep, "ORACLE": cov_orc}
        eigs = {m: {l: eig(mc[l]) for l in LAYERS} for m, mc in mems.items()}
        G = grad_at(dec, prev, s, n, C.load_features(cur), replay)
        geo = {}
        for tau in TAUS:
            g = {}
            for mname, mc in mems.items():
                per = {}
                for l in LAYERS:
                    U, _ = basis(eigs[mname][l], tau)
                    r = U.shape[1]
                    P = U @ U.T
                    w0, w1 = prev[l + ".weight"].double(), full[l + ".weight"].double()
                    d = w1 - w0
                    Co = cov_old[l]
                    act = float(torch.trace(d @ Co @ d.T))
                    resid = float(torch.trace(d @ (torch.eye(768, dtype=d.dtype) - P) @ Co
                                              @ (torch.eye(768, dtype=d.dtype) - P) @ d.T))
                    gl = G[l]
                    gp = gl - gl @ P                                   # projected BEFORE the optimizer
                    nrm = gp / (gp.abs() + 1e-30)                      # analytic step-1 Adam normaliser (not actual)
                    per[l] = {
                        "rank": r, "random_expectation": r / 768,
                        "old_capture": frac_in(cov_old[l], U), "new_overlap": frac_in(cov_new[l], U),
                        "dW_energy_in_U": float((d @ U).pow(2).sum() / d.pow(2).sum()),
                        "dW_old_action_residual_after_full_projection": resid / act,
                        "grad_energy_in_U": float((gl @ U).pow(2).sum() / gl.pow(2).sum()),
                        "pre_projected_step1_normalised_energy_in_U": float((nrm @ U).pow(2).sum() / nrm.pow(2).sum()),
                        "old_action_gain_vs_new": float((act / torch.trace(Co))
                                                        / (torch.trace(d @ cov_new[l] @ d.T) / torch.trace(cov_new[l]))),
                    }
                    if mname == "ACQ":
                        Uo, _ = basis(eigs["ORACLE"][l], tau)
                        per[l]["subspace_similarity_to_ORACLE"] = float((U.T @ Uo).pow(2).sum() / min(r, Uo.shape[1]))
                g[mname] = per
            geo[str(tau)] = g
        # per-category staleness of each task's own acquisition memory at theta_{s-1}
        stale = {c: {l: frac_in(cov_eval[c][l], acq_b[c][l]) for l in LAYERS} for c in old}
        w = {l: {"dW_rel_norm": float((full[l + ".weight"] - prev[l + ".weight"]).double().norm()
                                      / prev[l + ".weight"].double().norm()),
                 "wd_component_rel_to_dW": float((1 - wd_mult) * prev[l + ".weight"].double().norm()
                                                 / (full[l + ".weight"] - prev[l + ".weight"]).double().norm())}
             for l in LAYERS}
        # functional probe
        Pm = {}
        for mname in ("ACQ", "REP", "ORACLE"):
            Pm[mname] = {l: (lambda U: U @ U.T)(basis(eigs[mname][l], TAU)[0]) for l in LAYERS}
        gen = torch.Generator().manual_seed(SEED * 100 + s)
        rP = [{l: (lambda U: U @ U.T)(rand_basis(int(round(torch.trace(Pm["ACQ"][l]).item())), gen)) for l in LAYERS}
              for _ in range(RAND_DRAWS)]
        beta = {}
        for l in LAYERS:
            d = (full[l + ".weight"] - prev[l + ".weight"]).double()
            beta[l] = float((d - d @ Pm["ACQ"][l]).norm() / d.norm())
        eye = torch.eye(768, dtype=torch.float64)
        P_cfg = {"revert_proj": {l: eye for l in LAYERS},
                 "ACQ_a0.5": {l: 0.5 * Pm["ACQ"][l] for l in LAYERS}, "ACQ_a1": Pm["ACQ"],
                 "REP_a1": Pm["REP"], "ORACLE_a1": Pm["ORACLE"],
                 **{f"RAND_a1_{k}": rP[k] for k in range(RAND_DRAWS)},
                 "NORM_a1": {l: (1 - beta[l]) * eye for l in LAYERS}}
        old_x = torch.cat([eval_x[c] for c in old])
        groups = [c for c in old for _ in range(len(eval_x[c]))]
        cur_x = rows(cur, sub[cur]["cur"])
        fun = {}
        prev_old, ref_old = evaluate(dec, prev, old_x)
        prev_cur, ref_cur = evaluate(dec, prev, cur_x)
        fun["prev"] = {"old": prev_old, "cur": prev_cur}
        for name in CONFIGS[1:]:
            sd = full if name == "full" else edited(full, prev, P_cfg[name])
            fun[name] = {"old": evaluate(dec, sd, old_x, ref_old)[0], "cur": evaluate(dec, sd, cur_x, ref_cur)[0]}
            print(s, name, f"old {np.mean(fun[name]['old']['loss']):.5f} cur {np.mean(fun[name]['cur']['loss']):.5f}",
                  flush=True)
        res["stages"][str(s)] = {"current": cur, "old": old, "old_groups": groups, "geometry": geo,
                                 "staleness_own_acq_capture_tau97": stale, "weights": w, "norm_beta": beta,
                                 "functional": fun, "seconds": time.perf_counter() - ts}
        write_partial(RESULT.with_suffix(".partial.json"), res)
        print("stage", s, "done", round(time.perf_counter() - ts), "s", flush=True)
    res["timing"] = {**timing, "total_s": time.perf_counter() - t_all, "threads": threads}
    D.write_json(RESULT, res)
    RESULT.with_suffix(".partial.json").unlink()
    print("written", RESULT, sha(RESULT))


def boot_ci(x, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(draws)])
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def decide():
    pl, r = D.read(PLAN), D.read(RESULT)
    require(r["plan_sha256"] == sha(PLAN), "plan changed")
    per, rel = {}, []
    for s, st in r["stages"].items():
        f = st["functional"]
        lp, lf = np.array(f["prev"]["old"]["loss"]), np.array(f["full"]["old"]["loss"])
        D_full = float((lf - lp).mean())
        ci = boot_ci(lf - lp)
        A_full = float(np.mean(f["prev"]["cur"]["loss"]) - np.mean(f["full"]["cur"]["loss"]))
        row = {"D_full": D_full, "D_full_ci": ci, "A_full": A_full, "configs": {}}
        for name in CONFIGS[2:]:
            lo = np.array(f[name]["old"]["loss"])
            Dk = float((lo - lp).mean())
            Ak = float(np.mean(f["prev"]["cur"]["loss"]) - np.mean(f[name]["cur"]["loss"]))
            row["configs"][name] = {
                "protection": 1 - Dk / D_full if D_full != 0 else None,
                "acq_retained": Ak / A_full if A_full != 0 else None,
                "paired_old_loss_vs_full": float((lo - lf).mean()), "paired_ci": boot_ci(lo - lf),
                "old_score_vs_full": float(np.mean(f[name]["old"]["score"]) - np.mean(f["full"]["old"]["score"])),
                "old_drift": float(np.mean(f[name]["old"]["drift"])),
                "cur_drift": float(np.mean(f[name]["cur"]["drift"]))}
        row["full_old_drift"] = float(np.mean(f["full"]["old"]["drift"]))
        row["full_cur_drift"] = float(np.mean(f["full"]["cur"]["drift"]))
        c = row["configs"]
        prot = lambda k: c[k]["protection"]
        rand = float(np.mean([prot(f"RAND_a1_{k}") for k in range(RAND_DRAWS)]))
        row["informative"] = bool(D_full > 0 and ci[0] > 0 and prot("revert_proj") >= 0)
        row["rand_mean_protection"] = rand
        a = c["ACQ_a1"]
        row["protective"] = bool(a["protection"] >= 0.25 and a["acq_retained"] >= 0.75 and a["protection"] - rand >= 0.15
                                 and a["protection"] - prot("NORM_a1") >= 0.10
                                 and a["acq_retained"] >= c["NORM_a1"]["acq_retained"] - 0.05)
        row["not_selective"] = bool(a["protection"] - max(rand, prot("NORM_a1")) < 0.05)
        row["prohibitive"] = bool(a["protection"] >= 0.25 and a["acq_retained"] < 0.5)
        per[s] = row
        g = st["geometry"][str(TAU)]["ACQ"]
        rel += [v["dW_energy_in_U"] / v["random_expectation"] for v in g.values()]
    inf = [s for s, v in per.items() if v["informative"]]
    rel_med = float(np.median(rel))
    k = len(inf)
    if rel_med <= 1.2 or (k and sum(per[s]["not_selective"] for s in inf) >= 2 * k / 3) or \
            (k and sum(per[s]["prohibitive"] for s in inf) >= 2 * k / 3):
        label = "SLOW_MEMORY_MECHANISM_REJECTED"
    elif rel_med >= 1.5 and k >= 6 and sum(per[s]["protective"] for s in inf) >= 2 * k / 3:
        label = "SLOW_MEMORY_FEASIBLE"
    else:
        label = "SLOW_MEMORY_INCONCLUSIVE"
    out = {"plan_sha256": sha(PLAN), "result_sha256": sha(RESULT), "relevance_median_ratio": rel_med,
           "informative_stages": inf, "per_stage": per, "label": label, "gate": pl["gate"]}
    D.write_json(DECISION, out)
    print(label, "relevance", round(rel_med, 2), "informative", inf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "amend", "run", "decide"])
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    {"plan": plan, "run": lambda: run(a.threads), "decide": decide,
     "amend": lambda: amend(PLAN, SRC, AMEND, {
         "reason": "engineering only: the per-stage progress checkpoint used the create-exclusive D.write_json and "
                   "could not be overwritten after stage 7 (run aborted with FileExistsError; the first launch had "
                   "been OOM-killed at stage 8 by a concurrent smoke test). Now written via a temp file + os.replace.",
         "unchanged": "all computations, subsets, states, configs, metrics and the decision gate",
         "results_seen_before_amendment": "stage 7 only (reproduced bit-identically by the second launch)"})}[a.cmd]()


if __name__ == "__main__":
    main()
