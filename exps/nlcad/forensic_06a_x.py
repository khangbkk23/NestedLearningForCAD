# exps/nlcad/forensic_06a_x.py
"""NLCAD-06A exploratory follow-up (outside the predeclared 06A gate; written after stage 7 of the main run was read).

The main probe edits theta_s, where every other parameter co-adapted to the full attn.proj change. Two CPU checks
separate co-adaptation from what the old-subspace component of the update does on its own:

X1  isolated graft at theta_{s-1}: only the eight attn.proj weights move, by dW P (inside the causal ACQ memory),
    dW (I - P) (outside), the full dW, or dW (I - P_rand) (rank-matched random complement). Everything else stays
    at theta_{s-1}, so no co-adaptation is present. Old-normal damage and current acquisition as in the main probe.
X2  raw-gradient decomposition at theta_{s-1}: the current-task half and the Replay half of the actual first
    slow-event batches, separately. Energy of each in U, and alignment of the update's U-component dW P with the
    replay descent direction -G_rep P (a "replay repair" component) and with -G_cur P.

Same subsets, memories, checkpoints and labels policy as exps.nlcad.forensic_06a (normal images only).

  python -m exps.nlcad.forensic_06a_x plan
  python -m exps.nlcad.forensic_06a_x run
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nlcad import forensic_06a as F
from exps.nlcad.analyze_05a import inner_batches
from exps.nlcad.forensic_blocks import checkpoint
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

PLAN = F.OUT / "FORENSIC_06A_X_PLAN.json"
RESULT = F.OUT / "FORENSIC_06A_X_R15_2711.json"
AMEND = F.OUT / "FORENSIC_06A_X_PLAN_AMENDMENT_1.json"
SRC = "exps/nlcad/forensic_06a_x.py"
GRAFTS = ["graft_full", "graft_U", "graft_perp", "graft_rand_perp"]


def grads_split(dec, sd, stage, n, cur_feats, replay):
    """Raw gradients at theta_{s-1} of the current half and the replay half of each actual batch (dropout off)."""
    dec.load_state_dict(sd)
    dec.eval()
    mods = dict(dec.named_modules())
    out = {k: {l: torch.zeros(768, 768, dtype=torch.float64) for l in F.LAYERS} for k in ("cur", "rep")}
    for k, (_, ci, ri) in enumerate(inner_batches(stage, n)):
        if k == F.N_GRAD_BATCHES:
            break
        for part, x in (("cur", cur_feats[ci]), ("rep", replay[ri])):
            dec.zero_grad(set_to_none=True)
            x = x.float()
            for xb in x.split(4):
                en, de = dec(xb[:, 0], xb[:, 1])
                loss = O.global_cosine_hm_percent(en, de, p=O.RECIPE["hm_p_final"],
                                                  factor=O.RECIPE["hm_factor"]) * len(xb) / len(x)
                loss.backward()
            for l in F.LAYERS:
                out[part][l] += mods[l].weight.grad.double() / F.N_GRAD_BATCHES
    dec.zero_grad(set_to_none=True)
    return out


def graft(prev, full, M):
    """theta_{s-1} with attn.proj weights W_{s-1} + dW @ M (M = P, I - P, I or I - P_rand)."""
    sd = dict(prev)
    for l, m in M.items():
        w0, w1 = prev[l + ".weight"].double(), full[l + ".weight"].double()
        sd[l + ".weight"] = (w0 + (w1 - w0) @ m).to(prev[l + ".weight"].dtype)
    return sd


def plan():
    require(not PLAN.exists(), f"plan already written: {PLAN}")
    D.write_json(PLAN, {
        "task": "NLCAD-06A exploratory follow-up X1/X2 (not a gate input)", "source": SRC,
        "source_sha256": sha(D.ROOT / SRC), "states": F.STATES, "tau": F.TAU, "grafts": GRAFTS,
        "status_note": "written after stage 7 of FORENSIC_06A_R15_2711 had been read; descriptive only",
        "reading": {"X1": "per unit of update energy, compare old damage of graft_U vs graft_perp vs graft_rand_perp "
                          "at theta_{s-1}; graft_U < 0 damage means the old-subspace component repairs old normals",
                    "X2": "cos(dW P, -G_rep P) > 0 consistently -> the old-subspace component is replay-driven"},
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print("plan written", sha(PLAN))


def run(threads):
    pl = D.read(PLAN)
    require(F.source_ok(pl, SRC, AMEND), "source changed after the plan was written (no matching amendment)")
    require(not RESULT.exists(), f"result exists: {RESULT}")
    torch.set_num_threads(threads)
    t_all = time.perf_counter()
    proto = RB.verify_lock()
    n = R1.n_train(proto)
    sub = F.subsets(n)
    dec = O.Decoder(768, 12).eval()
    held = {c: DR.load_split("heldout", c) for c in C.EVAL}
    acq = {}
    for t in range(1, len(C.STREAM)):
        c = C.STREAM[t - 1]
        dec.load_state_dict(checkpoint(F.SEED, t)[0])
        acq[c] = F.tnorm(F.covs(dec, F.rows(c, sub[c]["mem"])))
    res = {"plan_sha256": sha(PLAN), "stages": {}}
    eye = torch.eye(768, dtype=torch.float64)
    for s in F.STATES:
        ts = time.perf_counter()
        cur, old = C.STREAM[s - 1], list(C.STREAM[:s - 1])
        prev, full = checkpoint(F.SEED, s - 1)[0], checkpoint(F.SEED, s)[0]
        cov = F.mean_cov([acq[c] for c in old])
        P = {l: (lambda U: U @ U.T)(F.basis(cov[l], F.TAU)[0]) for l in F.LAYERS}
        gen = torch.Generator().manual_seed(F.SEED * 100 + s)                      # same draws as the main run's first
        Pr = {l: (lambda U: U @ U.T)(F.rand_basis(int(round(torch.trace(P[l]).item())), gen)) for l in F.LAYERS}
        idx = C.buffer_index(F.SEED, n, old)
        replay = torch.stack([C.load_features(d)[i] for d in old for i in idx[d]])
        G = grads_split(dec, prev, s, n, C.load_features(cur), replay)
        x2 = {}
        for l in F.LAYERS:
            d = (full[l + ".weight"] - prev[l + ".weight"]).double()
            dP = d @ P[l]
            cosf = lambda a, b: float((a * b).sum() / (a.norm() * b.norm() + 1e-300))
            x2[l] = {"cur_energy_in_U": float((G["cur"][l] @ P[l]).pow(2).sum() / G["cur"][l].pow(2).sum()),
                     "rep_energy_in_U": float((G["rep"][l] @ P[l]).pow(2).sum() / G["rep"][l].pow(2).sum()),
                     "rep_share_of_U_grad": float((G["rep"][l] @ P[l]).pow(2).sum()
                                                  / ((G["rep"][l] @ P[l]).pow(2).sum() + (G["cur"][l] @ P[l]).pow(2).sum())),
                     "cos_dWP_minus_GrepP": cosf(dP, -G["rep"][l] @ P[l]),
                     "cos_dWP_minus_GcurP": cosf(dP, -G["cur"][l] @ P[l]),
                     "cos_dWperp_minus_Gcurperp": cosf(d - dP, -(G["cur"][l] - G["cur"][l] @ P[l])),
                     "dW_energy_in_U": float(dP.pow(2).sum() / d.pow(2).sum())}
        cfg = {"graft_full": {l: eye for l in F.LAYERS}, "graft_U": P,
               "graft_perp": {l: eye - P[l] for l in F.LAYERS},
               "graft_rand_perp": {l: eye - Pr[l] for l in F.LAYERS}}
        eval_x = {c: (held[c] if c in C.EVAL else F.rows(c, sub[c]["eval"])) for c in old}
        old_x = torch.cat([eval_x[c] for c in old])
        cur_x = F.rows(cur, sub[cur]["cur"])
        fun = {}
        p_old, ref_old = F.evaluate(dec, prev, old_x)
        p_cur, ref_cur = F.evaluate(dec, prev, cur_x)
        fun["prev"] = {"old": p_old, "cur": p_cur}
        for name in GRAFTS:
            sd = graft(prev, full, cfg[name])
            fun[name] = {"old": F.evaluate(dec, sd, old_x, ref_old)[0], "cur": F.evaluate(dec, sd, cur_x, ref_cur)[0]}
        energy = {name: {l: float(((full[l + ".weight"] - prev[l + ".weight"]).double() @ cfg[name][l]).pow(2).sum())
                         for l in F.LAYERS} for name in GRAFTS}
        res["stages"][str(s)] = {"current": cur, "x2": x2, "functional": fun, "update_energy": energy,
                                 "seconds": time.perf_counter() - ts}
        F.write_partial(RESULT.with_suffix(".partial.json"), res)
        print("stage", s, "done", round(time.perf_counter() - ts), "s", flush=True)
    res["total_s"] = time.perf_counter() - t_all
    D.write_json(RESULT, res)
    RESULT.with_suffix(".partial.json").unlink()
    print("written", RESULT, sha(RESULT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "amend", "run"])
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    {"plan": plan, "run": lambda: run(a.threads),
     "amend": lambda: F.amend(PLAN, SRC, AMEND, {
         "reason": "engineering only: progress checkpoint now overwritable (same defect as FORENSIC_06A amendment 1)",
         "forensic_06a_sha256": sha(D.ROOT / F.SRC),
         "unchanged": "all computations, configs and readings", "results_seen_before_amendment": "none"})}[a.cmd]()


if __name__ == "__main__":
    main()
