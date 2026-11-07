# exps/nl_align_01/g2_audit.py
"""NL-ALIGN-01 G2 hypergradient audit: the first-order delayed credit vs central finite differences (label-free).

Seed 2711, stage 7 (leather), from the verified R15 stage-6 checkpoint, slow window [96, 128) with H = 32. Every run
replays the identical fast data/dropout stream (the identity path is bitwise-deterministic, G1 PARITY.json), with
rho = 0.25 (psi = 0) before step 96. Inside the window psi = delta. The slow optimizer is frozen.
J(theta_128) is the Q3 ranking loss on one FIXED set of lawful OPT items with a re-seeded negative sampler.

  estimate_k = the analytic dJ/dpsi_k of the psi = 0 run (exactly what g2.train_stage feeds the slow optimizer)
  fd_k       = (J(psi = +eps e_k) - J(psi = -eps e_k)) / (2 eps),  eps = 0.5
  noise      = |J(psi = 0) - J(psi = 0, repeated)|

Registered rule (G2_AMENDMENT_1): component k is significant iff |J+ - J-| > max(4 noise, 1e-6).
  PASS               >= 1 significant component and sign(estimate_k) = sign(fd_k) for every significant k
  SIGN_FAILURE       a significant component disagrees in sign -> the credit is invalid; G2 training is blocked
  CREDIT_BELOW_NOISE no significant component -> G2 proceeds, flagged (the slow level cannot see its own effect)

  python -m exps.nl_align_01.g2_audit
"""
from __future__ import annotations

import math

import torch

from exps.dinomaly import common as D
from exps.dinomaly import run as DR
from exps.dinomaly.common import require
from exps.nl_align_01 import g1 as G
from exps.nl_align_01 import g2 as G2
from exps.rh01 import numerics as NUM
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

SEED, STAGE, WIN0, WIN1, EPS_FD = 2711, 7, 96, 128, 0.5
OUT = G2.G2 / "HYPERGRAD_AUDIT.json"


class FixedCredit:
    def __init__(self, aux, seed):
        self.aux, self.seed = aux, seed
        self.keys = aux.draw()
        self.arm = aux.arm

    def draw(self):
        return list(self.keys)

    def backward(self, dec, keys, p_hm, device="cuda"):
        self.aux.neg_gen = torch.Generator().manual_seed(self.seed)
        return self.aux.backward(dec, keys, p_hm, device)


class WindowSlow(G2.Slow):
    """psi = delta inside [WIN0, WIN1), 0 before; the slow optimizer never steps (records the hypergradient)."""

    def __init__(self, delta):
        super().__init__(3)
        self.delta = torch.tensor(delta, dtype=torch.float64)

    def rho(self, h):
        i = round(h[2] * R1.ITERS)
        psi = self.delta if i >= WIN0 else torch.zeros(3, dtype=torch.float64)
        hv = torch.tensor(h, dtype=torch.float64)
        s = torch.sigmoid(torch.dot(psi, hv))
        return float(G2.RHO_MAX * s), (G2.RHO_MAX * s * (1 - s) * hv).tolist()

    def step(self, grad):
        return False


def one_run(delta, ctx):
    from exps.nlcad.run import start_state
    sd0, _ = start_state(SEED, G.START)
    dec = DR.new_decoder(SEED)
    dec.load_state_dict(sd0)
    dseed, gseed = RB.stage_seeds("R15", SEED, STAGE)
    torch.manual_seed(dseed)
    gen = torch.Generator().manual_seed(gseed)
    aux = G.Aux("FS_A", *ctx["aux_args"])
    credit = FixedCredit(G.Aux("FS_A", *ctx["aux_args"]), SEED * 1000 + 901)
    credit.aux.host_gen = torch.Generator().manual_seed(SEED * 1000 + 701)
    credit.keys = credit.aux.draw()
    _, _, log, slog = G2.train_stage(dec, ctx["current"], R1.ITERS, gen, ctx["replay"], aux, credit,
                                     WindowSlow(delta), 32, stop_at=WIN1)
    ev = slog[-1]
    require(ev["iter"] == WIN1 - 1, "window misaligned")
    del dec
    return {"J": ev["J"], "hyper": ev["hyper"], "rho_window_mean": float(sum(log["rho"][WIN0:WIN1]) / (WIN1 - WIN0))}


def main():
    from exps.nlcad.run import cap
    require(not OUT.exists(), "exists")
    G2.verify()
    DR.configure()
    cap()
    proto = RB.verify_lock()
    NUM.use(True)
    n = R1.n_train(proto)
    items, hosts = G.load_bank(SEED)
    role = G.roles(hosts)
    done = list(C.STREAM[:STAGE - 1])
    idx = C.buffer_index(SEED, n, done)
    rk = [(d, i) for d in done for i in idx[d]]
    replay = torch.stack([C.load_features(d)[i] for d, i in rk])
    current = C.load_features(C.STREAM[STAGE - 1])
    ctx = {"current": current, "replay": replay,
           "aux_args": (items, hosts, role, SEED, n, STAGE, current, replay, rk)}
    runs = {"base": one_run([0., 0., 0.], ctx), "base_repeat": one_run([0., 0., 0.], ctx)}
    for k in range(3):
        for sgn in (+1, -1):
            d = [0., 0., 0.]
            d[k] = sgn * EPS_FD
            runs[f"{'+' if sgn > 0 else '-'}{k}"] = one_run(d, ctx)
            print(k, sgn, runs[f"{'+' if sgn > 0 else '-'}{k}"], flush=True)
    noise = abs(runs["base"]["J"] - runs["base_repeat"]["J"])
    comp = []
    for k in range(3):
        dj = runs[f"+{k}"]["J"] - runs[f"-{k}"]["J"]
        fd, est = dj / (2 * EPS_FD), runs["base"]["hyper"][k]
        sig = abs(dj) > max(4 * noise, 1e-6)
        comp.append({"k": k, "feature": ["bias", "cos(g_N,g_G)", "progress"][k], "fd": fd, "estimate": est,
                     "significant": sig, "sign_agree": (fd > 0) == (est > 0),
                     "ratio_estimate_over_fd": est / fd if fd else None})
    sigs = [c for c in comp if c["significant"]]
    status = ("SIGN_FAILURE" if any(not c["sign_agree"] for c in sigs) else "PASS" if sigs else "CREDIT_BELOW_NOISE")
    D.write_json(OUT, {"seed": SEED, "stage": STAGE, "window": [WIN0, WIN1], "eps": EPS_FD, "noise": noise,
                       "runs": runs, "components": comp, "status": status, "utc": DR.now()})
    print(status, comp, "noise", noise)
    require(status != "SIGN_FAILURE", "hypergradient sign failure: G2 credit invalid")


if __name__ == "__main__":
    main()
