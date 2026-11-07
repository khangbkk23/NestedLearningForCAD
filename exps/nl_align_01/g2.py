# exps/nl_align_01/g2.py
"""NL-ALIGN-01 G2: a genuine two-level Nested Learning mechanism on the R15 trajectory.

Runs ONLY if G1 = OBJECTIVE_OPTIMIZATION_TRANSFER_SUPPORTED. Written and preregistered before the G1 decision existed.

Fast level  theta (R15 decoder, StableAdamW unchanged, R15 data/replay/seeds), per fast step t:
            g = g_N + rho_t * ||g_N|| / ||g_G|| * g_G,   g_G = Q3 (CutPaste) ranking gradient on lawful OPT hosts
Slow level  psi, with the bounded state-dependent operator
            rho_t = RHO_MAX * sigmoid(psi . h_t),  h_t = [1, cos(g_N, g_G), t / iters]   (psi = 0 -> rho = 0.25 = FS_A)
            on its own clock H with delayed credit:
            psi <- Adam(psi, dJ/dpsi),  J = Q3 ranking loss at theta_{j+H} on FRESH lawful OPT items
            dJ/dpsi_k ~ <grad J(theta_{j+H}), U_k>,  U_k = sum_{t in window} (d rho_t / d psi_k) * u_t
            u_t = d theta_{t+1} / d rho_t ~ -lr_t * c_t * (||g_N||/||g_G||) * g_G / denom_t
            (c_t: clip factor; denom_t: StableAdamW's AMSGrad denominator after the step; first-order truncation)
Q2 (Perlin+DTD) is never used for gradients or slow updates. It is measured only on DIAG hosts afterwards, as in G1.

Arms:
  NL     H = 32, psi in R^3, persists across task boundaries
  SC     same-clock control: H = 1 (credit at theta_{t+1} after every fast step)
  RESET  NL with psi and its Adam state reset at every task boundary (slow-state-reset ablation)
  CLO    conventional learned optimizer: scalar hypergradient-descent on rho (psi = [bias]), H = 32
Controls reused from G1 without rerunning: ID (R15), FS_A (= the no-slow-learning ablation, psi frozen at 0), FP.

  python -m exps.nl_align_01.g2 prereg | seal | decide   (descriptive real-AP scoring is added only if G2 runs)
  python -m exps.nl_align_01.g2 train --arm NL --seed 2711
  python -m exps.nl_align_01.g2 diag  --arm NL --seed 2711
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import readout as R
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nl_align_01 import g1 as G
from exps.nlcad.fast import MAX_CONSECUTIVE, MAX_SKIPS, _rng_devices, block_optimizer
from exps.rh01 import numerics as NUM
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

ROOT = G.ROOT
G2 = G.OUT / "g2"
PREREG, SEAL, DECISION, REAL = G2 / "G2_PREREG.json", G2 / "G2_SEAL.json", G2 / "G2_DECISION.json", G2 / "G2_REAL.json"
ARMS = ("NL", "SC", "RESET", "CLO", "RND")
H = {"NL": 32, "SC": 1, "RESET": 32, "CLO": 32, "RND": 32}
DIM = {"NL": 3, "SC": 3, "RESET": 3, "CLO": 1, "RND": 3}
AMENDMENTS = sorted((G.OUT / "g2").glob("G2_AMENDMENT_*.json"))
RHO_MAX, LR_PSI = 0.5, 0.05
TRAIN_GEN, DIAG_GEN = "Q3", "Q2"
GATE = {"nl_min": 0.005, "harm_max": 0.003}
SOURCES = ["exps/nl_align_01/g2.py", "exps/nl_align_01/g1.py"]
VERDICT_LABEL = {"NESTED_LEARNING_MECHANISM_SUPPORTED": "NL_SPECIFIC_GAIN_SUPPORTED",
                 "OBJECTIVE_ONLY_IMPROVEMENT": "OBJECTIVE_GAIN_ONLY",
                 "LEARNED_SLOW_LEVEL_REJECTED": "NL_MECHANISM_NOT_SUPPORTED", "INCONCLUSIVE": "INCONCLUSIVE"}
RULES = {
    "contrasts": "stage-9 synthetic patch AP on DIAG hosts (G1 metric and bootstrap), seed-mean, paired: "
                 "NL-FS_A, NL-SC, NL-CLO, NL-RESET on Q2 (diagnostic) and Q3 (training generator, held-out hosts)",
    "supported": "NL-FS_A, NL-SC and NL-CLO each >= 0.005 on Q2 with CI lower > 0, AND NL-RESET CI lower > 0 on Q2, "
                 "AND old-normal held-out loss NL-ID <= 0.003",
    "rejected": "NL-FS_A on Q2 CI upper < 0",
    "objective_only": "neither supported nor rejected, and NL-FS_A on Q2 CI contains 0",
    "inconclusive": "otherwise",
    "real": "descriptive only (exposed MVTec): Delta_objective = FS_A - ID, Delta_NL = NL - FS_A",
}


class Slow:
    """psi and its Adam state. rho(h) -> (rho, d rho / d psi)."""

    def __init__(self, dim):
        self.psi = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float64))
        self.opt = torch.optim.Adam([self.psi], lr=LR_PSI)
        self.events = 0

    def rho(self, h):
        h = torch.tensor(h[:len(self.psi)], dtype=torch.float64)
        s = torch.sigmoid(torch.dot(self.psi.detach(), h))
        return float(RHO_MAX * s), (RHO_MAX * s * (1 - s) * h).tolist()

    def step(self, grad):
        g = torch.tensor(grad, dtype=torch.float64)
        if not bool(torch.isfinite(g).all()):
            return False
        self.opt.zero_grad(set_to_none=True)
        self.psi.grad = g
        self.opt.step()
        self.events += 1
        return True

    def snapshot(self):
        return {"psi": self.psi.detach().clone(), "opt": self.opt.state_dict(), "events": self.events}

    def restore(self, snap):
        with torch.no_grad():
            self.psi.copy_(snap["psi"])
        self.opt.load_state_dict(snap["opt"])
        self.events = snap["events"]


def _aux_backward(dec, aux, drops, device, p_hm):
    saved = [m.p for m in drops]
    for m in drops:
        m.p = 0.0
    try:
        with torch.random.fork_rng(devices=_rng_devices(device)):
            return aux.backward(dec, aux.draw(), p_hm, device)
    finally:
        for m, v in zip(drops, saved):
            m.p = v


def train_stage(dec, current, iters, gen, replay, aux, credit, slow, H_, device="cuda", autocast=True, rnd=None,
                stop_at=None):
    """g1.train_stage with rho_t from the slow level and the delayed-credit slow update every H_ fast steps.
    rnd: torch.Generator -> RND arm: the computed hypergradient is replaced by a random direction of the same norm.
    stop_at: stop after this many fast steps (hypergradient audit only)."""
    opt = block_optimizer(dec)
    sch = O.make_scheduler(opt, iters)
    ramp = round(0.1 * iters)
    batch = O.RECIPE["batch_size"]
    cur = DR.Batches(len(current), batch // 2, gen)
    rep = DR.Batches(len(replay), batch // 2, gen)
    dec.train()
    params = list(dec.parameters())
    drops = [m for m in dec.modules() if isinstance(m, torch.nn.Dropout)]
    dtype = torch.device(device).type
    dim = len(slow.psi)
    U = [[torch.zeros_like(q) for q in params] for _ in range(dim)]
    losses, skips, run_ = [], [], 0
    log = {k: [] for k in ("rho", "cos", "nN", "nA", "aux_loss", "norm_pre_clip")}
    slog = []
    for i in range(iters):
        p = min(O.RECIPE["hm_p_final"] * i / ramp, O.RECIPE["hm_p_final"])
        ci, ri = cur.next(), rep.next()
        x = torch.cat([current[ci], replay[ri]])
        x = x[torch.randperm(len(x), generator=gen)].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        total = 0.
        for xb in x.split(DR.MICRO):
            with torch.autocast(dtype, dtype=torch.bfloat16, enabled=autocast):
                en, de = dec(xb[:, 0], xb[:, 1])
            loss = O.global_cosine_hm_percent([e.float() for e in en], [d.float() for d in de],
                                              p=p, factor=O.RECIPE["hm_factor"]) * len(xb) / len(x)
            loss.backward()
            total += float(loss)
        g_n = [None if q.grad is None else q.grad.detach().clone() for q in params]
        for q in params:
            q.grad = None
        al = _aux_backward(dec, aux, drops, device, p)
        g_a = [q.grad for q in params]
        if al is None:
            g_a = [None] * len(params)
        nn_ = math.sqrt(sum(float(a.double().pow(2).sum()) for a in g_n if a is not None))
        na = math.sqrt(sum(float(b.double().pow(2).sum()) for b in g_a if b is not None))
        dot = sum(float((a.double() * b.double()).sum()) for a, b in zip(g_n, g_a) if a is not None and b is not None)
        ok = al is not None and math.isfinite(na) and na > 0 and math.isfinite(nn_)
        cos = dot / (nn_ * na + G.EPS) if ok else 0.0
        rho, drho = slow.rho([1.0, cos, i / iters])
        ratio = nn_ / (na + G.EPS) if ok else 0.0
        for q, a, b in zip(params, g_n, g_a):
            if a is None and b is None:
                q.grad = None
            elif a is None:
                q.grad = b * (rho * ratio)
            elif b is None:
                q.grad = a
            else:
                q.grad = a.add(b, alpha=rho * ratio)
        del g_n
        norm = torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=O.RECIPE["clip_grad_norm"])
        for k, v in (("rho", rho), ("cos", cos), ("nN", nn_), ("nA", na),
                     ("aux_loss", float("nan") if al is None else al), ("norm_pre_clip", float(norm))):
            log[k].append(v)
        if not (np.isfinite(total) and bool(torch.isfinite(norm))):
            opt.zero_grad(set_to_none=True)
            skips.append({"iteration": i, "loss": total, "grad_norm": float(norm)})
            run_ += 1
            require(len(skips) <= MAX_SKIPS and run_ < MAX_CONSECUTIVE, f"non-finite steps exceed limits: {skips}")
            sch.step()
            losses.append(float("nan"))
            continue
        run_ = 0
        base = sch.get_last_lr()[0]
        for g in opt.param_groups:
            g["lr"] = base * 1.0
        opt.step()
        sch.step()
        losses.append(total)
        if ok:                                              # u_t accumulation (first-order, truncated)
            c_t = min(1.0, O.RECIPE["clip_grad_norm"] / (float(norm) + 1e-6))
            coef = -base * c_t * ratio
            bc2 = None
            for j, (q, b) in enumerate(zip(params, g_a)):
                if b is None:
                    continue
                st = opt.state[q]
                bc2 = 1 - O.RECIPE["betas"][1] ** st["step"]
                denom = (st["max_exp_avg_sq"].sqrt() / math.sqrt(bc2)).add_(O.RECIPE["eps"])
                u = b.div(denom).mul_(coef)
                for k in range(dim):
                    U[k][j].add_(u, alpha=drho[k])
                del u, denom
        del g_a
        if i % H_ == H_ - 1:                                # slow event: delayed credit at theta_{j+H}
            for q in params:
                q.grad = None
            jl = _aux_backward(dec, credit, drops, device, p)
            hyper = [0.0] * dim
            if jl is not None:
                for k in range(dim):
                    hyper[k] = sum(float((q.grad.double() * Uk.double()).sum())
                                   for q, Uk in zip(params, U[k]) if q.grad is not None)
            used = hyper
            if rnd is not None:
                v = torch.randn(dim, generator=rnd, dtype=torch.float64)
                used = (v / v.norm() * math.sqrt(sum(h * h for h in hyper))).tolist()
            applied = jl is not None and slow.step(used)
            slog.append({"iter": i, "J": jl, "hyper": hyper, "used": used, "applied": applied,
                         "psi": slow.psi.detach().tolist()})
            for q in params:
                q.grad = None
            for Uk in U:
                for t in Uk:
                    t.zero_()
        if stop_at is not None and i + 1 >= stop_at:
            break
    return losses, skips, log, slog


def run_dir(arm, seed):
    return G2 / "runs" / f"{arm}_seed{seed}"


def checkpoint(arm, seed, stage):
    if arm in G.ARMS:
        return G.checkpoint(arm, seed, stage)
    if stage == G.START:
        return G.r15_record(seed, stage)
    rec = D.read(run_dir(arm, seed) / f"STAGE{stage}.json")
    path = ROOT / rec["checkpoint"]
    require(sha(path) == rec["checkpoint_sha256"], f"checkpoint changed: {path}")
    return path, rec["checkpoint_sha256"]


def prereg():
    require(not PREREG.exists(), "exists")
    require(not G.DECISION.exists(), "G2 must be preregistered before the G1 decision exists")
    G2.mkdir(parents=True, exist_ok=True)
    D.write_json(PREREG, {"g1_prereg": sha(G.PREREG), "sources": {s: sha(ROOT / s) for s in SOURCES},
                          "arms": ARMS, "H": H, "dim": DIM, "rho_max": RHO_MAX, "lr_psi": LR_PSI,
                          "train_gen": TRAIN_GEN, "diag_gen": DIAG_GEN, "gate": GATE, "rules": RULES,
                          "launch_condition": "G1_DECISION.json verdict == OBJECTIVE_OPTIMIZATION_TRANSFER_SUPPORTED",
                          "utc": DR.now()})


def verify():
    p = D.read(PREREG)
    digests = dict(p["sources"])
    for a in AMENDMENTS:
        for name, rec in D.read(a)["files"].items():
            require(digests[name] == rec["locked_sha256"], f"amendment chain broken: {name}")
            digests[name] = rec["amended_sha256"]
        digests.update(D.read(a).get("new_files", {}))
    bad = [s for s, d in digests.items() if sha(ROOT / s) != d]
    require(not bad and p["g1_prereg"] == sha(G.PREREG), f"changed after G2 prereg: {bad}")
    G.verify_prereg()
    require(D.read(G.DECISION)["verdict"] == "OBJECTIVE_OPTIMIZATION_TRANSFER_SUPPORTED", "G1 gate did not pass")


def train(arm, seed):
    from exps.nlcad.run import cap, start_state
    require(arm in ARMS and seed in G.SEEDS, "bad arm/seed")
    verify()
    DR.configure()
    cap()
    proto = RB.verify_lock()
    NUM.use(True)
    run = run_dir(arm, seed)
    run.mkdir(parents=True, exist_ok=True)
    n = R1.n_train(proto)
    held = {c: DR.load_split("heldout", c) for c in C.EVAL}
    sd0, _ = start_state(seed, G.START)
    dec = DR.new_decoder(seed)
    dec.load_state_dict(sd0)
    items, hosts = G.load_bank(seed)
    role = G.roles(hosts)
    slow = Slow(DIM[arm])
    rnd = torch.Generator().manual_seed(seed * 1000 + 77) if arm == "RND" else None
    torch.cuda.reset_peak_memory_stats()
    for s in range(G.START + 1, G.LAST + 1):
        c = C.STREAM[s - 1]
        ck, js = run / f"stage{s}_{c}.pt", run / f"STAGE{s}.json"
        if js.exists():
            rec = D.read(js)
            require(sha(ck) == rec["checkpoint_sha256"], f"stage {s} checkpoint changed")
            snap = torch.load(ck, weights_only=False)
            dec.load_state_dict(snap["state"])
            slow.restore(snap["slow"])
            if rnd is not None:
                rnd.set_state(snap["rnd"])
            continue
        require(not ck.exists(), f"orphan checkpoint: {ck}")
        if arm == "RESET":
            slow = Slow(DIM[arm])
        done = list(C.STREAM[:s - 1])
        idx = C.buffer_index(seed, n, done)
        rk = [(d, i) for d in done for i in idx[d]]
        replay = torch.stack([C.load_features(d)[i] for d, i in rk])
        current = C.load_features(c)
        dseed, gseed = RB.stage_seeds("R15", seed, s)
        torch.manual_seed(dseed)
        gen = torch.Generator().manual_seed(gseed)
        aux = G.Aux("FS_A", items, hosts, role, seed, n, s, current, replay, rk)
        credit = G.Aux("FS_A", items, hosts, role, seed, n, s, current, replay, rk)
        credit.host_gen = torch.Generator().manual_seed(seed * 1000 + 700 + s)
        credit.neg_gen = torch.Generator().manual_seed(seed * 1000 + 900 + s)
        psi0 = slow.psi.detach().tolist()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        losses, skips, log, slog = train_stage(dec, current, R1.ITERS, gen, replay, aux, credit, slow, H[arm], rnd=rnd)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        state = {k: v.detach().cpu() for k, v in dec.state_dict().items()}
        torch.save({"state": state, "slow": slow.snapshot(), "rnd": None if rnd is None else rnd.get_state(),
                    "arm": arm, "seed": seed, "stage": s}, ck)
        torch.save({"log": log, "slow_events": slog}, run / f"LOG{s}.pt")
        rho, cos = np.asarray(log["rho"]), np.asarray(log["cos"])
        rec = {"arm": arm, "seed": seed, "index": s, "name": c, "seconds": seconds, "nonfinite_skips": skips,
               "loss_curve_every50": [float(np.nanmean(losses[j:j + 50])) for j in range(0, len(losses), 50)],
               "psi_start": psi0, "psi_end": slow.psi.detach().tolist(), "slow_events": len(slog),
               "rho": {"mean": float(rho.mean()), "std": float(rho.std()), "min": float(rho.min()),
                       "max": float(rho.max()), "every50": [float(rho[j:j + 50].mean()) for j in range(0, len(rho), 50)],
                       "corr_with_cos": float(np.corrcoef(rho, cos)[0, 1]) if rho.std() > 0 and cos.std() > 0 else None},
               "cos_mean": float(cos.mean()), "aux_loss_mean": float(np.nanmean(log["aux_loss"])),
               "credit_J_every_event": [e["J"] for e in slog][:: max(1, len(slog) // 40)],
               "heldout_normal": DR.normal_stats(dec, held),
               "distance_to_R15_same_stage": G.displacement(state, G.load_state(G.r15_record(seed, s)[0])),
               "checkpoint": str(ck.relative_to(ROOT)), "checkpoint_sha256": sha(ck),
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "utc": DR.now()}
        D.write_json(js, rec)
        print(arm, seed, s, c, f"{seconds:.0f}s", "rho", round(rec["rho"]["mean"], 4), "+-", round(rec["rho"]["std"], 4),
              "psi", [round(v, 3) for v in rec["psi_end"]],
              {k: round(v["global_cosine"], 4) for k, v in rec["heldout_normal"].items()}, flush=True)
        del replay, current, aux, credit


@torch.no_grad()
def diag(arm, seed):
    verify()
    DR.configure()
    NUM.use(True)
    items, hosts = G.load_bank(seed)
    keys = G.diag_keys(items, G.roles(hosts), set(C.STREAM[:G.LAST]))
    out = G2 / "diag"
    out.mkdir(parents=True, exist_ok=True)
    feats = {g: torch.stack([items[g][k][0] for k in keys]) for g in G.GENS}
    covs = {g: torch.stack([items[g][k][1] for k in keys]) for g in G.GENS}
    for s in range(G.START + 1, G.LAST + 1):
        target = out / f"{arm}_seed{seed}_stage{s}.pt"
        if target.exists():
            continue
        path, digest = checkpoint(arm, seed, s)
        dec = O.Decoder(768, 12).cuda()
        dec.load_state_dict(G.load_state(path))
        dec.eval()
        res = {"keys": keys, "checkpoint_sha256": digest}
        for g in G.GENS:
            sc = [R.grid(*dec(feats[g][j:j + 8].cuda()[:, 0], feats[g][j:j + 8].cuda()[:, 1])).cpu()
                  for j in range(0, len(keys), 8)]
            res[g] = {"scores": torch.cat(sc), "coverage": covs[g]}
        torch.save(res, target)
        print("diag", arm, seed, s, flush=True)


def seal():
    require(not SEAL.exists(), "sealed")
    verify()
    files = {}
    for p in sorted((G2 / "runs").rglob("*")) + sorted((G2 / "diag").glob("*.pt")):
        if p.is_file():
            files[str(p.relative_to(ROOT))] = sha(p)
    for seed in G.SEEDS:
        for arm in ARMS:
            require((G2 / "diag" / f"{arm}_seed{seed}_stage{G.LAST}.pt").exists(), f"diag missing {arm} {seed}")
    D.write_json(SEAL, {"prereg": sha(PREREG), "g1_seal": sha(G.SEAL), "files": files, "utc": DR.now()})


def decide():
    require(SEAL.exists() and not DECISION.exists(), "seal first / decided")
    verify()
    lk = D.read(SEAL)
    require(all(sha(ROOT / n) == d for n, d in lk["files"].items()), "post-seal change")
    cats = list(C.STREAM[:G.LAST])
    boots, point = {}, {}
    for seed in G.SEEDS:
        for arm in ARMS + ("ID", "FS_A"):
            src = (G2 if arm in ARMS else G.G1) / "diag" / f"{arm}_seed{seed}_stage{G.LAST}.pt"
            res = torch.load(src, weights_only=False)
            W = G.boot_weights(res["keys"], cats, seed)
            for g in G.GENS:
                boots[(arm, seed, g)] = np.mean([G.syn_table(res, g, cats, W)[c]["ap"] for c in cats], 0)
                point[(arm, seed, g)] = float(np.mean([G.syn_table(res, g, cats)[c]["ap"] for c in cats]))

    def contrast(a, b, g):
        per = [point[(a, sd, g)] - point[(b, sd, g)] for sd in G.SEEDS]
        bb = np.mean([boots[(a, sd, g)] - boots[(b, sd, g)] for sd in G.SEEDS], 0)
        return {"delta": float(np.mean(per)), "ci95": np.percentile(bb, [2.5, 97.5]).tolist(), "per_seed": per}

    def heldout(arm, seed):
        if arm == "ID":
            rec = D.read(C.OUT / f"runs/R15_seed{seed}/STAGE{G.LAST}.json")
        else:
            rec = D.read(run_dir(arm, seed) / f"STAGE{G.LAST}.json")
        return float(np.mean([rec["heldout_normal"][c]["global_cosine"] for c in C.EVAL]))

    con = {f"{a}-{b}:{g}": contrast(a, b, g) for g in G.GENS
           for a, b in (("NL", "FS_A"), ("NL", "SC"), ("NL", "CLO"), ("NL", "RESET"), ("NL", "RND"), ("NL", "ID"),
                        ("FS_A", "ID"))}
    harm = float(np.mean([heldout("NL", sd) - heldout("ID", sd) for sd in G.SEEDS]))
    q = lambda a, b: con[f"{a}-{b}:{DIAG_GEN}"]
    supported = (all(q("NL", b)["delta"] >= GATE["nl_min"] and q("NL", b)["ci95"][0] > 0
                     for b in ("FS_A", "SC", "CLO", "RND"))
                 and q("NL", "RESET")["ci95"][0] > 0 and harm <= GATE["harm_max"])
    rejected = q("NL", "FS_A")["ci95"][1] < 0
    if supported:
        verdict = "NESTED_LEARNING_MECHANISM_SUPPORTED"
    elif rejected:
        verdict = "LEARNED_SLOW_LEVEL_REJECTED"
    elif q("NL", "FS_A")["ci95"][0] <= 0 <= q("NL", "FS_A")["ci95"][1]:
        verdict = "OBJECTIVE_ONLY_IMPROVEMENT"
    else:
        verdict = "INCONCLUSIVE"
    D.write_json(DECISION, {"rules": RULES, "amendments": [a.name for a in AMENDMENTS], "contrasts": con,
                            "harm_NL_minus_ID": harm, "mechanism_verdict": verdict,
                            "verdict": VERDICT_LABEL[verdict], "utc": DR.now()})
    print(verdict, {k: round(v["delta"], 4) for k, v in con.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prereg", "train", "diag", "seal", "decide"])
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int, choices=G.SEEDS)
    a = ap.parse_args()
    if a.cmd in ("train", "diag"):
        require(a.arm and a.seed, "--arm and --seed required")
        (train if a.cmd == "train" else diag)(a.arm, a.seed)
    else:
        {"prereg": prereg, "seal": seal, "decide": decide}[a.cmd]()


if __name__ == "__main__":
    main()
