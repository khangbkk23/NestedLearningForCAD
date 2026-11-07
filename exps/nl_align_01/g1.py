# exps/nl_align_01/g1.py
"""NL-ALIGN-01 G1: adaptive Goodhart test of the synthetic ranking objective on the R15 trajectory.

Stages 7-9 (leather, metal_nut, pill) from the verified RH-01b R15 stage-6 checkpoints, seeds 2711-2713.
No learned controller.

Arms (shared R15 stage-6 start; same data, Replay-24 identities, stage seeds, optimizer, clipping, schedule):
  ID    R15 itself (hash-verified RH-01b stage 7-9 checkpoints). This runner's identity path must reproduce
        R15 stage 7 bitwise (`parity`) before any other arm runs.
  FS_A  fixed synthetic-gradient optimizer; training generator Q3 (CutPaste), Q2 diagnostic only.
  FS_B  the same with training generator Q2 (Perlin+DTD), Q3 diagnostic only.
  CN    compute-matched conventional control. Same extra forward/backward and norm-matched mixing on the same
        sampled hosts, but the auxiliary gradient is the official normal training loss on their CLEAN images.
  FP    fixed-profile update rule: historical NLCAD F1 block multipliers, normal objective only.
        Seed 2711 is the existing F1 run (hash-verified); seeds 2712/2713 are run through
        exps.nlcad.fast.nl_train_stage, the code that produced F1.

Auxiliary step (every fast step, K_SYN = 1), at the same decoder state:
    g = g_N + RHO * ||g_N|| / ||g_aux|| * g_aux
then the unchanged clip-to-0.1 and StableAdamW step. g_aux is computed with dropout disabled inside a forked RNG,
and hosts are drawn from a generator of its own. The fast data/dropout stream is therefore identical across arms.

Host custody: bank hosts of each category are split OPT/DIAG by position parity (current-host list, overriding
the category's buffer-permutation prefix). Gradients only use OPT hosts lawful at the stage (nlcad.bank.allowed).
Every G1 synthetic measurement uses DIAG hosts only, for both generators.

  python -m exps.nl_align_01.g1 prereg | parity | seal | decide | real
  python -m exps.nl_align_01.g1 train --arm FS_A --seed 2711
  python -m exps.nl_align_01.g1 diag  --arm FS_A --seed 2711
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
from exps.nlcad import bank as BK
from exps.nlcad.controller import BLOCKS, BlockController, block_of
from exps.nlcad.fast import MAX_CONSECUTIVE, MAX_SKIPS, _rng_devices, block_optimizer, nl_train_stage
from exps.nlcad.objectives import ranking_loss
from exps.rh01 import numerics as NUM
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

ROOT = D.ROOT
OUT = ROOT / "results/hope_cad/nl_align_01"
G1 = OUT / "g1"
PREREG, SEAL, DECISION, REAL = G1 / "G1_PREREG.json", G1 / "G1_SEAL.json", G1 / "G1_DECISION.json", G1 / "G1_REAL.json"
SEEDS = (2711, 2712, 2713)
START, LAST = 6, 9
K_SYN, RHO, N_CUR, N_REP, TAU, N_NEG, EPS = 1, 0.25, 4, 4, 0.05, 64, 1e-12
GENS = ("Q2", "Q3")
TRAIN_GEN = {"FS_A": "Q3", "FS_B": "Q2", "CN": "Q3"}      # CN: the Q3 pool only defines the (shared) host sequence
DIRECTIONS = {"A": ("FS_A", "Q3", "Q2"), "B": ("FS_B", "Q2", "Q3")}
ARMS = ("ID", "FS_A", "FS_B", "CN", "FP")
F1_RUN = ROOT / "results/hope_cad/nlcad/runs/F1_per_task_seed2711_from6"
REPS, BOOT_SEED = 1000, 20261105
GATE = {"manip_min": 0.01, "transfer_min": 0.005, "harm_max": 0.003}
SOURCES = ["exps/nl_align_01/g1.py", "exps/nl_align_01/build_bank.py", "exps/nlcad/fast.py",
           "exps/nlcad/objectives.py", "exps/nlcad/bank.py", "exps/nlcad/controller.py", "exps/nlcad/run.py",
           "exps/dinomaly/official.py", "exps/dinomaly/readout.py", "exps/dinomaly/run.py", "exps/rh01b/run.py",
           "exps/rh01/run.py", "exps/rh01/common.py", "exps/cadfinal/metrics.py", "exps/cadfinal/common.py"]

RULES = {
    "metric": "synthetic patch AP on the 28x28 grid (R.grid, eval mode, FP32): positives coverage>=0.5, negatives "
              "coverage==0, partial patches ignored; per category over its DIAG-host items, macro over categories "
              "1..stage. Gate at stage 9 (macro over the 9 seen categories).",
    "contrast": "Delta(arm, gen) = seed-mean of [AP(arm, seed, gen) - AP(ID, seed, gen)]; 95% CI from 1,000 paired "
                "bootstraps of DIAG items stratified by category, independent per seed, seed-mean per replicate.",
    "manipulation": "direction d (arm, T, D): Delta(arm, T) >= 0.01 and CI lower > 0",
    "transfer": "manipulation AND Delta(arm, D) >= 0.005 AND CI lower(Delta(arm, D)) > 0 AND CI lower(Delta(arm, D) - "
                "Delta(CN, D)) > 0 AND per-seed Delta(arm, D) > 0 in 3/3 seeds",
    "goodhart": "manipulation AND CI upper(Delta(arm, D)) < 0",
    "harm": "seed-mean of old-normal held-out global-cosine loss (bottle, carpet, hazelnut; stage 9) arm - ID > 0.003",
    "verdict": "SYNTHETIC_GOODHART_CONFIRMED if goodhart in either direction; else "
               "OBJECTIVE_OPTIMIZATION_TRANSFER_SUPPORTED if transfer in both directions and no harm in either; "
               "else OBJECTIVE_ALIGNMENT_INCONCLUSIVE (sub-reasons recorded: no effective optimization, "
               "generator-specific overfitting = manipulation without transfer, one-direction transfer, harm).",
    "g2_launch": "G2 runs iff the G1 verdict is OBJECTIVE_OPTIMIZATION_TRANSFER_SUPPORTED. Real MVTec AP is read only "
                 "after the G1 seal and the decision file, is descriptive, and never enters the G1 or G2 decision.",
}

G2_DESIGN = {
    "launch": "only if G1 = OBJECTIVE_OPTIMIZATION_TRANSFER_SUPPORTED; direction A (train Q3, diagnostic Q2), "
              "fixed now, not chosen by results",
    "fast": "theta: R15 decoder, StableAdamW unchanged; g = g_N + rho_t * ||g_N||/||g_G|| * g_G; R15 is the oracle arm",
    "slow": "psi in R^3, rho_t = 0.5 * sigmoid(psi . h_t), h_t = [1, cos(g_N, g_G), t/iters]; psi = 0 gives "
            "rho = 0.25 (= FS_A)",
    "slow_clock": "H = 32 fast steps; delayed credit J = Q3 ranking loss on fresh lawful OPT items at theta_{j+H}",
    "hypergradient": "first-order truncated: dJ/drho_t ~ <grad J(theta_{j+H}), u_t>, u_t = -lr_t * c_t * "
                     "(||g_N||/||g_G||) * g_G / (sqrt(v_t)+eps) (c_t clip factor, v_t Adam second moment); "
                     "U_k = sum_t (drho_t/dpsi_k) u_t accumulated over the window; psi <- Adam(lr 0.05)",
    "controls": ["ID (R15)", "FS_A (= no-slow-learning ablation: psi frozen at 0)", "FP", "SC same-clock H=1",
                 "RESET (psi reset to 0 at every task boundary)", "CLO scalar hypergradient-descent rho (psi=[bias])"],
    "decision": "NESTED_LEARNING_MECHANISM_SUPPORTED iff NL - max(FS_A, SC, CLO) on the Q2 diagnostic >= 0.005 with "
                "CI lower > 0 AND NL - RESET CI lower > 0 AND no harm; LEARNED_SLOW_LEVEL_REJECTED iff "
                "NL - FS_A CI upper < 0; OBJECTIVE_ONLY_IMPROVEMENT if FS_A beat ID (G1) and NL is not "
                "distinguishable from FS_A; else INCONCLUSIVE",
}


# --- bank and host custody ---------------------------------------------------------------------------


def load_bank(seed, verify=True):
    """{generator: {(category, index, variant): (features [2,789,768] mmap, coverage [28,28])}}, hosts."""
    man = D.read(OUT / f"BANK_seed{seed}.json")
    require(man["seed"] == seed and man["stage_max"] == LAST and man["variants"] == 2, "bank manifest")
    items = {g: {} for g in GENS}
    for g in GENS:
        for c in C.STREAM[:LAST]:
            path = OUT / f"bank_seed{seed}" / f"{g}_{c}.pt"
            if verify:
                require(sha(path) == man["files"][f"{g}_{c}"]["sha256"], f"bank file changed: {path}")
            d = torch.load(path, weights_only=False, mmap=True)
            for j, (cat, i, v) in enumerate(d["keys"]):
                items[g][(cat, int(i), int(v))] = (d["features"][j], d["coverage"][j])
    require(set(items["Q2"]) == set(items["Q3"]), "Q2/Q3 bank keys differ")
    return items, {c: {k: [int(x) for x in v] for k, v in h.items()} for c, h in man["hosts"].items()}


def roles(hosts):
    """{(category, index): 'OPT'|'DIAG'} by position parity; the current-host list overrides the buffer list."""
    r = {}
    for c, h in hosts.items():
        for pos, i in enumerate(h["buffer"]):
            r[(c, i)] = "OPT" if pos % 2 == 0 else "DIAG"
        for pos, i in enumerate(h["current"]):
            r[(c, i)] = "OPT" if pos % 2 == 0 else "DIAG"
    return r


def diag_keys(items, role, cats):
    return sorted(k for k in items["Q2"] if k[0] in cats and role[(k[0], k[1])] == "DIAG")


class Aux:
    """Per-stage auxiliary-gradient source. The host/variant sequence depends only on (seed, stage) and the shared
    OPT pools, so FS_A, FS_B and CN draw the identical hosts at every step."""

    def __init__(self, arm, items, hosts, role, seed, n_train, stage, current, replay, replay_keys):
        c = C.STREAM[stage - 1]
        keys = sorted(k for k in BK.allowed(hosts, seed, n_train, stage) if role.get(k) == "OPT")
        self.pools = {"current": [(a, i, v) for a, i in keys if a == c for v in range(2)],
                      "replay": [(a, i, v) for a, i in keys if a != c for v in range(2)]}
        require(self.pools["current"] and self.pools["replay"], "empty OPT pool")
        self.arm, self.gname = arm, TRAIN_GEN[arm]
        self.syn = {k: (items[self.gname][k][0].clone(), items[self.gname][k][1].clone())
                    for p in self.pools.values() for k in p}
        self.clean = None
        if arm == "CN":
            row = {key: j for j, key in enumerate(replay_keys)}
            self.clean = {(a, i): (current[i] if a == c else replay[row[(a, i)]]).clone()
                          for a, i in {(a, i) for p in self.pools.values() for a, i, _ in p}}
        self.host_gen = torch.Generator().manual_seed(seed * 1000 + stage)
        self.neg_gen = torch.Generator().manual_seed(seed * 1000 + 500 + stage)

    def draw(self):
        out = []
        for src, k in (("current", N_CUR), ("replay", N_REP)):
            pool = self.pools[src]
            out += [pool[j] for j in torch.randperm(len(pool), generator=self.host_gen).tolist()[:k]]
        return out

    def backward(self, dec, keys, p_hm, device="cuda"):
        if self.arm == "CN":
            x = torch.stack([self.clean[(a, i)] for a, i, _ in keys]).to(device)
        else:
            x = torch.stack([self.syn[k][0] for k in keys]).to(device)
            cov = torch.stack([self.syn[k][1] for k in keys]).to(device)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            en, de = dec(x[:, 0], x[:, 1])
        en, de = [e.float() for e in en], [d.float() for d in de]
        if self.arm == "CN":
            loss = O.global_cosine_hm_percent(en, de, p=p_hm, factor=O.RECIPE["hm_factor"])
        else:
            loss = ranking_loss(R.grid(en, de), cov, TAU, N_NEG, gen=self.neg_gen)
        if loss is None:
            return None
        loss.backward()
        return float(loss)


def mix_(params, g_n, rho=RHO):
    """In place: p.grad <- g_N + rho * ||g_N|| / ||g_aux|| * g_aux (p.grad holds g_aux on entry). Returns stats."""
    nn_, na, dot = 0., 0., 0.
    for q, a in zip(params, g_n):
        b = q.grad
        if a is not None:
            nn_ += float(a.double().pow(2).sum())
        if b is not None:
            na += float(b.double().pow(2).sum())
            if a is not None:
                dot += float((a.double() * b.double()).sum())
    nn_, na = math.sqrt(nn_), math.sqrt(na)
    ok = math.isfinite(na) and na > 0 and math.isfinite(nn_)
    scale = rho * nn_ / (na + EPS) if ok else 0.0
    for q, a in zip(params, g_n):
        b = q.grad
        if a is None and b is None:
            continue
        if a is None:
            q.grad = b.mul_(scale)
        elif b is None:
            q.grad = a
        else:
            q.grad = a.add_(b, alpha=scale)
    return {"nN": nn_, "nA": na, "cos": dot / (nn_ * na + EPS) if ok else float("nan"), "scale": scale}


def train_stage(dec, current, iters, gen, replay, aux=None, device="cuda", autocast=True):
    """exps.nlcad.fast.nl_train_stage with ctrl=None and outer=None (the R15-parity path), plus the auxiliary
    gradient mixing before clipping when `aux` is given."""
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
    losses, skips, run_ = [], [], 0
    log = {k: [] for k in ("norm_pre_clip", "nN", "nA", "cos", "scale", "aux_loss")}
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
        if aux is not None and i % K_SYN == 0:
            g_n = [None if q.grad is None else q.grad.detach().clone() for q in params]
            for q in params:
                q.grad = None
            saved = [m.p for m in drops]
            for m in drops:
                m.p = 0.0
            try:
                with torch.random.fork_rng(devices=_rng_devices(device)):
                    al = aux.backward(dec, aux.draw(), p, device)
            finally:
                for m, v in zip(drops, saved):
                    m.p = v
            if al is None:
                for q in params:
                    q.grad = None
            st = mix_(params, g_n)
            del g_n
            for k in ("nN", "nA", "cos", "scale"):
                log[k].append(st[k])
            log["aux_loss"].append(float("nan") if al is None else al)
        norm = torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=O.RECIPE["clip_grad_norm"])
        log["norm_pre_clip"].append(float(norm))
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
    return losses, skips, log


# --- checkpoints -------------------------------------------------------------------------------------


def r15_record(seed, stage):
    rec = D.read(C.OUT / f"runs/R15_seed{seed}/STAGE{stage}.json")
    path = ROOT / rec["checkpoint"]
    require(sha(path) == rec["checkpoint_sha256"], f"R15 checkpoint changed: {path}")
    return path, rec["checkpoint_sha256"]


def run_dir(arm, seed):
    return G1 / "runs" / f"{arm}_seed{seed}"


def checkpoint(arm, seed, stage):
    """(path, sha256) of the decoder state of `arm` after `stage` (stage 6 = the shared R15 start)."""
    if stage == START or arm == "ID":
        return r15_record(seed, stage)
    if arm == "FP" and seed == 2711:
        rec = D.read(F1_RUN / f"STAGE{stage}.json")
    else:
        rec = D.read(run_dir(arm, seed) / f"STAGE{stage}.json")
    path = ROOT / rec["checkpoint"]
    require(sha(path) == rec["checkpoint_sha256"], f"checkpoint changed: {path}")
    return path, rec["checkpoint_sha256"]


def load_state(path):
    return torch.load(path, weights_only=False)["state"]


def f1_controller():
    cfg = D.read(F1_RUN / "CONFIG.json")
    require(cfg["arm"] == "F" and cfg["learn"] is False, "F1 config")
    ctrl = BlockController(lr=cfg["lr_phi"], weight_decay=cfg["wd_phi"])
    with torch.no_grad():
        ctrl.phi.copy_(torch.tensor([cfg["profile"][b] for b in BLOCKS], dtype=torch.float64))
    return ctrl


def displacement(state, ref):
    out = {b: 0.0 for b in BLOCKS}
    for k, v in state.items():
        if k in ref and v.dtype.is_floating_point:
            try:
                b = block_of(k)
            except KeyError:
                continue
            out[b] += float((v.double() - ref[k].double()).pow(2).sum())
    return {b: math.sqrt(v) for b, v in out.items()}


# --- prereg ------------------------------------------------------------------------------------------


def prereg():
    require(not PREREG.exists(), "exists")
    G1.mkdir(parents=True, exist_ok=True)
    banks = {s: sha(OUT / f"BANK_seed{s}.json") for s in SEEDS}
    D.write_json(PREREG, {
        "task": "agents/tasks/task_nl_align_01_user_brief.md", "codex_task": "agents/tasks/task_nl_align_01.md",
        "g0": {"file": "results/hope_cad/nl_align_01/G0.json", "sha256": sha(OUT / "G0.json")},
        "sources": {s: sha(ROOT / s) for s in SOURCES}, "bank_manifests": banks,
        "constants": {"K_SYN": K_SYN, "RHO": RHO, "N_CUR": N_CUR, "N_REP": N_REP, "TAU": TAU, "N_NEG": N_NEG,
                      "seeds": SEEDS, "start": START, "last": LAST, "reps": REPS, "boot_seed": BOOT_SEED, **GATE},
        "arms": ARMS, "directions": DIRECTIONS, "rules": RULES, "g2_design": G2_DESIGN,
        "amendment_vs_g0": "G0 (Codex) named K=32 events with rho=0.25. Before any G1 run this is changed to every "
                           "step (K=1), same rho: 21 events/stage x 0.25 is < 1% of the update mass and cannot test "
                           "'directly optimizing' the objective (no manipulation power). The fixed-profile arm is the "
                           "historical F1 rule on the normal objective (reusing the completed seed-2711 run), "
                           "and the compute-matched control uses the clean images of the same hosts.",
        "custody": "gradients: OPT hosts lawful at the stage only; measurements: DIAG hosts only; no real defect "
                   "image, mask or label is read before G1_SEAL.json and G1_DECISION.json exist",
        "utc": DR.now()})


def verify_prereg():
    p = D.read(PREREG)
    bad = [s for s, d in p["sources"].items() if sha(ROOT / s) != d]
    bad += [f"BANK_seed{s}" for s, d in p["bank_manifests"].items() if sha(OUT / f"BANK_seed{s}.json") != d]
    require(not bad, f"changed after G1 prereg: {bad}")
    return p


# --- training ----------------------------------------------------------------------------------------


def train(arm, seed, stages=None):
    from exps.nlcad.run import cap, start_state
    require(arm in ARMS and seed in SEEDS, "bad arm/seed")
    verify_prereg()
    if arm != "ID":
        require(D.read(G1 / "PARITY.json")["pass"], "identity parity first")
        require(not (arm == "FP" and seed == 2711), "FP seed 2711 is the existing F1 run")
    DR.configure()
    cap()
    proto = RB.verify_lock()
    NUM.use(True)
    run = run_dir(arm, seed)
    run.mkdir(parents=True, exist_ok=True)
    n = R1.n_train(proto)
    held = {c: DR.load_split("heldout", c) for c in C.EVAL}
    sd0, _ = start_state(seed, START)
    dec = DR.new_decoder(seed)
    dec.load_state_dict(sd0)
    items = hosts = role = None
    if arm in TRAIN_GEN:
        items, hosts = load_bank(seed)
        role = roles(hosts)
    ctrl = f1_controller() if arm == "FP" else None
    torch.cuda.reset_peak_memory_stats()
    for s in stages or range(START + 1, LAST + 1):
        c = C.STREAM[s - 1]
        ck, js = run / f"stage{s}_{c}.pt", run / f"STAGE{s}.json"
        if js.exists():
            rec = D.read(js)
            require(sha(ck) == rec["checkpoint_sha256"], f"stage {s} checkpoint changed")
            dec.load_state_dict(load_state(ck))
            continue
        require(not ck.exists(), f"orphan checkpoint: {ck}")
        done = list(C.STREAM[:s - 1])
        idx = C.buffer_index(seed, n, done)
        replay_keys = [(d, i) for d in done for i in idx[d]]
        replay = torch.stack([C.load_features(d)[i] for d, i in replay_keys])
        current = C.load_features(c)
        dseed, gseed = RB.stage_seeds("R15", seed, s)
        torch.manual_seed(dseed)
        gen = torch.Generator().manual_seed(gseed)
        theta0 = {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()}
        aux = Aux(arm, items, hosts, role, seed, n, s, current, replay, replay_keys) if arm in TRAIN_GEN else None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if arm == "FP":
            losses, skips, _ = nl_train_stage(dec, current, R1.ITERS, gen, replay, ctrl=ctrl, outer=None, learn=False)
            log = None
        else:
            losses, skips, log = train_stage(dec, current, R1.ITERS, gen, replay, aux=aux)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        state = {k: v.detach().cpu() for k, v in dec.state_dict().items()}
        torch.save({"state": state, "arm": arm, "seed": seed, "stage": s}, ck)
        if log is not None:
            torch.save(log, run / f"LOG{s}.pt")
        r15 = load_state(r15_record(seed, s)[0])
        summ = None
        if log is not None and log["nN"]:
            a = {k: np.asarray(v, dtype=np.float64) for k, v in log.items()}
            summ = {k: [float(np.nanmean(a[k][j:j + 50])) for j in range(0, len(a[k]), 50)]
                    for k in ("cos", "nN", "nA", "scale", "aux_loss")}
            summ["mean"] = {k: float(np.nanmean(a[k])) for k in ("cos", "nN", "nA", "scale", "aux_loss")}
        rec = {"arm": arm, "seed": seed, "index": s, "name": c, "seconds": seconds, "nonfinite_skips": skips,
               "loss_curve_every50": [float(np.nanmean(losses[j:j + 50])) for j in range(0, len(losses), 50)],
               "clip_saturation": None if log is None else float(np.mean(np.asarray(log["norm_pre_clip"]) >
                                                                          O.RECIPE["clip_grad_norm"])),
               "aux": summ, "heldout_normal": DR.normal_stats(dec, held),
               "displacement_from_stage_start": displacement(state, theta0),
               "distance_to_R15_same_stage": displacement(state, r15),
               "checkpoint": str(ck.relative_to(ROOT)), "checkpoint_sha256": sha(ck),
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "utc": DR.now()}
        D.write_json(js, rec)
        print(arm, seed, s, c, f"{seconds:.0f}s", "skips", len(skips),
              {k: round(v["global_cosine"], 4) for k, v in rec["heldout_normal"].items()},
              None if summ is None else {k: round(v, 4) for k, v in summ["mean"].items()}, flush=True)
        del replay, current, aux, r15


def parity():
    """The identity path of this runner vs the RH-01b R15 stage-7 checkpoint, seed 2711 (bitwise)."""
    verify_prereg()
    train("ID", 2711, stages=[START + 1])
    mine = load_state(ROOT / D.read(run_dir("ID", 2711) / f"STAGE{START + 1}.json")["checkpoint"])
    ref = load_state(r15_record(2711, START + 1)[0])
    diff = max(float((mine[k].double() - ref[k].double()).abs().max()) for k in ref)
    out = {"max_abs_diff": diff, "bitwise": all(torch.equal(mine[k], ref[k]) for k in ref),
           "pass": diff == 0.0, "utc": DR.now()}
    D.write_json(G1 / "PARITY.json", out)
    print(out)
    require(out["pass"], "identity path does not reproduce R15")


# --- synthetic diagnostics (label-free; DIAG hosts) -------------------------------------------------


@torch.no_grad()
def diag(arm, seed):
    verify_prereg()
    DR.configure()
    NUM.use(True)
    items, hosts = load_bank(seed)
    role = roles(hosts)
    keys = diag_keys(items, role, set(C.STREAM[:LAST]))
    out_dir = G1 / "diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = [START] if arm == "ID" else []
    stages += list(range(START + 1, LAST + 1))
    feats = {g: torch.stack([items[g][k][0] for k in keys]) for g in GENS}
    covs = {g: torch.stack([items[g][k][1] for k in keys]) for g in GENS}
    for s in stages:
        target = out_dir / f"{arm}_seed{seed}_stage{s}.pt"
        if target.exists():
            continue
        path, digest = checkpoint(arm, seed, s)
        dec = O.Decoder(768, 12).cuda()
        dec.load_state_dict(load_state(path))
        dec.eval()
        res = {"keys": keys, "checkpoint_sha256": digest}
        for g in GENS:
            sc = []
            for j in range(0, len(keys), 8):
                x = feats[g][j:j + 8].cuda()
                en, de = dec(x[:, 0], x[:, 1])
                sc.append(R.grid(en, de).cpu())
            res[g] = {"scores": torch.cat(sc), "coverage": covs[g]}
        torch.save(res, target)
        print("diag", arm, seed, s, len(keys), flush=True)
        del dec


def weighted_ap(scores, labels, item, W):
    """AP of patch scores with per-item bootstrap counts. scores/labels/item: patch arrays (labels 0/1); W: [reps,
    n_items] (unit weights give the plain AP; ties are not grouped, scores are continuous FP32)."""
    o = np.argsort(-scores, kind="stable")
    y, it = labels[o].astype(np.float64), item[o]
    w = W[:, it]
    tp = np.cumsum(w * y, axis=1)
    tot = np.cumsum(w, axis=1)
    prec = tp / np.maximum(tot, 1e-12)
    return (prec * (w * y)).sum(1) / np.maximum(tp[:, -1], 1e-12)


def syn_table(res, gen, cats, W=None):
    """Per category: AP (unit weights or bootstrap reps), ranking loss and background q99. res: diag file dict."""
    s_all, c_all = res[gen]["scores"].numpy(), res[gen]["coverage"].numpy()
    keys = res["keys"]
    out = {}
    for c in cats:
        rows = [j for j, k in enumerate(keys) if k[0] == c]
        s, cv = s_all[rows].reshape(len(rows), -1), c_all[rows].reshape(len(rows), -1)
        keep = (cv >= 0.5) | (cv == 0)
        item = np.broadcast_to(np.arange(len(rows))[:, None], s.shape)[keep]
        sc, lab = s[keep], (cv[keep] >= 0.5)
        w = np.ones((1, len(rows))) if W is None else W[c]
        ap = weighted_ap(sc, lab, item, w)
        neg = s[cv == 0]
        pos = s[cv >= 0.5]
        out[c] = {"ap": ap if W is not None else float(ap[0]), "neg_q99": float(np.quantile(neg, 0.99)),
                  "pos_mean": float(pos.mean()) if pos.size else float("nan"), "neg_mean": float(neg.mean()),
                  "n_items": len(rows)}
    return out


def boot_weights(keys, cats, seed):
    """{category: [REPS, n_items]} multinomial counts, stratified by category (independent per seed)."""
    rng = np.random.default_rng(BOOT_SEED + seed)
    out = {}
    for c in cats:
        n = sum(1 for k in keys if k[0] == c)
        out[c] = rng.multinomial(n, np.full(n, 1.0 / n), size=REPS).astype(np.float64)
    return out


def seal():
    require(not SEAL.exists(), "sealed")
    verify_prereg()
    files = {}
    for p in sorted((G1 / "runs").rglob("*")) + sorted((G1 / "diag").glob("*.pt")):
        if p.is_file():
            files[str(p.relative_to(ROOT))] = sha(p)
    for seed in SEEDS:
        for arm in ARMS:
            for s in range(START + 1, LAST + 1):
                path, digest = checkpoint(arm, seed, s)
                files[str(path.relative_to(ROOT))] = digest
                require((G1 / "diag" / f"{arm}_seed{seed}_stage{s}.pt").exists(), f"diag missing {arm} {seed} {s}")
        require((G1 / "diag" / f"ID_seed{seed}_stage{START}.pt").exists(), "diag missing stage 6")
    D.write_json(SEAL, {"prereg": sha(PREREG), "files": files, "utc": DR.now()})
    print("sealed", len(files))


def decide():
    require(SEAL.exists() and not DECISION.exists(), "seal first / decided")
    verify_prereg()
    lk = D.read(SEAL)
    require(all(sha(ROOT / n) == d for n, d in lk["files"].items()), "post-seal change")
    stages = list(range(START, LAST + 1))
    table, boots = {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            for s in stages:
                if s == START and arm != "ID":
                    continue
                res = torch.load(G1 / "diag" / f"{arm}_seed{seed}_stage{s}.pt", weights_only=False)
                cats = list(C.STREAM[:s])
                W = boot_weights(res["keys"], cats, seed) if s == LAST else None
                for g in GENS:
                    t = syn_table(res, g, cats)
                    table[(arm, seed, s, g)] = t
                    if W is not None:
                        b = syn_table(res, g, cats, W)
                        boots[(arm, seed, g)] = np.mean([b[c]["ap"] for c in cats], 0)
    def tab(arm, seed, s, g):
        return table[("ID" if s == START else arm, seed, s, g)]

    def macro(arm, seed, s, g, old=False):
        cats = list(C.STREAM[:s - 1] if old else C.STREAM[:s])
        return float(np.mean([tab(arm, seed, s, g)[c]["ap"] for c in cats]))

    def heldout(arm, seed):
        if arm == "ID":
            rec = D.read(C.OUT / f"runs/R15_seed{seed}/STAGE{LAST}.json")
        elif arm == "FP" and seed == 2711:
            rec = D.read(F1_RUN / f"STAGE{LAST}.json")
        else:
            rec = D.read(run_dir(arm, seed) / f"STAGE{LAST}.json")
        return float(np.mean([rec["heldout_normal"][c]["global_cosine"] for c in C.EVAL]))

    def delta(arm, g, ref="ID"):
        per = [macro(arm, sd, LAST, g) - macro(ref, sd, LAST, g) for sd in SEEDS]
        b = np.mean([boots[(arm, sd, g)] - boots[(ref, sd, g)] for sd in SEEDS], 0)
        return {"delta": float(np.mean(per)), "ci95": np.percentile(b, [2.5, 97.5]).tolist(), "per_seed": per,
                "_boot": b}

    res = {"rules": RULES, "seal": sha(SEAL), "directions": {}, "trajectory": {}, "harm": {}}
    for arm in ARMS:
        res["harm"][arm] = {"heldout_old_normal_minus_ID": float(np.mean([heldout(arm, sd) - heldout("ID", sd)
                                                                           for sd in SEEDS])),
                            "per_seed": [heldout(arm, sd) - heldout("ID", sd) for sd in SEEDS]}
        res["trajectory"][arm] = {g: {str(s): {"macro_all": float(np.mean([macro(arm, sd, s, g) for sd in SEEDS])),
                                               "macro_old": float(np.mean([macro(arm, sd, s, g, old=True)
                                                                           for sd in SEEDS])),
                                               "neg_q99": float(np.mean([np.mean([tab(arm, sd, s, g)[c]["neg_q99"]
                                                                                  for c in C.STREAM[:s]])
                                                                         for sd in SEEDS]))}
                                      for s in stages} for g in GENS}
    contrasts = {}
    for arm in ARMS[1:]:
        for g in GENS:
            d = delta(arm, g)
            contrasts[f"{arm}-ID:{g}"] = {k: v for k, v in d.items() if k != "_boot"}
    verdicts = {}
    for name, (arm, T, Dg) in DIRECTIONS.items():
        dT, dD = delta(arm, T), delta(arm, Dg)
        dCN = delta("CN", Dg)
        diff = dD["_boot"] - dCN["_boot"]
        vs_cn = {"delta": dD["delta"] - dCN["delta"], "ci95": np.percentile(diff, [2.5, 97.5]).tolist()}
        manip = dT["delta"] >= GATE["manip_min"] and dT["ci95"][0] > 0
        transfer = (manip and dD["delta"] >= GATE["transfer_min"] and dD["ci95"][0] > 0 and vs_cn["ci95"][0] > 0
                    and all(x > 0 for x in dD["per_seed"]))
        goodhart = manip and dD["ci95"][1] < 0
        harm = res["harm"][arm]["heldout_old_normal_minus_ID"] > GATE["harm_max"]
        reason = ("goodhart" if goodhart else "transfer" if transfer else
                  "no_effective_optimization" if not manip else "generator_specific_overfitting")
        verdicts[name] = {"arm": arm, "train_gen": T, "diag_gen": Dg, "manipulation": manip, "transfer": transfer,
                          "goodhart": goodhart, "harm": harm, "reason": reason,
                          "delta_train": {k: v for k, v in dT.items() if k != "_boot"},
                          "delta_diag": {k: v for k, v in dD.items() if k != "_boot"}, "diag_vs_CN": vs_cn}
    if any(v["goodhart"] for v in verdicts.values()):
        verdict = "SYNTHETIC_GOODHART_CONFIRMED"
    elif all(v["transfer"] and not v["harm"] for v in verdicts.values()):
        verdict = "OBJECTIVE_OPTIMIZATION_TRANSFER_SUPPORTED"
    else:
        verdict = "OBJECTIVE_ALIGNMENT_INCONCLUSIVE"
    res.update({"contrasts": contrasts, "direction_verdicts": verdicts, "verdict": verdict,
                "g2_authorized": verdict == "OBJECTIVE_OPTIMIZATION_TRANSFER_SUPPORTED", "utc": DR.now()})
    D.write_json(DECISION, res)
    print(verdict, {k: (v["reason"], round(v["delta_train"]["delta"], 4), round(v["delta_diag"]["delta"], 4))
                    for k, v in verdicts.items()})


# --- real anomaly evaluation (descriptive; after seal and decision) ---------------------------------


@torch.no_grad()
def real():
    """Native Pixel AP / Image AUROC of every arm's stage-9 decoder on the MVTec test of categories 1..9 (the
    exposed official test: BENCHMARK_DESCRIPTIVE). Paired image bootstrap of arm - ID, stratified by task x label."""
    from exps.cadfinal import common as K
    from exps.cadfinal import metrics as M
    from exps.dinomaly import encoder as E
    require(DECISION.exists() and not REAL.exists(), "decide first / done")
    lk = D.read(SEAL)
    require(all(sha(ROOT / n) == d for n, d in lk["files"].items()), "post-seal change")
    DR.configure()
    NUM.use(True)
    ids = K.benchmark_identities("mvtec")
    tasks = ids["tasks"][:LAST]
    require([t["task"] for t in tasks] == list(C.STREAM[:LAST]), "stream order")
    root = ROOT / "data/mvtec"
    enc = E.load(D.CKPT["base"][0], "base", "cuda")
    ap = {a: np.zeros((len(SEEDS), LAST)) for a in ARMS}
    au = {a: np.zeros((len(SEEDS), LAST)) for a in ARMS}
    boot = {a: np.zeros((len(SEEDS), REPS, LAST)) for a in ARMS}
    for si, seed in enumerate(SEEDS):
        decs = {}
        for a in ARMS:
            dec = O.Decoder(768, 12).cuda()
            dec.load_state_dict(load_state(checkpoint(a, seed, LAST)[0]))
            decs[a] = dec.eval()
        for j, t in enumerate(tasks):
            rows = t["test"]
            labels = np.array([r["label"] for r in rows])
            hws = [M.native_hw(root / r["relative_path"]) for r in rows]
            masks = [M.native_mask(root / r["mask_path"] if r["mask_path"] else None, hw) for r, hw in zip(rows, hws)]
            W = M.bootstrap_weights(labels, REPS, BOOT_SEED + 100 * si + j)
            low = {a: [] for a in ARMS}
            img = {a: [] for a in ARMS}
            for i in range(0, len(rows), 8):
                x = torch.stack([E.preprocess(root / r["relative_path"]) for r in rows[i:i + 8]]).cuda()
                f = enc.groups(x).float()
                for a, dec in decs.items():
                    en, de = dec(f[:, 0], f[:, 1])
                    s, im = R.dinomaly(en, de)
                    low[a].append(s.cpu())
                    img[a].append(im.cpu())
            for a in ARMS:
                lo_ = torch.cat(low[a]).numpy()
                maps = [M.to_native("dinomaly", x, hw) for x, hw in zip(lo_, hws)]
                ap[a][si, j] = M.exact_ap(maps, masks)["ap"]
                au[a][si, j] = M.auroc(labels, torch.cat(img[a]).numpy())
                lo, hi = max(float(lo_.min()), 0.0), float(lo_.max())
                pos = np.zeros((len(rows), 8192))
                tot = np.zeros((len(rows), 8192))
                for k, (m, y) in enumerate(zip(maps, masks)):
                    pos[k], tot[k] = M.hist(m, y, lo, hi, 8192)
                boot[a][si, :, j] = [M.ap_from_hist(pos.T @ w, tot.T @ w) for w in W]
                del maps
            print("real", seed, t["task"], {a: round(ap[a][si, j], 4) for a in ARMS}, flush=True)
        del decs
    names = list(C.STREAM[:LAST])
    out = {"cohort": "MVTec official test, categories 1-9 (exposed; BENCHMARK_DESCRIPTIVE)", "seal": sha(SEAL),
           "decision": sha(DECISION), "arms": {}}
    for a in ARMS:
        b_all = boot[a].mean(0).mean(1)
        b_old = boot[a].mean(0)[:, :LAST - 1].mean(1)
        rec = {"per_category_ap": dict(zip(names, ap[a].mean(0).tolist())),
               "macro_ap_9": float(ap[a].mean()), "macro_ap_old_1_8": float(ap[a][:, :LAST - 1].mean()),
               "macro_image_auroc_9": float(au[a].mean()), "per_seed_macro_ap_9": ap[a].mean(1).tolist()}
        if a != "ID":
            d_all = b_all - boot["ID"].mean(0).mean(1)
            d_old = b_old - boot["ID"].mean(0)[:, :LAST - 1].mean(1)
            rec["delta_vs_ID_macro_9"] = {"delta": float(ap[a].mean() - ap["ID"].mean()),
                                          "ci95_hist": np.percentile(d_all, [2.5, 97.5]).tolist()}
            rec["delta_vs_ID_macro_old"] = {"delta": float(ap[a][:, :LAST - 1].mean() - ap["ID"][:, :LAST - 1].mean()),
                                            "ci95_hist": np.percentile(d_old, [2.5, 97.5]).tolist()}
        out["arms"][a] = rec
    out["utc"] = DR.now()
    D.write_json(REAL, out)
    print({a: (round(r["macro_ap_9"], 4), round(r["macro_ap_old_1_8"], 4)) for a, r in out["arms"].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prereg", "parity", "train", "diag", "seal", "decide", "real"])
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    a = ap.parse_args()
    if a.cmd in ("train", "diag"):
        require(a.arm and a.seed, "--arm and --seed required")
        (train if a.cmd == "train" else diag)(a.arm, a.seed)
    else:
        {"prereg": prereg, "parity": parity, "seal": seal, "decide": decide, "real": real}[a.cmd]()


if __name__ == "__main__":
    main()
