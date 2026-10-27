"""NLCAD-05A record-only retention-credit harness.

The recorder copies the locked RH-01b fast loop locally so it can retain every
identity and optimizer invariant while recording several outer gradients.  No
recorded credit is applied to the trajectory.  It never reads anomaly labels
or invokes Stage D scoring.

Commands:
  python -m exps.nlcad.retention_credit preflight
  python -m exps.nlcad.retention_credit prereg
  python -m exps.nlcad.retention_credit run [--device cuda]
  python -m exps.nlcad.retention_credit summarize
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from exps.dinomaly import common as D
from exps.dinomaly import official as O
from exps.dinomaly import readout as RO
from exps.dinomaly import run as DR
from exps.dinomaly.common import require, sha
from exps.nlcad import bank as BK
from exps.nlcad.controller import BLOCKS, block_of
from exps.nlcad.fast import block_optimizer
from exps.nlcad.objectives import _forward, ranking_loss
from exps.rh01 import numerics as NUM
from exps.rh01 import run as R1
from exps.rh01b import common as C
from exps.rh01b import run as RB

OUT = D.ROOT / "results/hope_cad/nlcad"
TAG = "RETENTION_CREDIT_RH01b_R15_seed2711_stage7_9"
RUN = OUT / "runs" / TAG
SEED, START, FINAL, K = 2711, 6, 9, 32
TAU, N_NEG = 0.05, 64
Q2_EVENT_ITERS = (319, 639)                 # fixed before execution
CF_EVENTS = {(7, 319), (8, 319), (9, 319)}  # one mid/late phase event per stage
CF_HORIZON, CF_PHI_NORM = 1, 0.15            # fixed before Q2 is inspected
TASK = "agents/tasks/task_nlcad_05a_retention_credit.md"
PREREG = OUT / "PREREG_05A_R3.json"
PREREG_R2 = OUT / "PREREG_05A_R2.json"
PREFLIGHT = OUT / "PREFLIGHT_05A.json"
PHASES = ((0, 99, "warmup"), (100, 299, "mid_high"), (300, 499, "mid_low"), (500, 666, "tail"))


def _dev(device):
    return torch.device(device)


def _devices(device):
    d = _dev(device)
    return [d.index or 0] if d.type == "cuda" else []


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(dec):
    h = hashlib.sha256()
    for n, p in sorted(dec.state_dict().items()):
        h.update(n.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def object_hash(obj):
    h = hashlib.sha256()
    def rec(v):
        if torch.is_tensor(v):
            h.update(b"T"); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str): h.update(str(k).encode()); rec(v[k])
        elif isinstance(v, (list, tuple)):
            for z in v: rec(z)
        else: h.update(repr(v).encode())
    rec(obj)
    return h.hexdigest()


def cpu_tree(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cpu_tree(v) for v in obj)
    return copy.deepcopy(obj)


def block_grads(dec):
    out = {b: [] for b in BLOCKS}
    for n, p in dec.named_parameters():
        out[block_of(n)].append(torch.zeros_like(p, device="cpu") if p.grad is None else p.grad.detach().cpu().clone())
    return out


def block_norms(groups):
    return {b: float(torch.sqrt(sum((x.double() ** 2).sum() for x in groups[b]))) for b in BLOCKS}


def dot_groups(g, d):
    return {b: float(sum((x.double() * y.double()).sum() for x, y in zip(g[b], d[b]))) for b in BLOCKS}


def delta_groups(dec, old):
    out = {b: [] for b in BLOCKS}
    for n, p in dec.named_parameters():
        out[block_of(n)].append(p.detach().cpu() - old[n])
    return out


def phi_hypergrad(phi, s):
    """Full dJ/dphi for a=exp(c*tanh(phi))/mean(exp(c*tanh(phi)))."""
    p = phi.detach().double()
    h = math.log(2.0) / 2 * torch.tanh(p)
    a = torch.exp(h) / torch.exp(h).mean()
    sv = torch.tensor([s[b] for b in BLOCKS], dtype=torch.float64)
    weighted = torch.dot(a, sv) / len(BLOCKS)
    return (math.log(2.0) / 2 * (1 - torch.tanh(p) ** 2) * a * (sv - weighted)).tolist()


def cosine(u, v, centred=False):
    x, y = np.asarray([u[b] for b in BLOCKS], dtype=float), np.asarray([v[b] for b in BLOCKS], dtype=float)
    if centred:
        x, y = x - x.mean(), y - y.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-300))


def sign_agreement(u, v):
    x, y = np.asarray([u[b] for b in BLOCKS]), np.asarray([v[b] for b in BLOCKS])
    return float(np.mean(np.sign(x) == np.sign(y)))


def phase(i):
    return next(name for lo, hi, name in PHASES if lo <= i <= hi)


def fixed_gen(seed):
    return torch.Generator().manual_seed(int(seed))


class CreditRecorder:
    """Deterministic current/replay synthetic and normal query recorder."""
    def __init__(self, bank, hosts, allowed, current_cat, replay_keys, stage, seed, device):
        self.bank, self.hosts, self.allowed = bank, hosts, set(allowed)
        self.current_cat, self.replay_keys = current_cat, list(replay_keys)
        self.stage, self.seed, self.device = stage, seed, _dev(device)
        self.pools = {g: {"current": [x for x in bank[g] if x["host"] in self.allowed and x["host"][0] == current_cat],
                          "replay": [x for x in bank[g] if x["host"] in self.allowed and x["host"][0] != current_cat]}
                      for g in BK.GENERATORS}

    def inner_hosts(self, ctx):
        return {(self.current_cat, int(i)) for i in (ctx.get("ci") or [])} | \
               {self.replay_keys[int(i)] for i in (ctx.get("ri") or [])}

    def sample(self, gen, src, ctx, tag, n=8):
        inner = self.inner_hosts(ctx)
        pool = [x for x in self.pools[gen][src] if x["host"] not in inner]
        g = fixed_gen(self.seed + 1009 * self.stage + 37 * int(ctx["iter"]) + sum(map(ord, tag)))
        order = torch.randperm(len(pool), generator=g).tolist()[:n]
        return [pool[i] for i in order]

    def normal_query(self, current, replay, ctx, tag):
        inner_c = set(ctx.get("ci") or [])
        inner_r = set(ctx.get("ri") or [])
        g = fixed_gen(self.seed + 7001 * self.stage + 53 * int(ctx["iter"]) + sum(map(ord, tag)))
        pc = [i for i in range(len(current)) if i not in inner_c]
        pr = [i for i in range(len(replay)) if i not in inner_r]
        qc = [pc[i] for i in torch.randperm(len(pc), generator=g).tolist()[:8]]
        qr = [pr[i] for i in torch.randperm(len(pr), generator=g).tolist()[:8]]
        return qc, qr

    def _loss(self, dec, items, tag, grad=True):
        if not items:
            return {"loss": float("nan"), "used": 0, "skipped": 0, "hosts": []}, {b: [] for b in BLOCKS}
        total, used, skipped = 0.0, 0, 0
        dec_device = self.device.type
        ng = fixed_gen(self.seed + 9001 * self.stage + 17 * sum(map(ord, tag)))
        for k in range(0, len(items), 8):
            chunk = items[k:k + 8]
            x = torch.stack([it["features"] for it in chunk]).to(self.device)
            cov = torch.stack([it["coverage"] for it in chunk]).to(self.device)
            keep = (cov >= 0.5).flatten(1).any(1)
            skipped += int((~keep).sum())
            with torch.set_grad_enabled(grad):
                en, de = _forward(dec, x, self.device, self.device.type == "cuda")
                loss = ranking_loss(RO.grid(en, de), cov, TAU, N_NEG, gen=ng)
                if loss is not None and grad:
                    (loss * len(chunk) / len(items)).backward()
            if loss is not None:
                total += float(loss.detach()) * len(chunk) / len(items)
                used += int(keep.sum())
        return {"loss": total if used else float("nan"), "used": used, "skipped": skipped,
                "hosts": [list(x["host"]) for x in items]}, block_grads(dec)

    def _normal(self, dec, current, replay, ctx, tag):
        qc, qr = self.normal_query(current, replay, ctx, tag)
        x = torch.cat([current[qc], replay[qr]]).to(self.device)
        total = 0.0
        for xb in x.split(8):
            en, de = _forward(dec, xb, self.device, self.device.type == "cuda")
            loss = O.global_cosine(en, de) * len(xb) / len(x)
            loss.backward(); total += float(loss.detach())
        return {"loss": total, "query_current": qc, "query_replay": qr}, block_grads(dec)

    def event(self, dec, current, replay, ctx, q2):
        was_training = dec.training
        drops = [m for m in dec.modules() if isinstance(m, torch.nn.Dropout)]
        saved = [m.p for m in drops]
        # Keep train mode so the locked decoder's checkpoint=True path remains active.
        # Dropout is disabled explicitly, matching exps.nlcad.fast.slow_event.
        for m in drops: m.p = 0.0
        out, grads = {}, {}
        try:
            for key, gen, src, enabled in (("Q3_current", "Q3", "current", True),
                                           ("Q3_replay", "Q3", "replay", True),
                                           ("Q2_replay", "Q2", "replay", q2)):
                if not enabled:
                    out[key], grads[key] = None, None
                    continue
                dec.zero_grad(set_to_none=True)
                out[key], grads[key] = self._loss(dec, self.sample(gen, src, ctx, key), key, grad=True)
                dec.zero_grad(set_to_none=True)
            dec.zero_grad(set_to_none=True)
            out["normal"], grads["normal"] = self._normal(dec, current, replay, ctx, "normal")
            dec.zero_grad(set_to_none=True)
        finally:
            for m, p in zip(drops, saved): m.p = p
            dec.train(was_training)
        return out, grads


@torch.no_grad()
def score_items(dec, items):
    """Label-free score summaries for synthetic masks; no real anomaly data."""
    was = dec.training; dec.eval(); pos, neg, per = [], [], []
    for k in range(0, len(items), 8):
        chunk = items[k:k + 8]
        en, de = dec(*torch.stack([x["features"] for x in chunk]).to(next(dec.parameters()).device).unbind(1))
        maps = RO.grid([e.float() for e in en], [d.float() for d in de]).float().cpu()
        for it, s in zip(chunk, maps):
            p, n = s[it["coverage"] >= 0.5].numpy(), s[it["coverage"] == 0].numpy()
            if len(p) and len(n):
                pos.append(p); neg.append(n)
                y = np.r_[np.ones(len(p)), np.zeros(len(n))]
                per.append(float(average_precision_score(y, np.r_[p, n])))
    dec.train(was)
    if not pos: return {"pooled_ap": None, "per_image_ap": None, "background_p99": None, "margin": None, "items": 0}
    p, n = np.concatenate(pos), np.concatenate(neg)
    return {"pooled_ap": float(average_precision_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n])),
            "per_image_ap": float(np.mean(per)), "background_p99": float(np.quantile(n, .99)),
            "margin": float(np.mean(p) - np.mean(n)), "items": len(per)}


def ranking_audit(dec, bank, allowed, current_cat, seed):
    """A: within-image; B: category-balanced cross-image hard-negative ranking."""
    out = {"eligibility": {}, "within_image_A": {}, "cross_image_B": {}, "clean_counterpart_fallback": {}}
    for gen in BK.GENERATORS:
        items = [x for x in bank[gen] if x["host"] in allowed]
        cats = sorted({x["host"][0] for x in items})
        out["within_image_A"][gen] = score_items(dec, items)
        out["within_image_A"][gen]["per_category"] = {}
        out["cross_image_B"][gen] = {"per_category": {}, "eligible_categories": [], "ineligible_categories": []}
        for cat in cats:
            cat_items = [x for x in items if x["host"][0] == cat]
            hosts = sorted({tuple(x["host"]) for x in cat_items})
            eligible = len(hosts) >= 2
            out["eligibility"].setdefault(gen, {})[cat] = {"distinct_hosts": len(hosts), "eligible": eligible,
                                                             "items": len(cat_items), "source": "legal_allowed_hosts"}
            out["within_image_A"][gen]["per_category"][cat] = score_items(dec, cat_items)
            if not eligible:
                out["cross_image_B"][gen]["ineligible_categories"].append(cat); continue
            out["cross_image_B"][gen]["eligible_categories"].append(cat)
            # fixed category-balanced host sample, with hard-negative background tail.
            chosen = [h for h in hosts if h[0] == cat][:8]
            chosen_items = [x for x in cat_items if tuple(x["host"]) in chosen]
            vals = []
            was = dec.training; dec.eval()   # R3: B ran in the caller's train mode (bottleneck dropout active)
            with torch.no_grad():
                dev = next(dec.parameters()).device
                for k in range(0, len(chosen_items), 8):
                    ch = chosen_items[k:k + 8]
                    en, de = dec(*torch.stack([x["features"] for x in ch]).to(dev).unbind(1))
                    maps = RO.grid([e.float() for e in en], [d.float() for d in de]).float().cpu()
                    for it, m in zip(ch, maps):
                        vals.append((m[it["coverage"] >= .5].numpy(), m[it["coverage"] == 0].numpy()))
            dec.train(was)
            positives = np.concatenate([x[0] for x in vals]); backgrounds = np.concatenate([x[1] for x in vals])
            hard = np.sort(backgrounds)[-min(64, len(backgrounds)):]
            margins = [float(p.mean() - hard.mean()) for p, _ in vals]
            pooled_y = np.r_[np.ones(len(positives)), np.zeros(len(hard))]
            pooled_s = np.r_[positives, hard]
            per_ap = [float(average_precision_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n])) for p, n in vals]
            out["cross_image_B"][gen]["per_category"][cat] = {
                "hosts": [list(x) for x in chosen], "distinct_hosts": len(chosen), "positive_patches": len(positives),
                "hard_negative_patches": len(hard), "pooled_ap": float(average_precision_score(pooled_y, pooled_s)),
                "per_image_ap": float(np.mean(per_ap)), "background_p99": float(np.quantile(backgrounds, .99)),
                "ranking_margin": float(np.mean(margins)), "hard_negative_margin": float(positives.mean() - hard.mean())}
    return out


def _batch_preview(current, replay, gen, batch, i, iters):
    g = torch.Generator(); g.set_state(gen.get_state())
    cur = DR.Batches(len(current), batch // 2, g); rep = DR.Batches(len(replay), batch // 2, g)
    # Reconstruct the Batches queues by consuming the same number of calls from a fresh generator.
    # The actual loop has one Batches object whose queue persists, so callers pass a clone of those queues.
    return g


def fast_one_step(dec, opt, sch, x, p, a, device):
    dec.train(); opt.zero_grad(set_to_none=True); total = 0.0
    for xb in x.to(device).split(DR.MICRO):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            en, de = dec(xb[:, 0], xb[:, 1])
        loss = O.global_cosine_hm_percent([e.float() for e in en], [d.float() for d in de], p=p,
                                          factor=O.RECIPE["hm_factor"]) * len(xb) / len(x)
        loss.backward(); total += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=O.RECIPE["clip_grad_norm"])
    base = sch.get_last_lr()[0]
    for g in opt.param_groups: g["lr"] = base * a[g["block"]]
    opt.step(); sch.step()
    return total


def counterfactual(dec, opt, sch, recorder, current, replay, ctx, x_cpu, p, source_s, device):
    phi0 = torch.zeros(len(BLOCKS), dtype=torch.float64)
    q = torch.tensor([source_s[b] for b in BLOCKS], dtype=torch.float64)
    hg = torch.tensor(phi_hypergrad(phi0, source_s))
    rng = torch.get_rng_state()
    variants = {"identity": phi0.clone(), "q3_replay": -CF_PHI_NORM * hg / (hg.norm() + 1e-300)}
    rg = fixed_gen(5042711 + recorder.stage * 100 + ctx["iter"])
    z = torch.randn(len(BLOCKS), generator=rg, dtype=torch.float64); variants["matched_norm_random"] = CF_PHI_NORM * z / z.norm()
    both = {b: source_s[b] + ctx["q3_current_credit"][b] for b in BLOCKS}
    hboth = torch.tensor(phi_hypergrad(phi0, both)); variants["q3_current_plus_replay"] = -CF_PHI_NORM * hboth / (hboth.norm() + 1e-300)
    # Fixed independent query identities for every arm.
    q2r = recorder.sample("Q2", "replay", ctx, "cf_q2_replay")
    q2c = recorder.sample("Q2", "current", ctx, "cf_q2_current")
    q3r = recorder.sample("Q3", "replay", ctx, "cf_q3_replay")
    qc, _ = recorder.normal_query(current, replay, ctx, "cf_normal_current")
    # Keep snapshots on CPU.  Cloning a second decoder and Adam state on the
    # 6-GiB device exceeds the locked 2.6-GiB process cap.
    state0 = {n: p.detach().cpu().clone() for n, p in dec.state_dict().items()}
    opt0 = cpu_tree(opt.state_dict()); sch0 = cpu_tree(sch.state_dict())
    start_hash, opt_hash, rng_hash = state_hash(dec), object_hash(opt0), tensor_hash(rng)
    result = {"event": ctx["iter"], "phi_norm": CF_PHI_NORM, "horizon": 1, "starting_state_hash": start_hash,
              "optimizer_state_hash": opt_hash, "rng_hash": rng_hash,
              "queries": {"q2_replay": [list(x["host"]) for x in q2r], "q2_current": [list(x["host"]) for x in q2c],
                          "q3_replay": [list(x["host"]) for x in q3r], "normal_current": qc}, "variants": {}}
    cuda_rng = torch.cuda.get_rng_state() if device.type == "cuda" else None
    for name, phi in variants.items():
        # new_decoder seeds torch.manual_seed; keep construction and the branch update in a fork.
        with torch.random.fork_rng(devices=_devices(device)):
            torch.set_rng_state(rng)
            if cuda_rng is not None: torch.cuda.set_rng_state(cuda_rng)
            dec.load_state_dict(state0)
            opt.load_state_dict(cpu_tree(opt0))
            sch.load_state_dict(cpu_tree(sch0))
            h = math.log(2.0) / 2 * torch.tanh(phi); av = torch.exp(h) / torch.exp(h).mean()
            amap = dict(zip(BLOCKS, av.tolist()))
            fast_one_step(dec, opt, sch, x_cpu, p, amap, device)
            # R3: measure inside the fork with dropout off.  R2 ran the normal-query forward in train mode with the
            # bottleneck dropout active, outside the fork: it consumed the trajectory's RNG (parity loss) and made
            # the variants' normal metric differ by dropout noise.
            met = {"q2_replay": score_items(dec, q2r), "q2_current": score_items(dec, q2c),
                   "q3_replay": score_items(dec, q3r)}
            drops = [m for m in dec.modules() if isinstance(m, torch.nn.Dropout)]
            saved = [m.p for m in drops]
            for m in drops: m.p = 0.0
            try:
                with torch.no_grad():
                    x = current[qc].to(device); en, de = _forward(dec, x, device, device.type == "cuda")
                    met["normal_current"] = float(O.global_cosine(en, de))
            finally:
                for m, v in zip(drops, saved): m.p = v
        met["a"] = amap; met["phi"] = phi.tolist(); met["state_hash"] = state_hash(dec)
        met["effective_update_norm"] = float(sum((p.detach().cpu() - state0[n].float()).double().pow(2).sum()
                                                   for n, p in dec.named_parameters()).sqrt())
        result["variants"][name] = met
    dec.load_state_dict(state0)
    opt.load_state_dict(cpu_tree(opt0))
    sch.load_state_dict(cpu_tree(sch0))
    result["restored_state_hash"] = state_hash(dec)
    result["restored_optimizer_hash"] = object_hash(opt.state_dict())
    result["restoration_exact"] = result["restored_state_hash"] == start_hash and result["restored_optimizer_hash"] == opt_hash
    return result


def event_record(dec, recorder, current, replay, ctx, old, a, base_lr, device):
    q2 = ctx["iter"] in Q2_EVENT_ITERS
    out, grads = recorder.event(dec, current, replay, ctx, q2)
    delta = delta_groups(dec, old)
    groups = {}
    for key, g in grads.items():
        if g is None: groups[key] = None; continue
        s = dot_groups(g, delta); hn = phi_hypergrad(torch.zeros(len(BLOCKS), dtype=torch.float64), s)
        groups[key] = {"loss": out[key]["loss"], "used": out[key].get("used"), "skipped": out[key].get("skipped"),
                       "hosts": out[key].get("hosts", out[key].get("query_current")), "dJ_da": s,
                       "phi_hypergrad": dict(zip(BLOCKS, hn)), "credit_norm": float(np.linalg.norm(list(s.values()))),
                       "grad_norm": block_norms(g), "update_norm": block_norms(delta),
                       "credit_cosine_to_normal": None, "sign_agreement_to_normal": None,
                       "gradient_scale_to_normal": None}
    normal = groups["normal"]
    for key, row in groups.items():
        if row is not None and key != "normal":
            row["credit_cosine_to_normal"] = cosine(row["dJ_da"], normal["dJ_da"])
            row["sign_agreement_to_normal"] = sign_agreement(row["dJ_da"], normal["dJ_da"])
            row["gradient_scale_to_normal"] = row["grad_norm"].copy()
            for b in BLOCKS: row["gradient_scale_to_normal"][b] /= normal["grad_norm"][b] + 1e-300
    return {"stage": recorder.stage, "iter": ctx["iter"], "phase": phase(ctx["iter"]), "lr": base_lr,
            "a_used": a, "delta_norm": block_norms(delta), "groups": groups, "queries": out,
            "q2_replay_recorded": q2}


def train_stage(dec, current, replay, stage, bank, hosts, n, device, run):
    c = C.STREAM[stage - 1]; done = list(C.STREAM[:stage - 1]); idx = C.buffer_index(SEED, n, done)
    replay_keys = [(d, i) for d in done for i in idx[d]]
    allowed = BK.allowed(hosts, SEED, n, stage)
    recorder = CreditRecorder(bank, hosts, allowed, c, replay_keys, stage, 2711 * 1000 + 500 + stage, device)
    dseed, gseed = RB.stage_seeds("R15", SEED, stage); torch.manual_seed(dseed); gen = torch.Generator().manual_seed(gseed)
    opt = block_optimizer(dec); sch = O.make_scheduler(opt, R1.ITERS); batch = O.RECIPE["batch_size"]
    cur = DR.Batches(len(current), batch // 2, gen); rep = DR.Batches(len(replay), batch // 2, gen)
    named = list(dec.named_parameters()); params = [p for _, p in named]; losses, events = [], []
    pending = None; t0 = time.perf_counter()
    for i in range(R1.ITERS):
        p_hm = min(O.RECIPE["hm_p_final"] * i / round(.1 * R1.ITERS), O.RECIPE["hm_p_final"])
        ci, ri = cur.next(), rep.next(); x = torch.cat([current[ci], replay[ri]]); x = x[torch.randperm(len(x), generator=gen)]
        x = x.to(device); opt.zero_grad(set_to_none=True); total = 0.0
        for xb in x.split(DR.MICRO):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                en, de = dec(xb[:, 0], xb[:, 1])
            loss = O.global_cosine_hm_percent([e.float() for e in en], [d.float() for d in de], p=p_hm,
                                              factor=O.RECIPE["hm_factor"]) * len(xb) / len(x)
            loss.backward(); total += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=O.RECIPE["clip_grad_norm"])
        require(np.isfinite(total) and bool(torch.isfinite(norm)), f"non-finite identity step {stage}/{i}")
        base = sch.get_last_lr()[0]
        old = {n: p.detach().cpu().clone() for n, p in named} if i % K == K - 1 else None
        for g in opt.param_groups: g["lr"] = base
        opt.step(); sch.step(); losses.append(total)
        if old is not None:
            ctx = {"iter": i, "ci": ci, "ri": ri, "q3_current_credit": {b: 0.0 for b in BLOCKS}}
            er = event_record(dec, recorder, current, replay, ctx, old, {b: 1.0 for b in BLOCKS}, base, device)
            events.append(er)
            if (stage, i) in CF_EVENTS:
                # Use the recorded replay and current credits to construct both directions.
                # Peek the exact next Batches calls without advancing the real generator.
                gpeek = torch.Generator(); gpeek.set_state(gen.get_state())
                cpeek = DR.Batches(len(current), batch // 2, gpeek); cpeek.queue = list(cur.queue)
                rpeek = DR.Batches(len(replay), batch // 2, gpeek); rpeek.queue = list(rep.queue)
                c2, r2 = cpeek.next(), rpeek.next()
                x2 = torch.cat([current[c2], replay[r2]])
                x2 = x2[torch.randperm(len(x2), generator=gpeek)]
                ctx["q3_current_credit"] = er["groups"]["Q3_current"]["dJ_da"]
                er["counterfactual"] = counterfactual(dec, opt, sch, recorder, current, replay, ctx, x2,
                                                       min(O.RECIPE["hm_p_final"] * (i + 1) / round(.1 * R1.ITERS), O.RECIPE["hm_p_final"]),
                                                       er["groups"]["Q3_replay"]["dJ_da"], device)
                pending = (er, er["counterfactual"]["variants"]["identity"]["state_hash"])
        if pending is not None and i == pending[0]["iter"] + 1:
            pending[0]["counterfactual"]["identity_parity_next_step"] = state_hash(dec) == pending[1]
            pending = None
    seconds = time.perf_counter() - t0
    audit = ranking_audit(dec, bank, allowed, c, stage)
    return {"stage": stage, "category": c, "seconds": seconds, "losses": losses, "events": events,
            "audit": audit, "allowed_hosts": [list(x) for x in sorted(allowed)], "replay_keys": [list(x) for x in replay_keys],
            "controller": "identity/no feedback", "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None}


def start_checkpoint(stage):
    rec = D.read(C.OUT / f"runs/R15_seed{SEED}/STAGE{stage}.json"); path = D.ROOT / rec["checkpoint"]
    require(sha(path) == rec["checkpoint_sha256"], f"checkpoint changed: {path}")
    return rec, path, torch.load(path, weights_only=True)["state"]


def preflight():
    """CPU cost projection from locked R15 timings plus explicit recorder/counterfactual accounting."""
    rows = [D.read(C.OUT / f"runs/R15_seed{SEED}/STAGE{s}.json") for s in (7, 8, 9)]
    base = float(np.mean([r["seconds"] for r in rows]))
    events = math.floor(R1.ITERS / K)
    # Four gradient objectives at Q2 sampling events, three otherwise; each is a micro-batched forward/backward.
    outer_calls = events * 3 + len(Q2_EVENT_ITERS) * 1
    cf = len(CF_EVENTS) * 4  # identity, retention, random, current+replay; one fast step each
    projected = 3 * base * (1.0 + 0.035 * (outer_calls / (3 * events)) + 0.055 * (cf / (3 * events)))
    result = {"base_stage_seconds": base, "stages": 3, "slow_events_per_stage": events,
              "outer_backward_calls": outer_calls, "counterfactual_branch_steps": cf,
              "projected_gpu_hours": projected / 3600, "limit_gpu_hours": 1.5,
              "proceed": bool(projected / 3600 <= 1.5), "model": "R15 measured stage time + bounded call overhead",
              "q2_event_iters": list(Q2_EVENT_ITERS), "cf_events": [list(x) for x in sorted(CF_EVENTS)],
              "persistent_bytes": 0, "cpu_snapshot_bytes": 0,
              "storage_accounting": {"replay24_images": 24, "slow_controller_state": 0,
                                     "synthetic_feature_cache": "precomputed BANK_seed2711 bytes disclosed at prereg",
                                     "cpu_parameter_snapshots": "one per counterfactual event, measured during run"}}
    D.write_json(PREFLIGHT, result); print(result, flush=True)
    return result


def config():
    return {"tag": TAG, "seed": SEED, "start": START, "final": FINAL, "K": K, "q2_event_iters": list(Q2_EVENT_ITERS),
            "counterfactual_events": [list(x) for x in sorted(CF_EVENTS)], "counterfactual_phi_norm": CF_PHI_NORM,
            "counterfactual_horizon": CF_HORIZON, "tau": TAU, "n_neg": N_NEG, "identity_feedback": False,
            "audit_B": "category-balanced legal historical replay hosts; 8 hosts/category maximum; top 64 backgrounds",
            "fallback": "clean counterpart excluded from B and reported separately if evaluated"}


def prereg():
    require(PREFLIGHT.exists(), "run preflight first"); pf = D.read(PREFLIGHT); require(pf["proceed"], "projected cost exceeds 1.5 GPU-hours")
    rec, path, _ = start_checkpoint(START); bank_man = OUT / f"BANK_seed{SEED}.json"
    data = {"utc": DR.now(), "task": {TASK: sha(D.ROOT / TASK)}, "config": config(),
            "sources": {s: sha(D.ROOT / s) for s in ("exps/nlcad/retention_credit.py", "exps/nlcad/controller.py", "exps/nlcad/fast.py", "exps/nlcad/objectives.py", "exps/nlcad/objectives_s.py")},
            "start_checkpoint": {"path": rec["checkpoint"], "sha256": rec["checkpoint_sha256"]},
            "bank_manifest": {"path": str(bank_man.relative_to(D.ROOT)), "sha256": sha(bank_man)},
            "preflight_sha256": sha(PREFLIGHT), "gpu_policy": "one recorder job only; no S1/LH-REC/Stage-D/NLCAD-05B",
            "labels": "no real anomaly labels or Stage-D native AP"}
    require(not PREREG.exists(), "already preregistered"); D.write_json(PREREG, data); print("preregistered", sha(PREREG), flush=True)


def run(device="cuda"):
    pr = D.read(PREREG)
    for s, h in pr["sources"].items(): require(sha(D.ROOT / s) == h, f"source changed: {s}")
    require(sha(D.ROOT / TASK) == pr["task"][TASK], "task changed"); require(config() == pr["config"], "config changed")
    require(device == "cuda" and torch.cuda.is_available(), "full recorder requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; NUM.use(True); DR.configure()
    proto = RB.verify_lock(); n = R1.n_train(proto); bank, hosts = BK.load_bank(SEED); RUN.mkdir(parents=True, exist_ok=True)
    _, _, sd = start_checkpoint(START); dec = DR.new_decoder(SEED).to(device); dec.load_state_dict(sd)
    all_rows = []; t0 = time.perf_counter()
    for s in range(START + 1, FINAL + 1):
        done = list(C.STREAM[:s - 1]); idx = C.buffer_index(SEED, n, done)
        result = train_stage(dec, C.load_features(C.STREAM[s - 1]),
                             torch.stack([C.load_features(d)[i] for d in done for i in idx[d]]),
                             s, bank, hosts, n, torch.device(device), RUN)
        state = {k: v.detach().cpu() for k, v in dec.state_dict().items()}; ck = RUN / f"stage{s}_{C.STREAM[s - 1]}.pt"; torch.save({"state": state}, ck)
        ref = D.read(C.OUT / f"runs/R15_seed{SEED}/STAGE{s}.json"); ref_state = torch.load(D.ROOT / ref["checkpoint"], weights_only=True)["state"]
        parity = all(torch.equal(state[k], ref_state[k]) for k in state)
        row = {k: v for k, v in result.items() if k != "losses"}; row["checkpoint"] = str(ck.relative_to(D.ROOT)); row["checkpoint_sha256"] = sha(ck)
        row["identity_checkpoint_sha256"] = ref["checkpoint_sha256"]; row["identity_state_bitwise_parity"] = parity
        row["storage"] = {"replay24_images": 24, "synthetic_feature_cache_bytes": sum((BK.BANK / f"seed{SEED}" / f"{g}_{c}.pt").stat().st_size for g in BK.GENERATORS for c in C.STREAM if (BK.BANK / f"seed{SEED}" / f"{g}_{c}.pt").exists()),
                           "cpu_parameter_snapshot_bytes": sum(p.numel() * p.element_size() for p in dec.parameters()),
                           "slow_controller_persistent_bytes": 0}
        D.write_json(RUN / f"STAGE{s}.json", row); all_rows.append(row); print("stage", s, C.STREAM[s - 1], round(result["seconds"], 1), "parity", parity, flush=True)
    summary = summarize_rows(all_rows); summary["actual_gpu_seconds"] = time.perf_counter() - t0; summary["actual_gpu_hours"] = summary["actual_gpu_seconds"] / 3600
    D.write_json(RUN / "SUMMARY.json", summary); return summary


def summarize_rows(rows=None):
    rows = rows or [D.read(RUN / f"STAGE{s}.json") for s in range(7, 10)]
    ev = [e for r in rows for e in r["events"]]
    result = {"stages": [r["stage"] for r in rows], "events": len(ev), "identity_parity": all(r["identity_state_bitwise_parity"] for r in rows),
              "per_stage_seconds": {str(r["stage"]): r["seconds"] for r in rows}, "actual_gpu_hours": None, "credit": {}}
    for key in ("Q3_current", "Q3_replay", "Q2_replay", "normal"):
        vals = [e["groups"][key] for e in ev if e["groups"].get(key) is not None]
        result["credit"][key] = {"events": len(vals), "mean_norm": float(np.mean([v["credit_norm"] for v in vals])) if vals else None,
                                  "mean_cosine_to_normal": float(np.mean([v["credit_cosine_to_normal"] for v in vals if v["credit_cosine_to_normal"] is not None])) if key != "normal" and vals else None,
                                  "mean_sign_agreement_to_normal": float(np.mean([v["sign_agreement_to_normal"] for v in vals if v["sign_agreement_to_normal"] is not None])) if key != "normal" and vals else None,
                                  "mean_gradient_scale": float(np.mean([np.mean(list(v["gradient_scale_to_normal"].values())) for v in vals if v["gradient_scale_to_normal"]])) if key != "normal" and vals else None}
    result["counterfactuals"] = [e["counterfactual"] for e in ev if "counterfactual" in e]
    result["audit"] = {str(r["stage"]): r["audit"] for r in rows}
    return result


def amend_r3():
    """R3 amendment: same config, events, branches and queries as R2; only the two dropout/RNG defects are fixed."""
    require(not PREREG.exists(), "already amended")
    r2 = D.read(PREREG_R2)
    require(r2["config"] == config(), "config differs from R2")
    data = dict(r2, utc=DR.now(), parent_sha256=sha(PREREG_R2),
                sources={s: sha(D.ROOT / s) for s in r2["sources"]},
                amendment={"parent": str(PREREG_R2.relative_to(D.ROOT)),
                           "reason": "R2 stage 7 lost identity parity (identity_parity_next_step False, restoration "
                                     "exact): the counterfactual normal-query forward ran in train mode with the "
                                     "bottleneck dropout active outside the forked RNG, consuming the trajectory's "
                                     "CUDA RNG. R3 measures every branch metric inside the fork with dropout off, "
                                     "and runs ranking-audit B in eval mode. Config, events, queries unchanged.",
                           "r2_source_sha256": r2["sources"]["exps/nlcad/retention_credit.py"],
                           "r2_aborted_run": "results/hope_cad/nlcad/archive_05a_r2/run_R2_aborted (stage 7 only)"})
    D.write_json(PREREG, data); print("amended", sha(PREREG), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=("preflight", "prereg", "amend_r3", "run", "summarize")); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    {"preflight": preflight, "prereg": prereg, "amend_r3": amend_r3, "run": lambda: run(a.device), "summarize": summarize_rows}[a.command]()


if __name__ == "__main__": main()
