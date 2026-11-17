"""NSGP-F0 (agents/tasks/current_task.md): normal-support geometry preservation feasibility — shared definitions.

Protocol: agents/plans/nsgp_f0_protocol.md (hashed into results/hope_cad/nsgp_f0/LOCK.json).
Canonical modules are imported, never edited:
  bank        = C2-DOM DOM_PQ_B2 boundary checkpoints (exps.nl_sgm_01a_c2dom.stream)
  features    = CALM-02 ViT-B/8 layer-9 train/good cache via exps.nl_sgm_01a.common.Features (refuses test paths)
  PQ codec    = exps.nl_sgm_01a.codecs ; exact 1-NN = exps.nl_sgm_01a.exact.nn (FP64 direct-difference Euclidean)
Normal-only: nothing here opens MVTec test features, labels or masks.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from exps.nl_sgm_01a import am02_run as R2
from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a import exact as X
from exps.nl_sgm_01a_c2dom import common as D2
from exps.task_p.p0_normal_support import require

ROOT = C.ROOT
OUT = ROOT / "results/hope_cad/nsgp_f0"
LOCK, PROTOCOL = OUT / "LOCK.json", OUT / "PROTOCOL.json"
PLAN, TASK = "agents/plans/nsgp_f0_protocol.md", "agents/tasks/current_task.md"
ORDER = C.ORDER
MEM_ARM = "DOM_PQ_B2"
MEM_DIR = D2.OUT / "streams" / MEM_ARM
DIM, PATCHES = 768, 784

BOUNDARIES = (0, 4, 9, 14)                       # 0-based; reported as boundaries 1, 5, 10, 15
FINAL = 14
K_C0 = {"C": 94, "Q": 127}                       # PCA-int8 components at the 0.75*B0 / B0 final-boundary budgets
K_N0 = {"C": 90, "Q": 123}                       # learned decoder components at the same budgets
HID = 256
TRAIN = {"steps": 4000, "lr": 1e-3, "lr_min": 1e-5, "clip": 1.0, "seed": 20261114, "log_every": 100}
PQ_SEED, PQ_ITERS = C.SEED, 25                   # P0 static PQ refit: the canonical PQ fit with the C2-DOM seed
FITQ_KEY, FITQ_N = "NSGP-F0/fitq/v1|", 6
B0_FINAL_BYTES = 15_359_984
ABS_BUDGET = 15_360_000
GATE = {"quality_bytes_frac": 1.00, "quality_err_frac": 0.90, "compression_bytes_frac": 0.75,
        "compression_err_frac": 1.03, "compression_ci_hi_frac": 0.03, "guard_frac": 1.10,
        "guard_negligible_frac": 0.10, "select_rule_frac": 1.03, "quantile": 0.95}
BOOT = {"B": 2000, "seed": 20261108, "unit": "category (paired, resample categories with replacement)"}
CAPS = {"rss_gb": 8.5, "vram_gib": 3.0, "wall_h_per_command": 3.0}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


sha, text_sha, write_json = C.sha, C.text_sha, C.write_json


def read_json(path):
    return json.loads(Path(path).read_text())


class Log:
    def __init__(self, name):
        (OUT / "logs").mkdir(parents=True, exist_ok=True)
        self.f = (OUT / "logs" / f"{name}.log").open("a")

    def __call__(self, *a):
        s = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def verify_lock():
    lk = read_json(LOCK)
    require(not list(OUT.glob("AMENDMENT_*.json")), "NSGP-F0 admits no amendments: a science change needs a new protocol")
    bad = [s for s, h in lk["sources"].items() if sha(ROOT / s) != h]
    require(not bad, f"LOCK: source changed after lock: {bad}")
    require(sha(PROTOCOL) == lk["protocol_sha256"], "LOCK: protocol changed after lock")
    return lk


def setup(threads=6):
    C.set_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    co = C.cohorts()
    require(C.r15_digest()[:8] == C.R15_DIGEST, "R15 cohort digest mismatch")
    n = {c: len(co[c]["writers"]) for c in ORDER}
    return co, n


# ------------------------------------------------------------------ cohorts

def fitq(co, part, c):
    """Training-only selection queries: FITQ_N FIT-pool images of c (hash-first). Never stored, never a probe."""
    fit = [p for _, p in part[c]["fit"]]
    return sorted(sorted(fit, key=lambda p: text_sha(FITQ_KEY + p))[:FITQ_N])


def heldout_paths(c):
    from exps.vitb8_q2 import common as V
    return list(torch.load(V.CACHE / f"heldout_{c}.pt", weights_only=True, mmap=True)["paths"])


def heldout_rows(c):
    from exps.vitb8_q2 import common as V
    return V.heldout_features(c).reshape(-1, DIM).clone()


# ------------------------------------------------------------------ bank (identity-matched)

def bank(t):
    """{c: {"ids": int64 sorted, "codes": uint8 [R_c,96] (same order), "codebook": fp16}} at boundary t (hash-verified)."""
    rec = read_json(MEM_DIR / f"boundary_t{t:02d}.json")
    p = MEM_DIR / f"ckpt_t{t:02d}.pt"
    require(sha(p) == rec["ckpt_sha256"], f"bank checkpoint changed {p}")
    ck = torch.load(p, weights_only=False)
    nr = int(ck["header"][1:].sum())
    ids, own = ck["ids"][:nr].long(), ck["owner"][:nr].long()
    require(int(ck["header"][0]) == t and bool((ck["ids"][nr:] == -1).all()), "bank header/padding")
    out = {}
    for k in range(t + 1):
        m = own == k
        g, cd = ids[m], ck["codes"][:nr][m]
        o = torch.argsort(g)
        c = ORDER[k]
        require(len(g) == rec["per_category_rows"][c] and D2.gid_sha(g) == rec["per_category_gids_sha"][c], f"ledger {c}")
        out[c] = {"ids": g[o].contiguous(), "codes": cd[o].contiguous(), "codebook": ck["codebooks"][k]}
    return out, rec


def address(n, c, ids):
    """gid -> (writer index j, patch p) of category c (exps.nl_sgm_01a.am02_run.gid_base layout)."""
    rel = ids - R2.gid_base(n, c)
    require(bool((rel >= 0).all()) and bool((rel < n[c] * PATCHES).all()), f"gid outside category {c}")
    return rel // PATCHES, rel % PATCHES


def o0_rows(F, co, n, c, ids):
    """Original FP32 features of the stored ids (identity-matched oracle)."""
    j, p = address(n, c, ids)
    uj = torch.unique(j)
    imgs = F.images([co[c]["writers"][int(v)] for v in uj])            # [u, 784, 768]
    pos = torch.searchsorted(uj, j)
    return imgs[pos, p].contiguous()


def b0_rows(b):
    return K.decode(b["codes"], b["codebook"]).contiguous()


# ------------------------------------------------------------------ byte ledgers (persistent state only)

def bytes_common(R):
    return 64 + 5 * R                                   # header int32[16]; ids int32; owner int8 per stored row


def bytes_pq(R, n):
    return bytes_common(R) + R * K.PQ_ROW_BYTES + n * K.CODEBOOK_BYTES


def bytes_c0(R, n, k):
    return bytes_common(R) + n * (DIM * 2 + DIM * k * 2 + 4 * k) + R * k


def bytes_n0(R, n, k):
    return bytes_c0(R, n, k) + HID * k * 2 + HID * 2 + DIM * HID * 2


def state_bytes(st):
    return int(sum(v.numel() * v.element_size() for v in st.values()))


def common_state(t, cats, ids):
    hdr = torch.zeros(16, dtype=torch.int32)
    hdr[0] = t
    for c in cats:
        hdr[1 + ORDER.index(c)] = len(ids[c])
    return {"header": hdr, "ids": torch.cat([ids[c] for c in cats]).to(torch.int32),
            "owner": torch.cat([torch.full((len(ids[c]),), ORDER.index(c), dtype=torch.int8) for c in cats])}


def split_rows(st, cats):
    """Row slices per category from the header (state order = category order, ids ascending within category)."""
    out, a = {}, 0
    for c in cats:
        r = int(st["header"][1 + ORDER.index(c)])
        out[c] = slice(a, a + r)
        a += r
    require(a == len(st["ids"]), "state row count")
    return out


# ------------------------------------------------------------------ C0: per-category PCA, int8 codes

def pca(X, k):
    """Top-k principal axes of FP32 rows (FP64 covariance eigendecomposition, deterministic sign: max |entry| > 0)."""
    X64 = X.double()
    mu = X64.mean(0)
    Z = X64 - mu
    w, V = torch.linalg.eigh(Z.T @ Z / len(X))
    idx = torch.argsort(w, descending=True)[:k]
    U = V[:, idx]
    s = torch.sign(U.gather(0, U.abs().argmax(0, keepdim=True)))
    return mu, U * s, w[idx]


def quant8(u):
    """Per-component symmetric int8: scale = max|u| / 127 (FP32), q = round(u / scale) in [-127, 127]."""
    s = (u.abs().amax(0) / 127.0).float().clamp_min(1e-12)
    q = torch.round(u / s).clamp(-127, 127).to(torch.int8)
    return q, s


def build_c0(t, rows, k):
    cats = list(rows)
    st = common_state(t, cats, {c: rows[c]["ids"] for c in cats})
    mean, basis, scale, codes = [], [], [], []
    for c in cats:
        X = rows[c]["x"]
        mu, U, _ = pca(X, k)
        mu16, U16 = mu.half(), U.half()
        coef = (X.double() - mu16.double()) @ U16.double()
        q, s = quant8(coef)
        mean.append(mu16); basis.append(U16); scale.append(s); codes.append(q)
    st.update(mean=torch.stack(mean), basis=torch.stack(basis), scale=torch.stack(scale), codes=torch.cat(codes))
    require(state_bytes(st) == bytes_c0(len(st["ids"]), len(cats), k), "C0 ledger")
    return st


def decode_c0(st, cats):
    sl = split_rows(st, cats)
    out = {}
    for i, c in enumerate(cats):
        u = st["codes"][sl[c]].float() * st["scale"][i][None]
        out[c] = (st["mean"][i].float()[None] + u @ st["basis"][i].float().T).contiguous()
    return out


# ------------------------------------------------------------------ N0: minimal learned code-to-feature decoder

def n0_forward(mu, A, W1, b1, W2, u):
    """x_hat = mu_c + A_c u + W2 gelu(W1 u + b1)  (per-category affine head + shared nonlinear residual)."""
    return mu[None] + u @ A.T + F.gelu(u @ W1.T + b1[None]) @ W2.T


def decode_n0(st, cats):
    sl = split_rows(st, cats)
    W1, b1, W2 = st["W1"].float(), st["b1"].float(), st["W2"].float()
    out = {}
    for i, c in enumerate(cats):
        u = st["codes"][sl[c]].float() * st["scale"][i][None]
        with torch.no_grad():
            out[c] = n0_forward(st["mean"][i].float(), st["basis"][i].float(), W1, b1, W2, u).contiguous()
    return out


def train_n0(t, rows, k, device="cuda", steps=None, log=None):
    """Full-batch Adam on the stored bank rows (static, offline). Returns (state, diagnostics).
    Init = per-category PCA (A_c = U_c diag(sigma_c), u = coef / sigma_c) with W2 = 0, i.e. exactly C0 before
    quantization; the shared residual and all codes are then learned by mean squared reconstruction error."""
    steps = TRAIN["steps"] if steps is None else steps
    cats = list(rows)
    torch.manual_seed(TRAIN["seed"])
    g = torch.Generator().manual_seed(TRAIN["seed"])
    mus, As, us, Xs, init_mse = [], [], [], [], []
    for c in cats:
        X = rows[c]["x"]
        mu, U, _ = pca(X, k)
        coef = (X.double() - mu) @ U
        sig = coef.std(0).clamp_min(1e-6)
        mus.append(torch.nn.Parameter(mu.float().to(device)))
        As.append(torch.nn.Parameter((U * sig[None]).float().to(device)))
        us.append(torch.nn.Parameter((coef / sig[None]).float().to(device)))
        Xs.append(X.to(device))
    W1 = torch.nn.Parameter((torch.randn(HID, k, generator=g) / math.sqrt(k)).to(device))
    b1 = torch.nn.Parameter(torch.zeros(HID, device=device))
    W2 = torch.nn.Parameter(torch.zeros(DIM, HID, device=device))
    params = mus + As + us + [W1, b1, W2]
    opt = torch.optim.Adam(params, lr=TRAIN["lr"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=TRAIN["lr_min"])
    N = sum(len(x) for x in Xs) * DIM
    curve, t0 = [], time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for i in range(len(cats)):
            loss = (n0_forward(mus[i], As[i], W1, b1, W2, us[i]) - Xs[i]).square().sum() / N
            if step < steps:
                loss.backward()
            tot += float(loss.detach())
        require(math.isfinite(tot), f"N0: non-finite loss at step {step}")
        if step == 0:
            init_mse = tot
        if step % TRAIN["log_every"] == 0 or step == steps:
            with torch.no_grad():
                un = torch.cat([u.detach() for u in us])
                comp_sd = torch.cat([u.detach().std(0) for u in us])
                gn = math.sqrt(sum(float(p.grad.square().sum()) for p in params if p.grad is not None)) if step < steps else float("nan")
            curve.append({"step": step, "mse": tot, "grad_norm": gn, "code_norm_mean": float(un.norm(dim=1).mean()),
                          "code_norm_max": float(un.norm(dim=1).max()), "min_component_sd": float(comp_sd.min()),
                          "W2_norm": float(W2.detach().norm()), "lr": sch.get_last_lr()[0]})
            if log is not None and step % (TRAIN["log_every"] * 10) == 0:
                log("N0", f"k={k}", "step", step, "mse", round(tot, 6), "gn", round(gn, 5) if step < steps else None)
        if step == steps:
            break
        gn_t = torch.nn.utils.clip_grad_norm_(params, TRAIN["clip"])
        require(bool(torch.isfinite(gn_t)), f"N0: non-finite gradient at step {step}")
        opt.step()
        sch.step()
    train_s = time.time() - t0
    vram = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    st = common_state(t, cats, {c: rows[c]["ids"] for c in cats})
    codes, scales = [], []
    for u in us:
        q, s = quant8(u.detach().cpu().double())
        codes.append(q); scales.append(s)
    st.update(mean=torch.stack([m.detach().cpu().half() for m in mus]),
              basis=torch.stack([a.detach().cpu().half() for a in As]), scale=torch.stack(scales),
              codes=torch.cat(codes), W1=W1.detach().cpu().half(), b1=b1.detach().cpu().half(),
              W2=W2.detach().cpu().half())
    require(state_bytes(st) == bytes_n0(len(st["ids"]), len(cats), k), "N0 ledger")
    dec = decode_n0(st, cats)
    final_mse = sum(float((dec[c] - rows[c]["x"]).double().square().sum()) for c in cats) / N
    diag = {"k": k, "steps": steps, "init_mse_pca_unquantized": init_mse, "trained_mse_continuous": curve[-1]["mse"],
            "persistent_mse_fp16_int8": final_mse, "train_seconds": train_s, "vram_peak_bytes": vram, "curve": curve,
            "collapse_min_component_sd": curve[-1]["min_component_sd"], "device": device}
    return st, diag


# ------------------------------------------------------------------ P0: static PQ refit on the bank (secondary)

def build_p0(t, rows):
    cats = list(rows)
    st = common_state(t, cats, {c: rows[c]["ids"] for c in cats})
    cbs, codes, sec = [], [], {}
    for c in cats:
        t0 = time.time()
        cb = K.fit(rows[c]["x"][:100_000], seed=PQ_SEED, iters=PQ_ITERS)
        codes.append(K.encode(rows[c]["x"], cb, chunk=1024)); cbs.append(cb)
        sec[c] = time.time() - t0
    st.update(codes=torch.cat(codes), codebooks=torch.stack(cbs))
    require(state_bytes(st) == bytes_pq(len(st["ids"]), len(cats)), "P0 ledger")
    return st, sec


def decode_p0(st, cats):
    sl = split_rows(st, cats)
    return {c: K.decode(st["codes"][sl[c]], st["codebooks"][i]).contiguous() for i, c in enumerate(cats)}


DECODERS = {"C0": decode_c0, "N0": decode_n0, "P0": decode_p0}


# ------------------------------------------------------------------ scoring and statistics

def score(q, x, ids):
    """Canonical score: exact FP64 1-NN Euclidean distance (exps.nl_sgm_01a.exact.nn) and the NN id."""
    return X.nn(q, X.RowSet(x, ids))


def err_stats(d, d0, q=GATE["quantile"]):
    e = np.asarray(d, dtype=np.float64) - np.asarray(d0, dtype=np.float64)
    a = np.abs(e)
    return {"p95_abs": float(np.quantile(a, q)), "p99_abs": float(np.quantile(a, 0.99)), "max_abs": float(a.max()),
            "mean_abs": float(a.mean()), "mean_signed": float(e.mean()), "inflation_frac": float((e > 0).mean()),
            "suppression_frac": float((e < 0).mean()), "rel_p95_abs": float(np.quantile(a / np.asarray(d0), q))}


def hausdorff(A, B, chunk=2048):
    """Exact two-sided Hausdorff distance between FP32 row sets (FP64 direct-difference via exact.nn)."""
    ia = torch.arange(len(A), dtype=torch.int64)
    ib = torch.arange(len(B), dtype=torch.int64)
    dab = X.nn(A, X.RowSet(B, ib))[0]
    dba = X.nn(B, X.RowSet(A, ia))[0]
    return max(float(dab.max()), float(dba.max())), float(dab.max()), float(dba.max())


def boot(a, b):
    from exps.nl_sgm_01a import am03_eval as A3
    return A3.bootstrap_delta(a, b, seed=BOOT["seed"], B=BOOT["B"])


def gate(bytes_n, bytes_b, e_n, e_b):
    """e_n, e_b: per-category P95 abs score error on the gate cohort at the final boundary (same category order)."""
    e_n, e_b = np.asarray(e_n, dtype=np.float64), np.asarray(e_b, dtype=np.float64)
    mn, mb = float(e_n.mean()), float(e_b.mean())
    gain = boot(e_b, e_n)                              # B0 - N0
    loss = boot(e_n, e_b)                              # N0 - B0
    quality = bytes_n <= GATE["quality_bytes_frac"] * bytes_b and mn <= GATE["quality_err_frac"] * mb and gain["lo"] > 0
    compression = (bytes_n <= GATE["compression_bytes_frac"] * bytes_b and mn <= GATE["compression_err_frac"] * mb
                   and loss["hi"] < GATE["compression_ci_hi_frac"] * mb)
    material = e_b >= GATE["guard_negligible_frac"] * mb
    worse = material & (e_n > GATE["guard_frac"] * e_b)
    guard = not bool(worse.any())
    absolute = bytes_n <= ABS_BUDGET
    ok = bool((quality or compression) and guard and absolute)
    return {"macro_p95_N0": mn, "macro_p95_B0": mb, "ratio": mn / mb, "bytes_N0": bytes_n, "bytes_B0": bytes_b,
            "bytes_ratio": bytes_n / bytes_b, "B0_minus_N0": gain, "N0_minus_B0": loss, "quality_route": bool(quality),
            "compression_route": bool(compression), "guard": guard, "guard_material": material.tolist(),
            "guard_violations": worse.tolist(), "absolute_budget": bool(absolute), "pass": ok,
            "route": ("quality" if quality else "compression") if ok else None, "rules": GATE}
