"""BENCH-AUD-00: UCAD-protocol evaluation (task-agnostic, final model, 224x224) and Image AUROC audit. CPU only.

    python -m exps.bench_aud_00.run lock
    python -m exps.bench_aud_00.run evaluate

Prereg: agents/plans/bench_aud_00_prereg.md. Re-scores existing SAND-FINAL / SAND-E2A-00 artifacts; no training.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

from exps.sand_e2a_00 import common as E

OUT = E.ROOT / "results/hope_cad/bench_aud_00"
LOCK, RESULT = OUT / "LOCK.json", OUT / "RESULT.json"
DRIFT = E.ROOT / "results/hope_cad/nl_drift_00"
STATS, STATS_LOCK = DRIFT / "STATS.pt", DRIFT / "STATS_LOCK.json"
CACHE_DIR = Path.home() / "hope_cad_cache/vitb8_q2"
CACHE_JSON = E.ROOT / "results/hope_cad/vitb8_q2/CACHE.json"
LIT = E.ROOT / "docs/benchmarks/cad_literature_2026-10-27"
SOURCES = ("exps/bench_aud_00/__init__.py", "exps/bench_aud_00/run.py", "exps/bench_aud_00/test_bench_aud.py",
           "exps/sand_e2a_00/common.py", "exps/sand_e2a_00/run.py", "exps/calm02/score.py", "exps/vitb8_q2/common.py",
           "exps/nl_sgm_01a/exact.py", "exps/dinomaly/official.py", *E.CANONICAL,
           "agents/plans/bench_aud_00_prereg.md", "agents/tasks/task_bench_aud_00.md",
           "docs/benchmarks/cad_literature_2026-10-27/perclass_mvtec_image_auroc.csv",
           "docs/benchmarks/cad_literature_2026-10-27/perclass_mvtec_pixel_aupr.csv")
E2A_SHA = {"RESULT.json": "34462103e9cf73df072c5e91217d10b83249973bd726550e064c24968d25385f",
           "CALIB.json": "a9e8a122cb89c5a0600abb490b0a139cd6e6bb1a0493d0e3efbfb867b7eeddde",
           "LOCK.json": "6bee0362258adb70febb12673d515c66ad25c859d2e109d52fd44015d019cd37"}
W = [f"E2A1_w{j}" for j in range(E.N_SEL)]
SIZE = 224
BOOT = {"B": 10000, "seed": 20261118}
CAPS = {"wall_h": 2.5, "rss_gb": 8.5}
TOL_PARITY = 1e-9
PUBLISHED_ROWS = ("ReplayCAD", "UCAD")
ALT = ("max", "top1pct_mean", "gauss_max", "top10cell_mean")
_LOG = []


# ------------------------------------------------------------------ custody (prereg section 5)

def decide(path):
    try:
        p = Path(os.fsdecode(path)).resolve()
    except Exception:
        return True
    if not E._governed(str(p)):
        return True
    if p == OUT.resolve() or E._under(p, OUT) or E._under(p, E.OUT):
        return True
    if p in {STATS.resolve(), STATS_LOCK.resolve(), CACHE_JSON.resolve()}:
        return True
    if p.parent == CACHE_DIR.resolve() and p.name.startswith("test_") and p.suffix == ".pt":
        return True
    if p.parent == E.MEM.resolve() and p.name.startswith("ckpt_t") and p.suffix == ".pt":
        return True
    return E.decide(p, "L")


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        if not decide(args[0]):
            _LOG.append({"denied": os.fsdecode(args[0])})
            raise PermissionError(f"BENCH-AUD-00 custody: refused {os.fsdecode(args[0])}")
        p = os.fsdecode(args[0])
        try:
            if E._governed(str(Path(p).resolve())):
                _LOG.append({"path": p})
        except Exception:
            pass


def install_custody():
    sys.addaudithook(_audit)
    srcs = E.read_json(E.SAND / "LOCK.json")["sources"]
    E._STATE["sand_lock_sources"] = frozenset((E.ROOT / s).resolve() for s in srcs if s.startswith(("results/", "data/")))


def write_lock():
    E.require(not LOCK.exists(), "LOCK.json exists (create-exclusive)")
    for n, h in E2A_SHA.items():
        E.require(E.sha_file(E.OUT / n) == h, f"SAND-E2A-00 artifact changed: {n}")
    lk = {"schema": "bench_aud_00_lock_v1", "utc": E.now(), "sources": {s: E.sha_file(E.ROOT / s) for s in SOURCES},
          "sand_e2a_00": E2A_SHA, "boot": BOOT, "caps": CAPS, "size": SIZE, "alt_scores": ALT}
    print("LOCK", E.write_json_x(LOCK, lk))


def verify_lock():
    lk = E.read_json(LOCK)
    bad = [s for s, h in lk["sources"].items() if E.sha_file(E.ROOT / s) != h]
    E.require(not bad, f"LOCK: changed after lock: {bad}")
    for n, h in E2A_SHA.items():
        E.require(E.sha_file(E.OUT / n) == h, f"SAND-E2A-00 artifact changed: {n}")
    return lk


# ------------------------------------------------------------------ rules

def route(mv, cents, seen):
    """mv [n,768] float64 mean tokens; cents {c: [768]}; seen: categories available. Nearest centroid (L2)."""
    C = np.stack([np.asarray(cents[c], dtype=np.float64) for c in seen])
    d = ((mv[:, None, :] - C[None]) ** 2).sum(-1)
    return [seen[i] for i in d.argmin(1)]


def maps224(grid):
    """[n,784] -> [n,224,224] FP32 bilinear (align_corners=False), the CADIC-harness adapter applied row-wise."""
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(np.ascontiguousarray(grid, dtype=np.float32)).reshape(-1, 1, 28, 28)
    return F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[:, 0].numpy()


def mask224(m):
    from PIL import Image
    if not m.any():
        return np.zeros((SIZE, SIZE), dtype=bool)
    return np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((SIZE, SIZE), Image.Resampling.NEAREST)) > 0


_G = []


def image_scores(maps, grid):
    """Alternative image-score rules (prereg section 4); 'max' is the primary rule."""
    import torch
    from exps.dinomaly import official as O
    if not _G:
        _G.append(O.get_gaussian_kernel(*O.RECIPE["gaussian"]))
    flat = maps.reshape(len(maps), -1)
    k = max(1, int(flat.shape[1] * 0.01))
    with torch.no_grad():
        sm = _G[0](torch.from_numpy(maps[:, None])).numpy().reshape(len(maps), -1)
    return {"max": flat.max(1), "top1pct_mean": -np.sort(-flat, axis=1)[:, :k].mean(1), "gauss_max": sm.max(1),
            "top10cell_mean": -np.sort(-np.asarray(grid, dtype=np.float64), axis=1)[:, :10].mean(1)}


def metrics224(grid, masks, labels, alt=False):
    from sklearn.metrics import roc_auc_score
    from exps.cadfinal import metrics as MX
    maps = maps224(grid)
    out = {"pixel_aupr": float(MX.exact_ap(list(maps), masks)["ap"])}
    sc = image_scores(maps, grid) if alt else {"max": maps.reshape(len(maps), -1).max(1)}
    for r, s in sc.items():
        out[f"i_auroc_{r}" if r != "max" else "i_auroc"] = float(roc_auc_score(labels, np.asarray(s, dtype=np.float64)))
    if alt:
        out["_image_max"] = sc["max"]
    return out


def size_terciles(labels, masks_native, scores):
    """I-AUROC of normals vs each within-category tercile of anomalous mask area."""
    from sklearn.metrics import roc_auc_score
    lab = np.asarray(labels)
    area = np.array([m.mean() for m in masks_native])
    an = np.where(lab == 1)[0]
    q = np.quantile(area[an], [1 / 3, 2 / 3])
    groups = [an[area[an] <= q[0]], an[(area[an] > q[0]) & (area[an] <= q[1])], an[area[an] > q[1]]]
    nor = np.where(lab == 0)[0]
    out = []
    for g in groups:
        idx = np.concatenate([nor, g])
        out.append(float(roc_auc_score(lab[idx], np.asarray(scores)[idx])) if len(g) else None)
    return {"small": out[0], "medium": out[1], "large": out[2], "area_thresholds": q.tolist()}


def published():
    out = {}
    for key, fn in (("i_auroc", "perclass_mvtec_image_auroc.csv"), ("pixel_aupr", "perclass_mvtec_pixel_aupr.csv")):
        with (LIT / fn).open() as f:
            for row in csv.DictReader(f):
                if row["Method"] in PUBLISHED_ROWS:
                    out.setdefault(row["Method"], {})[key] = {c: float(row[c]) for c in E.ORDER}
    return out


def band(b):
    return "AHEAD" if b["lo"] > 0 else ("BEHIND" if b["hi"] < 0 else "NOT_ESTABLISHED")


# ------------------------------------------------------------------ run

def evaluate():
    t_start = time.time()
    install_custody()
    verify_lock()
    E.require(not RESULT.exists(), "RESULT.json exists (one-shot)")
    from exps.sand_e2a_00 import run as R
    torch = R._torch()
    from exps.nl_sgm_01a import am03_eval as A3
    from exps.nl_sgm_01a import exact as X
    from exps.sand import common as S
    from exps.sand import score as SC
    from exps.vitb8_q2 import common as V
    E.verify_inputs(E.read_json(E.LOCK))
    S.verify_lock()
    logf = (OUT / "logs" / "evaluate.log").open("a")

    def log(*a):
        s = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    E.require(E.sha_file(STATS) == E.read_json(STATS_LOCK)["stats_sha256"], "NL-DRIFT-00 STATS changed")
    cents = {c: np.asarray(v, dtype=np.float64) for c, v in torch.load(STATS, weights_only=False)["centroid"].items()}
    cal = E.read_json(E.CALIB)["cal"]
    man = E.read_json(E.MANIFEST)
    bnd = [E.read_json(E.MEM / f"boundary_t{t:02d}.json") for t in range(E.NT)]
    pub = published()
    cells, audit, routing, parity, n_metric = {}, {}, {}, {"dec": 0.0, "fus_w0": 0.0}, 0
    sup_cache = {}
    for k, c in enumerate(E.ORDER):
        t0 = time.time()
        labels, masks, items = A3.load_masks(c)
        hw = tuple(man["categories"][c]["images"][0]["hw"])
        paths = [x["relative_path"] for x in items]
        m224 = [mask224(m) for m in masks]
        feats, fpaths = V.test_features(c)
        E.require(list(fpaths) == paths, f"feature identity order {c}")
        mv = np.stack([feats[i].double().mean(0).numpy() for i in range(len(paths))])
        for t in range(k, E.NT):
            seen = list(E.ORDER[:t + 1])
            rt = route(mv, cents, seen)
            routing[f"{t}|{c}"] = {"n": len(rt), "correct": int(sum(r == c for r in rt)),
                                   "wrong_to": sorted({r for r in rt if r != c})}
            G, Z = R._inputs(torch, t, c, bnd[t])
            E.require(G["paths"] == paths, f"grid identity order {t + 1} {c}")
            a = np.maximum(G["grid"].numpy().reshape(-1, E.TOKENS).astype(np.float64), 0.0)
            d = Z["test"].copy()
            for i, r in enumerate(rt):                        # mis-routed: exact distance to the routed support
                if r != c:
                    if (t, r) not in sup_cache:
                        sup_cache.clear()
                        sup_cache[(t, r)] = SC.supports(E.MEM, t, only=r)[r]
                    d[i] = X.nn(feats[i].float().contiguous(), sup_cache[(t, r)])[0].numpy()
            if t == E.NT - 1:                                  # parity (native, true category, canonical scorer)
                ref_th = E.read_json(E.CELLS / f"stage15_{c}.json")["theta_P"]
                mp, im = S.readout_native(a.astype(np.float32), hw)
                dv = S.metrics(mp, im, labels, masks)
                ref_f = E.read_json(E.OUT / "cells" / f"stage15_{c}.json")["arms"]["E2A1_w0"]
                g0, _ = S.fuse(G["grid"].numpy().reshape(-1, E.TOKENS), Z["test"], cal[f"{t}|{c}"]["E2A1_w0"])
                mp, im = S.readout_native(g0, hw)
                fv = S.metrics(mp, im, labels, masks)
                del mp
                for key, x, y in (("dec", dv, ref_th), ("fus_w0", fv, ref_f)):
                    e = max(abs(x[m] - y[m]) for m in ("pixel_ap", "image_auroc"))
                    parity[key] = max(parity[key], e)
                    E.require(e <= TOL_PARITY, f"PARITY FAIL {key} {c}: {e}")
                n_metric += 2
            final = t == E.NT - 1
            cell = {"DEC": metrics224(a, m224, labels, alt=final), "MEM": metrics224(d, m224, labels, alt=final)}
            fus = []
            for w in W:
                g = np.empty_like(a, dtype=np.float32)
                for r in sorted(set(rt)):
                    idx = [i for i, x in enumerate(rt) if x == r]
                    g[idx], _ = S.fuse(G["grid"].numpy().reshape(-1, E.TOKENS)[idx], d[idx], cal[f"{t}|{r}"][w])
                fus.append(metrics224(g, m224, labels, alt=final))
            n_metric += 2 + len(W)
            if final:
                audit[c] = {"alt_i_auroc": {arm: {r: (cell[arm]["i_auroc"] if r == "max" else cell[arm][f"i_auroc_{r}"])
                                                  for r in ALT} for arm in ("DEC", "MEM")},
                            "size_terciles": {"DEC": size_terciles(labels, masks, cell["DEC"]["_image_max"]),
                                              "FUS_w0": size_terciles(labels, masks, fus[0]["_image_max"])},
                            "score_separation": {arm: {"normal_median": float(np.median(s[labels == 0])),
                                                       "normal_q99": float(np.quantile(s[labels == 0], 0.99)),
                                                       "anomalous_median": float(np.median(s[labels == 1]))}
                                                 for arm, s in (("DEC", cell["DEC"]["_image_max"]),
                                                                ("FUS_w0", fus[0]["_image_max"]))}}
                audit[c]["alt_i_auroc"]["FUS"] = {r: float(np.mean([f["i_auroc" if r == "max" else f"i_auroc_{r}"]
                                                                    for f in fus])) for r in ALT}
            for x in (cell["DEC"], cell["MEM"], *fus):
                x.pop("_image_max", None)
            cell["FUS"] = {m: float(np.mean([f[m] for f in fus])) for m in ("pixel_aupr", "i_auroc")}
            cell["FUS_w0"] = {m: fus[0][m] for m in ("pixel_aupr", "i_auroc")}
            cells[f"{t}|{c}"] = {arm: {m: v[m] for m in ("pixel_aupr", "i_auroc")} for arm, v in cell.items()}
            wall = time.time() - t_start
            E.require(wall <= CAPS["wall_h"] * 3600 and E.peak_rss_gb() <= CAPS["rss_gb"], "CAP")
        del feats, masks, m224
        f = cells[f"{E.NT - 1}|{c}"]
        log(c, "final FUS", round(f["FUS"]["i_auroc"], 4), round(f["FUS"]["pixel_aupr"], 4), "DEC",
            round(f["DEC"]["i_auroc"], 4), round(f["DEC"]["pixel_aupr"], 4), "routing final",
            routing[f"{E.NT - 1}|{c}"]["correct"], "/", routing[f"{E.NT - 1}|{c}"]["n"], f"{time.time() - t0:.0f}s",
            f"rss {E.peak_rss_gb():.2f}GB")
    boot = lambda x, y: A3.bootstrap_delta(x, y, seed=BOOT["seed"], B=BOOT["B"])  # noqa: E731
    res = summarize(cells, routing, pub, boot)
    res.update({"schema": "bench_aud_00_result_v1", "audit": audit, "parity_max_abs_diff": parity,
                "lock_sha256": E.sha_file(LOCK),
                "resources": {"wall_s": time.time() - t_start, "rss_peak_gb": E.peak_rss_gb(), "n_metric": n_metric,
                              "cuda_initialized": bool(torch.cuda.is_initialized())},
                "custody": {"n_entries": len(_LOG), "n_denied": sum("denied" in x for x in _LOG)},
                "protocol": "UCAD protocol: MVTec 1x15 alphabetical, final model, task-agnostic (centroid routing), "
                            "224x224 full-image maps and masks, image score = map max, Chaudhry FM",
                "exposure": "MVTec AD official test, historically exposed; exploratory development cohort",
                "utc": E.now()})
    E.require(not res["resources"]["cuda_initialized"], "CUDA initialised")
    h = E.write_json_x(RESULT, res)
    log("RESULT sha", h[:16], {a: {m: round(v, 4) for m, v in x.items()} for a, x in res["final_macro"].items()})


def summarize(cells, routing, pub, boot):
    O = E.ORDER
    arms = ("DEC", "MEM", "FUS", "FUS_w0")
    fin = {a: {m: [cells[f"{E.NT - 1}|{c}"][a][m] for c in O] for m in ("pixel_aupr", "i_auroc")} for a in arms}
    macro = {a: {m: float(np.mean(v)) for m, v in fin[a].items()} for a in arms}
    fm = {}
    for a in arms:
        for m in ("pixel_aupr", "i_auroc"):
            val, per = E.forgetting({(t, c): cells[f"{t}|{c}"][a][m] for t, c in E.KEYS})
            fm.setdefault(a, {})[m] = val
    vs = {}
    for name, rows in pub.items():
        for m in ("i_auroc", "pixel_aupr"):
            if m in rows:
                b = boot(fin["FUS"][m], [rows[m][c] for c in O])
                vs[f"FUS - {name} ({m})"] = {**b, "band": band(b), "published_macro": float(np.mean(list(rows[m].values())))}
    acc = {t + 1: float(sum(routing[f"{t}|{c}"]["correct"] for c in O[:t + 1]) /
                        sum(routing[f"{t}|{c}"]["n"] for c in O[:t + 1])) for t in range(E.NT)}
    return {"final_macro": macro, "forgetting": fm, "vs_published": vs, "routing_accuracy_by_stage": acc,
            "routing": routing, "final_per_category": {a: {m: dict(zip(O, v)) for m, v in fin[a].items()} for a in arms},
            "published": pub, "cells": cells}


if __name__ == "__main__":
    {"lock": write_lock, "evaluate": evaluate}[sys.argv[1]]()
