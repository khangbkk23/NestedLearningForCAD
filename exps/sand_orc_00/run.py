"""SAND-ORC-00: oracle headroom of the SAND fusion (retention, complementarity, calibration). CPU only.

    python -m exps.sand_orc_00.run lock
    python -m exps.sand_orc_00.run evaluate

Prereg: agents/plans/sand_orc_00_prereg.md. Reuses exps.sand_e2a_00 (inputs, calibrators, custody rules) and the
canonical SAND scorer unmodified. Oracle calibrators use test labels: non-deployable reference arms only.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from exps.sand_e2a_00 import common as E

OUT = E.ROOT / "results/hope_cad/sand_orc_00"
LOCK, RESULT = OUT / "LOCK.json", OUT / "RESULT.json"
PREREG = E.ROOT / "agents/plans/sand_orc_00_prereg.md"
TASK = E.ROOT / "agents/tasks/task_sand_orc_00.md"
SOURCES = ("exps/sand_orc_00/__init__.py", "exps/sand_orc_00/run.py", "exps/sand_orc_00/test_sand_orc.py",
           "exps/sand_e2a_00/common.py", "exps/sand_e2a_00/run.py", *E.CANONICAL,
           "agents/plans/sand_orc_00_prereg.md", "agents/tasks/task_sand_orc_00.md")
E2A_SHA = {"RESULT.json": "34462103e9cf73df072c5e91217d10b83249973bd726550e064c24968d25385f",
           "CALIB.json": "a9e8a122cb89c5a0600abb490b0a139cd6e6bb1a0493d0e3efbfb867b7eeddde",
           "LOCK.json": "6bee0362258adb70febb12673d515c66ad25c859d2e109d52fd44015d019cd37"}
BOOT = {"B": 10000, "seed": 20261117}
PRESENT = {"min_point": 0.010, "lower_bound_gt": 0.0}
CAPS = {"wall_h": 1.0, "rss_gb": 8.5}
TOL_PARITY = 1e-9
ARMS = ("PAR", "ORC_FIN", "ORC_RET", "INS_RET")
_LOG = []


# ------------------------------------------------------------------ custody (prereg section 3)

def decide(path):
    try:
        p = Path(os.fsdecode(path)).resolve()
    except Exception:
        return True
    if not E._governed(str(p)):
        return True
    if p == OUT.resolve() or E._under(p, OUT):
        return True
    return E.decide(p, "L")


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        if not decide(args[0]):
            _LOG.append({"denied": os.fsdecode(args[0])})
            raise PermissionError(f"SAND-ORC-00 custody: refused {os.fsdecode(args[0])}")
        p = os.fsdecode(args[0])
        try:
            governed = E._governed(str(Path(p).resolve()))
        except Exception:
            governed = False
        if governed:
            _LOG.append({"path": p})


def install_custody():
    sys.addaudithook(_audit)
    srcs = E.read_json(E.SAND / "LOCK.json")["sources"]
    E._STATE["sand_lock_sources"] = frozenset((E.ROOT / s).resolve() for s in srcs if s.startswith(("results/", "data/")))


# ------------------------------------------------------------------ lock

def write_lock():
    E.require(not LOCK.exists(), "LOCK.json exists (create-exclusive)")
    for n, h in E2A_SHA.items():
        E.require(E.sha_file(E.OUT / n) == h, f"SAND-E2A-00 artifact changed: {n}")
    lk = {"schema": "sand_orc_00_lock_v1", "utc": E.now(), "sources": {s: E.sha_file(E.ROOT / s) for s in SOURCES},
          "sand_e2a_00": E2A_SHA, "boot": BOOT, "present": PRESENT, "caps": CAPS, "arms": ARMS}
    print("LOCK", E.write_json_x(LOCK, lk))


def verify_lock():
    lk = E.read_json(LOCK)
    bad = [s for s, h in lk["sources"].items() if E.sha_file(E.ROOT / s) != h]
    E.require(not bad, f"LOCK: changed after lock: {bad}")
    for n, h in E2A_SHA.items():
        E.require(E.sha_file(E.OUT / n) == h, f"SAND-E2A-00 artifact changed: {n}")
    return lk


# ------------------------------------------------------------------ rules (prereg sections 1-2)

def oracle_cal(S, a, d, labels):
    """Calibrators from the test images labelled normal (non-deployable reference)."""
    normal = np.asarray(labels) == 0
    E.require(int(normal.sum()) >= 1, "no normal test image")
    return {k: float(v) for k, v in S.calibrate(a[normal], d[normal]).items() if k in E.CAL_KEYS}


def present(b):
    return bool(b["point"] >= PRESENT["min_point"] and b["lo"] > PRESENT["lower_bound_gt"])


def verdict(ret, comp):
    if not present(ret):
        return "NO_SF_HEADROOM"
    return "SF_HEADROOM_AND_COMPLEMENTARY" if present(comp) else "SF_HEADROOM_FUSION_REDUNDANT"


def analyse(per, boot):
    """per[c] = {arm or reference: {'pixel_ap', 'image_auroc', ...}} for every category in E.ORDER."""
    col = lambda a, m="pixel_ap": [per[c][a][m] for c in E.ORDER]  # noqa: E731
    names = ("ORC_FIN", "ORC_RET", "INS_RET", "PAR", "TH_P_acq", "TH_P_fin", "A2_fin", "E2A1_fin")
    macro = {a: {"pixel_ap": float(np.mean(col(a))), "image_auroc": float(np.mean(col(a, "image_auroc")))} for a in names}
    pairs = {"RET": ("ORC_RET", "ORC_FIN"), "COMP": ("ORC_RET", "TH_P_acq"), "CAL_FIN": ("ORC_FIN", "E2A1_fin"),
             "BIAS": ("ORC_RET", "INS_RET"), "TOTAL_OVER_E2A1": ("ORC_RET", "E2A1_fin"),
             "FRESH_DECODER_OVER_E2A1": ("TH_P_acq", "E2A1_fin"), "INS_RET_OVER_E2A1": ("INS_RET", "E2A1_fin")}
    comps = {}
    for k, (x, y) in pairs.items():
        b = boot(col(x), col(y))
        comps[k] = {**b, "present": present(b), "n_categories_positive": int(sum(p > q for p, q in zip(col(x), col(y)))),
                    "per_category": {c: per[c][x]["pixel_ap"] - per[c][y]["pixel_ap"] for c in E.ORDER}}
    comps["RET_image_auroc"] = boot(col("ORC_RET", "image_auroc"), col("ORC_FIN", "image_auroc"))
    comps["COMP_image_auroc"] = boot(col("ORC_RET", "image_auroc"), col("TH_P_acq", "image_auroc"))
    return {"verdict": verdict(comps["RET"], comps["COMP"]), "macro": macro, "comparisons": comps,
            "per_category": {c: {a: per[c][a] for a in names} for c in E.ORDER}}


# ------------------------------------------------------------------ run

def evaluate():
    t_start = time.time()
    install_custody()
    verify_lock()
    E.require(not RESULT.exists(), "RESULT.json exists (one-shot)")
    E.verify_inputs(E.read_json(E.LOCK))
    from exps.sand_e2a_00 import run as R
    torch = R._torch()
    from exps.nl_sgm_01a import am03_eval as A3
    from exps.sand import common as S
    S.verify_lock()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    logf = (OUT / "logs" / "evaluate.log").open("a")

    def log(*a):
        s = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    for n, h in E.SAND_PUBLISHED.items():
        E.require(E.sha_file(E.SAND / n) == h, f"SAND published artifact changed: {n}")
    cal = E.read_json(E.CALIB)["cal"]
    e2a = E.read_json(E.RESULT)["per_category_final"]
    sand = E.read_json(E.SAND / "PHASE0_RESULTS.json")
    th = sand["variants_non_gating"]["TH_theta_channel_under_P"]["per_category"]
    a2 = sand["summary"]["A2"]["per_category"]
    E.require(abs(np.mean([th[c]["final_pixel_ap"] for c in E.ORDER]) - E.PUBLISHED["TH_P"][0]) <= E.TOL["published"]
              and abs(np.mean([a2[c]["final_pixel_ap"] for c in E.ORDER]) - E.PUBLISHED["A2"][0]) <= E.TOL["published"],
              "published single-detector macros")
    man = E.read_json(E.MANIFEST)
    bnd = [E.read_json(E.MEM / f"boundary_t{t:02d}.json") for t in range(E.NT)]
    per, parity, n_metric = {}, 0.0, 0
    for k, c in enumerate(E.ORDER):
        t0 = time.time()
        labels, masks, items = A3.load_masks(c)
        hw = tuple(man["categories"][c]["images"][0]["hw"])
        E.require(all(tuple(x["hw"]) == hw for x in items), f"mixed native sizes {c}")
        paths = [x["relative_path"] for x in items]
        Gk, Zk = R._inputs(torch, k, c, bnd[k])
        Gf, Zf = R._inputs(torch, E.NT - 1, c, bnd[E.NT - 1])
        E.require(Gk["paths"] == paths and Gf["paths"] == paths, f"test identity order {c}")
        a_acq, a_fin = (G["grid"].numpy().reshape(-1, E.TOKENS) for G in (Gk, Gf))
        d_acq, d_fin = Zk["test"], Zf["test"]
        E.require(a_acq.shape == a_fin.shape == d_acq.shape == d_fin.shape == (len(labels), E.TOKENS), f"shapes {c}")
        frozen, d1_fin = cal[f"{k}|{c}"]["frozen"], cal[f"{E.NT - 1}|{c}"]["D1"]
        ins = {"mu_theta": frozen["mu_theta"], "sd_theta": frozen["sd_theta"], "mu_m": d1_fin["mu_m"], "sd_m": d1_fin["sd_m"]}
        jobs = {"PAR": (a_acq, d_acq, frozen), "ORC_FIN": (a_fin, d_fin, oracle_cal(S, a_fin, d_fin, labels)),
                "ORC_RET": (a_acq, d_fin, oracle_cal(S, a_acq, d_fin, labels)), "INS_RET": (a_acq, d_fin, ins)}
        out = {}
        for name in ARMS:
            a, d, cl = jobs[name]
            g, frac = S.fuse(a, d, cl)
            maps, img = S.readout_native(g, hw)
            out[name] = {**S.metrics(maps, img, labels, masks), "theta_is_min_fraction": frac, "calibrators": cl}
            del maps, g
            n_metric += 1
            if name == "PAR":                                  # parity before any oracle arm is trusted
                ref = E.read_json(E.CELLS / f"stage{k + 1:02d}_{c}.json")["fusion_P"]
                dv = max(abs(out["PAR"][m] - ref[m]) for m in ("pixel_ap", "image_auroc"))
                parity = max(parity, dv)
                E.require(dv <= TOL_PARITY, f"PARITY FAIL acquisition fusion {c}: {dv}")
        out["TH_P_acq"] = {"pixel_ap": th[c]["acq_pixel_ap"], "image_auroc": th[c]["acq_image_auroc"]}
        out["TH_P_fin"] = {"pixel_ap": th[c]["final_pixel_ap"], "image_auroc": th[c]["final_image_auroc"]}
        out["A2_fin"] = {"pixel_ap": a2[c]["final_pixel_ap"], "image_auroc": a2[c]["final_image_auroc"]}
        out["E2A1_fin"] = {"pixel_ap": e2a[c]["E2A1_mean"], "image_auroc": e2a[c]["E2A1_image_auroc_mean"]}
        per[c] = out
        del masks
        wall = time.time() - t_start
        E.require(wall <= CAPS["wall_h"] * 3600 and E.peak_rss_gb() <= CAPS["rss_gb"], "CAP")
        log(c, "ORC_FIN", round(out["ORC_FIN"]["pixel_ap"], 4), "ORC_RET", round(out["ORC_RET"]["pixel_ap"], 4),
            "INS_RET", round(out["INS_RET"]["pixel_ap"], 4), "TH_P_acq", round(out["TH_P_acq"]["pixel_ap"], 4),
            "E2A1", round(out["E2A1_fin"]["pixel_ap"], 4), f"{time.time() - t0:.0f}s", f"rss {E.peak_rss_gb():.2f}GB")
    boot = lambda x, y: A3.bootstrap_delta(x, y, seed=BOOT["seed"], B=BOOT["B"])  # noqa: E731
    res = analyse(per, boot)
    lab = [i for i, x in enumerate(_LOG) if "path" in x and ("/data/mvtec/" in x["path"]
                                                             or x["path"].endswith("DATASET_MANIFEST.json"))]
    res.update({"schema": "sand_orc_00_result_v1", "lock_sha256": E.sha_file(LOCK), "parity_max_abs_diff": parity,
                "resources": {"wall_s": time.time() - t_start, "rss_peak_gb": E.peak_rss_gb(),
                              "n_metric_computations": n_metric, "cuda_initialized": bool(torch.cuda.is_initialized())},
                "custody": {"n_entries": len(_LOG), "n_denied": sum("denied" in x for x in _LOG),
                            "n_label_file_opens": len(lab)},
                "note": "ORC_* arms use test-label calibrators: non-deployable upper references",
                "exposure": "MVTec AD official test, historically exposed; exploratory development cohort", "utc": E.now()})
    E.require(not res["resources"]["cuda_initialized"], "CUDA initialised")
    h = E.write_json_x(RESULT, res)
    log("RESULT", res["verdict"], {k: (round(v["point"], 4), round(v["lo"], 4), round(v["hi"], 4))
                                   for k, v in res["comparisons"].items()}, "sha", h[:16])


if __name__ == "__main__":
    {"lock": write_lock, "evaluate": evaluate}[sys.argv[1]]()
