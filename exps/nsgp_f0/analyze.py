"""NSGP-F0 locked analysis: gate, secondary analyses, tables, machine-readable verdict.

    python -m exps.nsgp_f0.analyze
"""
from __future__ import annotations

import csv

import numpy as np
import torch

from exps.cadfinal import metrics as MX
from exps.nl_sgm_01a import am03_eval as A3
from exps.nl_sgm_01a import common as C
from exps.nsgp_f0 import common as S
from exps.task_p.p0_normal_support import require

ARMS = ("B0", "C0_C", "C0_Q", "N0_C", "N0_Q", "P0")
DATA = S.ROOT / "data/mvtec"


def load(t):
    d = S.OUT / "eval" / f"t{t:02d}"
    require((d / "DONE.json").exists(), f"eval t={t} incomplete")
    return {c: dict(np.load(d / f"{c}.npz")) for c in S.ORDER[:t + 1]}


def image_errors(ev, c, paths, arm):
    """S-f: per EVAL image, |max native(arm) - max native(O0)| with the AM03 bilinear readout."""
    out = []
    d0 = torch.from_numpy(ev["EVAL|O0|d"]).reshape(-1, S.PATCHES)
    da = torch.from_numpy(ev[f"EVAL|{arm}|d"]).reshape(-1, S.PATCHES)
    for i, p in enumerate(paths):
        hw = MX.native_hw(DATA / p)
        m0 = A3.native_maps(d0[i:i + 1], hw).max()
        ma = A3.native_maps(da[i:i + 1], hw).max()
        out.append(abs(float(ma) - float(m0)))
    return out


def main():
    S.verify_lock()
    target = S.OUT / "VERDICT.json"
    require(not target.exists(), "one-shot")
    sel = S.read_json(S.OUT / "SELECTION.json")
    nstar = sel["selected"]
    cmatch = "C0_C" if nstar == "N0_C" else "C0_Q"
    co = C.cohorts()
    res = {"boundaries": {}, "selection": sel}
    rows_csv = []
    for t in S.BOUNDARIES:
        ev = load(t)
        fit = S.read_json(S.OUT / "fit" / f"t{t:02d}" / "FIT_SUMMARY.json")
        bt = {"bytes": {"B0": fit["bank_bytes_B0"], **{a: fit["decode"][a]["bytes"] for a in ARMS if a != "B0"},
                        "O0_fp32_rows_transient": int(fit["rows"] * (768 * 4 + 5) + 64)}, "cohorts": {}}
        for k in ("EVAL", "SEL", "HELD"):
            cats = [c for c in S.ORDER[:t + 1] if f"{k}|O0|d" in ev[c]]
            if not cats:
                continue
            per = {}
            for a in ARMS:
                per[a] = {}
                for c in cats:
                    d0 = ev[c][f"{k}|O0|d"]
                    st = S.err_stats(ev[c][f"{k}|{a}|d"], d0)
                    st["nn_id_agreement"] = float((ev[c][f"{k}|{a}|id"] == ev[c][f"{k}|O0|id"]).mean())
                    tail = d0 >= np.quantile(d0, 0.95)
                    st["tail_p95_abs"] = float(np.quantile(np.abs(ev[c][f"{k}|{a}|d"][tail] - d0[tail]), 0.95))
                    per[a][c] = st
                    rows_csv.append({"boundary": t + 1, "cohort": k, "arm": a, "category": c, **st})
            macro = {a: {m: float(np.mean([per[a][c][m] for c in cats])) for m in per[a][cats[0]]} for a in ARMS}
            bt["cohorts"][k] = {"categories": cats, "per_category": per, "macro": macro}
        if t == S.FINAL:
            lat = {a: np.concatenate([ev[c][f"latency|{a}"] for c in S.ORDER]) * 1000 for a in ("O0",) + ARMS}
            bt["latency_ms_per_eval_image"] = {a: {"p50": float(np.median(v)), "p95": float(np.quantile(v, 0.95)),
                                                   "n": int(len(v))} for a, v in lat.items()}
            bt["image_readout_abs_err"] = {a: {c: image_errors(ev[c], c, co[c]["eval"], a) for c in S.ORDER} for a in ARMS}
            bt["b0_vs_sand_eval_max_abs_diff"] = max(float(ev[c]["B0_vs_sand_eval_max_abs_diff"]) for c in S.ORDER
                                                     if "B0_vs_sand_eval_max_abs_diff" in ev[c])
        res["boundaries"][t + 1] = bt
    fin = res["boundaries"][S.FINAL + 1]
    E = fin["cohorts"]["EVAL"]["per_category"]
    e = {a: [E[a][c]["p95_abs"] for c in S.ORDER] for a in ARMS}
    g = S.gate(fin["bytes"][nstar], fin["bytes"]["B0"], e[nstar], e["B0"])
    res["gate"] = {"N0_star": nstar, **g, "categories": list(S.ORDER)}
    res["secondary"] = {
        "S_b_learning_contribution": {"comparison": f"{nstar} - {cmatch}", **S.boot(e[nstar], e[cmatch]),
                                      "macro": {nstar: float(np.mean(e[nstar])), cmatch: float(np.mean(e[cmatch]))}},
        "S_c_refit_effect": {"comparison": "P0 - B0", **S.boot(e["P0"], e["B0"])},
        "other_vs_B0": {a: {"comparison": f"{a} - B0", **S.boot(e[a], e["B0"])} for a in ARMS if a != "B0"},
        "fit_side": {t + 1: S.read_json(S.OUT / "fit" / f"t{t:02d}" / "FIT_SUMMARY.json")["fitq_macro_p95"]
                     for t in S.BOUNDARIES},
        "n0_training": {t + 1: {a: {k: v for k, v in S.read_json(S.OUT / "fit" / f"t{t:02d}" / f"{a}.json").items()
                                    if k not in ("curve", "categories")} for a in ("N0_C", "N0_Q")} for t in S.BOUNDARIES},
    }
    geo = S.OUT / "GEOMETRY_FINAL.json"
    res["secondary"]["S_e_geometry"] = S.read_json(geo) if geo.exists() else None
    verdict = "F0_FEASIBLE" if g["pass"] else "F0_NOT_SUPPORTED"
    S.write_json(S.OUT / "RESULTS.json", res)
    with (S.OUT / "PER_CATEGORY_SCORE_ERRORS.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)
    S.write_json(target, {"schema": "nsgp_f0_verdict_v1", "verdict": verdict, "route": g["route"], "N0_star": nstar,
                          "gate": {k: v for k, v in g.items() if k != "rules"}, "rules": S.GATE,
                          "macro_p95_eval_final": {a: float(np.mean(e[a])) for a in ARMS},
                          "bytes_final": fin["bytes"], "results_sha256": S.sha(S.OUT / "RESULTS.json"),
                          "lock_sha256": S.sha(S.LOCK), "scope": "static normal-score fidelity only; no anomaly "
                          "detection, continual learning or Nested Learning claim", "utc": S.now()})
    print(verdict, g["route"], {a: round(float(np.mean(e[a])), 4) for a in ARMS})


if __name__ == "__main__":
    main()
