# exps/nlcad/lhrec_analyze.py
"""NLCAD-03 Phase 1 analysis of LH-REC records (CPU; preregistered in PREREG_03_LHREC.json). With a = 1 the credit of
block b is the dot product <g_b, theta_end - anchor>_b.

  python -m exps.nlcad.lhrec_analyze
"""
from __future__ import annotations

import itertools

import numpy as np

from exps.dinomaly import common as D
from exps.nlcad import lhrec as L
from exps.nlcad.controller import BLOCKS
from exps.rh01b import common as C

WARMUP = 100                                             # recipe warm-up; Phase 2 skips periods starting before it
PHASES = ((0, 100), (100, 300), (300, 500), (500, 667))


def vec(credit):
    return np.array([credit["dot"][b] for b in BLOCKS])


def ccos(u, v):
    u, v = u - u.mean(), v - v.mean()
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300))


def periods(records, span=32):
    """Per span-period ending at a recorded step: the TBPTT-8 sum over its 8-step windows and the period-end record."""
    by_iter = {r["iter"]: r for r in records}
    out = []
    for end in range(span - 1, max(by_iter) + 1, span):
        wins = [by_iter.get(end - 8 * j) for j in range(span // 8)]
        if all(w is not None for w in wins):
            out.append({"start": end + 1 - span, "end_rec": by_iter[end], "tbptt": sum(vec(w["q"]["K8"]) for w in wins)})
    return out


def stage_analysis(records):
    P = periods(records)
    res = {"periods": len(P)}
    for tag, sel in (("all", P), ("post_warmup", [p for p in P if p["start"] >= WARMUP])):
        tb, k32 = [p["tbptt"] for p in sel], [vec(p["end_rec"]["q"]["K32"]) for p in sel]
        one = [vec(p["end_rec"]["q"]["onestep"]) for p in sel]
        pre = [vec(p["end_rec"]["q"]["onestep_pre"]) for p in sel]
        r = {"n": len(sel),
             "cos_tbptt_K32": [ccos(a, b) for a, b in zip(tb, k32)],
             "cos_tbptt_onestep": [ccos(a, b) for a, b in zip(tb, one)],
             "cos_K32_onestep": [ccos(a, b) for a, b in zip(k32, one)],
             "global_sign_tbptt_vs_onestep": [bool(np.sign(a.sum()) == np.sign(b.sum())) for a, b in zip(tb, one)],
             "global_tbptt": [float(a.sum()) for a in tb], "global_onestep": [float(b.sum()) for b in one],
             "global_onestep_first_order": [float(b.sum()) for b in pre],
             "consecutive_cos_tbptt": [ccos(a, b) for a, b in zip(tb[:-1], tb[1:])],
             "tbptt_dirs": [((a - a.mean()) / (np.linalg.norm(a - a.mean()) + 1e-300)).tolist() for a in tb],
             "onestep_dirs": [((b - b.mean()) / (np.linalg.norm(b - b.mean()) + 1e-300)).tolist() for b in one],
             "abs_tbptt": np.abs(np.array(tb)).ravel().tolist(), "abs_onestep": np.abs(np.array(one)).ravel().tolist()}
        post, first = np.array(r["global_onestep"]), np.array(r["global_onestep_first_order"])
        curv = post - first
        r["curvature_share"] = (np.abs(curv) / (np.abs(first) + np.abs(curv) + 1e-300)).tolist()
        for span in (64, 128):
            Q = [p for p in periods(records, span) if tag == "all" or p["start"] >= WARMUP]
            r[f"cos_tbptt_K{span}"] = [ccos(p["tbptt"], vec(p["end_rec"]["q"][f"K{span}"])) for p in Q]
        res[tag] = r
    held = [x for x in records if "held" in x]
    res["objective"] = {k: {"cos_q_held": [ccos(vec(x["q"][k]), vec(x["held"][k])) for x in held if k in x["held"]],
                            "global_sign_agree": [bool(np.sign(vec(x["q"][k]).sum()) == np.sign(vec(x["held"][k]).sum()))
                                                  for x in held if k in x["held"]],
                            "held_global_positive": [bool(vec(x["held"][k]).sum() > 0) for x in held if k in x["held"]]}
                        for k in ("K8", "K32", "K64", "K128", "onestep")}
    phases = {}
    for lo, hi in PHASES:
        sel = [x for x in held if lo <= x["iter"] < hi]
        post = np.array([vec(x["q"]["onestep"]).sum() for x in sel])
        first = np.array([vec(x["q"]["onestep_pre"]).sum() for x in sel])
        phases[f"{lo}-{hi}"] = {"n": len(sel), "lr_mean": float(np.mean([x["lr"] for x in sel])) if sel else None,
                                "onestep_global_mean": float(post.mean()) if sel else None,
                                "first_order_mean": float(first.mean()) if sel else None,
                                "curvature_mean": float((post - first).mean()) if sel else None,
                                "first_order_positive_frac": float((first > 0).mean()) if sel else None}
    res["phases"] = phases
    return res


def cross_stage(per_stage, key, tag):
    means = []
    for s in per_stage.values():
        d = np.array(s[tag][key])
        m = d.mean(0)
        means.append(m / (np.linalg.norm(m) + 1e-300))
    M = np.array(means)
    C_ = M @ M.T
    iu = np.triu_indices(len(M), 1)
    return {"mean": float(C_[iu].mean()), "min": float(C_[iu].min()), "matrix": C_.tolist()}


def main():
    pr = L.verify_prereg()
    per_stage, overhead = {}, {}
    for s in range(L.START + 1, len(C.STREAM) + 1):
        st = D.read(L.RUN / f"STAGE{s}.json")
        per_stage[s] = stage_analysis(D.read(L.RUN / f"RECORDS{s}.json"))
        ref = D.read(C.OUT / f"runs/R15_seed2712/STAGE{s}.json")
        overhead[s] = {"seconds": st["seconds"], "R15_2712_seconds": ref["seconds"], "ratio": st["seconds"] / ref["seconds"],
                       "bitwise_equal_to_R15": st["bitwise_equal_to_R15"], "anchor_peak_live": st["anchor_peak_live"],
                       "peak_allocated_mib": st["peak_allocated_mib"]}
    pool = lambda tag, k: [v for s in per_stage.values() for v in s[tag][k]]
    med = lambda xs: float(np.median(xs)) if len(xs) else None
    summary = {}
    for tag in ("post_warmup", "all"):
        summary[tag] = {k: med(pool(tag, k)) for k in ("cos_tbptt_K32", "cos_tbptt_onestep", "cos_K32_onestep",
                                                         "cos_tbptt_K64", "cos_tbptt_K128", "consecutive_cos_tbptt",
                                                         "curvature_share")}
        summary[tag]["global_sign_tbptt_vs_onestep"] = float(np.mean(pool(tag, "global_sign_tbptt_vs_onestep")))
        summary[tag]["onestep_global_positive"] = float(np.mean(np.array(pool(tag, "global_onestep")) > 0))
        summary[tag]["tbptt_global_positive"] = float(np.mean(np.array(pool(tag, "global_tbptt")) > 0))
        summary[tag]["first_order_global_positive"] = float(np.mean(np.array(pool(tag, "global_onestep_first_order")) > 0))
        summary[tag]["cross_stage_tbptt"] = cross_stage(per_stage, "tbptt_dirs", tag)
        summary[tag]["cross_stage_onestep"] = cross_stage(per_stage, "onestep_dirs", tag)
    summary["objective"] = {k: {"cos_q_held_median": med([v for s in per_stage.values() for v in s["objective"][k]["cos_q_held"]]),
                                "global_sign_agree": float(np.mean([v for s in per_stage.values() for v in s["objective"][k]["global_sign_agree"]])),
                                "held_global_positive": float(np.mean([v for s in per_stage.values() for v in s["objective"][k]["held_global_positive"]]))}
                            for k in ("K8", "K32", "K64", "K128", "onestep")}
    pw = summary["post_warmup"]
    go = {"stable": pw["consecutive_cos_tbptt"] > 0, "differs_a": pw["cos_tbptt_onestep"] < 0.5,
          "differs_b": pw["curvature_share"] > 0.5,
          "differs_d": pw["cross_stage_tbptt"]["mean"] < pw["cross_stage_onestep"]["mean"] - 0.1}
    go["go"] = bool(go["stable"] and (go["differs_a"] or go["differs_b"] or go["differs_d"]))
    lam = 1e-5 * float(np.median(pool("post_warmup", "abs_tbptt")) / np.median(pool("post_warmup", "abs_onestep")))
    res = {"prereg_sha256": D.sha(L.PREREG), "summary": summary, "go_no_go": go, "lambda_LH": lam,
           "overhead": overhead, "per_stage": per_stage, "rules": pr["go_no_go"]}
    D.write_json(L.OUT / "LHREC_ANALYSIS.json", res)
    for tag in ("post_warmup", "all"):
        print(tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary[tag].items()
                    if not k.startswith("cross")},
              "cross tbptt/onestep", round(summary[tag]["cross_stage_tbptt"]["mean"], 3),
              round(summary[tag]["cross_stage_onestep"]["mean"], 3))
    print("objective", summary["objective"])
    print("go", go, "lambda_LH", f"{lam:.3e}", "overhead", {s: round(v["ratio"], 2) for s, v in overhead.items()})


if __name__ == "__main__":
    main()
