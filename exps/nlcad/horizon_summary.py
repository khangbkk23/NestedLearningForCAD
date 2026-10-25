# exps/nlcad/horizon_summary.py
"""Summarise HORIZON_TOY_GRID_*.json: per (regime, horizon, objective) and proxy, across seeds."""
from __future__ import annotations

import argparse

import numpy as np

from exps.dinomaly import common as D

OUT = D.ROOT / "results/hope_cad/nlcad"
PROXIES = ("endpoint", "onestep", "online")


def summarise(rows):
    keys = sorted({(r["regime"], r["T"], r["objective"]) for r in rows}, key=lambda k: (k[0], k[2], k[1]))
    out = []
    for reg, T, obj in keys:
        rs = [r for r in rows if (r["regime"], r["T"], r["objective"]) == (reg, T, obj)]
        rec = {"regime": reg, "T": T, "objective": obj, "n": len(rs),
               "exact_global_positive": float(np.mean([r["m_endpoint"]["global_exact"] > 0 for r in rs]))}
        for p in PROXIES:
            m = [r[f"m_{p}"] for r in rs]
            rec[p] = {"cos_centred_median": float(np.median([x["cos_centred"] for x in m])),
                      "cos_centred_min": float(np.min([x["cos_centred"] for x in m])),
                      "sign_agree_centred_mean": float(np.mean([x["sign_agree_centred"] for x in m])),
                      "global_sign_agree": float(np.mean([x["global_sign_agree"] for x in m])),
                      "slope_median": float(np.median([x["slope"] for x in m]))}
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="GRID_ones")
    a = ap.parse_args()
    src = D.read(OUT / f"HORIZON_TOY_{a.tag}.json")
    table = summarise(src["rows"])
    D.write_json(OUT / f"HORIZON_TOY_{a.tag}_SUMMARY.json", table)
    print("regime     obj     T  exact_G>0 | " + " | ".join(f"{p}: cosC(med/min) signC  G_agree slope" for p in PROXIES))
    for r in table:
        print(f"{r['regime']:9s} {r['objective']:6s} {r['T']:3d}  {r['exact_global_positive']:.2f}     | " + " | ".join(
            f"{r[p]['cos_centred_median']:+.2f}/{r[p]['cos_centred_min']:+.2f} {r[p]['sign_agree_centred_mean']:.2f} "
            f"{r[p]['global_sign_agree']:.2f} {r[p]['slope_median']:+.2f}" for p in PROXIES))


if __name__ == "__main__":
    main()
