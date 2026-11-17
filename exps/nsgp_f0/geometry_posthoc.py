"""NSGP-F0 secondary analysis S-e, POST HOC replacement of the locked `run.py geometry` command.

    python -m exps.nsgp_f0.geometry_posthoc

Why this file exists: the locked command aborted on its own consistency check `d_H <= paired max + 1e-9` (C0_C,
bottle). The paired maximum was computed as `(x_hat - x).double().norm()`: the subtraction ran in FP32 and lost
~1.4e-8, while d_H was computed in FP64 direct difference by exact.nn. The theorem itself was not violated (max
score error 7.58 <= d_H 12.96). NSGP-F0 admits no amendment to locked sources, so this new, unlocked,
reporting-only file recomputes S-e with every difference in FP64. It reads only the locked fit states, the locked
EVAL distances and the bank; it changes no gate input and is written after the EVAL probes were read.
Mechanical failure record: results/hope_cad/nsgp_f0/MECHANICAL_FAILURES.json.
"""
from __future__ import annotations

import numpy as np
import torch

from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a import exact as X
from exps.nsgp_f0 import common as S
from exps.nsgp_f0 import run as RN
from exps.task_p.p0_normal_support import require


def main():
    S.verify_lock()
    target = S.OUT / "GEOMETRY_FINAL_POSTHOC.json"
    require(not target.exists(), "one-shot")
    co, n = S.setup()
    F = C.Features()
    t = S.FINAL
    rows, _ = RN.load_rows(t, co, n, F)
    cats = list(rows)
    fd = S.OUT / "fit" / f"t{t:02d}"
    supp = {"B0": {c: rows[c]["b0"] for c in cats}}
    for arm in RN.ARMS_FIT:
        supp[arm] = S.DECODERS[arm[:2]](torch.load(fd / f"{arm}.pt", weights_only=False), cats)
    out = {}
    for c in cats:
        x = rows[c]["x"]
        ids = torch.arange(len(x), dtype=torch.int64)
        rar = X.nn(x, X.RowSet(x, ids), exclude=ids)[0]
        thr = torch.quantile(rar, 0.9)
        rare = rar >= thr
        ev = np.load(S.OUT / "eval" / f"t{t:02d}" / f"{c}.npz")
        d0 = ev["EVAL|O0|d"]
        out[c] = {"rare_threshold_nn_dist": float(thr), "n_rows": len(x)}
        for arm, xs in supp.items():
            pe = (xs[c].double() - x.double()).norm(dim=1)                 # FP64 difference of FP32 rows: exact
            h, h_ab, h_ba = S.hausdorff(x, xs[c])
            e = np.abs(ev[f"EVAL|{arm}|d"] - d0)
            out[c][arm] = {"paired_max": float(pe.max()), "paired_mean": float(pe.mean()), "hausdorff": h,
                           "hausdorff_O0_to_arm": h_ab, "hausdorff_arm_to_O0": h_ba,
                           "max_abs_score_err_eval": float(e.max()),
                           "score_err_le_hausdorff": bool(e.max() <= h + 1e-9),
                           "hausdorff_le_paired_max": bool(h <= float(pe.max()) + 1e-9),
                           "recon_rare_mean": float(pe[rare].mean()), "recon_common_mean": float(pe[~rare].mean()),
                           "recon_rare_p95": float(torch.quantile(pe[rare], 0.95)),
                           "recon_common_p95": float(torch.quantile(pe[~rare], 0.95))}
            require(out[c][arm]["score_err_le_hausdorff"] and out[c][arm]["hausdorff_le_paired_max"],
                    f"bound violated {arm} {c}: {out[c][arm]}")
        print("geometry", c, {a: round(v["hausdorff"], 3) for a, v in out[c].items() if isinstance(v, dict)}, flush=True)
    S.write_json(target, {"t": t, "categories": out, "status": "post hoc reporting code (see module docstring)",
                          "peak_rss_gb": C.peak_rss_gb(), "utc": S.now()})


if __name__ == "__main__":
    main()
