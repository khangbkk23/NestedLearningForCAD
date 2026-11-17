"""NSGP-F0 POST HOC diagnostic (unlocked, reporting only, written after the EVAL probes were read): reconstruction
error of the stored rows that are the O0 nearest neighbours of EVAL probe patches, versus all stored rows, at the
final boundary. Explains why bank-average reconstruction (NMSE) and normal-score fidelity disagree. No gate input.

    python -m exps.nsgp_f0.nnrow_posthoc
"""
from __future__ import annotations

import numpy as np
import torch

from exps.nl_sgm_01a import common as C
from exps.nsgp_f0 import common as S
from exps.nsgp_f0 import run as RN


def main():
    S.verify_lock()
    target = S.OUT / "POSTHOC_NN_ROW_ERRORS.json"
    co, n = S.setup()
    F = C.Features()
    rows, _ = RN.load_rows(S.FINAL, co, n, F)
    cats = list(rows)
    fd = S.OUT / "fit" / f"t{S.FINAL:02d}"
    supp = {"B0": {c: rows[c]["b0"] for c in cats}}
    for arm in RN.ARMS_FIT:
        supp[arm] = S.DECODERS[arm[:2]](torch.load(fd / f"{arm}.pt", weights_only=False), cats)
    out = {}
    for c in cats:
        ev = np.load(S.OUT / "eval" / f"t{S.FINAL:02d}" / f"{c}.npz")
        pos = torch.searchsorted(rows[c]["ids"], torch.from_numpy(ev["EVAL|O0|id"]))
        used = torch.unique(pos)
        out[c] = {"n_rows": len(rows[c]["ids"]), "n_rows_used_as_nn": int(len(used))}
        for a, xs in supp.items():
            e = (xs[c].double() - rows[c]["x"].double()).norm(dim=1)
            out[c][a] = {"all_rows_mean": float(e.mean()), "nn_rows_query_weighted_mean": float(e[pos].mean()),
                         "nn_rows_unique_mean": float(e[used].mean())}
    macro = {a: {k: float(np.mean([out[c][a][k] for c in cats])) for k in out[cats[0]]["B0"]} for a in supp}
    S.write_json(target, {"status": "post hoc diagnostic, not preregistered, no gate input", "macro": macro,
                          "frac_rows_used_as_nn": float(np.mean([out[c]["n_rows_used_as_nn"] / out[c]["n_rows"] for c in cats])),
                          "categories": out, "utc": S.now()})
    print({a: {k: round(v, 3) for k, v in m.items()} for a, m in macro.items()})


if __name__ == "__main__":
    main()
