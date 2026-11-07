"""NL-RET-02 G2 (CPU): signal tables from the frozen probe records, merged with the NL-RET-01A lawful features, and the
frozen delayed-outcome table (Y1-Y3 unchanged from NL-RET-01A; Y4 is NOT copied here, it is read only by `align`)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from exps.cadfinal import common as K
from exps.dinomaly.common import require, sha
from exps.nl_ret_02 import common as C
from exps.nl_ret_02 import prereg as R

KEY = ["arm", "seed", "category", "s"]
OUTCOLS = ["Y1_primary", "Y1_abs", "Y2", "Y3", "n_nonmembers", "n_eval_items", "nonmember_q90_s", "ORACLE_prev_Y1"]


def records():
    out = []
    for seed in C.SEEDS:
        for arm in ("ID", "FS_B"):
            for s in C.boundaries(arm):
                p = C.EXT / f"seed{seed}" / f"{arm}_s{s}.json"
                require(p.exists(), f"missing probe state {p}")
                r = json.loads(p.read_text())
                require(r["param_sha_before"] == r["param_sha_after"], "parameter change")
                out.append((r, sha(p)))
    return out


def main():
    require((C.OUT / "ANALYSIS_LOCK.json").exists(), "analysis lock first")
    require(not (C.OUT / "FEATURE_TABLE.csv").exists(), "one-shot")
    man = K.read(C.OUT / "PROBE_MANIFEST.json")
    rows, rec_sha = [], {}
    for r, h in records():
        rec_sha[f"{r['arm']}|{r['seed']}|{r['s']}"] = h
        require(r["probe"]["incoming_idx"] == man["probes"][str(r["seed"])][str(r["s"])]["incoming_idx"], "manifest")
        b = {k: v for k, v in r["boundary"].items() if k != "FSQ_per_draw"}
        for c, v in r["rows"].items():
            rows.append({"arm": r["arm"], "seed": r["seed"], "category": c, "s": r["s"], "incoming": r["probe"]["incoming"],
                         "state_owner": r["owner"], **b, **v})
    P = pd.DataFrame(rows)
    want = pd.DataFrame([{**x, "seed": sd} for sd in C.SEEDS for a in ("ID", "FS_B") for x in C.rows(a)])
    require(len(P) == len(want) == 3 * 195, f"coverage {len(P)} vs {len(want)}")
    gcols = KEY + ["incoming", "state_owner"] + R.GN + ["GNB_split_half_cos"] + [c for c in P if c == "FSQ_cos_aux_c"]
    ecols = KEY + ["incoming", "state_owner"] + R.GE
    fsq = [c for c in P if c.startswith("FSQ_") and c != "FSQ_cos_aux_c"]
    fcols = KEY + ["incoming", "state_owner", "n_members"] + R.FN + fsq
    P[gcols].to_csv(C.OUT / "INCOMING_GRADIENT_SIGNALS.csv", index=False)
    P[ecols].to_csv(C.OUT / "INCOMING_GEOMETRY_SIGNALS.csv", index=False)
    P[fcols].to_csv(C.OUT / "INCOMING_FUNCTION_SIGNALS.csv", index=False)
    # merge with NL-RET-01A lawful features (ID s = 1..14; FS_B s = 6..14, i.e. its non-shared rows)
    F1 = pd.read_csv(C.OUT01 / "FEATURE_TABLE.csv")
    F1 = F1[(F1.arm == "ID") | ((F1.arm == "FS_B") & (F1.s >= 6))].drop(columns=[c for c in F1 if c.startswith("ORACLE")])
    Fm = F1.merge(P.drop(columns=["FSQ_per_draw"], errors="ignore"), on=KEY, how="outer", validate="1:1", indicator=True)
    require((Fm._merge == "both").all(), "row sets differ between NL-RET-01A features and probes")
    require((Fm.n_members == Fm.D_n_members).all(), "occupancy disagreement")
    Fm = Fm.drop(columns=["_merge"]).sort_values(KEY).reset_index(drop=True)
    Fm.to_csv(C.OUT / "FEATURE_TABLE.csv", index=False)
    O1 = pd.read_csv(C.OUT01 / "OUTCOME_TABLE.csv")
    O = Fm[KEY].merge(O1[KEY + OUTCOLS], on=KEY, how="left", validate="1:1")
    require(O.loc[O.arm == "ID", "Y1_primary"].notna().all(), "missing primary outcome")
    O.to_csv(C.OUT / "OUTCOME_TABLE.csv", index=False)
    miss = {c: float(Fm[c].isna().mean()) for c in R.GN + R.GE + R.FN if Fm[c].isna().any()}
    K.write_json(C.OUT / "TABLES_SEAL.json", {
        "probe_records_sha256": rec_sha, "feature_table_sha256": sha(C.OUT / "FEATURE_TABLE.csv"),
        "outcome_table_sha256": sha(C.OUT / "OUTCOME_TABLE.csv"), "rows": {a: int((Fm.arm == a).sum()) for a in ("ID", "FS_B")},
        "new_feature_missing_fraction": miss, "y4_in_outcome_table": False,
        "source_outcomes": {"table": "results/hope_cad/nl_ret_01a/OUTCOME_TABLE.csv",
                            "sha256": sha(C.OUT01 / "OUTCOME_TABLE.csv")}, "utc": K.now()})
    print("tables", len(Fm), "rows; missing", miss)


if __name__ == "__main__":
    main()
