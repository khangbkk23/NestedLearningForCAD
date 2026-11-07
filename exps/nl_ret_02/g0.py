"""NL-RET-02 G0 (CPU): independent audit + NL-RET-01A baseline reproduction. Read-only on every historical artifact.

  python -m exps.nl_ret_02.g0 audit        -> BOUNDARY_COVERAGE.json, CAUSAL_TIMING_AUDIT.json (audit part)
  python -m exps.nl_ret_02.g0 reproduce    -> BASELINE_REPRODUCTION.json
"""
from __future__ import annotations

import argparse
import subprocess

import numpy as np
import pandas as pd

from exps.cadfinal import common as K
from exps.dinomaly import common as D
from exps.dinomaly.common import require, sha
from exps.nl_ret_02 import common as C

TOL = 1e-9
AGENTS_PIN = "b88393815586295209c2704a6e16262730f2aba4000a6904b7c5c6d2b65ed622"
SCORER_SOURCES = ["exps/dinomaly/official.py", "exps/dinomaly/readout.py", "exps/dinomaly/run.py",
                  "exps/dinomaly/encoder.py", "exps/cadfinal/dino.py", "exps/rh01/common.py", "exps/rh01/run.py",
                  "exps/rh01b/run.py", "exps/nl_align_01/g1.py", "exps/nl_align_02/m3.py"]


def git(*a):
    return subprocess.run(["git", *a], cwd=C.ROOT, capture_output=True, text=True).stdout.strip()


def audit():
    C.OUT.mkdir(parents=True, exist_ok=True)
    proto, n = C.proto_and_n()
    out = {"branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
           "git_status_short": git("status", "--short").splitlines(),
           "agents_md_sha256": sha(C.ROOT / "AGENTS.md"), "agents_md_pin_ok": sha(C.ROOT / "AGENTS.md") == AGENTS_PIN}
    # 1. checkpoints (local ext4 copies vs historical run records) for every observation state used
    ck, bad = [], []
    for seed in C.SEEDS:
        for arm in ("ID", "FS_B"):
            for s in C.boundaries(arm):
                p, h, prov, rec = C.checkpoint(arm, seed, s)
                ok = sha(C.local_ckpt(arm, seed, s)) == h
                ck.append({"arm": arm, "seed": seed, "s": s, "owner": C.owner(arm, s), "provenance": prov,
                           "checkpoint": str(p.relative_to(C.ROOT)), "sha256": h, "local_copy_ok": ok,
                           "record_name": rec.get("name"), "order_ok": rec.get("name") == C.STREAM[s - 1]})
                if not ok or rec.get("name") != C.STREAM[s - 1]:
                    bad.append((arm, seed, s))
    # 2. task order of the incoming stage records
    order = []
    for seed in C.SEEDS:
        for t in range(2, 16):
            r = D.read(C.ROOT / f"results/hope_cad/rh01b/runs/R15_seed{seed}/STAGE{t}.json")
            order.append(r["name"] == C.STREAM[t - 1])
        for t in range(7, 16):
            r = D.read(C.checkpoint("FS_B", seed, t)[0].parent / f"STAGE{t}.json")
            order.append(r["name"] == C.STREAM[t - 1])
    # 3. replay membership at every boundary vs the R15 record of the incoming stage (row order included)
    mism = []
    for seed in C.SEEDS:
        for s in range(1, 15):
            rec = D.read(C.ROOT / f"results/hope_cad/rh01b/runs/R15_seed{seed}/STAGE{s + 1}.json")["replay"]["identities"]
            want = [proto["train"][d][i]["relative_path"] for d, i in C.replay_rows(seed, n, s)]
            if rec != want:
                mism.append((seed, s))
    # 4. encoder feature caches + scorer sources
    from exps.rh01 import common as RC
    cache = D.read(RC.OUT / "CACHE.json")
    enc = {c: {"sha256": cache[c]["sha256"], "ok": sha(RC.CACHE / f"train_{c}.pt") == cache[c]["sha256"]}
           for c in C.STREAM}
    src = {s: sha(C.ROOT / s) for s in SCORER_SOURCES}
    # 5. NL-RET-01A custody: its locked sources and frozen predictions unchanged
    al = K.read(C.OUT01 / "ANALYSIS_LOCK.json")
    fr = K.read(C.OUT01 / "G4_FREEZE.json")
    cust = {"analysis_lock_sources_unchanged": {s: sha(C.ROOT / s) == h for s, h in al["sources"].items()},
            "frozen_predictions_unchanged": sha(C.OUT01 / "FROZEN_RISK_PREDICTIONS.csv") == fr["predictions_sha256"],
            "feature_table_sha256": sha(C.OUT01 / "FEATURE_TABLE.csv"),
            "outcome_table_sha256": sha(C.OUT01 / "OUTCOME_TABLE.csv"),
            "eval_only_nonmembers_disjoint": all(not set(C.nonmembers(sd, n, c)) & set(C.members(sd, n, s).get(c, []))
                                                 for sd in C.SEEDS for c in C.STREAM for s in range(1, 16))}
    cov = {"rows": {a: len(C.rows(a)) for a in ("ID", "FS_B")},
           "rows_ID_per_seed": len(C.rows("ID")), "ID_boundaries": C.boundaries("ID"),
           "FS_B_boundaries": C.boundaries("FS_B"),
           "FS_B_state_distinct_rows_per_seed": sum(r["state_distinct_from_ID"] for r in C.rows("FS_B")),
           "FS_B_outcome_distinct_rows_per_seed": sum(r["outcome_distinct_from_ID"] for r in C.rows("FS_B")),
           "shared_history_note": "FS_B stages <= 6 ARE the R15 states. FS_B boundary 6->7 observes the shared R15 "
                                  "theta_6 (features identical to the ID row) but a distinct outcome; FS_B rows with "
                                  "s <= 5 are the ID rows and are not used. Primary FS_B replication uses s >= 7 "
                                  "(independently trained theta_s); s = 6 is reported separately.",
           "per_category_rows_per_seed": {c: sum(r["category"] == c for r in C.rows("ID")) for c in C.STREAM},
           "occupancy_low_n": "see NL-RET-01A REPLAY_OCCUPANCY.csv (re-derived here in the probe manifest)",
           "checkpoints": ck, "bad_checkpoints": bad, "task_order_ok": all(order),
           "replay_record_mismatches": mism, "encoder_feature_caches": enc,
           "encoder_caches_ok": all(v["ok"] for v in enc.values()), "scorer_and_training_sources": src,
           "fsb_replay_note": "FS_B stage records do not store replay identities; FS_B calls the identical "
                              "buffer_index(seed, n, STREAM[:s]) (NL-RET-01A audit, g1.py train) -> same members",
           "nl_ret_01a_custody": cust, **out, "utc": K.now()}
    K.write_json(C.OUT / "BOUNDARY_COVERAGE.json", cov)
    ok = (not bad and all(order) and not mism and cov["encoder_caches_ok"] and out["agents_md_pin_ok"]
          and all(cust["analysis_lock_sources_unchanged"].values()) and cust["frozen_predictions_unchanged"]
          and cust["eval_only_nonmembers_disjoint"])
    print("audit ok" if ok else "AUDIT FAILED", len(ck), "states", bad, mism)
    require(ok, "integrity")


def reproduce():
    """Recompute NL-RET-01A's primary comparison from its archived tables with its archived code, compare to its
    archived JSON, and re-derive its registered decision."""
    from exps.nl_ret_01a import validate as V
    V.RNG = np.random.default_rng(20261106)
    Dd, Dfs = V.load("ID"), V.load("FS_B")
    S = V.specs()
    cc = K.read(C.OUT01 / "CROSS_CLASS_VALIDATION.json")
    cs = K.read(C.OUT01 / "CROSS_SEED_VALIDATION.json")
    tv = K.read(C.OUT01 / "TEMPORAL_VALIDATION.json")
    arch = {"LOCO": cc["Y1_primary"], "LOSO": cs["Y1_primary"], "temporal": tv["Y1_primary"]}
    rep, worst = {}, 0.0
    det = [k for k, v in S.items() if v[0] != "shuffle"]           # B7 uses an RNG stream: excluded from equality
    for scheme in ("LOCO", "LOSO", "temporal"):
        rep[scheme] = {"archived_best_lawful": arch[scheme]["best_lawful_baseline"], "models": {}}
        names = det if scheme == "LOCO" else V.LAWFUL_BASE + ["controls"]
        for k in names:
            p, thr = V.oof(Dd, S[k], "Y1_primary", scheme)
            m = V.metrics(Dd, p, thr, "Y1_primary")
            a = arch[scheme]["models"][k]
            d = max(abs(m["MAE"] - a["MAE"]), abs(m["spearman"] - a["spearman"]))
            worst = max(worst, d)
            rep[scheme]["models"][k] = {"MAE": m["MAE"], "archived_MAE": a["MAE"], "spearman": m["spearman"],
                                        "archived_spearman": a["spearman"], "abs_diff": d,
                                        "per_seed_MAE": m["per_seed_MAE"]}
        rep[scheme]["recomputed_best_lawful"] = min(V.LAWFUL_BASE, key=lambda k: rep[scheme]["models"][k]["MAE"])
    # FS_B replication of B5 (ID-trained LOCO folds predicting FS_B distinct rows)
    Dall = pd.concat([Dd.assign(_src="ID"), Dfs.assign(_src="FS")], ignore_index=True)
    pred = pd.Series(np.nan, index=Dall.index)
    for c in C.STREAM:
        tr = Dall.index[(Dall._src == "ID") & (Dall.category != c)]
        te = Dall.index[(Dall._src == "FS") & (Dall.category == c)]
        if len(te):
            pred.loc[te] = V.fit_predict(Dall, tr, te, ("sched",), "Y1_primary")
    fs = Dall._src == "FS"
    mae = float((pred[fs] - Dall.loc[fs, "Y1_primary"]).abs().mean())
    a = cc["fsb_replication"]["Y1_primary"]["models"]["B5_fixed_schedule"]["MAE"]
    worst = max(worst, abs(mae - a))
    rep["FS_B_replication_B5"] = {"MAE": mae, "archived_MAE": a}
    # age schedule itself (training means per age bucket, all ID rows) for the report
    rep["age_schedule_full_data"] = Dd.groupby("AGE_BUCKET")["Y1_primary"].agg(["mean", "count"]).reset_index().to_dict("records")
    # registered decision, re-derived from the archived gate
    fv = K.read(C.OUT01 / "FINAL_VERDICT.json")
    g = fv["gate_Y1"]
    full = [f for f, v in g.items() if v["PASS"] and v["temporal_ok"] and v["fsb_replication_ok"]]
    cond = [f for f, v in g.items() if v["ii_ci_vs_best_lt0"] and v["iii_ci_vs_controls_lt0"] and not v["PASS"]
            and v["v_categories_beating"] >= 4]
    rederived = "MULTICLASS_or_UNALIGNED" if full else ("CLASS_CONDITIONAL_SIGNAL_ONLY" if cond
                                                        else "NO_INCREMENTAL_PREDICTIVE_SIGNAL")
    # independent recomputation of the gate conditions (ii)/(iii) is stochastic (bootstrap); recheck (i) exactly
    bl = rep["LOCO"]["recomputed_best_lawful"]
    i_cond = {f: rep["LOCO"]["models"][f"controls+{f}"]["MAE"] <= 0.95 * rep["LOCO"]["models"][bl]["MAE"]
              for f in V.FAM}
    res = {"tolerance": TOL, "worst_abs_diff": worst, "pass": worst <= TOL and rederived == fv["label"]
           and all(rep[s]["recomputed_best_lawful"] == rep[s]["archived_best_lawful"] for s in ("LOCO", "LOSO", "temporal")),
           "archived_verdict": fv["label"], "rederived_verdict": rederived,
           "condition_i_recomputed_any_family": any(i_cond.values()), "condition_i_by_family": i_cond,
           "schemes": rep, "note": "B7 (row-shuffled negative control) consumes a module RNG stream and is not part "
                                   "of the equality check", "utc": K.now()}
    V.W(C.OUT / "BASELINE_REPRODUCTION.json", res)
    print("reproduction", "PASS" if res["pass"] else "FAIL", worst, rederived,
          {s: (rep[s]["recomputed_best_lawful"], round(rep[s]["models"][rep[s]["recomputed_best_lawful"]]["MAE"], 5))
           for s in ("LOCO", "LOSO", "temporal")}, round(mae, 5))
    require(res["pass"], "baseline reproduction")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit", "reproduce"])
    globals()[ap.parse_args().cmd]()
