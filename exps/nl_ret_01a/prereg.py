"""NL-RET-01A Phase 1 preregistration: signal dictionary, targets, baselines, folds, thresholds, decision gate.
Written and SHA-locked BEFORE any feature/outcome table exists. `python -m exps.nl_ret_01a.prereg` writes
SIGNAL_DICTIONARY.json and AUDIT_LOCK.json (refuses to overwrite)."""
from __future__ import annotations

from exps.cadfinal import common as K
from exps.dinomaly.common import require, sha
from exps.nl_ret_01a import common as C

# z(c, r, arm, s): end of stage s, before any stage-(s+1) update. "members" = Replay-24 members of c held at the end
# of s (buffer replayed during s+1). Image score = canonical R_D top-1% mean; pixel stats on the R_D 256x256 map.
SIGNALS = {
    "A_replay_tail": {
        "A_img_mean": "mean image score over c's members at s",
        "A_pix_q99": "mean over c's members of per-image pixel q99 at s",
        "A_pix_q999": "mean over c's members of per-image pixel q99.9 at s",
        "A_img_max": "max image score over c's members at s",
        "A_tail_mass": "mean over members of the fraction of 64x64-pooled map cells above tau_c (tau_c = pooled q99 of "
                       "the pooled maps of c's members at its acquisition state)",
        "A_grow_acq": "log(mean per-image q99 at s / same images at acquisition state) [fixed images]",
        "A_grow_prev": "log(mean per-image q99 at s / same images at s-1) [fixed images; NaN at s = a(c)]",
        "A_cv": "coefficient of variation of member image scores (NaN if n_members < 2)",
        "A_global_grow_prev": "log ratio s/s-1 of mean image score over ALL members of all completed tasks (fixed images)",
        "A_cur_vs_rep": "log(mean member image score of c / mean image score of the current task's train normals) at s",
        "A_membership_effect": "A_pix_q99(members at s, scored at s) - A_pix_q99(members at s-1, scored at s)",
    },
    "C_synthetic_Q2": {
        "C_ap": "pooled patch AP (28x28 grid, positives coverage>=0.5, negatives coverage==0) over Q2 items hosted on "
                "c's members at s",
        "C_margin": "mean positive patch score - q99 negative patch score (same items)",
        "C_rankloss": "mean over items of mean softplus((neg - pos)/0.05) over all pos/neg pairs",
        "C_sat": "fraction of items with rank loss < 1e-5",
        "C_bg_q99": "q99 of negative patch scores",
        "C_ap_grow_acq": "C_ap(s) - C_ap(acquisition state) on the same member-hosted items",
    },
    "D_coverage": {
        "D_n_members": "number of c's members at s",
        "D_dispersion": "mean pairwise cosine distance of members' token-mean encoder features (NaN if n < 2)",
        "D_cur_dist": "cosine distance between c's member centroid and the current task's train centroid",
        "D_cur_nn": "mean over c's members of the nearest current-task train distance",
    },
    "E_function_drift": {
        "E_param_rel": "||theta_s - theta_{s-1}|| / ||theta_{s-1}||",
        "E_param_block_max": "max over decoder blocks of the relative block change",
        "E_member_img_drift": "mean |image score(s) - image score(s-1)| over c's members (fixed images)",
        "E_member_map_corr": "mean Pearson correlation of 64x64 maps s vs s-1 over c's members",
        "E_global_img_drift": "mean |image score(s) - image score(s-1)| over all members",
    },
    "F_optimization": {
        "F_loss_last": "logged mean training loss of the last 50 iterations of stage s",
        "F_loss_shock": "logged loss at iteration 0 / last logged loss of stage s",
        "F_g_member_norm": "fresh no-update gradient norm of global-cosine on c's members at theta_s",
        "F_cos_cur_member": "cosine(fresh gradient on 16 fixed task-s normals, fresh gradient on c's members)",
        "F_cos_rep_member": "cosine(gradient on all members, gradient on c's members)",
        "F_aux_eff": "FS_B only: mean effective injected aux norm ratio in stage s (0 for ID)",
        "F_clip_sat": "FS_B only: logged clip saturation in stage s (missing for ID: NOT_OBSERVABLE_FROM_ARCHIVE)",
    },
    "G_history": {
        "G_Agrow_prev_lag1": "A_grow_prev at s-1",
        "G_Agrow_ewma": "EWMA(alpha=0.5) of A_grow_prev over a(c)+1..s",
        "G_Agrow_maxexc": "max A_grow_prev over a(c)+1..s",
        "G_Cap_delta_prev": "C_ap(s) - C_ap(s-1)",
        "G_Edrift_ewma": "EWMA(alpha=0.5) of E_member_img_drift",
    },
    "CLOCK": {"K_s": "stage s", "K_age": "s - a(c)", "K_inv_age": "1/(1 + s - a(c))"},
}
NOT_OBSERVABLE = ["optimizer first/second moments (not checkpointed)", "historical per-update gradients",
                  "historical per-step old-category scores", "R15 clip saturation (not logged)"]

TARGETS = {
    "Y1_primary": "log q90(s+1) - log q90(s) of image scores of c's FIXED eval-only nonmember cohort (never members)",
    "Y1_abs": "q90(s+1) - q90(s) (same cohort)",
    "Y2": "gap(s+1) - gap(s), gap = log mean image score(nonmembers) - log mean image score(M), M = c's members at s "
          "(the cohort replayed during s+1), both evaluated at s and s+1",
    "Y3": "C_ap_eval(s) - C_ap_eval(s+1): pooled patch AP of Q2 items hosted on c's eval-only nonmember 'current' "
          "hosts (positive = deterioration)",
    "Y4_exposed": "real native Pixel AP(s) - AP(s+1) from NL-RET-00B STAGE_TRAJECTORY.json; POSTHOC only",
    "event": "Y1_primary > 0.05 (about +5% nonmember q90 in one transition)",
}
BASELINES = {
    "B0_constant": "training-fold mean",
    "B1_clock": "ridge on CLOCK",
    "B2_occupancy": "ridge on [D_n_members, 1/D_n_members]",
    "B3_last_tail": "ridge on [A_pix_q99, A_img_mean]",
    "B4_fixed_threshold": "two-group training means split at A_grow_prev > 0 (NaN -> group 0); threshold fixed at 0",
    "B5_fixed_schedule": "training mean per age bucket {0,1,2,>=3}",
    "B6_lawful_persistence": "ridge on [A_grow_prev, G_Agrow_prev_lag1] (lawful replay-tail change persistence)",
    "B7_shuffled_history": "ridge on ALL features with rows permuted inside the training fold (negative control)",
    "O1_oracle_persistence": "previous transition's Y1 (EVAL-ONLY input: ORACLE_DIAGNOSTIC_BASELINE, never lawful)",
}
MODELS = {
    "single_feature": "OLS on one standardized feature (every dictionary feature), LOCO OOF; descriptive ranking",
    "family_ridge": "ridge on one family's features",
    "controls": "ridge on CLOCK + D_n_members + A_pix_q99 + A_img_mean + 1[A_grow_prev>0]",
    "controls_plus_family": "ridge on controls + family",
    "all": "ridge on all lawful features",
    "alpha_grid": [0.1, 1.0, 10.0, 100.0, 1000.0],
    "alpha_selection": "inner leave-one-category-out CV inside the training fold (min MAE)",
    "preprocessing": "inside training folds: median imputation + missing indicator, standardization",
    "nonlinear": "not used (n ~ 300 grouped rows)",
}
VALIDATION = {
    "primary_data": "ID arm, 315 rows (105 x 3 seeds)",
    "LOCO": "15 folds; train on the other 14 categories (all seeds), predict all rows of the held-out category",
    "LOSO": "3 folds; train on two seeds, predict the third",
    "temporal": "for t = 4..14: train on rows with s+1 <= t, predict rows with s = t (all categories)",
    "FS_B_replication": "models trained on ID rows (LOCO folds) predict FS_B rows not shared with ID",
    "uncertainty": "category-clustered bootstrap (2,000 resamples of the 15 categories) of OOF MAE differences",
    "metrics": "MAE, RMSE, Spearman, bias, calibration slope; event precision/recall/PR-AUC at the 80th-percentile "
               "training-fold prediction; per-category and per-seed MAE",
}
GATE = {
    "best_lawful_baseline": "lowest OOF MAE among B0-B6 under the same protocol (chosen post hoc -> conservative)",
    "family_PASS": "(i) LOCO MAE(controls+family) <= 0.95 x MAE(best lawful baseline); (ii) clustered 95% CI of "
                   "MAE(controls+family) - MAE(best lawful) < 0; (iii) clustered 95% CI of MAE(controls+family) - "
                   "MAE(controls) < 0; (iv) LOSO: beats best lawful baseline in >= 2 of 3 held-out seeds; (v) per-category "
                   "OOF MAE below the best lawful baseline's in >= 9 of 15 categories",
    "MULTICLASS_SLOW_SIGNAL_SUPPORTED": ">= 1 family PASSES (i)-(v) on Y1_primary, temporal MAE <= best lawful "
                                        "temporal baseline, and FS_B replication MAE <= best lawful baseline",
    "NORMAL_ONLY_PREDICTIVE_BUT_REAL_AP_UNALIGNED": "the MULTICLASS conditions hold but the POSTHOC Spearman between "
                                                    "frozen OOF Y1 predictions and Y4 is <= 0.10",
    "CLASS_CONDITIONAL_SIGNAL_ONLY": "some family meets (ii) and (iii) but fails (iv) or (v), with >= 4/15 categories "
                                     "improved",
    "NO_INCREMENTAL_PREDICTIVE_SIGNAL": "no family meets (ii) and (iii)",
    "UNDERPOWERED_OR_INCOMPLETE": "any eligible ID row missing a primary outcome, or < 15 Y1 events",
    "INTEGRITY_BLOCKED": "audit/parity failure",
    "secondary": "Y2 and Y3 evaluated with the identical protocol and reported; they do not change the gate",
}


def main():
    require(not (C.OUT / "AUDIT_LOCK.json").exists(), "lock exists")
    require(K.read(C.OUT / "PARITY.json")["pass"], "parity first")
    cov = K.read(C.OUT / "FULL_STREAM_COVERAGE.json")
    require(not cov["bad_checkpoints"] and not cov["replay_record_mismatches"], "audit")
    K.write_json(C.OUT / "SIGNAL_DICTIONARY.json", {"signals": SIGNALS, "not_observable": NOT_OBSERVABLE,
                                                    "targets": TARGETS, "baselines": BASELINES, "models": MODELS,
                                                    "validation": VALIDATION, "gate": GATE})
    src = ["exps/nl_ret_01a/common.py", "exps/nl_ret_01a/extract.py", "exps/nl_ret_01a/prereg.py"]
    K.write_json(C.OUT / "AUDIT_LOCK.json", {
        "task": "NL-RET-01A", "agents_md_sha256": sha(C.ROOT / "AGENTS.md"),
        "signal_dictionary_sha256": sha(C.OUT / "SIGNAL_DICTIONARY.json"),
        "sources": {s: sha(C.ROOT / s) for s in src},
        "coverage_sha256": sha(C.OUT / "FULL_STREAM_COVERAGE.json"),
        "occupancy_sha256": sha(C.OUT / "REPLAY_OCCUPANCY.csv"), "parity_sha256": sha(C.OUT / "PARITY.json"),
        "note": "frozen before any feature or outcome value exists; analysis code (features.py, validate.py) is "
                "written during extraction without reading its outputs and hashed into ANALYSIS_LOCK.json before "
                "the first table is built",
        "utc": K.now()})
    print("prereg locked")


if __name__ == "__main__":
    main()
