"""NL-RET-02 G1/G3 preregistration. Written and SHA-locked BEFORE any incoming-task feature exists and before any new
feature is paired with an outcome. `python -m exps.nl_ret_02.prereg` writes SIGNAL_DICTIONARY.json,
PROBE_MANIFEST.json, CAUSAL_TIMING_AUDIT.json and AUDIT_LOCK.json (all exclusive-create).

Disclosure: the NL-RET-01A outcomes (Y1/Y2/Y3) and its feature tables were inspected in NL-RET-01A; the targets are
reused unchanged (continuity, as the brief requires). The incoming-task features below have never been computed.
"""
from __future__ import annotations

from exps.cadfinal import common as K
from exps.dinomaly.common import require, sha
from exps.nl_ret_02 import common as C

PROBE_K = 4            # first 4 sampler batches of stage s+1 -> 32 distinct incoming images (min floor(n/8) = 7)

GN_ROW = {
    "GN_old_norm": "||g_c||, g_c = grad of the normal loss on c's Replay-24 members M_s^c at theta_s",
    "GN_cos": "cos(g_new, g_c), g_new = grad on the 32 probe images",
    "GN_dot": "g_new . g_c",
    "GN_negalign": "max(0, -GN_cos)",
    "GN_ratio": "||g_new|| / ||g_c||",
    "GN_orth_rel": "||g_new - proj_{g_c} g_new|| / ||g_c||",
    "GN_blk_cos_min": "min over the 9 decoder blocks (bottleneck + 8) of the per-block cos(g_new, g_c)",
    "GN_blk_cos_mean": "mean over blocks of the per-block cosine",
    "GN_blk_negfrac": "fraction of blocks with negative per-block cosine",
    "GN_blk_conc": "Herfindahl index over blocks of |block dot| shares (concentration of the interaction)",
    "GN_fo_sgd": "-g_c . g_new: first-order change of c's member loss under a unit-lr SGD step on g_new (LOCAL "
                 "approximation; not the StableAdamW update)",
    "GN_fo_sign": "-(g_c . sign(g_new)) / ||g_c||_1: first-order change per unit lr under a sign step, the direction "
                  "of StableAdamW's first steps from its fresh moments (optimizer re-created every stage); still an "
                  "approximation (later steps, clipping, weight decay, amsgrad, train-mode dropout not modelled)",
    "GN_b1_cos": "cos(g_b1, g_c), g_b1 = grad on the exact first training batch (8 incoming + 8 replay, as permuted)",
    "GN_b1_fo_sign": "-(g_c . sign(g_b1)) / ||g_c||_1",
    "GN_cos_mix": "cos(0.5 g_new + 0.5 g_rep, g_c), g_rep = sum_d (n_d/N) g_d (replay-weighted aggregate direction)",
}
GN_BOUNDARY = {"GNB_new_norm": "||g_new||", "GNB_b1_norm": "||g_b1||", "GNB_rep_norm": "||g_rep||",
               "GNB_cos_new_rep": "cos(g_new, g_rep)"}
GE_ROW = {
    "GE_cent_dist": "1 - cos(centroid of probe image vectors, centroid of c's member vectors); image vector = "
                    "L2-normalised mean of patch tokens of both frozen-encoder groups (1536-d)",
    "GE_nn_c2i": "mean over c's members of the min cosine distance to the probe images",
    "GE_nn_i2c": "mean over probe images of the min cosine distance to c's members",
    "GE_rel_nn": "GE_nn_i2c / min over old categories d of the same quantity (1 = c is the nearest old category)",
    "GE_rank": "rank of c by centroid distance among old categories / (k - 1) (0 = closest)",
    "GE_patch_cov": "mean over all patch tokens of c's members of the min cosine distance to the probe images' patch "
                    "tokens (1536-d per patch); patches of one image are NOT independent samples",
    "GE_patch_q90": "q90 of the same per-patch distances",
    "GE_redund": "mean over c's members of the max cosine similarity to members of other old categories (NaN if k=1)",
    "GE_cov_overlap": "mean squared cosine of principal angles between the top-r centred subspaces of probe and "
                      "member vectors, r = min(n_c - 1, 5); UNDEFINED (NaN) if n_c < 3",
}
GE_BOUNDARY = {"GEB_novelty": "mean over probe images of the min cosine distance to all M_s members",
               "GEB_novelty_q90": "q90 of the same", "GEB_inc_dispersion": "mean pairwise cosine distance of probe images"}
FN_ROW = {"FN_old_loss": "mean global-cosine loss of c's members at theta_s", "FN_old_img": "mean R_D image score of "
          "c's members at theta_s", "FN_loss_ratio": "FNB_inc_loss / FN_old_loss",
          "FN_shift": "log(mean probe image score / FN_old_img)"}
FN_BOUNDARY = {"FNB_inc_img_mean": "mean R_D image score of the probe images at theta_s", "FNB_inc_img_q90": "q90 of the "
               "same", "FNB_inc_loss": "mean global-cosine loss of the probe images",
               "FNB_inc_vs_buffer": "log(FNB_inc_img_mean / member-weighted mean image score of all M_s members)"}
FSQ = {"FSQ_loss": "mean Q2 ranking loss over stage s+1's first K Aux host draws at theta_s",
       "FSQ_nA": "mean ||g_aux||", "FSQ_sat": "fraction of draws with ranking loss < 1e-5",
       "FSQ_margin": "mean (positive patch mean - q99 background patch) on drawn items",
       "FSQ_bg_q99": "mean q99 background patch score", "FSQ_cos_new": "mean cos(g_aux, g_new)",
       "FSQ_step1_nN": "||g_b1|| (the normal gradient the first draw is norm-matched to)",
       "FSQ_step1_cos": "cos(g_aux_1, g_b1)", "FSQ_step1_scale": "rho ||g_N|| / (||g_aux|| + EPS) (unchanged rule)",
       "FSQ_step1_dose_rel": "scale * ||g_aux|| / ||g_N|| (effective EPS-attenuated dose)",
       "FSQ_cos_aux_c": "cos(g_aux_1, g_c) per old category",
       "use": "FS_B-only, DESCRIPTIVE (ID has no Q2): never enters an ID-trained predictor"}

GN = list(GN_ROW) + list(GN_BOUNDARY)
GE = list(GE_ROW) + list(GE_BOUNDARY)
FN = list(FN_ROW) + list(FN_BOUNDARY)
I0 = ["K_s", "K_age", "K_inv_age", "D_n_members", "INV_N", "AB0", "AB1", "AB2", "AB3"]
INFO_SETS = {
    "I0": "ridge on task clock + replay occupancy + age-bucket indicators (fixed-schedule information)",
    "I1": "I0 + every NL-RET-01A lawful feature (families A, C, D, E, F, G) + THR",
    "I2": "I1 + GE", "I3": "I1 + GN", "I4": "I1 + GE + GN (PRIMARY P1)", "I5": "I4 + FN",
    "I0+GE": "compact", "I0+GN": "compact", "I0+GE+GN": "compact (PRIMARY P2)", "I0+GE+GN+FN": "compact",
}
PRIMARY = ["I4", "I0+GE+GN"]
CONTROLS = {
    "CTRL_shuffle_incoming": "I4 where, inside each training fold only, the GE+GN block is permuted across training "
                             "rows of the same age bucket (test rows keep true features)",
    "CTRL_shuffle_grad_to_category": "I4 where, in every (arm, seed, s) group, the GN row-level block is permuted "
                                     "among the old categories (fixed RNG, label-free, whole table)",
    "CTRL_shuffle_history_order": "I4 where, in every (seed, category), the G_history columns are permuted across s",
    "B1_clock / B5_fixed_schedule": "age-only", "B2_occupancy": "occupancy-only",
}
BASELINES = ["B0_constant", "B1_clock", "B2_occupancy", "B3_last_tail", "B4_fixed_threshold", "B5_fixed_schedule",
             "B6_lawful_persistence", "I0"]
MODELS = {"family": "ridge, alpha in {0.1,1,10,100,1000} by inner leave-one-category-out CV inside the training fold; "
                    "fold-internal median imputation + missing indicators + standardisation (NL-RET-01A code)",
          "single_feature": "OLS-ridge on one feature, LOCO OOF (descriptive ranking only)",
          "nonlinear": "none (n = 315 grouped rows)"}
VALIDATION = {"primary": "ID arm, 315 rows (105 x 3 seeds), target Y1_primary (NL-RET-01A, unchanged), event Y1 > 0.05",
              "LOCO": "15 folds (14 with rows; zipper has none)", "LOSO": "3 folds",
              "temporal": "t = 4..14: train on rows with s + 1 <= t, predict rows with s = t",
              "FSB_replication": "ID-trained LOCO models predict FS_B rows; PRIMARY: s >= 7 (independently trained "
                                 "theta_s, 84 rows/seed); SECONDARY: s = 6 (shared theta_6, distinct outcome)",
              "uncertainty": "category-clustered paired bootstrap, 2,000 resamples of the 14 categories with rows; "
                             "two primary sets -> Bonferroni: 98.75% intervals",
              "metrics": "MAE (primary), RMSE, Spearman, event PR-AUC, precision/recall/false-alarm at the training "
                         "fold's 80th-percentile prediction, per-category and per-seed MAE"}
GATE = {
    "best_fixed": "lowest LOCO MAE among BASELINES (chosen post hoc -> conservative)",
    "P_PASS": "for a primary set P: (i) LOCO MAE(P) <= 0.95 x MAE(best_fixed); (ii) upper 98.75% clustered bound of "
              "MAE(P) - MAE(best_fixed) < 0; (iii) upper 98.75% bound of MAE(P) - MAE(I1) < 0; (iv) LOSO: MAE(P) < "
              "MAE(best fixed LOSO baseline) in >= 2 of 3 held-out seeds; (v) temporal MAE(P) <= 1.02 x best fixed "
              "temporal MAE; (vi) FS_B s >= 7 replication MAE(P) < best fixed replication MAE",
    "INTEGRITY_BLOCKED": "audit, reproduction, causal-timing or parameter-invariance failure",
    "UNDERPOWERED_OR_INCOMPLETE": "any eligible ID row lacking Y1 or any probe state missing, < 15 Y1 events, OR for "
                                  "some P: (i), (iv), (v), (vi) hold and both point differences in (ii), (iii) are "
                                  "< 0 but an interval crosses 0",
    "INCOMING_TASK_SIGNAL_SUPPORTED": "some P passes (i)-(vi) and the POSTHOC Spearman(frozen OOF P prediction, Y4) "
                                      "> 0.10",
    "NORMAL_PREDICTIVE_BUT_AP_UNALIGNED": "some P passes (i)-(vi) but that Spearman <= 0.10",
    "INCOMING_SIGNAL_RANK_ONLY": "no P passes, but for some P the LOCO Spearman exceeds both best_fixed's and I1's with "
                                 "lower 98.75% clustered bootstrap bounds of both Spearman differences > 0",
    "NO_INCREMENTAL_INCOMING_SIGNAL": "otherwise",
    "order": "INTEGRITY -> UNDERPOWERED (data) -> SUPPORTED/UNALIGNED -> UNDERPOWERED (variance) -> RANK_ONLY -> NO",
    "secondary": "Y2, Y3 evaluated identically and reported; they do not change the verdict",
}
TIMELINE = [
    "1. restore theta_s: SHA-verified end-of-stage-s checkpoint (R15 or FS_B own state; FS_B s=6 is the R15 state)",
    "2. restore M_s: buffer_index(seed, n, STREAM[:s]) = the replay identities recorded in R15 STAGE{s+1}.json",
    "3. incoming task s+1 arrives: its normal training set is loaded in full before stage s+1's first update "
    "(exps/rh01b/run.py: `current = C.load_features(c)` precedes train_stage_guarded; same in FS_B); probe = the "
    "incoming halves of the first K = 4 sampler batches, regenerated with the identical DR.Batches/generator path",
    "4. compute diagnostics in eval mode, no optimizer object, parameters hashed before/after",
    "5. freeze the record (per-state JSON, SHA in the feature-table lock)",
    "6. no optimizer step; no stage-(s+1) checkpoint, log or score is read by probe.py",
    "7. pair with the historical after-stage-(s+1) outcome (NL-RET-01A OUTCOME_TABLE, Y1-Y3) only in tables.py",
]
FACTS = {"optimizer_state": "O.make_optimizer / block_optimizer is called inside every stage: StableAdamW moments "
                            "start at zero at each boundary, so no historical moments are needed for the first step",
         "first_step_lr": "WarmCosineScheduler warm-up starts at 0: the iteration-0 step has lr = 0 (moments only); "
                          "the first parameter change happens at iteration 1 with lr = 2e-3/99",
         "iteration0_loss": "hm p = 0 at iteration 0 -> global_cosine_hm_percent masks nothing == global_cosine",
         "probe_vs_training_numerics": "training: train mode (dropout active), bf16 autocast, FP32 core; probes: eval "
                                       "mode (dropout off, deterministic), bf16 autocast, FP32 core, same reduction"}


def main():
    require(not (C.OUT / "AUDIT_LOCK.json").exists(), "lock exists")
    cov = K.read(C.OUT / "BOUNDARY_COVERAGE.json")
    rep = K.read(C.OUT / "BASELINE_REPRODUCTION.json")
    require(not cov["bad_checkpoints"] and not cov["replay_record_mismatches"] and rep["pass"], "G0 first")
    require(not C.EXT.exists() or not any(C.EXT.rglob("*.json")), "probe outputs must not exist before the lock")
    proto, n = C.proto_and_n()
    probes = {str(sd): {str(s): C.probe(sd, n, s, PROBE_K) for s in range(1, 15)} for sd in C.SEEDS}
    for sd in probes.values():
        for p in sd.values():
            require(len(p["incoming_idx"]) == 8 * PROBE_K, "probe images must be distinct (first epoch)")
    occ = {str(sd): {str(s): {c: len(v) for c, v in C.members(sd, n, s).items()} for s in range(1, 15)} for sd in C.SEEDS}
    K.write_json(C.OUT / "PROBE_MANIFEST.json", {
        "K": PROBE_K, "label": "FRESH_PREUPDATE_PROBES", "selection": "K fixed from BENCH.json (compute + label-free "
        "split-half gradient stability: K2 0.987, K4 0.995, K8 0.996) and the first-epoch constraint (toothbrush "
        "floor(60/8) = 7 batches); no outcome read", "bench_sha256": sha(C.OUT / "BENCH.json"),
        "probes": probes, "occupancy_M_s": occ, "utc": K.now()})
    K.write_json(C.OUT / "SIGNAL_DICTIONARY.json", {
        "GN_row": GN_ROW, "GN_boundary": GN_BOUNDARY, "GE_row": GE_ROW, "GE_boundary": GE_BOUNDARY, "FN_row": FN_ROW,
        "FN_boundary": FN_BOUNDARY, "FSQ_descriptive": FSQ, "families": {"GN": GN, "GE": GE, "FN": FN, "I0": I0},
        "information_sets": INFO_SETS, "primary_sets": PRIMARY, "negative_controls": CONTROLS, "baselines": BASELINES,
        "models": MODELS, "validation": VALIDATION, "gate": GATE,
        "targets": "NL-RET-01A Y1_primary (log nonmember q90 ratio s->s+1), Y2, Y3 unchanged; Y4 post hoc only",
        "missingness": "definitional NaN (GE_redund k=1, GE_cov_overlap n_c<3) -> fold-internal median + indicator"})
    K.write_json(C.OUT / "CAUSAL_TIMING_AUDIT.json", {"timeline": TIMELINE, "facts": FACTS,
                 "observation_timestamp": "after stage-s checkpoint, before stage s+1 iteration 0",
                 "lawful_inputs": ["theta_s", "M_s members (features, identities)", "incoming task s+1 train normals "
                                   "(probe identities)", "Q2 bank items hosted on stage-(s+1) allowed hosts (FS_B)"],
                 "excluded": ["any state after s", "nonmember (eval-only) images", "test images, masks, labels",
                              "tasks > s+1"], "utc": K.now()})
    src = ["exps/nl_ret_02/common.py", "exps/nl_ret_02/g0.py", "exps/nl_ret_02/probe.py", "exps/nl_ret_02/prereg.py",
           "exps/nl_ret_01a/common.py", "exps/nl_ret_01a/validate.py"]
    K.write_json(C.OUT / "AUDIT_LOCK.json", {
        "task": "NL-RET-02", "agents_md_sha256": sha(C.ROOT / "AGENTS.md"),
        "signal_dictionary_sha256": sha(C.OUT / "SIGNAL_DICTIONARY.json"),
        "probe_manifest_sha256": sha(C.OUT / "PROBE_MANIFEST.json"),
        "causal_timing_sha256": sha(C.OUT / "CAUSAL_TIMING_AUDIT.json"),
        "coverage_sha256": sha(C.OUT / "BOUNDARY_COVERAGE.json"),
        "reproduction_sha256": sha(C.OUT / "BASELINE_REPRODUCTION.json"),
        "sources": {s: sha(C.ROOT / s) for s in src}, "utc": K.now(),
        "note": "frozen before any incoming-task feature exists; tables.py / validate.py are hashed into "
                "ANALYSIS_LOCK.json before the first table is built"})
    print("prereg locked")


if __name__ == "__main__":
    main()
