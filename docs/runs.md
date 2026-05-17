# Run Log

## MetaNATH_Phase1_2_20260516_120342_raw_baseline_full15

Baseline locked before Phase 3.

| Field | Value |
| --- | --- |
| Run dir | `results/MetaNATH_Phase1_2_20260516_120342_raw_baseline_full15` |
| Phase | Phase 1-2 Light |
| Backbone | `facebook/dinov2-base` |
| Dataset | MVTec AD, 15 tasks |
| tau_acc | 0.6 |
| nearest_neighbors | 9 |
| max_coreset_size | 1000 |
| store_images | false |
| checkpoint_mode | `phase12_light` |
| DTD | missing, self-shift fallback |
| pixel_score_norm | `none` |
| gaussian_smoothing_sigma | 0.0 |
| patch_grid / n_patch | [16, 16] / 256 |
| avg current image AUROC | 0.9667 |
| avg current pixel AUROC | 0.9562 |
| avg current pixel AUPR | 0.4842 |
| final cumulative image AUROC | 0.9409 |
| final cumulative pixel AUPR | 0.3499 |

### Per-Task Breakdown

| Task | Category | Image AUROC | Pixel AUPR | Pixel AUROC | Image AP | Approval | Updates | Coreset |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | bottle | 1.0000 | 0.7523 | 0.9864 | 1.0000 | 1.0000 | 99 | 99 |
| 1 | cable | 0.9140 | 0.4594 | 0.9275 | 0.9517 | 1.0000 | 115 | 214 |
| 2 | capsule | 0.9418 | 0.4356 | 0.9707 | 0.9814 | 1.0000 | 89 | 303 |
| 3 | carpet | 0.9988 | 0.5364 | 0.9876 | 0.9996 | 1.0000 | 122 | 425 |
| 4 | grid | 1.0000 | 0.2905 | 0.9871 | 1.0000 | 1.0000 | 120 | 545 |
| 5 | hazelnut | 0.9971 | 0.7300 | 0.9938 | 0.9985 | 1.0000 | 188 | 733 |
| 6 | leather | 1.0000 | 0.2315 | 0.9816 | 1.0000 | 1.0000 | 109 | 842 |
| 7 | metal_nut | 1.0000 | 0.7921 | 0.9746 | 1.0000 | 1.0000 | 93 | 935 |
| 8 | pill | 0.9490 | 0.5883 | 0.9495 | 0.9902 | 1.0000 | 138 | 1000 |
| 9 | screw | 0.8301 | 0.1706 | 0.8641 | 0.9336 | 1.0000 | 134 | 1000 |
| 10 | tile | 1.0000 | 0.4381 | 0.9439 | 1.0000 | 1.0000 | 91 | 1000 |
| 11 | toothbrush | 0.9611 | 0.4432 | 0.9849 | 0.9848 | 1.0000 | 8 | 1000 |
| 12 | transistor | 0.9371 | 0.4617 | 0.9292 | 0.9356 | 1.0000 | 46 | 1000 |
| 13 | wood | 0.9781 | 0.5450 | 0.9053 | 0.9931 | 1.0000 | 122 | 1000 |
| 14 | zipper | 0.9932 | 0.3877 | 0.9571 | 0.9983 | 1.0000 | 48 | 1000 |

### Cumulative Eval

| After Task | Category | Cumulative Image AUROC | Cumulative Pixel AUPR | Cumulative Pixel AUROC | Images | Seconds |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | grid | 0.9513 | 0.3909 | 0.9550 | 560 | 72.2 |
| 9 | screw | 0.9378 | 0.3309 | 0.9399 | 1236 | 549.3 |
| 14 | zipper | 0.9409 | 0.3499 | 0.9334 | 1725 | 759.6 |

### Final Coreset Task Counts

- task 0: 32
- task 1: 107
- task 2: 15
- task 3: 59
- task 4: 112
- task 5: 164
- task 6: 63
- task 7: 49
- task 8: 52
- task 9: 86
- task 10: 61
- task 11: 8
- task 12: 27
- task 13: 122
- task 14: 43

### Notes

- Image-level detection is close to the Phase 1-2 target, but pixel localization is still below the instruction target.
- `tau_acc=0.6` behaved as pass-through on this run; all tasks had approval rate 1.0.
- Weakest pixel AUPR classes in this run: screw, leather, grid, zipper, capsule.
- This is not a Phase 3 run: no NSP2, CBP, Subspace Recycling, or N2B-NC consolidation yet.

## MetaNATH_Phase3_20260516_220707_phase3_balanced_top64_step3_patchdistill

Accepted Phase 3.0 4-task candidate.

| Field | Value |
| --- | --- |
| Source checkpoint | `results/MetaNATH_Phase3_20260516_151237_phase3_anchor_warmup_4task/last_checkpoint.pt` |
| Candidate checkpoint | `results/MetaNATH_Phase3_20260516_220707_phase3_balanced_top64_step3_patchdistill/last_checkpoint.pt` |
| Eval dir | `results/MetaNATH_Eval_20260516_220740_after_phase3_balanced_top64_step3_patchdistill_4task` |
| Acceptance report | `results/MetaNATH_Phase3_20260516_220707_phase3_balanced_top64_step3_patchdistill/acceptance_report.json` |
| Tasks | 4 |
| top_k_anchors | 64 |
| anchor selection | balanced by task id, 16 anchors per task |
| steps | 3 |
| loss weights | CLS 1.0, patch 1.0, LEJEPA 0.1 |
| drift | 0.0229 |
| patch_drift_mse | 0.0392 |
| rollback | false |
| coreset refresh | true, 466 entries |

### Before vs After

| Metric | Before | After | Delta | Gate |
| --- | ---: | ---: | ---: | --- |
| avg image AUROC | 0.9609 | 0.9628 | +0.0019 | n/a |
| avg pixel AUROC | 0.9679 | 0.9710 | +0.0031 | n/a |
| avg pixel AUPR | 0.5463 | 0.5486 | +0.0022 | n/a |
| final cumulative image AUROC | 0.9397 | 0.9483 | +0.0086 | pass |
| final cumulative pixel AUPR | 0.4307 | 0.4259 | -0.0049 | pass within -0.005 tolerance |

### Notes

- Earlier unbalanced Phase 3 candidates improved image AUROC but regressed final cumulative pixel AUPR beyond the gate.
- Balanced anchor selection reduced cumulative pixel regression enough to pass the configured acceptance threshold while improving image-level cumulative AUROC.
- NSP2, CBP reset, and Subspace Recycling remain disabled for this accepted Phase 3.0 candidate.

## 8-Task Phase 3 Scale Sanity

Checked Phase 3.0 scaling on the first 8 MVTec tasks. Both candidates were
correctly rejected by the acceptance gate because final cumulative pixel AUPR
regressed beyond the configured `-0.005` tolerance.

| Candidate | Drift | Before Pixel AUPR | After Pixel AUPR | Delta | Image AUROC Delta | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `phase3_balanced_top64_step3_patchdistill_8task` | 0.0187 | 0.3255 | 0.3164 | -0.0091 | +0.0022 | rejected |
| `phase3_balanced_top64_step1_8task` | 0.0037 | 0.3255 | 0.3181 | -0.0074 | +0.0008 | rejected |

### Notes

- Phase 3 plumbing scales to 8 tasks: anchors were balanced at 8 per task,
  coreset refresh covered 1000 entries, and rollback was not triggered.
- The acceptance gate is doing useful work: it prevents accepting image-level
  gains that damage cumulative pixel localization.
- Do not run Phase 3 full-15 as an accepted benchmark yet. The next research
  step is improving the pixel-preserving objective or per-task calibration.

## MetaNATH_Phase3_20260517_121318_phase3_conservative_8task

Accepted conservative Phase 3.0 8-task candidate.

| Field | Value |
| --- | --- |
| Config | `conf/config_phase3_conservative.yaml` |
| Source checkpoint | `results/MetaNATH_Phase3_20260516_222751_phase3_anchor_warmup_8task/last_checkpoint.pt` |
| Candidate checkpoint | `results/MetaNATH_Phase3_20260517_121318_phase3_conservative_8task/last_checkpoint.pt` |
| Eval dir | `results/MetaNATH_Eval_20260517_121406_after_phase3_conservative_8task` |
| Acceptance report | `results/MetaNATH_Phase3_20260517_121318_phase3_conservative_8task/acceptance_report.json` |
| Tasks | 8 |
| top_k_anchors | 64 |
| anchor selection | balanced by task id, 8 anchors per task |
| steps | 1 |
| lr | 0.000003 |
| unfreeze_last_blocks | 1 |
| loss weights | CLS 0.5, patch 5.0, LEJEPA 0.0 |
| drift | 0.000048 |
| patch_drift_mse | 0.000093 |
| rollback | false |
| coreset refresh | true, 1000 entries |

### Before vs After

| Metric | Before | After | Delta | Gate |
| --- | ---: | ---: | ---: | --- |
| avg image AUROC | 0.9800 | 0.9800 | +0.0000 | n/a |
| avg pixel AUROC | 0.9758 | 0.9759 | +0.0000 | n/a |
| avg pixel AUPR | 0.5267 | 0.5267 | +0.0000 | n/a |
| final cumulative image AUROC | 0.9715 | 0.9717 | +0.0002 | pass |
| final cumulative pixel AUPR | 0.3255 | 0.3254 | -0.0001 | pass |

### Per-Task Final Pixel AUPR Delta

| Category | Delta |
| --- | ---: |
| hazelnut | -0.000250 |
| leather | -0.000158 |
| bottle | -0.000123 |
| metal_nut | -0.000113 |
| carpet | -0.000039 |
| capsule | +0.000044 |
| grid | +0.000103 |
| cable | +0.000901 |

### Notes

- This candidate fixes the 8-task acceptance failure by reducing update strength:
  lower LR, one final backbone block, lower CLS distillation, and LEJEPA disabled.
- With `steps=1`, patch-token distillation starts at zero because teacher and
  student patch tokens are initially identical. The pass is mainly due to the
  conservative update, not proof that patch distillation is strong enough.
- Score distribution checks on the rejected 8-task step1 candidate showed
  hazelnut/leather anomaly pixel tails dropping more than normal tails, while
  cable improved because normal high-tail scores dropped more than anomaly scores.

## MetaNATH_Phase3_20260517_131327_phase3_experimental_nsp2_cbp_smoke

Experimental smoke only. This run proves the NSP2/CBP CLI path executes, but it
is not an accepted benchmark because no before/after acceptance evaluation was
run for this experimental config.

| Field | Value |
| --- | --- |
| Config | `conf/config_phase3_experimental_nsp2_cbp.yaml` |
| Source checkpoint | `results/MetaNATH_Phase3_20260516_222751_phase3_anchor_warmup_8task/last_checkpoint.pt` |
| Candidate checkpoint | `results/MetaNATH_Phase3_20260517_131327_phase3_experimental_nsp2_cbp_smoke/last_checkpoint.pt` |
| Tasks in source checkpoint | 8 |
| NSP2 | enabled |
| CBP | enabled, reset allowed |
| drift | 0.000044 |
| patch_drift_mse | 0.000089 |
| rollback | false |
| NSP2 null_dim | 761 |
| NSP2 recycled | false |
| CBP dead_units / reset_units | 0 / 0 |

### Notes

- NSP2 projection, Subspace Recycling stats, and CBP reset code paths are covered
  by `scripts/test_integration_2.py`.
- In this real checkpoint smoke, Subspace Recycling was not needed because the
  fitted task subspace left a large null space (`null_dim=761`).
- CBP reset did not fire on the real backbone because no units were below the
  configured utility threshold. This is healthy behavior, not a failure.
- Do not report NSP2/CBP as improving benchmark metrics until this experimental
  config passes the same before/after acceptance gate used by Phase 3.0.
