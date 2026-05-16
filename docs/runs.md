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
