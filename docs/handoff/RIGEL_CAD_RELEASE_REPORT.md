# RIGEL-CAD v1.0: release results report

> **For readers outside the project.** This is a copy of the internal report `agents/reports/rigel_cad_release_01.md`; `agents/` is not in git. Paths under `/mnt/k/...` are local caches and are not in the repository. `.rigel` state files are not committed: re-create them with the commands in §15.

**Date:** 2026-11-26.

**Status:** release candidate, frozen method. Every number below is read from a result file in the repository (paths in §15).

**Scope.** This is the single reference report for writing the paper chapters. It covers:
- accuracy on MVTec AD, VisA, MPDD and MVTec AD 2;
- forgetting;
- ablations;
- the budget curve;
- robustness;
- a matched-backbone check;
- storage;
- training and inference speed;
- the claims the evidence supports.

It supersedes `docs/handoff/RIGEL_CAD_FINAL_RESULTS_HANDOFF.md` (the G3E-PAPER-01 copy). Numbers on MVTec AD, VisA and MPDD are unchanged from that report. The local MVTec AD 2 test_public numbers are **corrected** (§4.3, erratum E1).

**Names.** RIGEL-CAD is the research arm G3-E.

| Name in this report | Research name | Meaning |
|---|---|---|
| **RIGEL-CAD** (8 MiB) | E@8, E | the proposed method, 8 MiB persistent state |
| RIGEL-CAD @4 / @16 / @32 | E@4 / E@16 / E@32 | the same method at another budget |
| RIGEL-CAD-∞ | U | unbounded reference: every per-task coreset is kept |
| No-slow (F) | F | ablation: no slow level, all 8 MiB spent on Euclidean keys |

---

## 0. Release summary

**RIGEL-CAD at 8 MiB.** Final model, task-agnostic. Order a, seed 2711 (MVTec AD 2: the 8-scenario stream). Measured on an RTX 4050 Laptop GPU.

| Dataset (tasks) | Pixel AP [CI95] | Image AUROC | FM | Persistent state | Train, whole stream (compute / with decode) | Inference per image (compute / with decode) |
|---|---|---|---|---|---|---|
| MVTec AD (15) | **0.593** [0.575, 0.612] | 0.983 | 0.033 | 8,388,104 B | 3.6 / 8.0 min (3,629 images) | 50 / 137 ms |
| VisA (12) | **0.460** [0.430, 0.485] | 0.935 | 0.012 | 8,388,104 B | 15.6 / 20.1 min (8,659 images) | 50 / 89 ms |
| MPDD (6) | **0.508** [0.490, 0.530] | 0.940 | 0.013 | 8,388,104 B | 0.8 / 1.5 min (888 images) | 50 / 99 ms |
| MVTec AD 2 (8), server | AU-PRO 0.05 **38.2** (private) / **34.0** (mixed) | ClassF1 75.4 / 76.6 | — | 8,388,104 B | 3.4 / 18.6 min (2,528 images) | 74 / 401 ms |
| MVTec AD 2, local test_public (corrected) | AU-PRO 0.05 / 0.30 **0.359 / 0.592**, SegF1 0.168 | — | 0.012 (AU-PRO) | | | |

**Key facts for the paper.**
1. **Bounded memory, high retention.** At a constant 8 MiB, RIGEL-CAD keeps 93–98 % of the unbounded memory's pixel AP on MVTec AD, VisA and MPDD; U uses 13–107 MB. On MVTec AD 2 it matches U's AU-PRO 0.05 (0.359 vs 0.358; U uses 32.8 MB).
2. **The slow level matters (the NL contribution).** At the same 8 MiB, whitening by the slow statistic beats spending the bytes on more keys: +0.025 / +0.079 / +0.035 pixel AP (CI > 0) on MVTec AD, VisA and MPDD, and +0.066 AU-PRO 0.05 on MVTec AD 2.
3. **Robust.** Over 3 orders × 3 seeds, the std of pixel AP is ≤ 0.008.
4. **Cheap.** There is no gradient training. The per-task update is closed-form, taking seconds to about a minute. The head costs about 2.8 ms per image, and total compute is about 50 ms per image, dominated by the frozen backbone.
5. **Qualifiers.**
   - The raw advantage over published continual-AD methods depends on the DINOv2 backbone. With matched ViT-B/8, RIGEL-CAD is below CADIC.
   - The MVTec AD 2 server result is far below the high-resolution leaderboard.
   - SegF1 is limited by the untuned threshold rule.

**What changed since the G3E-PAPER-01 handoff.**
- **Erratum E1:** MVTec AD 2 local masks were fixed, and the local AU-PRO was re-run. 0.263 → 0.359 (§4.3).
- **New:** MVTec AD 2 controls U and F with corrected masks (§4.2).
- **New:** speed and GPU-memory benchmark from raw images (§11) and a storage breakdown (§10).
- No method change. MVTec AD, VisA and MPDD numbers are unchanged.

---

## 1. Method in one page

RIGEL-CAD is a continual anomaly detector for a stream of normal-only tasks. It learns one category (task) at a time, never sees a task again, and runs at test time with no task ID. It has no gradient training.

**Encoder (frozen).**
- DINOv2-reg ViT-B/14 at 392 px, full-field square resize.
- The token is the mean of 4 intermediate blocks.
- Result: 28×28 = 784 patch tokens per image, unit-L2, stored in fp16 (768-d).

**Two memory levels with different update frequencies.** This is the Nested-Learning reading: NL Def. 2–3 and CMS Eq. 70–71, PAPER_INFERRED.

| Level | What it stores | When and how it is updated | Bytes at 8 MiB |
|---|---|---|---|
| Fast | key bank K of unit fp16 tokens (cap 3,925) | Once per task. A 1 % greedy k-centre coreset C_t of the task's tokens is computed, then K ← k-centre(K ∪ C_t, cap); both steps use a seeded 128-d projection | 3,925 × 1,536 = 6,028,800 |
| Slow | pooled residual second moment S (768×768) and a count n | Accumulated over all tasks: S += Σ r rᵀ, with r = x − nearest coreset key, over the task's non-key tokens | 2,359,304 (S counted at fp32 plus the count) |

**Read-out.**
- The whitener is W = (½·S/n + ½·tr(S/n)/768·I)^(−1/2), from an fp64 eigendecomposition.
- Patch energy is e(q) = ½·min_{m∈K} ‖qW − mW‖².
- The 28×28 energy grid is upsampled bilinearly to 392 px and smoothed with a Gaussian (σ 4).
- The image score is the mean of the top 1 % of pixels.

**Budget.**
- Persistent state = |K|·1,536 B + 2,359,304 B ≤ 8 MiB. At 3,925 keys this is 8,388,104 B = 8 MiB − 504 B.
- The budget is asserted after every task.
- The frozen backbone (86.6 M parameters, about 346 MB in fp32) is shared and not counted. The literature makes the same convention for CADIC, UCAD and GCR.

**Code:** `rigel_cad/` (`python -m rigel_cad {stream,ad2,bench}`). Its grids are bitwise equal to the frozen research code at every stage (`rigel_cad/tests/test_equivalence.py`).

---

## 2. Protocol

| Item | Setting |
|---|---|
| Datasets | MVTec AD (15 tasks), VisA (12), MPDD (6); MVTec AD 2 (8 scenarios, held-out confirmation) |
| Training data | train/good only, every image, one pass per task |
| Order | a = alphabetical (main). Orders b and c are the registered NL03-MULTIRATE-01 permutations |
| Seeds | 2711 (main); 2712 and 2713 for robustness |
| Evaluation | Final model (after the last task), task-agnostic, full official test set of each category |
| Pixel AP | Native-resolution binned pixel AUPR (8,192 bins), category mean |
| Image AUROC | From the top-1 % image score, category mean |
| FM (forgetting) | Mean over tasks of (max over stages − final) pixel AP. The last task is excluded |
| acq AP | Pixel AP of a task right after it was learned |
| Prequential AP | Mean over stages of the AP on the tasks seen so far |
| CI95 | Paired bootstrap, 2,000 replicates, registered weights `boot_weights(n, position)`, order a only |
| MVTec AD 2 | Stream over the 8 scenarios (alphabetical, train/ only). θ = mean + 3σ of the validation/ pixel scores at the acquisition stage (D-THR, no test labels). AU-PRO at FPR 0.05 / 0.30 (official integration), SegF1 |

Design decisions were taken on MVTec AD, VisA and MPDD. The MVTec AD 2 private test, scored by the official server, is the held-out confirmation. It was submitted once.

---

## 3. Main results on MVTec AD, VisA and MPDD (order a, seed 2711, final model)

Source: `results/hope_cad/g3e_paper_01/RESULTS.json` → `order_a_seed2711` and `A4`. The same numbers under RIGEL-CAD names are in `results/rigel_cad/paper/RIGEL_CAD_TABLES.md`.

| Dataset | Method | Pixel AP [CI95] | Image AUROC [CI95] | FM | acq AP | Preq. AP | State (B) |
|---|---|---|---|---|---|---|---|
| MVTec AD | RIGEL-CAD-∞ (U) | 0.6348 [0.6164, 0.6540] | 0.9953 [0.9917, 0.9979] | 0.0034 | 0.6293 | 0.6552 | 46,070,792 |
| MVTec AD | **RIGEL-CAD 8 MiB** | **0.5927** [0.5747, 0.6123] | **0.9826** [0.9767, 0.9877] | 0.0333 | 0.6129 | 0.6395 | 8,388,104 |
| MVTec AD | No-slow F (8 MiB) | 0.5681 | 0.9740 | 0.0368 | — | — | 8,388,096 |
| VisA | RIGEL-CAD-∞ (U) | 0.4868 [0.4573, 0.5129] | 0.9746 [0.9680, 0.9800] | 0.0030 | 0.4784 | 0.5068 | 106,638,344 |
| VisA | **RIGEL-CAD 8 MiB** | **0.4603** [0.4298, 0.4853] | **0.9349** [0.9242, 0.9448] | 0.0119 | 0.4638 | 0.4934 | 8,388,104 |
| VisA | No-slow F (8 MiB) | 0.3816 | 0.8760 | 0.0406 | — | — | 8,388,096 |
| MPDD | RIGEL-CAD-∞ (U) | 0.5175 [0.5000, 0.5379] | 0.9627 [0.9449, 0.9785] | 0.0012 | 0.5178 | 0.3454 | 13,057,544 |
| MPDD | **RIGEL-CAD 8 MiB** | **0.5082** [0.4895, 0.5296] | **0.9399** [0.9164, 0.9606] | 0.0128 | 0.5163 | 0.3433 | 8,388,104 |
| MPDD | No-slow F (8 MiB) | 0.4733 | 0.9550 | 0.0211 | — | — | 8,388,096 |

**Paired differences in pixel AP (CI95):**

| Dataset | RIGEL-CAD − F (slow level, same bytes) | U − RIGEL-CAD (cost of the 8 MiB bound) |
|---|---|---|
| MVTec AD | **+0.0245** [0.0177, 0.0335] | +0.0421 [0.0367, 0.0468] |
| VisA | **+0.0786** [0.0693, 0.0878] | +0.0265 [0.0201, 0.0338] |
| MPDD | **+0.0349** [0.0234, 0.0460] | +0.0092 [0.0015, 0.0186] |

**Reading.**
- At 8 MiB, RIGEL-CAD keeps 93.4 % (MVTec), 94.6 % (VisA) and 98.2 % (MPDD) of the unbounded pixel AP. It uses 18.2 %, 7.9 % and 64.2 % of U's bytes.
- The slow level beats spending the same bytes on more keys on all three datasets (CI above 0). This is the Nested-Learning contribution (owner decision 2026-11-25, concluded).
- Image AUROC on MPDD is the exception: F is higher by 0.0151, but the CI crosses 0 (§5).
- Forgetting is small in absolute terms: FM ≤ 0.033 AP. It is still 4–11× the FM of the unbounded reference.

---

## 4. MVTec AD 2 (held-out confirmation)

### 4.1 Official server (test_private, test_private_mixed)

These are owner-reported numbers in %, from a single submission named RIGEL-CAD on 2026-11-25. The archive `E.tar.gz` has sha256 `1294efac…0626`. The method was locked before submission, and the threshold rule D-THR was declared before submission.

| Split | AU-PRO 0.05 | SegF1 | ClassF1 |
|---|---|---|---|
| test_private | **38.2** | 20.5 | 75.4 |
| test_private_mixed | **34.0** | 15.3 | 76.6 |

AU-PRO 0.05 per scenario (private / mixed):

| Can | Fabric | Fruit jelly | Rice | Sheet metal | Vial | Wallplugs | Walnuts |
|---|---|---|---|---|---|---|---|
| 10.8 / 8.1 | 24.0 / 25.2 | 51.5 / 51.4 | 34.1 / 35.5 | 32.4 / 30.2 | 62.4 / 58.9 | 30.6 / 15.3 | 59.8 / 47.0 |

**The mask fix does not change these numbers.** The server computes them with its own ground truth. The submitted maps and thresholds do not depend on test masks; θ comes from validation/ only, and the re-run θ is bitwise equal.

**Context (reported by the MVTec AD 2 authors).** This is not a matched comparison: those baselines train one model per category, at 256 px.

| Method | AU-PRO 0.05 (private / mixed) | SegF1 | ClassF1 |
|---|---|---|---|
| EfficientAD @256 | 30.8 / 19.2 | 15.4 / 8.0 | 76.8 / 77.8 |
| PatchCore @256 | 28.8 / 26.0 | 3.7 / 3.4 | 8.0 / 9.9 |
| RD++ @256 | 27.1 / 25.2 | 19.2 / 16.7 | 70.5 / 73.0 |
| MSFlow @256 | 24.3 / 11.9 | 21.8 / 9.0 | 51.1 / 72.0 |
| PatchCore, high resolution | 62.3 / 52.6 | — | — |
| Leaderboard top (joint, high resolution) | about 83–89 / 78–82 | about 60 / 57 | about 78–86 |

RIGEL-CAD is one continual model at 8 MiB. Its AU-PRO 0.05 is above all four 256 px per-category baselines. Its SegF1 and ClassF1 are in the range of the best of them. It is far below high-resolution PatchCore and the leaderboard top.

### 4.2 Local test_public, corrected (re-run 2026-11-26)

**Run.** `results/rigel_cad/ad2/run_ad2_maskfix.sh`:
- `python -m rigel_cad ad2 --budget-mib 8 --seed 2711 --per-stage --no-submission`, with the token cache;
- 384 s on the RTX 4050;
- result in `results/rigel_cad/ad2/RESULTS_ad2.json`.

**Controls.** U and F were run through the same package and the same threshold rule (`run_ad2_controls.sh`, final stage only): `results/rigel_cad/ad2/controls/{U,F}/RESULTS_ad2.json`.

**Checks.**
- θ for RIGEL-CAD is bitwise equal to the run before the fix, and to the locked D-THR values.
- θ for U is bitwise equal to the locked D-THR U values.
- Mask audit, per scenario (`mask_audit` in the JSON): 0 of 379 good images carry defect pixels, and 705 of 705 bad images do.

| Scenario | RIGEL-CAD AU-PRO .05 / .30 / SegF1 | U (32.8 MB) | F (8 MiB, no slow level) |
|---|---|---|---|
| can | 0.110 / 0.313 / 0.001 | 0.080 / 0.322 / 0.000 | 0.003 / 0.166 / 0.000 |
| fabric | 0.238 / 0.519 / 0.053 | 0.186 / 0.497 / 0.224 | 0.246 / 0.493 / 0.146 |
| fruit_jelly | 0.391 / 0.615 / 0.236 | 0.400 / 0.633 / 0.138 | 0.343 / 0.580 / 0.253 |
| rice | 0.490 / 0.694 / 0.055 | 0.511 / 0.717 / 0.211 | 0.336 / 0.581 / 0.353 |
| sheet_metal | 0.180 / 0.377 / 0.114 | 0.204 / 0.406 / 0.250 | 0.181 / 0.404 / 0.316 |
| vial | 0.706 / 0.918 / 0.334 | 0.774 / 0.947 / 0.333 | 0.612 / 0.839 / 0.300 |
| wallplugs | 0.222 / 0.483 / 0.078 | 0.232 / 0.512 / 0.078 | 0.160 / 0.422 / 0.063 |
| walnuts | 0.538 / 0.817 / 0.474 | 0.478 / 0.818 / 0.456 | 0.470 / 0.780 / 0.478 |
| **Mean** | **0.3594 / 0.5921 / 0.1680** | 0.3582 / 0.6067 / 0.2114 | 0.2939 / 0.5333 / 0.2388 |

**Streaming quantities (RIGEL-CAD, per stage):**
- FM of AU-PRO 0.05 = 0.0116. This uses the locked definition: mean over the first 7 scenarios of the max over stages j..T−2, minus the final value.
- Acquisition AU-PRO 0.05 = 0.3502.
- Final 0.3594 is above acquisition, so later tasks improved some earlier ones: can 0.059 → 0.110, fabric 0.225 → 0.238.

| Scenario | AU-PRO 0.05 at stages j … 7 |
|---|---|
| can | .059 .115 .146 .127 .120 .108 .080 .110 |
| fabric | .225 .210 .194 .191 .191 .254 .238 |
| fruit_jelly | .365 .367 .358 .377 .383 .391 |
| rice | .512 .531 .518 .533 .490 |
| sheet_metal | .177 .176 .175 .180 |
| vial | .719 .708 .706 |
| wallplugs | .206 .222 |
| walnuts | .538 |

**Reading (facts only).**
- **Slow level on AD 2.** RIGEL-CAD − F is +0.066 AU-PRO 0.05 (6/8 scenarios higher) and +0.059 AU-PRO 0.30 (7/8). There are no CIs: the bootstrap is not registered for AD 2, and there are 8 scenarios.
- **Bound cost on AD 2.** U − RIGEL-CAD is −0.001 AU-PRO 0.05 (U higher on 5/8) and +0.015 AU-PRO 0.30 (7/8). At 8 MiB, RIGEL-CAD matches the 32.8 MB unbounded memory on AU-PRO 0.05.
- **SegF1 is threshold-limited.** RIGEL-CAD has the lowest SegF1 of the three (0.168 vs U 0.211, F 0.239), even though its AU-PRO is highest. SegF1 depends only on the D-THR calibration (mean + 3σ of validation scores). Its θ is not tuned, and the energy scales differ between arms. Rice is the clearest case: RIGEL-CAD AU-PRO 0.490 with SegF1 0.055, versus F 0.336 with 0.353. A better threshold rule is an open improvement, and it must be chosen without test labels.
- **Local vs server.** Local test_public AU-PRO 0.05 (35.9) is now close to the server's test_private value (38.2). The pre-fix local value was 26.3.

### 4.3 Erratum E1: test_public masks (found 2026-11-26, fixed)

**What was wrong.**
- In MVTec AD 2, every good test_public image has the same file name as a bad one, for example `good/000_regular.png` and `bad/000_regular.png`.
- The old `ad2_mask_of` matched on file stem only. All 379 good images therefore received the defect mask of their namesake.
- The research script `exps/nl03_multirate_01/ad2_aupro.py` (frozen, not edited) had the same lookup.

**What it affected.**
- Every local test_public AU-PRO and SegF1 published before this report: NL03-MULTIRATE-01 §10, RIGEL-CAD-01 §9, and the AD 2 table in `rigel_cad/README.md` (commit 12514e2).
- It did not affect the server numbers, the thresholds, the submission maps, or anything on MVTec AD, VisA or MPDD.

**Fix.**
- `rigel_cad/data.py::ad2_mask_of` now takes the label from the directory. `good/` returns None, meaning all-normal. `bad/` returns `ground_truth/bad/<stem>_mask.png` and raises an error if it is missing.
- `run_ad2` asserts that no good image carries defect pixels, and writes a `mask_audit`.
- Two new tests: a synthetic same-name test, and an all-images check on the real dataset.
- The equivalence test keeps the old lookup inside the test file only, so it can still prove that the package's metric code equals the research code on the locked numbers.

| RIGEL-CAD, test_public | Before (defective masks) | After (corrected) |
|---|---|---|
| AU-PRO 0.05 | 0.2633 | **0.3594** |
| AU-PRO 0.30 | 0.4750 | **0.5921** |
| SegF1 | 0.1493 | **0.1680** |
| FM (AU-PRO 0.05) | 0.0077 | 0.0116 |
| Acquisition AU-PRO 0.05 | 0.2557 | 0.3502 |
| U AU-PRO 0.05 / 0.30 / SegF1 | 0.2622 / 0.4954 / 0.1864 | 0.3582 / 0.6067 / 0.2114 |

The pre-fix file is kept as `results/rigel_cad/ad2/RESULTS_ad2_STEM_MASK_DEFECT.json`.

---

## 5. Ablations at 8 MiB (A4; order a, seed 2711), paired RIGEL-CAD (E) − X

**The arms.**
- **Q:** hindsight k-centre over the union of *all* seen coresets, at E's cap. Q has to keep every past coreset to make its choice, so it is a hindsight selection reference, not a streaming-feasible method within 8 MiB.
- **E4:** E's keys, with W built from the current task's S only.
- **F:** Euclidean (W = I), all 8 MiB spent on keys (5461 keys).
- **G:** a generic covariance W (canonical G1b), at (B − G1_B)/1536 = 3923 keys.
- **FIFO:** evicts the oldest keys first.
- **B1:** E's keys without whitening (6.03 MB).

| Dataset | X | Pixel AP | Image AUROC | FM | E − X AP [CI95] | E − X AUROC [CI95] |
|---|---|---|---|---|---|---|
| MVTec | (E) | 0.5927 | 0.9826 | 0.0333 | | |
| MVTec | Q | 0.6092 | 0.9865 | 0.0211 | **−0.0166** [−0.0210, −0.0123] | −0.0039 [−0.0096, 0.0015] |
| MVTec | E4 | 0.5885 | 0.9793 | 0.0550 | +0.0042 [0.0003, 0.0087] | +0.0033 [−0.0007, 0.0069] |
| MVTec | F | 0.5681 | 0.9740 | 0.0368 | **+0.0245** [0.0177, 0.0335] | +0.0086 [0.0010, 0.0155] |
| MVTec | G | 0.5970 | 0.9786 | 0.0216 | −0.0043 [−0.0103, 0.0018] | +0.0040 [−0.0018, 0.0100] |
| MVTec | FIFO | 0.3945 | 0.8520 | 0.2614 | +0.1981 [0.1824, 0.2131] | +0.1306 [0.1125, 0.1493] |
| MVTec | B1 | 0.5425 | 0.9702 | 0.0488 | +0.0501 [0.0438, 0.0581] | +0.0124 [0.0066, 0.0183] |
| VisA | (E) | 0.4603 | 0.9349 | 0.0119 | | |
| VisA | Q | 0.4518 | 0.9516 | 0.0179 | +0.0084 [0.0019, 0.0140] | **−0.0167** [−0.0255, −0.0078] |
| VisA | E4 | 0.4492 | 0.9189 | 0.0337 | +0.0110 [0.0078, 0.0145] | +0.0160 [0.0107, 0.0213] |
| VisA | F | 0.3816 | 0.8760 | 0.0406 | **+0.0786** [0.0693, 0.0878] | +0.0589 [0.0454, 0.0727] |
| VisA | G | 0.4359 | 0.9415 | 0.0214 | +0.0244 [0.0145, 0.0322] | −0.0066 [−0.0159, 0.0020] |
| VisA | FIFO | 0.1668 | 0.6764 | 0.3366 | +0.2934 [0.2713, 0.3131] | +0.2586 [0.2362, 0.2812] |
| VisA | B1 | 0.3725 | 0.8478 | 0.0424 | +0.0878 [0.0802, 0.0963] | +0.0871 [0.0739, 0.1000] |
| MPDD | (E) | 0.5082 | 0.9399 | 0.0128 | | |
| MPDD | Q | 0.5047 | 0.9338 | 0.0169 | +0.0035 [−0.0035, 0.0106] | +0.0062 [−0.0068, 0.0187] |
| MPDD | E4 | 0.5036 | 0.9388 | 0.0221 | +0.0046 [0.0001, 0.0104] | +0.0011 [−0.0064, 0.0083] |
| MPDD | F | 0.4733 | 0.9550 | 0.0211 | **+0.0349** [0.0234, 0.0460] | −0.0151 [−0.0324, 0.0015] |
| MPDD | G | 0.4911 | 0.9349 | 0.0333 | +0.0172 [0.0101, 0.0240] | +0.0050 [−0.0082, 0.0183] |
| MPDD | FIFO | 0.4657 | 0.9078 | 0.0638 | +0.0426 [0.0302, 0.0543] | +0.0321 [0.0011, 0.0607] |
| MPDD | B1 | 0.4550 | 0.9466 | 0.0280 | +0.0532 [0.0437, 0.0644] | −0.0067 [−0.0261, 0.0131] |

**Reading (facts only).**
- **Pixel AP.** E's CI is above 0 against:
  - F, B1 and FIFO on 3/3 datasets;
  - E4 on 3/3, with the MVTec and MPDD lower bounds close to 0;
  - G on VisA and MPDD (MVTec is a tie);
  - Q on VisA.
- **Q > E on MVTec pixel AP** (−0.0166) and on VisA image AUROC (−0.0167). Q is a hindsight reference that needs all past coresets.
- **Image AUROC.** F is higher than E on MPDD (−0.0151), but the CI crosses 0.

---

## 6. Budget curve (A3; order a, 3 seeds, mean ± std)

| Dataset | Arm | Bytes | Pixel AP | Image AUROC | FM | ms/img |
|---|---|---|---|---|---|---|
| MVTec | E@4 | 4,193,288 | 0.5583 ± 0.0151 | 0.9661 ± 0.0070 | 0.0445 | 1.09 |
| MVTec | E@8 | 8,388,104 | 0.5946 ± 0.0063 | 0.9848 ± 0.0020 | 0.0286 | 2.13 |
| MVTec | E@16 | 16,776,200 | 0.6201 ± 0.0016 | 0.9898 ± 0.0023 | 0.0177 | 4.13 |
| MVTec | E@32 | 33,553,928 | 0.6326 ± 0.0028 | 0.9946 ± 0.0005 | 0.0056 | 8.56 |
| MVTec | U | 46,070,792 | 0.6342 ± 0.0004 | 0.9954 ± 0.0004 | 0.0030 | 9.75 |
| VisA | E@4 | 4,193,288 | 0.3882 ± 0.0199 | 0.8996 ± 0.0071 | 0.0634 | 1.07 |
| VisA | E@8 | 8,388,104 | 0.4540 ± 0.0056 | 0.9419 ± 0.0084 | 0.0153 | 2.03 |
| VisA | E@16 | 16,776,200 | 0.4651 ± 0.0045 | 0.9627 ± 0.0013 | 0.0103 | 4.11 |
| VisA | E@32 | 33,553,928 | 0.4784 ± 0.0021 | 0.9696 ± 0.0017 | 0.0051 | 8.55 |
| VisA | U | 106,638,344 | 0.4881 ± 0.0012 | 0.9738 ± 0.0013 | 0.0033 | 20.20 |
| MPDD | E@4 | 4,193,288 | 0.4618 ± 0.0097 | 0.8986 ± 0.0182 | 0.0348 | 1.16 |
| MPDD | E@8 | 8,388,104 | 0.5099 ± 0.0054 | 0.9455 ± 0.0091 | 0.0114 | 2.01 |
| MPDD | E@16 / E@32 / U | 13,057,544 | 0.5175 ± 0.0053 | 0.9616 ± 0.0015 | 0.0010 | 2.6–2.7 |

- On MPDD, U needs only 13.06 MB. At 16 and 32 MiB the budget never binds, so E@16 = E@32 = U exactly. This is a consistency check that holds.
- The curve is monotone on every dataset.
- E@32 reaches 99.7 % (MVTec) and 98.0 % (VisA) of U's pixel AP, at 73 % and 31 % of U's bytes.
- Latency scales with the number of keys: E@8 is 4.6× (MVTec) and 9.9× (VisA) faster than U.

---

## 7. Robustness over orders × seeds (A2; 3 × 3 = 9 cells per dataset, mean ± std)

| Dataset | Arm | n | Pixel AP | Image AUROC | FM | acq AP |
|---|---|---|---|---|---|---|
| MVTec | U | 9 | 0.6342 ± 0.0004 | 0.9954 ± 0.0003 | 0.0047 ± 0.0047 | 0.6315 ± 0.0032 |
| MVTec | **E@8** | 9 | **0.5944 ± 0.0063** | **0.9832 ± 0.0029** | 0.0314 ± 0.0075 | 0.6157 ± 0.0050 |
| VisA | U | 9 | 0.4881 ± 0.0010 | 0.9738 ± 0.0011 | 0.0036 ± 0.0010 | 0.4806 ± 0.0043 |
| VisA | **E@8** | 9 | **0.4505 ± 0.0052** | **0.9457 ± 0.0056** | 0.0225 ± 0.0091 | 0.4635 ± 0.0068 |
| MPDD | U | 9 | 0.5175 ± 0.0046 | 0.9616 ± 0.0013 | 0.0015 ± 0.0017 | 0.5169 ± 0.0048 |
| MPDD | **E@8** | 9 | **0.5102 ± 0.0078** | **0.9473 ± 0.0070** | 0.0104 ± 0.0025 | 0.5165 ± 0.0059 |

**E@8 pixel AP per cell** (order|seed: a|2711, a|2712, a|2713, b|2711, b|2712, b|2713, c|2711, c|2712, c|2713):

| Dataset | Values | Range |
|---|---|---|
| MVTec | .5927 .6016 .5895 .5930 .5925 .5879 .5974 .6065 .5889 | 0.588–0.607 |
| VisA | .4603 .4522 .4494 .4489 .4497 .4409 .4475 .4520 .4535 | 0.441–0.460 |
| MPDD | .5082 .5160 .5056 .5030 .5164 .5028 .5020 .5244 .5131 | 0.502–0.524 |

**Notes.**
- U's final state is order-invariant, so its std reflects the coreset seed only.
- Prequential AP depends on the order's composition. MPDD's std is 0.168, so it is not averaged across orders in the paper; it is in RESULTS.json.
- Order-a seed 2711 (§3) lies inside the 9-cell range on every dataset.
- The seed-2711 cells of orders b and c are the reused MULTIRATE-01 cells.

---

## 8. Matched backbone: ViT-B/8 IN-21k at 224 (B; order a, seed 2711)

**Contract.** Documented in `exps/g3e_paper_01/vitb8.py` before running.
- Model: timm `vit_base_patch8_224` augreg in21k, local checkpoint `checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth` (sha256 755e9f5f…a415). It was already on disk, so nothing was downloaded.
- Features: output of block 9, before the final norm, with CLS dropped: 784 = 28×28 tokens × 768.
- Input: PIL RGB, bilinear direct resize to 224 (the same full-field geometry, so the A1 readout is unchanged).
- Normalisation: ImageNet mean/std. This is the repository's CADIC contract. timm's augreg default would be 0.5/0.5; this is a recorded ambiguity and was not tuned.
- Tokens: unit-L2 fp16.
- The cache is new, under `/mnt/k/NLCAD/cache/g3e_paper_01/vitb8/` (110 files, with per-file sha in a json). The canonical caches were not touched.
- Test row order was checked against the DINOv2 benchmark rows.

| Dataset | Arm | Pixel AP [CI95] | Image AUROC | FM | acq AP | Bytes | ms/img |
|---|---|---|---|---|---|---|---|
| MVTec | U | 0.5748 [0.5532, 0.5936] | 0.9453 | 0.0050 | 0.5755 | 46,070,792 | 9.86 |
| MVTec | **E@8** | **0.5018** [0.4789, 0.5207] | **0.8869** | 0.0630 | 0.5490 | 8,388,104 | 2.17 |
| MVTec | F | 0.4232 [0.3987, 0.4419] | 0.9018 | 0.1246 | 0.5349 | 8,388,096 | 2.40 |
| VisA | U | 0.2910 [0.2679, 0.3125] | 0.8783 | 0.0023 | 0.2741 | 106,638,344 | 20.34 |
| VisA | **E@8** | **0.2356** [0.2105, 0.2541] | **0.7644** | 0.0341 | 0.2420 | 8,388,104 | 2.12 |
| VisA | F | 0.2176 [0.1936, 0.2361] | 0.7863 | 0.0330 | 0.2378 | 8,388,096 | 2.30 |

- **E − F pixel AP (paired):** MVTec +0.0786 [0.0708, 0.0888]; VisA +0.0179 [0.0098, 0.0249].
- **U − E pixel AP (paired):** MVTec +0.0731 [0.0655, 0.0817]; VisA +0.0554 [0.0471, 0.0674].

**For reference only:** CADIC (ViT-B/8) as reported by its authors. This is not a matched run, and the protocol may differ.
- 10k patches (≈ 30.7 MB): MVTec 0.972 I-AUROC / 0.584 P-AUPR; VisA 0.891 / 0.438.
- 2.5k patches (≈ 7.7 MB): MVTec 0.955 / 0.559; VisA 0.828 / 0.415.
- Source: `agents/reports/lit_01_cad_benchmark_accuracy_memory_time.md`.

**Reading (facts only).**
- With this backbone contract, U and E@8 are below CADIC's reported numbers on both datasets.
- The advantage over CADIC in the main comparison table therefore depends on the DINOv2 at 392 front end, not on the memory rule alone.
- The slow-level effect (E > F) holds on both backbones.
- The normalisation ambiguity above is one untested possible contributor to the low ViT-B/8 numbers.

---

## 9. Per-class pixel AP (A5; final model, order a, seed 2711)

Columns: U final AP, E@8 final AP, U AUROC, E@8 AUROC, E@8 acq AP, E@8 FM. The last task in the order has no FM (—).

### MVTec

| Class | U AP | E AP | U AUROC | E AUROC | E acq AP | E FM |
|---|---|---|---|---|---|---|
| bottle | 0.855 | 0.833 | 1.000 | 1.000 | 0.842 | 0.019 |
| cable | 0.497 | 0.485 | 0.991 | 0.978 | 0.487 | 0.061 |
| capsule | 0.590 | 0.472 | 0.978 | 0.897 | 0.592 | 0.120 |
| carpet | 0.727 | 0.703 | 1.000 | 1.000 | 0.702 | 0.032 |
| grid | 0.494 | 0.485 | 1.000 | 1.000 | 0.509 | 0.024 |
| hazelnut | 0.808 | 0.756 | 1.000 | 1.000 | 0.764 | 0.011 |
| leather | 0.468 | 0.427 | 1.000 | 1.000 | 0.449 | 0.054 |
| metal_nut | 0.723 | 0.637 | 1.000 | 1.000 | 0.680 | 0.043 |
| pill | 0.575 | 0.510 | 0.994 | 0.989 | 0.541 | 0.038 |
| screw | 0.600 | 0.491 | 0.983 | 0.887 | 0.544 | 0.053 |
| tile | 0.727 | 0.725 | 1.000 | 1.000 | 0.707 | −0.001 |
| toothbrush | 0.669 | 0.661 | 1.000 | 1.000 | 0.674 | 0.013 |
| transistor | 0.509 | 0.449 | 0.991 | 0.996 | 0.451 | 0.002 |
| wood | 0.679 | 0.669 | 0.992 | 0.994 | 0.667 | −0.003 |
| zipper | 0.601 | 0.586 | 1.000 | 0.998 | 0.586 | — |

### VisA

| Class | U AP | E AP | U AUROC | E AUROC | E acq AP | E FM |
|---|---|---|---|---|---|---|
| candle | 0.444 | 0.422 | 0.977 | 0.906 | 0.426 | 0.008 |
| capsules | 0.628 | 0.606 | 0.967 | 0.937 | 0.576 | −0.002 |
| cashew | 0.692 | 0.652 | 0.968 | 0.938 | 0.671 | 0.030 |
| chewinggum | 0.702 | 0.671 | 0.990 | 0.994 | 0.682 | 0.018 |
| fryum | 0.467 | 0.411 | 0.958 | 0.905 | 0.431 | 0.031 |
| macaroni1 | 0.157 | 0.182 | 0.933 | 0.867 | 0.165 | −0.000 |
| macaroni2 | 0.137 | 0.094 | 0.946 | 0.839 | 0.110 | 0.020 |
| pcb1 | 0.798 | 0.784 | 0.996 | 0.955 | 0.779 | 0.004 |
| pcb2 | 0.427 | 0.417 | 0.982 | 0.943 | 0.421 | 0.004 |
| pcb3 | 0.444 | 0.391 | 0.993 | 0.971 | 0.404 | 0.013 |
| pcb4 | 0.419 | 0.375 | 0.997 | 0.992 | 0.381 | 0.006 |
| pipe_fryum | 0.527 | 0.520 | 0.989 | 0.974 | 0.520 | — |

### MPDD

| Class | U AP | E AP | U AUROC | E AUROC | E acq AP | E FM |
|---|---|---|---|---|---|---|
| bracket_black | 0.197 | 0.182 | 0.872 | 0.842 | 0.202 | 0.020 |
| bracket_brown | 0.407 | 0.383 | 0.975 | 0.922 | 0.400 | 0.032 |
| bracket_white | 0.081 | 0.061 | 0.982 | 0.932 | 0.082 | 0.022 |
| connector | 0.758 | 0.767 | 1.000 | 1.000 | 0.756 | −0.011 |
| metal_plate | 0.885 | 0.884 | 1.000 | 1.000 | 0.885 | 0.001 |
| tubes | 0.777 | 0.772 | 0.947 | 0.942 | 0.772 | — |

**Largest E losses versus U:** MVTec capsule (−0.118 AP, FM 0.120), screw (−0.109) and metal_nut (−0.086). On these classes the image AUROC also drops: capsule 0.897, screw 0.887.

---

---

## 10. Storage

### 10.1 What RIGEL-CAD keeps between tasks (8 MiB budget)

| Component | Shape and dtype | Counted bytes | In memory / on disk |
|---|---|---|---|
| Fast keys K | 3,925 × 768, fp16 | 6,028,800 | 6,028,800 |
| Slow statistic S | 768 × 768 (symmetric) | 2,359,296 (fp32) | 4,718,592 (kept in fp64 for exact accumulation) |
| Count n | scalar | 8 | 8 |
| **Total** | | **8,388,104 = 8 MiB − 504 B** | 10,747,400 |
| `.rigel` file (`torch.save`, with config and task names) | | | **10,749,509** (AD 2), 10,749,997 (MPDD) |

**Notes.**
- **Budget convention.** The budget counts S at fp32. This is the research protocol, and it is the same for every bounded arm.
- **Stored size.** The file stores S in fp64, so it is 10.75 MB. Storing S in fp32 would bring the file to about 8.39 MB.
- **Numerical caveat.** W rebuilt from an fp32 S may differ in the last bits from the in-memory run. This option was not applied to the frozen release; it is an owner decision, listed in §14.
- **Read-out cache.** W (768 × 768 fp32, 2.36 MB) and the whitened keys K·W are recomputed from the state and are not persistent.
- **Not counted.** The frozen DINOv2-reg ViT-B/14 is shared and not counted: 86,583,552 parameters, 346 MB fp32 checkpoint. This is the same convention as the CAD literature.
- **Constant over tasks.** The state is the same 8,388,104 B from task 2 on, on every dataset. Task 1 uses fewer bytes when its coreset is smaller than the cap.

### 10.2 Unbounded reference and budget sweep

| Dataset | U (all per-task coresets) | RIGEL-CAD 4 / 8 / 16 / 32 MiB |
|---|---|---|
| MVTec AD (15 tasks) | 46,070,792 | 4,193,288 / 8,388,104 / 16,776,200 / 33,553,928 |
| VisA (12) | 106,638,344 | same |
| MPDD (6) | 13,057,544 | same (16 and 32 MiB do not bind: = U) |
| MVTec AD 2 (8) | 32,808,968 | 8,388,104 at 8 MiB |

U grows linearly with the data: about 3.1 MB per MVTec task and 8.9 MB per VisA task. RIGEL-CAD is constant.

### 10.3 Persistent continual-learning state in the literature

Labels: [P] = paper-reported; [I] = inferred from tensor shapes stated in the paper (fp32 unless stated). The source is `agents/reports/lit_01_cad_benchmark_accuracy_memory_time.md`. The backbone is not counted for any row.

| Method | Persistent state (MVTec, 15 tasks) | Grows with tasks | Backbone |
|---|---|---|---|
| **RIGEL-CAD** | **8.39 MB** (counted), 10.75 MB file | No | DINOv2-reg ViT-B/14 (frozen) |
| ReplayCAD | 3.0 MB [P], plus a 1.07 B-parameter diffusion model | Yes, about 0.2 MB/class | LDM / SD 1.5 |
| CADIC 2.5k / 10k / 20k | 7.68 / 30.72 / 61.4 MB [I] | No | ViT-B/8 IN-21k |
| GCR | 9.0 MB [I] | Yes, 0.6 MB/class | OpenCLIP ViT-B/16 |
| ONER | about 9.1 MB [I] | Yes | ViT-B/16 |
| UCAD | 23.3 MB [P] | Pre-allocated | ViT-B/16 (+ SAM in training) |
| DNE | about 35 MB [I] | Yes | ViT-B/16 (fine-tuned) |
| ContCore | about 82–123 MB [I] | No | WRN50 |
| DINOSaur | about 107 MB [P/I] | Yes | DINOv3 ViT-S/16 |
| DeCoFlow | about 136 MB of adapters [I] | Yes | WRN50 / ViT-B/16 |
| CDAD | 3.5 GB [P] | Yes | SD-based diffusion |

---

## 11. Speed and GPU memory (measured 2026-11-26, from raw images)

**Run.** `results/rigel_cad/bench/run_bench.sh` → `python -m rigel_cad bench --dataset {mpdd,mvtec,visa,ad2}`, module `rigel_cad/bench.py`.
- No token cache is used. Each task's training images are decoded and encoded once, and the same tokens are fitted by RIGEL-CAD, F and U.
- Every phase is CUDA-synchronised and timed separately.
- Inference uses 100 test images drawn evenly over the categories (seeded), scored one at a time end to end, with the final 8 MiB state resident on the GPU.
- Times are p50 per image.

**Machine.**
- Laptop: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), 12 CPU threads.
- Software: WSL2 (Linux 6.6), torch 2.10.0+cu126, Python 3.11.
- Backbone in fp32, batch 8 for training, batch 1 for inference.
- Images are read from a Windows drive through WSL. That makes **decoding I/O-bound and variable**: it is the largest and noisiest term. Use the "without decode" columns to compare compute.

**Training.** The whole continual stream: decode, then backbone, then RIGEL-CAD `fit_task` per task. There is no gradient training and no epochs.

| Dataset | Tasks | Train images | Decode (s) | Backbone (s) | img/s backbone | Fit RIGEL-CAD (s) | Fit / task (s) | Fit F / U (s) | **RIGEL-CAD total** (s) | Total w/o decode (s) | Peak GPU train (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MVTec AD | 15 | 3,629 | 264 | 112 | 32.5 | 103.7 | 6.91 | 105.4 / 95.6 | **480** | 215 | 1.96 |
| VisA | 12 | 8,659 | 268 | 268 | 32.4 | 668.9 | 55.74 | 670.4 / 660.6 | **1204** | 937 | 4.01 |
| MPDD | 6 | 888 | 45 | 27 | 32.5 | 19.4 | 3.23 | 19.3 / 16.9 | **92** | 47 | 1.53 |
| MVTec AD 2 | 8 | 2,528 | 912 | 114 | 22.1 | 90.8 | 11.35 | 91.3 / 86.2 | **1117** | 205 | 2.12 |

**Inference.** Batch 1, end to end, p50 ms per image. The map and score step is the 392 px Gaussian-smoothed map plus the top-1 % image score, on the CPU.

| Dataset | Decode | Backbone | Head RIGEL-CAD | Map + score | **Total** | Compute only (no decode) | Head F / U | Keys E / F / U | Peak GPU infer (GB) |
|---|---|---|---|---|---|---|---|---|---|
| MVTec AD | 87.1 | 33.0 | 2.84 | 13.8 | **136.6** | 49.7 (20 img/s) | 2.72 / 18.25 | 3,925 / 5,461 / 28,458 | 0.79 |
| VisA | 37.6 | 33.5 | 2.84 | 13.9 | **88.6** | 50.2 (20 img/s) | 2.72 / 41.91 | 3,925 / 5,461 / 67,890 | 1.33 |
| MPDD | 49.4 | 33.4 | 2.87 | 13.9 | **99.3** | 50.1 (20 img/s) | 2.69 / 4.43 | 3,925 / 5,461 / 6,965 | 0.50 |
| MVTec AD 2 | 335.6 | 56.5 | 4.37 | 13.4 | **400.8** | 74.3 (13 img/s) | 3.58 / 18.26 | 3,925 / 5,461 / 19,824 | 0.67 |

**Reading.**
- **The backbone dominates compute.** DINOv2 at 392 px runs at about 32 img/s for training (batch 8) and 33 ms per image at batch 1.
- **The RIGEL-CAD head is cheap.** It takes 2.8–2.9 ms per image on MVTec AD, VisA and MPDD (4.4 ms on AD 2, see below), because the state is a constant 3,925 keys. Unbounded U grows with the data: 4.4 ms (MPDD), 18.3 ms (MVTec) and 41.9 ms (VisA). The same holds for its memory: 13 / 46 / 107 MB.
- **Compute-only latency is about 50 ms per image (about 20 img/s)** on MVTec AD, VisA and MPDD. Decoding adds 38–87 ms on this disk setup.
- **MVTec AD 2 is slower:** 74 ms compute and 336 ms decode per image. Its images are large (up to several megapixels), so decoding and resizing to 392 px dominate.
  - Its backbone ran at 22 img/s instead of 32, and the head at 4.4 ms instead of 2.8, with the same 392 px input and the same key count.
  - AD 2 ran last, after about 80 minutes of continuous GPU load on a laptop. These AD 2 compute numbers are therefore probably inflated by GPU clock throttling. This was not separately verified.
- **The continual update is closed-form.** `fit_task` costs 3.2 s (MPDD), 6.9 s (MVTec), 11.4 s (MVTec AD 2) and 55.7 s (VisA) per task. It scales with the square of the images per task, because the 1 % coreset makes k proportional to N.
- **The bound costs little update time.** At 8 MiB, the fit takes about the same time as F (±2 %), and 1–13 % more than U. U skips the boundary consolidation, a k-centre over ≤ 3,925 + k keys.
- **Peak GPU memory.** Training needs 1.5–4.0 GB, driven by one task's fp32 tokens; VisA's largest tasks have 900 images. Inference needs 0.5–1.3 GB, including the 0.35 GB backbone.
- **Agreement with locked timings.** The research timings of G3E-PAPER-01 (head only, cached tokens) agree: update 107.4 / 677.4 / 20.3 s and 2.0–2.3 ms per image for MVTec / VisA / MPDD.

**Incident (benchmark only).** The first benchmark version kept each task's tokens on the GPU between encoding and fitting. On VisA candle (900 images), the 6 GB GPU spilled into shared system memory, and the fit took 289 s instead of 79 s. That run was stopped. The fixed version moves tokens to the CPU, as the token store does, and re-ran every dataset. The first version's MPDD and MVTec files are kept in `results/rigel_cad/bench/superseded_gpu_tokens/`. No accuracy number is involved.

**Training and inference cost in the literature.** The hardware differs, so this is context only.

| Method | Train (whole MVTec stream) | Test per image | Hardware | Source |
|---|---|---|---|---|
| **RIGEL-CAD 8 MiB** | **215 s** compute (480 s with decode) | **≈ 50 ms** compute (2.8 ms head) | RTX 4050 Laptop | this report |
| CADIC 2.5k / 10k | 63 s / 2,503 s | — | RTX 4070 | [P] Table VII |
| ContCore | 754 s | 39.1 ms | same GPU for all rows of the ContCore table | [P] |
| UCAD | 786 s | 15.3 ms | same as ContCore | [R:ContCore] |
| DNE | 1,233 s | 21.3 ms | same as ContCore | [R:ContCore] |
| IUF | 39,192 s (10.9 h) | 80.8 ms | same as ContCore | [R:ContCore] |
| CDAD | 281,434 s (78.2 h) | 2,694 ms | same as ContCore | [R:ContCore] |

---

## 12. Position against published continual-AD methods (MVTec AD 1×15, VisA 1×12; final model, task-agnostic)

**This is not a matched comparison.**
- Published numbers use their own backbones and resolutions. RIGEL-CAD uses DINOv2-reg ViT-B/14 at 392 px.
- With a matched ViT-B/8 at 224 px, RIGEL-CAD is **below** CADIC (§8).
- [P] means paper-reported. [R:X] means re-run and reported by X.
- Source: `agents/reports/lit_01_cad_benchmark_accuracy_memory_time.md`.

| Method | Backbone | State | MVTec I-AUROC | MVTec P-AUPR | VisA I-AUROC | VisA P-AUPR | Source |
|---|---|---|---|---|---|---|---|
| **RIGEL-CAD** | DINOv2-reg B/14 @392 | 8.39 MB | **0.983** | **0.593** | **0.935** | **0.460** | this report |
| RIGEL-CAD-∞ (reference) | same | 46 / 107 MB | 0.995 | 0.635 | 0.975 | 0.487 | this report |
| RIGEL-CAD, matched backbone | ViT-B/8 @224 | 8.39 MB | 0.887 | 0.502 | 0.764 | 0.236 | this report §8 |
| CADIC 10k | ViT-B/8 | 30.7 MB | 0.972 | 0.584 | 0.891 | 0.438 | [P] |
| CADIC 2.5k | ViT-B/8 | 7.7 MB | 0.955 | 0.559 | 0.828 | 0.415 | [P] |
| CMPMB | CLIP | about 23 MB | 0.974 | 0.604 | 0.901 | 0.365 | [P], not reproduced |
| DeCoFlow | WRN50 / ViT-B/16 | about 136 MB | 0.984 | 0.582 | 0.930 | 0.370 | [P] |
| GCR | OpenCLIP B/16 | 9.0 MB | 0.978 | 0.513 | 0.893 | 0.302 | [P] |
| ReplayCAD | + diffusion | 3.0 MB + LDM | 0.948 | 0.537 | 0.903 | 0.415 | [P] |
| UCAD | ViT-B/16 | 23.3 MB | 0.930 | 0.456 | 0.874 | 0.300 | [P] |
| Joint PatchCore (upper bound, not CL) | WRN50 | — | 0.978 | 0.594 | 0.916 | 0.440 | [R:CADIC] |

**Reading.**
- With its backbone, RIGEL-CAD at 8.39 MB has the highest VisA P-AUPR (0.460 vs CADIC 0.438) and VisA I-AUROC (0.935 vs DeCoFlow 0.930) of the published rows. The RIGEL-CAD-∞ reference row is not counted here.
- On MVTec AD:
  - P-AUPR is second to CMPMB (0.604, not reproduced), and level with joint PatchCore (0.594);
  - I-AUROC is level with DeCoFlow (0.983 vs 0.984).
- **The backbone is a stated condition of every such claim.** With matched ViT-B/8, CADIC is higher on every number.

---

## 13. Claim audit (what the paper may say)

| Claim | Status | Evidence |
|---|---|---|
| NL contribution: a two-level, two-frequency memory whose slow level beats spending the same bytes on keys | **Can claim** (owner decision 2026-11-25) | RIGEL-CAD − F > 0 with CI on 3/3 datasets (DINOv2) and 2/2 (ViT-B/8); +0.066 AU-PRO 0.05 on AD 2 (6/8 scenarios, no CI). Mapping to NL Def. 2–3 is PAPER_INFERRED (closed-form slow level) |
| 8 MiB keeps 93–98 % of unbounded pixel AP at 8–64 % of its bytes | **Can claim** | §3; U − RIGEL-CAD > 0 with CI on all 3 datasets |
| On MVTec AD 2, 8 MiB matches the 32.8 MB unbounded memory on AU-PRO 0.05 | **Can claim, local test_public, no CI** | §4.2: 0.359 vs 0.358 (AU-PRO 0.30: 0.592 vs 0.607) |
| Constant state, independent of the number of tasks | **Can claim** | §10 |
| Robust over orders and seeds | **Can claim** | §7: std ≤ 0.008 AP over 9 cells |
| Smooth budget curve; 32 MiB ≈ unbounded | **Can claim** | §6 |
| Training-free, fast continual update and fast inference | **Can claim, with the machine stated** | §11 |
| Higher raw numbers than published CAD methods | **Only with the backbone stated** | §12; below CADIC with matched ViT-B/8 (§8) |
| AD 2: above the dataset authors' 256 px baselines | **Can claim with qualifiers** | §4.1: one submission; 392 px DINOv2 vs 256 px per-category models; far below the leaderboard top; not SOTA |
| Task-boundary k-centre is the best selection rule | **Cannot claim** | §5: hindsight Q is higher on MVTec pixel AP and VisA image AUROC; GG-01 (whitened k-centre) was rejected at DEV |
| Image AUROC parity with unbounded | **Cannot claim** | VisA 0.935 vs 0.975 |
| Good segmentation thresholds | **Cannot claim** | §4.2: SegF1 is threshold-limited (rice) |

---

## 14. Known limitations, errata and open items

| ID | Item | Status |
|---|---|---|
| E1 | MVTec AD 2 test_public masks (stem lookup) | **Fixed 2026-11-26**, all numbers re-run (§4.3) |
| L1 | The `.rigel` file stores S in fp64 (10.75 MB vs 8.39 MB counted) | Open, owner decision: store S as fp32, or state the convention (§10.1) |
| L2 | Backbone dependence: matched ViT-B/8 is below CADIC | Stated in every comparison. The ViT-B/8 normalisation ambiguity (ImageNet vs 0.5/0.5) is untested |
| L3 | Hindsight selection Q > task-boundary k-centre on MVTec | Documented. GG-01 (geometry-guided consolidation) was rejected at DEV on 2026-11-26 (−0.049 AP). A ContCore-style expansion is a lead only (+0.007, CI > 0, without whitening) |
| L4 | SegF1 limited by the untuned D-THR threshold | Open. A label-free threshold rule is a possible improvement |
| L5 | Weak classes: capsule, screw, metal_nut (largest loss vs U, MVTec); macaroni1/2 (VisA, AP < 0.19); bracket_white, bracket_black (MPDD); can (AD 2) | Documented (§9, §4.2) |
| L6 | No CI on AD 2 or on the per-class numbers | The bootstrap is registered for the order-a dataset means only |
| L7 | Speed numbers are for one laptop (RTX 4050 6 GB, WSL2, data on a Windows drive) | Image decoding is I/O-bound there (§11) |
| T1 | `test_equivalence.py` failed when run as one session: the research encoder loader switches the process to deterministic algorithms. That is test pollution, not a code difference; the test passed alone | **Fixed 2026-11-26** (fixture restores torch globals). 6/6 pass in one session |

---

## 15. Reproducibility

The environment is `~/miniconda3/envs/nested_learning`. MVTec AD 2 needs `tifffile` only for the submission files.

```bash
# tests
python -B -m pytest -q rigel_cad/tests/test_unit.py               # 11 CPU tests
python -B -m pytest -q rigel_cad/tests/test_equivalence.py        # bitwise vs frozen research code (CUDA + caches)
# accuracy (MVTec / VisA / MPDD), order a, seed 2711, 8 MiB
python -m rigel_cad stream --dataset mvtec --data-root data/mvtec --order a --seed 2711 --budget-mib 8
# MVTec AD 2, corrected masks, per-stage forgetting (+ controls)
bash results/rigel_cad/ad2/run_ad2_maskfix.sh
bash results/rigel_cad/ad2/run_ad2_controls.sh
# speed and memory
bash results/rigel_cad/bench/run_bench.sh
# all paper tables under RIGEL-CAD names (from the locked research results)
python -m rigel_cad.paper_tables
```

| Result | File |
|---|---|
| MVTec AD, VisA and MPDD tables (§3, §5–§9) | `results/hope_cad/g3e_paper_01/RESULTS.json` (sha256 f450d20b9cd59e5e…); `results/rigel_cad/paper/` |
| MVTec AD 2, corrected (§4.2) | `results/rigel_cad/ad2/RESULTS_ad2.json`, `run_ad2_maskfix.log` |
| MVTec AD 2 controls U and F | `results/rigel_cad/ad2/controls/{U,F}/RESULTS_ad2.json`, `run_ad2_controls.log` |
| MVTec AD 2 before the fix | `results/rigel_cad/ad2/RESULTS_ad2_STEM_MASK_DEFECT.json`, `run_ad2.log` |
| Speed and memory (§11) | `results/rigel_cad/bench/BENCH_{mpdd,mvtec,visa,ad2}.json`, `run_bench.log` |
| Server result (§4.1) | owner-reported; `agents/reports/nl03_multirate_01.md` §10.5 |

---

## Handoff

**Task:** RIGEL-CAD release report. Owner request, 2026-11-26: fix the MVTec AD 2 mask defect, re-run it, and write one release report with every metric, storage figure and speed figure.

**Status:** DONE.

**Last agent:** Codex (role held by Claude Code Opus).

**Next agent:** User.

**Branch / HEAD:** dikhang_space / 54c86e0. The changes are staged in the index, **not committed and not pushed**; commit and push wait for the owner.

**Completed work:**
- Fixed `ad2_mask_of` (directory label, strict) and added a mask audit with an assertion in `run_ad2`.
- Re-ran MVTec AD 2 (RIGEL-CAD per stage, plus U and F controls) through the package.
- Added `ad2_stream_summary` (FM and acquisition AU-PRO).
- Added the `--no-submission` option.
- Added the `bench` command and module, and benchmarked 4 datasets from raw images.
- Wrote this report.
- Added erratum banners to the earlier handoff documents (`docs/handoff/RIGEL_CAD_FINAL_RESULTS_HANDOFF.md`, `RIGEL_CAD_CODE_REPORT.md`) and the internal reports (`rigel_cad_01.md`, `nl03_multirate_01.md`).
- Updated `rigel_cad/README.md`.

**Changed files:**
- `rigel_cad/data.py`, `pipeline.py`, `cli.py`, `bench.py` (new), `README.md`.
- `rigel_cad/tests/test_unit.py` (+2 tests).
- `rigel_cad/tests/test_equivalence.py` (the legacy lookup is kept inside the test only; a new fixture restores the torch globals).
- `results/rigel_cad/ad2/`:
  - `RESULTS_ad2.json` (corrected);
  - `RESULTS_ad2_STEM_MASK_DEFECT.json` (renamed, the pre-fix file);
  - `run_ad2_maskfix.{sh,log}`, `run_ad2_controls.{sh,log}`, `controls/{U,F}/`.
- `results/rigel_cad/bench/**`.
- `docs/handoff/RIGEL_CAD_RELEASE_REPORT.md` (a copy of this report).
- `exps/` is untouched. The canonical caches are untouched: they were read only, and the new tokens were written only under `/mnt/k/NLCAD/cache/rigel_cad`.

**Verification / Tests:**
- `python -B -m pytest -q rigel_cad/tests/test_unit.py`: 11 passed (13 s).
- `python -B -m pytest -q rigel_cad/tests/test_equivalence.py`: 6 passed in one session (290 s). It first failed in one session through test pollution (T1, §14); a fixture fixed that.
- AD 2 re-run θ is bitwise equal to the pre-fix run and to the locked D-THR (E and U).
- Mask audit: 0 of 379 good images and 705 of 705 bad images carry defect pixels.
- Benchmark fit times agree with the locked research timings (MVTec 103.7 vs 107.4 s; VisA 668.9 vs 677.4 s).

**Unresolved issues:** L1 (fp32 S on disk, owner decision), L4 (threshold rule), and L2 (ViT-B/8 normalisation), all in §14. Pushing is left to the owner.

**Exact next action:** the owner reviews the report and the staged changes, and decides whether to commit and push them, together with the GG-01 commits daef1dc and 54c86e0. The paper chapters use §0, §3–§13. The next agent may proceed only on a new owner task.

