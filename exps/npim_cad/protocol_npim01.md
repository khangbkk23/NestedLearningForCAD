# exps/npim_cad/protocol_npim01.md

# EXP-NPIM-01 — frozen development-evaluation protocol

Status: **FROZEN BEFORE THE FIRST NPIM-01 DEV IMAGE DECODE.** This protocol is additive to `exps/npim_cad/protocol.md` (EXP-NPIM-00) and does not modify any NPIM-00 artifact.

## 1. Provenance and correction

The authoritative NPIM-00 training stream is the resolved P1/P0c stream:

```
events   1–100 : bottle   (100 normal images)
events 101–200 : carpet   (100 normal images)
events 201–300 : hazelnut (100 normal images)
events 301–350 : bottle   ( 50 new bottle-return normal images)
```

This is **bottle 150 total = 100 initial + 50 bottle-return**, carpet 100, hazelnut 100. It is **not** a contiguous `150 bottle → 100 carpet → 100 hazelnut` stream. The earlier NPIM-01 task text said “150/100/100, no bottle return”; that was a task-specification error and is corrected here and in `agents/tasks/current_task.md`. No NPIM-00 state, report, cache or protocol is changed.

| Item | Pin |
|---|---|
| Branch / HEAD | `dikhang_space` / `293dc2e46c36f088e6681f7be00c047cae7bc394` |
| P1 resolved protocol | `results/hope_cad/task_p/p1/protocol_resolved.json`, SHA-256 `11c1cc03b20cb1f79a6e8b7e472bb6f44758b43046b61043368ea11b4ce1e08f` |
| P0c resolved protocol | `results/hope_cad/task_p/p0c/protocol_resolved.json`, SHA-256 `0090f697518587e51eda91a3bd5def44b5ceb2c572b49eb75f5b9a0dfd4fb182` |
| NPIM-00 pilot manifest | SHA-256 `e310d1e4edc4bf16b43dc5ecc0342fac72eb02f9fd6eed414ec5047cd6ab486e` |
| NPIM-00 event log | SHA-256 `984bd53a78ce7e2e3c67b0d93eadc9a0c97580b9baf6ab98ae9877e3af07c9aa` |
| Frozen encoder | `checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth`, SHA-256 `755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415`, 343,282,995 B |

The 12 NPIM-00 C1/C2/C3 boundary states at 100/200/300/350 are reused exactly, with their file SHA-256 and reloaded `state_hash` pinned by `npim01_contract.py`. **No retraining, no reordering, no hyperparameter change, no checkpoint modification.**

## 2. Admitted development cohort

Only the 60 records in `p1/protocol_resolved.json["evaluations"]` (role `development_eval`) are in scope:

* 3 categories × (10 normal + 10 anomalous) = 60 unique images; 30 normal, 30 anomalous.
* These are the historical **exposed P1 development** identities; they are development evidence, not independent confirmation.
* The P1 `sealed_identity_paths` (next-10 per class) are never opened, hashed, decoded, extracted or scored.
* Cohort identity digest (canonical 60-record JSON SHA-256): `63d1dbb48e9b6dd4665e50a0fa2702616834adddf95fd9889323804c21875447`.
* Writer identity digest (350 records): `9549f24aae822ea8cccdbe63dd9731790633cf5eba80b37506c11c2944df3c24`.
* Writer/DEV path and content sets are disjoint; DEV/sealed identity sets are disjoint.

No `test/` directory listing or discovery is used: the evaluator iterates only the pinned 60 records.

## 3. Frozen scoring

* Frozen ViT-B/8; `Z` at `blocks[0]` pre-attention input, `Y` at block index 8; CLS excluded; the NPIM-00 architecture/preprocessing contract is unchanged.
* Context keys from `Z` only, `N4` + leave-one-out global (0.5/0.5), fixed projection seed 20261021, gamma 0, epsilon 1e-8.
* C1: single level, r=145, period 1, lambda 0.1.
* C2: three levels, r=64, periods (1,1,1), lambdas (0.1, 1.0, 10.0).
* C3: three levels, r=64, periods (1,16,128), lambdas (0.1, 1.0, 10.0).
* Fixed equal mixture over ready levels; patch score = mean squared residual over 768 dimensions; image score = max patch score; diagonal covariance disabled.
* Native map: `28×28 → (H_orig,W_orig)` bilinear, `align_corners=False`, `antialias=False`; native mask, no resize.
* Pixel metric: tie-aware native Pixel AP (non-interpolated), not trapezoidal PR area. Image metric: tie-aware Image AUROC. Normal diagnostics: image-score mean/p95, hard-normal cases.
* No per-category calibration, threshold selection, score inversion or category input to the scorer.

## 4. Boundaries and admitted evaluation

| Boundary | Admitted categories | Pairs (images × arms) |
|---:|---|---:|
| 100 | bottle | 20 × 3 = 60 |
| 200 | bottle, carpet | 40 × 3 = 120 |
| 300 | bottle, carpet, hazelnut | 60 × 3 = 180 |
| 350 | bottle, carpet, hazelnut | 60 × 3 = 180 |

Total: **180 boundary-image pairs × 3 arms = 540 score maps**. The same 60 images and the same 12 states are used for every arm. No unseen category is scored before its boundary. C3's slow level is not ready at 100 (first commit at 128), so C3 vs C2 at 100 is not a clean schedule comparison; report this cold-start difference and emphasize 200/300/350 for the scheduling contrast.

## 5. E1/E2 isolation and seal

**E1 (prediction-only):** verify the contract, encode each of the 60 images at most once to obtain `Z` and `Y`, score only admitted categories, assert 540 finite maps and unchanged state/RNG hashes, then atomically write `predictions.pt` and `predictions_seal.json` with SHA-256. **No numeric mask is read in E1.** The prediction payload contains no labels or masks.

**E2 (metrics-only):** require and verify the E1 seal SHA, then open native masks for the same 60 identities, verify mask SHA/geometry/non-emptiness, map scores to native geometry, and compute AP/AUROC and diagnostics. A missing or mismatched seal aborts before any mask read.

## 6. Pre-registered decision rule

Before viewing anomaly results (recorded here and in the task):

* Primary contrast: **C3 − C2** (isolates update-frequency scheduling; equal architecture/rank/tensor count).
* Secondary contrast: **C3 − C1** (single-level vs three-level, approximately byte-matched; C1 is 0.39 % smaller).
* Practical progression threshold at boundary 350:
  * macro per-category native Pixel AP `mean_c(AP_C3,c − AP_C2,c) ≥ +0.01` absolute, **and**
  * no category has `AP_C3,c − AP_C2,c < −0.03` absolute.
* This is an engineering progression gate, not a p-value or scientific proof. Report the results regardless of pass/fail.
* Threshold is not optimized after seeing results.

Verdict vocabulary: `NPIM01_MULTITIMESCALE_SIGNAL_DEV`, `NPIM01_COMPRESSION_ONLY_DEV`, `NPIM01_INCONCLUSIVE_DEV`, `NPIM01_BLOCKED_<REASON>`.

## 7. Uncertainty and retention

* Whole-image paired bootstrap, stratified by normal/anomalous within category, identical multiplicities across arms; image AUROC and native Pixel AP. If native Pixel-AP resampling is not practical within the bounded run, report exact point AP and explicitly “descriptive only, no Pixel-AP CI” rather than inventing significance.
* Retention/acquisition: report each category's AP/AUROC at its acquisition boundary and at later boundaries, with deltas. Do not call normal-only score drift “pre-return forgetting.” The final 50 events are the bottle-return continuation; bottle 300→350 is reported as a within-stream change, not a new task.

## 8. Resource and prohibitions

* Reuse the 12 states; no retraining. No C4. No coreset, exemplar, retrieval or ContCore method. No 15-category benchmark, no multi-seed sweep, no hyperparameter search.
* Use CUDA for encoding/scoring if safe (RTX 4050, 6 GiB; target peak reserved <4.5 GiB); CPU fallback allowed with the batch/chunk change recorded. Bound the real-data stage to about 30 minutes.
* Do not modify historical reports, protocols, caches, checkpoints or states; write new run directories and a new report.

## 9. Protocol hash

The SHA-256 of this file is recorded in the run manifest and the E1 seal before any DEV image is decoded.
