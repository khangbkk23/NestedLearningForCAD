# Meta-NATH CAD — Phase 2 protocol and implementation plan

**Status:** working protocol; Track A code path added, benchmark not run, 2026-09-26  
**Purpose:** make the implementation, experiments, and thesis claims agree. This document supersedes neither the original `instruction_CAD.md` nor the Phase 1 report; those remain historical records.

## 1. Research tracks

| Track | Purpose | Backbone / method | What may be claimed |
|---|---|---|---|
| A. CADIC reproduction | Reproduce the published continual anomaly detection baseline | ViT-Base-Patch8-224, ImageNet-21K pretrained, selected intermediate layer, frozen backbone, patch-vector coreset | Reproduction under the exact recorded protocol and deviations; do not claim exact reproduction where the paper is underspecified |
| B. Meta-NATH ablations | Measure the contribution of each proposed memory/consolidation component | Same backbone, data order, feature pipeline, memory budget, and evaluator as Track A | A component's measured effect, supported by controlled ablations |
| C. Foundation-backbone extension | Test transfer to modern self-supervised backbones | DINOv2 first; DINOv3 only when its weights and runtime are available | Within-backbone comparisons; do not attribute differences from Track A to the algorithm alone |

The Phase 1 DINOv2 results are historical development results. They are not a CADIC-parity baseline. DINOv3 remains a planned extension until an actual run is recorded.

## 2. Track A protocol

### Data and stream

- Dataset: MVTec AD, 15 categories, in a fixed alphabetical order.
- Training stream: original `train/good` images only. Synthetic anomaly generation is disabled for this track.
- Test data: official category test sets and masks. Test metrics are reporting-only; they must not select checkpoints, tune hyperparameters, or accept/reject Phase 3 changes.
- Freeze and record the category order, image transforms, random seed, dataset version/checksums, and code commit before the benchmark run.

### Features and distances

- Backbone family: ViT-Base-Patch8-224 pretrained on ImageNet-21K; input 224×224; expected patch grid 28×28 and feature dimension 768.
- The current configuration pins the model identifier to `hf-hub:timm/vit_base_patch8_224.augreg_in21k`; it uses direct 224×224 bicubic resize and mean/std `[0.5, 0.5, 0.5]`. This is a recorded reproduction choice; verify the exact paper checkpoint and preprocessing against the authors' released implementation before calling it bit-for-bit reproduction.
- Published CADIC selects layer 9 for its reported configuration. The paper reports that layer 7 scores better on pixel localization in its layer ablation. Use layer 9 for the reproduction configuration; treat layer 7 as a separate localization ablation.
- Record the exact weight repository/revision, library version, preprocessing, block indexing, and whether a normalization layer is applied to the selected block. The paper does not fully specify all of these implementation details.
- Distance: raw Euclidean L2 for Track A. Do not add L2 feature normalization unless a verified source establishes that the reference pipeline uses it.
- Coreset capacity is **K patch vectors in total**, not K images. Reference capacity experiments: K ∈ {2,500, 5,000, 10,000}; use 10,000 for the main comparison if resources allow.
- Implement CADIC batch update on patch vectors: find the candidate with maximum distance to its nearest coreset vector; find the closest pair already in the coreset; replace a member of that pair when the candidate distance is larger; repeat over the incoming batch until the condition fails.
- Keep image replay/anchor storage, if later needed, as a separate budget from the patch-vector bank.

### Scoring and metrics

- Pixel scores are nearest-neighbor distances from query patch vectors to the shared patch-vector bank. Record the interpolation, smoothing, score normalization, and image-score neighborhood setting.
- Main reported pixel metrics use every pixel. Debug sampling is permitted only when marked approximate and must not be mixed with full-pixel results.
- Report per-category values and their unweighted macro average from the **same final checkpoint**. Pooled all-category metrics may be included separately and labeled pooled.
- For continual forgetting, save task-stage checkpoints on a fixed schedule and evaluate the fixed protocol at each stage. Those measurements are for reporting only and cannot choose a checkpoint.
- Record image AUROC, image AP, pixel AUROC, pixel AP/AUPR, forgetting, runtime, and memory separately. State the metric implementation (including the AP interpolation convention).
- The CADIC paper's tables label 0.584 as MVTec P-AUPR, while one prose passage calls that pixel AUROC. Report both metrics and identify 0.584 specifically as the table's P-AUPR value, not as an unambiguous statement from every passage of the paper.

## 3. Test isolation and Phase 3

- `best_and_last` must not select checkpoints using official test metrics. Fixed task checkpoints may be saved for a predeclared forgetting analysis; saving is not selection.
- Phase 3 acceptance must not read official test metrics. For now, keep metric-based acceptance disabled. The consolidator's anchor representation-drift rollback is an internal stability guardrail, not evidence that anomaly-detection quality improved.
- A held-out normal subset from `train/good` can measure score drift and false positives on normal data. It cannot measure defect recall, AUROC, or P-AUPR. Any anomaly-metric-based model selection requires a separate, predeclared labeled validation set; if taken from the official test set, that split is no longer the final untouched benchmark.
- Drift/FPR thresholds must be calibrated and fixed before the final evaluation. Values such as 0.05 drift or 1% FPR change are not accepted as defaults without evidence.

## 4. Current implementation facts and gaps

| Area | Current behavior | Phase 2 treatment |
|---|---|---|
| Phase 1 backbone | Transformers remains the legacy default; fallback is disabled unless explicitly enabled for local debugging | Benchmark loading fails closed; Track A uses the configured timm ViT adapter |
| Dataset | Synthetic anomaly probability is configurable and defaults to 0.5 for legacy runs | Track A config sets it to 0 and requires all 15 ordered categories |
| Coreset | Legacy mode counts image entries and selects/replaces using CLS; patch-vector mode counts individual patch features | Keep checkpoint formats separate; Phase 3 rejects patch-vector mode until a separate anchor memory exists |
| Evaluation | Final-checkpoint evaluator reports same-checkpoint per-task macro metrics; training summary labels task-end-state means separately; pooled evaluation is optional | Summaries include split, aggregation, pixel-sampling, and runtime provenance |
| Pixel metrics | Default sampling limit remains 10,000 pixels per image | `null` or `full` selects every pixel; sampled metrics record whether sampling actually occurred |
| Fast Memory | TITANS currently processes CLS; inference patch scoring uses raw patch features and the stored bank. CLS memory can indirectly affect admission/gating | Keep it out of Track A; test patch-memory as a later ablation, not as a presumed fix for Phase 1 metrics |
| Utility/config | Utility update methods are not wired into the stream; several optimizer/loss fields are not consumed in Phase 1/2 | Do not claim utility-guided selection; document unused config until a specific component consumes it |
| Phase 3 selection | Acceptance is disabled in the distributed Phase 3 configs; enabling it requires validation summaries | Test-split summaries are rejected, and patch-vector mode is blocked from Phase 3 |

## 5. Ordered delivery plan

1. Preserve Phase 1 artifacts and freeze the Track A protocol above. **Done:** original report/config history retained; protocol recorded here.
2. Add the CADIC parity config, disable synthetic anomalies for that config, and make backbone loading fail fast. **Done:** `conf/cadic_phase2.yaml` and explicit fallback control.
3. Add a patch-vector coreset without changing old image-entry checkpoints. Make Phase 3 explicitly incompatible with this baseline mode until image-anchor support is designed. **Done:** separate coreset mode and explicit Phase 3 guard.
4. Add the ViT-B/8 feature adapter and record exact feature extraction assumptions. **Done:** timm adapter; layer, final normalization, resize, and normalization are explicit in config.
5. Run K=2,500 as a resource/pipeline check, then K=5,000 and K=10,000 for the capacity study. Do not use the official test to change the method after seeing its results.
6. Freeze and report Track A, including per-task and macro metrics, full-pixel settings, forgetting, runtime, and peak memory.
7. Add one Meta-NATH component at a time on the same ViT-B/8 protocol: existing CLS memory as a control where meaningful, fixed-frequency patch memory, adaptive update frequency, then slow consolidation.
8. Move the strongest controlled variant to Track C and compare DINOv2/DINOv3 variants only within their backbone track.

## 6. Resource accounting

For a float32 ViT-B/8 feature bank, vector storage is `K × 768 × 4` bytes: K=2,500 uses 7.68 MB (decimal), K=5,000 uses 15.36 MB, and K=10,000 uses 30.72 MB. Also measure pairwise-distance temporaries, model/activation memory, any cache copies, and image replay separately. Record peak CUDA allocated bytes for feature extraction, coreset update, and nearest-neighbor scoring; synchronize CUDA around timed regions. The optional host RSS profiler samples process memory at stage boundaries; record elapsed time and host RAM when available.

## 7. Claims not yet established

- Phase 1's reported pixel result does not establish that CLS-only Fast Memory caused the result.
- Patch-level TITANS, adaptive frequency, utility-guided selection, DINOv3 support, and edge readiness are hypotheses or plans until implemented and measured.
- The project is Nested Learning inspired; the current code is not the full Hope architecture or its Continuum Memory System.
- A result near a published CADIC number is a reference point, not a hard pass/fail threshold, because the paper leaves some checkpoint/preprocessing/metric details underspecified and contains the P-AUPR/P-AUROC wording inconsistency noted above.
