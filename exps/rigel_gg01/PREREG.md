# GG-01 pre-registration (frozen before any GG-01 outcome is computed)

**Registered:** 2026-11-26. Owner task: "RIGEL-CAD — GG-01: Geometry-Guided Exemplar Learning" (session message of 2026-11-26).

**Pre-registration status:**
- This file is committed locally before the DEV stream runs.
- `exps/rigel_gg01/dev.py` refuses to compute metrics or the gate if this file's sha256 differs from the one recorded at the stream stage.

---

## 1. Hypothesis

**H1.** Historical residual geometry can guide exemplar consolidation to improve normal-support retention and downstream anomaly detection under the same memory constraint.

**Single candidate: GG-01.** Only the global task-boundary consolidation changes.

| | Consolidation rule | Metric |
|---|---|---|
| v1 | `K_t = KCenter_P(K_{t-1} ∪ C_t, m)` | P = seeded Gaussian projection to 128 dimensions |
| GG-01 | `K_t = KCenter_{W_{t-1}}(K_{t-1} ∪ C_t, m)` | `d²(x, y) = ‖(x − y) W_{t−1}‖²`, computed exactly in the full 768-d space (no approximation) |

**Definitions in GG-01:**
- `W_{t−1}` = `whitener(Σ_{τ<t} S_τ, Σ_{τ<t} n_τ)`, cast to fp32. It is the same object that v1 uses to score after task t−1.
- At task 1, or whenever no slow statistic exists yet, GG-01 applies the v1 rule.

**Seeded semantics:**
- The generator is advanced through the same draws as v1: the projection is drawn and discarded, then the start index.
- The first centre is therefore the same index as in v1.
- Survivors keep their insertion order.

**Unchanged from v1:**
- the per-task coreset `C_t` (1 %, `ops.greedy_kcenter`) and the residual moment `S_t`;
- the shrinkage of 0.5 and the capacity m = 3,925;
- the encoder, the scoring `½ min ‖qW_t − mW_t‖²` and the A1 read-out;
- image and token order.

**Not used:**
- no old images and no discarded tokens;
- no task ID at inference.

**Code:**
- `exps/rigel_gg01/gg.py` (`kcenter_w`, `ArmMemory.write` rule `gg`);
- v1 primitives are imported from `rigel_cad/ops.py` unchanged.

## 2. Arms

- **Gating comparison:** GG01 versus V1.
- **Controls:** reported with paired CIs; they never gate the decision.

| Arm | Rule | Capacity | Scoring |
|---|---|---|---|
| V1 | v1 consolidation | 3,925 | whitened with `W_t` |
| GG01 | GG-01 consolidation | 3,925 | whitened with `W_t` |
| F | v1 consolidation | 5,461 (B/1536) | Euclidean |
| U | union of all `C_t` | none | whitened |
| CCW | ContCore-style (see below), then v1 consolidation | 3,925 | whitened |
| CCE | same as CCW | 5,461 | Euclidean |

**ContCore-style expansion (CCW, CCE).**
- Expansion `G_{n_t}(Z_t; K_{t−1})` over all task tokens, with `n_t = |C_t|`.
- The first point is `argmax d(z, B)`.
- Distances use the v1 projected metric P.
- When `K_{t−1}` is empty, the canonical `C_t` is used.
- This is a PROJECT_MAPPING of arXiv 2608.15277, Eq. 2–3. It uses no nearest-neighbour approximation (q = 0). WRN50 is replaced by the frozen DINOv2 tokens, and m is fixed by the byte budget.
- `S_t` stays the canonical one (residuals against `C_t`), so CCW and V1 have identical W at every stage. CCW − V1 isolates the selection rule.

**Pre-registered control questions:**
1. CCW − V1: does base-set-aware expansion help under whitened scoring?
2. CCE − F: does it help without whitening?
3. GG01 − CCW: geometry-guided consolidation versus ContCore-style expansion.
4. V1 − F: the slow level on the DEV stream.
5. U − V1: the headroom left by the budget.

## 3. DEV split

**VisA-DEV is rejected.**
- VisA-DEV is 240 VisA test images (`agents/reports/cad_dataset_custody_audit.md` §4).
- These are a subset of the official VisA test set, which the FULL validation evaluates in full (2,162 images).
- Its independence from the official evaluation samples therefore fails.

**Registered DEV set: MVTec AD 2 `test_public`.**

**Stream:**
- 8 scenarios in alphabetical order: can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts.
- Training uses `train/` only, i.e. normal images.
- `validation/` is not used.

**Evaluation:**
- 1,084 `test_public` images: 705 bad, 379 good.
- Masks come from `ground_truth/bad/<stem>_mask.png`, for images under `test_public/bad/` only.

**Image identifiers:**
- the sorted `rigel_cad.data.ad2_files` lists of `train` and `test_public`;
- sha256 of the newline-joined list: `fac5201b5236230a22ca57b77047d235e85806136bddbae6f3efd7e2f84eee8e`.

**Independence:**
- No image overlaps MVTec AD, VisA or MPDD.
- No GG-01 detail is chosen on it.
- The pixel AP of any method on `test_public` has never been computed in this project.

**Exposure disclosure:**
- v1-family AU-PRO and SegF1 on `test_public` were computed and reported before (NL03-MULTIRATE-01 §10, RIGEL-CAD-01 §9).
- Those local numbers used a stem-only mask lookup that gives the good images the mask of the same-named bad image. That is a pre-existing defect, reported separately.
- This DEV evaluation uses the corrected mask assignment above.

## 4. Fixed inputs

**Encoder:**
- DINOv2-reg ViT-B/14 at 392 px.
- Checkpoint `checkpoints/dinov2/dinov2_vitb14_reg4_pretrain.pth`, sha256 `73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71`.

**Tokens:**
- `/mnt/k/NLCAD/cache/rigel_cad/dinov2_vitb14_reg4/ad2/*.pt`.
- Each file is sha256-verified against its JSON by `rigel_cad.pipeline.TokenStore`.
- These are the same tokens as the RIGEL-CAD end-to-end AD 2 run. They were bitwise equal to the locked AD 2 path.

**Fixed settings:**
- seed 2711;
- one order (alphabetical);
- budget 8 MiB (8,388,608 B).

**Source commits:**
- baseline v1 = `rigel_cad/` at `45422f6`;
- candidate = the commit that adds this file.

**Environment:**
- torch 2.10.0+cu126, Python 3.11.17, numpy 2.4.6, scikit-learn 1.9.1;
- RTX 4050 Laptop GPU, 6 GB.

**v1 parity (correctness gate):**
- The V1 arm's `test_public` grids must equal the locked AD 2 arm-E grids (`/mnt/k/NLCAD/cache/nl03_multirate_01/ad2/grids/E`) bitwise, with identical rows.
- On MPDD order a, seed 2711, the V1, U and F arms already equal the locked grids at every stage (`exps/rigel_gg01/test_gg.py::test_parity_with_locked_v1_mpdd`, passed 2026-11-26).

## 5. Metrics and statistics (DEV)

**Primary metric:**
- **Macro pixel AP**: the mean over the 8 scenarios of the final-stage native-resolution binned pixel AP.
- Computed with `rigel_cad.metrics.evaluate_category`: 8,192 bins over [0, max energy of that arm's scenario grid + 1e-12], A1 read-out, bilinear `to_native_torch`.

**Secondary metrics:**
- image AUROC (top-1 % score);
- forgetting AP (max over the stages after acquisition, minus final, averaged over scenarios 1–7);
- acquisition AP and prequential AP;
- per-scenario final AP;
- update time, latency (p50 per image), persistent bytes;
- the diagnostics of §8.

**CI:**
- Paired image bootstrap: 2,000 replicates per scenario, using `rigel_cad.metrics.boot_weights(n_s, position_s)`. The weights are identical for every arm.
- Each replicate's macro AP is the mean of the per-scenario APs.
- The 95 % CI of `macro(GG01) − macro(V1)` is the 2.5 / 97.5 percentiles.
- **Limitation:** there is one stream and one seed, so the CI covers test-image sampling only, not order or seed variance. The FULL stage covers those.

## 6. DEV gate (conjunctive; frozen)

| Gate | Condition |
|---|---|
| G1 | Δ macro pixel AP (GG01 − V1) ≥ +0.005 |
| G2 | Paired bootstrap 95 % CI lower bound of Δ > 0 |
| G3 | No scenario loses more than 0.03 final pixel AP |
| G4 | Forgetting AP(GG01) − forgetting AP(V1) ≤ 0.005 |
| G5 | Physical persistent bytes(GG01) ≤ V1, and nominal bytes(GG01) ≤ V1 |
| G6 | Correctness: V1 parity bitwise; finite W; positive-definite shrunk moment at every stage; unit tests pass |

**Verdict rules:**
- `GG01_REJECTED_DEV` if any of G1, G3, G4, G5 or G6 fails.
- `GG01_INCONCLUSIVE_DEV` if G1 and G3–G6 pass but G2 fails.
- `GG01_DEV_PASS` if all gates pass. The candidate is then sealed and FULL runs automatically.

**Forbidden:** no second seed, metric, shrinkage, capacity or heuristic is tried after the DEV outcome.

## 7. FULL protocol (frozen now; runs only after `GG01_DEV_PASS`)

**Cells:**
- MVTec AD (15 tasks), VisA (12), MPDD (6);
- orders a / b / c (`exps.nl03_multirate_01.final.bench_order`);
- seeds 2711 / 2712 / 2713;
- 27 cells in total.

**Fixed across arms:**
- frozen DINOv2-reg ViT-B/14;
- the canonical token caches (`RN.fit_tokens`, `G.load_test`);
- the full official test sets.

**v1 values:**
- The locked E@8 metrics of each cell, located read-only by `exps.g3e_paper_01.tables.row(ds, order, seed, 'E@8')`.
- Correctness gate: in every FULL cell, the V1 arm's final-stage grids are recomputed and must equal the locked E@8 final-stage grids bitwise.

**GG-01 metrics:**
- The locked evaluator `exps.nl03_pressure_00.evaluate.run_task`, identical to v1.
- Order a gets the bootstrap (2,000 replicates).

**FULL gate (conjunctive; frozen).** Δ = GG01 − V1, and "mean" is over the 9 cells of a dataset.

| Gate | Condition |
|---|---|
| F1 | All correctness and reproducibility checks pass (parity of every cell, finite W, tests) |
| F2 | The equally weighted mean over the 3 datasets of mean Δ pixel AP ≥ +0.005 |
| F3 | At least 2 of 3 datasets have mean Δ pixel AP > 0 |
| F4 | No dataset has mean Δ pixel AP < −0.01 |
| F5 | No category loses more than 0.03 final pixel AP in the primary comparison (order a, seed 2711, per dataset) |
| F6 | No dataset has mean Δ forgetting > +0.005 |
| F7 | No dataset has mean Δ image AUROC < −0.005 |
| F8 | Physical persistent bytes(GG01) ≤ V1 in every cell |
| F9 | No protocol, cache, preprocessing or metric mismatch (F1 parity plus identical evaluator and rows) |

**Verdicts:**
- `GG01_FULL_PASS` if every gate passes;
- `GG01_FULL_REJECTED` if any of F2–F8 fails;
- `GG01_BLOCKED` if F1 or F9 fails.

**Order-a paired CIs:** reported, using the registered bootstrap weights.

**Independence:** order, seed and dataset cells are not treated as independent image-level observations.

## 8. Diagnostics (explanatory only; never gating, never tuned on)

**Retention:**
- keys retained per historical task at every boundary;
- evictions per boundary;
- age (stage − insertion stage) of the retained keys;
- the stage at which the capacity first binds;
- the fraction of each task's original keys retained;
- bytes per stage.

**Coverage:**
- Mean normal-token energy of a fixed held-aside sample of 4,096 training tokens per task (seed 20261126 + task index), under each arm's state after every boundary.
- Measured both after consolidation and on the pre-consolidation candidate set (V1, GG01, F).
- Coverage uses the stage's W; the Euclidean metric for F/CCE.

**Geometry drift:** `D_t = ‖W_t − W_{t−1}‖_F / ‖W_{t−1}‖_F`.

**Numerics:**
- condition number of the shrunk moment per stage;
- determinism of `kcenter_w` (unit test).

**Memory:**
- nominal (v1 accounting);
- physical tensor bytes;
- serialized bytes;
- inference-derived state;
- peak extra VRAM of each consolidation;
- the shared encoder, reported separately.

## 9. Compute budget and stop conditions

**Budget:**
- DEV ≤ 1.5 GPU-h; FULL ≤ 6 GPU-h; CPU metrics ≤ 8 h.
- Only one GPU job at a time.

**Stop conditions:**
- any free-space check on D:, E: or K: below 30 GiB;
- a parity failure (gives `GG01_BLOCKED`);
- a non-finite or non-positive-definite W;
- the budget exceeded (the stage is reported as not run).

**Storage:**
- New caches are written only under `/mnt/k/NLCAD/cache/rigel_gg01/`; canonical caches are read-only.
- Results go only under `results/hope_cad/rigel_gg01/`.
