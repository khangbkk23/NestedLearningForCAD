# Nested Learning for CAD — architecture checkpoint after normal-only r1

## Decision status

**Proposed; not implemented, not experimentally validated, not yet the frozen
benchmark method.** This note is a concrete decision for review after the bounded
normal-only diagnostic. It does not edit or replace the three frozen v3 documents.

Recommendation: retain the Nested Learning for CAD topic and the HOPE-only
predictor architecture, but do not invest in the full v3 benchmark suite as though
W6 demonstrated useful multi-timescale learning. Prepare one coherent corrective
revision before viewing additional anomaly-test feedback. Do not add CADIC
coreset scoring, TITANS prototypes, NSP2, CBP or unrelated mechanisms.

Evidence: `normal_only_diagnostic_report_v1.md` and
`results/hope_cad_v2/normal_forensics_r1/{report,summary}.json`.

## Contracts retained in the proposed revision

- Same frozen ViT-B/8 ImageNet-21k checkpoint, block 8/layer 9, raw 784x768 features,
  224x224 geometry and preprocessing as the matched benchmark.
- Same four RGB masks, detached L2-unit clean teacher and unnormalized decoder
  prediction. Score remains squared teacher discrepancy from final CMS output;
  image score is maximum patch score. No nearest-neighbor or coreset path is added.
- Same input projection, spatial wrapper, four directional visual maps, five coupled
  working memories and independent per-image/view reset. Query remains outer-learned.
- Same two residual CMS MLP levels, widths, FP32 mean accumulators, local parameter
  ownership, global image clock and default periods `(1,8)`.
- Same exact 16-support differentiable meta unroll. No detach, first-order
  approximation, shortened support, task-boundary reset or accumulator flush.
- Same first-task normal-prefix privileges, held-out validation/probe exclusions,
  baseline adapter lock, no future-task calibration and read-only test evaluation.
- Same C0/C1/C2/C3/C4/A1/M conceptual controls, paired seeds/orders, co-primary
  I-AUROC/P-AUPR, pixel resolution, uncertainty policy and resource disclosure.

## Three targeted architecture/objective corrections

### 1. Express default visual retention in row/chunk units

The v3 working maps collapse over a 784-token scan, with last-chunk alpha returning
to 0.9. The proposed default is a fixed row retention `rho=0.9`, translated to
per-token alpha by geometry:

`alpha_init(C) = rho ** (1/C)`; `C=28` for the canonical row/column chunks.

The generated control becomes
`alpha_t = sigmoid(z_t @ M_alpha_boundary.T + logit(alpha_init(C)))`.
All rank-one recurrence equations, scalar-map initialization, key normalization,
injection/spectral safeguards, epsilons and readout order remain unchanged.
The row retention is fixed before any revised fit, not searched on probe/test.

This is an adaptation proposed for CAD, NOT a claim that VisionHOPE recommends
this setting. In a retention-only approximation, old `0.9**784` is about `1e-36`,
whereas this policy gives `(0.9**(1/28))**784 = 0.9**28`, about `0.0523`.
The actual recurrence also contains rank-one and injection terms, so these values
are explanatory approximations, not bounds on its realized map norms or accuracy.
Existing non-expansion guard assumptions still require alpha in [0,1), but all
new numerical/gradient behavior must be tested.

### 2. Align inner updates with the final teacher objective

At a pre-event snapshot, let `h2 = CMS2(CMS1(h0))` and
`L_final = 0.5 * mean_masked(sum_channels((D(h2)-teacher)**2))`.

For BOTH levels, use `g_l = partial L_final / partial current_weights_l`, computed
before any commit. Each level owns only its parameters; decoder and encoder stay
frozen during deployment updates. During exact meta training, derivatives through
inputs, decoder, initials and the resulting update histories remain intact.
Periods/mean accumulation/rates do not change. Intermediate teacher losses may
still be logged diagnostically but do not drive the first-level update.

This removes the known mismatch between intermediate teacher fitting and the
only output actually used for anomaly scoring. It does not guarantee a finite
step decreases loss or protects past tasks, so the registered stability gate,
rate search and later acquisition/forgetting evaluation are still necessary.
The corrected objective remains a CMS instantiation with per-level context and
update clocks; it is not claimed as a new foundational Nested Learning algorithm.

### 3. Give meta training an explicit zero-clock objective

For the same allowed query identity/view, measure prediction from learned CMS
initials and from the state after the exact 16 support events:

`L_outer = (L_query_at_0 + L_query_at_16) / 2`.

Both terms differentiate through the shared outer parameters. Only the second
term traverses the inner-update history. Query never commits. The query feature
encoding may be reused exactly because visual memory is functional per call and
does not depend on CMS history. No new images or teacher rights are introduced.

The weights 1/2 and horizons {0,16} are fixed before revised fitting, not selected
on inspected probe losses. This directly discourages a learned initialization
that requires support adaptation merely to repair an enormous output scale.
It does not train cross-category retention; that remains a W7/W8 question.

## Registered optimization and selection for the proposed revision

To avoid calling the current boundary checkpoint converged, use a disclosed
400-step maximum for BOTH independent meta and ordinary fits. Checkpoints are
0/25/50/100/200/300/400, common materialized episodes and seed pairing. This is a
new recipe, not a rerun of v3 claiming a bug fix. Even 400 does not guarantee
convergence; the complete learning curves must be reported.

- Dynamic Adam LR remains 1e-4; beta grid remains {1e-4,1e-3,1e-2}.
- Ordinary C0/C1 LR grid remains {3e-5,1e-4,3e-4}; same 400-step cap.
- Dynamic checkpoint/candidate ranking uses the mean normal-validation loss at
  horizons 0 and 16, matching the outer objective. Both component losses are saved.
- Static validation has no adaptation, so the two horizons are identical; ordinary
  training keeps its explicit 17-image mean-loss objective.
- Shortlist one checkpoint per completed fit; filter with one unchanged-constant
  1024-event normal-only stress gate; rank eligible fits by the original joint
  short-validation score, earlier step and smaller beta. No gate-loss ranking.
- C4 inherits selected M beta and rates `(beta_M,beta_M/8)` with periods `(1,1)`;
  own fit/step, no independent beta search.
- A1 reuses revised C1 weights and revised M beta, no additional outer fitting.
- Final seeds and VisA lock each method's selected recipe; no test-driven refit.
- Finite failures invalidate their fit; failed candidates are not rescued/retried.

Proposed design ID: `hope_cad_v4_warm_joint_cms_row_retention`.
Proposed selection-rule ID: `prefix_warm_joint_stress_v4`.
These IDs are reserved by this proposal, not labels for existing v3 checkpoints.

## Baselines, attribution and acceptance before benchmark

The main matched CADIC route/protocol is retained. Existing v3 normal evidence
remains reportable only at its stated scope. Revised learned parameters must be
fit fresh; old v3 initializations must not be relabeled or silently modified.
Frozen feature tensors may be reused only through explicit content/contract
verification and parent provenance; do not rewrite their historical hashes.

Before any revised anomaly benchmark:

1. Verify FP64/FP32 recurrence and first/second derivative parity under the new
   retention controls, including zero/near-floor/saturated fixtures.
2. Verify joint-final local partials, pre-event ownership, mean accumulation and
   complete meta gradient to every active outer group on the exact 16-event graph.
3. Verify the warm joint objective and joint-validation ranking independently;
   zero/query passes cannot commit or access withheld/future identities.
4. Repeat byte/source/checkpoint/clock/RNG resume, RGB/cache, read-only scoring,
   masks and canonical T4 memory/time checks. No reduced surrogate acceptance.
5. Run the revised normal-only fits, registered gates and same-history mechanism
   diagnostics. Numerical PASS is not a utility claim. Report whether updates
   improve/worsen normal fitting and whether center spatial readout remains active.
6. Review learning curves and compute before launching the full suite. Do not
   invent a normal-loss cutoff that guarantees anomaly utility.

The three changes form one corrective method. C0/C1/C2/C3/C4/A1 comparisons support
its registered component questions, but they do not alone isolate all differences
between original v3 and the revision. Any claim attributing gain to exactly one
of the three corrections requires an explicit one-change control before test
evaluation. Normal-only controlled studies can establish mechanism effects;
anomaly superiority still requires the locked paired benchmark.

## Remaining risks and decision boundary

This proposal does not establish higher AUROC/AUPR, eliminate catastrophic
forgetting, or solve frozen-first-category representation generalization. A better
normal predictor can still reconstruct anomalies too well. Joint-final gradients
can still overshoot at finite rates. Longer fitting increases cost, and the full
suite may require multiple saved sessions or the separately confirmed final GPU.
No resource duration or success probability is promised before profiling.

If approved, inherit the FULL three v3 documents into v4 and replace only the
affected equations/recipes/IDs/status sections, implement/test the correction and
prepare a bounded normal-only Kaggle pilot before benchmarking. Do not use this
short proposal as an A-to-Z replacement specification.

The current architecture checkpoint therefore recommends a single registered
revision rather than automatic full W7 on v3. It is awaiting the human's design
review because the requested freeze prohibits silently changing the algorithm
after observing development evidence. No architecture, training recipe or frozen
specification has been changed by the forensic run itself.
