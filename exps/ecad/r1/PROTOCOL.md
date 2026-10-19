# ECAD-R1A corrected normal-only protocol — REVIEW DRAFT, NOT AN EXECUTION LOCK

This document authorizes no data access. Stages 0–3 end with independent review.
No evaluation-normal features have been consumed. No efficacy, anomaly, test,
mask or sealed-identity reader is part of this package. The R0 adversarial review
is a required correction, not independent certification. There is no ADSM arm.

## Representation, source and update contract

Use the existing Task P representation: raw FP32 `[784,768]` patches from frozen
ViT-B/8, zero-based block 8 before final normalization, CLS removed, direct square
224px PIL bilinear resize and the recorded ImageNet mean/std. The extractor
executes all twelve blocks. Task P checkpoint SHA-256 is
`755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415`.
It is a reconstructed representation, distinct from the missing historical
`ae3012808a9b406a19b799381bd26b253634ad26125937c26d02dbcbbc85dd92` checkpoint.
The current Task P checkpoint is also absent. Do not provision, replace or rewrite
either identity. Existing clean cache file presence and lineage documents are
verified; their tensor contents and file digests have not been revalidated here.
An authorized custodian must verify them before any real-normal execution.

Author reference: `jungyg/ContCore@0579a727e2320ea93c8a657e02600080837ee2a0`,
five byte-verified files and SHA-256/Git blob digests in `reference/SOURCE_PIN.json`.
The original paper is https://arxiv.org/html/2608.15277v1. The author implementation
accumulates task features and consolidates after each task. This adaptation commits
after every image, in full feature dimension, with no task boundary, projection,
FAISS copy, retained task pool or raw-feature archive. It is not byte-identical
author reproduction. The main loop, expansion base/index discrepancy and tie/
numeric choices are documented in the handoff report.

For image t, observe all 784 patches once. Generate serial `784*t+p`, select 78
distinct incoming rows by farthest-first conditioned on the ENTIRE OLD bank and
previous selected incoming rows. Empty base starts with smallest serial. Equal
computed distances choose smallest serial. Form old bank plus 78 candidates.
Keep it in serial order if size <=K. Otherwise, in the union compute each row's
nearest OTHER ROW distance (exclude only its diagonal, including zero distances
between duplicate-valued rows). Keep floor(.75*K) largest radii with serial ties;
complete to K by farthest-first conditioned on these retained rows. Mask already
selected indices, including duplicates, and commit in serial order. No value
deduplication. q=0 is a synthetic oracle control only. K=2500, p=.1, q=.75 are fixed.

`distance.py` supplies every matched arm's Euclidean values: FP32 norm/dot,
autocast/TF32 off, cancellation repair by direct FP64 subtraction when
`d_squared <= 64*eps32*(norm_x+norm_y)`, sqrt before FP32 output. Chunk=256,
repair chunk=1024. P0 promotes these same rounded values to FP64 for its existing
guard arithmetic/cache. Score all banks with the SAME global NN, image maximum,
and any later authorized native map with FP32 bilinear interpolation,
align_corners=False. No squared-score substitution, smoothing or image-support
weighting. Real-normal numerical validation is still required: tolerance
`abs(error)<=2e-6+2e-6*abs(direct_FP64_distance)` and exact decision parity on the
registered reduced oracle fixtures; stop rather than change the kernel after lock.
CPU tests do not establish CUDA numerical parity.

An arrival service must enforce source identity as well as contiguous event
numbers. Learner APIs have no category argument or storage-reader callback.
Opaque event checks alone cannot identify maliciously mislabeled input tensors.
No previously delivered image is replayed into a learner, including self replay;
retained support rows and P0's bounded prior witnesses are the only history.
Scoring is read-only. Checkpoints preserve counters, RNG and warm caches.

## Six mandatory arms and two byte ledgers

ContCore, Algorithm-R full-grid reservoir, global CADIC closest-pair replacement,
existing P0 WITNESS_PROTECTED, spatial S4 and balanced random R4 are mandatory.
S4 means 4x4=16 groups, NOT four groups: 49 grid positions each, row-major group
quotas with remainder first. R4 uses the existing geometry-only generator,
seed 17011, 24 candidates, at most 40 attempts/candidate, no within-group lattice
adjacency and maximum cell load 10. Save assignment digest before the lock.
Every group contributes to the global scoring union. CADIC retains its historical
prefix fill and slot-based pair ties; those are not ContCore's serial ties.
P0 preserves 128 witnesses, shortlist16, mean gain/cost and q90 <=.05*historical
scale, RNG seed7203, and witness insertion AFTER the image. Its known guard
limitations are not repaired. Reservoir RNG seed7202 is independent of order and
R4 RNGs. Use paired RNG streams across ledgers where their opportunity laws agree;
different capacities naturally produce different states.

1. Equal descriptors: every arm has capacity2500; record actual occupancy and all
   online bytes. Full-grid controls typically fill by image4; ContCore fills at
   image33. Acquisition uses the actual states, without forcing equal occupancy.
2. Equal online bytes: ceiling **7,716,384 B**, the materialized ContCore2500 charge:
   `(3072+8)*2500+16384`. This replaces the R0 analytic norm-cache envelope with
   this implementation's measured storage contract. Assigned capacities are
   ContCore2500, reservoir2498, global CADIC1055, P02358, S4/R42122.

The exact formulas are `controls.charged_bytes`; each capacity is maximal under
the ceiling. Charge FP32 bank values, int64 row IDs where retained, P0 nearest
distance/index arrays and witnesses, all generators, int64 assignments, and LIVE
K² pair caches. Metadata envelopes are 16KiB for single-bank arms and 64KiB for
the 16-bank implementations; observed Python headers must fit. Fail closed if
actual storage/envelopes exceed the formula. Include allocated unused slots,
not just occupied rows. Device mirrors or retained indices must be added.
No timings from cached code may use an uncached byte count. This frontier uses
these exact implementations; it does not prove a quadratic cache is intrinsic to
CADIC. A future uncached engineering variant requires its own time/byte pairing.

Common encoder parameter payload343,231,488 B is added identically to deployment
ledgers; the absent serialized Task P checkpoint is documented as343,282,995 B
historically, not newly measured. Inference exports contain only the score-bank
union and fixed score contract; witnesses/caches/RNG are online-update state.
Encoder, allocator/context/library workspace, CPU RSS, checkpoint bytes and peak
CUDA allocated/reserved are separate measures. No archive is learner-accessible.

## Identities and six recurrence orders

Only existing train/good normal provenance is eligible. The source manifest's
canonical SHA-256 is
`060579457d344992eef159b1d4f1a86c34dd9283b97b70d3c3a386006b0f4dcc`.
It lists880 globally distinct content hashes: bottle209, carpet280, hazelnut391.
Every category needs206 identities across the six permutations:

| Disjoint role per category | Images |
| --- | ---: |
| Calibration writers |24|
| Calibration held-out witnesses |8|
| First-pass evaluation writers |100|
| Return evaluation writers |50|
| Evaluator-only scale-reference images |8|
| Evaluation held-out normal witnesses |16|
| Total |206|

This count is feasible IN THE MANIFEST, leaving3/74/185 unused respectively.
All 880 current raw train/good PNG byte hashes match the manifest; no image was
decoded. Current cache-to-manifest tensor alignment remains a custodian check.
Do not traverse any official-test path, even test/good, or any sealed membership.
Do not replace missing normals with test normals. Do not import P1 discovery.
Do not hash/decode evaluation feature payloads to perform this metadata stage.

Before access, a custodian creates a disjoint content-hash/identity manifest using
normal-only metadata: within each category order by SHA-256 of
`"ecad-r1a-split-11021:" + raw_png_sha256`; take successive role counts in table
order. This allocation is a proposed deterministic rule, not an already frozen
split. Existing identities are historically exposed normal data, not fresh
confirmation. Preserve role sets across all six orders/arms/ledgers.

Orders are all lexicographic permutations of (bottle,carpet,hazelnut), indexed by
8011..8016; each contains100+100+100+50 distinct writer images, with only its FIRST
category returning. Within-category first-pass and return shuffles use separate
NumPy PCG64 generators seeded by `(order_seed,category_index,role_index)` via
SeedSequence. Category metadata stays in the evaluator/arrival service.
Boundaries100/200/300/350. Report pre-return300 for both older categories, plus
first-category200 loss and return350 recovery. Do not let return recovery erase
pre-return loss. Six fixed orders are conditions, not independent datasets.

## Calibration is not synthetic saturation

Synthetic resource fixtures use no real-normal data and say nothing about
statistical power, useful headroom, category geometry or detection performance.
Real calibration requires separate authorization. Use24 calibration writers per
category and8 disjoint calibration witnesses; engineering seeds7011/7012 with
fixed forward/reverse category orders. Each fresh72-image stream can fill
ContCore2500 at33, leaving40 arrivals including first saturation,39 fully saturated
starting states. No old raw-image replay; reset between independent calibration
streams. This is enough to EXERCISE saturated updates at2500, not to establish
tail precision. The first24-image category is still unfilled at its acquisition.
It is unsuitable for certifying saturated first-task power. Do not extend to5k/10k.

No calibration outcome may choose p/q/K/arms/kernel/score. Use calibration only
for numeric checks, end-to-end I/O/resource forecasts and precision/headroom
assessment. No bank/witness/scale/RNG state transfers into evaluation. Eight
calibration witnesses make an upper-eighth image tail a single-image endpoint;
16 evaluation witnesses make it two images. These are major precision limits.
The R0 `c+.841621*SE` calculation is merely an approximate sensitivity diagnostic,
not demonstrated80% population power. No inferential readiness is claimed now.
If authorized calibration cannot support useful precision/headroom, stop as
UNDERPOWERED/NO_HEADROOM, or obtain explicit authorization for a descriptive-only
screen. Do not enlarge calibration by consuming evaluation roles: adding nine
writers/category to fill the first segment would require215>209 bottle identities.

## Exact endpoints and paired uncertainty

The eight evaluator-only reference images become accessible only on their
category's arrival. For each patch compute distance to the union of patches from
the OTHER seven reference images; scale s_c is median of all8*784 distances,
floored at1e-6. Freeze it for every arm/order/ledger. Reference images and witness
images never enter a learner. These eight images are for scale only; there is no
fitted evaluator clustering or claimed semantic-mode endpoint in this draft.

For held-out image i, patch p, category c, order o, arm m, let d_a be the global NN
distance at that category's acquisition boundary and d_t at a later boundary.
Define signed inflation `(d_t-d_a)/s_c` and positive inflation
`max(d_t-d_a,0)/s_c`. TopFraction(v) is the arithmetic mean of the largest
ceil(len(v)/8) observations, retaining multiplicities. First reduce784 patches
WITHIN EACH IMAGE; then reduce16 image values within category. The positive tail
is `F_m,c,o,t = TopFraction_i(TopFraction_p(positive_inflation))`.
Keep the analogous signed endpoint, absolute current normalized NN distances,
mean, per-image q95/q99 and max. Discrete q95/q99 use sorted rankceil(p*N)-1.

PRIMARY retention risk is the MEAN over six orders of the MAXIMUM of the TWO
old-category F values at t300. Maximum is taken BEFORE averaging orders.
`endpoints.pre_return_risk` implements this order. Report both category terms,
their earlier boundaries and return separately. Neither a good mean nor positive
clipping may conceal an absolute q99 regression or a poor acquisition baseline.

At each category's OWN acquisition boundary define A_mean as the witness-image
average of image mean patch distance/s_c and A_q95 as the witness-image average
of its rank-q95 patch distance/s_c. These are absolute coverage endpoints, not
change from an arm-specific poor birth baseline. Acquisition and old-category
absolute q99 comparisons remain category-specific, with equal category weights.

For control j, `B_j=F_j-F_ContCore`, positive ALWAYS favors ContCore. If F_j<.02,
report absolute B and NO_HEADROOM; relative benefit is null, never epsilon-divided.
For material5% improvement use `M_j=.95*F_j-F_ContCore`, recomputed inside each
bootstrap rather than attaching an uncertain plug-in denominator afterward.
Do not require superiority over a zero-forgetting under-acquiring control.
For acquisition regression use `(A_ContCore-A_j)/A_j` if A_j>=1e-6; positive is
worse. If smaller, report ZERO_REFERENCE, the absolute difference and compare
to1e-8 normalized units, with no relative claim. Regular noninferiority margin1%
uses contrast `(A_ContCore-1.01*A_j)/max(A_j,1e-6)` and upper bound<=0.
Old absolute q99 safety uses analogous5%/zero-reference contrasts at t300.

Use2000 whole-image paired bootstrap draws, seed9011, stratified by category.
One draw of16 witness-image indices/category is reused across ALL arms, ledgers,
six orders and every occurrence/boundary of that image. Resample complete image
grids and recompute nested tails, maxima, means, ratios and denominators. Never
resample patches or treat repeated scores as independent. Writers/orders remain
fixed; intervals are conditional on this normal cohort and these writer streams.

One declared primary family contains164 coordinates:10 B_j (five controls times
two ledgers),10 M_j,24 acquisition-margin contrasts (reservoir/CADIC times three
categories times two metrics times two ledgers, averaged over fixed orders within
category), and120 old absolute-q99 safety contrasts (five controls times two old
categories per order times six orders times two ledgers). The exact count is
10+10+24+120=164. Save coordinate names/order in the execution
lock. For each draw compute the maximum absolute centered error across the
entire family; radius is its rank-higher95th percentile. Use simultaneous basic
intervals `[point-radius,point+radius]` for every coordinate. This is an explicitly
approximate finite-cohort bootstrap; extreme tails/maxima may be biased/unstable.
No independent coverage or power guarantee is claimed. There are no larger-K
secondary p-values or an unspecified Holm procedure.

Report each material contrast and acquisition safety with interval widths and
headroom. Retain adequate-acquisition non-dominated conventional cost/coverage
points; a useful frontier point need not beat every weaker control. No normal-only
result is an anomaly-efficacy result, and no automated new-architecture promotion
is included. If acquisition uncertainty cannot exclude the margin, label the
contrast inconclusive. A later mechanism-superiority claim must beat the strongest
matched adequate-acquisition frontier, with a separately reviewed decision lock.

## Saturated cost preflight and strict120-minute admission

`profile.py` first runs CPU correctness and endpoint tests. If they fail, no CUDA
worker starts. It then checks CUDA; inaccessible hardware produces null metrics
and preserves NOT_RUN failures. At accessible CUDA,12 sequential jobs cover six
arms times two byte ledgers. Each starts empty, measures fill, actually reaches
capacity through synthetic causal arrivals, then measures two stationary Gaussian
and two shifted/broader high-replacement-demand images. High demand may trigger
P0 refusal; report actual replacements rather than relabel a refusal as high
replacement. A geometry that fails to exercise replacement is an explicit
coverage limitation. Worker45-second and global600-second deadlines preserve
partial records, errors, timeouts and unstarted cells. Larger K is rejected.

Measure synchronized selection-only time, global784-query NN time, bank-union
preparation and cached-image wall from synthetic CPU feature creation through
H2D/update/NN/D2H. Encoder-inclusive latency is UNMEASURED. Include process RSS,
allocated/reserved CUDA peaks through resume, complete live-cache ledger and
checkpoint/inference bytes. Distance pair/repair/block counters are separate
from profiled scalar reads, dynamic compactions and actual CUDA synchronization
API events. Diagnostic profiling overhead is separate from timing samples but
inside the stage cap. Two observations/geometry DO NOT certify p95. Candidate
counts, actual occupancy and replacement/rejection counts accompany each sample.

Complete primary matrix:6 arms*2 ledgers*6 orders*350=25,200 updates. Four boundary
snapshots evaluate16*(1+2+3+3)=144 held-out normal images per order/arm/ledger,
or10,368784-query NN evaluations overall. A conservative forecast also charges
one global NN at every writer arrival via the measured full cached-image wall;
those self scores are discarded and cannot be acquisition evidence or learning.
Eight reference-image scales/category add evaluator computation/I/O separately.

For each of12 method/ledger cells let u_m be1.25 times the maximum saturated full
cached-image wall (stationary AND high demand), and r_m be1.25 times its maximum
NN+bank-preparation+held-out-result-transfer wall. Then

`T_forecast = 2100*sum_m(u_m) + 864*sum_m(r_m) + T_fixed`.

T_fixed has explicit initial reservations:600s preflight,600s authorized calibration
and scale validation,300s analysis,300s I/O/hash/checkpoint residual overhead and
120s final verification =1920s. Reservations are NOT measurements. Replace each
with a conservative observed bound before lock; do not keep a300s I/O allowance
if actual cold reads/hash costs exceed it. Evaluation readers may decode large
cache objects even when delivering one causal row: those host costs must be
measured and included, never treated as free. Data access infrastructures and
evaluator witness archives are charged as infrastructure, never learner replay.

Admit only if the ENTIRE forecast<=6480s (10% reserve under7200s), reserved CUDA
<=4.5GiB and process selection/extraction RSS<=4GiB. Hard stop at7200s from start
of authorized calibration through final verification, including CPU/I/O/context
release time. Save complete failure matrix on stop. A1s p95 selection target or
20ms p95 NN target is supplementary and cannot establish full-run feasibility.

Predeclared cost-only reductions, BEFORE any evaluation-normal outcome: restrict
to K2500 already; eliminate per-update full checkpoints/hashes in favor of compact
decision counters and only the four boundary checkpoints; reuse the exact same
source/reference scales across methods as evaluator infrastructure. If still
over budget, STOP RESOURCE_INFEASIBLE. Do not drop reservoir/S4/R4, another arm,
a ledger, an order, a safety endpoint or a difficult input after exposure. A
smaller scientific matrix needs a new reviewed task, not an automatic reduction.
No real-normal run starts from a null/assumed GPU latency forecast.

## Current prerequisites and next authorization boundary

Required before a normal screen: independent OpenCode review of actual diff,
source pins/oracle and this protocol; accessible CUDA with completed12-cell
cost preflight and CUDA numerical/resume validation; custodian-verified cache/
normal provenance and versioned representation availability; explicit authorization
for disjoint calibration; a precision decision and complete<=6480s forecast;
then a SHA-256 execution lock naming all source/config/manifest/statistic inputs.
None of those later data phases is automatically authorized by R1A.

Do not train a model, implement ADSM, access MVTec official tests/anomalies/sealed
identities, reuse historical predictions, rerun a benchmark, edit historical
protocols/caches/checkpoints, or change Git state. Stop after this preparation.
