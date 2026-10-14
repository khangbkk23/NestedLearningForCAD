# Nested Learning for CAD — normal-only diagnostic r1 results

## Status and evidence boundary

The bounded forensic run is complete. It did not fit a new initialization, select
hyperparameters, access anomaly test data, or change the frozen v3 source/specs.
It did run the registered inner CMS updates on independent diagnostic states.

- Protocol: `normal_only_diagnostic_protocol_v1.json` and its companion explanation.
- Runner: `scripts/hope_cad_v2/diagnose_normal_forensics.py`.
- Raw evidence: `results/hope_cad_v2/normal_forensics_r1/report.json`.
- Compact evidence: `results/hope_cad_v2/normal_forensics_r1/summary.json`.
- Input W6 ZIP SHA256: `5706cd906d27f686ad0e1132100dfd29fcf67d18a18a47055e048d0f88879544`.
- Protocol SHA256: `83bf6a567cec7cf38659bc670f8fd3c4a3229380485e702cb43d3c346afba170`.
- Runner SHA256: `4a2e16772aa108629a41de1bd209dae4c227788d824aded1edf3ba87011705a0`.
- Raw report SHA256: `5983fc811a7f4dcbd9c99ca835065a1ec4e96ca7d25cfb79b7447a4c68538ab3`.

Execution used the local RTX 3050 Ti Laptop GPU, PyTorch 2.12.0+cu126, FP32,
highest matmul precision and disabled TF32. Elapsed time was 438.63 seconds
(about 7 minutes 19 seconds), peak allocated GPU memory 413,174,784 bytes.
This is local forensic evidence, not a new Kaggle selection gate.
Host peak-memory accounting was unavailable and is recorded as null.

The two new intervention-bookkeeping tests passed. They are not a rerun of the
original 83-test Kaggle suite. The original W6 ZIP and all 64 archived source/spec
hashes were checked unchanged after the diagnostic. Selected slow/initial weights
were checked unchanged for every arm. Diagnostic states were not exported.

Logical read counts: 350 canonical stream image-views, 128 derivative fixture
image-views, 32 spatial fixture image-views and 1,320 probe prediction image-views.
Only the first-task normal caches were used; no raw dataset/backbone extraction
or future-task feature access was needed.

## Same-history trajectories

Each dynamic arm starts once from its parent selected initialization and consumes
the same canonical 50 support + 20 query images. Probe measurements at 0/16/32/70
are successive points of that one history. Lower normal loss is better; these
numbers are not anomaly-detection metrics.

| Arm | Event 0 | Event 16 | Event 32 | Event 70 |
|---|---:|---:|---:|---:|
| M | 200.972599 | 0.195559 | 0.217133 | 0.239584 |
| A1, ordinary C1 initialization + M beta | 0.126801 | 0.251171 | 0.270749 | 0.259642 |
| M, visual off during adaptation and scoring | 201.151091 | 0.195634 | 0.217321 | 0.239661 |
| M, first CMS weights frozen | 200.972599 | 239.563633 | 24.719089 | 2.254293 |
| M, second CMS weights frozen | 200.972599 | 1.360275 | 0.915512 | 0.827093 |
| C0 static | 0.117644 | 0.117644 | 0.117644 | 0.117644 |
| C1 static CMS | 0.126801 | 0.126801 | 0.126801 | 0.126801 |

For M/A1/visual-off, actual CMS weight applications are `(70,8)`. First-freeze
has `(0,8)` and second-freeze `(70,0)`, while scheduled commit counters still
follow `(70,8)`. C0/C1 have no inner applications or CMS clock advancement.

M worsens by about 22.51% between the canonical event-16 and event-70 probe
measurements in this short stream. This now is actual longitudinal evidence.
It is still not catastrophic forgetting across tasks. The earlier W6 report's
random-16 result (0.197150) and canonical-70 result (0.239584) were independently
initialized histories and should not themselves have been called a trajectory.

Both CMS levels are useful to this already co-adapted M initialization: freezing
either from event zero severely degrades fitting. This does not prove that two
timescales beat a separately trained one-level/equal-period model, and is not an
argument for simply deleting a CMS level. M also fits better than A1 under the
same dynamic update recipe. Meta calibration is therefore not wholly ineffective.

## Local/final gradient alignment

Eight canonical support images, four views each, were evaluated independently at
states 0 and 16. The 32 views per row are dependent observations from eight images,
not 32 independent experimental replicates. The local derivative is of the first
CMS auxiliary loss; the final derivative is of final loss with respect to those
same first-level weights. Hypothetical updates operate on discarded separate states.

| Model/history | Median cosine | Negative cosine | One hypothetical event increases final loss |
|---|---:|---:|---:|
| M / zero | 0.846064 | 0/32 | 0/32 |
| M / 16 events | 0.139523 | 0/32 | 11/32 |
| A1 / zero | 0.148152 | 2/32 | 32/32 |
| A1 / 16 events | -0.339640 | 32/32 | 32/32 |

At A1's adapted state, every measured first-level local direction conflicts with
the corresponding final-loss derivative. The mean local auxiliary loss is
0.354240 while mean final loss is 0.250036. At zero history A1 already has good
final fitting (mean 0.126897) but much worse intermediate teacher fitting
(mean 2.102594); online updates try to repair an intermediate output that the
second layer already compensates for. Curvature/finite update size also matters:
positive cosine does not prevent a finite update from worsening final loss.

For M, the conflict is less severe: there are no negative first-level cosines in
the measured fixtures, but 11/32 finite next events worsen the same-image final
loss after adaptation. Thus local-objective conflict is established for these A1
states, not a universal explanation of all M errors. Meta learning partially
compensates for the selected local dynamics.

## Spatial contribution over multiple image/views

Across 32 support image-views and four directions:

- Median center-to-border readout RMS ratio: `5.42909e-16`.
- Maximum ratio: `7.60788e-16`.
- Median center RMS: `1.59907e-18`; maximum `1.89395e-18`.
- Median border RMS: `0.00270412`.
- Mean branch/input RMS ratio: `0.00333102`.
- Last-chunk alpha is FP32 `0.899999976` on every measured fixture.

This is a robust local spatial-contraction observation, not just a single-image
or final-state artifact. The frozen ViT and residual/FFN remain active, so it does
not mean the entire detector is blind in the center.

Visual off during BOTH adaptation and scoring changes event-70 mean loss by only
`7.60546e-5` (about 0.03174%). It is a more complete intervention than W6's
inference-only visual-off measurement. It still uses the same learned slow weights
and does not measure a retrained visual-off architecture.

## Result interpretation and next decision

The findings strengthen two concerns: the row-spanning visual core collapses
spatially, and auxiliary local teacher objectives can oppose final teacher fitting.
M's zero-clock error and finite-step behavior also justify a startup/horizon
correction. No hidden detach, cache, loss-reduction or checkpoint mistake was found
that would erase the unfavorable W6 evidence.

The 200-step W6 cap is still an optimization limitation: all selected checkpoints
are at that boundary and validation was still improving. This forensic run did not
extend fitting and cannot claim that more steps would solve the problem.

Do not start the full W7 suite automatically. The recommended next architecture
checkpoint is described in `normal_only_architecture_checkpoint_v1.md`. It is a
single proposed corrective revision, not an implemented or validated method. If
v3 is benchmarked instead, it must retain its existing identity and all negative
evidence. Any revision based on these inspected normals must disclose development
exposure; the probe cannot become untouched confirmation again.
