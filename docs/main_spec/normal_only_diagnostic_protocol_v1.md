# Nested Learning for CAD — normal-only forensic protocol r1

This is a bounded diagnostic of the existing v3 checkpoint, not a new method,
calibration search, benchmark, or replacement for the frozen v3 specifications.
The machine-readable policy is `normal_only_diagnostic_protocol_v1.json`, created
before this diagnostic run. Its hash is stored in the resulting report.

## Reference and data permissions

- Worktree: `D:/src/DATN/hope_cad_v2_2`, branch `hope_cad_v2_2`.
- Reference source: W6 archive based on `d78faa69c066094deb9bb9decc59df84b5d470db`
  with the archived dirty-source fingerprint and explicit original file hashes.
- Input: `docs/w6_kaggle/hope_cad_v2_2_w1_w6_evidence.zip`, SHA256
  `5706cd906d27f686ad0e1132100dfd29fcf67d18a18a47055e048d0f88879544`.
- Stream: exactly the 50 `meta_support` and 20 `meta_query` normal bottle images.
- Derivative/spatial fixtures: first eight canonical support identities, all four masks.
- Evaluation: all 15 previously exposed normal probe images, read-only.
- No official test images, anomaly labels/masks, future-task features, outer fitting,
  rate search, checkpoint selection, or deployment initialization export.
- The archive membership/CRC/SHA, original source hashes, cache contracts and
  selected-initialization receipts must validate before measurements begin.

The probe has already been inspected during W6 review and is development/forensic
evidence for any subsequent design revision. It must not be described as untouched
confirmation. Reusing it does not authorize fitting or selecting parameters on it.

## Measurements

1. **Continuous deployment history:** initialize once per arm, consume the same
   canonical 70-image stream with view `event_index % 4`, and measure probe loss at
   events 0/16/32/70. The states at these events are successive states in one history.
   They must not be confused with W6's independently initialized random-16 and
   canonical-70 histories.
2. **Local/final alignment:** at states 0 and 16, compare the partial derivative
   of the first-level auxiliary loss with the derivative of final loss with respect
   to that same level's weights. Record dot product, cosine and both norms. On a
   separate state, measure the final loss after one hypothetical registered update.
   This state is discarded and never enters the canonical stream or saved model.
3. **Spatial contraction:** on 8 x 4 support fixtures, record all four directional
   spatial readout RMS, center rows/columns 12:16, outer-two-row/column border,
   branch/input RMS, first/last chunk retention and final map RMS. Only the actually
   dynamic M operator is used for this measurement.
4. **Update-path interventions:** M full; A1; M with fixed-map visual readout during
   both adaptation and inference; M with first-level weights held fixed; M with
   second-level weights held fixed. C0/C1 are static references.

Weight-freeze interventions still compute both local gradients and follow the
registered event clocks, accumulator schedule and scheduled commit counters. The
frozen level's preceding weights are retained. Actual applied-weight-update counts
are reported separately. These are diagnostic interventions, not independently
trained deployable controls, resource optimizations or newly validated methods.

## Interpretation and technical checks

- No post-hoc numerical gain cutoff is introduced.
- Normal prediction loss does not establish anomaly separation, AUROC/AUPR, or
  retention across categories. One category and eight fixtures limit generalization.
- A small/negative local-final cosine is evidence about that state/fixture, not a
  universal theorem of conflict. A positive first-order dot product does not ensure
  that a finite nonlinear update decreases final loss.
- Freeze/visual-off outcomes concern the same learned weights and the declared
  adaptation histories; they do not estimate a retrained variant's performance.
- GPU/runtime differences from Kaggle are disclosed; this local diagnostic is not
  a new Kaggle scientific selection gate.
- Parameters, saved initializations and frozen v3 source/spec files remain unchanged.
- Read accounting distinguishes stream views, derivative fixtures, spatial fixtures
  and probe prediction views instead of calling provider-only counters total reads.
- A failed run retains its report. A rerun requires a new output directory.

The diagnostic is implemented by `scripts/hope_cad_v2/diagnose_normal_forensics.py`.
The freeze bookkeeping is checked by two focused tests in
`tests/hope_cad_v2/test_normal_forensics.py` before running on real normal features.
After completion, a separate report will state which mechanisms are supported,
which are unfavorable/inconclusive, and whether to keep v3 or propose a registered
revision before anomaly-test evaluation. The existing v3 documents are not edited
by this diagnostic.
