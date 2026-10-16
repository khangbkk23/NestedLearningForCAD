# W7 JSON lock transport correction v1

Date: 2026-10-16. Scope: engineering resume/handoff correction only. The frozen
HOPE-CAD v4 architecture, objective, learning schedule, preprocessing, data roles,
scoring, baseline routes and registered W7 suite remain unchanged. No AD metric
is used to select or modify this correction.

## Observed failure and root cause

The 7.2 attempt stopped at `run_w7.py -> lock_json(run_lock.json)` before any
new evaluation cell. Saved JSON lists and live dataclass tuples differ under
Python equality despite identical canonical JSON/digest. Example: periods
`[1,8]` versus `(1,8)`. The original lock writer accepts tuples and its digest
already serializes them as arrays; its reader did not normalize the proposal.
The completed 7.1 archive remains intact.

A second handoff issue was that the final notebook cell exported failure evidence
but never raised after a nonzero runner exit, so a finished notebook could appear
successful while its handoff was `failed_preserve_evidence`.

## Correction and reproducibility boundary

Directly editing the existing Python runner would invalidate source/suite/session
identities tied to the completed 7.1 checkpoints. This delivery therefore keeps
ALL 119 original experiment dependencies byte-identical and packages a separate
launcher `notebook_support/w7_resume_fix/launcher.py`. This is supplemental runtime
code, explicitly not hidden or attributed to the original source identity.

The launcher verifies the complete pinned original source map, converts ONLY the
proposed lock identity through a strict JSON round-trip (`allow_nan=False`), then
calls the ORIGINAL strict lock function. Its existing-record equality AND digest
checks remain in force. Actual beta, geometry, checkpoint, manifest, runtime and
source changes still fail; no locks/checkpoint states are deleted or rebound.
The fix does not rewrite existing lock bytes or change their canonical digests.

The supplemental launcher is hashed by the new bundle manifest and copied into
every run output with a receipt containing its SHA256, pinned source-map digest,
phase/job, verified file count and exit code. Receipt path:
`runs/<JOB>/transport_fixes/w7_json_lock_transport_v1/`. This records actual
execution provenance alongside the immutable historical experiment identity.

Original suite digest: `17cd635b35e62484c5a000b4be4af503284e7d00da9aec79d64576c8d5055240`.
Pinned original source-map digest: `86c941b4aa1b2e82682c2d0ed31e97325b3e02442f6f88efa333511884db25ad`.
Parent source ZIP: `1b402113ae91aa2ee9f07ff976d3ad73d1fc173327bd8bc6d93c4c438473394a`.
Completed 7.1 ZIP: `9bdf8089155e2c325e96057892d52db4d641391546680bc09d0bc765fd831e65`.

## Notebook handoff

Updated notebooks select completed W6 by its reviewed manifest SHA rather than
rejecting equivalent extracted copies. 7.2 checks that final 7.1 metrics exist.
All four GPU phase notebooks use the guarded launcher. The final cell always
exports ZIP first, then checks BOTH runner exit code and expected handoff status;
failure is raised explicitly. A valid budget pause remains resumable.

Use the NEW source ZIP at `ultils/w7_resume_fix/hope_cad_w7_source_bundle.zip`,
the updated 7.2 notebook, completed 7.1 evidence, reviewed W6, MVTec and ViT.
Attach only one source and one latest W7 evidence. Do not use the failed 7.2 output
as a supposed completed matrix. No W6 or 7.1 retraining/new selection is needed.
The prior preflight applies to identical model/engine dependencies; local tests
cover this transport correction separately. Actual 7.2 GPU matrix acceptance is
still pending until its new saved output is reviewed.

## Validation artifacts

See `results/hope_cad_v2/w7_resume_fix/`: focused serialization/failure/export
tests; full existing W7/shared-v4 regression; real 7.1 lock/checkpoint compatibility;
and delivery checksum validation. Focused regression includes real predictor
metadata, JSON-to-disk handoff into matrix evaluation, no retraining or final-row
rescore, and rejection of actual identity/source/digest drift. None of these
engineering checks increases or validates the already measured AD performance.
