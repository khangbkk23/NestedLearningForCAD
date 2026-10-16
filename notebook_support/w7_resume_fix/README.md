# W7 JSON lock transport fix v1 — 2026-10-16

The failed 7.2 run stopped BEFORE any new matrix cell. The original run_lock
contains JSON lists while live predictor metadata contains tuples. Its canonical
digest is unchanged, but Python equality rejected resume.

This delivery leaves all 119 experiment Python/spec dependencies byte-identical.
The added launcher checks the pinned source map, JSON-normalizes only the proposed
identity, and calls the ORIGINAL strict lock function. It never deletes/rewrites
old locks, checkpoints, raw maps or metrics; altered beta/geometry/runtime/source
still fail. Each execution exports the exact helper plus a hashed receipt under
`runs/<JOB>/transport_fixes/w7_json_lock_transport_v1/`. This supplemental code is
part of the NEW source bundle manifest; it is not presented as original runtime code.

Notebook-only changes: identify W6 by its reviewed manifest hash (duplicate copies
accepted), run the guarded launcher, check completed 7.1 evidence before 7.2, and
raise failure AFTER exporting evidence. No model/learning/scoring revision.

## Kaggle

1. Replace the old source INPUT with `ultils/w7_resume_fix/hope_cad_w7_source_bundle.zip`.
   Attach only one source version. Kaggle extraction or intact ZIP both work.
2. Import the updated `notebooks/hope_cad_w7_7_2_kaggle.ipynb` from this delivery.
3. Keep completed W6, MVTec, ViT. Attach ONLY completed 7.1 evidence as previous W7
   input (460054213 bytes; SHA256 9bdf8089155e2c325e96057892d52db4d641391546680bc09d0bc765fd831e65).
   Detach the FAILED 7.2/preflight evidence from this notebook, but keep backups.
4. PHASE=7.2, JOB=M_seed0; W6_OVERRIDE=None now handles reviewed duplicate folders.
5. Save Version -> Save & Run All on the same T4/runtime. No W6/7.1 refit or
   new preflight is required: all frozen dependencies and the suite are unchanged.
6. Expect `[W7 LOCK FIX] ... 119 frozen files verified`, then `[W7 EVAL]` progress.
   Success is `7.2_complete_pending_review`; runner failure exports ZIP then raises.

Rebuild: `pixi run --locked python -B notebook_support/w7_resume_fix/build_delivery.py`.
Tests: `pixi run --locked python -B -m unittest discover -s notebook_support/w7_resume_fix -p test_*.py -v`.

This engineering fix does not improve or reinterpret 7.1 AD performance.
