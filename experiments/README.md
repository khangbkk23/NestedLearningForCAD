# Experiment records

This repository currently supports the B319 seed-0 workload. An experiment is a
question/design/decision; a run is one execution. New experiments get their own
card. Shared source stays in `src/`, recipes in `configs/`, and launchers in
`scripts/` and `notebooks/`. A recipe has one authoritative copy.

## Index

- [Lịch sử v6/v7 → hiện tại](HISTORY.md): sáu mốc hồi cứu.
- [Catalog](history/catalog.json): records, aliases, dependencies và metrics nhỏ.
- [Audit](history/audit.json): verification theo record, gaps và nguy cơ độc bản.

| Card | Trạng thái |
| --- | --- |
| [v7/CADIC-NL](H01-v7-cadic-nl/card.md) | Historical admission/internal-control evidence |
| [D0/D1](H02-d0-d1/card.md) | Normal-only/diagnostic; no F2 acceptance inferred |
| [D2](H03-d2/card.md) | Mechanism separated from AD disposition |
| [Dinomaly](H04-dinomaly/card.md) | Detector transition and population changes |
| [B317](B317/card.md) | Scoped numerical/context qualification |
| [B319](B319/card.md) | Seed0 workload; recorded reproduction, screen not met |

## Historical evidence rules

- `source_fidelity` classifies evidence bytes: `exact_bytes`,
  `exact_reconstruction`, `semantic_derivative`, `reference_only`, `unknown`.
  `executed_source_fidelity` separately describes retained execution-code
  identities. Neither certifies scientific validity.
- Measured and declared hashes are separate. Exact reconstruction requires a
  known original hash match. Changed-link/semantic editions are not raw bytes.
- Purpose/hypothesis requires an explicit contemporary source. When ordering
  cannot be proven, preregistration stays `not_proven`; later interpretation is
  retrospective, never silently assigned as the original hypothesis.
- Every record has its own verification pointer. Generic read/hash entries do
  not validate raw ranks, behavior, usefulness, or phase closure.
- Public history dùng source/artifact IDs và hash thay đường dẫn riêng. Catalog
  records là semantic derivatives; source entries là reference-only đối với
  người đọc public. Exact payloads/selectors được maintainer giữ, chưa công bố.
- Public `publication.json` bên từng run là derivative có hash receipt gốc, không
  giả thành terminal receipt nguyên bản. Matrices và results giữ số liệu gốc.
- Exporters vẫn sinh full receipts phục vụ audit nội bộ. Trước lần public tiếp
  phải review disclosure projection; không tự overwrite producer hoặc bỏ failure.

## Before execution

1. Write the question, hypothesis, primary metric, arms, controls, data exposure,
   seed plan, decision rule, caps and stopping conditions in the card. Mark
   exploratory/confirmatory/replication/diagnostic explicitly.
2. Commit the card and recipe before the first new run. Packaging/training checks
   these files are committed and unchanged; dirty code is allowed only because
   the exact source bytes are bundled. Do not claim retrospective registration.
3. A code commit, full source bundle, recipe, train/evaluation manifests, lock and
   launcher jointly define the execution. A hash alone cannot replace missing
   bytes. For local execution, keep the source worktree unchanged while a run is
   active; remote runs execute their packaged snapshot. Preserve the bundle or a reconstructable commit+patch, and its locator.
4. For independent confirmation, freeze the candidate and holdout contract before
   accessing outcomes. MVTec test has informed development, so claims based on
   its observed results remain exploratory. The user reports VisA has not been
   used to select/tune the current method (2026-11-11); VisA is a reserved
   candidate, not an approved/frozen evaluation protocol. No VisA execution yet.

## During and after execution

- Live `run.json` may change. Every execution entering training, every intentional
  stop, and all failed attempts are retained. Never select only favorable runs.
- Training seals `receipts/receipt.json` on normal completion, exception or
  KeyboardInterrupt. A hard kill can prevent sealing: retain the live record,
  provider status and logs; document the interruption instead of inventing a
  successful end. Provider terminal receipts cover remote cancellation/failure.
- Each receipt records source identity/bytes, command, seed/arm, config/manifest/
  lock hashes, observed runtime/device, start/end/state/reason, parent run and
  checkpoint (null for a fresh run), and artifact locators. Unknown means unknown.
- Keep full boundary-by-task matrices of pixel AP/AUROC and image AP/AUROC, with
  null for unseen tasks, plus `metric_version`. Retain per-seed values. A seed
  distribution measures uncertainty; it does not define numerical replay tolerance.
- `summarize.py` generates `metrics.json`, `summary.json`, and `results.md` from
  stage records. Reports do not replace producer evidence. Numerical/reporting
  corrections are new analyses; do not edit a closed producer receipt.
- Import **all** terminal runs using `scripts/record_run.py` or automatic Kaggle
  `fetch`. Commit every exported receipt and available matrix before the next
  scientific comparison/report. Export never auto-commits or pushes Git.
- Closed receipts are write-once. A new design/seed/attempt gets a new run ID.
  A new analysis version must preserve the previous analysis and explain changes.
  Use `record_run.py --analysis-id <new-id>`; analysis script bytes are retained
  under the run audit directory, identified by their SHA256.
- Source/raw bytes live under `.local` or a recorded recoverable external store;
  compact receipts/matrices/results live here. A Git clone alone need not contain
  artifacts, but must identify which bytes are needed and where they are held.

## Commands

```sh
pixi run --locked -e cad-test test
pixi run --locked -e cad package --owner qoctrn --run-id <new-run-id>
# On Linux GPU with prepared inputs:
pixi run --locked -e cad train --run-id <new-run-id>
# After a local terminal execution, including a failure:
pixi run --locked -e cad python scripts/record_run.py --run .local/runs/b319/<run-id>
```

Kaggle evidence import uses the existing Kaggle CLI environment (not another ML
lock): `python scripts/kaggle_run.py fetch --kernel owner/kernel/version
--run-id <run-id> --token-file <private-token-file>`. Credentials never enter a
bundle or receipt. Keep raw outputs/bundles until a verified alternate copy exists.
Historical cards cite receipt/code/note sources; unsupported fields stay unknown.
No requirement to revive historical implementations in the active source tree.
