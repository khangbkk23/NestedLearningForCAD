# CADIC standalone diagnostic — K2500/K5000

Contract `cadic_fulltrain_diagnostic_v1`. A portable **host/pipeline diagnostic**,
not a new method or an acceptance bypass for `research`.

## Frozen scope

- Default two independent fresh banks, K=2500 then5000. Each consumes all3629
  MVTec train/good normals in the canonical15-category, lexicographic image order.
  K means **patch vectors**, not images. No sampling, shuffle or merging images.
- Existing checkpoint SHA, frozen ViT-B/8 block8 zero-based, raw FP32
  [1,784,768], ImageNet normalization/direct224bilinear. Retain TF32/AMP-off
  numerical policy,2CPU threads, deterministic algorithms, imagewise CADIC
  Eq1–7 and full Eq9 b9. Existing cached pair/fixed-shape NN backend, all distance
  chunks256 as in the registered research route (only K changes).
- Direct bank updates. No utility learner, proposal fork, anchors, normal-risk,
  per-image bank hashes, learner snapshots, or study gates in the timed path.
- This command intentionally runs **before study preflight**. It checks actual
  Linux/CUDA/package/checkpoint/data/source identities and a small CUDA smoke
  first; it does not claim study preflight/pilot acceptance or paper-exact parity.
- No source/runtime/data identity migration. Resume only identical bindings.
  A new host with changed runtime gets a new run-dir and fresh runs. No main,
  pilot or historical checkpoints can be imported as diagnostic state.

## Timing and correctness

Report synchronized wall components: `input_seconds` (file hash, decode,
resize/normalize), `features_seconds` (transfer, ViT and feature finite checks),
`update_seconds` (bank.update). Sum is named `pipeline_seconds`; report whole
invocation, checkpoint/QA time and export separately. Each component boundary
synchronizes CUDA: this gives comparable host wall scopes but adds measurement
overhead and is not a CUDA-kernel profiler or guaranteed author timing scope.

Fresh extraction is repeated for each K; no second arm gets a precomputed feature
cache advantage. OS file caches may still warm. Bind source/runtime/checkpoint,
full image hashes, bank config and producer. Record per-image replacement counts,
per-category totals and progress, so cheap later categories are not extrapolated
from Bottle. No automatic tuning or hardware ranking from one run.

Validate finite/raw shape, bank count/counters and every image identity. At
completion, check bank serialization roundtrip, read-only finite Eq9 scoring and
reference/cached pair/update/score parity on a discarded four-patch clone. These
QA operations run outside `pipeline_seconds`; they are finite fixtures, not proof
of author parity or absence of all bugs. Save final bank and hashed state.

Paper TableVII reports MVTec K2500=63s,K5000=379s on RTX4070/i5-13600KF.
Timing inclusions and author implementation/numerical identity are not fully
established. Show both pipeline and update-only scopes; do not call a ratio a
hardware speedup. Source: https://arxiv.org/html/2511.08634v1#S4.T7

## Operation

After bootstrap/Pixi/assets, **before research preflight**:

```bash
bash scripts/remote/run.sh cadic-diagnostic \
  --phase train --run-dir workspace/experiments/cadic_check_01/run \
  --max-hours 1
```

Default GPU is cuda:0; actual GPU is detected and recorded. Optionally assert
`--expected-gpu 'NVIDIA GeForce RTX 3090'`. No edits to scientific resources,
no allocation questions. `--max-hours` is an invocation deadline (0,4]hours,
not permission for unlimited rental; runner never Stops/Deletes the instance.
Checkpoint after64images, at task boundaries, pause or completion. Reissue the
same command to resume. Bounded operations/export can overrun the soft deadline.
Do not delete writer locks without verifying the old process has exited.

State plus committed timing/trace use two atomic checkpoint slots and a hashed
pointer. Crash replays only the uncommitted suffix. Retain all invocation receipts
and cloud billing; unrecorded crash/idle/setup time is not silently a fresh-run
timing. Source/runtime/budgets list/backend/preprocessing changes reject resume.

Progress is printed every16images. `diagnostic_train_complete` means both selected
K runs and QA finished. A pause is not completion; a failure exports evidence and
does not become PASS. Every invocation exports verified `evidence.zip` to a fresh
`exports/session_*`; logs live beside run/. Preserve the entire experiment and
records, verify backup, then follow the user's Stop/Destroy policy.
`run/summary.json` and the final terminal JSON show timing/QA for every completed
K; `diagnostic_lock.json` identifies the host, CPU, GPU, driver and producer.

## Optional quality check — separate timing

```bash
bash scripts/remote/run.sh cadic-diagnostic \
  --phase evaluate --run-dir workspace/experiments/cadic_check_01/run \
  --max-hours 1
```

Requires **both selected K trainings complete** before discovering official test.
Uses the existing evaluator (1725test images,15final cells/K,raw Eq9,bilinear maps,
nonzero nearest masks,macro I-AUROC/P-AP). It does not train again or construct a
120-cell matrix. Per-cell chunks/hashes permit resume and raw recomputation;
train state must remain unchanged. Test metrics never choose K, preprocessing or
hyperparameters; MVTec is development-exposed. A timing/normal-QA PASS alone does
not establish anomaly quality, paper reproduction, or H32 benefit.

Select only one fixed budget with `--budgets 2500` if needed; default comparisons
use both. Selected budgets are part of the lock and cannot be changed on resume.
Use fresh run IDs for repeated measurements; retain slow/failed results too.
