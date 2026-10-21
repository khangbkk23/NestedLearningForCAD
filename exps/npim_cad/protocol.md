# exps/npim_cad/protocol.md

# EXP-NPIM-00 — frozen planned protocol (normal-only)

Status: **PLANNED / FROZEN BEFORE THE FIRST NPIM REAL-DATA WRITER RUN IN THIS SESSION.**
This protocol is additive. It does not modify, replace or reinterpret HOPE-CAD,
ECAD-R1, CCNM, ContCore, P0/P1/P2 or any historical report, cache, checkpoint or
prediction. Scientific validity is **not** claimed.

## 1. Identity and source preservation

| Item | Value |
|---|---|
| Repository | `NestedLearningForCAD`, branch `dikhang_space` |
| Local HEAD at protocol write | `293dc2e46c36f088e6681f7be00c047cae7bc394` |
| `AGENTS.md` SHA-256 pin | `b88393815586295209c2704a6e16262730f2aba4000a6904b7c5c6d2b65ed622` (verified) |
| Frozen encoder | `checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth` |
| Encoder canonical SHA-256 | `755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415` |
| Encoder canonical bytes | `343,282,995` |
| Encoder public source | HF `timm/vit_base_patch8_224.augreg_in21k`, revision `3c68ba3fef589ef43b862ec19254ecb1146506c3` |
| Encoder provisioning code | `scripts/hope_cad/provision_vit_b8_checkpoint.py` (unmodified) |

The checkpoint is rebuilt by the repository's pinned provisioning script and is
a **new, explicitly versioned frozen representation**. It is not claimed
bitwise-identical to the vanished historical file
`ae3012808a9b406a19b799381bd26b253634ad26125937c26d02dbcbbc85dd92`.
Historical block-8 caches were produced with this canonical representation
(`checkpoint_sha256 = 755e9f5f…`), so Y parity to those caches is testable.

## 2. Preprocessing and representation

* RGB, PIL direct square bilinear resize to 224x224, `uint8/255`, then fixed
  dataset constants mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`.
  No per-image statistic, no augmentation, no random crop.
* `Z ∈ R^{784×768}`: **pre-attention** patch embeddings captured at a
  forward-pre-hook on `blocks[0]`. The tensor is
  `[CLS, 784 patch tokens]` after the non-overlapping `Conv2d(3,768,k=8,s=8,pad=0)`
  patch embedding, the fixed per-token positional embedding and the `p=0`
  dropout, and **before any cross-token attention**. CLS is dropped.
* `Y ∈ R^{784×768}`: output of block index 8 (one-based layer 9), final norm,
  CLS dropped — the same tensor as the existing CADIC/P1 readout.
* Architecture preconditions are asserted at load time: kernel/stride 8/8,
  padding 0, dilation 1, groups 1, patch-embed norm `Identity`, fixed
  `pos_embed`, `pos_drop.p == 0`, `patch_drop`/`norm_pre` identity, 12 blocks.
  Any failure stops NPIM; no fallback backbone and no substitute representation.

## 3. Leakage-safe context keys (Z only)

For target patch `p` on the 28x28 row-major grid, `N4(p)` are the valid
up/down/left/right positions; there is **no padding rule that copies `p`** at
borders (corners have 2 neighbours, edges 3).

```
zbar_N,p   = mean_{q in N4(p)} Z[q]
zbar_G,!p  = ( sum_{q != p} Z[q] ) / 783
u_p        = 0.5 * zbar_N,p + 0.5 * zbar_G,!p + gamma * e(p)
k_p        = normalize(P u_p, eps)
```

* The global leave-one-out sum uses prefix/suffix accumulators that never
  contain `Z[p]` (FP64 accumulation). The FP32 `sum(Z) - Z[p]` shortcut is not
  used.
* `P ∈ R^{64×768}` is fixed, seeded (`projection_seed = 20261021`), row-unit-norm
  FP32, persisted and SHA-256-pinned. It is not a learned adapter in NPIM-00.
* `gamma = 0` for EXP-NPIM-00. `e(p)` is implemented but inert; a coordinate
  ablation is a separate, later experiment.
* `eps = 1e-8`. Zero context yields a zero key (no NaN).
* `Y`, block-8 CLS and any deep-token neighbour are **never** context inputs.

## 4. Multi-clock proximal memory

Three levels with image-event periods and ridge strengths:

| Level | Period (completed normal images) | lambda |
|---|---:|---:|
| fast | 1 | 0.1 |
| mid | 16 | 1.0 |
| slow | 128 | 10.0 |

Per level, only sufficient statistics are persisted between commits:
`Skk = Σ k kᵀ` (r×r), `Syk = Σ y kᵀ` (d×r), `N_pairs`. At a due image event:

```
A = Skk / N,  B = Syk / N
W_new = (B + lambda W0) (A + lambda I)^{-1}     (Cholesky solve, never inverse)
```

then `W ← W_new`, accumulators are zeroed, `ready ← True`, `update_count += 1`.
Clocks count **completed normal images**, never patches, batches, task
boundaries or category transitions. There is no period reset at a category
boundary and no flush at evaluation/stream end. At image 350 the slow level
last committed at 256; its 257–350 sufficient statistics remain pending.

Readout: fixed registered weights `w = (1/3, 1/3, 1/3)` renormalised over
**ready** levels only. Before any level is ready the detector is explicitly
uncalibrated; a zero prediction is never treated as a normality model.

## 5. Innovation score and readout

* Default score: `s_p = mean_j (y_pj - yhat_pj)²` (L2). The diagonal-normalised
  score is **disabled** (`use_diag = False`) in this pilot. If enabled later it
  uses one fused-residual variance updated only from normal writer pre-update
  residuals, with an explicit floor; it is a separate registered arm.
* Map: `s.reshape(1,1,28,28) → F.interpolate(size=(H_orig,W_orig),
  mode='bilinear', align_corners=False, antialias=False)`, identical to
  `exps/task_p/p1_readout.py`.
* Image score: `max_p s_p`. Native Pixel AP uses exact tie-aware AP, never
  trapezoidal PR area. Interval units are whole images, never pixels.

## 6. Data roles and isolation

* **Writer stream (normal only):** the pre-existing resolved P0c order of 350
  `train/good` images: events 1–100 bottle, 101–200 carpet, 201–300 hazelnut,
  301–350 bottle. Exact identities and raw SHA-256 come from
  `results/hope_cad/task_p/p0c/protocol_resolved.json`; no reordering, no
  resampling, no task-ID input to any learner decision.
* **Normal witnesses:** the 59 content-disjoint `probes` rows of the same P0c
  protocol (`train/good` only). Used read-only for normal residual/stability
  telemetry. Never used for architecture, hyperparameter, checkpoint, scoring
  or threshold selection.
* **Never opened by NPIM-00:** official-test images, anomaly labels, ground
  truth masks, sealed confirmation payloads, masked-feature caches.
* No category ID, image label, mask, task boundary or test-specific calibration
  enters `score`, `commit`, key construction, clock decisions or mixture weights.
  Category may appear in provenance/reporting metadata only.

## 7. Arms (matched writer events)

| Arm | Levels | Periods | lambda | Key rank | Bytes |
|---|---|---|---:|---:|---:|
| C1 (single-rate, byte-matched) | 1 | (1) | 0.1 | 112 | ≈ C3 total |
| C2 (same-frequency) | 3 | (1,1,1) | (0.1,1.0,10.0) | 64 | = C3 |
| C3 (hypothesis) | 3 | (1,16,128) | (0.1,1.0,10.0) | 64 | = C3 |

C1 raises the key rank so its total persistent detector bytes (projection +
W + pending statistics + calibration) approximately equal C3's; the exact bytes
are measured and reported. C2 is C3 with all periods equal to 1, isolating the
multi-timescale hypothesis from "three different ridge strengths". All arms
consume identical writer images in identical order.

Deferred and **not implemented**: C4 self-modifying update. C0a/C0b ContCore
comparators are for a separately authorized Stage E only.

## 8. Execution gates

| Gate | Requirement | Failure action |
|---|---|---|
| G0 | branch/HEAD, clean tree, checkpoint SHA, data permission | BLOCKED |
| G1 | target-patch perturbation leaves `key_p` unchanged; autograd target gradient exactly zero | STOP learned-image work |
| G2 | `Z`,`Y` shapes `[784,768]`, same 28×28 order, Y parity to cached block-8 | STOP |
| G3 | Cholesky vs FP64 solve, SPD/strong-convexity, proximal identities | STOP |
| G4 | exact 1/16/128 image clocks and ready flags at 100/128/200/256/300/350 | STOP |
| G5 | sufficient statistics only; no raw patch replay | STOP |
| G6 | atomic commit, deterministic save/load/resume, rollback on nonfinite | STOP |
| G7 | read-only scoring leaves state, pending buffers, counters and RNG unchanged | STOP |
| G8 | native bilinear map, tie-aware AP, image max | STOP |
| G9 | exact byte ledger incl. projection, pending stats, variance, metadata, serialization | STOP compression claim |
| G10 | bounded CUDA smoke with VRAM/time recorded; safety budget ≤ 3.5 GB allocated by this process | STOP if exceeded |
| G11 | normal-only proxy only; anomaly DEV only under separate authorization | report negative |

## 9. Stage E authorization rule

Stage E (any anomaly label/mask read) is **not authorized** by this protocol.
`exps/npim_cad/evaluate_dev.py` refuses to run without a separate, explicit
authorization manifest naming the exposed DEV cohort, the frozen protocol
SHA-256 and the exact score-lock hash. NPIM-00 does not create, infer or
back-fill that authorization.

## 10. Pre-registered decision vocabulary

Only as evidenced: `BLOCKED_INPUT_LINEAGE`, `BLOCKED_LEAKAGE_FREE_CONTEXT`,
`NPIM_SYNTHETIC_MECHANICS_PASS`, `NPIM_VISION_MECHANICS_PASS`,
`NPIM_NORMAL_ONLY_PILOT_PASS`, `NPIM_NO_DETECTION_ADVANTAGE`,
`NPIM_COMPRESSION_ONLY`, `NPIM_PROMISING_DEV_ONLY`, `INCOMPLETE`.

Anomaly-efficacy threshold, only if Stage E is later authorized: a candidate
must reach at least **+0.01 absolute native Pixel AP** versus the strongest
matched baseline, with paired whole-image intervals and no category reversal.
Passing normal-only gates or improving a normal-only objective does **not**
establish anomaly detection efficacy.

## 11. Theory qualifications reproduced here

* The proximal update is lambda-strongly convex and has a unique minimiser.
* The proved one-update magnitude bound is
  `||W_new - W0||_F ≤ ||B - W0 A||_F / lambda`. It bounds **one update**, not
  general forgetting, old-category AP or memory compression.
* No universal anti-forgetting guarantee follows without task identity/replay.
* The clean-context positive-gap result requires `||δ|| > 2b` and a fixed
  uncontaminated context; real defects can contaminate neighbouring context.
* This is a multi-rate learned parameter memory, not a demonstrated full HOPE
  module or self-modifying system.

## 12. Reproducibility

Deterministic CPU execution (8 threads), fixed seeds, no dropout/augmentation,
batched writer order fixed, states written to new immutable paths with atomic
promotion, and every run manifest records protocol/checkpoint/config hashes.
Encoder extraction is the only expensive shared cost and is performed once per
writer image and once per probe image.
