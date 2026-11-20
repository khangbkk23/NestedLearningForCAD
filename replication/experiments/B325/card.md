# B325: independent replication of C, F and HOPE-S on new seeds (MVTec)

Registered 18 Nov 2026, before any run. Results:

| Job | Question | Label | Results |
| --- | --- | --- | --- |
| Q | does the port reproduce Khang's seed-2711 numbers? | `PORT_FAITHFUL` | [b325-q-20261119-03](runs/b325-q-20261119-03/results.md) |
| R | Q-NL and Q-HP on seeds 2712, 2713 | `NL_NULL_REPLICATED`, `HARDPATCH_SUPPORTED` | [b325-r-20261119-01](runs/b325-r-20261119-01/results.md) |
| C | does arm C through `acquire(γ=0)` match Khang's `fit_task` C? | `C_PATH_FAITHFUL` | [b325-c-20261119-01](runs/b325-c-20261119-01/results.md) |

Scope: MVTec only, three arms (C, F, HOPE-S), new seeds also change BASE. The MVTec test set was exposed during
development on both sides, so every result is exploratory. No VisA.

Reference: `docs/handoff/HOPE_S_HANDOFF.md` and Khang's sealed `RESULT.json` files, which Khang sent separately on
18 Nov 2026 (section 9). Every reference number below was checked against those files and matches the handoff text.

## 1. Questions

Khang's results had one seed (2711). One point the handoff states only implicitly: in `hope_s_01`, HOPE-S beat C by
+0.0108 (interval [+0.0017, +0.0198]); only after the post-hoc fixed-γ arm F was added in `hope_s_02` did the
HOPE-S gain over F become −0.0007.

- **Q-NL:** does the learned slow level (HOPE-S) beat the fixed γ (F) at new seeds? Khang: H − F = −0.0007.
- **Q-HP:** does the fixed hard-patch loss (F, γ = 2.1) beat the plain patch loss (C, γ = 0)? Khang: F − C = +0.0115,
  interval [−0.0018, +0.0261].
- **Q-SYS (descriptive):** how much does F vary with the seed?

Priors written before the runs: Q-NL null at both seeds; Q-HP positive but possibly below the threshold.

## 2. Code: reused unchanged and re-implemented

- **Reused unchanged** (Khang's `exps/`, MIT, commit 5217cb1; pinned in `replication/khang_source.json`):
  - front end: `dinomaly.encoder.load/preprocess`, `nar_b14_01.run.tokens` (DINOv2-reg B/14, 392 px, mean of blocks
    6–9);
  - model: `nar_pilot_00.full.model_new/rows_for`, `core.attach/init_state/get_state/set_state/normal_partition`;
  - learning: `hope_s_01.run.acquire`;
  - scoring: `nar_lite_01.run.lite_encoder/expert_grids/smooth392/cell_metrics`, `nar_pilot_00.detector.prefix_grids/
    fm_eq8`, `nl_sgm_01a.am03_eval.bootstrap_delta`.
- **Re-implemented** in one driver, `replication/scripts/hope_replicate.py`: the train loop, sealing, test
  extraction, scoring and metric loops. Khang's own versions of these steps read historical result paths
  (`HIST/BASE.pt`, `REF/RESULT.json`) and hash git-ignored files, so they cannot run outside his machine. The driver
  follows Khang's own seed-parameterised F runner (`exps/vbench_00/run.py`, function `train`), with VisA paths replaced
  by MVTec. **This is where the port could diverge**, which is why job Q exists.
- CPU gate at the start of each job: the driver tests and Khang's tests `nar_pilot_00/test_core.py`,
  `test_detector.py`, `hope_s_01/test_run.py`, `nar_lite_01/test_run.py`.

## 3. Seeds and arms

- **The seed s controls:** BASE (`torch.manual_seed(s)` in `model_new`, then `trunc_normal_` on CUDA), the LoRA
  initialisation (`init_state(s)`), and per-step sampling (`s + 100·stage`). The task order (alphabetical) and the
  FIT/CREDIT/EVAL roles (hash-based) do not depend on the seed.
- **Three arms per seed**, same BASE, same `acquire`, warm start from the previous task's endpoint, 300 steps per
  task, rank 8, 34 sites:

| Arm | γ | Slow level |
| --- | --- | --- |
| C | 0 | not learned (`learn=False`) |
| F | 2.1 fixed | not learned; slow-level gradients computed and discarded, as in Khang's design |
| H (HOPE-S) | starts at 0, learned | `learn=True`, η_γ 0.05, η_λ 0.2, top-8 tail, as Khang |

- **Our C is not exactly Khang's C.** Khang trains C with `fit_task` and checked only that its first 10 steps are
  bitwise equal to `acquire(γ=0)`. B325 uses `acquire(γ=0)` so that the three arms share one code path (checked by
  job C, section 11).
- **Implementation:** train in fp32; score with Khang's bf16 "lite" encoder (blocks 0–9) if Khang's normal-image
  fidelity gate passes (relative error ≤ 0.05), otherwise fp32 (fallback fixed in advance). Readout A1: bilinear to
  392, Gaussian σ 4, kernel 33, image score = mean of the top 1%. The raw readout is reported alongside.

## 4. Metrics (Khang's definitions, unchanged)

Macro over 15 categories of: final Pixel AP at original resolution (A1), final Image AUROC, Pixel AP at acquisition,
Pixel FM and Image FM (ReplayCAD Eq. 8). Per-category numbers are kept. Differences between two arms use Khang's
category bootstrap (B = 2,000, seed 20261117).

## 5. Jobs

**Q (seed 2711): F on GPU 0, H on GPU 1.** Q is a port check only if BASE equals Khang's: Khang always loads a sealed
`BASE.pt` and never regenerates it, and a BASE generated on another GPU may differ, which would make seed 2711 a
different seed in practice (seed noise ≈ 0.01, so port errors and BASE differences could not be separated).
Decision of 18 Nov: Khang sent the `BASE.pt` itself (section 9); Q **loads** it.

`PORT_FAITHFUL` rule (all three):
- F: final Pixel AP within ±0.005 of 0.6013 and Image AUROC within ±0.005 of 0.9783;
- H: final Pixel AP within ±0.005 of 0.6006;
- H, γ fingerprint (added 19 Nov, after receiving Khang's files and before any run): `gamma_end` at tasks 1, 4, 8, 15
  within 10% relative of 0.3603, 1.1645, 2.0532, 3.4407.

Otherwise `PORT_DIVERGES`: stop and diagnose, do not run R. Bitwise equality across GPUs is not expected, hence
tolerances. Superseded in part by section 10.

Extra CPU gates for Q: every MVTec train image must equal, byte for byte, the image Khang used (SHA-256 per image and
image count against his 15 manifests, bottle 209 … zipper 240); the encoder SHA-256 must be `73182a08…5dba71`
(`ENC_SHA` in `exps/dinomaly/common.py`); `BASE.pt` must be `dfe6689e…10cc31`. Informational only: the train feature
cache SHA-256 against Khang's manifests (not expected to match across GPUs), and a BASE generated with
`model_new(2711)` on the T4 compared with Khang's (the only way for Q to touch R's BASE path).

**R (seeds 2712 on GPU 0, 2713 on GPU 1, each GPU runs C, F, H in turn).** The frozen-encoder feature cache does not
depend on the seed and is extracted once. Time limits were set from Q's measured timings.

**Disk:** train cache fp32 ≈ 12.9 GB, test cache ≈ 4.2 GB (fp32). No fp16 caches, which would change the numbers.

## 6. Decision rules (fixed before the runs)

Labels use only the new seeds 2712 and 2713. Seed 2711 (Khang's, and ours from Q) is reported alongside.
Superseded by section 10.

## 7. Memory

Requested by Khang; see the separate card [B326](../B326/card.md).

## 8. Out of scope

No VisA. No change to γ, η, steps, rank or readout. No seeds added after seeing results. F is not to be called
"nested learning improves CAD". One run per job, no retry.

## 9. Files Khang sent separately (18 Nov 2026)

Kept outside git.

| File | SHA-256 |
| --- | --- |
| `BASE.pt` (three parts joined, 245,731,383 bytes) | `dfe6689e34d103c0a8b61931b71701c901cc4e9048ca1d04037bdf633d10cc31` |
| `hope_s_01/RESULT.json` | `e8f0392284074c5ae4bea4aeccce4c299741b8837e921f66fb02d3dea58e4e82` |
| `hope_s_02/RESULT.json` | `5c48a639ab0aa405a81f9643d39a74dad13ba0c093977008ac22b6b23419678e` |
| `nar_full_15/RESULT.json` | `e16e5979351dd322a9c1a397f25ecf093fee8d8357963763d80ea2eb3bf855d2` |
| `nar_warm_01/RESULT.json` | `8ff1d4a3f76a7ad2435998985bbb286b758743b1ad95d03a9aaa3c3252ecd1db` |
| `nar_b14_01/RESULT.json` | `07b0d3e6616ae00855a06b07bbbe45517311e7f017bc5563c7c2948fefa623c9` |
| 15 manifests `<category>_train.json` | hashes in `replication/configs/b325.json` |

**Known limit: the BASE identity rests on a sidecar.** The joined file matches the `BASE.pt.sha256` Khang sent with
it, which proves intact transfer only. None of the five `RESULT.json` files records `base_sha256`; that field is in
Khang's `LOCK.json` / `TRAIN_COMPLETE.json`, which we did not have. Decision of 19 Nov: accept the sidecar (the chance
that a wrong file was sent is very low) and ask for the `LOCK.json` line only if Q gives `PORT_DIVERGES`. The
`PORT_FAITHFUL` label therefore carries "BASE identified by sidecar".

**Reference numbers checked on file** (A1, macro over 15 categories, seed 2711):

| Arm | Final Pixel AP | Image AUROC | Pixel AP at acquisition | Pixel FM |
| --- | ---: | ---: | ---: | ---: |
| C | 0.5898 | 0.9649 | 0.5814 | 0.0057 |
| F | 0.6013 | 0.9783 | 0.5934 | 0.0031 |
| H | 0.6006 | 0.9761 | 0.5928 | 0.0052 |

**Environments.** Khang: RTX 4050 Laptop, PyTorch 2.10, CUDA 12.6 (confirmed 20 Nov 2026). Ours: 2 × T4,
PyTorch 2.11.0+cu130. Every Khang-vs-ours comparison therefore differs in GPU, CUDA build and PyTorch version at once;
differences (0/15 bitwise feature caches, CUDA BASE, per-category AP) cannot be attributed to the GPU alone. Labels are
unaffected: the tolerances in section 5 were set in advance precisely to absorb environment differences. (An earlier
version of this card assumed PyTorch 2.11 on Khang's side; corrected on 20 Nov.)

## 10. Revisions before writing code (19 Nov 2026)

These replace the corresponding parts of sections 3, 5 and 6.

1. **One precision per seed.** The bf16 gate runs per arm; if any arm of a seed fails, all three arms of that seed
   are scored in fp32. Labels use that common precision.
2. **No label without complete runs.** If a seed lacks a complete C/F/H triple, no Q-NL or Q-HP label is assigned;
   results are descriptive only. Labels are computed by the driver from the `RESULT.json` files, not by hand.
3. **One BASE per seed.** Generated once per seed, saved, hashed; all three arms load that file and the driver checks
   its SHA-256 before each arm. In Q the BASE is Khang's file.

**Decision rules:**
- **Q-NL** (Δ = H − F, final Pixel AP). H learns both γ and λ, so this compares learned (γ, λ) with fixed
  (γ = 2.1, λ = 0), not γ alone.
  - `NL_SIGNAL`: Δ ≥ +0.010 at both seeds and the lower interval bound > 0 (Khang's gate). Only these two conditions;
    the Image AUROC / acquisition / FM tolerances of Khang's `W.gate` are reported, not used.
  - `NL_NEGATIVE`: Δ ≤ −0.010 at both seeds.
  - `NL_NULL_REPLICATED`: |mean Δ| < 0.005 and |Δ| < 0.010 at each seed.
  - otherwise `NL_MIXED`.
- **Q-HP** (Δ = F − C; deliberately without the interval, since this is a descriptive effect-size question).
  - `HARDPATCH_SUPPORTED`: Δ ≥ +0.005 at both seeds and mean ≥ +0.010;
  - `HARDPATCH_NEGATIVE`: Δ ≤ −0.005 at both seeds;
  - `HARDPATCH_NOT_SUPPORTED`: Δ < +0.005 at both seeds, but not `HARDPATCH_NEGATIVE`;
  - otherwise `HARDPATCH_MIXED`.
  - The "threshold" of section 1 means these +0.005 / +0.010 values, not the per-category −0.01 floor (reported only).
- **Port check in Q** (replaces the `PORT_FAITHFUL` rule of section 5): the AP / AUROC conditions decide whether R
  runs. If they hold but the γ fingerprint is off by more than 10%, the label is `PORT_FAITHFUL_GAMMA_DRIFT` and R may
  still run: γ accumulates over 450 credit steps of noisy cosines, so drift across GPUs is plausible. If they fail:
  `PORT_DIVERGES`. Another known source of difference: the decoder bottleneck has dropout 0.2 driven by the CUDA RNG,
  whose masks can differ between GPUs at the same seed.

**Driver details fixed in advance:** call `R.configure()` (deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG`, TF32
off, 5 GiB limit); set `R.loss = P.patch_loss` so the before/after `normal_eval` logs match Khang; set `R.SEED = s`
before `model_new`, `init_state` and every `acquire`; record torch and CUDA versions; extract test features once per
scoring precision with the lite encoder in that dtype (bf16 ≈ 2.1 GB, fp32 ≈ 4.2 GB). F keeps γ = 2.1 at every seed
and γ is not re-derived from the new seeds' trajectories. Every statement about F carries the note that γ was chosen
post hoc at seed 2711.

**Implementation (19 Nov).** Driver `replication/scripts/hope_replicate.py`, config `replication/configs/b325.json`.
The runs used the platform's Python with torch, scipy and scikit-learn preinstalled (Khang's code needs scipy via
`calm02.native` and scikit-learn for AUROC and the bootstrap); versions are recorded per run. Pre-data gate:
`replication/tests/test_b325.py` plus Khang's four test files; any failure stops the job. The torch-dependent tests
could not be run on the authoring machine (no torch there); they ran for the first time inside each job.

## 11. Job C: code-path check for arm C (added 19 Nov 2026, after job R, before running job C)

Purpose: close limitation 2 of Q-HP (our C runs `acquire(γ=0)`, Khang's runs `fit_task`).

- Seed 2711, Khang's `BASE.pt` (as in Q), arm C only; same driver, gates and readout; bf16 scoring if fidelity passes.
- Reference: `c_a1` in Khang's `hope_s_02/RESULT.json`: Pixel AP 0.5898, Image AUROC 0.9649.
- `C_PATH_FAITHFUL`: final Pixel AP and Image AUROC each within ±0.005 of the reference; otherwise
  `C_PATH_DIVERGES`, in which case limitation 2 of Q-HP stays open (R is not re-run).
- Job C does not change R's labels.
