# B325 job Q, seed 2711: port check (b325-q-20261119-03)

Exploratory (exposed MVTec test). This job checks the port; it does not answer the scientific questions. Label:
**`PORT_FAITHFUL`** (card sections 5 and 10), with BASE identified by its sidecar SHA-256 only (card section 9).
Full numbers: [metrics.json](metrics.json); environment, gates and hashes: [execution.json](execution.json). Two
earlier attempts of this job stopped on infrastructure errors before any data or model step (source loading; a
missing cache directory) and produced no results.

## Execution checks

- **Environment:** Python 3.13.16, torch 2.11.0+cu130, 2 × T4. Khang used torch 2.10, so torch version and CUDA
  build differ as well as the GPU.
- **Pre-data gate:** 21 driver tests and 30 of Khang's tests passed.
- **Data gate:** encoder hash `73182a08…`; `BASE.pt` hash `dfe6689e…`; all 3,629 train images byte-identical to
  Khang's manifests.
- **Training:** 15 tasks per arm, 300 steps and 30 credit evaluations per task; F kept γ = 2.1 and λ = 0 for all
  15 tasks (checked after every task); F 675 s, H 719 s of update time; CUDA peak 0.78 GB.
- **bf16 fidelity:** relative error 0.0172 (F) and 0.0173 (H), threshold 0.05; both arms scored in bf16.
- **Whole job:** 4,759 s, no traceback in the log.

## Comparison with Khang (same BASE, seed 2711, readout A1, macro over 15 categories)

| Arm | Final Pixel AP | Khang | Δ | Image AUROC | Khang | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F (γ = 2.1) | 0.6024 | 0.6013 | +0.0011 | 0.9772 | 0.9783 | −0.0011 |
| H (HOPE-S) | 0.5988 | 0.6006 | −0.0018 | 0.9760 | 0.9761 | −0.0001 |

γ trajectory of H at the pre-set tasks:

| Task | Ours | Khang | Relative Δ |
| --- | ---: | ---: | ---: |
| 1 | 0.361 | 0.360 | +0.2% |
| 4 | 1.240 | 1.164 | +6.5% |
| 8 | 2.198 | 2.053 | +7.1% |
| 15 | 3.520 | 3.441 | +2.3% |

The largest deviation over all 15 tasks is at task 7 (+8.9%), which is not part of the fingerprint. Acquisition and
FM are also close: Pixel AP at acquisition +0.0021 (F) and +0.0006 (H); Pixel FM +0.0021 (F) and +0.0003 (H).

**H − F at seed 2711, ours:** −0.0036, interval [−0.0090, +0.0021]. Khang: −0.0007, interval [−0.0100, +0.0083].

## Reading

1. **The port is faithful at the macro level.** Both Pixel APs within 0.002, Image AUROC within 0.0012, learned γ
   on Khang's trajectory. The B325 code path (new driver around Khang's unchanged learning and scoring functions)
   reproduces Khang's macro numbers on another GPU and CUDA build. Scope: this is fidelity of the functions called
   plus the Q gate (and later the C gate); the driver replaces Khang's orchestration and does not run his 10-step
   parity check against the original B14 path, as `hope_s_01.run` does.
2. **Per-category differences between the two runs are not small.** The largest are 0.038 (F, screw) and 0.023
   (H, zipper), with the same BASE and seed. The two runs differ in GPU, CUDA build, PyTorch version and
   orchestration at once, and no control isolates dropout or GPU, so no cause is assigned. Per-category comparisons
   need care; the per-category −0.01 floor is descriptive only, as registered.
3. **HOPE-S does not beat F at seed 2711, for the second time.** Two independent runs (Khang's machine and ours) give
   H − F near zero and slightly negative. This is not yet a Q-NL label, which uses seeds 2712 and 2713 only.
4. **A BASE generated from the seed on the T4 differs from Khang's BASE.** `model_new(2711)` on the T4 gives SHA
   `b8ccbbe7…`; the largest absolute difference from Khang's file is 0.0596, the same size as the weights themselves
   (`trunc_normal_`, std 0.01, cut at ±0.03): a different random draw, not rounding. GPU and PyTorch version cannot be
   separated as the cause. Consequence for B326 (M1): BASE cannot be stored as "seed + code" while it is generated on
   CUDA. Consequence for R: BASE for 2712 and 2713 is generated once per seed, saved and hashed, and shared by the
   three arms, as the card specifies.
5. **The train feature cache is not bitwise equal to Khang's (0/15).** Expected; informational only.

## Measured timings (used to set R's time limits)

| Phase | Seconds |
| --- | ---: |
| Setup, tests, input loading | 264 |
| Gate | 18 |
| Train feature extraction | 396 |
| Train (one lane, one arm) | 776 |
| Fidelity (two arms in parallel) | 112 |
| Test feature extraction, bf16 | 198 |
| Scoring (one lane, one arm) | 1,192 |
| Metrics (2 workers in parallel, one arm each) | 1,794 |
