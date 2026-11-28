# RIGEL-CAD

RIGEL-CAD is a continual anomaly detector for a stream of industrial categories. It learns from normal images only, keeps
a fixed persistent-memory budget (8 MiB by default) and needs no task identity at test time.

Version 1.1 adds an optional compact storage format, `int4_triangular`. It stores the same two memories in fewer bytes
per element, so more exemplar keys fit in the same budget. The default format, `fp16`, is RIGEL-CAD 1.0 unchanged.
Version 1.1.1 is an engineering release: faster coreset fit, image decoding and scoring, with bitwise-identical results.

## Architecture

A frozen self-supervised ViT encodes each image into 28 × 28 unit-norm patch tokens: DINOv2 ViT-B/14 with 4 registers,
at 392 px. Two memories with different update frequencies sit on top of the encoder.

| Memory | Class | Update | Content |
|---|---|---|---|
| **Exemplar Memory** (fast) | `FastKeyMemory` | Once per task. A 1 % greedy k-centre coreset of the task's tokens is computed. Then {keys} ∪ {new coreset} is consolidated back to the key capacity by a seeded greedy k-centre; survivors keep their insertion order | key bank K (M × 768) |
| **Residual Geometry Memory** (slow) | `SlowResidualWhitener` | Accumulated over the whole stream: uncentred second moment S of the residuals r = x − (nearest coreset key of its task), over the task's non-key tokens; plus the count n | S (768 × 768) and n |

**Read-out.** The slow memory defines a whitening metric W = shrink(S / n, 0.5)^(−1/2), with linear shrinkage towards
the scaled identity. A query token q is scored by its whitened nearest-key energy:

    e(q) = ½ · min_m ‖qW − mW‖²

**Anomaly maps and scores.**
- Anomaly map: 28 × 28 energies → bilinear 392 × 392 → Gaussian smoothing (σ = 4).
- Image score: the mean of the top 1 % of the map.

**Not stored and not trained.**
- The encoder is shared and frozen, and is not counted as continual-learning state (144.8 MB, reported separately).
- Nothing is trained by gradient descent.

## Storage formats

The storage format decides only how the two memories are represented in memory and on disk. Every format uses the same
memories, the same update schedule, the same consolidation rule and the same read-out.

| `storage` | Exemplar key | Residual statistic | Keys in 8 MiB |
|---|---|---|---|
| `'fp16'` (default) | fp16 unit vector, 1,536 B | full fp32 768 × 768 matrix + fp64 count, 2,359,304 B (accumulated in fp64) | 3,925 |
| `'int4_triangular'` | signed int4 codes with per-key absmax scaling: scale = fp16(max\|x\| / 7), q = clamp(round(x / scale), −7, 7), two codes per byte (384 B) + fp16 scale (2 B) = **386 B** | fp32 upper triangle (295,296 values, diagonal included) + fp64 count, **1,181,192 B** | **18,672** |

**How `int4_triangular` runs in the stream.** It is used throughout the continual stream; it is not applied to a
checkpoint after training.
- Each new coreset is quantised when it is written.
- The k-centre consolidation runs on the dequantised candidates, so the keys that are selected are the keys that are
  scored.
- After every task, S_t is added in fp64 to the unpacked triangle, and the sum is rounded back to the fp32 triangle.
- Scoring uses the dequantised fp32 keys (q · scale) and the W computed from the triangle.

**Restrictions.**
- `int4_triangular` needs a finite budget, `slow_level=True` and `selection='kcenter'`. Other combinations raise
  `ValueError`; they were not part of the verified configuration.
- The key capacity is ⌊(budget − slow bytes) / key bytes⌋ for either format.

## Install

You need Python ≥ 3.10, `torch`, `numpy`, `pillow`, `scikit-learn`, `scipy` and `opencv-python`.

Optional packages:
- `timm`, for the ViT-B/8 backbone;
- `tifffile`, for the MVTec AD 2 submission files.

Run every command from the repository root; the package is `rigel_cad/`. The package does not import `exps/`.

**Encoder checkpoints.** They are verified by sha256 and loaded strictly.

| Backbone | Path | sha256 |
|---|---|---|
| default | `checkpoints/dinov2/dinov2_vitb14_reg4_pretrain.pth` (official DINOv2 ViT-B/14 reg4) | `73182a08…ba71` |
| optional | `checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth` (timm ViT-B/8 IN-21k) | `755e9f5f…a415` |

## Python API

```python
from rigel_cad import RigelCAD, RigelConfig, readout
from rigel_cad.encoders import build_encoder

enc = build_encoder('dinov2_vitb14_reg4', device='cuda')
model = RigelCAD(RigelConfig(budget_mib=8))                             # default: fp16 storage
# model = RigelCAD(RigelConfig(budget_mib=8, storage='int4_triangular'))  # compact memory

for task_images in stream:                                  # lists of normal training image paths, one per task
    tokens = enc.encode_paths(sorted(task_images))          # [n, 784, 768] fp16 unit tokens
    model.fit_task(tokens.reshape(-1, 768).cuda())

grid = model.patch_energy_grid(enc.encode_paths(test_paths))   # [n, 28, 28] patch energies
maps = readout.anomaly_maps(grid)                              # [n, 392, 392]
scores = readout.image_scores(maps)                            # [n]

model.save('state.rigel')                                      # persistent state only
model = RigelCAD.load('state.rigel')                           # either format; learning can continue with fit_task
```

**Split fit (1.1.1).** `fit_task(tokens)` is `write_task(compute_fit(tokens), name)`:
- `compute_fit` returns a `TaskFit`: coreset keys, residual moment S_t, count n_t. It does not depend on the budget or
  the storage format;
- `write_task` adds a `TaskFit` to the memories and consolidates.

One `TaskFit` can therefore be written into several models (for example a budget sweep), with the same result as
calling `fit_task` on each of them.

**Prepared scoring (1.1.1).** `patch_energy_grid(tokens)` with no state caches the current read-out on the GPU: keys
in fp32, W, and keys @ W. The cache is rebuilt after every `write_task`, `load` or change of the stored keys.
- `prepare(state)` returns the same cache for a past state, `state_at(l)`;
- `clear_cache()` frees it.
Scores are bitwise identical to passing the `RigelState`.

**`RigelConfig` options.**

| Option | Values | Effect |
|---|---|---|
| `budget_mib` | 4 / 8 / 16 / 32 / `None` | persistent-state budget; `None` keeps every coreset (unbounded reference, fp16 only) |
| `storage` | `'fp16'` (default), `'int4_triangular'` | persistent-state format (see above) |
| `slow_level` | `True` (default) / `False` | `False` is an ablation: fast keys only, Euclidean scoring, the whole budget spent on keys |
| `selection` | `'kcenter'` (default) / `'fifo'` | `'fifo'` is an ablation that evicts the oldest keys first |
| `backbone` | `'dinov2_vitb14_reg4'` / `'vit_b8_in21k'` | the second option is the matched-backbone ablation at 224 px |
| `seed` | int (2711) | seed of every greedy k-centre call |

**State and history.**
- `RigelCAD.state()` returns a `RigelState` with these fields:
  - `keys` as scored: fp16 for `fp16`, dequantised fp32 for `int4_triangular`;
  - `W`;
  - `bytes`, the registered persistent bytes;
  - for `int4_triangular`, also the packed codes `q4` and the scales `scale`.
- `RigelCAD(keep_history=True)` keeps the read-out state after every task, so past stages can be scored with
  `state_at(l)`; forgetting is computed from this. With `int4_triangular` the history keeps the packed codes (386 B per
  key) and dequantises them on demand.

## Command line

**Continual stream on MVTec AD, VisA or MPDD** (task order a|b|c, seed, budget):

```bash
python -m rigel_cad stream --dataset mvtec --data-root data/mvtec --order a --seed 2711 --budget-mib 8 \
       --cache-dir /path/to/token_cache --out results/rigel_cad
python -m rigel_cad stream --dataset mpdd --data-root data/mpdd --storage int4_triangular --budget-mib 8
```

**Ablations:**

```bash
python -m rigel_cad stream --dataset visa --data-root data/visa --budget-mib none       # unbounded reference
python -m rigel_cad stream --dataset mpdd --data-root data/mpdd --fast-only            # without the slow level
```

**MVTec AD 2.** This runs a stream over the 8 scenarios and writes:
- the thresholds from validation/;
- the local test_public AU-PRO and SegF1;
- the submission folder for test_private and test_private_mixed.

```bash
python -m rigel_cad ad2 --data-root /path/to/mvtec_ad_2 --out results/rigel_cad/ad2 [--per-stage] [--no-submission] \
       [--storage int4_triangular]
```

**Speed and memory benchmark from raw images** (arm E in either storage format):

```bash
python -m rigel_cad bench --dataset mvtec --data-root data/mvtec --out results/rigel_cad/bench [--storage int4_triangular]
```

**Image decoding.** `stream`, `ad2` and `bench` decode training images with a thread pool. `--decode-workers N` sets
the number of threads (default `min(8, CPU count)`); `1` decodes sequentially. The preprocessing is unchanged, so the
tokens are identical for any N.

**Outputs** (under `--out`). Non-default storage adds the suffix `_int4tri` to every file name, so a compact-memory run
never overwrites an fp16 result.

| File | Content |
|---|---|
| `RESULTS_<tag>.json` | final pixel AP and image AUROC, forgetting, acquisition and prequential AP, per-task and per-stage metrics, registered bytes per stage, update time, scoring latency, bootstrap CI95 (order a), `storage` |
| `BOOT_<tag>.npz` | bootstrap replicates |
| `states/<tag>.rigel` | the persistent state |
| `grids/` | energy grids (with `--save-grids`) |
| `RESULTS_ad2[_int4tri].json`, `submission[_int4tri]/` | MVTec AD 2 results and submission maps. Check and compress the folder with the official MVTec AD 2 code utils before uploading |

`<tag>` has the form `<dataset>_<order><seed>_<backbone>_B<budget>[_fastonly][_fifo][_int4tri]`.

**Data layouts.**
- MVTec AD and MPDD: the standard folder layout.
- VisA: `split_csv/1cls.csv`.
- MVTec AD 2: `<scenario>/{train,validation,test_public,test_private,test_private_mixed}`.

Training images are always read in sorted path order. The order is part of the method, because greedy k-centre starts from a
seeded random index.

## Memory accounting

The package reports several kinds of memory. They are separate quantities and should be reported separately.

| Quantity | `fp16` (8 MiB) | `int4_triangular` (8 MiB) |
|---|---|---|
| **Registered logical bytes** (`RigelState.bytes`, `persistent_bytes()`, asserted ≤ budget after every task) | 3,925 × 1,536 + 2,359,304 = **8,388,104** | 18,672 × 386 + 1,181,192 = **8,388,584** |
| **Persistent tensor payload** of the saved state | 10,747,400 (S is saved at its fp64 accumulator precision) | **8,388,584** (codes, scales, triangle, count) |
| **Serialized file** (`torch.save`, including container metadata, config and task names) | ≈ 10,750,000 | 8,391,364 (VisA) – 8,391,437 (MVTec AD), which is **≈ 2.8 kB above 8 MiB** |
| **Live model on GPU** (keys as held between tasks) | 6.0 MB (fp16 keys) | 71.7 MB (unpacked int8 codes + fp32 view) |
| **Scoring working set**: fp32 keys, keys·W and W | 26.5 MB | 117.1 MB |
| **Peak extra GPU memory** while scoring one category (measured) | 52–53 MB | 233–234 MB |

**Notes on the table.**
- File sizes are for the full key bank. Under the budget, the int4 tensor payload equals the registered logical bytes
  exactly. The serialized file does not fit under 8 MiB, because of the `torch.save` container.
- In GPU memory the live int4 memory is held unpacked: int8 codes plus a dequantised fp32 view, 3,840 B per key, about
  72 MB at 18,672 keys. Packing is applied when the memory is saved or recorded in the history.
- Memory figures were measured on MVTec AD, VisA and MPDD, order a, seed 2711 (`results/rigel_cad/q4t_integration/`).

## Evaluation protocol

- **Data:** train/good only, every image, one pass per task.
- **Model evaluated:** the final model, after the last task, task-agnostic, on the full official test set of every
  category.
- **Pixel AP:** native resolution, binned into 8,192 bins over [0, max energy of the category + 1e-12]; category mean.
- **Image AUROC:** computed from the image score.
- **Forgetting (FM):** mean over tasks, except the last, of the maximum AP over the stages from acquisition to T−2,
  minus the final AP.
- **Also reported:** acquisition AP, the AP right after a task is learned; and prequential AP.
- **Bootstrap CI95:** 2,000 paired replicates, seeded by [20261120, position in the order].
- **Task orders:**
  - a = alphabetical;
  - b = reversed;
  - c = SHA-256 permutation with the salt `NANM-EVAL-01/order-c/v1`.
  - Seeds: 2711, 2712, 2713.
- **MVTec AD 2:**
  - θ_s = mean + 3·std of the validation pixel scores at the scenario's acquisition stage;
  - AU-PRO at FPR limits 0.05 / 0.30, from an exact float16 histogram, equivalent to the official MVTec evaluation code;
  - SegF1 aggregated per scenario.

## Results

All numbers below were measured with the protocol above, using the DINOv2 ViT-B/14 reg4 encoder at 392 px and an
8 MiB budget.

### Main cell (order a, seed 2711)

| Dataset | Storage | Pixel AP | Image AUROC | FM | Acq. AP | Preq. AP | Keys |
|---|---|---|---|---|---|---|---|
| MVTec AD (15) | fp16 | 0.5927 | 0.9826 | 0.0333 | 0.6129 | 0.6395 | 3,925 |
| | int4_triangular | 0.6304 | 0.9926 | 0.0074 | 0.6269 | 0.6559 | 18,672 |
| VisA (12) | fp16 | 0.4603 | 0.9349 | 0.0119 | 0.4638 | 0.4934 | 3,925 |
| | int4_triangular | 0.4749 | 0.9707 | 0.0053 | 0.4680 | 0.4974 | 18,672 |
| MPDD (6) | fp16 | 0.5082 | 0.9399 | 0.0128 | 0.5163 | 0.3433 | 3,925 |
| | int4_triangular | 0.5170 | 0.9598 | 0.0012 | 0.5175 | 0.3475 | 6,965 (cap not reached) |

**Bootstrap CI95 for fp16.** Pixel AP:
- MVTec AD [0.575, 0.612];
- VisA [0.430, 0.485];
- MPDD [0.490, 0.530].

**Reference without the budget.** The unbounded fp16 reference reaches pixel AP 0.6348 / 0.4868 / 0.5175 on MVTec AD /
VisA / MPDD, using 46.1 / 106.6 / 13.1 MB.

### Robustness (3 task orders × 3 seeds, 9 cells per dataset; mean ± std)

| Dataset | fp16 pixel AP | int4_triangular pixel AP | Δ AP (cells > 0) | Δ image AUROC | Δ FM |
|---|---|---|---|---|---|
| MVTec AD | 0.5944 ± 0.0063 | 0.6310 ± 0.0017 | +0.0366 ± 0.0066 (9/9) | +0.010 | −0.024 |
| VisA | 0.4505 ± 0.0052 | 0.4738 ± 0.0031 | +0.0233 ± 0.0052 (9/9) | +0.022 | −0.011 |
| MPDD | 0.5102 ± 0.0078 | 0.5171 ± 0.0046 | +0.0069 ± 0.0058 (7/9) | +0.013 | −0.009 |

**Categories where `int4_triangular` is worse.** The gain is not uniform across categories. Category-level final pixel
AP is lower than fp16 in these cases:
- MVTec AD zipper: −0.073 in one cell;
- MVTec AD tile: lower in 6 of 9 cells, by up to −0.022;
- VisA macaroni1: as low as −0.031;
- VisA capsules: as low as −0.019;
- MPDD connector: lower in 7 of 9 cells, by up to −0.020;
- MPDD metal_plate: lower in 9 of 9 cells, by −0.003 to −0.010;
- MPDD bracket_white: lower in 4 of 9 cells, by up to −0.024.

**Quantisation fidelity.** int4 is lossy:
- median single-token energy change of 15–25 %;
- 19–25 % of nearest-key assignments change.
The accuracy gain comes from the 4.8× larger key bank, not from fidelity to the fp16 keys.

### MVTec AD 2 (fp16 only)

These are RIGEL-CAD 1.0 (`fp16`) results. `int4_triangular` has not been evaluated on MVTec AD 2. It runs through the
same `ad2` command, but none of these numbers apply to it.

| Split | AU-PRO 0.05 | Other |
|---|---|---|
| test_private (official server) | 38.2 % | SegF1 20.5, ClassF1 75.4 |
| test_private_mixed (official server) | 34.0 % | SegF1 15.3, ClassF1 76.6 |
| test_public (local, corrected masks) | 0.3594 | AU-PRO 0.30 0.5921, SegF1 0.1680 |

### Budget curve, both storage formats (order a, seed 2711)

Final model, task-agnostic pixel AP. The script is `exps/rigel_budget_00/sweep.py`; it writes
`results/rigel_cad/budget_sweep/` with image AUROC, forgetting and per-stage metrics.
- Descriptive only: the test sets were used in earlier development, and nothing is selected from this curve.
- Where a budget never binds, its state is identical at every stage to a smaller budget's, and the same numbers are shown.
- The fp16 rows at 4, 8 and 16 MiB and int4 at 8 MiB equal the locked results exactly.

| Budget (MiB) | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | unbounded |
|---|---|---|---|---|---|---|---|---|---|---|
| MVTec AD, fp16 | — | — | 0.5606 | 0.5840 | 0.5927 | 0.6075 | 0.6219 | 0.6281 | 0.6315 | 0.6348 |
| MVTec AD, int4_triangular | 0.5827 | 0.6090 | 0.6206 | 0.6263 | 0.6304 | 0.6331 | 0.6331 | 0.6331 | | |
| VisA, fp16 | — | — | 0.3672 | 0.4418 | 0.4603 | 0.4620 | 0.4614 | 0.4744 | 0.4808 | 0.4868 |
| VisA, int4_triangular | 0.4213 | 0.4532 | 0.4659 | 0.4711 | 0.4749 | 0.4759 | 0.4821 | 0.4869 | | |
| MPDD, fp16 | — | — | 0.4685 | 0.4889 | 0.5082 | 0.5172 | 0.5175 | 0.5175 | 0.5175 | 0.5175 |
| MPDD, int4_triangular | 0.4879 | 0.5096 | 0.5170 | 0.5170 | 0.5170 | 0.5170 | 0.5170 | 0.5170 | | |

Notes:
- fp16 needs at least 2.25 MiB for the slow statistic alone, so it has no 2 or 3 MiB point.
- The fp16 32 MiB and unbounded columns are locked G3E-PAPER-01 values; they were not recomputed.
- int4 keeps every coreset key from 12 MiB on for MVTec AD (28,458 keys, 12.2 MB) and from 4 MiB on for MPDD.
- Without consolidation, its gap to the unbounded fp16 reference is the int4 quantisation: −0.0017 on MVTec AD and
  −0.0005 on MPDD.

### Runtime (RTX 4050 Laptop, 6 GB; head only, excluding the encoder)

Measured with CUDA synchronisation, order a, seed 2711, on cached tokens.

**1.1.1** (`exps/rigel_eng_01/check.py`, `results/rigel_cad/eng_01/`):

| | fp16 | int4_triangular |
|---|---|---|
| Scoring latency per image, p50, prepared (MVTec AD / VisA / MPDD) | 1.84 / 1.79 / 1.97 ms | 8.20 / 8.23 / 3.00 ms |
| Same, passing the `RigelState` (keys moved and multiplied per call) | 3.68 / 3.63 / 3.92 ms | 17.9 / 18.0 / 7.02 ms |
| Coreset fit per stream, `compute_fit` total (MVTec AD / VisA / MPDD) | 37 / 243 / 7 s | same fit |
| k-centre alone with the 1.1.0 plain loop, same tokens | 92 / 640 / 16 s | same fit |
| Write + consolidation per stream, `write_task` total (MVTec AD / VisA / MPDD) | 2.6 / 3.0 / 0.8 s | 11.0 / 17.3 / 0.3 s |

Notes:
- The int4 scoring cost that remains is the fp32 distance computation over 18,672 keys.
- Lowering it further would need TF32 or fp16 arithmetic, which changes the scores, so it was not done.

**MVTec AD from raw images**, `bench`, 3,629 training images, arms E, F and U:
- Training wall time is 305 s in 1.1.1 against 682 s in 1.1.0.
- Decoding takes 76 s against 264 s, and the coreset fit of E 38 s against 104 s.
- Encoding (115 s) is unchanged.
- The int4 arm takes 313 s.
- The saved E states are tensor-for-tensor equal to the 1.1.0 checkpoints.
- Outputs: `results/rigel_cad/bench_v111/` against `results/rigel_cad/bench/`.

**1.1.0** (before the engineering changes):

| | fp16 | int4_triangular |
|---|---|---|
| Scoring latency per image, p50 (MVTec AD / VisA / MPDD) | 2.17 / 2.08 / 2.51 ms | 7.95 / 7.96 / 3.43 ms |
| Update time per stream, `fit_task` total: coreset fit + write + consolidation (MVTec AD / VisA / MPDD) | 113 / 669 / 21 s | 140 / 709 / 19 s |
| Update time per task, median (MVTec AD / VisA) | 7.2 / 79.1 s | 7.8 / 81.8 s |

**Notes.**
- MPDD has fewer keys (6,965), so its int4 latency is lower.
- The per-task coreset fit dominates update time and is the same for both formats.
- `int4_triangular` adds about 6 ms per image at full capacity.
- Encoding an image with the frozen encoder costs about 50 ms of compute on the same GPU, so end-to-end inference
  slows by roughly 10 %.

## Reproducibility

**Tests.**

```bash
python -m pytest -q rigel_cad/tests/test_unit.py          # CPU unit tests (no data needed)
python -m pytest -q rigel_cad/tests/test_equivalence.py   # needs CUDA + the locked research caches
```

The unit tests cover:
- capacities and byte accounting for both formats;
- k-centre determinism;
- the whitener;
- the int4 quantiser: range, idempotence, pack/unpack round trip and 386 B per key;
- exact reconstruction of the triangular statistic, with W bitwise equal to the W from the full matrix;
- int4 consolidation and the compact history;
- save → load → continue for int4, bitwise equal to an uninterrupted run;
- an unchanged fp16 checkpoint layout;
- loading a RIGEL-CAD 1.0 checkpoint;
- 1.1.1:
  - the CUDA-graph k-centre equals the plain loop;
  - prepared scoring equals state scoring and is invalidated by every write;
  - `compute_fit` + `write_task` equals `fit_task`;
  - threaded decoding equals sequential decoding;
- metric primitives;
- the MVTec AD 2 mask lookup.

The equivalence tests check, on MPDD, order a, seed 2711:
- **fp16:** 8 MiB, unbounded, fast-only and FIFO grids are bitwise equal to the locked research grids at every stage.
- **int4_triangular:** at every task, the int4 codes, scales, triangle, count and whitener are bitwise equal to the
  research implementation, and so is every stage's grid.
- **Metrics:** metrics and bootstrap replicates are bitwise equal to the locked files.
- **Task orders:** the task orders match.
- **Encoder:** encoder tokens are bitwise equal to the research encoder on the same machine. Older token caches differ
  by at most one fp16 ulp on 0.0045 % of entries.
- **MVTec AD 2:** the thresholds and AU-PRO / SegF1 match the locked values.

**End-to-end check from raw images** (`python -m rigel_cad stream --dataset mpdd --data-root data/mpdd --storage
int4_triangular --budget-mib 8`):
- pixel AP 0.51704891, against 0.51704892 from the cached tokens;
- image AUROC 0.95981625 (identical);
- 3,869,682 registered bytes.
The encoder differs from the old token caches by at most one fp16 ulp on a tiny fraction of entries.

**Integration check of `int4_triangular` on all three datasets** (order a, seed 2711):
- The script is `exps/rigel_q4t_int/verify.py`; it writes `results/rigel_cad/q4t_integration/`.
- At every stage, the key state and grids are bitwise equal to the research implementation, the metrics are equal to
  the locked values, and the final checkpoint tensors are equal.

**1.1.1 engineering check on all three datasets** (`exps/rigel_eng_01/check.py`, order a, seed 2711):
- Every task's `compute_fit` equals the cached fit behind the locked results, and the k-centre equals the plain loop.
- Both formats at 8 MiB, written with `write_task`:
  - the int4 key count per stage equals the integration run;
  - the final state equals the saved 1.1.0 checkpoints;
  - the last category's grids at every stage are bitwise equal across the `RigelState`, the prepared cache and the
    locked grids.
- `ALL_BITWISE: true` in `results/rigel_cad/eng_01/CHECK_SUMMARY.json`.
- The budget sweep re-checks this end to end: fp16 at 4, 8 and 16 MiB and int4 at 8 MiB equal the locked metrics at
  every stage on all three datasets.

## Provenance

RIGEL-CAD is a clean re-implementation of the frozen research code under `exps/`. In that code the method was called G3-E:
fast-key arm E with pooled residual whitening. `exps/` is kept unchanged as the evidence trail.

| RIGEL-CAD | Research code |
|---|---|
| `ops.greedy_kcenter` | `exps/nanm_r0/audit.py::greedy_coreset` |
| `ops.residual_moment` | `exps/nanm_r2b/run.py::residual_stats` |
| `ops.whitener`, `ops.shrink`, `ops.inv_sqrt_psd` | `exps/nanm_r2b/run.py::whiten`, `exps/nanm_r2/run.py` |
| `ops.energy_whitened` / `energy_euclidean` | `nanm/g3.py::score`, `exps/bench_g3/run.py::euclid` |
| `FastKeyMemory` (k-centre / FIFO) | `exps/nl03_pressure_00/core.py::kc_select` / `fifo_select` |
| `FastKeyMemory` + `SlowResidualWhitener` with `storage='int4_triangular'`; `storage.py` | `exps/rigel_q8/d1.py::ArmQ4`, `quantize4`, `pack4`, `unpack4`; `exps/rigel_q8/q8.py::dequantize`, `tri_pack`, `tri_unpack` |
| `RigelCAD.fit_task` | `exps/g3e_paper_01/learner.py::Learner.task` (arms E@B, U, F, FIFO) |
| `encoders.RigelDinoEncoder` | `exps/dinomaly/encoder.py` + `exps/nar_b14_01/run.py::tokens` |
| `readout`, `metrics` | `exps/nanm_r2/run.py::maps_of`, `exps/nanm_r0/audit.py`, `exps/nl03_pressure_00/evaluate.py` |
| `ad2`, `metrics.ProAccumulator` | `exps/nl03_multirate_01/ad2_dthr.py`, `ad2_pack16.py`, `ad2_aupro.py` |

**Paper arm names.**
- E@B → `RigelConfig(budget_mib=B)`;
- U = E@∞ → `budget_mib=None`;
- F → `slow_level=False`;
- FIFO → `selection='fifo'`;
- the compact format → `storage='int4_triangular'`.

### Checkpoint formats

| Format string | Storage | Contents |
|---|---|---|
| `rigel-cad-state/1` | `fp16` | config, task names, fp16 `keys` (or unbounded `blocks`), fp64 `S`, `n` |
| `rigel-cad-state/2` | `int4_triangular` | config (with `storage`), task names, packed `q4` (uint8, M × 384), `scale` (fp16), `S_tri` (fp32, 295,296), `n` |

**Compatibility.**
- RIGEL-CAD 1.1 loads both formats, and refuses a file whose format and storage disagree.
- fp16 checkpoints written by 1.1 have the same layout as 1.0, with no `storage` key, so RIGEL-CAD 1.0 can read them.
- RIGEL-CAD 1.0 rejects `rigel-cad-state/2` files.

## Limitations

- **Unmatched comparisons.** Comparisons with published continual AD methods use a different backbone (DINOv2 at
  392 px). With a matched ViT-B/8 backbone at 224 px, RIGEL-CAD (fp16) is below CADIC's reported numbers. No
  state-of-the-art claim is made.
- **MVTec AD 2.** The official-server result (38.2 / 34.0 AU-PRO 0.05) is above the dataset authors' 256 px per-category
  baselines, and far below high-resolution, jointly trained leaderboard methods.
- **Evidence for `int4_triangular`.**
  - The format was chosen among several compact-storage variants on development data before its benchmark evaluation.
  - The MVTec AD, VisA and MPDD test sets had been used in earlier development of RIGEL-CAD.
  - Its results are therefore benchmark evidence, not an independent held-out confirmation. It has no MVTec AD 2
    evaluation.
- **Costs of `int4_triangular`.**
  - Inference costs about 3.7× in head latency and about 4.4× in scoring working memory.
  - Some categories get worse (see Results).
  - The serialized file is about 2 kB over the 8 MiB budget, although the tensor payload is within it.
- **Selection rule.** Hindsight k-centre selection over all past coresets beats the task-boundary consolidation on MVTec
  AD. That selection needs past data, so it is not streaming-feasible.
- **fp16 checkpoint size.** The fp16 state file stores S in its fp64 accumulator precision, about 10.7 MB on disk. The
  budget counts S at fp32, as in the research protocol.
- **AU-PRO credit.** The AU-PRO integration follows `generic_util.trapezoid` of the MVTec AD evaluation code (Copyright
  2022 MVTec Software GmbH, BSD-style licence).

## Release notes

**1.1.1** (engineering only; every result is bitwise unchanged)
- `greedy_kcenter` runs as a captured CUDA graph over row blocks of 8,192, with no host synchronisation per step. The
  plain loop remains as `greedy_kcenter_reference` (CPU path and test oracle).
- `RigelCAD.compute_fit` / `write_task` split `fit_task` into its budget-independent and state-dependent parts. New
  dataclasses: `TaskFit` and `Prepared`.
- Prepared scoring: `prepare`, `clear_cache`, and a cache for the current state.
- Threaded training-image decoding in the encoder (`--decode-workers`).
- `bench --storage` measures arm E in either format; non-default storage writes `_int4tri` files.
- `paper_tables` adds the 8 MiB storage-format table and the budget-sweep table (`--q4t`, `--sweep`).

**1.1.0**
- Adds the optional `storage='int4_triangular'` compact-memory format: CLI `--storage`, checkpoint format
  `rigel-cad-state/2`, and the `_int4tri` file suffix.
- The default fp16 behaviour, its outputs and its checkpoints are unchanged.
- `run_stream` logs the key count with `len(model.fast)`. The previous log line failed whenever the history holds no
  key copies: the unbounded reference and compact history. This changes logging only.

**1.0.0, erratum (2026-11-26): MVTec AD 2 test_public masks.**
- In MVTec AD 2 every good test_public image has the same file name as a bad one.
- The original mask lookup matched on the file stem, so good images were scored against a defect mask.
- The lookup now takes the label from the directory, and `run_ad2` asserts it.
- The corrected local numbers are above: AU-PRO 0.05 / 0.30 = 0.3594 / 0.5921, SegF1 0.1680; they were 0.2633 /
  0.4750 / 0.1493 before.
- Thresholds, submission maps and the official server numbers do not depend on test masks and are unchanged.
- Details: `docs/handoff/RIGEL_CAD_RELEASE_REPORT.md`.
