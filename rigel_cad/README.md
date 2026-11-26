# RIGEL-CAD

RIGEL-CAD is a continual anomaly detector for a stream of industrial categories. It learns from normal images only and keeps a fixed persistent-memory budget (8 MiB by default).

It has two learning levels with different update frequencies. This follows the Nested Learning mechanism: a fast memory, plus a slower consolidating level that shapes how that memory is read.

| Level | Component | Update | State |
|---|---|---|---|
| Fast | `FastKeyMemory` | once per task: 1 % greedy k-centre coreset of the task's tokens, then a k-centre consolidation of {keys} ∪ {new coreset} down to the key capacity | fp16 keys, 1,536 B each |
| Slow | `SlowResidualWhitener` | accumulated over the whole stream: uncentred second moment S of the residuals "token − nearest key of its task" | S (fp32) + count = 2,359,304 B |

**Scoring.**
- Patch energy: `e(q) = ½ · min_m ‖qW − mW‖²`, with `W = shrink(S/n, 0.5)^(-1/2)`.
- Anomaly map: 28×28 → 392×392 bilinear, followed by a Gaussian with σ = 4.
- Image score: mean of the top 1 % of the map.

**What is not stored or trained.** The frozen encoder is shared and not counted as continual-learning state. Nothing is trained by gradient descent, and no task ID is used at test time.

## Install

You need Python ≥ 3.10, `torch`, `numpy`, `pillow`, `scikit-learn`, `scipy` and `opencv-python`.

Optional packages:
- `timm`, for the ViT-B/8 backbone;
- `tifffile`, for the MVTec AD 2 submission files.

Run every command from the repository root; the package is `rigel_cad/`.

**Checkpoints.** They are verified by sha256 and loaded strictly.

| Backbone | Path | sha256 |
|---|---|---|
| default | `checkpoints/dinov2/dinov2_vitb14_reg4_pretrain.pth` (official DINOv2 ViT-B/14 reg4) | `73182a08…ba71` |
| optional | `checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth` (timm ViT-B/8 IN-21k) | `755e9f5f…a415` |

## Python API

```python
import torch
from rigel_cad import RigelCAD, RigelConfig
from rigel_cad.encoders import build_encoder
from rigel_cad import readout

enc = build_encoder('dinov2_vitb14_reg4', device='cuda')
model = RigelCAD(RigelConfig(budget_mib=8))                 # budget_mib=None -> unbounded

for task_images in stream:                                  # lists of normal training image paths, one per task
    tokens = enc.encode_paths(sorted(task_images))          # [n, 784, 768] fp16 unit tokens
    model.fit_task(tokens.reshape(-1, 768).cuda())

grid = model.patch_energy_grid(enc.encode_paths(test_paths))   # [n, 28, 28]
maps = readout.anomaly_maps(grid)                              # [n, 392, 392]
scores = readout.image_scores(maps)                            # [n]
model.save('state.rigel')                                      # persistent state only (≤ budget)
model = RigelCAD.load('state.rigel')
```

**Options** (`RigelConfig`):

| Option | Effect |
|---|---|
| `budget_mib` | 4 / 8 / 16 / 32 / `None` |
| `slow_level=False` | ablation: fast keys only, Euclidean scoring, the whole budget spent on keys |
| `selection='fifo'` | ablation: evict the oldest keys first |
| `backbone='vit_b8_in21k'` | matched-backbone ablation |
| `seed` | seed of every greedy k-centre call |

`RigelCAD(keep_history=True)` keeps the read-out state after every task, so past stages can be scored with `state_at(l)`. This is what forgetting is computed from.

## Command line

```bash
# continual stream: MVTec AD / VisA / MPDD, task order a|b|c, seed, budget
python -m rigel_cad stream --dataset mvtec --data-root data/mvtec --order a --seed 2711 --budget-mib 8 \
       --cache-dir /path/to/token_cache --out results/rigel_cad
python -m rigel_cad stream --dataset visa --data-root data/visa --budget-mib none          # unbounded reference
python -m rigel_cad stream --dataset mpdd --data-root data/mpdd --fast-only               # ablation without the slow level

# MVTec AD 2: stream over the 8 scenarios, thresholds from validation/, local test_public AU-PRO / SegF1,
# submission folder for test_private + test_private_mixed
python -m rigel_cad ad2 --data-root /path/to/mvtec_ad_2 --out results/rigel_cad/ad2 [--per-stage]

# paper tables under RIGEL-CAD names (from the locked research results; nothing is recomputed)
python -m rigel_cad.paper_tables
```

**Outputs** (under `--out`):

| File | Content |
|---|---|
| `RESULTS_<tag>.json` | final pixel AP and image AUROC, forgetting, acquisition and prequential AP, per-task-per-stage metrics, bytes, update time, latency, bootstrap CI95 (order a) |
| `BOOT_<tag>.npz` | bootstrap replicates |
| `states/<tag>.rigel` | the persistent state |
| `grids/` | energy grids, with `--save-grids` |

The MVTec AD 2 run writes `RESULTS_ad2.json` and `submission/anomaly_images{,_thresholded}/…`. Check and compress that folder with the official MVTec AD 2 code utils before uploading.

**Data layouts.**
- MVTec AD and MPDD: the standard folder layout.
- VisA: `split_csv/1cls.csv`.
- MVTec AD 2: `<scenario>/{train,validation,test_public,test_private,test_private_mixed}`.

Training images are always read in sorted path order. The order is part of the method, because greedy k-centre starts from a seeded random index.

## Evaluation protocol (fixed)

- **Pixel AP:** native resolution, binned into 8,192 bins over [0, max energy of the category + 1e-12].
- **Image AUROC:** computed from the image score.
- **Final-model, task-agnostic evaluation** after the last task.
- **Forgetting:** max over the stages after acquisition, minus the final value. Prequential AP is also reported.
- **Bootstrap CIs:** 2,000 paired replicates, seeded by [20261120, position in the order].
- **Task orders:**
  - a = alphabetical;
  - b = reversed;
  - c = SHA-256 permutation with the salt `NANM-EVAL-01/order-c/v1`.
- **MVTec AD 2:**
  - θ_s = mean + 3·std of the validation pixel scores at the scenario's acquisition stage;
  - AU-PRO (FPR limits 0.05 / 0.30) from an exact float16 histogram, equivalent to the official MVTec evaluation code;
  - SegF1 aggregated per scenario.

## Equivalence with the research code (provenance)

RIGEL-CAD is a clean re-implementation of the frozen research code under `exps/`, where the method was called G3-E (fast-key arm E with pooled residual whitening). `exps/` is kept unchanged as the evidence trail.

| RIGEL-CAD | Research code |
|---|---|
| `ops.greedy_kcenter` | `exps/nanm_r0/audit.py::greedy_coreset` |
| `ops.residual_moment` | `exps/nanm_r2b/run.py::residual_stats` |
| `ops.whitener`, `ops.shrink`, `ops.inv_sqrt_psd` | `exps/nanm_r2b/run.py::whiten`, `exps/nanm_r2/run.py` |
| `ops.energy_whitened` / `energy_euclidean` | `nanm/g3.py::score`, `exps/bench_g3/run.py::euclid` |
| `FastKeyMemory` (k-centre / FIFO) | `exps/nl03_pressure_00/core.py::kc_select` / `fifo_select` |
| `RigelCAD.fit_task` | `exps/g3e_paper_01/learner.py::Learner.task` (arms E@B, U, F, FIFO) |
| `encoders.RigelDinoEncoder` | `exps/dinomaly/encoder.py` + `exps/nar_b14_01/run.py::tokens` |
| `readout`, `metrics` | `exps/nanm_r2/run.py::maps_of`, `exps/nanm_r0/audit.py`, `exps/nl03_pressure_00/evaluate.py` |
| `ad2`, `metrics.ProAccumulator` | `exps/nl03_multirate_01/ad2_dthr.py`, `ad2_pack16.py`, `ad2_aupro.py` |

Paper arm names map to RIGEL-CAD as follows:
- E@B → `RigelCAD(RigelConfig(budget_mib=B))`;
- U = E@∞ → `budget_mib=None`;
- F → `slow_level=False`;
- FIFO → `selection='fifo'`.

### Tests

```bash
python -m pytest -q rigel_cad/tests/test_unit.py          # CPU unit tests (no data needed)
python -m pytest -q rigel_cad/tests/test_equivalence.py   # needs CUDA + the locked research caches
```

**`test_equivalence.py`** (MPDD, order a, seed 2711):
1. Learner grids for 8 MiB, unbounded, fast-only and FIFO are **bitwise equal** to the locked grids at every stage.
2. Metrics and bootstrap replicates are bitwise equal to the locked METRICS file.
3. Task orders equal the registered orders on MVTec, VisA and MPDD.
4. Encoder tokens are bitwise equal to the research encoder run on the same machine.
5. MVTec AD 2:
   - the D-THR thresholds of all 8 scenarios are bitwise equal to the locked values;
   - test_public AU-PRO 0.05 / 0.30 and SegF1 (scenario `can`, with its tp/fp/fn counts) are bitwise equal to the locked values.
6. ViT-B/8 tokens are bitwise equal to the matched-backbone cache.

The token caches written earlier, in another software/hardware environment, differ by at most one fp16 ulp on 0.0045 % of entries.

**End-to-end check from images** (`python -m rigel_cad stream --dataset mpdd …`, 2026-11-26):

| Metric | End-to-end run | Locked value |
|---|---|---|
| Pixel AP | 0.50822631 | 0.50822630 |
| Image AUROC | 0.93990785 | 0.93990785 (identical) |
| Bytes | 8,388,104 | 8,388,104 |

**End-to-end check on MVTec AD 2** (`python -m rigel_cad ad2 … --budget-mib 8 --seed 2711 --per-stage`, 2026-11-26, 70 min on an RTX 4050):

| Quantity | Result vs locked values |
|---|---|
| θ, 8 scenarios | bitwise equal |
| test_public AU-PRO 0.05 / 0.30, SegF1, every stage (80 values) | bitwise equal |
| mean AU-PRO 0.05 / 0.30 / SegF1 | 0.2633 / 0.4750 / 0.1493 (bitwise equal) |
| forgetting of AU-PRO 0.05 | 0.0077 (bitwise equal) |
| submission files (300 sampled TIFF + PNG) | byte-identical to the submitted folder |

Result file: `results/rigel_cad/ad2/RESULTS_ad2.json`.

The saved `.rigel` file stores S in its fp64 accumulator precision, so it is about 10.7 MB on disk. The budget counts S at fp32 (2,359,304 B with the count), as in the research protocol.

## Scope and limitations

- **Unmatched comparisons.** Comparisons with published continual AD methods use a different backbone (DINOv2 at 392 px). With a matched ViT-B/8 at 224 px backbone, RIGEL-CAD is below CADIC's reported numbers.
- **MVTec AD 2.** The result on the official server (38.2 / 34.0 AU-PRO₀.₀₅, private / mixed) is above the dataset authors' 256 px per-category baselines, and far below high-resolution, jointly trained leaderboard methods.
- **Selection rule.** Hindsight k-centre selection over all past coresets beats the task-boundary consolidation on MVTec AD. That selection needs past data, so it is not streaming-feasible.
- **AU-PRO credit.** The AU-PRO integration follows `generic_util.trapezoid` of the MVTec AD evaluation code (Copyright 2022 MVTec Software GmbH, BSD-style licence).
