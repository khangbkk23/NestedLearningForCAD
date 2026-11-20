# Independent replication of C / F / HOPE-S, and a memory study of F

This directory is an independent reproduction of the non-replay HOPE-CAD results in
[`docs/handoff/HOPE_S_HANDOFF.md`](../docs/handoff/HOPE_S_HANDOFF.md), run on a different machine and on two new
seeds (experiment **B325**), followed by a study of how much persistent memory detector F really needs
(experiment **B326**). It adds files only under `replication/`; nothing outside it is modified.

Setting (unchanged from the handoff): sequential, normal-only continual anomaly detection on MVTec AD, 15 tasks in
alphabetical order, no replay, no task identity at inference. Frozen DINOv2-reg ViT-B/14 encoder, frozen randomly
initialised decoder (BASE), one rank-8 LoRA expert per task on 34 decoder sites, task-free minimum over experts at
scoring, readout A1.

**All results are exploratory.** The MVTec test set was already used during development of these methods, so no
number here is a held-out estimate. VisA was not used.

## Arms

| Arm | Training objective | Slow level |
| --- | --- | --- |
| C | plain patch loss (γ = 0) | none |
| F | fixed hard-patch (surprise-weighted) patch loss, γ = 2.1, λ = 0 | none |
| H (HOPE-S) | γ starts at 0; γ and the per-site rates λ are learned by the nested slow level | learned |

All three arms run Khang's `hope_s_01.acquire` with the same fast/slow compute; C and F compute and discard the
slow-level gradients. γ = 2.1 for F was chosen by Khang **post hoc**, from the HOPE-S γ trajectory at seed 2711, and
was kept unchanged at the new seeds.

## Main results

Final Pixel AP, readout A1, macro over 15 categories. Seed 2711: Khang's run (RTX 4050) and ours (same BASE file).
Seeds 2712 and 2713: ours only, each with its own BASE generated from the seed.

| Seed | Source | C | F (γ = 2.1) | H (HOPE-S) |
| --- | --- | ---: | ---: | ---: |
| 2711 | Khang | 0.5898 | 0.6013 | 0.6006 |
| 2711 | this replication | 0.5860 (job C) | 0.6024 (job Q) | 0.5988 (job Q) |
| 2712 | this replication | 0.5912 | 0.6076 | 0.6063 |
| 2713 | this replication | 0.5849 | 0.6055 | 0.5982 |

Differences at the new seeds (95% bootstrap interval over the 15 categories, B = 2000, within one seed):

| Seed | F − C | H − F |
| --- | --- | --- |
| 2712 | +0.0164 [+0.0009, +0.0341] | −0.0014 [−0.0071, +0.0059] |
| 2713 | +0.0207 [+0.0074, +0.0333] | −0.0074 [−0.0139, −0.0008] |

What the data support:

1. **The port is faithful at seed 2711.** With Khang's BASE file, our F and H are within 0.002 Pixel AP and 0.0012
   Image AUROC of his numbers, and the learned γ follows his trajectory (within 10% at tasks 1, 4, 8, 15). Label
   `PORT_FAITHFUL` (defined in the [B325 card](experiments/B325/card.md), section 5). Our C, trained through
   `acquire(γ = 0)` instead of Khang's `fit_task`, is 0.0038 below his C: label `C_PATH_FAITHFUL` (tolerance 0.005).
2. **The fixed hard-patch loss beats the plain loss at both new seeds.** F − C is +0.0164 and +0.0207, and the lower
   end of the category-bootstrap interval is above 0 at both seeds. The gain also shows in Image AUROC, acquisition
   AP and lower forgetting. Label `HARDPATCH_SUPPORTED`.
3. **The learned slow level does not beat the fixed γ.** H − F is negative at all three seeds we ran (−0.0036,
   −0.0014, −0.0074); at seed 2713 the interval excludes 0. Label `NL_NULL_REPLICATED` (mean −0.0044, inside the
   pre-set ±0.005 band with 0.0006 to spare). The accurate wording is "no gain, a small consistent shortfall", not
   "equivalent". Note that the comparison is against a γ chosen post hoc from HOPE-S itself.
4. **F is stable across seeds:** 0.6024 / 0.6076 / 0.6055, mean 0.6052, standard deviation 0.0026.

Limits that go with every statement above: two new seeds; intervals are over categories within a seed, not over
seeds; one task order; exposed MVTec test; gains are not uniform across categories (several texture categories lose
more than 0.01 in F − C). F uses no slow-level learning and must not be described as a nested-learning improvement.

## Memory of detector F

Byte counts from `cost.persistent_bytes` of Khang's runs, plus our measured int8 payload. MB = 10^6 bytes. Full table
and literature comparison: [memory_table.md](experiments/B326/memory_table.md).

| Part | MB | Counted? |
| --- | ---: | --- |
| (1) CL state: 15 LoRA experts (bf16) + λ | 25.4 | yes |
| (2) BASE, frozen random decoder (bf16) | 122.9 | yes, method-specific |
| (3) Public DINOv2 encoder, blocks 0–9 (bf16) | 144.8 | reported, not compared |

- **Headline: 148.3 MB** of method-specific state, (1) + (2), in bf16.
- **Secondary: 135.6 MB** with the archive stored as int8 (job M2, label `INT8_LOSSLESS`: |Δ Pixel AP| ≤ 0.0001 and
  |Δ Image AUROC| ≤ 0.0003 at seeds 2711–2713). This was measured by scoring an **in-RAM dequantised** copy of the
  stored int8 tensors. Reloading `archive_int8.pt` from file and resuming training from an int8 archive are
  **untested**.
- **BASE stays counted.** BASE generated on CUDA is not reproducible across machines (T4 vs RTX 4050: max abs
  difference 0.0596). BASE generated from a seeded CPU generator is reproducible on one machine and gives the same
  detector quality within noise (Δ Pixel AP −0.0027 and −0.0017, label `BASE_FROM_SEED_OK`), but it is **not
  bit-portable**: regenerating seed 2712 on Windows/Intel instead of Linux gave 1–2 ULP differences on 0.66% of the
  elements (402,455 of 61,425,408). The pre-set rule required identical SHA-256, so the label is
  `BASE_SAME_MACHINE_ONLY` and BASE cannot be replaced by "seed + code hash".
- Compared with the literature table, only column (1) is comparable (other methods do not count their frozen
  backbone). Column (2) has no counterpart and is reported next to it, not dropped.

## Environments

| | Khang (reference runs) | This replication |
| --- | --- | --- |
| GPU | RTX 4050 Laptop | 2 × Tesla T4 (Kaggle) |
| PyTorch | 2.10, CUDA 12.6 | 2.11.0+cu130 (CUDA 13.0), Python 3.13.16, scipy 1.16.3, scikit-learn 1.6.1 |
| Code | `exps/` at 5217cb1 | the same files (see below), driven by `replication/scripts/` |

GPU, CUDA build and PyTorch version differ at the same time, so no difference between the two sides can be
attributed to one of them. The CPU-BASE portability check (B326 M1a) used torch 2.11.0+cpu on Windows 11 / Intel
against torch 2.11.0+cu130 on Linux.

## Code and provenance

- `scripts/hope_replicate.py` (B325) re-implements only the orchestration (train loop, sealing, test extraction,
  scoring, metrics); every numerical function is Khang's, imported from `exps/` unmodified.
- `scripts/b326_memory.py` (B326) adds int8 quantisation, the CPU-seeded BASE, and their labels, and reuses the B325
  phases. `scripts/b326_memory_table.py` builds the memory table (stdlib only).
- **Pin.** The published runs executed Khang's code at commit `5217cb1`. [`khang_source.json`](khang_source.json)
  records the SHA-256 of all 108 repository files the drivers can import (static import closure, including Khang's
  four gate test files). The drivers refuse to run if any pinned file differs, and
  `python -I -B replication/scripts/check_khang_pin.py` reports any changed or newly imported file. At the base of
  this branch (`dikhang_space` at 75384d5) all 108 files are byte-identical to 5217cb1; the branch differs from
  5217cb1 only in `dataset/download_mvtec*` and `requirements.txt`, which are not imported.
- **What changed relative to the executed code.** The runs used the same drivers in a different repository layout
  (Khang's files copied under a vendor directory and checked against a per-file hash list). Here the drivers import
  Khang's files from the repository root, the hash check reads `khang_source.json`, record fields named
  `vendored_commit` are now written as `khang_code_commit`, the configs no longer hold execution-platform settings,
  and a `--gpus` option was added. The numerical path is unchanged. The adapted drivers are covered by the CPU unit
  tests in `replication/tests` but **have not been re-run end to end on a GPU**; `code_sha256` values inside the old
  records refer to the executed driver bytes, not to these files. The published `metrics.json` files are the original
  ones, except that one free-text note in job Q (`base_identity`) was reworded.

## Layout

```text
replication/
  README.md                this file
  khang_source.json        pin of Khang's code (commit 5217cb1)
  configs/b325.json        B325 jobs Q, C, R: seeds, arms, lanes, time limits, reference numbers, input hashes
  configs/b326.json        B326 jobs M1, M2: pinned checkpoints, decision rules
  scripts/                 drivers (see above) and check_khang_pin.py
  tests/                   CPU unit tests of the drivers (torch-dependent tests skip without torch)
  experiments/B325/        card (design and pre-set decision rules) and runs/<run-id>/
  experiments/B326/        card, memory_table.md and runs/<run-id>/
```

Each GPU run directory `runs/<run-id>/` holds `results.md` (reading of the run), `metrics.json` (full numbers, per
category) and `execution.json` (environment, gates, hashes of BASE files and checkpoints, timings, labels).
`execution.json` is a projection of the original run record with execution-platform bookkeeping removed (its
`derivation` field lists what was removed or rewritten); it carries the SHA-256 of the original record. The CPU-only
check `runs/b326-m1a-local-20261119/` instead holds its results, the scripts as run, the environment spec and the
measured `base_diff_2712.json` and `BASE_SEED.json` files. Run ids encode the job and the date (`b325-r-20261119-01`: B325, job R, 19 Nov 2026, run 1).
Failed infrastructure attempts that stopped before any scientific step are not reproduced here; the runs listed are
all runs that produced results.

## Reproducing

**Requirements.** Linux with an NVIDIA GPU (Khang's `exps/nar_pilot_00/full.py` imports the Unix-only `resource`
module). Python with PyTorch, numpy, scipy, scikit-learn, Pillow, opencv-python (`cv2`), timm and pytest: the third-party imports of the pinned `exps/` closure.
About 20 GB of free disk for the feature caches (train cache fp32 12.9 GB, test cache 2.1 GB in bf16 or 4.2 GB in
fp32). One job took 1.2–3.4 hours on 2 × T4.

**Inputs.**

| Argument | Content |
| --- | --- |
| `--data` | MVTec AD root (`<category>/train/good`, `<category>/test`, ground truth) |
| `--weights` | DINOv2-reg ViT-B/14 checkpoint, SHA-256 `73182a08…5dba71` (`ENC_SHA` in `exps/dinomaly/common.py`) |
| `--khang` | directory with Khang's sealed `BASE.pt` for seed 2711 (SHA-256 `dfe6689e…10cc31`, from his `nar_full_15` run) and his 15 manifests `<category>_train.json` from `nar_b14_01` (hashes in `configs/b325.json`) |
| `--work`, `--cache` | new output directory and feature-cache directory |

The gate phase refuses to run unless the encoder, `BASE.pt`, the 15 manifests and every one of the 3,629 MVTec
train images match these hashes.

**Commands** (from the repository root):

```sh
python -m pytest -q replication/tests                  # CPU unit tests; needs only pytest (torch optional)
python -I -B replication/scripts/check_khang_pin.py    # Khang's code equals the pinned 5217cb1 bytes

# B325. Job Q: port check at seed 2711 with Khang's BASE (F on GPU 0, H on GPU 1).
python -B replication/scripts/hope_replicate.py job --job Q --data <mvtec> --weights <encoder.pth> \
    --khang <khang-dir> --work <out>/b325-q --cache <cache>
# Job C: arm C at seed 2711. Job R: seeds 2712 and 2713, arms C, F, H (one seed per GPU).
python -B replication/scripts/hope_replicate.py job --job R --data <mvtec> --weights <encoder.pth> \
    --khang <khang-dir> --work <out>/b325-r --cache <cache>

# B326. --inputs must contain the B325 work directories named by run id, and khang-reference/BASE.pt.
python -B replication/scripts/b326_memory.py job --job M2 --data <mvtec> --weights <encoder.pth> \
    --inputs <inputs> --khang <khang-dir> --work <out>/b326-m2 --cache <cache>
```

On a single GPU add `--gpus 0`: the lanes then run one after another on that GPU. Each lane is its own process with
the same seeds, so the schedule does not change the computation; the wall time roughly doubles. Every phase is also
a subcommand (`gate`, `extract-train`, `make-base`, `train`, `fidelity`, `extract-test`, `score`, `metrics`,
`label`; see `--help`). Runs never overwrite existing outputs.

**What to expect.** Bit-identical numbers are only expected on the same GPU type, CUDA build and PyTorch version:
on 2 × T4 with torch 2.11.0+cu130, re-running F from the same BASE reproduced the checkpoint SHA-256 and the macro
Pixel AP / Image AUROC exactly (B326 job M1). On other hardware expect macro differences of the size seen between
Khang's machine and ours (≤ 0.004 Pixel AP) and larger per-category differences (up to 0.038). Jobs R and M1 generate
BASE on the GPU, so on another machine they produce a different BASE draw for the same seed.

The CPU-BASE portability check (`experiments/B326/runs/b326-m1a-local-20261119/`) keeps the scripts exactly as they
were run; they refer to the earlier vendor-directory layout and must be given paths accordingly.

## License and attribution

Khang's code (`exps/`, `models/`, `dataset/`) is MIT, Copyright (c) 2026 Khang Bui Tran Duy, see
[LICENSE](../LICENSE). The files in this directory reuse it without modification and follow the same license.
