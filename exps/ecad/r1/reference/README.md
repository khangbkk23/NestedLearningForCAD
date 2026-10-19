# ContCore

Continual / incremental coreset learning for image anomaly detection. `main.py` trains a
PatchCore-style memory bank incrementally over a sequence of tasks (each task = one or more
object classes), and after every task it evaluates on **all classes seen so far**.

Per task the pipeline is:

1. extract patch features for the task's training images (`model.update`),
2. build a per-task coreset (`_sample_task_coresets`),
3. merge it into the global coreset with meta-greedy sampling (`_meta_greedy_sampling`),
4. save the coreset, build a FAISS index, and evaluate on the accumulated test sets.

---

## 1. Requirements

- Linux, **Python 3.8**
- An **NVIDIA GPU with CUDA** — `main.py` hardcodes `torch.device("cuda:0")`, so CPU-only runs are not supported.
- PyTorch 1.13.1 + cu116, `faiss-gpu`, `timm`, `torchvision`, `scikit-learn`, `scikit-image`, `opencv-python`, `tqdm`, `pandas`, `tifffile`.

### Option A — conda (from the recorded environment)

`requirements.txt` is a `conda list --export` dump of the environment this code was developed in:

```bash
conda create --name contcore --file requirements.txt
conda activate contcore
```

Note: that dump contains pip-installed packages (marked `=pypi_0`) and one local dev package
(`aivad=0.0.0=dev_0`), which conda cannot resolve from channels. If `conda create` fails on those
lines, use Option B instead — the versions there match the dump.

### Option B — conda + pip (recommended, most reliable)

```bash
conda create --name contcore python=3.8 -y
conda activate contcore

# PyTorch 1.13.1 / CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116

pip install faiss-gpu==1.7.2 timm==0.9.12 numpy==1.24.4 scipy==1.10.1 \
    scikit-learn==1.3.2 scikit-image==0.21.0 opencv-python==4.8.1.78 \
    pandas==1.5.3 matplotlib==3.7.4 pillow==10.1.0 tqdm==4.66.1 \
    tifffile==2023.7.10 einops==0.7.0 psutil==5.9.8 gputil==1.4.0 \
    tabulate==0.9.0 natsort==8.4.0 pyyaml==6.0.1
```

Backbone weights (`wideresnet50` by default) are downloaded automatically by torchvision/timm on
first run, so the machine needs internet access once.

---

## 2. Dataset setup

Download **MVTec AD** and/or **VisA** and point the code at them by editing `PATH_DICT` in
`datasets/common.py` (this is the dict `get_datasets` actually reads):

```python
PATH_DICT = {'mvtec': '/path/to/mvtec_ad',
             'visa':  '/path/to/visa',
             ...}
```

`utils.py` has a second `PATH_DICT` with the same keys; keep the two in sync to avoid confusion.

Expected layouts:

```
mvtec_ad/
├── bottle/
│   ├── train/good/*.png
│   ├── test/good/*.png
│   ├── test/broken_large/*.png
│   └── ground_truth/broken_large/*_mask.png
├── cable/
└── ...  (15 classes)
```

For VisA, use the official release **plus the `1cls.csv` split file at the dataset root** — the
loader reads every image and mask path from that CSV (columns 4 and 5), relative to the root, and
never scans directories itself. So the only hard requirement is that `1cls.csv` sits at
`PATH_DICT['visa']` and its relative paths resolve from there.

Class order (this fixes the incremental task order):

- **mvtec** (15): bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper
- **visa** (12): candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum

---

## 3. Running

```bash
python main.py                       # defaults: mvtec, wideresnet50, 15 tasks of 1 class
```

VisA (12 classes → 1 + 1×11 tasks):

```bash
python main.py --dataset visa --incremental_class_list 1-1-11
```

Larger coreset, 5 classes in the first task then 2 at a time:

```bash
python main.py --dataset mvtec \
    --incremental_class_list 5-2-5 \
    --coreset_size 50000 --sampling_ratio 0.01 --approximate_ratio 0.75 \
    --batch_size 12 --num_workers 8
```

Pick a GPU with `CUDA_VISIBLE_DEVICES` (the process always uses `cuda:0` of what it can see):

```bash
CUDA_VISIBLE_DEVICES=1 python main.py --dataset mvtec
```

### `--incremental_class_list`

Format `A-B-N`: the first task has **A** classes, followed by **N** more tasks of **B** classes
each — i.e. `A + B*N` classes in total, taken in the order listed above. `1-1-14` = 15 tasks of
1 class (all of MVTec AD). `A + B*N` must not exceed the number of classes in the dataset.

### Output

Results are written under:

```
./results/<dataset>/<incremental_class_list>/Coreset<coreset_size>_SampRatio<sampling_ratio>_Apprx<approximate_ratio>/
├── model/model_s<step>.pt       # coreset features + sample ids after each task
├── metrics/results_s<step>.csv  # per-class AUROC/F1/AP/AUPRO after each task
├── segmentations/
└── time.txt                     # total sampling (training) time in seconds
```

`results/` is gitignored.

---

## 4. Arguments

Run `python main.py --help` for the full list. Booleans accept `True/False/yes/no/1/0`.

### Feature extractor

| Arg | Default | Description |
| --- | --- | --- |
| `--backbone_name`, `-b` | `wideresnet50` | Backbone key from `corelib/backbones.py` (`resnet50`, `wideresnet101`, `vit_base`, `efficientnet_b5`, …). |
| `--layers_to_extract_from`, `-le` | `layer2 layer3` | Layer names to pull features from (must exist in the chosen backbone). |
| `--pretrain_embed_dimension` | `1024` | Dimension features are preprocessed to before aggregation. |
| `--feature_dim` | `1024` | Final aggregated patch-feature dimension. |
| `--patchsize` | `3` | Neighbourhood size used to form patch features. |
| `--patchstride` | `1` | Stride of the patch extractor. |

A DINOv2 variant is stubbed out in the commented block of `get_argparse()`
(`-b dinov2reg_vit_base_14 -le 11 --feature_dim 768`), but `corelib/backbones.py` has no DINOv2
entry yet, so use the timm/torchvision backbones.

### Sampler / coreset

| Arg | Default | Description |
| --- | --- | --- |
| `--metric` | `L2` | Distance for greedy sampling: `L2` or `cosine`. |
| `--coreset_size` | `20000` | Size of the global (merged) coreset kept across tasks. |
| `--sampling_threshold` | `20000` | Number of accumulated features at which sampling kicks in. |
| `--sampling_ratio` | `0.01` | Fraction of a task's features kept as its task coreset. |
| `--approximate_ratio` | `0.75` | Approximation ratio for the inter-task (meta-greedy) sampling. |
| `--chunk_size` | `100000` | Max features per distance-matrix chunk. Automatically capped to what fits in memory. |
| `--project_dim` | `128` | Random-projection dimension used during inter-task sampling. |
| `--reduce_feature_dim` | `True` | Enable that low-dimensional projection. Set `False` to sample in full feature space (slower, more memory). |

### Scoring / FAISS

| Arg | Default | Description |
| --- | --- | --- |
| `--anomaly_score_num_nn` | `1` | Number of nearest coreset neighbours used for the anomaly score. |
| `--faiss_on_gpu` | `True` | Run the FAISS index on GPU. |
| `--faiss_num_workers` | `8` | FAISS worker threads. |

### Data

| Arg | Default | Description |
| --- | --- | --- |
| `--dataset` | `mvtec` | `mvtec` or `visa`. |
| `--incremental_class_list` | `1-1-14` | Task schedule, see above. |
| `--batch_size` | `12` | Batch size for both train feature extraction and test. |
| `--num_workers` | `8` | DataLoader workers. |
| `--resize` | `256` | Image resize before crop. |
| `--image_size` | `224` | Center-crop size fed to the backbone. |
| `--augment` | `True` | Train-time augmentation (used by the VisA loader; the MVTec loader ignores it). |

### Misc

| Arg | Default | Description |
| --- | --- | --- |
| `--seed` | `42` | Seed for `torch`, `numpy`, `random`. |
| `--gpus` | `"0"` | GPU id. Since `main.py` pins `cuda:0`, prefer `CUDA_VISIBLE_DEVICES` to select a device. |

Currently parsed but unused by this entry point: `--train`, `--max_chunk_size`,
`--preprocessing`, `--patchscore`, `--patchoverlap`, `--patchsize_aggregate`.

---

## 5. Troubleshooting

- **`KeyError` on the dataset path / `FileNotFoundError`** — `PATH_DICT` in `datasets/common.py` still points at the original machine's paths; update it.
- **CUDA OOM during sampling** — lower `--chunk_size`, `--coreset_size`, or `--batch_size`; keep `--reduce_feature_dim True`.
- **CUDA OOM / crash in FAISS** — set `--faiss_on_gpu False`.
- **Empty / crashing final tasks** — `A + B*N` in `--incremental_class_list` exceeds the number of classes in the dataset, so the last tasks get empty datasets. Use `1-1-14` for MVTec AD (15) and `1-1-11` for VisA (12).
- **Slow first run** — backbone weights are being downloaded and cached under `~/.cache/torch`.
