# Meta-NATH CAD: Nested Learning for Continual Anomaly Detection

<div align="center">

**A continual industrial anomaly detection pipeline built around frozen DINOv2 features, TITANS-style fast memory, ACC gating, CADIC coreset memory, and metric-gated Phase 3 consolidation.**

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.6-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Backbone-facebook%2Fdinov2--base-yellow.svg)](https://huggingface.co/facebook/dinov2-base)
[![Pixi](https://img.shields.io/badge/Environment-Pixi-purple.svg)](https://pixi.sh/)
[![Dataset](https://img.shields.io/badge/Dataset-MVTec%20AD%20%2B%20VisA-green.svg)](https://www.mvtec.com/company/research/datasets/mvtec-ad)

[Overview](#-overview) . [Architecture](#-architecture) . [Results](#-verified-results) . [Installation](#-installation) . [Quick Start](#-quick-start) . [Project Structure](#-project-structure)

</div>

---

## 📖 Overview

**Meta-NATH CAD** is a research-oriented implementation of **Continual Anomaly Detection (CAD)** for industrial visual inspection. The project studies how an anomaly detector can process a sequence of object categories as continual tasks while preserving previous knowledge without generative replay.

The active reportable path uses:

- **Frozen DINOv2 feature extraction** with `facebook/dinov2-base`.
- **TITANS-style fast memory** for lightweight test-time adaptation.
- **ACC gating** to decide whether updated features are stable enough to enter memory.
- **CADIC unified coreset** for normal-only incremental memory.
- **Patch nearest-neighbor scoring** for image-level and pixel-level anomaly detection.
- **Phase 3 N2B-NC consolidation** with rollback and before/after acceptance gates.

> DINOv2 is the intentional stable backbone for the current code path. DINOv3 remains a future migration target and should not be reported as the current verified backbone.

---

## 📊 Dataset Overview

### MVTec AD

The main verified experiments use the full [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) benchmark as a 15-task continual stream.

| Field | Value |
| :--- | :--- |
| Dataset | MVTec AD |
| Continual tasks | 15 object/texture categories |
| Task order | `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper` |
| Input size | `224 x 224` |
| Patch grid | `16 x 16` |
| Patch embeddings per image | `256` |
| Backbone | `facebook/dinov2-base` |
| Nearest neighbors | `b = 9` |
| Pixel score normalization | `none` |
| Gaussian smoothing | `0.0` |

### VisA

The repository also includes VisA configs and workflow wrappers:

- `conf/visa_phase3.yaml`
- `conf/visa_experimental_nsp2_cbp.yaml`
- `conf/visa_max_power.yaml`
- `scripts/run_server_visa.sh`
- `notebooks/kaggle_visa_workflow.ipynb`

VisA is prepared as an optional validation path. Do not claim accepted VisA benchmark results until the corresponding run is executed and recorded.

---

## ✨ Project Objectives

This project aims to:

1. Build a reproducible **continual anomaly detection** pipeline for industrial images.
2. Avoid generative replay by using a bounded **normal-only coreset memory**.
3. Support image-level detection and pixel-level localization from patch features.
4. Add a controlled Phase 3 consolidation stage that updates only a small part of the backbone.
5. Use metric-gated acceptance to prevent image-level gains from silently damaging pixel-level localization.
6. Provide Kaggle/server workflows that can reproduce full MVTec experiments end to end.

---

## 🧠 Architecture

The active Meta-NATH CAD pipeline is:

```text
Input image
    |
    v
Frozen DINOv2 backbone
    |
    +--> CLS embedding --------------------+
    |                                      |
    +--> Patch embeddings                  |
                                           v
                                 TITANS-style fast memory
                                           |
                                           v
                                      ACC gating
                                           |
                                           v
                        CADIC normal-only unified coreset
                                           |
                                           v
                    Patch nearest-neighbor anomaly scoring
                                           |
                                           v
                       Image score + pixel anomaly map
```

### Main Components

| Component | File | Role |
| :--- | :--- | :--- |
| Meta-NATH core | `models/meta_nath_core.py` | Frozen backbone, TITANS memory, ACC gating, CADIC orchestration |
| CADIC coreset | `models/cadic_coreset.py` | Unified memory bank and patch nearest-neighbor scoring |
| Training engine | `training/meta_nath_engine.py` | Normal-only streaming update and evaluation |
| Experiment runner | `training/run_experiment.py` | Main Phase 1-2 experiment entrypoint |
| Phase 3 engine | `training/consolidation_engine.py` | N2B-NC consolidation, drift checks, coreset refresh |
| Dataset loader | `dataset/load_dataset.py` | MVTec/VisA continual task stream |
| Acceptance gate | `scripts/pipeline/phase3_acceptance.py` | Before/after metric comparison |

### Phase 3 Modes

| Mode | Config | Status | Notes |
| :--- | :--- | :--- | :--- |
| Conservative reportable | `conf/full_demo.yaml` | Accepted on full MVTec 15-task run | `LEJEPA = 0`, high patch distillation weight, one final block |
| Experimental NSP2/CBP | `conf/experimental_nsp2_cbp.yaml` | Experimental unless accepted in a run | Enables NSP2 and CBP reset path |
| Max-power MVTec | `conf/mvtec_max_power.yaml` | Accepted in the recorded stress run | Enables NSP2, Subspace Recycling, CBP, LEJEPA, stronger Phase 3 |
| VisA conservative | `conf/visa_phase3.yaml` | Prepared | Requires VisA data |
| VisA max-power | `conf/visa_max_power.yaml` | Prepared | Requires VisA data and accepted run |

---

## ✅ Verified Results

### MVTec AD Full 15-Task Run

The latest verified run was executed on Kaggle with the full 15 MVTec AD tasks.

| Field | Value |
| :--- | :--- |
| Date | 2026-05-20 |
| Backbone | `facebook/dinov2-base` |
| Main config | `conf/full_demo.yaml` |
| Experimental config | `conf/mvtec_max_power.yaml` |
| Tasks | 15 |
| Patch grid | `16 x 16` |
| Nearest neighbors | `9` |
| Mechanism smoke | `72 / 72` tests passed |

### Main Reportable Phase 3

Conservative Phase 3 was accepted by the configured before/after gate.

| Metric | Before | After | Delta | Gate |
| :--- | ---: | ---: | ---: | :--- |
| Avg image AUROC | 0.963984 | 0.964069 | +0.000084 | n/a |
| Avg pixel AUROC | 0.954502 | 0.954588 | +0.000086 | n/a |
| Avg pixel AUPR | 0.474363 | 0.474474 | +0.000111 | n/a |
| Final cumulative image AUROC | 0.947035 | 0.947269 | +0.000233 | Pass |
| Final cumulative pixel AUPR | 0.351101 | 0.351049 | -0.000052 | Pass, threshold `>= -0.005` |
| Forgetting measure | 0.000000 | 0.000000 | 0.000000 | n/a |

**Decision:** accepted.

### Experimental Max-Power Phase 3

The max-power stress profile also passed the acceptance gate.

| Metric | Before | After | Delta | Gate |
| :--- | ---: | ---: | ---: | :--- |
| Avg image AUROC | 0.963984 | 0.964111 | +0.000127 | n/a |
| Avg pixel AUROC | 0.954502 | 0.954800 | +0.000298 | n/a |
| Avg pixel AUPR | 0.474363 | 0.474530 | +0.000167 | n/a |
| Final cumulative image AUROC | 0.947034 | 0.947791 | +0.000757 | Pass |
| Final cumulative pixel AUPR | 0.351101 | 0.350401 | -0.000701 | Pass, threshold `>= -0.005` |
| Forgetting measure | 0.000000 | 0.000000 | 0.000000 | n/a |

**Decision:** accepted.

Max-power Phase 3 details:

| Field | Value |
| :--- | :--- |
| `top_k_anchors` | 128 |
| Steps | 2 |
| Drift | 0.000315 |
| Patch drift MSE | 0.000672 |
| Drift threshold | 0.05 |
| Trainable parameter groups | 38 |
| NSP2 | Enabled, rank 14, null dim 754 |
| Subspace Recycling | Enabled, not triggered |
| CBP | Enabled, 0 dead units, 0 reset units |

> The max-power run is useful as a stress test of the full mechanism stack. For strict academic reporting, separate ablations are still needed before assigning individual performance gains to NSP2, CBP, or Subspace Recycling.

---

## 📁 Project Structure

```text
NestedLearningForCAD/
├── conf/                              # Reproducible YAML run profiles
│   ├── full_demo.yaml                 # Main reportable MVTec/Kaggle config
│   ├── experimental_nsp2_cbp.yaml     # Experimental NSP2 + CBP config
│   ├── mvtec_max_power.yaml           # MVTec max-power stress profile
│   ├── visa_phase3.yaml               # Optional VisA conservative profile
│   ├── visa_experimental_nsp2_cbp.yaml
│   └── visa_max_power.yaml
├── dataset/                           # MVTec/VisA loading and anomaly generation
├── docs/                              # Research target, pipeline notes, run log
│   ├── instruction_CAD.md
│   ├── PIPELINE.md
│   └── runs.md
├── legacy/                            # Old prototypes kept for reference only
├── models/                            # Meta-NATH mechanisms and core modules
├── notebooks/                         # Kaggle orchestration notebooks
│   ├── kaggle_full_phase3_workflow.ipynb
│   └── kaggle_visa_workflow.ipynb
├── results/                           # Generated experiment outputs
├── scripts/                           # CLI entrypoints, workflows, diagnostics
│   ├── diagnostics/
│   ├── pipeline/
│   ├── workflows/
│   ├── run_full_demo.sh
│   ├── run_server_phase3.sh
│   └── run_server_visa.sh
├── training/                          # Training, evaluation, consolidation engines
├── utils/                             # Shared helpers
│   └── global_seed.py
├── pixi.toml                          # Pixi environment
├── requirements.txt                   # Pip dependency fallback
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- Git
- Python environment with CUDA-capable PyTorch for full GPU runs
- [Pixi](https://pixi.sh/) for the recommended local environment
- MVTec AD dataset mounted under `data/mvtec` or the path configured in YAML/Kaggle

### Clone

```bash
git clone https://github.com/taitrn/NestedLearningForCAD.git
cd NestedLearningForCAD
```

### Recommended Setup with Pixi

```bash
pixi install
```

Windows check:

```powershell
.\.pixi\envs\default\python.exe --version
```

### Pip Fallback

```bash
pip install -r requirements.txt
```

For Kaggle, the notebook uses the system Python and installs only what is missing when `INSTALL_DEPS=True`.

---

## 🎯 Quick Start

### 1. Run the Mechanism Smoke Test

Windows:

```powershell
.\.pixi\envs\default\python.exe scripts\diagnostics\mechanism_smoke.py
```

Linux/Kaggle:

```bash
python -u scripts/diagnostics/mechanism_smoke.py
```

Expected result:

```text
ALL 72 TESTS PASSED
```

Optional data/config verification:

```powershell
.\.pixi\envs\default\python.exe scripts\data\prepare_data.py --run_verify --config .\conf\reference\phase1_baseline.yaml
```

### 2. Run the Full MVTec Demo

Default full demo:

```bash
bash scripts/run_full_demo.sh
```

Full 15-task MVTec run:

```bash
MAIN_MAX_TASKS=15 EXPERIMENTAL_MAX_TASKS=15 bash scripts/run_full_demo.sh
```

Full 15-task MVTec max-power stress run:

```bash
MAIN_MAX_TASKS=15 EXPERIMENTAL_MAX_TASKS=15 EXPERIMENTAL_CONFIG=conf/mvtec_max_power.yaml bash scripts/run_full_demo.sh
```

Useful knobs:

```bash
REQUIRE_EXPERIMENTAL_ACCEPTED=1 bash scripts/run_full_demo.sh
PYTHON_BIN=/path/to/python bash scripts/run_full_demo.sh
STEP_TIMEOUT_SECONDS=21600 bash scripts/run_full_demo.sh
```

### 3. Run on Kaggle

Use:

```text
notebooks/kaggle_full_phase3_workflow.ipynb
```

Recommended full-demo first cell settings:

```python
MAIN_MAX_TASKS = int(os.environ.get("MAIN_MAX_TASKS", os.environ.get("MAX_TASKS", "15")))
EXPERIMENTAL_MAX_TASKS = int(os.environ.get("EXPERIMENTAL_MAX_TASKS", str(MAIN_MAX_TASKS)))
MAX_POWER = os.environ.get("MAX_POWER", "1")
STEP_TIMEOUT_SECONDS = os.environ.get("STEP_TIMEOUT_SECONDS", "21600")
```

The notebook calls `scripts/run_full_demo.sh`, inspects result artifacts, and packages a Kaggle artifact ZIP.

### 4. Optional VisA Workflow

After mounting VisA under the expected data path:

```bash
bash scripts/run_server_visa.sh
```

VisA max-power:

```bash
EXPERIMENTAL_CONFIG=conf/visa_max_power.yaml bash scripts/run_server_visa.sh
```

---

## 🔍 Inspecting Results

Summarize a run directory:

```powershell
.\.pixi\envs\default\python.exe scripts\diagnostics\summarize_run.py results\<run_dir>
```

Important generated files:

| File | Meaning |
| :--- | :--- |
| `run_summary.json` | Main training/evaluation summary |
| `task_records.json` | Per-task records |
| `final_cumulative_metrics.json` | Final cumulative evaluation |
| `forgetting_matrix.json` | Forgetting matrix when enabled |
| `phase3_summary.json` | Phase 3 consolidation summary |
| `acceptance_report.json` | Before/after acceptance decision |
| `last_checkpoint.pt` | Default saved checkpoint |
| `manifest.json` | Full-demo manifest when all workflow stages finish |

By default, runs save `last_checkpoint.pt` only. Change `logging.checkpoint_policy` to `all` or `best_and_last` if per-task or best checkpoints are needed.

---

## 🧪 Reproducibility Notes

- Global seeding is centralized in `utils/global_seed.py`.
- Configs are intentionally separated instead of merged into one large YAML.
- `scripts/run_full_demo.sh` is the public MVTec submission entrypoint.
- `scripts/pipeline/evaluate_checkpoint.py` evaluates saved checkpoints without retraining.
- `scripts/pipeline/phase3_acceptance.py` enforces the before/after metric gate.
- `scripts/pipeline/compare_checkpoint_scores.py` is a slow optional diagnostic and is not part of the default demo.

Acceptance gate defaults:

| Metric | Role | Minimum allowed delta |
| :--- | :--- | ---: |
| Final cumulative pixel AUPR | Primary | `-0.005` |
| Final cumulative image AUROC | Guardrail | `-0.002` |

---

## ⚠️ Current Scope and Limitations

Implemented and verified:

- Frozen DINOv2 backbone path.
- TITANS-style memory.
- ACC gating.
- CADIC coreset update with normal samples only.
- Patch nearest-neighbor image/pixel anomaly scoring.
- Conservative Phase 3 consolidation.
- NSP2, CBP reset, and Subspace Recycling code paths.
- Full 15-task MVTec accepted conservative and max-power runs.

Prepared but not yet a final claim:

- DINOv3 production migration.
- Accepted VisA benchmark result.
- Fine-grained ablation proving the individual contribution of NSP2, CBP, and Subspace Recycling.
- Advanced Subspace Recycling policy beyond the current fallback projection.

Known metric caveat:

- Image-level AUROC is strong on MVTec AD, but final cumulative pixel AUPR remains much lower than image-level metrics. The project therefore treats pixel-level localization as the primary safety gate for Phase 3 acceptance.

### Legacy Code

Older ViT-CMS and supervised-style prototypes are kept under `legacy/` for reference only. They are not part of the active Meta-NATH CAD benchmark path.

| Legacy area | Files |
| :--- | :--- |
| Old model prototype | `legacy/models/vit_cms.py` |
| Old training stack | `legacy/training/trainer.py`, `legacy/training/evaluator.py`, `legacy/training/cms_optim.py` |
| Old sweep runner | `legacy/training/run_sweep.py` |
| Old scripts | `legacy/scripts/train_mvtec.py`, `legacy/scripts/test_forward.py`, `legacy/scripts/test_integration.py`, `legacy/scripts/get_mvtec_meta.py` |
| Old notebook | `legacy/notebooks/eval_checkpoint.ipynb` |

---

## 🎓 Project Information

| Information | Details |
| :--- | :--- |
| Project | Meta-NATH CAD / Nested Learning for Continual Anomaly Detection |
| Main task | Continual industrial anomaly detection |
| Main dataset | MVTec AD |
| Optional dataset | VisA |
| Main backbone | `facebook/dinov2-base` |
| Environment | Pixi, PyTorch, Hugging Face Transformers, Kaggle GPU |
| Primary workflow | `scripts/run_full_demo.sh` |
| Main notebook | `notebooks/kaggle_full_phase3_workflow.ipynb` |

### Reports and Documentation

- 📄 Research Papers: [`docs/papers`](./docs/papers)
- 🧭 Pipeline reference: [`docs/PIPELINE.md`](./docs/PIPELINE.md)
- 📈 Verified run log: [`docs/runs.md`](./docs/runs.md)
- ⚙️ Config guide: [`conf/README.md`](./conf/README.md)
- 🧰 Script guide: [`scripts/README.md`](./scripts/README.md)

---

<div align="center">

**Built for continual anomaly detection experiments with Meta-NATH CAD.**

</div>
