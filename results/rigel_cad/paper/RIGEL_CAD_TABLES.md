# RIGEL-CAD paper tables

Source: `results/hope_cad/g3e_paper_01/RESULTS.json` (sha256 f450d20b9cd59e5e…). Final model, task-agnostic; native binned pixel AP, image AUROC; FM = forgetting.

## Main (order a, seed 2711)

| Dataset | Method | Pixel AP [CI95] | Image AUROC | FM | Bytes |
|---|---|---|---|---|---|
| MVTec AD | RIGEL-CAD (unbounded) | 0.6348 [0.6164, 0.6540] | 0.9953 | 0.0034 | 46,070,792 |
| MVTec AD | RIGEL-CAD (8 MiB) | 0.5927 [0.5747, 0.6123] | 0.9826 | 0.0333 | 8,388,104 |
| VisA | RIGEL-CAD (unbounded) | 0.4868 [0.4573, 0.5129] | 0.9746 | 0.0030 | 106,638,344 |
| VisA | RIGEL-CAD (8 MiB) | 0.4603 [0.4298, 0.4853] | 0.9349 | 0.0119 | 8,388,104 |
| MPDD | RIGEL-CAD (unbounded) | 0.5175 [0.5000, 0.5379] | 0.9627 | 0.0012 | 13,057,544 |
| MPDD | RIGEL-CAD (8 MiB) | 0.5082 [0.4895, 0.5296] | 0.9399 | 0.0128 | 8,388,104 |

## Robustness (3 orders x 3 seeds, mean ± std)

| Dataset | Method | Pixel AP | Image AUROC |
|---|---|---|---|
| MVTec AD | RIGEL-CAD (unbounded) | 0.6342 ± 0.0004 | 0.9954 ± 0.0003 |
| MVTec AD | RIGEL-CAD (8 MiB) | 0.5944 ± 0.0063 | 0.9832 ± 0.0029 |
| VisA | RIGEL-CAD (unbounded) | 0.4881 ± 0.0010 | 0.9738 ± 0.0011 |
| VisA | RIGEL-CAD (8 MiB) | 0.4505 ± 0.0052 | 0.9457 ± 0.0056 |
| MPDD | RIGEL-CAD (unbounded) | 0.5175 ± 0.0046 | 0.9616 ± 0.0013 |
| MPDD | RIGEL-CAD (8 MiB) | 0.5102 ± 0.0078 | 0.9473 ± 0.0070 |

## Ablations at 8 MiB (paired RIGEL-CAD − X, pixel AP)

| Dataset | X | Pixel AP of X | Δ [CI95] |
|---|---|---|---|
| MVTec AD | Ablation: fast level only (no slow whitening, all bytes as keys) | 0.5681 | +0.0245 [+0.0177, +0.0335] |
| MVTec AD | Ablation: RIGEL-CAD keys without whitening | 0.5425 | +0.0501 [+0.0438, +0.0581] |
| MVTec AD | Ablation: slow level from the current task only | 0.5885 | +0.0042 [+0.0003, +0.0087] |
| MVTec AD | Ablation: generic feature-covariance whitening | 0.5970 | -0.0043 [-0.0103, +0.0018] |
| MVTec AD | Ablation: FIFO key eviction | 0.3945 | +0.1981 [+0.1824, +0.2131] |
| MVTec AD | Reference: hindsight k-centre over all past coresets (not streaming-feasible) | 0.6092 | -0.0166 [-0.0210, -0.0123] |
| VisA | Ablation: fast level only (no slow whitening, all bytes as keys) | 0.3816 | +0.0786 [+0.0693, +0.0878] |
| VisA | Ablation: RIGEL-CAD keys without whitening | 0.3725 | +0.0878 [+0.0802, +0.0963] |
| VisA | Ablation: slow level from the current task only | 0.4492 | +0.0110 [+0.0078, +0.0145] |
| VisA | Ablation: generic feature-covariance whitening | 0.4359 | +0.0244 [+0.0145, +0.0322] |
| VisA | Ablation: FIFO key eviction | 0.1668 | +0.2934 [+0.2713, +0.3131] |
| VisA | Reference: hindsight k-centre over all past coresets (not streaming-feasible) | 0.4518 | +0.0084 [+0.0019, +0.0140] |
| MPDD | Ablation: fast level only (no slow whitening, all bytes as keys) | 0.4733 | +0.0349 [+0.0234, +0.0460] |
| MPDD | Ablation: RIGEL-CAD keys without whitening | 0.4550 | +0.0532 [+0.0437, +0.0644] |
| MPDD | Ablation: slow level from the current task only | 0.5036 | +0.0046 [+0.0001, +0.0104] |
| MPDD | Ablation: generic feature-covariance whitening | 0.4911 | +0.0172 [+0.0101, +0.0240] |
| MPDD | Ablation: FIFO key eviction | 0.4657 | +0.0426 [+0.0302, +0.0543] |
| MPDD | Reference: hindsight k-centre over all past coresets (not streaming-feasible) | 0.5047 | +0.0035 [-0.0035, +0.0106] |

## Budget curve (order a, 3 seeds, pixel AP mean ± std)

| Dataset | Method | Bytes | Pixel AP |
|---|---|---|---|
| MVTec AD | RIGEL-CAD (4 MiB) | 4,193,288 | 0.5583 ± 0.0151 |
| MVTec AD | RIGEL-CAD (8 MiB) | 8,388,104 | 0.5946 ± 0.0063 |
| MVTec AD | RIGEL-CAD (16 MiB) | 16,776,200 | 0.6201 ± 0.0016 |
| MVTec AD | RIGEL-CAD (32 MiB) | 33,553,928 | 0.6326 ± 0.0028 |
| MVTec AD | RIGEL-CAD (unbounded) | 46,070,792 | 0.6342 ± 0.0004 |
| VisA | RIGEL-CAD (4 MiB) | 4,193,288 | 0.3882 ± 0.0199 |
| VisA | RIGEL-CAD (8 MiB) | 8,388,104 | 0.4540 ± 0.0056 |
| VisA | RIGEL-CAD (16 MiB) | 16,776,200 | 0.4651 ± 0.0045 |
| VisA | RIGEL-CAD (32 MiB) | 33,553,928 | 0.4784 ± 0.0021 |
| VisA | RIGEL-CAD (unbounded) | 106,638,344 | 0.4881 ± 0.0012 |
| MPDD | RIGEL-CAD (4 MiB) | 4,193,288 | 0.4618 ± 0.0097 |
| MPDD | RIGEL-CAD (8 MiB) | 8,388,104 | 0.5099 ± 0.0054 |
| MPDD | RIGEL-CAD (16 MiB) | 13,057,544 | 0.5175 ± 0.0053 |
| MPDD | RIGEL-CAD (32 MiB) | 13,057,544 | 0.5175 ± 0.0053 |
| MPDD | RIGEL-CAD (unbounded) | 13,057,544 | 0.5175 ± 0.0053 |

## Matched backbone ViT-B/8 @224 (ablation)

| Dataset | Method | Pixel AP | Image AUROC |
|---|---|---|---|
| MVTec AD | RIGEL-CAD (unbounded) | 0.5748 | 0.9453 |
| MVTec AD | RIGEL-CAD (8 MiB) | 0.5018 | 0.8869 |
| MVTec AD | Ablation: fast level only (no slow whitening, all bytes as keys) | 0.4232 | 0.9018 |
| VisA | RIGEL-CAD (unbounded) | 0.2910 | 0.8783 |
| VisA | RIGEL-CAD (8 MiB) | 0.2356 | 0.7644 |
| VisA | Ablation: fast level only (no slow whitening, all bytes as keys) | 0.2176 | 0.7863 |
