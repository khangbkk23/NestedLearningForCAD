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

## Storage formats at 8 MiB (order a, seed 2711)

Source: `results/rigel_cad/q4t_integration/VERIFY_SUMMARY.json` (sha256 42cf5ffb2844294e…). fp16 = RIGEL-CAD v1 (default); int4_triangular = optional compact memory (same algorithm, 18,672 int4 keys).

| Dataset | Storage | Pixel AP | Image AUROC | FM |
|---|---|---|---|---|
| MVTec AD | fp16 | 0.5927 | 0.9826 | 0.0333 |
| MVTec AD | int4_triangular | 0.6304 | 0.9926 | 0.0074 |
| VisA | fp16 | 0.4603 | 0.9349 | 0.0119 |
| VisA | int4_triangular | 0.4749 | 0.9707 | 0.0053 |
| MPDD | fp16 | 0.5082 | 0.9399 | 0.0128 |
| MPDD | int4_triangular | 0.5170 | 0.9598 | 0.0012 |

## Budget sweep, both storage formats (order a, seed 2711)

Source: `results/rigel_cad/budget_sweep/SWEEP_SUMMARY.json` (sha256 f2766ff7bb48d7aa…). Descriptive (exposed test sets).

| Dataset | Storage | Budget (MiB) | Keys (final) | Pixel AP | Image AUROC | FM |
|---|---|---|---|---|---|---|
| MVTec AD | fp16 | 4 | 1,194 | 0.5606 | 0.9662 | 0.0495 |
| MVTec AD | fp16 | 6 | 2,559 | 0.5840 | 0.9797 | 0.0349 |
| MVTec AD | fp16 | 8 | 3,925 | 0.5927 | 0.9826 | 0.0333 |
| MVTec AD | fp16 | 12 | 6,655 | 0.6075 | 0.9847 | 0.0263 |
| MVTec AD | fp16 | 16 | 9,386 | 0.6219 | 0.9914 | 0.0170 |
| MVTec AD | fp16 | 24 | 14,847 | 0.6281 | 0.9912 | 0.0113 |
| MVTec AD | int4_triangular | 2 | 2,372 | 0.5827 | 0.9738 | 0.0360 |
| MVTec AD | int4_triangular | 3 | 5,089 | 0.6090 | 0.9865 | 0.0235 |
| MVTec AD | int4_triangular | 4 | 7,805 | 0.6206 | 0.9895 | 0.0156 |
| MVTec AD | int4_triangular | 6 | 13,239 | 0.6263 | 0.9917 | 0.0115 |
| MVTec AD | int4_triangular | 8 | 18,672 | 0.6304 | 0.9926 | 0.0074 |
| MVTec AD | int4_triangular | 12 | 28,458 | 0.6331 | 0.9934 | 0.0031 |
| MVTec AD | int4_triangular | 16 | 28,458 | 0.6331 | 0.9934 | 0.0031 |
| MVTec AD | int4_triangular | 24 | 28,458 | 0.6331 | 0.9934 | 0.0031 |
| VisA | fp16 | 4 | 1,194 | 0.3672 | 0.8956 | 0.0792 |
| VisA | fp16 | 6 | 2,559 | 0.4418 | 0.9341 | 0.0275 |
| VisA | fp16 | 8 | 3,925 | 0.4603 | 0.9349 | 0.0119 |
| VisA | fp16 | 12 | 6,655 | 0.4620 | 0.9595 | 0.0126 |
| VisA | fp16 | 16 | 9,386 | 0.4614 | 0.9643 | 0.0133 |
| VisA | fp16 | 24 | 14,847 | 0.4744 | 0.9704 | 0.0081 |
| VisA | int4_triangular | 2 | 2,372 | 0.4213 | 0.9140 | 0.0376 |
| VisA | int4_triangular | 3 | 5,089 | 0.4532 | 0.9535 | 0.0194 |
| VisA | int4_triangular | 4 | 7,805 | 0.4659 | 0.9540 | 0.0130 |
| VisA | int4_triangular | 6 | 13,239 | 0.4711 | 0.9676 | 0.0102 |
| VisA | int4_triangular | 8 | 18,672 | 0.4749 | 0.9707 | 0.0053 |
| VisA | int4_triangular | 12 | 29,538 | 0.4759 | 0.9712 | 0.0099 |
| VisA | int4_triangular | 16 | 40,404 | 0.4821 | 0.9741 | 0.0057 |
| VisA | int4_triangular | 24 | 62,136 | 0.4869 | 0.9723 | 0.0031 |
| MPDD | fp16 | 4 | 1,194 | 0.4685 | 0.9195 | 0.0432 |
| MPDD | fp16 | 6 | 2,559 | 0.4889 | 0.9300 | 0.0246 |
| MPDD | fp16 | 8 | 3,925 | 0.5082 | 0.9399 | 0.0128 |
| MPDD | fp16 | 12 | 6,655 | 0.5172 | 0.9627 | 0.0015 |
| MPDD | fp16 | 16 | 6,965 | 0.5175 | 0.9627 | 0.0012 |
| MPDD | fp16 | 24 | 6,965 | 0.5175 | 0.9627 | 0.0012 |
| MPDD | int4_triangular | 2 | 2,372 | 0.4879 | 0.9229 | 0.0186 |
| MPDD | int4_triangular | 3 | 5,089 | 0.5096 | 0.9559 | 0.0098 |
| MPDD | int4_triangular | 4 | 6,965 | 0.5170 | 0.9598 | 0.0012 |
| MPDD | int4_triangular | 6 | 6,965 | 0.5170 | 0.9598 | 0.0012 |
| MPDD | int4_triangular | 8 | 6,965 | 0.5170 | 0.9598 | 0.0012 |
| MPDD | int4_triangular | 12 | 6,965 | 0.5170 | 0.9598 | 0.0012 |
| MPDD | int4_triangular | 16 | 6,965 | 0.5170 | 0.9598 | 0.0012 |
| MPDD | int4_triangular | 24 | 6,965 | 0.5170 | 0.9598 | 0.0012 |
