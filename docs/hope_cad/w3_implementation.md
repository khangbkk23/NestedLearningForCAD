# W3 — Visual self-modifying memory cho Nested Learning for CAD

Ngày cập nhật: 2026-10-09. Branch: `hope_cad_v2`, base commit `a89fd48f94ed9c5deb26041151ea63b79911342f`. Implementation mới đang ở working tree; SHA256 từng source/test được ghi trong verification artifact, không gán code mới cho base commit.

**Trạng thái: W3 implemented và tested trên synthetic tensors CPU/CUDA local. Chưa experimentally validated cho CAD; chưa có CMS/outer learning/anomaly head v2.** Không thay architecture decision; W4 nhận output `h0` của visual block này.

## 1. Những gì đã triển khai

| File / symbol | Contract thực hiện |
|---|---|
| [scan_geometry.py](../../models/hope_cad_v2/scan_geometry.py), `scan_routes`, `ScanRoute.scan/restore`, `chunk_spans` | Row/column forward và whole-sequence reverse; inverse permutation; row chunk=width, column chunk=height; hỗ trợ rectangular fixtures |
| [visual_memory.py](../../models/hope_cad_v2/visual_memory.py), `VisualMemoryState` | 5 functional maps; learned initials riêng mỗi direction/head; không persistent working state |
| Cùng file, `stability_matched_step` | Soft injection cap, spectral clamp, inward factors, nextafter forward và continuous pre-rounding backward; xử lý alpha=1 không chia zero |
| Cùng file, `scan_chunked` | Boundary projections/readout; rank-one shared-gain recurrence; chunk-final commit cả 5 maps; autograd xuyên history |
| Cùng file, `FourScanVisualMemory` | Shared slow q, query normalization, four independent scans, inverse geometry và learned channel fusion |
| [reference.py](../../models/hope_cad_v2/reference.py), `scan_reference` | CPU FP64 oracle theo từng head/token; trực tiếp update từng map bằng `M_previous @ A + M_boundary @ B`, không dùng optimized recurrence/step helper |
| [visual_block.py](../../models/hope_cad_v2/visual_block.py), `VisualBlockConfig`, `HOPECADVisualBlock` | P 768→256; pre-LN, 256→128, depthwise 3×3; visual memory; 128→256 residual; pre-LN FFN 256→1024→256 |
| [test_visual_w3.py](../../tests/hope_cad_v2/test_visual_w3.py) | 23 mathematical/geometry/gradient/lifecycle/precision tests; dùng standard-library unittest |
| [verify_visual.py](../../scripts/hope_cad_v2/verify_visual.py) | Chạy tests và xuất JSON có environment, base commit, source hashes, failures/skips và numerical measurements |

Main defaults đúng frozen spec: 4 directions, internal width 128, 8 heads/direction, head dimension 16; grid 28×28, chunk 28; initial matrices truncated-normal std .01, bounds ±.02; controls zero; retention bias logit(.9); rate scale .025; fusion .25. Query không inner-update. Không thêm CPE, per-route skip, dropout hoặc upstream auxiliary branches.

Optimized path vectorize batch/head/direction và projections trong chunk. Square grid xử lý cả 4 routes trong một group; rectangular fixtures dùng hai chunk-length groups. Gain update khai triển rank-one để tránh dense d³ product mỗi token; vòng token vẫn tuần tự. Đây là PyTorch differentiable implementation, chưa phải custom CUDA/kernel throughput optimization.

## 2. Interface để W4 tiếp quản

```python
import torch
from models.hope_cad_v2 import HOPECADVisualBlock

visual = HOPECADVisualBlock()
masked_patch_features = torch.randn(1, 784, 768)  # synthetic fixture only
h0 = visual(masked_patch_features)               # [1, 784, 256]
# W4/W5: h0 -> CMS1 -> CMS2 -> decoder -> teacher discrepancy
```

Input thật phải là frozen-ViT features từ **masked RGB trước ViT**. W3 không nhận clean teacher target, task ID hoặc anomaly masks, và không thực hiện teacher extraction/RGB masking/CMS/decoder. Vì chưa có checkpoint ViT trong workspace, example không giả vờ là real-feature validation.

Mỗi forward tự bắt đầu từ learned initial maps. Không carry `inspect(...).final_state` sang ảnh/view sau; trạng thái đó chỉ phục vụ diagnostics. Không dùng `reset_parameters()` như per-image reset vì nó sẽ randomize learned weights. Working maps không nằm trong module buffers/checkpoint; state_dict chứa learned parameters và geometry/schema metadata. Shape/config mismatch bị reject khi load.

`scan_chunked` giữ autograd qua boundary states, quantities, gains và all initial maps; tests gồm gradcheck/gradgradcheck. Không lấy `.detach()` ở local input để dùng cho W4 exact unroll. Diagnostics được detach để tránh giữ training graph, nhưng `output` và `final_state` của recurrence vẫn có graph.

`FourScanVisualMemory.inspect(z)` trả fused/directional outputs, final working maps và chunk diagnostics. Nó không thay module/RNG state. Có thể dùng để đo gain/contraction theo vị trí; finite/bounded không chứng minh memory có ích cho anomaly detection.

FP32/FP64 được chấp nhận; half/bfloat16 input/state bị reject. Bên trong operator/block, autocast bị tắt để giữ safeguards và recurrent state đúng precision. Không âm thầm đổi half inputs thành reference features. Không có custom kernel hoặc dependency timm/pytest mới cho W3.

## 3. Lệnh đã chạy và evidence

Từ repository root:

```powershell
pixi run python scripts/hope_cad_v2/verify_visual.py
```

Latest successful execution: `2026-10-09T08:13:36Z`; **23 tests, 0 failures, 0 errors, 0 skips**. Python 3.14.4, PyTorch 2.12.0+cu126, Windows; CPU FP64/FP32 và CUDA test trên NVIDIA RTX 3050 Ti Laptop GPU. Không cần dataset, checkpoint ViT hoặc network. CUDA test tự skip nếu máy không có CUDA; report giữ skip count nên không được claim CUDA-tested từ một CPU-only run.

Artifact local: `results/hope_cad_v2/w3_verification.json` (gitignored). Nó ghi SHA256 từng file implementation/test/runner; lưu kèm source khi chuyển máy vì commit gốc không chứa working-tree changes này. Re-running verifier thay báo cáo latest tại path mặc định; dùng `--output <path>` để giữ một run riêng.

| Kiểm tra số học | Observed value |
|---|---:|
| FP64 optimized vs independent oracle, max output absolute error trên chunk sizes 1/3/7/20 | 1.39e-17 |
| FP32 recurrence vs FP64 oracle, max output absolute error | 1.43e-8 |
| FP32 final 5 maps vs FP64 oracle, max absolute error trong các maps | 3.65e-9 |
| Canonical full-grid block FP32 vs FP64 cùng weights, max absolute error | 2.45e-6 |
| Canonical full-grid block relative L2 error | 3.01e-7 |
| CUDA FP32 vs CPU reduced-fixture block, max absolute error | 4.77e-7 |
| Max chunk-relative gain spectral norm, 20 fixtures × 56 tokens, chunks 28 | 0.9172714613632416 |

FP64 assertions dùng atol 1e-9/rtol 1e-7; FP32 assertions dùng atol 1e-5/rtol 1e-4. Không nới tolerances so frozen decision. Finite-difference tests dùng float64, eps 1e-6, gradient tolerances atol 1e-5/rtol 1e-3, tránh clamp kinks; kiểm cả first/second derivatives. Oracle gradient parity kiểm trực tiếp input/query/đủ năm maps. Full spatial block kiểm finite nonzero gradients cho mọi named trainable group trên fixture không suy biến.

Lifecycle coverage: state_dict/RNG unchanged qua nhiều forwards, mỗi ảnh/view reset độc lập, batch vs từng ảnh và reversed image order, direction isolation, invalid input/failure không carry working state, state_dict roundtrip, schema mismatch. Canonical 784×768 → 784×256 có CPU forward/backward test; CUDA parity/backward hiện dùng reduced fixture, không được mô tả thành canonical T4 resource validation.

## 4. Paper/upstream provenance

Nguồn quy phạm là [architecture decision v2 §3](architecture_decision_v2.md) và [VisionHOPE arXiv v1](https://arxiv.org/html/2609.33325v1), đặc biệt chunk recurrence và Appendix B.3/B.4. Local PDF SHA256:

```text
73f157ecd534be17861bc1580ca629730201d8c708d9df28b2bc04ceebc6f97c
```

Repository tham khảo: [PSRben/VisionHOPE](https://github.com/PSRben/VisionHOPE/tree/98cfb5d6004472a0ba3d212d3cf884393e39c134), upstream main SHA quan sát ngày 2026-10-09: `98cfb5d6004472a0ba3d212d3cf884393e39c134`. [License tại commit đó](https://github.com/PSRben/VisionHOPE/blob/98cfb5d6004472a0ba3d212d3cf884393e39c134/LICENSE) là MIT, copyright VisionHOPE contributors. Đã kiểm tra README/license/reference commit; không claim đã audit toàn upstream source.

W3 được viết từ mathematical contract/paper, không vendor/copy upstream implementation và không import CUDA extensions/backbone của tác giả. Root project LICENSE vẫn áp dụng code mới. Đây là minimal CAD visual wrapper đã freeze, không upstream architecture parity claim.

## 5. Trạng thái nghiệm thu và phần tiếp theo

W3 local implementation/correctness acceptance đã đạt: geometry, five-map recurrence, guards, independent oracle, float32 precision, reset isolation và gradients. **Không có architectural deviation**; các reduced configs chỉ là test fixtures, metadata canonical=false khi khác defaults.

W4/W5 vẫn chưa triển khai: persistent CMS, exact 16-image outer calibration, normal prefix loader, RGB mask/teacher head, final CMS anomaly score và common benchmark adapter. W2 vẫn cần checkpoint ViT thật/Linux-Kaggle validation; chưa đo T4 memory/speed hoặc 16-image higher-order graph. W6/W7/W8 vẫn cần mechanism/AD/forgetting/VisA experiments. Vì vậy không claim full HOPE-CAD đã hoàn thành hoặc đã có AD gain từ kết quả 23 tests này.
