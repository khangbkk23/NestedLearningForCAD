# W3 — Visual self-modifying memory cho Nested Learning for CAD — báo cáo v2

Ngày cập nhật báo cáo: 2026-10-10. Nhánh hiện hành `hope_cad_v2_1` đã có source W3 `363634b`, bắt đầu chặng 1 tại `5187e4d`; suite W3 chạy lại pass 24/24. Không sửa các source W3 đã audit. Historical local 23-test và artifact Kaggle 24-test vẫn giữ riêng. [Báo cáo chặng 1](stage1_implementation.md) bổ sung init contraction, W4 correctness và handoff upstream/T4.

**Trạng thái: W3 implemented/tested tại source `363634b`, đã có trên nhánh hiện hành và pass lại suite. Chưa experimentally validated CAD.** Historical Kaggle 24/24 có 7/7 hashes khớp source, canonical CPU và reduced CUDA. Chặng 1 thêm CMS/outer learning ở mức feature predictor; full RGB adapter, upstream numerical parity và exact canonical-T4 profile vẫn pending. Không suy hiệu quả CAD từ test numerical.

Các đường dẫn source trong bảng tiếp tục trỏ commit `363634b` khớp historical artifact Kaggle; source đó hiện cũng có trong checkout. [Decision](architecture_decision_v3.md), [planning](planning_CAD_v3.md), [instruction](instruction_CAD_v3.md) giữ contract; [báo cáo chặng 1](stage1_implementation.md) ghi trạng thái mới. Namespace `hope_cad_v2` giữ nguyên.

## 1. Những gì đã triển khai

| File / symbol | Contract thực hiện |
|---|---|
| [scan_geometry.py](https://github.com/khangbkk23/NestedLearningForCAD/blob/363634b18d98eefc6da28660a99b967988077ef3/models/hope_cad_v2/scan_geometry.py), `scan_routes`, `ScanRoute.scan/restore`, `chunk_spans` | Row/column forward và whole-sequence reverse; inverse permutation; row chunk=width, column chunk=height; hỗ trợ rectangular fixtures |
| [visual_memory.py](https://github.com/khangbkk23/NestedLearningForCAD/blob/363634b18d98eefc6da28660a99b967988077ef3/models/hope_cad_v2/visual_memory.py), `VisualMemoryState` | 5 functional maps; learned initials riêng mỗi direction/head; không persistent working state |
| Cùng file, `stability_matched_step` | Soft injection cap, spectral clamp, inward factors, nextafter forward và continuous pre-rounding backward; xử lý alpha=1 không chia zero; spectral limit dùng rsqrt² tại `363634b` |
| Cùng file, `scan_chunked` | Boundary projections/readout; rank-one shared-gain recurrence; chunk-final commit cả 5 maps; autograd xuyên history |
| Cùng file, `FourScanVisualMemory` | Shared slow q, query normalization, four independent scans, inverse geometry và learned channel fusion |
| [reference.py](https://github.com/khangbkk23/NestedLearningForCAD/blob/363634b18d98eefc6da28660a99b967988077ef3/models/hope_cad_v2/reference.py), `scan_reference` | CPU FP64 oracle theo từng head/token; trực tiếp update từng map bằng `M_previous @ A + M_boundary @ B`, không dùng optimized recurrence/step helper |
| [visual_block.py](https://github.com/khangbkk23/NestedLearningForCAD/blob/363634b18d98eefc6da28660a99b967988077ef3/models/hope_cad_v2/visual_block.py), `VisualBlockConfig`, `HOPECADVisualBlock` | P 768→256; pre-LN, 256→128, depthwise 3×3; visual memory; 128→256 residual; pre-LN FFN 256→1024→256 |
| [test_visual_w3.py](https://github.com/khangbkk23/NestedLearningForCAD/blob/363634b18d98eefc6da28660a99b967988077ef3/tests/hope_cad_v2/test_visual_w3.py) | 24 mathematical/geometry/gradient/lifecycle/precision tests, gồm regression spectral-backward gần floor; dùng standard-library unittest |
| [verify_visual.py](https://github.com/khangbkk23/NestedLearningForCAD/blob/363634b18d98eefc6da28660a99b967988077ef3/scripts/hope_cad_v2/verify_visual.py) | Chạy tests và xuất JSON có environment, base commit, source hashes, failures/skips và numerical measurements |

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

Lệnh historical dưới đây chạy từ repository root **có source W3**; HEAD clone hiện chưa thể chạy/import example này trước khi được phép tích hợp code:

```powershell
pixi run python scripts/hope_cad_v2/verify_visual.py
```

Artifact verifier lịch sử đã xác minh: `2026-10-09T08:13:36Z`; **23 tests, 0 failures, 0 errors, 0 skips**. Python 3.14.4, PyTorch 2.12.0+cu126, Windows; CPU FP64/FP32 và CUDA test trên NVIDIA RTX 3050 Ti Laptop GPU. Không cần dataset, checkpoint ViT hoặc network. CUDA test tự skip nếu máy không có CUDA; report giữ skip count nên không được claim CUDA-tested từ một CPU-only run.

Artifact local: `results/hope_cad_v2/w3_verification.json` (gitignored). Ngày 2026-10-10 đã đối chiếu SHA256 của đủ 7 source/test/runner từ artifact với Git blobs `6dbe4f1`: tất cả khớp. Trường `source_commit=a89fd48` trong artifact ghi base lúc W3 còn untracked, không chứng minh base chứa implementation. Tại `363634b`, `visual_memory.py` và `test_visual_w3.py` không còn khớp artifact; 5 file còn lại khớp. Không overwrite artifact cũ hoặc chuyển 23 tests thành 24 tests đã chạy. Verifier có `--output <path>` để lưu run mới riêng sau tích hợp.

| Kiểm tra số học — historical local `6dbe4f1`, artifact 23-test | Observed value |
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

### 3.1. Source update sau artifact và trạng thái kiểm chứng

**Spectral-guard implementation đã kiểm thử tại `363634b`.** Với `d=key.square().sum(-1).clamp_min(1e-20)`, tính `inverse_key_norm=torch.rsqrt(d)` rồi `spectral=(2*alpha*(1-1e-6))*inverse_key_norm.square()` trước `nextafter` hướng zero và backward mapping hiện có. Dạng này tương đương bound trong real arithmetic, nhưng FP32/autograd có thể khác phép chia trực tiếp; phải kiểm numerical/gradient parity, không suy từ đẳng thức đại số rằng mọi expression có cùng backward ổn định. Traceback Kaggle trước fix chỉ tới `DivBackward0` tại phép chia spectral limit. Không nâng denominator floor, đổi retention/init, bỏ guard hoặc nới tolerance để xử lý lỗi. Regression `test_spectral_limit_backward_is_finite_near_denominator_floor` tại `tests/hope_cad_v2/test_visual_w3.py` dùng key FP32 `1.01e-10`, kiểm first-order gradients hữu hạn gần floor; zero-key/saturated-alpha và full-grid backward cũng được suite kiểm. Gradgrad tests hiện có dùng fixtures riêng, không tự chứng nhận higher-order gần floor hoặc exact 16-event W4. Đây là numerical bug fix có source hash riêng; topology, phương trình, design/method ID và selection rule v3 giữ nguyên.

[Commit `363634b`](https://github.com/khangbkk23/NestedLearningForCAD/commit/363634b18d98eefc6da28660a99b967988077ef3) đã được artifact Kaggle xác nhận ở scope suite 24-test dưới đây; không còn chỉ là test-source-present. Task tài liệu này kiểm lại artifact/source hashes, không tự chạy lại tests hoặc tích hợp code vào clone.

#### Lịch sử failure → fix → verification

| Mốc / source | Evidence | Kết luận đúng phạm vi |
|---|---|---|
| `6dbe4f1`, local Windows | `results/hope_cad_v2/w3_verification.json`, 23/23 pass; 7/7 hashes khớp | Historical numerical/lifecycle acceptance trên runtime local đã ghi ở §3 |
| `6dbe4f1`, Kaggle trước fix | User logs trong `chat.json`, `requests[13]`, `[14]`, `[19]`: 22/23 pass; canonical CPU backward fail; anomaly traceback chỉ tới `DivBackward0` trong `stability_matched_step` | Runtime portability/numerical failure đã được quan sát. CUDA reduced test vẫn pass; không quy lỗi riêng cho GPU T4 |
| `363634b`, fix | `visual_memory.py::stability_matched_step`, `RecurrenceTests.test_spectral_limit_backward_is_finite_near_denominator_floor` | Giữ floor/bound/init, đổi expression thực thi và thêm regression; không nới assertion |
| `363634b`, Kaggle sau fix | `w3_verification_kaggle.json`, 24/24 pass, 0 failures/errors/skips; 7/7 source hashes khớp | W3 correctness suite đã tested trong môi trường Kaggle có T4; chưa là full-method/T4 resource/AD validation |

Transcript nguồn do người dùng cung cấp là `C:/Users/trana/Downloads/chat.json`; nó không được copy vào repo trong task này. Logs trước fix là transcript evidence, không có failed-run JSON riêng đã được kiểm hash trong repo; không suy diễn ra source hashes hoặc environment fields bị thiếu. Kết quả sau fix có artifact độc lập đã kiểm tra, không chỉ lời mô tả của Copilot.

#### Artifact Kaggle đã xác minh

Artifact repo: [artifact Kaggle](../../results/hope_cad_v2/w3_verification_kaggle.json). File được tạo `2026-10-09T15:46:02Z`, tương ứng **22:46:02 ngày 2026-10-09, Asia/Bangkok**; ngày đối chiếu source/hash trong task tài liệu là 2026-10-10. SHA256 bytes của artifact:

```text
a4e765c2ff2654945877327764ec6fea2c10cbd63a971f23cc5ba88e5ffc3251
```

| Trường | Giá trị |
|---|---|
| Source commit | `363634b18d98eefc6da28660a99b967988077ef3` |
| Tests / status | 24, success=true, 0 failures, 0 errors, 0 skips |
| Python | 3.13.15, GCC 13.3.0 |
| OS | Linux-6.18.48+-x86_64-with-glibc2.39 |
| PyTorch / CUDA runtime | 2.11.0+cu128 / 12.8 |
| CUDA device | Tesla T4 |
| Elapsed suite | 6.706445296999959 s; không phải full-model throughput |
| Source integrity | 7/7 SHA256 khớp Git blobs tại source commit |

Hai files thay đổi so với artifact local 23-test:

| Source | SHA256 tại `363634b` |
|---|---|
| `models/hope_cad_v2/visual_memory.py` | `932ef162d02eae47aef5b3dd2ac21b9e1bf5d548318a9d53c228100644ea9ee1` |
| `tests/hope_cad_v2/test_visual_w3.py` | `c040b4347ac52ce837cbcf6af9a243bb27ee305486bd87737b358fde0ffca5c5` |

Năm source/runner files còn lại cũng đã được đối chiếu, hashes giữ nguyên giữa hai commits; danh sách đầy đủ nằm trong JSON. Bản sao mới có tên `_kaggle` để phân biệt với historical local artifact, không thay nội dung run hoặc overwrite file cũ.

Lệnh notebook đã dùng, từ checkout `363634b` trên Kaggle:

```bash
python -u scripts/hope_cad_v2/verify_visual.py --output /kaggle/working/w3_verification.json
```

| Kiểm tra số học — Kaggle `363634b` | Observed value |
|---|---:|
| FP64 optimized vs independent oracle, max output absolute error trên chunk sizes 1/3/7/20 | 2.0816681711721685e-17 |
| FP32 recurrence vs FP64 oracle, max output absolute error | 1.4260912145624971e-8 |
| FP32 final 5 maps vs FP64 oracle, max absolute error trong các maps | 3.6489945639747723e-9 |
| Canonical CPU full-grid block FP32 vs FP64, max absolute error | 3.8356514462378755e-6 |
| Canonical CPU full-grid block relative L2 error | 3.9874268127461064e-7 |
| CUDA FP32 vs CPU reduced-fixture block, max absolute error | 3.5762786865234375e-7 |
| Max chunk-relative gain spectral norm | 0.9172714613632416 |

**Phạm vi đã kiểm:** canonical `[1,784,768] → [1,784,256]` forward/backward trên CPU của phiên Kaggle; CUDA parity/autocast/backward trên reduced fixture (`VisualBlockConfig(10,8,4,2,3,4)`, input `[2,12,10]`). Không có full-grid GPU peak VRAM/throughput, exact 16-event higher-order CMS, backbone/real data, upstream-core comparison hoặc CAD score/forgetting trong suite. First/second derivatives trên existing small fixtures không chứng minh mọi higher-order edge case gần norm floor đã đạt. W2 có thêm runtime smoke evidence nhưng chưa đóng data/weights/extractor/harness/resume acceptance.

`results/` bị `.gitignore` loại khỏi Git; file đang có trong workspace không đồng nghĩa đã có trên remote. Khi handoff cần export artifact cùng source/spec hashes vào bundle lưu bền vững; giữ cả successful và failed-run evidence theo scope đang có. Không force-add/commit artifact trong task chỉ cập nhật tài liệu này.

### 3.2. Contraction audit bổ sung

**Bằng chứng và phạm vi.** Audit trước đã chạy `FourScanVisualMemory.inspect` với seed 103, input Gaussian `[1,784,128]`, FP32, chưa outer calibration. Norm tính FP64: mean initial content norm ≈`1.40e-1`, final ≈`1.36e-37`; direction-0 output RMS ở chunk đầu ≈`8.73e-3`, chunk thứ 14 ≈`1.42e-19`, chunk cuối ≈`1.63e-37`. Đây là synthetic measurement trong transcript audit trên source W3 trước fix NaN, **không nằm trong artifact 23-test và không phải CAD experiment**. Cần lưu artifact tái lập khi làm diagnostics mới; không gán các số này cho `a89fd48` hoặc coi artifact Kaggle 24-test là phép đo lại contraction sau fix.

Ở initialization, zero α-map cho retention 0.9; `0.9^784` giải thích một bậc độ lớn khả dĩ, **không phải định luật co chính xác cho mọi input** vì injection, normalized key và boundaries cùng tham gia recurrence. Zero control maps giữ zero qua inner right-action trước outer learning không có nghĩa outer gradient tới initialization bằng zero. Không suy ra mọi outer gradient giảm đúng `0.9^t`, hoặc outer learning không thể phục hồi utility.

**Thứ tự thực hiện:** ngay sau W3, trước đầu tư đầy đủ W4, đo init trên fixtures cố định; thêm real features chỉ từ `meta_support`/`meta_query` khi W2 có extractor/weights. Theo từng direction/head/chunk và restored spatial position, ghi memory/gain norms, output RMS, zero/subnormal fractions, α/η distributions và cap activity. Với full spatial wrapper, ghi `RMS(P_out(V(Z)))/max(RMS(X),1e-12)` và visual-on/off output deltas. Khi minimal W4 đủ chạy vài exact outer steps, đo lại ở steps 0/1/5/25 và selected checkpoint; chạy gradient diagnostics theo nhóm initial maps/q/projections/decoder/CMS initials, rồi final-score interventions ở W5/W6. Không chờ tới W7 mới nhìn contraction.

**Acceptance:** numerical NaN/Inf, state mutation sai hoặc gradient bị disconnect do implementation là correctness failure. Biên độ nhỏ ở init là **warning cần đo tiếp**, không tự động fail kiến trúc hoặc ép đổi initialization. Diagnostics completeness là gate sớm; utility được đánh giá sau calibration bằng trained static/reset/freeze controls và final-score sensitivity. Không đặt một cutoff dựa vào fixture đã xem rồi gọi là preregistered pass. Nếu chưa có bằng chứng utility thì ghi inconclusive/no-go cho claim H3, không bịa threshold đảm bảo gain. Mọi sửa init/retention/guards để xử lý quan sát phải thành method revision trước anomaly-test evaluation.

Cách tái lập fixture lịch sử trên source W3 được ghi (không chạy được ở HEAD clone thiếu source):

```python
import torch
from models.hope_cad_v2 import FourScanVisualMemory
torch.set_num_threads(2)
torch.manual_seed(103)
model = FourScanVisualMemory()
z = torch.randn(1, 784, 128)
with torch.no_grad():
    result = model.inspect(z)
print(model.initial_content.double().norm(dim=(-2, -1)).mean().item())
print(result.final_state.content.double().norm(dim=(-2, -1)).mean().item())
for chunk in (0, 3, 9, 13, 27):
    values = result.directional_outputs[:, 0, chunk*28:(chunk+1)*28].double()
    print(chunk, values.square().mean().sqrt().item())
```

Để tạo evidence mới cần lưu source/config/input hashes, environment, dtype/device và raw measurements, không chép số cũ làm test result mới. Phép đo trên là memory-only; ratio full-wrapper-vs-residual, real features, post-outer learning và final-score utility chưa được fixture này đo.

## 4. Paper/upstream provenance

Contract hiện hành là [architecture decision v3 §3](architecture_decision_v3.md); source W3 ban đầu theo [decision v2 §3](architecture_decision_v2.md) và [VisionHOPE arXiv v1](https://arxiv.org/html/2609.33325v1), đặc biệt chunk recurrence và Appendix B.3/B.4. Local PDF SHA256:

```text
73f157ecd534be17861bc1580ca629730201d8c708d9df28b2bc04ceebc6f97c
```

Repository tham khảo: [PSRben/VisionHOPE](https://github.com/PSRben/VisionHOPE/tree/98cfb5d6004472a0ba3d212d3cf884393e39c134), upstream main SHA quan sát ngày 2026-10-09: `98cfb5d6004472a0ba3d212d3cf884393e39c134`. [License tại commit đó](https://github.com/PSRben/VisionHOPE/blob/98cfb5d6004472a0ba3d212d3cf884393e39c134/LICENSE) là MIT, copyright VisionHOPE contributors. Ngoài README/license, audit tiếp theo đã đối chiếu các paths `models/srnl.py`, `models/srnl_core/chunk.py`, `models/srnl_core/csrc/chunk.cu` và `memory.cuh`; không claim đã audit toàn upstream source. Direct numerical parity vẫn chưa chạy.

W3 được viết từ mathematical contract/paper, không vendor/copy upstream implementation và không import CUDA extensions/backbone của tác giả. Root project LICENSE vẫn áp dụng code mới. Đây là minimal CAD visual wrapper đã freeze, không upstream architecture parity claim.

### 4.1. Upstream parity mới cần thực hiện

Recurrence ở trên giữ nguyên: `M_t = M_(t-1) A_t + M_b B_t`, tương đương `G_t=α_t G_(t-1)-η̃_t(G_(t-1)k_t+δ_t)k_tᵀ`. Đối chiếu đại số với Eq.45–46 Appendix B.4 không cho thấy lỗi “đếm đôi”; token oracle và optimized path cùng khớp spec chưa đủ để chứng minh numerical parity với tác giả.

Nguồn upstream ghim `98cfb5d6004472a0ba3d212d3cf884393e39c134`, MIT: [SRNL wrapper](https://github.com/PSRben/VisionHOPE/blob/98cfb5d6004472a0ba3d212d3cf884393e39c134/models/srnl.py), [chunk CUDA](https://github.com/PSRben/VisionHOPE/blob/98cfb5d6004472a0ba3d212d3cf884393e39c134/models/srnl_core/csrc/chunk.cu). Source audit chỉ bao phủ các operator paths liên quan, không toàn repository. Upstream có clamp pre-activations η/α ở ±10, query-normalization conventions cần đồng nhất, và custom operator hỗ trợ first-order gradients. W3 giữ unclamped paper-form controls và query eps 1e-6; không tự thêm clamp hoặc thay wrapper/init theo upstream trong revision tài liệu này.

**Gate bổ sung W3 — proposed/not executed:** gọi trực tiếp low-level upstream core và local reference trên cùng explicit tensors/initial maps/normalized queries, cùng layout, guards, dtype và chunk boundaries. Không dùng “cùng seed” thay việc copy chính xác weights; map states theo tên content/key/value/eta/alpha, không giả định positional argument order của hai APIs giống nhau. Chốt fixtures seeds 103/104/105, head dimensions 4/16, chunk lengths 1/7/28, nhiều chunks và ít nhất một sequence 784 tokens; bao gồm multi-image isolation và boundary readout. Main state-parity fixtures có sequence length chia hết chunk length. Với partial tail, [upstream normalized-query path](https://github.com/PSRben/VisionHOPE/blob/98cfb5d6004472a0ba3d212d3cf884393e39c134/models/srnl_core/chunk.py) padding tới đủ chunk: so valid-token outputs trước, ghi riêng padded terminal state, không nhầm nó với state ngay sau last real token. Fixtures parity chính giữ logits trong vùng clamp upstream không hoạt động. Fixtures ngoài vùng này ghi expected divergence do contract khác, không nới tolerance để gọi khớp.

So output FP32 và boundary/final states mà API thực sự expose: `atol=1e-5, rtol=1e-4`. So first-order gradients với inputs, normalized queries và đủ năm initial maps trên fixtures không tại clamp/min kink: `atol=1e-5, rtol=1e-3`; ghi max absolute/relative errors theo nhóm. Nếu cần instrumentation để expose states, lưu patch/hash harness; nếu chưa quan sát được state thì đánh dấu pending, không claim state parity. Ghi upstream SHA/license, build/runtime, source hashes và inputs hash vào artifact riêng.

Higher-order/gradgrad vẫn kiểm bằng local FP64 oracle và finite differences; không dùng upstream first-order kernel để chứng nhận Hessian hoặc thay vào exact 16-event outer unroll. Chưa chạy gate trực tiếp thì nhãn là **paper/operator mapping audited; upstream numerical parity pending**. W4 reference có thể được dựng tiếp sau diagnostics W3; fidelity claim và scientific use của backend cần gate phù hợp đã đạt, không giả định wrapper/full-backbone parity.

## 5. Trạng thái nghiệm thu và phần tiếp theo

W3 numerical/lifecycle acceptance **theo phạm vi suite hiện có** đã đạt tại source `363634b` trong môi trường Kaggle có T4, với 24 tests và 7/7 hashes khớp; historical local `6dbe4f1` vẫn giữ riêng. Full-grid CPU và reduced CUDA được phân biệt như §3.1. Không có architectural deviation so với mathematical contract W3 v2; điều này không là upstream/full-backbone parity. Reduced configs chỉ là test fixtures, metadata canonical=false khi khác defaults. V3 bổ sung upstream-core parity và early contraction diagnostics: các phần chưa chạy giữ pending, không đánh dấu toàn bộ expanded W3 acceptance đã xong từ suite này. Tiny initialization output không tự fail kiến trúc; post-calibration utility cần minimal W4 rồi W5/W6.

Trên clone hiện tại, W3 source chưa tích hợp; cần source/test handoff được cho phép trước W4. W4/W5 vẫn **not implemented**: persistent CMS, exact 16-image outer calibration, prefix loader/long gate `prefix_cms_stress_v3`, C4 inherited β, RGB masks/frozen teacher, final CMS anomaly score và benchmark adapter. W2 đã có Kaggle W3 runtime smoke nhưng còn weights/data/extractor, full-pipeline dependency/harness/resume acceptance; early exact-unroll resource spike chưa có measurement. W6/W7/W8 cần mechanism/AD/forgetting/VisA experiments. 24-test Kaggle pass không chứng minh full HOPE-CAD hoàn thành hoặc có AD gain. Task này chỉ cập nhật bốn tài liệu mới, giữ code, các tài liệu v2 gốc và HEAD clone.
