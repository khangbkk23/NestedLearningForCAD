# Phương pháp B319

Đối chiếu source: `74d975b151f59ad0b5fbf23be392e2d891f350bf` — 11/11/2026.

[Đường đọc](../README.md#đường-đọc) · [Protocol](protocol.md) · [B317](../experiments/B317/card.md)

## Ba arms và detector

Hướng này kiểm học trọng số replay có ích hơn replay đều trên cùng nền detector
hay không. Memory presence không bảo đảm AD gain.

| Arm | Replay weights | Slow controller |
| --- | --- | --- |
| `er_uniform` | Đều trên sampler-selected support IDs | Không |
| `a0_tail` | Learned policy trên raw contexts | Historical-tail query objective |
| `a0_tail_ctx` | Cùng policy, thêm CTX trên residual features | Cùng tail objective; primary arm |

Không có `a0_mean`/`current_only` trong B319 CLI; chúng thuộc trials lịch sử khác.
ER là baseline của gói này trên nền Dinomaly với cùng sampler/memory, không đồng
nghĩa toàn Dinomaly paper hoặc NL-CAD của Khang.

[Encoder](../src/nested_replay_cad/model/encoder.py) frozen tạo hai groups từ các
blocks DINOv2 đã chọn, giữ CLS/register tokens. [Decoder](../src/nested_replay_cad/model/decoder.py)
học reconstruction. `global_cosine_hm_percent` dùng global cosine loss và gradient
hook giảm đóng góp vị trí ít khó theo hard-mining schedule; không đơn thuần là
trung bình top-k patch losses. Ramp nằm trong config, recipe optimizer/hook ở decoder.

## Memory và fast update

[Sampler](../src/nested_replay_cad/core/memory/sampler.py) đọc pre-event buffers:
reservoir Algorithm R và FIFO gần nhất. Khi eligible, reserve hai recent queries,
hai historical queries và bốn replay supports; roles tách nhau, chỉ lấy arrivals
trước current. Warmup replay đều, chưa dùng learned policy/slow query. Capacities,
eligibility và RNG streams do sampler/rng định nghĩa.

Mỗi arrival có một fast step. Có replay thì support gradient tương ứng
`0.5 * current_loss + 0.5 * Σ(w_i * replay_loss_i)`; không replay thì current loss
đơn lẻ. ER dùng `w_i = 1/m`. Gradient clipping trước StableAdamW; fast optimizer
có moments/AMSGrad và tensor-wise RMS scaling. Loop không gọi scheduler dù decoder
còn chứa scheduler class. Admission diễn ra sau event, trước boundary observer;
store giữ hợp features được reservoir/FIFO tham chiếu, không đếm trùng một mẫu.
Policy học trọng số support, không học admission hoặc task order.

## Contexts và CTX

[replay.py](../src/nested_replay_cad/replay.py) tạo bốn features cho mỗi support:

1. Mean patch reconstruction residual, chia 2.
2. Mean top-79 patch residuals, chia 2.
3. Age transform `age / (age + 32)`.
4. Cosine similarity của pooled old/current features, đưa về `[0,1]`.

Contexts được detach về contiguous CPU FP32. Policy dùng hidden tanh, bounded
logits, softmax và fixed uniform floor. Layout/constants/forward/backward ở
[core/policy/layout.py](../src/nested_replay_cad/core/policy/layout.py) và
[numerical.py](../src/nested_replay_cad/core/policy/numerical.py).

[CTX](../src/nested_replay_cad/context.py) chỉ đổi **cột 0 và 1**, riêng từng cột
trên active replay set: trừ mean, chia population RMS có sigma floor, clip rồi
affine map về thang policy. Age/similarity giữ nguyên. Constants nằm trong
`normalize_context`; không có learned state, labels/category IDs/test scores,
query losses hoặc gradient ngược qua contexts.

## Slow update và numerical scope

Controller đếm eligible commits; mỗi 16 eligible events có slow step. Từ cùng
pre-event decoder/moments, dựng fast proposal rồi đánh giá query. Recent queries
dùng mean patch residual; historical queries dùng top-79 mean. Bốn contributions
có trọng số bằng nhau. Có thêm KL regularizer về uniform. Query chỉ lấy normal
training memory, không lấy test/anomaly images.

Đường tác động: policy weights → support gradient → fast proposal → query loss.
[fast_update.py](../src/nested_replay_cad/fast_update.py) bind primals/branch choices
theo forward-source dtype, tính FP64 chain-rule **first conditional VJP**. Policy
backward dùng hiệu Jacobian theo cặp; [slow Adam](../src/nested_replay_cad/core/learning/adam.py)
cập nhật phi. Loop khôi phục pre-event model, replay support RNG draws rồi commit
fast step thật; proposal không thành fast step thứ hai. Slow state được commit
sau fast-update consistency probes.

Đây là derivative convention có scope, không là literal derivative của mọi phép
làm tròn FP32 hoặc higher-order oracle. Xem B317 card cho qualification và
actual-episode gaps. Không tự chuyển qualification của source/bundle cũ sang code
đã đổi namespace/runtime. `COMPLETE_REVIEW_PENDING` khác scientific acceptance.

## B321 khác B319 ở đâu

Các mục trên mô tả B319. B321 là biến thể grouped, chạy qua
[configs/b321.json](../configs/b321.json) và [scripts/budget_train.py](../scripts/budget_train.py);
thiết kế, giới hạn và kết quả ở [card B321](../experiments/B321/card.md). Dưới đây là mô tả
vai trò sau khi đọc source, không phải audit hành vi.

- **Lịch update:** 667 update/task, không còn một fast step mỗi arrival. Stage 1 lấy 16
  current mỗi batch, chưa replay. Từ stage 2 mỗi batch có 8 current và 8 replay phân biệt,
  chia hai nhóm bốn support (`Batches` và `Schedule.reserve`).
- **Loss khi có replay:** mỗi current mang hệ số `0.5/số_current`, mỗi replay mang `0.25·w`,
  tức hai nhóm bốn; ở uniform `w=0.25` tổng replay bằng 0.5. Khi không có replay (stage 1),
  current mang `1/số_current` (`Learner.step`). Công thức B319
  `0.5·current + 0.5·Σ(w_i·replay_i)` không áp dụng nguyên cho B321.
- **Controller:** chỉ arm learned có controller; trọng số học chỉ dùng khi batch đủ điều
  kiện, nếu không dùng 0.25 đều. Query gồm 2 recent + 2 historical, slow step mỗi 16
  eligible batch.
- **Checkpoint và resume:** B321 có checkpoint theo update
  ([budget_checkpoint.py](../scripts/budget_checkpoint.py)) mà B319 chưa có. Qualification
  native resume (run `b321-resume-qual-20261114-02`, version 4) chỉ là tổng hợp và có phạm
  vi hẹp; không coi là qualification cho run thật.

## B322–B324

Ba thí nghiệm sau B321 giữ recipe B319 (một fast update mỗi arrival) và thử ba cơ chế
khác nhau: readout bằng trung bình trọng số theo thời gian (B322), neo hàm số về một
bản sao decoder chụp ở mỗi boundary (B323), và đổi định dạng bộ nhớ replay sang ảnh
đầu vào uint8 rồi so lịch cập nhật hai tần số với lịch phẳng (B324). Định nghĩa chính xác
ở card [B322](../experiments/B322/card.md), [B323](../experiments/B323/card.md),
[B324](../experiments/B324/card.md).
