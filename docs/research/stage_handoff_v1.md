# Stage handoff v1 — hợp đồng triển khai

Ngày chốt: 24/10/2026. User giao implement ZIP + gate trong bootstrap. Đây là operational amendment, không đổi phương pháp, study schedule, precision, gates hoặc frozen resources. Proposal rộng hơn [stage_handoff_spec](stage_handoff_spec.md) giữ lại cho future revisions; file này chốt **subset v1**.

Trạng thái: implementation và CPU fixtures đã thêm; chưa real Linux/cross-host CUDA acceptance. Không tự chạy paid stage chỉ vì local tests PASS.

## Quyền hỗ trợ

- Phase1=B preflight+pilot; Phase2=C ba seed0jobs train/final/matrix; Phase3=D sáu controls seed0+sáu replicates fresh. Hai nhánh3090/4090 độc lập.
- Chỉ exact source/study/runtime/data/backbone/producer/backend identity. Same GPU khácdriver vẫn blocked. Không source mapping, mixed-runtime aggregation, changed-runtime learner resume hoặc tự acceptance.
- Restore toàn experiment đã quiescent vào một experiment trống bằng staging+atomic rename. Original locks/receipts/events/states/cells giữ bytes. Engine hiện tại skip completed work; không cần nới loader hoặc aggregator khi identity bằng nhau.
- Pack sau từng invocation/phase bằng `handoff-pack`; một ZIP có live run, logs, immutable export sessions, records, asset receipts và verified source snapshot. Không dataset/weights/.pixi/keys. Gói không tự là trusted reviewer attestation; chỉ nhận từ nguồn user tin cậy và giữ SHA ngoài host.
- Gate chỉ đọc/kiểm/probe trên discarded state. Không train, ghi review acceptance, Stop/Destroy hoặc upload cloud. Fail không tự chạy lại từ đầu.

## Interface và dữ liệu

`handoff-pack --run-dir ... [--output ...]`: canonical experiment/run; output mặc định sibling `handoffs/<timestamp_uuid>.zip`. Không lấy outer ZIP cũ vào payload. Manifest `hope_stage_handoff_v1` gồm files{path:{sha256,bytes}}, experiment_id, source_commit, source_digest, source_file_hashes, suite_identity. Payload prefixes experiment/, records/, assets/, source/. Từng ZIP và manifest được verify trước publish; SHA cuối in ra màn hình.

`handoff-check --archive ZIP [--restore] [--device cuda:0] [--expected-gpu NAME] [--rate USD] [--cap USD]`: verify toàn ZIP, current source + resource/data/checkpoint/runtime identity trước extract vào staging; structural coverage và actual state loads trên đúng numerical device đã qualified, rồi discarded real CUDA feature/read-only score probe. Pending predictions được loader tính lại bằng exact equality: không dùng CPU để xác nhận một pending state sinh bởi GPU. Report JSON dưới workspace/records/handoff_checks và stdout. Chỉ publish restore nếu hợp lệ. Imported records/source ở experiment/origin/<archiveSHA>/, không overwrite target records/source. Read-only check không restore có `compatible=true,allowed=false,restore_required=true`, không in lệnh chạy vào thư mục chưa tồn tại.

Không archive → bootstrap báo fresh Phase1. Archive invalid/khácsource/driver/budget → blocked, giữ ZIP và report. Existing target không bị overwrite: nếu đã restore thì resume bằng lệnh đã in khi restore; không import lại ZIP cũ lên live run đã tiến triển và không rename logical job để lách gates. Review pending/negative hoặc fully completed chỉ xuất report, chưa publish run để chạy tiếp; giữ ZIP để review/inspect riêng.

## Qualification và coverage

Kiểm manifest train3629/209Bottle/15tasks/zero exclusions và digest, registered suite/job/method/rows/order, active state + SHA + event chain + seen IDs/finalized, task checkpoint SHA/boundary, initialization và feature receipts. Pack trên host gốc kiểm recorded runtime trước actual state load/forward cho pending; không train nhưng không phải công cụ hoàn toàn CPU-only. Nếu runtime gốc không còn, preserve raw experiment/exports, không coi metadata-only copy là qualified handoff v1. Completed main matrix phải đủ120cells, exact job/test/checkpoint/chunk/row bindings; no fake completion từ summary filenames. Phase1 cần chín completed arms, pilot gates PASS và actual review bound receipt/source. Phase2 cần ba complete matrices và recomputed registered development screen; Phase3 cần12conditional jobs còn lại. Partial jobs được resume chỉ với committed state hợp lệ; thiếu/corrupt không coi là fresh.

Fixed target probe dùng allowed train IDs ở indices0/128/255 nếu đã observed, tối thiểu ID0 cho từng existing job. Re-extract cùng rawFP32, đối chiếu source tensor receipts; load active/task model và score read-only/finite trên disposable clone, không update learner/bank. Probe có thể chạy CUDA vài forwards; không thay research preflight/pilot hoặc chứng minh mọi future trajectory. Không reference feature receipt → blocked. Không có model/jobstate mới chỉ preflight: verify runtime+raw preflight feature receipts nếu tồn tại, chưa mở main trước accepted pilot.

## Budget và bootstrap

Mọi budget_B/C/D.json và accounting sidecars giữ nguyên. Import chỉ nhận fully closed reservations; interrupted/open/unbound reconciliation → budget_unresolved. Next stage chưa có ledger có thể cấp allocation mới theo actual rate/cap user nhập, riêngB/C/D. D user capUSD/rate xác định GPU-wall cap được cấp, không unlimited. Existing stage ledger phải cùng allocation/rate/cap; đổi rate giữa partial stage **blocked ở v1**, không reset hay tự sửa lịch sử. Completed B khác giá không ngăn tạo C allocation mới; completed C không được tính lại giá khi sang D. Pack/upload/target setup/idle chưa được ledger đo tự động: operator phải ghi một lần bằng `--external-billed-seconds` theo scope stage; gate report nhắc rõ, không giả phép kiểm đã tạo hóa đơn/charge cho toàn khoảng đó.

Bootstrap nhận `HOPE_HANDOFF_ZIP=/workspace/file.zip` optional, truyền qua tmux worker; kiểm manifest header để pin source_commit trước clone (existing checkout mismatch thì fail, không reset). Không tự download ZIP từDrive. Sau Pixi/assets/tests, gate restore+report, tạo đúng next-stage allocation khi user input hợp lệ, in lệnh resume/next-job. Runtime mismatch không clone/sửa suite-lock; source thiếu trên remote thì fail rõ. Same source commit với Linux-normalized files phải khớp digest trong ZIP. Dirty source pack bị từ chối; phải commit/push trước cohort muốn portable.

## Kiểm chứng trước phát hành

CPU fixtures: ZIP roundtrip/hash/path traversal/symlink/casecollision, missing/corrupt payload, concurrent writer, foreign run, source/runtime mismatch, partial/complete B/C/D scheduling, blocked negative gates, actual state load/read-only probe, immutable restore, allocation/rate/open-reservation rejection. Mock bootstrap fresh/archive/source-pin/fail-before-clone, shell syntax; full regression và clean Pixi install. Real Linux/cross-host CUDA acceptance vẫn pending cho đến có actual artifacts. Không gọi local test là fully portable/GPU acceptance.
