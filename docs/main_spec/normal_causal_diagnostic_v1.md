# Nested Learning for CAD — Gói chẩn đoán nhân quả normal-only v1

Ngày khóa protocol: 17/10/2026. Worktree: `hope_cad_v5`. Đây là **thí nghiệm phụ trên clone bị discard**, không phải method revision, không thay đặc tả v4, không refit toàn W6 và không chạy full W7.3/VisA.

## 1. Vấn đề cần phân biệt

Recovery đã xác minh M seed0 có I-AUROC/P-AUPR 66,8010/17,0163%, matched CADIC 10k có 88,3571/46,0137%. Các số này là thông tin development đã biết; không dùng để chọn category, arm, LR hay checkpoint trong gói này. Review gốc: `docs/recovery_stage1_kaggle/main/review.md`; audit trước revision: `pre_revision_audit.md` cùng thư mục. Hai tài liệu và các kết quả gốc giữ nguyên.

Một decoder cố định sau calibration trên task đầu có thể hạn chế khả năng CMS khớp target của category mới. Tuy nhiên sai hướng, acquisition yếu, retention giảm và fit chạm cap 400 chỉ là dấu hiệu, chưa chứng minh nguyên nhân. Gói hữu hạn này hỏi:

1. Với **cùng state, encoded features, dữ liệu và số event**, tối ưu CMS bằng Adam có giúp hơn dynamics CMS triển khai không? Cho decoder học có tạo thêm dư địa khớp normal trên ảnh khác không? Khi khớp current normals, loss ở past normals biến đổi thế nào?
2. Với **đúng một recipe M đã chọn**, nối tiếp optimizer/RNG từ 400 tới 500 có còn giảm normal validation loss rõ rệt không?

Không kiểm tra exhaustive hyperparameters, mọi biến thể Nested Learning, hiệu quả anomaly ranking, hoặc chứng minh một nguyên nhân duy nhất. Không có yêu cầu rằng loss phải giảm để thí nghiệm được coi là hoàn tất.

## 2. Nguồn chuẩn và quyền dữ liệu

Protocol máy đọc: `normal_causal_diagnostic_v1.json`. SHA đầy đủ được khóa trong `diagnostic_lock.json` của mỗi run.

| Parent | Điều kiện bắt buộc |
|---|---|
| Completed recovery main | manifest SHA `087f167eb0fbcb935e36148ceaf64d373aa196894e928ac9b17aad49b46db6cc` |
| Completed W6 v4 | manifest SHA `5892af759a096a6f2ba658eff5a3095aadc94c05bdb7c5849bfb6b122f37d4cd` |
| Immutable source r4 | ZIP SHA `2ad2a4a29c6aa99f1ad5fbbb337ca7e3f1462d722c83bc19f69fafa27a68923d` |
| ViT-B/8 ImageNet21k | SHA `2c4839589adbbf701573dd155e1814f8adb145cd4d235bb86a93269369e111ec` |

Chỉ import receipts, train manifest, normal sample manifest và M states 01/07/14 từ recovery. Không import score/map/test manifest/metric benchmark vào learner. ZIP parent có benchmark cũ, nhưng code chẩn đoán không đọc chúng. MVTec archive chỉ giải nén `train/good/*.png`, không giải nén official test/ground truth. ViT và tiền xử lý dùng đúng producer đã khóa. Task ID chỉ chọn scope/state, không vào predictor.

Capacity sử dụng **chính 8 normal/category đã chọn bằng SHA256** trong recovery, đối chiếu lại với `manifest.online`. Đầu 4 là auxiliary fit, cuối 4 là auxiliary query. Query chỉ tách khỏi auxiliary fit: model M đã thấy những ảnh này trong online stream trước đây. Đây **không phải independent holdout**. Past query không được dùng để fit. Mỗi checkpoint chỉ đọc category có task ID không lớn hơn checkpoint. Các chỉ số task 1,7,14 được định trước theo tiến trình stream; không chọn theo metric anomaly thấp/cao.

Continuation chỉ dùng task-0 prefix: 50 support, 20 meta query, 15 normal validation. Không đọc 15 normal probe. Dùng lại **nguyên cache detached 85 ảnh của W6**, được kiểm archive SHA và `validate_bundle`: tensor SHA, geometry, unit teacher, source/preprocessing/role identity. Không thay feature bằng cache khác và không tái tạo approximation.

MVTec đã development-exposed, kể cả các anomaly metrics cũ. Normal-only ở đây tránh thêm test feedback trực tiếp; không xóa lịch sử exposure hoặc bảo đảm hoàn toàn không có thiên kiến của nhà nghiên cứu. VisA vẫn không được đọc; cần khóa method sau revision trước confirmation.

## 3. Các phép can thiệp trên clone

Đầu vào parent: current weights của visual/decoder, **current CMS weights + accumulators + pending + clock** từ M checkpoint sau task 1/7/14. Không reset CMS ở task boundary. Mỗi arm clone độc lập từ cùng parent. Encoder/visual luôn đóng băng ở phần này và encoded features được tái dùng đúng model đó.

Teacher target là frozen unit feature của ảnh clean. Objective giữ nguyên:

`L = (1/2) mean_masked_tokens sum_channels (D(CMS(z)) - teacher)^2`.

Bốn mask RGB phủ mỗi patch đúng một lần. Auxiliary fit dùng 96 events: tại bước s bắt đầu 0, `view=(parent_cursor+s)%4`, `image=(s+floor(s/4))%4`; mỗi block 16 events phủ đúng tích Descartes 4 ảnh × 4 views. Không để một ảnh chỉ gặp một view.

| Arm | Tham số được thay đổi | Update |
|---|---|---|
| `deployed_cms` | current CMS; decoder/visual cố định | Gọi chính `cms.event(..., meta=False)`; periods/rates/accumulators/clock nguyên bản |
| `cms_adam` | current CMS weights | Adam 1e-4; clip global norm 1; không diễn giải là deployment CMS |
| `decoder_adam` | decoder | Adam cùng LR/budget; CMS current weights giữ nguyên |
| `joint_adam` | current CMS weights và decoder | Adam cùng LR/budget; không thay visual |

Adam dùng betas (.9,.999), eps 1e-8, weight decay 0. Các arm Adam giữ counters parent như metadata, **không giả rằng Adam có commit semantics của v4**; accumulators không được áp vào Adam update. Baseline của mỗi arm là cùng đo ở step 0; đo tiếp 24 và 96. Đo current fit, current auxiliary query, past auxiliary query (task 0 và floor(checkpoint/2), dedup). Mỗi ảnh đủ 4 views. Report mean normal loss, prediction norm, cosine với teacher và L2 parameter change. Không tính AUROC/AP/AUPR từ normals.

### Oracle latent cùng ảnh

Tại parent chưa auxiliary fit, với 4 current query × 4 views × 3 checkpoints = 48 phép fit: lấy output cuối CMS tại masked 196 positions `[1,196,256]`; freeze decoder, cho latent trở thành biến độc lập. Tối ưu tối đa 64 Adam steps LR .01 theo teacher của **chính ảnh đó**. Ghi initial/terminal/best-found loss và norm/độ thay đổi latent. Best-found bao gồm initial candidate.

Đây là kiểm tra reachability lạc quan trong latent space, đã dùng query teacher để fit chính query. Không phải generalization, không phải model khả dụng, và một finite nonconvex search không chứng minh lower bound toàn cục. Chỉ oracle này được fit query teacher; bốn capacity arms không được dùng query gradients.

Tổng budget phần này: **12 capacity fits × 96 events = 1.152 events; 48 latent fits × 64 = 3.072 decoder-only steps**. Canonical preflight thêm 4 capacity events + 1 latent step trên một real normal, và 1 exact outer episode forward/backward; fixtures tests có scope kỹ thuật riêng. Không mở rộng budget sau khi nhìn output.

## 4. Kiểm tra cap tối ưu hóa 400

Chỉ recipe M β=.01, outer LR=1e-4, seed0 đã được W6 chọn. Đọc original `selection/M/M_b0.01_lr0.0001_fit_resume.pt`: step/max_steps400, model metadata, weights, Adam moments, RNG, role schedule và read counters. Không tái khởi tạo optimizer và không chọn β/LR khác.

Tạo schedule 500 bằng cùng generator, xác minh 400 episodes đầu cùng validation support và roles khớp original. Dùng nguyên `episode_loss` exact unroll16, objective mean query tại clock0/16, all active parameters, Adam/clip1 như v4. Sidecar continuation không gọi lại hoặc sửa `fit`/selection frozen vốn bắt buộc max400.

Chỉ thêm **100 outer optimizer steps**. Đo normal validation bằng `short_validation_components` ở 400/425/450/475/500. Ghi toàn bộ loss/gradient norm/read counters/timing. Không chọn best checkpoint, không lập shortlist, không chạy long-horizon gate mới, không xuất deployment initialization. Validation này là dữ liệu development đã được selection dùng trước đây.

Nếu loss còn giảm, kết luận cap cũ có dư địa trong recipe này, **không suy ra** AD performance sẽ tăng hay hội tụ đã đạt. Nếu không giảm, chưa loại trừ objective/capacity/optimizer khác. Không tiếp tục tự tăng tới 1.000 hay 10.000 bước.

## 5. Cách đọc kết quả và điều kiện dừng

Đây là phép so sánh fixed budget, không có threshold tùy tiện để tự freeze một model. Review paired deltas tại cả 3 checkpoint, current fit/query và past; báo cả bất lợi. Không dùng một category hay oracle in-sample để kết luận chung.

- CMS Adam tốt hơn deployed CMS trên current auxiliary query: update dynamics/conditioning là ứng viên nghẽn; chưa chứng minh chỉ cần thay LR hoặc bỏ CMS periods.
- Decoder/joint giúp query rõ hơn CMS-only: fixed readout/representation transfer đáng khảo sát; không tách hoàn toàn capacity khỏi conditioning/số parameter trainable. Thay decoder cũng thay function class.
- Latent oracle tốt hơn nhiều nhưng CMS-only không tiến: fixed decoder có nghiệm tốt hơn trong vùng latent tìm được, nhưng học đường tới vùng đó có thể khó. Oracle không chứng minh CMS đủ sức biểu diễn nó.
- Cả oracle và decoder fits không giúp: có thể do finite budget/optimizer/tín hiệu target, không được kết luận bất khả thi toán học.
- Current query cải thiện nhưng past query xấu: plasticity–retention conflict trong can thiệp này; **không có replay/anchor arm**, nên chưa đo hiệu quả của retention regularization.
- Continuation còn tiến: có optimization headroom trên task đầu, không phải chứng minh sửa được gap 21,56/29,00 điểm với CADIC.

Scientific acceptance là **đủ cell/đúng identity/finite/scope/parent bất biến**, không phải loss phải thắng. Kết thúc `complete_pending_review` rồi người thực hiện kiểm evidence; chỉ sau review mới quyết định một revision. Không auto-run W7.3 hay VisA.

## 6. Codespace, checkpoint và tái lập

Toàn bộ code mới ở `notebook_support/normal_causal/`; tests ở `tests/normal_causal/`. `contract.py` khóa parent/protocol/sampling; `bootstrap.py` scoped import/runtime/data; `features.py` frozen normal cache; `experiments.py` clone interventions/latent/continuation; `runner.py` canonical profile và orchestration; `entry.py` heartbeat/failure ZIP; `build_delivery.py` notebook/source builder; `verify_delivery.py` kiểm bundle/bootstrap và compatibility với checkpoint thật. Không sửa file thuộc 242-file r4 delivery hoặc 119-file frozen W7 map.

`diagnostic_lock.json` khóa canonical JSON identity: source files, protocol, parent manifests, runtime, backbone, original cache và imported hashes. Tuple/list được canonicalize trước lock. Không đưa mount path vào scientific identity. Runtime yêu cầu đúng W6 GPU runtime; CPU hoặc package drift fail sớm, không tự pip install.

Original W6 cache và extracted RGB features ở `/tmp`, không nhét cache nặng vào output. Evidence lưu cache tensor receipts và parent subset đủ để xác minh input states. Active cloned fit checkpoints dùng atomic replace và chứa optimizer/RNG/step/state; chỉ tồn tại khi chưa hoàn tất arm. Khi fit hoàn tất, xóa active clone, giữ trace/result; không xuất weights đã fit làm initialization. Parent M checkpoints và original W6 checkpoint vẫn được lưu như evidence đầu vào.

Session soft budget mặc định 3 giờ, kiểm trước mỗi atomic step, chừa 180 giây export; **không phải ETA hoặc bảo đảm platform không kill**. Một exact outer step hoặc validation có thể vượt soft deadline một chút. Đã complete rồi không fit lại; pause resume latest output với cùng source/parents. Lỗi numeric/identity là failure, không âm thầm rescue checkpoint tốt hơn. Runtime, source hoặc protocol khác phải dừng để review.

## 7. Hướng dẫn Kaggle từng bước

1. Upload `ultils/normal_causal/hope_cad_normal_causal_source_v1.zip` thành Kaggle Dataset mới. Giữ tên ZIP hoặc để Kaggle giải nén đều được; schema và hashes mới là định danh.
2. Import `notebooks/hope_cad_normal_causal_kaggle.ipynb`. Attach source mới, completed recovery **main**, completed W6 v4, MVTec, ViT. Không cần completed W7.2 riêng vì recovery main đã chứa ba M states cần thiết. Không attach recovery preflight hoặc source r4 làm source chính.
3. Settings: GPU T4 x2. Giữ `PHASE='run'`, `SESSION_HOURS=3.0`; có thể điều chỉnh soft session budget .1–11 giờ, **không sửa budget fit trong protocol**. Không cần chạy preflight riêng: nó tích hợp đầu notebook. Nếu muốn profile trước, `PHASE='preflight'` rồi attach output PASS vào lần `run` tiếp.
4. Save Version → Save & Run All. Cell 1 báo source verified; cell 2 kiểm runtime/parents/cache, chạy tests, profile canonical rồi continuation và capacity. Worker heartbeat mỗi 30 giây khi chưa có log bước mới. Thời gian chờ máy Kaggle trước cell đầu không nằm trong profile compute.
5. Nếu `paused_at_atomic_boundary`: giữ output ZIP/latest notebook output; attach nó cùng 5 inputs rồi chạy tiếp notebook giữ nguyên source và `PHASE='run'`. Tháo các diagnostic outputs cũ hơn. Nếu khác source/failed identity, giữ ZIP và gửi review trước, không cố bypass guard.
6. Khi `complete_pending_review`: tải `hope_cad_normal_causal_evidence.zip` và log về `docs/normal_causal_kaggle/main/` trong `hope_cad_v5`, báo để kiểm. ZIP phải có summary, 12 capacity results, 48 oracle results, continuation trace tới500, locks/receipts/logs và handoff. Chạy hết cell chưa đủ nếu status là pause/fail.

**Kiểm local:** 19 tests mới + 30 focused regressions CMS/v4/normal-forensics = 49 PASS, không failure/error/skip; runtime local RTX3050Ti khác Kaggle. Đã xác minh cả ba parent checkpoints (clock403/2022/3599), original cache85 ảnh, Adam39 groups ở step400 và schedule400-prefix. Bootstrap đã kiểm trên source ZIP và kiểu Kaggle tự giải nén. Đây là engineering verification, không là kết quả chẩn đoán real GPU.

**Trạng thái bàn giao:** implementation và local tests được ghi ở `ultils/normal_causal/local_verification.json`; real Kaggle experiment pending. Không gắn hiệu quả hay chọn revision trước khi có artifacts.
