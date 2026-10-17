# Nested Learning for CAD — Review completed normal-only causal diagnostic v1

Ngày review: 17/10/2026. Worktree: `hope_cad_v5`. **Nghiệm thu trong phạm vi gói chẩn đoán normal-only đã khóa. Không cần chạy lại gói này.** Chưa chọn method revision và chưa có anomaly benchmark mới.

## 1. Nguồn, kỹ thuật và phạm vi kiểm

Input gốc giữ tại `docs/recovery_stage1_kaggle/hope_cad_normal_causal_evidence.zip`, SHA256 `46124f3202776138fd39a2cffe4fe0fc2d043fe650bfe53a1f69f377fad0f109`, kích thước33.295.548 bytes,119 payload files. Manifest SHA `ad477fbe44d6df4b6caa152d7014edaf30c0155261911c71726fd37e448b1f94`. Notebook handoff `complete_pending_review`, worker exit0; thời gian tính từ bootstrap tới export1.723,719 giây, khoảng28 phút44 giây. Con số này không bao gồm thời gian Kaggle chờ cấp máy.

Receipt và script tái lập: `results/normal_causal/main_review.json`, `review_evidence_v1.py`. Đã kiểm119 payload hashes và inventory; source255 files và live bytes khớp delivery; parent manifests/hash entries, runtime W6, protocol/case/arm/oracle locks; regenerate độc lập400/500 episode schedules; sample rights, read counters, clocks, summary consistency và arithmetic aggregates.1.044 assertions là các kiểm điều kiện/integrity, **không phải1.044 experiments hoặc statistical tests**.

Đủ12 auxiliary capacity fits ×96 events,48 same-image latent fits ×64 steps, một selected-M continuation100 steps. Canonical T4 preflight PASS,19 engineering tests PASS; one-episode exact-meta forward/backward khoảng12,09 giây. Preflight GPU peak allocated khoảng1,60GiB chỉ là phạm vi đo preflight, không phải peak của toàn run. Không phát hiện thiếu cell hoặc lỗi khiến phải rerun.

Source digest `53c3caafbdf3a9294cdcb717de4f9a0adf4546869dd08e3dc63db92aca80dc90`, diagnostic lock `3dea73455cb19b313428898ef7c0d2615d35218578da6ea2d9c0c8f553ca2965`. Ba parent M states giữ hash và clock403/2022/3599; original W6 cache đúng85 ảnh, original fit/optimizer step400. Chỉ ba frozen parent `.pt` được xuất; không có active/fitted learned weights để lén triển khai hoặc thay v4. Frozen specs/model/source r4 và các benchmark cũ giữ nguyên.

**Giới hạn verification:** đã tính lại aggregates/deltas từ JSON và đối chiếu code/provenance; không tính lại prediction loss từ final fitted weights vì các clone đó đã bị discard đúng protocol. Feature tensor SHA receipts có, raw/encoded caches nằm/tmp và không được export. Không mô tả review này như independent forward replay của12 fitted models.

## 2. Phát hiện A — Có dư địa normal fitting rõ, update policy đang hạn chế tốc độ thích nghi

Bốn arms cùng parent, cùng frozen encoded inputs, cùng4 auxiliary-fit images,4 mask views và96 events. Query gồm4 ảnh khác auxiliary-fit nhưng **đã từng nằm trong original online stream**. Không phải independent holdout. Normal loss nhỏ hơn là tốt hơn trên phép đo này; không phải AUROC/AP/AUPR.

| Category / checkpoint | Parent query loss | Deployed CMS | CMS Adam | Decoder Adam | Joint Adam |
|---|---:|---:|---:|---:|---:|
| cable / 1 | 0.777488 | 0.758239 | 0.639165 | 0.527287 | 0.463669 |
| metal_nut / 7 | 0.464599 | 0.455644 | 0.389479 | 0.289495 | 0.259027 |
| zipper / 14 | 0.504643 | 0.494193 | 0.420427 | 0.291458 | 0.261752 |

Giảm tương đối so với parent query loss:

- Deployed CMS: cable2,48%, metal_nut1,93%, zipper2,07%.
- Chỉ CMS Adam:17,79% /16,17% /16,69%.
- Chỉ decoder Adam:32,18% /37,69% /42,24%.
- Joint CMS +decoder Adam:40,36% /44,25% /48,13%.

Evidence: ZIP `capacity/task_01|07|14/{arm}/result.json::trace`, owner changes và clock fields; implementation `notebook_support/normal_causal/experiments.py::capacity_arm`. Deployed arm gọi nguyên `cms.event(meta=False)` với pending3/6/7 ban đầu; không reset/flush. Clock tăng96; các Adam interventions giữ clock metadata, không giả vờ có semantics deployment.

**Kết luận có cơ sở:** trong ba probe này, policy CMS đang triển khai khai thác khả năng khớp normal chậm hơn auxiliary Adam. Cho readout thích nghi bổ sung dư địa lớn; joint intervention tốt nhất về current query ở cả ba mốc. Đây là bằng chứng thực nghiệm cho ràng buộc optimization/readout transfer trong instantiation v4, thay vì suy diễn mọi điểm yếu đều là bản chất CAD hoặc toàn bộ Nested Learning.

**Không suy ra:** “Adam luôn tốt hơn”, “do period8”, hoặc “decoder là nguyên nhân duy nhất của gap CADIC”. Optimizer/preconditioning/step magnitude, số parameter trainable và function class khác nhau. CMS parameter change L2 của Adam khoảng1,68–1,72 trong khi deployed khoảng0,10–0,15; fixed budget không equalize effective update distance. Chưa có multi-seed/optimizer-matched/long-stream study để tách mọi biến. Joint update không tự là một kiến trúc Nested Learning mới đã hoàn thiện.

Cosine với teacher cũng tăng: joint cable0,206→0,417; metal_nut0,466→0,690; zipper0,388→0,687. Vì vậy không nên giải thích toàn bộ giảm loss chỉ bằng prediction norm co nhỏ. Mặt khác norm cũng đổi, và normal loss vẫn chưa chứng minh abnormal/normal separation tốt hơn.

## 3. Phát hiện B — Plasticity tốt hơn đi kèm interference; mở decoder đơn giản chưa đủ

Past query dùng task0 và floor(checkpoint/2), dedup; không được fit. Tỷ lệ thay đổi **past normal loss** so với parent, số dương là xấu hơn:

| Category | Deployed CMS | CMS Adam | Decoder Adam | Joint Adam |
|---|---:|---:|---:|---:|
| cable | -2.21% | -14.86% | +17.17% | +8.36% |
| metal_nut | +1.20% | +2.84% | +9.08% | +8.81% |
| zipper | -0.10% | +0.88% | +13.95% | +14.07% |

Evidence: cùng `result.json::trace[0|2].groups.past_aux_query`, case lock rows. Decoder-only làm past loss tăng ở cả ba mốc; joint tăng8,36–14,07%. Cải thiện current query có thật trong auxiliary task, nhưng chưa giải quyết retention. CMS-only cũng tăng nhẹ ở hai mốc sau; deployed gần giữ nguyên nhưng học current chậm.

Không gọi những tỷ lệ này là forgetting/BWT benchmark mới: past normals được tái dùng, mỗi case chỉ4 hoặc8 past images, hai category lịch sử được định trước, không phải toàn stream. Không có replay/anchor intervention trong gói này nên chưa biết retention regularizer sẽ hiệu quả đến đâu. Đủ căn cứ bác bỏ bước đi “chỉ unfreeze decoder rồi coi đã fix”.

## 4. Phát hiện C — Decoder cố định vẫn có latent solutions tốt hơn, nhưng finite oracle không là global bound

Oracle fit latent tại final CMS với decoder frozen, dùng teacher của chính query image. Mean initial→best-found normal loss:

| Checkpoint | Initial | Best found | Giảm tương đối |
|---|---:|---:|---:|
|cable /1|0,777488|0,399077|48,67%|
|metal_nut /7|0,464599|0,250000|46,19%|
|zipper /14|0,504643|0,291821|42,17%|

Evidence:48 `capacity/task_*/latent/*/result.json`; implementation `experiments.py::latent_oracle`. Initial mean khớp parent query baseline trong tolerance1e−6; best-found gồm initial candidate, finite và không cao hơn terminal.

Kết quả này **không hỗ trợ phát biểu tuyệt đối rằng decoder frozen không thể tạo output normal tốt hơn**: trong latent space đã tìm được nghiệm tốt hơn. Có thể có khó khăn học đường từ encoded input qua CMS tới những latent đó. Đồng thời decoder adaptation vẫn hữu ích; đó là thay function/conditioning chứ không mâu thuẫn với oracle. Oracle được fit trực tiếp theo query target, nên không deployable, không generalization và không chứng nhận lower bound toàn cục. Joint tốt hơn finite oracle ở một case cũng không vi phạm lý thuyết, vì hai phép search có parameterization/budget khác nhau và oracle không tối ưu toàn cục.

## 5. Phát hiện D — Cap400 còn optimization headroom trên task đầu

Chỉ selected M β=.01/LR1e−4, không grid mới. Giữ original Adam/RNG/85-image features và400-episode prefix; thêm100 outer steps. Normal validation giữ nguyên warm objective mean(query loss tại clock0/16):

| Step | Normal validation loss |
|---|---:|
|400|0,144900376|
|425|0,138138314|
|450|0,132748480|
|475|0,133003783|
|500|0,131809950|

400→500 giảm **9,0341%** tương đối. Step400 tái hiện loss0/loss16/mean của W6 cũ trong1e−7; không phải bắt đầu bằng state khác. Trace có bump tại475 nên không gọi monotonic convergence. Read delta support1680/query100/validation600/probe0 khớp100×16support,100metaqueries và5 lần short validation có16support +15images×4views×2clocks. Không dùng probe hoặc anomaly test để chọn checkpoint mới.

Evidence: `continuation/result.json::trace/training/reads`, original W6 `M_b0.01_lr0.0001_fit_resume.pt`, `experiments.py::continuation`. Có100 rows401..500, không có selection/best export.

**Kết luận:** cap400 chưa làm hết dư địa tối ưu hóa normal task đầu trong recipe này. **Không suy ra:** tăngsteps sẽ chữa transfer/retention, loss500 là hội tụ, hoặc9% loss là9 điểm AUROC. Không mở rộng tiếp budget một cách hậu nghiệm. Thí nghiệm này kết thúc đúng500 theo protocol.

## 6. Trạng thái khoa học và data exposure

- Implemented/tested và completed/accepted trong phạm vi bounded normal development diagnostic.
- Evidence ủng hộ optimization/readout transfer headroom, và interference khi tăng plasticity; không còn chỉ là giả thuyết không có intervention.
- Causal share của từng module, objective mismatch masked predictability vs normal-support membership, lợi thế riêng của nhiều timescale và AD gain vẫn mở.
- Chỉ source/runtime/input hashes không tự bảo đảm không có mọi dạng cognitive bias. Gói được thiết kế sau khi đã biết MVTec W7/recovery results; phải ghi development-informed. Actual fitting/query accesses trong gói tuân thủ quyền normals cố định, không anomaly-test/VisA mới.
-4 auxiliary-query images/category, một parent seed,3 checkpoints; không extrapolate sang15category/3seeds/VisA. Latent fits dùng queryteacher trực tiếp và không độc lập. Không có phép kiểm thống kê chứng minh generality từ những averages này.
- Giữ `reportable=false` của artifacts. Có thể báo normal diagnostic results với đúng phạm vi/limitations, không chuyển chúng thành official anomaly benchmark hoặc superiority.

## 7. Quyết định tiếp theo được khuyến nghị

**Kết thúc vòng chẩn đoán này. Không cần rerun gói, W6 hoặc full W7.3 chỉ để tìm lại những câu trả lời vừa có.** Chuyển sang chốt thiết kế một revision nghiên cứu rồi triển khai, vẫn là đề tài Nested Learning for CAD.

Revision phải xử lý đồng thời **acquisition của category mới** và **retention**, đồng thời có anomaly score gắn với normal support hợp lý. Không lấy `joint_adam` diagnostic làm production patch: nó chưa là nested-learning system đầy đủ, đã có past interference và chưa có AD evidence. Cũng không chọn500 thay400 rồi claim cùng frozen v4.

Hướng normal-support memory giữ raw ViT geometry vẫn là một proposal đáng so sánh; lần này có thêm bằng chứng cho hướng cải thiện adaptive predictor/readout. Gói này không xác định hướng nào sẽ thắng CADIC, nên không được tự đóng dấu support memory là tốt nhất hoặc tự thêm hàng loạt module. Bước kế tiếp là một architecture decision có cơ sở cho **một** hướng, gồm: vai trò thực của CMS/HOPE trên inference/update, lý thuyết normal support và retention, data rights/budget, matched CADIC baseline, endpoint/ablations/stop criteria và VisA freeze. Sau đó mới code revision.

Không có yêu cầu người dùng chạy thêm Kaggle ngay trong bước chốt spec. Performance gain chỉ được kết luận từ benchmark của revision đã khóa, không từ tỷ lệ giảm normal loss trong report này.
