# Cơ sở thiết kế và giới hạn của H32

## 1. Vấn đề được quan sát, chưa phải nguyên nhân đã chứng minh

Matched seed0 v5.1 có pixel AP và pixel retention tốt hơn CADIC trong khi final image AUROC gần ngang; xem [kết quả, protocol và provenance](../results.md). Trên trace được review: **388/401 meta-updates** dùng tám records cùng category; **3.199/3.211 delayed labels** đến từ ảnh kế tiếp cùng category; **1.935/3.179** quyết định post-warmup khác geometry prior trên trajectory đã đi qua. Các thống kê này được kế thừa từ audit của trace, chưa được tái tính ở lần công bố tài liệu này.

Implementation hiện hành ở [meta_loss](../../src/hope_cad/models/learner.py) dùng support r0–r3 và query r4–r7 của cùng block tám completed records. [Session](../../src/hope_cad/models/session.py) resolve pending từ normal kế tiếp trước khi dựng proposal mới. Đây là đường code cần kiểm trực tiếp khi phản biện; episode counts không tự chứng minh một cơ chế nhân quả.

Tầng chậm vẫn tích lũy quá khứ trong weights/Adam và label đã chứa anchors. Vì thế không được nói v5.1 hoàn toàn không có retention. Giả thuyết hẹp hơn: sau khi fast thích nghi trên recent support, slow objective có thể cần đo utility trên cả completed contexts cũ. H32 thay hai trong bốn queries bằng records lịch sử; giữ support steps/topology/scoring để kiểm giả thuyết này.

Không thể quy pixel gain hiện tại cho exact meta-gradient hoặc nhiều timescale chỉ từ các counts/norms. D0 là observational fixed-pair analysis, còn H32−R_EXACT, FO, DIRECT, FAST_ONLY, BIAS và RANDOM là các contrasts được định nghĩa để kiểm một phần các giải thích khác.

## 2. Các hướng đã xem xét và quyết định

Bảng này ghi các quyết định thiết kế cần thiết cho người đọc độc lập. Các phản biện về bias/clock, stale utility, query eligibility, ownership, exact/FO/direct và tính công bằng đã được đưa thành controls/contracts cụ thể. Không cần đọc chatlog để thực hiện. Các đường dẫn paper dưới đây là nguồn học thuật, không thay evidence thực nghiệm của adaptation này.

| Point nghiên cứu | Quyết định trong vòng này |
|---|---|
| Slow queries cần thấy ngữ cảnh cũ | **Đặc tả một candidate H32**: 32 records, 4 support, 2 recent + 2 history; không gọi đây là root cause đã chứng minh |
| Gain có thể chỉ do hạn chế ghi/bias | **D0 trước**, BIAS và pilot-rate random-veto là controls đã định nghĩa, không dùng rate theo task nhìn trước |
| Exact meta so với FO/replay | **2×2 context×gradient**, thêm direct replay riêng; FO không bị gọi nhầm là ER |
| Anchors tám patches có thể thiếu normal modes hiếm | **Giữ 32×8 trong H32**, báo giới hạn; không đồng thời sửa sampler để tránh confound |
| Candidate/victim context và action quá thô | **Giữ 16 stats và whole-proposal action**; nghiên cứu riêng khi có evidence, không ghép vào H32 |
| Learned learning-rate/retention/momentum | **Hoãn**, giữ normalized fast rule/Adam; cần ownership và stability contract riêng |
| Thêm CMS levels/width/unroll | **Không thêm**; chưa có evidence capacity là nút nghẽn, không mở grid |
| ContCore | **Related work/alternative proposer**, không thay CADIC trong H32 hoặc mượn coverage theorem để bảo đảm veto thắng |
| Native normalization khác current recipe | **Giữ recipe matched hiện hành** trong toàn bundle; disclose mismatch, không claim paper parity |
| EMA/synchronous | **Chưa triển khai**: theta/phi khác topology; claim độc lập về tần suất chưa được mở |
| Retrieval interference/WNR | Báo shared-bank risk; saved bank thiếu per-row provenance nên **không hứa exact WNR offline** |
| ReplayCAD | Baseline bổ sung theo scope luận văn được duyệt, **chưa có matched artifacts**, không mặc định dự án đã hoàn thành đủ external baselines |
| VisA | Suite xác nhận riêng sau freeze; **không dùng để chọn H32 hoặc sửa pipeline** |

Không áp dụng mọi ý kiến bằng cách cộng thêm component. Những hướng hoãn được giữ trong danh mục nghiên cứu; chúng có điều kiện mở nghiên cứu mới, không tự bật nếu H32 thua.

## 3. Cơ sở lý thuyết và phạm vi đóng góp

NL mô hình hóa các quá trình học có context, objective và clock riêng. V5.1 có fast adaptation theo label, slow meta-gradient mỗi 8 labels và visual self-modification trong ảnh. H32 chỉ mở rộng **context của slow objective**; không tự trở thành một bản tái lập đầy đủ HOPE/Titans. [NL, §§3,7–8](https://arxiv.org/html/2512.24695v1).

ATLAS là động cơ cho việc học trên cửa sổ context thay vì chỉ một sự kiện; đó không phải chứng minh H32 utility replay thắng CADIC. MIRAS nhắc phải thiết kế memory/objective/retention/optimizer đồng bộ. [ATLAS](https://arxiv.org/html/2505.23735v1), [MIRAS](https://arxiv.org/html/2504.13173v1).

Titans và VisionHOPE cung cấp nguyên tắc update/self-modification; các bounds của visual scan không tự là bounds cho whole-bank retention hoặc nonlinear fast learner. ContCore phân tích sampling/compression theo điều kiện của chính proposer đó; không chuyển theorem sang hard admission policy. [Titans](https://arxiv.org/html/2501.00663v1), [VisionHOPE](https://arxiv.org/html/2609.33325v1), [ContCore](https://arxiv.org/html/2608.15277v1).

Lập luận có thể bảo vệ: khi population predictor đủ rộng và c là positive scale đã biết tại quyết định, MSE optimum có cùng sign với conditional expected one-step advantage. Finite capacity, compressed context, policy-dependent bank distribution và actual optimization không bảo đảm đạt optimum. Historical y chỉ mô tả original bank pair; giữ khả năng dự đoán y cũ có thể không cải thiện current bank hoặc AD. Không có claim “tận dụng tối đa NL”, global optimality, không còn bug, hay dự báo tăng bao nhiêu AP.

Kết quả giữ nguyên: seed 0 CADIC I-AUROC 86,9983% / P-AP 44,9012%; NESTED 86,9561% / 47,4331%; pixel forgetting 8,9104 → 6,8252 điểm phần trăm. Chưa attribution riêng exact/CMS/timescale, chưa multi-seed/VisA superiority. MVTec đã development-exposed.


## 4. Trade-offs có thể bác bỏ

Historical label là utility của bank pair gốc, không phải reward của bank hiện tại. Reservoir 32 có thể stale, aliasing hoặc không đại diện mode hiếm; không dùng category labels để sửa sampler. Equal query count không chứng minh equal FLOPs, latency hoặc peak memory; phải đo history, debug copies, transaction và I/O. Một controller dự đoán label cũ tốt hơn vẫn có thể không cải thiện anomaly ranking.

H32 chỉ đáng được chọn sau các gates [study](study.md), không nhờ tên Nested Learning hoặc paper. Không hứa tăng AP, không tuyên bố đã tận dụng tối đa NL. Không đạt safety/spend-screen thì giữ baseline, công bố kết quả âm và dừng candidate dưới ID này; không mở tuning loop trên MVTec rồi xem VisA là phép chọn tiếp.
