# Cập nhật Nested Learning for Continual Anomaly Detection

**Ngày cập nhật: 22/10/2026**  
**Code đang dùng:** branch/worktree `hope_cad_v7`  
**Phương pháp đã triển khai:** `hope_cad_v5_1_causal_scaled_utility`

Tài liệu này tóm tắt phần phát triển của nhánh HOPE-CAD, kết quả đã đo và công việc tiếp theo để trao đổi trong nhóm.

**Mốc đối chiếu code:** `dikhang_space@794d7e498ea093f99ae861e9f43f4ab59d887772` và baseline `hope_cad_v7@89df9a21658b265731f98a7c2014afb1876f1924`. Với nhánh dikhang_space, phần so sánh CADIC dưới đây lần theo script `scripts/benchmarks/run_cadic_paperfaithful.sh` → config/protocol → adapter/engine thực sự được gọi. Các experiments ECAD/NPIM/CSG là những đường chạy riêng, không gộp thành một “protocol của Khang”. Cấu hình mặc định và code không chứng minh mọi run thực tế đã dùng đúng các giá trị đó; overrides phải đối chiếu từ từng run.

**Amendment đã chốt trong branch:** [spec/plan](docs/pipeline_alignment_spec.md) và [contract máy đọc](docs/pipeline_alignment_contract.json) là nguồn prospective pipeline. Full-train3.629 đã được chọn; preprocessing **giữ recipe hiện hành** để matched với CADIC Khang và cô lập history effect. Native-metadata mismatch vẫn công bố; không gọi nó đã được sửa. Runtime/resources baseline cũ giữ nguyên; successor full-train/H32 đã implement và test local, chưa remote acceptance. Quyết định docs này không thay kết quả benchmark đã báo.

## Tài liệu đi cùng branch

Người nhận chỉ cần branch này để đọc đầy đủ thiết kế: [bản đồ docs](docs/README.md), [method đang chạy](docs/method.md), [benchmark và provenance](docs/results.md), cùng [spec/plan nghiên cứu đầy đủ](docs/research/README.md). H32 có [thuật toán/gradient/RNG/state](docs/research/candidate.md), [D0](docs/research/attribution.md), [controls/gates/budget](docs/research/study.md), [contract tổng hợp](docs/research/contract.json) và [kế hoạch implementation](docs/research/implementation_plan.md). Không có nguồn bắt buộc nào là một đường dẫn context riêng ngoài branch.

Hai file `pipeline_alignment_spec.md` và `pipeline_alignment_contract.json` chốt thay đổi dữ liệu/recipe/identity cho cohort tiếp theo; chúng chưa được CLI hiện tại thực thi. QUICK_UPDATE là báo cáo trao đổi, không là nơi duy nhất cất thuật toán. Spec/contract chuyên biệt là nguồn triển khai; các phụ lục bên dưới giữ vai trò giải thích. Raw artifacts chưa public trong Git; cần xin riêng nếu tự tính lại metrics.

## Implementation mới trong branch

D0 offline, successor full-train3629 và H32/controls đã có code cùng CLI riêng `research`, ledger ngân sách, review gates, atomic resume và aggregation. Frozen baseline contracts/evidence giữ nguyên; chưa chạy GPU thuê, pilot hay AD benchmark mới. [Implementation, kiểm chứng và lệnh SSH](docs/research/implementation.md).

## Overview

**Tình hình nhanh để Khang nắm:**

1. **Đã có model chạy thật:** NESTED học utility để chấp nhận/từ chối proposal cập nhật của CADIC, giữ frozen ViT, raw-feature bank và Eq9. Không phải chỉ thêm HOPE/CMS rồi để output không ảnh hưởng prediction.
2. **Đã có đối chứng nội bộ:** full stream 15 categories của MVTec, seed 0, K=10k; cả CADIC và NESTED có final metrics và đủ ma trận 120 ô. NESTED đạt **86,96% image AUROC / 47,43% pixel AP**, CADIC **87,00% / 44,90%**. Pixel forgetting giảm **8,91 → 6,83 điểm %**; image-level chưa có lợi thế rõ.
3. **Protocol khác route CADIC chuẩn ở `dikhang_space`:** bên này loại Bottle `070–099`, còn **3.599** normal images thay vì **3.629**. Cả CADIC đối chứng và NESTED bên này cùng bị loại 30 ảnh, nên bảng nội bộ matched; chưa dùng bảng đó để kết luận hơn run riêng của Khang. **Cohort mới sẽ trở lại full-train 3.629 normals theo route chuẩn của Khang**, áp dụng đồng thời cho mọi comparator; kết quả cũ giữ nguyên.
4. **Source đã dọn thành v7:** một package `src/hope_cad`, CLI cho train/evaluate/matrix/resume/export và workflow SSH. Method vẫn v5.1; cleanup không được tính thành gain benchmark. V7 chưa có full benchmark mới trên RTX 5090.
5. **Điểm cần thống nhất trong nhóm:** exact train IDs, K, checkpoint tensors, evaluator và trạng thái CADIC/ReplayCAD/VisA; ContCore là baseline bổ sung hay nhánh nghiên cứu riêng.
6. **Việc sắp làm:** phân tích utility bằng evidence có sẵn, rồi một candidate dùng historical queries cho slow learning. **Đã implement candidate/controls và runner SSH, test local đạt; chưa có remote pilot/AD results mới.** Giữ baseline hiện tại để đối chiếu/quay lại.

Phần dưới tự đủ để đọc cùng branch v7. Không cần kho knowledge trên local. Các giá trị benchmark là bản tổng hợp từ evidence đã được review; raw artifacts không được commit nên người nhận cần chúng nếu muốn tự tính lại số liệu từ đầu.

### Bản đồ đọc cho AI hoặc người review

- Đọc mục 1 để biết **đã đo được gì**; mục 2 để biết **đã phát triển gì so với nền Git**.
- Đọc mục 3 để hiểu **khác biệt protocol/route** và điều kiện so với kết quả phía Khang.
- Mục 4 ghi giới hạn của kết quả; mục 5 ghi công việc mới, chưa được triển khai.
- Mục 6 ghi điểm phối hợp; mục 7 là code map nằm trong branch.
- Phụ lục A–D trình bày thuật toán/ownership, benchmark chi tiết, candidate/controls và khả năng tái lập. Không suy status chỉ từ tên file hoặc trường pending trong immutable contract lịch sử.

## 1. Kết quả chính hiện tại

Đã hoàn tất học trên stream normal của MVTec AD và đánh giá cả **CADIC đối chứng cùng điều kiện** lẫn **NESTED**, seed 0. Có kết quả cuối stream và ma trận đánh giá các category đã học tại từng checkpoint.

**Kết quả quan sát được:** NESTED có image AUROC cuối gần bằng CADIC, pixel AP cao hơn và pixel forgetting thấp hơn. Chi phí cập nhật khi học stream cao hơn CADIC. Chưa đủ bằng chứng để kết luận ưu thế tổng quát, lợi ích riêng của exact meta-gradient hoặc khả năng giữ kết quả này trên VisA.

| Chỉ số                                       |  Matched CADIC |             NESTED |     NESTED − CADIC |
| -------------------------------------------- | -------------: | -----------------: | -----------------: |
| Final image AUROC                            |       86,9983% |           86,9561% |     −0,0422 điểm % |
| Final pixel AP                               |       44,9012% |       **47,4331%** | **+2,5319 điểm %** |
| Image AUROC forgetting — thấp hơn tốt hơn    |  6,8349 điểm % |      6,9384 điểm % |     +0,1035 điểm % |
| Pixel AP forgetting — thấp hơn tốt hơn       |  8,9104 điểm % |  **6,8252 điểm %** | **−2,0852 điểm %** |
| Pixel AP backward transfer — cao hơn tốt hơn | −8,9047 điểm % | **−6,4952 điểm %** | **+2,4096 điểm %** |

Pixel forgetting giảm khoảng **23,4% tương đối** trong lượt đo này. Backward transfer vẫn âm ở cả hai phương pháp: NESTED chưa loại bỏ hiện tượng quên.

Các con số trên là so sánh nội bộ giữa hai phương pháp dùng cùng pipeline. **Không phải so sánh trực tiếp với kết quả một lượt chạy khác của Khang hoặc với số trong paper CADIC.**

## 2. Đã thay đổi gì so với nền code chung?

### 2.0. Các milestone từ nền chung đến source v7 hiện tại

Hai nhánh không có quan hệ “v7 là HEAD mới nhất của Khang cộng thêm một file”. **Tổ tiên chung là `b985944`** (`Sync to continue`, 15/10). Sau mốc đó, `dikhang_space` có thêm experiments riêng; nhánh này tích hợp source HOPE rồi phát triển/đóng gói riêng. Git có một số commit merge mang lịch sử trước ngày tổ tiên chung; không lấy ngày trong một dòng log làm thứ tự khoa học của mọi revision.

| Milestone                                     | Thành quả giữ lại trong bản hiện hành                                                                                   | Phạm vi                                                                                                |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Xây primitives Nested Learning cho thị giác   | Visual memory/self-modification, fast/slow CMS, functional updates và numerical oracles                                 | Các primitives còn dùng được giữ trong `models/primitives`; không bê toàn bộ prototype vào đường chạy  |
| Tích hợp nền CADIC và quyền dữ liệu           | Frozen ViT-B/8, raw patch coreset, per-image update, full Eq9, protocol/manifests/evaluator                             | Đây là nền chung để learned controller có comparator cùng geometry                                     |
| Tích hợp vào workspace riêng từ nền `b985944` | Giữ baseline scaffolds cần thiết, tách runtime/evidence và loại experiments không thuộc dependency closure              | Commit tổ chức nền `84cccba`; không phải bằng chứng đã có mọi sửa khoa học sau đó tại đúng commit này  |
| Chốt phương pháp utility v5.1                 | Causal scale, geometry prior + learned residual, persistent fast readout/CMS và exact four-support meta-learning        | Đây là method của các benchmark trong tài liệu                                                         |
| Kiểm cơ chế online trước benchmark            | Normal-only pilot trên bốn controls, causal labels, gradient/state/resume và cơ chế thay bank thật                      | Pilot đã review; không dùng pilot metrics thay anomaly benchmark                                       |
| Tối ưu execution không thay objective         | Pair-cache/dirty tiles, NN/write path, state-preserving continuation; checkpoints và raw evidence có provenance         | Đã có local/remote engineering checks trong phạm vi cụ thể; không gọi đây là model accuracy gain       |
| Hoàn tất matched MVTec seed 0                 | Cả CADIC/NESTED học toàn online stream, có 15 final category metrics và 120-cell task matrix mỗi method                 | Gain pixel AP/retention đã quan sát; chưa multi-seed/VisA                                              |
| Chuẩn bị source portable cho máy thuê         | Bỏ phụ thuộc notebook launcher/sibling source, CLI và remote setup/preflight; tách data/checkpoints/artifacts khỏi repo | Có thể gửi branch để chuẩn bị fresh execution; actual GPU acceptance vẫn phải chạy                     |
| Hợp nhất package và cố định baseline v7       | `src/hope_cad`, functional tests, docs hiện hành, Git publication đủ data modules                                       | `26f2169` hợp nhất; `3731503` sửa tracking/newline. Cùng method v5.1, không tăng benchmark do refactor |
| Chốt hướng nghiên cứu tiếp theo               | Một candidate historical-query có sampler/state/controls/gates hữu hạn                                                  | Chỉ đặc tả; không tự đưa vào code hiện hành                                                            |

**Không dùng commit message thay source identity của experiment.** Một số lần phát triển trước được chạy bằng source ZIP từ working tree có thay đổi chưa commit; benchmark được bind bằng file hashes/runtime/manifests. Vì vậy commit trong bảng là mốc tổ chức dễ tra, không giả định mọi kết quả tương ứng chính xác với HEAD của một commit có cùng tên phiên bản.

#### Git diff đang thể hiện điều gì?

Đối chiếu tree `dikhang_space@794d7e4` với `hope_cad_v7@89df9a2` có **520 paths thay đổi, khoảng 8.462 dòng thêm và 75.703 dòng xóa** theo `git diff --shortstat`. Phần lớn là đổi tổ chức và loại code/experiments ngoài runtime hiện hành, không phải xóa thuật toán baseline rồi tạo gain. Rename detection cũng ảnh hưởng cách Git đếm thêm/xóa; số dòng không là thước đo contribution hoặc chất lượng model.

| Ở nhánh nền/Khang                                                          | Trong v7                                                                                | Ý nghĩa                                                                                                  |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Top-level `models/`, `dataset/`, `training/`, nhiều scripts theo phiên bản | Một `src/hope_cad/` chia models/data/baselines/benchmark/runtime                        | Imports nhất quán, cài package và giảm khả năng chạy nhầm implementation                                 |
| Nhiều experiment families trong `exps/`                                    | Không có các experiment families đó trên đường runtime hiện hành                        | NPIM/CSG/ECAD/Task P không được tự coi là phần model NESTED hoặc baseline đã nghiệm thu                  |
| Adapter CADIC chuẩn và module override trong shell script                  | Main matched driver chọn job CADIC/NESTED rõ; compatible/ReplayCAD ở integrations riêng | Tách comparator chính khỏi route tham khảo                                                               |
| `conf/` rải method/protocol và các cấu hình lịch sử                        | Normative JSON/YAML ở `src/hope_cad/resources/`                                         | Source inventory/locks xác định đúng recipe; không chọn config bằng tên gần giống                        |
| Root dependency/pytest scaffolds                                           | `pyproject.toml`, local Pixi lock và remote CUDA pins riêng                             | Không thay runtime bằng việc cài requirements của experiment khác                                        |
| Các notebook/scripts bàn giao riêng cho từng lần chạy                      | CLI thống nhất + `scripts/remote/`                                                      | Chạy qua SSH, tách train/evaluation, resume/export; không tự thuê GPU                                    |
| Một số helper/dataset/geometry bị lặp theo phiên bản                       | Common geometry/validation, static ablation helper và primitives có trách nhiệm rõ      | Dọn dependency; giữ serialized IDs/schema nếu thuộc contract                                             |
| Artifacts/checkpoints/context gắn với workspace cá nhân                    | Không là dependency của source public                                                   | Clone có thể chuẩn bị chạy với dataset/checkpoint của người nhận; tài liệu này mang bản tổng hợp kết quả |

Không merge toàn bộ các commit sau `b985944` của Khang vào v7: các route nghiên cứu mới có protocol/namespace/locks khác. Nhánh Khang vẫn là reference độc lập; một helper/ý tưởng đáng dùng phải port có chủ đích và kiểm lại, không nhập cả cây experiments vào baseline.

### 2.1. Thêm learned admission policy trên CADIC

Giữ frozen ViT features, raw patch memory bank và bộ tạo proposal cập nhật của CADIC. Thay đổi chính là thêm HOPE/CMS để học **utility của proposal**:

1. CADIC tạo bank ứng viên từ ảnh normal vừa đến.
2. Controller nhìn projected features và các thống kê hình học của bank/proposal, dự đoán lợi ích cập nhật.
3. Controller chấp nhận hoặc từ chối toàn proposal; quyết định này thực sự thay đổi bank dùng để tính anomaly scores sau đó.
4. Khi ảnh normal kế tiếp thực sự đến, đánh giá hai bank alternatives đã giữ cố định để tạo delayed utility label. Objective kết hợp risk trên ảnh mới và anchors lịch sử.
5. Fast learner cập nhật theo từng completed label; slow learner dùng meta-gradient qua bốn support updates và bốn queries sau mỗi tám completed labels.

Backbone không được fine-tune. Raw bank không bị thay bằng reconstruction hoặc neural embeddings. Scoring vẫn dùng nearest-neighbor distances và **đầy đủ CADIC Eq9**, không thêm learned anomaly decoder. Test images/masks không cập nhật learner, bank hoặc anchors.

Đây là đóng góp đang kiểm chứng của Nested Learning: **học chính sách cập nhật normal-support memory** nhằm cân bằng thích nghi với dữ liệu mới và giữ normal support cũ. Nó không bảo đảm mọi proposal bị từ chối đều có lợi cho anomaly detection dài hạn.

### 2.2. Tối ưu execution và độ tin cậy

- Cache kết quả tìm closest pair theo các vùng bank bị thay đổi để tránh tính lại phần không đổi; tối ưu NN/write backend đã có kiểm parity và migration trên phạm vi được review.
- Có source/runtime/data identities, checkpoint atomic, lưu RNG/pending labels/partial episodes và resume có kiểm tra.
- Lưu raw scores/maps, task checkpoints và receipts để tính lại metrics và kiểm provenance.
- Tách training khỏi read-only evaluation; evaluation không làm đổi trạng thái học.

**Phần kế thừa và phần bổ sung:** nhánh Khang đã có manifest, task checkpoints, đánh giá final/forgetting và kiểm scoring không mutate state. Ta không coi những phần đó là contribution mới. Bổ sung của nhánh này tập trung vào lifecycle phức tạp hơn của controller: alternative banks, delayed labels, fast/slow optimizer states, partial meta episodes, atomic rollback và resume theo từng ảnh; cùng receipts/hash của raw outputs để review lại. Với backend CADIC, route `cadic_paperfaithful` của Khang đã dùng GEMM cho distance; khác biệt chính của ta là cache closest-pair theo dirty tiles và tối ưu cập nhật, không phải “Khang chưa dùng GPU/matrix multiplication”.

Các tối ưu execution này nhằm giảm chi phí tính toán, không đổi phương pháp để tạo gain benchmark. Tuy nhiên, NESTED vẫn phải dựng proposal, đánh giá các bank alternatives/anchors và cập nhật learner; **overhead so với CADIC vẫn đáng kể**. Chưa có timing fresh end-to-end trên RTX 5090 cùng môi trường cho cả hai để báo hệ số chính xác.

### 2.3. Tổ chức lại source thành workspace v7

- Một Python package tại `src/hope_cad`, có `pyproject.toml`, CLI và hướng dẫn chạy remote GPU.
- Hợp nhất code theo chức năng: data, models, baselines, benchmark, runtime và tests; bỏ code tiền thân không còn thuộc đường chạy hiện hành.
- Giữ nguyên method/study contracts; source identity mới phản ánh việc đổi package/import/layout.
- Kiểm refactor bằng AST của 16 core modules, paired state/event/score/resume trên tám modes và các bộ tests local/clean-source.

**V7 không phải một model mới đã tăng benchmark.** Kết quả trong bảng là evidence của phương pháp v5.1 đã chạy trước refactor. Chưa chạy lại full benchmark trên v7/RTX 5090; các kiểm tra engineering hỗ trợ bảo toàn logic nhưng không thay thế lượt chạy thực trên runtime mới.

Theo quyết định mới nhất của người thực hiện, baseline source chính thức để phát triển và rollback trước nâng cấp tiếp theo là commit (đã gồm docs/spec/plan công khai):

```text
89df9a21658b265731f98a7c2014afb1876f1924
```

## 3. Protocol của nhánh hope_cad_v7 khác nhánh dikhang_space ở đâu?

### 3.1. Khác biệt chắc chắn: loại 30 ảnh normal của Bottle

Route CADIC chuẩn của Khang dùng `mvtec_1x15_v1`, `dev_mode: disabled`. Hàm `build_training_manifest` lấy **toàn bộ PNG trong `train/good`**, theo task order rồi tên file, không tách 30 ảnh validation/probe. Với full MVTec AD, route này dùng **3.629 normal images**, trong đó Bottle có **209 ảnh**.

Protocol mà cả CADIC đối chứng và NESTED của ta đã chạy là `mvtec_1x15_prefix_cal_v2`: **3.599 normal images**, Bottle còn **179 ảnh**. Manifest thật trong evidence xác nhận đúng 30 IDs bị loại:

| Role lịch sử        | Ảnh Bottle loại khỏi online stream        | Số ảnh |
| ------------------- | ----------------------------------------- | -----: |
| `normal_validation` | `bottle/train/good/070.png` đến `084.png` |     15 |
| `normal_probe`      | `bottle/train/good/085.png` đến `099.png` |     15 |

Stream Bottle của ta đi `000–069`, rồi `100–208`; 14 category còn lại không bị cắt ảnh theo rule này. Không phải bỏ 30 ảnh mỗi category, không lấy từ official test và không phải mất ảnh ngẫu nhiên do loader.

**Vì sao còn partition này?** Nó được kế thừa từ thiết kế trước có phân vai first-task prefix cho học/chọn/kiểm tra phương pháp. V5.1 hiện tại học online từ đầu và **không chạy lại offline calibration/search hoặc dùng 30 ảnh bị loại để chọn checkpoint**. Hai role lịch sử `meta_support` (`000–049`) và `meta_query` (`050–069`) vẫn nằm trong online stream, được học khi thực sự đến, không phải thêm một lượt pretraining ngoài stream. Tên `prefix_cal` là lịch sử protocol, không mô tả một bước calibration đang diễn ra trong benchmark này.

**Hệ quả khi so kết quả:**

- Bảng CADIC/NESTED trong tài liệu là fair về tập train **giữa hai phương pháp của ta**, vì cả hai cùng dùng 3.599 IDs/order.
- CADIC chạy full 3.629 ảnh ở phía Khang **không phải cùng training protocol**. Không lấy chênh lệch giữa hai run rồi quy hoàn toàn cho Nested Learning, kể cả khi cùng K và checkpoint.
- Chỉ khoảng 0,83% tổng train images bị loại, nhưng tương đương khoảng **14,35% Bottle normals**. Không có cơ sở bỏ qua tác động vì “chỉ 30 ảnh”.
- Bank dùng chung và cập nhật phụ thuộc trajectory, nên bank khởi đầu khác có thể tác động các category về sau, không chỉ metric Bottle. Cũng không có bảo đảm thêm đủ 30 ảnh sẽ tăng mọi metric một cách đơn điệu.
- Chưa có paired experiment chỉ thay partition để đo causal impact của 30 ảnh. Không dùng khác biệt này làm lời giải thích đã chứng minh cho gap với paper.

### 3.2. Đối chiếu code thực thi, không chỉ tên phương pháp

| Thành phần         | Route CADIC chuẩn trong `dikhang_space`                                                                             | CADIC/NESTED đã đo của ta                                                             | Ý nghĩa                                                                                             |
| ------------------ | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Tập train          | Full `train/good`, dự kiến 3.629 ảnh trên full dataset                                                              | 3.599 IDs đã kiểm manifest; loại đúng 30 Bottle normals                               | **Khác protocol**                                                                                   |
| Task/order         | 15 category: Bottle → Cable → … → Zipper; filename sort; không shuffle                                              | Cùng task order; thứ tự tương đối giữ nguyên sau khi loại IDs                         | Không gọi là cùng exact stream                                                                      |
| Offline fitting    | Route CADIC không neural calibration                                                                                | V5.1 không offline calibration/search; controller học causal online                   | Partition lịch sử không là extra training                                                           |
| Loại backbone      | Frozen ViT-B/8 AugReg IN21K, `num_classes=0`                                                                        | Cùng lựa chọn kiến trúc/tag                                                           | Chưa đủ để khẳng định mọi checkpoint tensor khớp                                                    |
| Checkpoint loading | Script trỏ `.pth` state dict; extractor `torch.load`, strict load                                                   | `.safetensors` SHA `2c483958…`, loại đúng classification-head keys khi load           | Cần tensor-level comparison, không so SHA hai định dạng rồi kết luận weights khác                   |
| Feature layer      | Layer 9 one-based/block 8 zero-based, bỏ CLS; 784×768                                                               | Cùng block và raw geometry; không thêm LN/L2                                          | Thành phần kế thừa, không phải cải tiến mới                                                         |
| RGB transform      | PIL resize trực tiếp 224×224 bilinear, mean(.485,.456,.406), std(.229,.224,.225)                                    | Cùng recipe                                                                           | **Không có thay đổi normalization giữa hai route này**                                              |
| K                  | Config mặc định 2.500, cho phép 2.500/5.000/10.000; shell script bắt truyền K                                       | Các kết quả báo ở **K=10.000**                                                        | Phải lấy K thật từ run của Khang, không lấy default thay actual K                                   |
| Feature batching   | Loader batch 8 theo config; extractor xử lý batch; script khóa `update_unit=image`                                  | Benchmark stream lấy một ảnh/event; ghi receipt features theo ảnh                     | Batch extraction khác không có nghĩa bank cập nhật gộp tám ảnh                                      |
| Proposal CADIC     | Per-image repeated farthest-candidate/closest-bank-pair replacements; deterministic fill                            | Giữ proposal semantics; cached backend; NESTED thêm admission                         | Không thay memory unit thành một image hoặc bỏ repeated replacement                                 |
| Distance route     | Paper-faithful adapter được script chọn bằng module override; GEMM và quét bank pairs                               | Cached pair minima/dirty tiles và optimized NN/write                                  | Numerical route/execution identity khác; parity có phạm vi, không claim mọi tensor CUDA bit-exact   |
| Image scoring      | Code `score` thực thi đầy đủ `weight * star_score` của Eq9, b=9                                                     | Cùng Eq9 b=9                                                                          | YAML có chuỗi mô tả weight chưa đủ; code đã có nhân s\*                                             |
| Pixel scoring      | Raw NN map, bilinear 28→224, `align_corners=False`                                                                  | Cùng map policy, không thêm Gaussian                                                  | Không gán pixel AP gain cho việc thêm decoder/smoothing                                             |
| GT masks           | `convert(L)` → nearest resize → >0                                                                                  | Raw nonzero encoding trước nearest resize; hash/QA mask                               | Có hardening cho palette/0–1 masks; không tự khẳng định đã gây lệch trên MVTec, cần so actual masks |
| Metrics            | I-AUROC; `pixel_aupr` thực gọi sklearn average precision                                                            | I-AUROC/P-AP cùng loại định nghĩa; bổ sung metric receipts/BWT                        | Tên AUPR/AP có thể khác nhãn nhưng cùng phép tính ở code này; không mặc định khác số                |
| Task matrix        | Engine có final và `evaluate_forgetting` trên learned tasks; shell script hiện gọi train rồi eval, **không gọi fm** | Đã có đủ 120 cells/phương pháp và tính lại metrics                                    | Khang có khả năng chạy matrix, nhưng script này riêng nó chưa chứng minh đã chạy matrix             |
| Học utility        | CADIC route không controller                                                                                        | NESTED có fast/slow learner, anchors và delayed labels; CADIC đối chứng không learner | **Thay đổi khoa học chính**                                                                         |

**Chi tiết checkpoint cần trao đổi:** script provision trên nhánh Khang đã pin upstream safetensors cùng SHA `2c483958…` rồi tạo `.pth` bỏ classifier, đồng thời ghi rõ checkpoint lịch sử không còn để kiểm byte identity. Như vậy có nguồn upstream chung có thể đối chiếu, nhưng **file provision hiện tại không tự chứng minh checkpoint của mọi run CADIC trước đây được tạo bằng script đó**. Nếu dùng đúng rebuilt file thì xác minh non-head tensors; nếu là run lịch sử thì cần checkpoint receipt thực tế. Không gọi khác SHA container là khác backbone features khi chưa kiểm tensors.

Mặt khác, cả hai route đang giữ ImageNet mean/std và bilinear resize; native metadata của tag checkpoint khai báo recipe khác. Đây là **giới hạn chung so với native checkpoint config**, không phải một cải tiến preprocessing của ta so với Khang. Author recipe và tác động benchmark vẫn chưa được xác nhận đầy đủ.

### 3.3. Cái gì đã có kết quả, cái gì cần khớp trước khi so trực tiếp?

Tập test của benchmark ta đã nghiệm thu có 1.725 ảnh và 15 category. Nhánh Khang cũng có code enumerate official test toàn category, nhưng phải đọc test manifest/counts của run cụ thể để xác nhận byte-level equivalence. Không đọc config rồi tuyên bố artifacts của Khang đã đạt parity.

**Quyết định cho cohort tiếp theo: trở lại full-train 3.629 normals theo route CADIC chuẩn của Khang (`mvtec_1x15_v1`), không tiếp tục loại 30 Bottle normals.** Bottle có đủ 209 ảnh, các category khác giữ nguyên; ảnh được đưa vào đúng thứ tự causal, không tạo offline pretraining/calibration mới. CADIC, phương pháp recent-only hiện hành và candidate phải chạy fresh trên cùng tập đầy đủ này.

Đây là **thay đổi protocol đã implement trong successor research route**, local verified và chưa remote acceptance. Manifest/guards/resources mới tách baseline; trước run mới cần source/runtime preflight thật; không resume bank/checkpoint 3.599 ảnh rồi thêm bù 30 ảnh. Prefix pilot cũng đổi thành 209 Bottle + 47 Cable nếu vẫn dùng 256 ảnh đầu; gates/bindings phụ thuộc partition phải được rà lại trước execution, không tự coi spec cũ đã đồng bộ.

“Protocol chuẩn” ở đây là full-normal training của route Khang, **không có nghĩa đã khớp toàn bộ paper CADIC**. Checkpoint tensors, K, transforms, numeric route, masks/metrics và scope evaluation vẫn cần đối chiếu. Không đổi số liệu/provenance của benchmark 3.599 ảnh đã hoàn tất; cohort full-train mới không được dùng để quy toàn bộ chênh lệch so với kết quả cũ cho H32.

Ma trận hiện có gồm 120 ô/phương pháp: đánh giá từng category ngay sau khi học và sau các category tiếp theo. Forgetting tính mức giảm từ performance tốt nhất trước cuối stream; backward transfer so cuối stream với thời điểm vừa học xong category. Cả hai lấy trung bình trên 14 category có task theo sau. Final AUROC/AP là macro average qua 15 categories. Đây là thông tin bổ sung về retention, không thêm training hoặc làm performance tăng.

**Kết luận fairness:** có evidence matched cho **CADIC đối chứng của ta vs NESTED của ta**. Chưa có evidence đủ để đổi nhãn bảng này thành “NESTED vs run CADIC/ReplayCAD của Khang” hoặc reproduction chính xác paper. Protocol 30 ảnh là một khác biệt phải công bố, không phải sai sót được phép bỏ qua và cũng chưa là nguyên nhân duy nhất đã chứng minh.

## 4. Những vấn đề còn tồn tại

- Mới có một seed; chưa chứng minh statistical superiority hoặc generalization sang dataset khác.
- Lợi ích không đồng đều: pixel AP tăng mạnh ở một số category, nhưng có category giảm. Image AUROC final gần ngang không có nghĩa từng category đều không thua.
- Có transient image-ranking drop sau một lần chuyển category rồi hồi phục; final macro không thể hiện hết bất ổn này.
- Chưa cô lập đầy đủ gain đến từ learned policy, mức giảm ghi bank, fast learning, slow consolidation hay exact higher-order gradient.
- Phần lớn delayed labels và meta episodes hiện nằm trong cùng category gần đây. Đây là động cơ nghiên cứu, **chưa chứng minh đó là root cause** của các hạn chế.
- MVTec đã được dùng để phát triển phương pháp; các revision tiếp theo phải công bố development exposure. VisA cần khóa method/protocol và xác nhận exposure của cả nhóm trước đánh giá xác nhận.
- Cùng K=10.000 chỉ khớp **dung lượng raw main bank**, không phải tổng memory. NESTED còn giữ alternative bank cho delayed credit, anchors, learner/optimizer và replay records; báo memory/cost phải tính các phần này. Không gọi hai methods “equal total memory” chỉ vì cùng K.

## 5. Công việc đang chuẩn bị và sắp làm

### A. Phân tích offline từ evidence đã có

Đã có spec cho đối chiếu prequential predictions với geometry prior, mean/bias và các đối chứng đơn giản có clock. Mục tiêu là xem learned utility có cung cấp tín hiệu vượt các lời giải thích đơn giản hay không. Chỉ dùng normal-event records; không cần thêm lượt training GPU. **Analyzer đã triển khai và exercise normal trace trong local verification; không là policy intervention/AD benchmark mới.** Phân tích logged bank pairs không thay được benchmark của một policy mới.

### B. Một candidate nâng cấp historical-query đã đặc tả

Giữ geometry, CADIC proposer, bank budget, Eq9 và actual fast updates hiện tại. Thay đổi tập trung vào **slow meta-objective**:

- Lưu reservoir tối đa 32 completed decision records cũ.
- Giữ bốn support updates; sau startup, bốn queries gồm hai recent và hai historical records.
- Khóa causal eligibility, sampler/RNG, gradient semantics và checkpoint/resume trước implementation.

Mục tiêu là gây áp lực học để controller vẫn đánh giá được utility cũ sau khi thích nghi với context mới. Đổi lại, objective dành ít trọng số hơn cho dữ liệu gần đây; historical labels thuộc bank pairs cũ và có thể stale. **Chưa implementation, chưa adoption và chưa có benchmark gain của candidate này.**

### C. Kiểm chứng có giới hạn trước khi mở rộng

1. Implement phân tích offline và candidate/controls; kiểm numerical gradients, parity, causality và resume local.
2. Tạo source delivery mới, chạy preflight trên GPU thuê và normal-only pilot; đo cả memory, latency và cơ chế thực sự tác động bank.
3. Nếu đạt gates, chạy fresh matched CADIC, phương pháp recent-only hiện hành và candidate historical-query trên cùng môi trường/dữ liệu.
4. Chỉ khi kết quả phù hợp và có ngân sách mới mở rộng controls, seeds và attribution. Không thử vô hạn hoặc sửa threshold để cứu kết quả âm.
5. Khóa phương pháp và pipeline trước VisA; learner/bank khởi tạo fresh cho dataset mới, không chỉ test bank đã học MVTec.

## 6. Phối hợp với nhánh `dikhang_space`

Đã xem các nhóm thay đổi Task P, ECAD/ContCore, NPIM và CSG; mức review khác nhau, chưa nghiệm thu toàn bộ experiments hoặc kết quả chạy của nhánh đó. Những hướng này hiện được giữ làm tham khảo, **chưa merge vào phương pháp v7**.

Với phần **CADIC chuẩn**, ta đã xác định được từ code: full normal train, cùng feature layer/RGB recipe/Eq9, K được truyền qua CLI, và script có module override để dùng paper-faithful adapter. Điều còn cần Khang gửi là **run-resolved manifests/config/checkpoint receipts/metrics**, không phải đoán lại những defaults đã đọc được.

**ReplayCAD là route khác:** config hiện trỏ native InvAD ở 256×256, encoder pretrained, diffusion/condition-learning assets và generated replay; không phải ViT-B/8 raw-NN pipeline ở bảng CADIC/NESTED. So sánh với ReplayCAD phải công bố architecture, input size, replay storage và compute riêng, đồng thời thống nhất data rights/evaluator nơi có thể. Chưa có matched ReplayCAD artifacts được nghiệm thu phía ta; không lấy trường `reported_reference` trong YAML làm kết quả reproduce đã chạy.

Để thống nhất đối chứng trong nhóm, cần trao đổi rõ:

- CADIC/ReplayCAD trên MVTec và VisA đang dùng commit, checkpoint, preprocessing, K, split/order, scoring và metrics nào; trạng thái runs và artifacts tương ứng.
- ContCore có được thêm làm baseline chính thức hay chỉ là exploratory experiment; phân biệt adaptation cùng backbone với author reproduction.
- NPIM/CSG có thay pipeline baseline hay là các hướng nghiên cứu riêng. Không gộp kết quả giữa các phương pháp/protocol khác nhau.

VisA: phía ta có constants/manifest helpers cho official split, nhưng chưa có completed matched VisA suite. Có code hỗ trợ dataset không đồng nghĩa đã train/evaluate xong. Không suy trạng thái VisA phía Khang từ sự tồn tại của config.

Các experiment mới của Khang không tự thay baseline CADIC đã đo ở đây. Ngược lại, kết quả NESTED nội bộ cũng không tự chứng minh hơn các run riêng của Khang khi chưa khớp protocol.

## 7. Nguồn để kiểm lại

### Code hai nhánh đã đọc cho lần đối chiếu này

Các paths sau tính từ root của từng checkout, tương ứng hai commit ghi đầu tài liệu:

| Câu hỏi             | `dikhang_space`                                                                                     | `hope_cad_v7`                                                                                    |
| ------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Route chạy CADIC    | `scripts/benchmarks/run_cadic_paperfaithful.sh`; `scripts/benchmarks/1_run_benchmark.py`            | `src/hope_cad/runtime/execution.py`; `src/hope_cad/benchmark/`                                   |
| Partition/order     | `conf/benchmarks/protocols/mvtec_1x15_v1.yaml`; `dataset/benchmark_manifest_v1.py`                  | `src/hope_cad/resources/protocols/mvtec_matched.yaml`; `src/hope_cad/data/prefix.py`             |
| RGB/GT loader       | `dataset/benchmark_protocol_v1.py`                                                                  | `src/hope_cad/data/protocol.py`; `src/hope_cad/data/prefix.py`; `src/hope_cad/benchmark/data.py` |
| Backbone/checkpoint | `models/feature_extractors/cadic_vit_v1.py`; `scripts/hope_cad/provision_vit_b8_checkpoint.py`      | `src/hope_cad/models/features.py`; `src/hope_cad/resources/method.json`                          |
| Proposal/scoring    | `models/cadic_paperfaithful/cadic_patch_coreset.py`; `cadic_benchmark_adapter.py` trong cùng folder | `src/hope_cad/baselines/cadic/`; `src/hope_cad/models/geometry.py`                               |
| Metrics/matrix      | `training/benchmark_engine_v1.py`; `training/benchmark_metrics_v1.py`                               | `src/hope_cad/benchmark/metrics.py` và các modules trong `benchmark/`                            |
| ReplayCAD config    | `conf/benchmarks/methods/replaycad.yaml`; `scripts/benchmarks/replaycad/`                           | Integration/reference riêng; chưa có matched completed run                                       |

30 IDs bị loại đã được đối chiếu với **`benchmark/manifest_train.json` trong evidence NESTED đã hoàn tất**, không chỉ suy từ YAML. Các nhận định về Khang ở đây là source inspection của route được chỉ rõ; chưa chạy lại code nhánh Khang hoặc xác minh mọi run artifacts. Lần cập nhật này chỉ sửa báo cáo, không đổi source thuật toán, protocol hay benchmark results.

### Những nguồn thực sự có trong branch public

- [Method](docs/method.md): equations, ownership, causal utility và learning rules hiện hành.
- [Architecture](docs/architecture.md): package/function ownership, runtime entry và integrations.
- [Benchmark](docs/benchmark.md): protocol, metric definitions, registered controls và acceptance.
- [Operations](docs/operations.md): setup, CLI, preflight, fresh train/evaluate/matrix, resume và export.
- `src/hope_cad/resources/method.json`, `study.json`, `runtime.json`, `lineage.json`: contracts/identities; các trạng thái ghi lúc chốt spec là metadata lịch sử, không thay current evidence status trong tài liệu này.

Không cần truy cập bất kỳ kho context ngoài branch nào để hiểu các quyết định và kế hoạch bên dưới. Raw run ZIP/logs không có trong branch. Để tự tái tính benchmark cũ, cần yêu cầu riêng hai ZIP matrix đã định danh ở Phụ lục B; không có chúng thì chỉ kiểm được code/contracts và bản tổng hợp, chưa tự xác minh raw experiment.

---

## Phụ lục A — Phương pháp hiện hành, đủ context để đọc code

### A1. Ba loại memory và ba quá trình cập nhật

| Thành phần       | Lưu gì?                                                 | Khi nào thay đổi?                                 | Vai trò                                                    |
| ---------------- | ------------------------------------------------------- | ------------------------------------------------- | ---------------------------------------------------------- |
| Raw bank C       | Tối đa 10.000 raw patch vectors 768D đã quan sát        | Khi proposal được nhận                            | **Trực tiếp** quyết định patch/image anomaly scores        |
| Anchor reservoir | 32 ảnh lịch sử, mỗi ảnh giữ 8 sampled raw patches       | Theo causal image reservoir                       | Tham chiếu normal-risk để hạn chế proposal gây hại quá khứ |
| Utility learner  | Slow parameters theta và persistent fast parameters phi | Fast mỗi completed label; slow mỗi block 8 labels | Học có nên nhận proposal tiếp theo                         |

Visual memory bên trong encoder được self-modify theo scan/chunks trong một ảnh rồi reset working state giữa ảnh; persistent fast CMS/readout giữ adaptation xuyên ảnh. Slow visual/geometry/CMS parameters nhận meta-gradient. Vì vậy “memory” không chỉ một raw bank và “slow” không có nghĩa một teacher checkpoint dùng để tính anomaly score.

Parameter ownership cụ thể:

- **Frozen:** ViT, Rademacher projection 768→32, raw bank/risks không có gradients; fast initialization template không học.
- **Fast phi:** CMS_fast và linear readout gồm 33 weights + bias. Readout khởi tạo zero; không reset theo category.
- **Slow theta:** visual wrapper/learned initial maps, affine geometry16→32 và CMS_slow.
- **Không có:** optimizer trên ViT, gradient xuyên hard admission/coreset replacement, task-label routing, reset fast state ở task boundary hoặc learned decoder trên anomaly test.

Context gồm projected image tokens `[1,784,32]`, được per-image RMS normalize theo rule đã khóa, và 16 geometry statistics. Raw 768D bank **không** dùng projected/RMS-normalized tokens này; scoring geometry không bị controller representation thay thế. Utility context có causal stream clock; đó là một khả năng shortcut cần đối chứng, không tự là leakage vì clock đã biết khi quyết định.

### A2. Causal utility label và admission

Với image/reference patch set X và bank C, normal coverage risk là:

```text
R(X,C) = 0.5 * mean_j d(x_j,C)^2 + 0.5 * max_j d(x_j,C)^2
d(q,C) = min_c ||q-c||_2
```

Khi normal image x_t đến, resolve credit cũ trước. Đặt C− là bank hiện tại, C+ là bank sau đúng CADIC proposer. Dùng anchor reservoir trước khi thêm x_t để tính a−/a+. Với anchor weight lambda=1:

```text
c_t     = 1 + R(x_t,C−) + R(x_t,C+) + a− + a+       # known, >= 1
gamma_t = [(R(x_t,C−)+a−) - (R(x_t,C+)+a+)] / c_t   # geometry prior
p_t     = gamma_t + w_fast · concat(h32, log1p(c_t)) + b_fast
```

Trong 32 completed labels đầu, changed proposals được force accept để khởi động học; sau đó nhận proposal khi `p_t >= 0`. Nếu proposal không đổi bank, không tạo fake learning target. Fill/bootstrap giữ semantics của CADIC.

Giữ alternative bank và cached anchor costs đến khi **normal x\_(t+1) thật sự đến**:

```text
J−  = R(x_(t+1), C−) + a−
J+  = R(x_(t+1), C+) + a+
y_t = (J− - J+) / stored_c_t
```

Nhãn y đo lợi ích realized của **hai bank alternatives cũ**, không dùng future image trước khi nó đến. Cả accepted và rejected proposal đều có label. y không phải anomaly label/probability, không bị ép [-1,1]. c được cố định khi quyết định, không lấy future denominator để đổi target. Không replay raw anomaly test để sinh utility targets.

Anchors được cache cho fixed-pair contrast; label không so sánh hai rollout dài hạn. Sau khi y đến, learner mới cập nhật; rồi mới predict proposal hiện tại. Pause/task boundary không tạo label, không flush meta block. True stream end chỉ censor nếu còn pending; không bịa ảnh kế tiếp.

### A3. Fast rule và exact slow meta-gradient

Đặt `u = concat(h32, log1p(c))`, `d = ||u||² + 1`. Fast update:

```text
loss     = 0.5 * (p - y)^2
phi_next = phi - 0.05 * partial_phi(loss) / d
```

Đây là lấy gradient của loss rồi chia context energy; **không** lấy gradient của `loss/d`. Neural state FP32; dot/scalar và quotient bridges theo numeric contract hiện hành. Actual phi detach giữa các completed labels để không giữ đồ thị toàn stream.

Trong một block tám labels r0…r7:

1. Lưu detached phi_start trước actual fast update đầu block.
2. Actual learner cập nhật cả tám labels; theta giữ nguyên tới cuối block.
3. Functional replay clone phi_start, thực hiện bốn support updates r0…r3.
4. Outer loss là mean bốn query losses r4…r7 tại functional phi4; không học tiếp trên queries.
5. Exact meta-gradient đi qua cả bốn normalized updates, gồm denominator derivative. Adam slow LR=1e−4, global grad clip=1.
6. Deploy **actual phi8**, không deploy functional phi4 hoặc reset phi sau slow step.

First-order control giữ forward values nhưng detach **toàn normalized increment g/d**, không chỉ numerator. Nhờ vậy control kiểm phần higher-order credit assignment, không âm thầm thay update values ở cùng trạng thái đầu vào. Qua nhiều updates, theta và policy trajectories vẫn có thể khác.

### A4. Tại sao có cơ sở nghiên cứu nhưng không có guarantee thắng?

Nếu context Z chứa c>0 đã biết, MSE predictor đủ khả năng biểu diễn có population optimum `E[y|Z]=E[J−−J+|Z]/c`. Sign ở optimum phù hợp expected one-step advantage. Nhưng actual model có compressed context, finite data, optimization và policy-dependent distribution; không bảo đảm đạt optimum. Ngay cả perfect one-step utility predictor cũng chưa tối ưu multi-step anomaly AP.

Raw-support geometry có một bound đơn giản: nếu C là subset của observed support S với coverage error epsilon, thì `d(q,S) <= d(q,C) <= d(q,S)+epsilon`. Normal coverage có liên hệ đến khả năng giữ features, nhưng bound không xác nhận assumptions về anomalies, Eq9 ranking hoặc AUROC/AP. Vì vậy ta dùng normal objective để học và benchmark anomaly để kiểm, không gọi normal-risk PASS là AD superiority.

### A5. State/resume và scoring

Session sở hữu image cursor, seen IDs, bank, anchors, pending alternative, learner/Adam, replay block và RNG. Một image transaction phải commit hoặc rollback đồng bộ; không để bank đã tiến mà learner/clock chưa tiến. Checkpoints lưu mỗi tám ảnh, task boundary, pause và stream end. Cached feature receipts kiểm replay sau interrupt tạo lại đúng features.

Inference dùng bank cuối đã chọn, không chạy utility learner để sửa test scores. Evaluator không tăng image/label clocks, không sample anchors hoặc mutate RNG. Task checkpoints là benchmark artifacts để đánh giá retention, không tất cả là persistent deployment memory.

## Phụ lục B — Benchmark chi tiết và phạm vi bằng chứng

### B1. Final results theo category

Tất cả ở cùng matched stream 3.599 normals, K=10k, seed 0. Δ là NESTED−CADIC, đơn vị điểm phần trăm.

| Category   | CADIC I-AUROC % | NESTED I-AUROC % |        Δ | CADIC P-AP % | NESTED P-AP % |        Δ |
| ---------- | --------------: | ---------------: | -------: | -----------: | ------------: | -------: |
| bottle     |         94,3651 |          97,1429 |  +2,7778 |      57,9602 |       72,6291 | +14,6689 |
| cable      |         91,5105 |          90,2924 |  −1,2181 |      17,6164 |       20,7456 |  +3,1292 |
| capsule    |         60,3510 |          68,8472 |  +8,4962 |      11,5457 |       14,8455 |  +3,2997 |
| carpet     |         97,3917 |          98,7159 |  +1,3242 |      67,4795 |       67,3478 |  −0,1316 |
| grid       |         98,3292 |          99,4987 |  +1,1696 |      40,5168 |       40,5461 |  +0,0293 |
| hazelnut   |         97,0000 |          96,9286 |  −0,0714 |      64,5981 |       64,5424 |  −0,0556 |
| leather    |        100,0000 |         100,0000 |   0,0000 |      22,6877 |       22,3989 |  −0,2888 |
| metal_nut  |         92,2776 |          86,8035 |  −5,4741 |      79,4315 |       85,5602 |  +6,1286 |
| pill       |         79,2417 |          63,9389 | −15,3028 |      67,5811 |       66,4171 |  −1,1640 |
| screw      |         22,0947 |          34,7407 | +12,6460 |       5,2631 |       13,7311 |  +8,4680 |
| tile       |         99,9278 |          99,9639 |  +0,0361 |      60,9595 |       63,0764 |  +2,1169 |
| toothbrush |         92,5000 |          91,6667 |  −0,8333 |      52,7279 |       52,9095 |  +0,1816 |
| transistor |         88,9583 |          84,4167 |  −4,5417 |      28,8548 |       29,4742 |  +0,6194 |
| wood       |         99,8246 |          99,7368 |  −0,0877 |      67,5419 |       67,9879 |  +0,4460 |
| zipper     |         91,2027 |          91,6492 |  +0,4464 |      28,7538 |       29,2841 |  +0,5303 |

NESTED tăng pixel AP ở 11/15 category; mức tăng tập trung đáng kể ở bottle/screw/metal_nut. Không loại bốn category giảm khỏi macro. Screw có image AUROC yếu ở cả hai và ngay khi vừa học xong task; chưa thể giải thích chỉ bằng forgetting hoặc learned gate.

### B2. Acquisition và retention

Gọi a[i,j] là metric category j sau khi học xong category i, indices 0…14. Chỉ đánh giá j<=i:

```text
Final = mean_j a[14,j], j=0..14
FM    = mean_j (max_{i=j..13} a[i,j] - a[14,j]), j=0..13
BWT   = mean_j (a[14,j] - a[j,j]), j=0..13
```

Không clamp negative forgetting hoặc tính future cells bằng 0. Mean diagonal P-AP: CADIC **53,2123%**, NESTED **53,4952%**; diagonal I-AUROC: **93,1858% / 93,1691%**. Macro acquisition gần nhau, còn pixel final/retention khác; không phải proof mọi category được học equally well.

Sau category metal_nut, NESTED có một transient dip về mean image AUROC trên **cùng bảy category cũ**: 93,8241% → 80,3633%, sau đó về 93,3766% ở checkpoint tile. Pixel AP không tụt tương ứng. Chi tiết này được giữ vì final macro có thể che bất ổn image ranking, không phải nhật ký các prototype. Chưa có intervention cô lập nguyên nhân.

### B3. Tính công bằng đã kiểm thực tế

Hai jobs dùng canonical T4 environment: Python3.13.15, PyTorch2.11.0+cu128; core chạy **một GPU `cuda:0`**, không DDP tự chia hai T4. Đã kiểm:

- Cùng train IDs/order và test manifest; 3.599 raw-feature hashes khớp từng ảnh giữa CADIC và NESTED.
- Cùng checkpoint/producer/raw geometry, K, Eq9 và mask/metric policy; CADIC fresh không nhận learned NESTED weights.
- Đủ task checkpoints/owner/locks và 120 matrix cells mỗi phương pháp; matrices không có additional training.
- Hai ZIP matrix có tổng **22.943 payloads**, đã kiểm inventory/member SHA; raw score/maps được dùng để tính lại **240 cells** trên CPU. Image metric error <1e−12, pixel <1e−6 theo review tolerances.
- 27.900 image-evaluations qua matrices là đánh giá lặp các 1.725 test images, **không phải 27.900 ảnh độc lập**.

Việc review này đã diễn ra trong workflow trước. Lần viết tài liệu hiện tại đọc lại source/receipts và selected train manifest, không chạy inference/training hoặc tự tính lại mọi cell lần nữa.

### B4. Artifact identities nếu cần chuyển riêng để tự kiểm

Raw data/weights/logs/ZIPs không nằm trong branch Git. Hai completed matrix archives tương ứng bảng này là:

```text
NESTED: hope_cad_matrix_NESTED_seed0_evidence.zip
SHA256: 668f9e348561d5ec55c4cf51762c5890b19328c2a950c7ba2fb09f037b1bea0d

CADIC: hope_cad_matrix_CADIC_seed0_evidence.zip
SHA256: 035f88639f8faab8bf02fdfbcd5f502677620c3756af53dcf4815dd40e8cc889

ViT source checkpoint: vit_base_patch8_224.augreg_in21k.safetensors
SHA256: 2c4839589adbbf701573dd155e1814f8adb145cd4d235bb86a93269369e111ec

Method design digest:
6f6b2c847a815619b6ad5ed899a0859aab1d7b532d62b27ae3217d926695db99
```

Các SHA là locator/integrity anchors, **không thay việc nhận và đọc artifact**. Người chỉ có branch và báo cáo này chưa tự xác minh raw results được. Để review đúng cần các member `benchmark/manifest_train.json`, test manifest, per-job task states, events, score/map chunks, metrics và artifact manifests trong ZIP. NESTED ZIP có namespace `benchmark/legacy/` giữ lịch sử; không glob mọi `/events/` rồi cộng trùng với canonical events.

### B5. Runtime và tradeoff đã đo

| Phạm vi                          | Thời gian T4 đã ghi                          | Cách diễn giải                                                             |
| -------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------- |
| CADIC fresh học đủ 3.599 normals | 7.918,867 s ≈ **2 giờ 12 phút**              | Gồm features, bank updates, checkpoints/I/O                                |
| CADIC final evaluation/staging   | 464,089 s ≈ **7 phút 44 giây**               | Tách khỏi training                                                         |
| NESTED continuation 384→3.599    | ≈ **4 giờ 34 phút 44 giây** tới hết phần học | 3.215 ảnh; có worker initialization/load/I/O; không phải fresh toàn stream |
| NESTED final evaluation/staging  | ≈ **8 phút**                                 | Không bao gồm full historical-task matrix                                  |
| Bổ sung task matrix CADIC        | ≈ **62,78 phút** tới export                  | Read-only evaluation; không train                                          |
| Bổ sung task matrix NESTED       | ≈ **73,63 phút** tới export                  | Read-only evaluation; không train                                          |

Không lấy 4h35/2h12 để báo một hệ số fresh-training chính xác vì khác phạm vi/prefix. Không cộng các lượt profiling/reference hoặc thời gian export vào training rồi so với timing paper không cùng scope. Dữ kiện hiện tại vẫn cho thấy NESTED có overhead đáng kể và chưa có lợi thế accuracy trên cả hai endpoints.

Raw main bank K=10k FP32 là 10.000×768×4 = **30.720.000 bytes**, khoảng **29,30 MiB**. Đó không phải tổng memory CADIC hoặc NESTED. NESTED có additional learner/optimizer, anchor records, replay và alternative full bank; backend có caches/temporary buffers; task checkpoints là disk artifacts. Một memory-matched comparator lớn hơn đã được đăng ký trong study hiện hành nhưng chưa có completed result ở đây.

## Phụ lục C — Thiết kế candidate đã implement local, chưa nghiệm thu remote

### C1. Tại sao chọn historical-query consolidation?

Trace hiện có cho thấy **388/401 slow episodes** gồm records cùng category và **3.199/3.211 delayed labels** nhận reward từ ảnh kế tiếp cùng category. Controller đã tác động thực tế: 1.935/3.179 post-warmup decisions khác geometry-prior action trên trajectory đã đi qua. Điều này cho động cơ mở rộng slow query context, không chứng minh short context là root cause hoặc các quyết định khác prior đều đúng.

Slow weights/Adam và anchors đã mang lịch sử; không mô tả model hiện hành là “chỉ nhớ ảnh kế tiếp”. Candidate đặt thêm câu hỏi rõ: **sau recent fast adaptation, slow representation có tiếp tục giúp dự đoán utility của completed contexts cũ không?**

Giữ raw geometry/proposer/fast updates để thay đổi có thể attribution. Không đồng thời ghép ContCore, rare-mode anchors, candidate/victim context mới, learned rates, tăng width/CMS depth/unroll hoặc đổi normalization. Những hướng đó có thể nghiên cứu riêng, không tự bật nếu candidate chưa tốt.

### C2. Phân tích offline trước GPU

Phân tích prequential trên logged normal bank pairs so factual NESTED predictions với gamma, mean, gamma+bias, ridge affine gamma, ridge gamma+clock và online scalar bias. Prediction tại quyết định chỉ dùng labels đã thực sự tới; tại một cursor resolve old credit trước prediction mới. Không random split retrospective để baseline nhìn tương lai.

Recipes chính: ridge sum-loss float64, lambda1e−6, không penalize intercept; causal clock `log1p(cursor)/log1p(1e6)`; scalar bias b0=0, update `b←b−.05*(cached_p−y)`. Statistical fits dùng gamma khi chưa có 32 past labels nhưng vẫn cập nhật statistics; online bias học từ label đầu. Metrics: target MSE/MAE, raw-advantage error, sign/action agreement, fixed-pair one-step regret và margin/cancellation diagnostics. Đây không là policy counterfactual rollout hoặc AD benchmark.

Chỉ đọc canonical 3.599 events, 3.211 completed credits và 3.179 post-warmup rows; loại 376 legacy event copies khỏi analysis. Không model forward/tensor loading/test scores/GT. Budget dự kiến của analysis đã khóa ở tối đa2 CPU threads,2 GiB targetRAM,30 phút softcap gồm hashing; đây là cap không ETA. Chưa có analyzer/results theo spec mới tại ngày tài liệu.

### C3. Candidate H32: thuật toán đã xác định

Candidate contract ID sau amendment full-train **`hope_cad_hq32_fulltrain_v1`** (tiền thân đặc tả `hope_cad_hq32_v1`), đã implementation/local verification, chưa remote acceptance/adoption. Các constants kế thừa: K=10k, anchors 32×8, stats 16, warmup 32, fast LR .05, slow Adam 1e−4, precision và Eq9. Chỉ slow query mixture thay đổi.

| Thành phần             | Định nghĩa                                                                                                                   |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| History                | Reservoir tối đa32 immutable completed utility records; không chia theo task                                                 |
| Record                 | Projected tokens `[1,784,32]`, stats 16, c/gamma/target/J±/decision denominator, identity/cursor và reward-observed metadata |
| Actual fast learning   | Cả tám new completed labels của block; deployed phi8 giữ như baseline                                                        |
| Functional support     | Bốn records mới đầu block r0…r3; replay từ detached phi_start                                                                |
| Startup queries        | Block0 dùng r4…r7, khớp baseline                                                                                             |
| Queries từ block sau   | `[r6,r7,h0,h1]`, mean bốn losses; không query adaptation                                                                     |
| Historical eligibility | Reward đã tới và record thuộc completed block trước khi block hiện tại bắt đầu                                               |
| Sample                 | Hai distinct records, private `random.sample`, giữ returned order; không chọn theo task/loss/utility                         |
| Reservoir update       | Algorithm R trên completed records; insert cả tám chronological records **sau** successful slow commit                       |
| RNG                    | Reservoir `Random(seed+96136)` và query `Random(seed+96137)` riêng, không làm đổi model/anchor RNG                           |
| Mid-block resume       | Giữ phi_start, partial records, fixed selected IDs, history và hai RNG states; không resample                                |
| Failure/transaction    | Rollback bank/learner/Adam/history/RNG/clocks/pending/anchors đồng bộ                                                        |
| End of stream          | Không flush partial block hoặc tạo labels giả; chỉ censor pending thật nếu có                                                |

History labels mô tả original bank pair; replay recompute model/denominator dưới functional theta/phi, **không** coi stored target là reward current bank. Historical metadata không đưa category/identity vào model inputs. Selected historical records là immutable references, không buffer tokens nhân đôi.

Token payload của 32 history records là **3.0625 MiB**; đây chưa phải total/peak overhead. Replay 8, metadata, debug snapshot, optimizer, alternative bank và transaction copies vẫn phải đo. Giữ tổng 4 queries không chứng minh runtime bằng baseline.

Không import learned state của lượt MVTec trước để làm initialization cho cohort mới. New history/state schemas và per-job method identity phải khác baseline, old guards giữ strict. Code hiện hành vẫn chỉ có baseline recent-query semantics; các lệnh CLI hiện chưa chạy H32 được.

### C4. Controls đã chốt cho candidate study mới

| Variant                | Câu hỏi                                                                                         |
| ---------------------- | ----------------------------------------------------------------------------------------------- |
| CADIC                  | Toàn hệ thống có đáng chi phí so với geometry baseline?                                         |
| R_EXACT                | Phương pháp recent-only hiện hành trên cùng cohort/runtime mới                                  |
| R_FO                   | Vai trò higher-order gradient khi queries chỉ recent                                            |
| H32_EXACT              | Candidate history-query chính                                                                   |
| H32_FO                 | Context history có ích khi bỏ higher-order support path không?                                  |
| H32_DIRECT             | Cùng queries tại detached phi_start, zero functional support steps; phân biệt với direct replay |
| FAST_ONLY              | Chỉ actual fast learning, không slow Adam                                                       |
| BIAS_VETO              | Gamma+scalar bias với fast update đơn giản, không CMS                                           |
| PILOT_RATE_RANDOM_VETO | Admission ngẫu nhiên có rate cố định lấy từ normal pilot trước, không theo future task          |

Đây là **study mới đã implement/test local**, cả chín variants có code; chưa có remote pilot hoặc full benchmark hoàn tất. Không nhầm chúng với danh sách controls hiện hành trong `resources/study.json`. Recent exact/FO sẽ giữ history 32 bookkeeping để đối chiếu storage/sampler tốt hơn, nhưng loss chỉ recent. DIRECT vẫn có actual fast learning và ít functional compute hơn exact; zero head làm first-block slow gradient zero là đúng dự kiến. FIRST_ORDER detach toàn g/d.

Random veto khóa reject probability 145/211, accept 66/211, private RNG seed+95537 và warmup 32 force accept; không lấy full-stream rejection rate theo task nhìn trước. Bias dùng p thật tại decision khi delayed target đến. Các control đơn giản giúp kiểm khả năng gain chỉ do giảm ghi hoặc bias, không gán mọi hiệu ứng cho NL.

### C5. Kế hoạch có giới hạn và điều kiện dừng

1. **Nghiệm thu local:** triển khai analysis offline và candidate. Kiểm historical meta-gradient bằng reduced-FP64 finite differences và full-grid FP32 exact backward; kiểm baseline parity, Algorithm R/private RNG, chronology, mọi độ dài partial block, rollback và scoring read-only. Tests không thay anomaly metrics.
2. **Preflight GPU thật và normal pilot:** dùng 24 discarded normals cùng bốn normal events tiếp theo để kiểm resume, cộng các historical-gradient synthetic fixtures. Sau amendment full-train, chạy chín variants fresh trên 256 allowed normals đầu: **209 Bottle + 47 Cable**, thay prefix 179 Bottle + 77 Cable của spec partition cũ; bindings/gates cần được rà lại trước run. Không chuyển pilot weights sang benchmark. Tổng cap 12 GPU wall-hours, mỗi session soft limit tối đa bốn giờ và reserve 180 giây. Cap gồm setup/features/I/O; không phải ETA hoặc quyền tự thuê/chạy.
3. **Gates của pilot:** cả chín arms hoàn tất. Mỗi non-CADIC arm có ít nhất 64 completed labels; R/H có ít nhất tám slow commits, các fast/bias/random arms không có slow Adam. H32 có historical-gradient contribution hữu hạn, khác zero sau startup. Post-warmup reject fraction trong [0,01; 0,99], bank khác CADIC. Sau 256 ảnh, score read-only 24 allowed train IDs đầu; max absolute pixel-map difference giữa H32 và fresh CADIC phải >1e−6. Mean prequential normal-risk ratio H32/CADIC trên 128 ảnh cuối <=1,10. Đây là kiểm cơ chế/safety, không AD guarantee.
4. **Fresh main comparison:** CADIC, R_EXACT, H32_EXACT seed 0; mỗi job dự kiến học **3.629 ảnh** và đủ 120 matrix cells theo protocol full-train đã chọn; phải implement successor contracts theo amendment đã chốt trước khi chạy. Batch có cap 24 GPU wall-hours; hết cap chưa đủ không được coi complete hoặc tự mở thêm jobs.
5. **Development spend-screen:** H32−R pixel AP >0; H32−R image AUROC >=−2 điểm %. H32−CADIC phải >=−2 điểm % trên cả image AUROC và pixel AP. Screen chỉ quyết định có đáng đầu tư attribution tiếp không, không phải statistical NI/superiority hoặc proof gain có ý nghĩa.
6. **Nếu được review và cấp budget tiếp:** chạy sáu remaining seed-0 controls, rồi seeds 1/2 cho CADIC, R_EXACT và H32_EXACT. Tổng tối đa 15 full jobs: ba main + sáu controls + sáu replications; không tự chạy mọi control × ba seeds. Kết quả âm/inconclusive vẫn báo, không sửa method để vượt screen.
7. **Statistics/adoption:** paired whole-image bootstrap theo category/image-label strata, 2.000 replicates, bootstrap seed 20261020, percentiles 1,25/98,75. NI criterion yêu cầu cả hai primary lower bounds >−0,02; superiority criterion yêu cầu cả hai >0. Finite-three-seed reporting và các contrasts/conditional screening vẫn là exploratory trên MVTec, không tự thành population-seed hoặc confirmatory proof.
8. **VisA:** chỉ sau method/protocol freeze mới học fresh bank/learner trên official train normals, theo official 1cls split có 12 categories. Nếu giữ cùng evaluation design, matrix có 78 cells. Chưa có completed runner/suite; không đổi mỗi data path hoặc dùng test VisA để chọn history ratio/capacity/LR.

Chỉ một candidate, một normal pilot và một first main batch; không hyperparameter search vô hạn. Budget extension là quyết định tài nguyên có record, không quyền tự đổi thuật toán. Proven bug/contradiction cần version/amendment công khai trước dependent run, không đổi recipe âm thầm sau khi xem metrics. Các gates là lựa chọn nghiên cứu/quản trị đã xác định, không phải theorem của Nested Learning.

### C6. Những lựa chọn cố ý chưa thay

- **Normalization:** giữ matched recipe để cô lập history effect; native config mismatch chưa đo causal impact. Nếu đổi sau này cần new producer/features/banks và fresh comparators.
- **Anchors:** giữ 32×8 uniform; chưa sửa sampler cho rare modes cùng lúc.
- **Action/context:** giữ whole-proposal binary action và stats 16; chưa thêm candidate/victim-level representation.
- **Learning rule:** giữ rates/optimizer/topology/unroll; chưa học rate/retention/momentum mới.
- **ContCore:** chưa thay proposer; có thể related work/baseline riêng với explicit protocol.
- **Timescale attribution:** exact/FO/direct không tự chứng minh multi-frequency gain; cần frequency/synchronous control riêng trước claim đó.

Những hướng hoãn không bị coi là vô ích, chỉ không được ghép cùng candidate khiến gain không giải thích được. Không có cam kết “khai thác hết NL” hoặc dự báo tăng bao nhiêu AP.

## Phụ lục D — Đọc/chạy branch public và bàn giao giữa hai phía

### D1. Trạng thái nào đã đạt?

| Hạng mục                           | Trạng thái tại ngày cập nhật                                                |
| ---------------------------------- | --------------------------------------------------------------------------- |
| Utility controller v5.1            | Đã code, local numerical/causal/state tests và normal pilot đã review       |
| Matched CADIC/NESTED MVTec seed 0  | Đã hoàn tất final endpoints + hai task matrices, có raw-score-backed review |
| Statistical superiority/multi-seed | Chưa xác nhận; không suy từ một seed                                        |
| Refactor/package/CLI v7            | Đã implementation và local/clean-source checks                              |
| Fresh full benchmark v7/RTX 5090   | Chưa chạy/nghiệm thu                                                        |
| Analysis offline theo recipes mới  | Đã implement/exercise normal events; chưa AD intervention                                      |
| H32 candidate/controls study       | Đã code/local verified; chưa remote run/adopt                            |
| ReplayCAD matched đối chiếu NESTED | Chưa có accepted artifacts phía này                                         |
| VisA matched suite                 | Chưa có completed benchmark; cần protocol/exposure freeze                   |
| ContCore/NPIM/CSG của Khang        | Tham khảo source, không tự gán acceptance/status từ phía ta                 |

### D2. Clone/runtime và CLI hiện có

Không cần context ngoài repo để setup. Cần dataset và checkpoint riêng; cài dependencies phù hợp trước editable install. Windows local có Pixi, Linux remote có `requirements/remote-cu128.txt`, `resources/runtime.json` và `scripts/remote/setup.sh`; direct pins không phải full transitive environment lock hoặc chứng nhận mọi GPU đã chạy.

```bash
python -m pip install -e . --no-deps --no-build-isolation
python -m hope_cad --help
python -m hope_cad.local_tests
```

Những lệnh này chỉ hợp lệ trong môi trường đã chuẩn bị dependencies. V7 CLI gồm `preflight`, `train`, `evaluate`, `matrix`, `aggregate`, `export`, `source`. `preflight` thật bind source/runtime/data/checkpoint/device; `--local-smoke` không mở benchmark GPU. `train` giữ stream state; `evaluate` đọc final state; `matrix` đánh giá các learned-history cells còn thiếu từ task states, không train lại. Các commands đã có cho **baseline v5.1**, chưa có job H32 trong registered study.

Runtime mới phải fresh cohort. Không dùng old T4 checkpoint vào current v7 bằng cách bypass source/producer locks. Chạy SSH trong tmux hoặc process manager, giữ run-directory trên máy/server; export và kiểm copy SHA trước stop/destroy. Source không tự thuê GPU, gọi SSH, upload Drive hoặc setup W&B. Source/CLI resources có thể import từ wheel, nhưng execution/shipping cần verified source checkout; kho context/sibling repo không runtime dependency.

### D3. Điều đã kiểm khi tổ chức lại code

Review refactor ghi:

- Local và clean extracted source: **125 tests +45 subtests mỗi suite** PASS.
- **Tám modes × 40 reduced CPU images**, interrupt tại 13/resume: initialization, events, final state và raw scores khớp exact với bản trước refactor.
- **16 core modules** giữ function/class AST ngoại trừ import relocation/static helper reference.
- Wheel build/install isolated target, resource JSON/YAML/CLI help, Linux bash syntax và full-grid synthetic CPU smoke có kiểm.
- Publication fix đảm bảo sáu data modules được track và study JSON không newline drift;137 tracked source paths ở baseline HEAD khớp working bytes trong lần review.

Đây là engineering evidence có phạm vi, không full-stream GPU equivalence proof. Các tests của tiền thân đã bỏ không cộng vào count hiện hành. Không gọi 125 tests bằng mọi test từng tồn tại trong lịch sử. Kết quả model/benchmark hiện hành không được làm mới chỉ bằng cách đổi tên package.

### D4. Baseline chính thức và mốc quay lại

`hope_cad_v7@89df9a21658b265731f98a7c2014afb1876f1924` là source baseline chính thức trước candidate nâng cấp. Các sửa sau đó phải giữ lineage và tách method/execution identity rõ. Nếu candidate chưa được chấp nhận, giữ baseline v5.1; rollback source không xóa logs/artifacts hoặc kết quả âm, không hứa khôi phục môi trường chỉ bằng Git checkout.

File này là báo cáo bàn giao thêm vào root repo, không scientific module hoặc raw artifact. Commit baseline được chỉ theo hash bất biến; commit chứa chính báo cáo này về sau có thể có hash khác. Không nhầm branch tip mới với baseline đã chọn. Nội dung thuật toán/spec trong report phải được đối chiếu với normative contracts code nếu implementation thay đổi sau ngày này.

### D5. Những thông tin cần trao đổi để hợp nhất kết quả nhóm

Đối với mỗi run CADIC/ReplayCAD/ContCore phía Khang muốn đưa vào bảng chung, gửi tối thiểu: commit/source identity, resolved method + protocol, dataset train/test manifest, checkpoint provenance, K/total memory, task order/seed, metric definitions, full final/per-category metrics, nếu có thì task matrix, timing/resource scope. Không yêu cầu hai methods khác architecture phải ép cùng backbone để thành native paper reproduction; phân biệt rõ **native reproduction** và **matched adaptation**.

Tài liệu này không đoán ý định hoặc phân công từ tên experiments. Nếu Khang chỉ đang thử thêm ngoài baseline, chúng được giữ exploratory; nếu chọn ContCore thành baseline chính thức hoặc NPIM/CSG thành phương pháp mới thì cần khai báo tên/recipe/evidence tương ứng. Không đổi nhãn CADIC đã chạy để bao gồm các component mới mà không tạo revision.

**Mốc hiện tại để cả nhóm cùng hiểu:** đã có một phương pháp NESTED hoạt động, giữ image AUROC gần matched CADIC và có pixel AP/retention gain ở một seed; đã có source v7 tối giản để phát triển tiếp. Việc tiếp theo là attribution và candidate lịch sử có budget, rồi mới xác nhận nhiều seed/dataset. Những phần chưa chạy được ghi rõ, không cộng thành thành quả đã có.

### D6. Disposition các điểm pipeline đã audit

Bỏ30ảnh là nợprotocol cần loại trong cohort mới, không lỗi/leakage làm vô hiệu matchedresults cũ. Recipe ảnh được giữ có lý do, chưa đo native impact. Route image-vs-batch và actual checkpoint là yêu cầu provenance/fixture. SameK không equalmemory; seed/order/timing/numericparity là các giới hạn phải báo và đo, không tự tăngcomponent/rate/K để gọi đã fix. Thứ tự là fulltrainbaseline+route/checkpoint locks → candidate và controls trên cùng cohort → review memory/timing/seeds → riêng VisAfreeze. Không gộp paperparity với fulltrainparity.


## Vận hành Pixi và assets trong checkout

Luồng mới dùng duy nhất `pixi install --locked` để cài Python/dependencies/projecteditable theo platformlock. `pixi run assets` tải/kiểm MVTec và ViT; data/weights/runs/records/logs/results nằm trong repo và ignore, không vào sourcehash/Git. Linux vẫn giữ scientificpinsPython3.13/torch2.11+cu128; Windows localgiữstackkhác. Đây là thay đổi provisioning/đónggói, không thuật toán/gate/ADgain. Chưa actualLinux/5090acceptance; usercommit/pushsource mới trước freshpreflight, không pin predecessor720b3cd/672abd1 rồi mong cótasks mới. Đọc [runbook](docs/operations.md) và [research handoff](docs/research/implementation.md).
