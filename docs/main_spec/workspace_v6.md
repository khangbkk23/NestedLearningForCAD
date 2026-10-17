# V6 integration and operational contract

Ngày17/10/2026. Đây là thay đổi tổ chức/vận hành, **không đổi method v5** `hope_cad_v5_causal_support_controller`. Đọc full architecture/instruction/planning v5 và JSON contract; các phương trình, hyperparameters, quyền dữ liệu, controls và pilot gates giữ nguyên.

## Nguồn và Git

Điểm xuất phát là `origin/dikhang_space@b9859440832422af20890aed9dc479d8c0e3b1d8`; tích hợp `hope_cad_v5@9e86478b150ec1ca100ffa20918e91de6a89de71`. Khang có5commits từ10/10: a3699a3,afac073,7ef2be0,d89f2f1,b985944. Common ancestor a89fd48,66paths phía Khang/249paths phía v5 không giao nhau; merge không conflict nhưng vẫn phải kiểm dependency/tests.

Các experiments Khang mới được bảo toàn trong Git/source backup rồi loại khỏi checkout sản phẩm nếu không thuộc method hiện hành. Downloader MVTec giữ bản Khang. ReplayCAD gitlink6d09dc3 không có mapping cũ và khác official HEAD acc6195 đã quan sát; `refs/replaycad.json` lưu cả hai. V6 dùng explicit external REPLAYCAD_ROOT, không tự nhận hai SHA tương đương hoặc đổi baseline của Khang.

Merge được chuẩn bị để người dùng review/commit; không tự push hoặc xóa branch/checkouts cũ. V5 là legacy dự phòng. Git chung vẫn ở primary repository, không được xóa theo tên thư mục.

## Dependency và nguồn chuẩn hiện tại

Các docs v5 ban đầu mô tả delivery266files/242r4 làm lịch sử Stage2. Chỉ dẫn vận hành trong tài liệu này và README là hiện hành cho v6:

| Trước | V6 |
|---|---|
| recovery_stage1.cadic_accel | models.cadic_paperfaithful.cache |
| recovery notebook stream helper | notebook_support.common.process |
| W7 evidence transport/W6 import dependency | notebook_support.common.evidence, không import W6 |
| hope_runtime_v3/v4 và W7 lock dependency | training.runtime, function bodies cần dùng giữ nguyên |
| source builder phải đọc ZIP r4 | explicit conf/delivery/support_source.json, build trực tiếp từ current files |
| ultils chứa bundles/receipts | shared .agents/artifacts/deliveries |
| docs chứa run ZIP/log/review | shared artifacts và evidence reviews |

Visual/CMS primitives có namespace v2 vẫn là dependency hoạt động; independent oracle phải giữ. `conf/provenance/algorithm_invariance.json` đối chiếu với v5, không tạo expected fingerprint từ bản đã sửa để che drift. Source schema v2 tách identity/layout mới; old source-v1 pilot không resume qua v6. Evidence schema v5 vẫn cùng phạm vi normal pilot, source digest guard phân biệt delivery.

## Chạy và artifacts

`pixi install --locked`, `pixi run test`, `pixi run package`, `pixi run verify`. Package/receipts vào `D:\src\DATN\.agents\artifacts\deliveries\hope_support_v5_v6_r4` mặc định; override bằng HOPE_ARTIFACTS_ROOT. Source ZIP mới `hope_cad_support_v5_source_v6_r4.zip` dùng với notebook hiện hành.

Ba inputs Kaggle: source mới, MVTec AD, frozen ViT-B/8 `vit_base_patch8_224.augreg_in21k.safetensors` SHA2c4839589adbbf701573dd155e1814f8adb145cd4d235bb86a93269369e111ec. Không W6/recovery learned weights. Default pilot có preflight24 normals discarded clone và4fresh jobs×256normals, seed0,một trial. Save Version/Save & Run All; soft budget3h có resume, không ETA. Lưu outputZIP/log vào shared artifacts/runs/<run-id>, review status/gates rồi mới bàn benchmark.

Local Pixi giữ numerical versions đã kiểm; Pillow12.2.0 dùng PyPI wheel thay Conda build để khép DLL import collision trên Windows. Canonical Kaggle runtime lock không đổi. Kiểm pixel bilinear và exact state/events/scores tám modes so với v5; đó là engineering parity, không AD gain.

## Lịch sử và relocation

`.agents/artifacts/index.json` ghi canonical objects; relocation-map ánh xạ đường dẫn cũ sang payload có SHA256. Mỗi file chỉ bỏ bản nguồn sau copy/hash verification. Original register/reviews/ZIP bytes giữ nguyên; validator resolve aliases thay vì rewrite receipts cũ. Các source backups và Git bundle ở artifacts/maintenance/20261017-v6-integration. Dữ liệu và Pixi environments không được chuyển vào evidence store.

V6 docs chỉ giữ current specs/papers. Đọc nguồn lịch sử từ `.agents/snapshots`, archived source ZIPs hoặc branches v5/v2_2; không khôi phục toàn bộ clutter vào checkout để đọc context. Phần B chưa thực hiện.

## Acceptance và pending

Phải có: tests hiện hành, independent oracle/meta-gradients, baseline/Eq9/cache, causal permissions, rollback/resume, packaged ZIP và extracted bootstrap, exact cross-checkout parity, archive relocation integrity và context/live validation. Report ghi counts/receipts thực, không phần trăm hoàn thành.

Chưa có T4 real-normal pilot v5, AD benchmark v5 hoặc VisA confirmation. Full v5 multi-seed orchestration/matrices/uncertainty/equal-total-memory control còn tiếp tục sau pilot. Không sửa ngưỡng để cứu gate, không tune bằng anomaly/VisA test, không claim max NL capability hoặc bảo đảm thắng CADIC.
