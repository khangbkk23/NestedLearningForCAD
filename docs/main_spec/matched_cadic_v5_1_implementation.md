# Matched CADIC seed 0 — implementation và bàn giao

Đặc tả đầy đủ: `matched_cadic_v5_1.md`. Lớp vận hành riêng trong `notebook_support/matched_cadic_v5_1/{common,entry,worker,build}.py`. Dùng nguyên frozen engine/SupportSessionCADIC và scoped backend NN tốiưu. Không nhập learned state của NESTED hoặc CADIC cost. Metadata source gồm full trainmanifest và **hash từng raw ViT feature** đã đo của NESTED; features phải khớp trước khi vào CADIC. Testmanifest phải khớp sau traincomplete. Source base235entries giữ byte-exact.

## Chạy lần đầu

1. Import `notebooks/hope_cad_v5_1_matched_cadic_seed0_kaggle.ipynb` mới.
2. Attach đúng **ba inputs**: source `hope_cad_matched_cadic_seed0_source_r1.zip` mới (hoặc Kaggle giải nén), MVTec AD đang dùng, checkpoint `vit_base_patch8_224.augreg_in21k.safetensors` đang dùng. Tháo source/evidence NESTED, bridge, pilot/W6 hoặc CADICcost. Không cần chạy notebook preflight riêng.
3. GPU **T4×2**, kernel/session mới, giữ `PHASE='7.1'`, `JOB='CADIC_seed0'`, `SESSION_HOURS=9.0`. Dùng **Save Version → Save & Run All**.9h là softbudget gồm600s exportreserve, khôngETA. Kernel runtime phải khớp NESTED; không tự upgrade packages hoặc chuyểncard.
4. Notebook verify source/runtime/backbone; stage/hash3629trainbytes để tái tạo đúng manifest; probe2normals trên discardedowners; freshCADIC học3599ảnh rồi evaluate15finalcells. Cóheartbeat30s/progress8ảnh. Không cần rawfeaturecache, W&B hoặc Drive.
5. Giữ ZIP và Kagglelog sau phiên. `7.1_complete_pending_review` là đã chạyxong và cầnreview; `paused_at_atomic_boundary` là chưaxong; `failed_preserve_evidence` cầnđiềutra, khôngtựretry.

## Resume

Thêm **latest matchedCADIC output** làm inputthứ4, dùng cùngsource/notebook/MVTec/ViT. Tháo cácphiênCADICcũ khác; giữRESUME_OVERRIDENone nếu discovery rõ. Không attach NESTED output/costclone. Mỗi phiên workdir/kernel mới. Logs/history append, operation/source/runtime/job locks phải khớp. Engine xác minh active/eventtrace rồi nối đúngcursor, không học lại prefix; traincomplete thì chỉverify, evaluation chunks resume readonly.15taskcheckpoints được giữ cho7.2 sau này; notebook hiện chỉ7.1.

## Files bàn giao và nơi lưu kết quả

- Source: `D:/src/DATN/.agents/artifacts/deliveries/hope_support_v5_1_matched_cadic_seed0_r1/hope_cad_matched_cadic_seed0_source_r1.zip`.
- Notebook: `D:/src/DATN/hope_cad_v6/notebooks/hope_cad_v5_1_matched_cadic_seed0_kaggle.ipynb`.
- Output: `hope_cad_matched_cadic_seed0_evidence.zip`, metrics tại `benchmark/runs/CADIC_seed0/metrics/`.
- ZIP/log phiên đầu: `D:/src/DATN/.agents/artifacts/runs/support_v5_1_matched_CADIC_seed0/session_01/`; phiênsau session_02,03… Khôngđèfilephiêncũ.

## Trạng thái bằng chứng

Implementation/local verification chỉ được xác nhận bằng receipt/review của delivery sau khi chạy kiểm thử. Source hash mô tả đúng dirty source; HEAD không đủ. Canonical full CADIC seed0 **chưa chạy** khi viết tài liệu này. Preflight2normals không là fullbank hoặc AD acceptance; fullbank backend parity/speed đã có reviewed CADICcost riêng. Matched endpoint phải so protocol/assets/source/featurehashes/metrics và artifactcoverage sau Kaggle. Không dùng timing16ảnh suy ETAtoànstream, không dùng mộtseed kếtluậnfullstudy/CI. Không có VisA authorization.
