# Nested Learning for CAD — continuation 7.1 sau G4

Ngày 21/10/2026. Đây là đặc tả lớp vận hành mới, không là revision thuật toán. Giữ nguyên full benchmark study v1, method v5.1, source bridge r2 và toàn bộ dữ liệu/evidence cũ. Không retune từ anomaly metrics. Chỉ expose `NESTED_seed0`, phase `7.1`; CADIC và các seed/control khác vẫn là jobs độc lập, chưa được gói này tự chạy.

## Danh tính và nguồn chuẩn

Source lõi là nguyên 222 entries của ZIP bridge r2 (221 payloads và bridge_manifest.json), SHA256 `244e49b892fa8bc691290b7f97fbc4cfcb8ad370fc62ed312ca20c7e39b3cb59`. File map lõi có digest `07b2b53d2927332feac56c6b86ddea2f69f1cbdeb23a5d0153896684450a06ee`. Giữ chính xác mọi byte. Gói continuation bổ sung namespace riêng `continuation_runtime_v1`, docs, tests và notebook; manifest riêng khóa cả core và phần bổ sung. Không ghi đè engine, model, loader, task resolver hoặc guard gốc. `Execution` vẫn kiểm source r2/new execution G4; receipt orchestration riêng ràng buộc manifest continuation. Không gọi source mới là source r2 thuần túy hoặc bỏ guard vì hai source digests khác nhau.

Lần đầu chỉ nhận bridge đã review: manifest SHA `05350f6f35cc2cbfa4e79f3fd906889229898f9d451e792564621569fa7a0993`, ZIP SHA `8d173d666f82fd03ab7c2a10cef6d63a1243a9b9c5a8ca117c08b94297e0b3ad`. Checkpoint384 SHA `337aaf57025cf74f60a8f669053e24ced6d47125dcad614eb3b2678a608221df`. Review/audit G4 được đóng kèm và hash-bound. Kernel mới, T4 canonical/runtime exact; không upgrade packages hoặc chuyển GPU bằng cách bỏ lock.

## Học và đánh giá

Tiếp tục384→3599, không replay học lại384ảnh đã chốt. Resume hard-crash có thể replay tối đa7events theo engine frozen và kiểm exact records; không reset learner/pending/partial/clock/anchors. Giữ original task00 ở legacy owner; các task còn lại ở new G4 job owner. Không finalize tại pause/task boundary; true end3599 mới censor pending theo method.

Dataset archive được copy những normal train/good bytes trong registered rows sang temporary assets bằng một lượt giải nén; đây là staging bytes, không trích features hoặc cho learner thấy future rows. Có receipt ghi quyền này. NormalFeatures vẫn đọc/hash/trích từng ảnh theo causal order. Không test/GT features hoặc metrics trước train_complete. Sau train_complete mới stage official test/GT, tạo manifest1725ảnh bằng frozen evaluator, kiểm hashes/QA và tính15final cells. Không task matrix hoặc VisA trong interface này. Scoring readonly, Eq9/masks/interpolation/metrics nguyên bản. State/rows/order/source/runtime sai thì fail closed.

## Resume và export

Mỗi phiên dùng work directory sạch. Lần đầu attach reviewed bridge output; lần sau **tháo bridge/output cũ**, attach latest continuation output cùng source và assets. Discovery phân biệt schema gốc, bỏ nested legacy manifest; nếu thấy nhiều phiên khác nhau thì yêu cầu override, không tự chọn cursor cao nhất. ZIP và Kaggle-extracted directory đều hợp lệ nếu full manifest khớp. Partial failure trước khi restore/receipt hoàn tất không là resumable run; giữ evidence để review. Chỉ schema continuation có handoff `paused_at_atomic_boundary` hoặc `7.1_complete_pending_review` được resume tự động, với orchestration lock và immutable bridge publication hợp lệ. Failed run không tự retry.

Soft SESSION_HOURS mặc định9h, hợp lệ0.1–11h, reserve600s cho export. Nếu budget không đủ reserve thì fail trước train. Checkpoint mỗi8ảnh/task boundary/pause. Deadline kiểm ở atomic train hoặc evaluation chunk boundaries; một event dài có thể vượt soft deadline. Platform hard kill có thể ngăn export; không bảo đảm tuyệt đối. Heartbeat30s, progress mỗi8ảnh, log subprocess unbuffered. Mỗi lần luôn export verified ZIP cả khi fail; lỗi sau export vẫn raise. Receipt timing tách restore/bootstrap, worker, export; active elapsed của G4 kế thừa cách tính cũ, không coi là full cold-start notebook wall time. Không dùng estimated ETA làm learning early stop.

Giữ `benchmark/` toàn bộ immutable legacy/source/bridge files và appended states/events/eval chunks. Root `origin/` giữ bridge manifest + handoff gốc; `continuation_lock.json` khóa orchestration identity; `session_history/` chỉ append, không overwrite prior session logs/receipts. Không tự đổi status failed thành pass. Complete chỉ khi train_complete và final_complete15cells tồn tại và worker return0. Mọi AD gain cần artifacts review, matched CADIC và study coverage; `reportable=false` không tự nâng.

## Kiểm thử và bàn giao

Cần kiểm ZIP/directory discovery, same-manifest duplicates, ambiguous/failed/tampered parents, source core byte parity, operational lock drift, safe tar + no test before training, deadline, failure export, notebook bootstrap thật ZIP/directory; interrupted/resumed training và inherited task00 + final15cells qua Execution synthetic. Local CPU tests không là canonical T4 acceptance. Không yêu cầu chạy lại W6/pilot/G3/bridge. Hướng dẫn inputs/output và trạng thái thực trong `continuation_7_1_v5_1_implementation.md`.
