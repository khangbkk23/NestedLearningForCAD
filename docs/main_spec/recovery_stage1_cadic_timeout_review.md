# Recovery preflight timeout: audit và exact CADIC execution r2

Ngày2026-10-16; scope duy nhất `hope_cad_v5`. Worktree `hope_cad_v2_2` và W6/M7.1/M7.2 giữ nguyên. Không commit/push trong lượt này.

## A. Artifact đã xác minh

Input `docs/recovery_stage1_kaggle/preflight/hope_cad_recovery_stage1_evidence.zip`,365020539bytes, SHA256 `ff034113ab9f846054585f6b02eb938a1b14bdbd5e0781e624fc9e39962152f5`. Verified exact inventory và148entry SHA256s. Machine audit: `results/recovery_stage1/preflight_timeout_review.json`.

130frozen tests và27recovery tests PASS trênT4; runtime/source/pinned parents khớp. M/C1/A1 plain và interrupted/resumed histories đều event_cursor9, task counts6+3. Resource worker chỉ commit8normal images,6272vectors, accepted6272/replaced0/rejected0. Batch2 chưa hoàn tất trước deadline. Không suy ra chỉ xử lý8ảnh trong toàn600s: report commit theo batch, không ghi được intermediate work của batch2.

Worker start322.2s, timeout922.5s; ZIP export945.7s; version kết thúc953.6s. Đây là resource failure, không phải AD score, NaN, OOM hoặc chứng cứ HOPE performance xấu hơn. Preflight chưa PASS, không cho controls dùng receipt này.

## B. Nguyên nhân ở implementation

Frozen `models/cadic_paperfaithful/cadic_patch_coreset.py:update` chọn farthest candidate theo nearest-bank distance, so với global closest pair, thay member đầu của pair và lặp. `_closest_pair` quét mọi upper tile256x256 và đồng bộ CPU/GPU theo từng tile. Khi bank10k đã đầy, nhiều replacements lặp lại pair search mà phần lớn pair distances không đổi.

Đây là chi phí execution; không sửa K, feature layer, input geometry, update unit image, candidate order, FP32 distance primitive hoặc Eq9 để làm baseline nhanh hơn.

## C. Tối ưu và lập luận tương đương

Thực thi ở `notebook_support/recovery_stage1/cadic_accel.py:CachedPairCoreset`;119frozen dependency bytes không sửa.

Chia bank C thành đúng các tiles reference, cùng start/shape và `euclidean_distance_mm`. Với mỗi upper tile(a,b), lưu minimum value và lexicographic global pair key `i*K+j`. Minimum trong tile dùng flatten-first tie như reference; minimum toàn cục chọn value rồi key nhỏ nhất. Diagonal masking loại i>=j đúng như reference.

Nếu row r thay, tile không chứa r trong bất kỳ trục nào nhận đúng tensors cũ, nên output distance/minimum cũ vẫn đúng. Chỉ invalid các tiles có block(r) ở một trong hai trục. Dirty tiles được recompute bằng đúng reference primitive/slices/shapes. Quy nạp từ initial full cache cho thấy global closest pair giống reference ở mỗi iteration; `_nearest`, farthest selection, comparison, replacement/candidate removal/counters giữ nguyên, nên cùng inputs dẫn tới cùng bank. Đây là conditional exact execution argument, không phải proof CADIC vượt HOPE hoặc proof từ normal loss sang anomaly ranking.

Không đổi GEMM batching/shapes, không approximate NN, không FP16/TF32, không epsilon tie-breaking. Cache chỉ giữ minima/pair keys/mask/layout; không giữ thêm raw support vectors. Nó không phải cache full10000x10000 matrix. Nếu feature tensor/version đổi ngoài expected replacement hoặc load checkpoint, rebuild cache. Checkpoint vẫn chỉ scientific state; resumed cache rebuild từ bank.

Cache là auxiliary compute memory có cost thật: khai current/peak tensor payload và Python layout bytes trong profile/memory accounting. Không claim memory-budget equality của full execution chỉ từ raw bank count. Fingerprint mới `cadic_exact_pair_tile_cache_v1` vào CADIC method metadata/run identity; không giả nhận original reference execution digest. Full W7 cần verified bridge nếu tái dùng artifacts. HOPE dynamics và identities không đổi.

## D. Kiểm chứng trước sử dụng

Tests CPU/CUDA so reference từng update: features/counters/decisions exact, Eq9 image/pixel scores exact; random streams, 2D/3D inputs, ties/duplicates/FP32 cancellation, singleton, tail tile, selective invalidation, external mutation, checkpoint/resume. Adapter memory accounting và source-revision rejection được kiểm. Bộ hiện hành37tests; receipt sau final build phải cùng operations source digest.

Local K10000,D768 trên RTX3050Ti, features giả: cold pair reference0.519s/cached0.562s;4single-vector replacement updates reference0.586–0.703s, cached0.038–0.050s, bank/counters exact. Không lấy tỷ lệ này làm T4/full-stream ETA hoặc accuracy claim. Receipts `cadic_cache_local_profile.json` và `cadic_cache_16_synthetic_profile.json` có scope rõ.

Preflight r2 vẫn watchdog600s, chạy16normal images thực qua cached backend. Sau đó trên cloned full bank, so reference và cache về closest pair,4normal candidate operator updates và Eq9. Main bank/RNG read-only. Không dual-run toàn16ảnh với reference chậm rồi tuyên bố đã exhaustive parity. Supporting receipt kiểm backend/parity/count/budget/dim/update unit trước controls; thiếu parity phải fail. Exact argument + CPU/CUDA regressions + T4 real-normal operator check cùng tạo engineering gate.

## E. Bước tiếp theo

Import preflight notebook mới, source r2 + original W6 + completed7.2 + MVTec + ViT, không input failed recovery ZIP. Không chạy W6 lại. Nếu r2 PASS mới chạy controls notebook mới, thay7.2 bằng r2 preflight output. Nếu fail, giữ ZIP/log để review. Không tăng timeout tự động, không đổi scientific method hoặc chạy VisA.

Chi tiết copy/paste thao tác trong `recovery_stage1_handoff.md` mục10. Original delivery đã backup trước sửa. Mọi kết quả cũ giữ nguyên, chưa có matched CADIC/C1/A1 final metrics từ gói này.
