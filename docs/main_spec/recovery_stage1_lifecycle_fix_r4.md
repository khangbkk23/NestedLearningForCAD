# Recovery r3 review và lifecycle-r4

Ngày2026-10-16. Chỉ sửa `hope_cad_v5`, không commit/push hoặc thay sibling/frozen119dependencies. Machine review `results/recovery_stage1/preflight_r3_review.json`; raw regression trước sửa `r4_lifecycle_before_fix.txt` cùng thư mục results.

## A. Kết quả thật của run mới

ZIP365041328bytes, SHA256 `db16e48ab160b44811ecf602bfad098998783e42e8c61f69a6047d71bcaac9bb`;153entries verified byte hashes/inventory. RuntimeT4/CUDA12.8;130frozen và40recovery tests PASS. R3 source ops digest `413eda33d9b798fa728a1e7fbbc7855ba783631a987b317c1e4781d7f3954e78` khớp source đã bàn giao.

Resource đã chạy16normal images/12544patches, bank10000x768, accepted10000/replaced2141/rejected403, update_batches16. Training elapsed122.4015s, batch2 update121.3335s. Binding actual registry/shared bank true; cache2144closest-pair calls/86460tile evaluations/1fullrebuild. Reference cloned full-bank pair/4normal candidates/features/counters/Eq9 scores đều exact; score read-only/state-RNG và checkpoint roundtrip đạt. Cache tensor75376bytes, peak auxiliary accounting258980bytes theo receipt. Không extrapolate full-stream ETA từ16images.

Run fail SAU các phép kiểm này: `cadic_profile.run` đã ghi functional passed fields rồi `finally:cadic.close()` ném AttributeError. Main ghi resource.status=failed và subprocess exit1; outer runner fail-preserve-evidence. Đây không là timeout mới, NaN hoặc AD result. Global preflight vẫn KHÔNG accepted; không sửa lại status gốc để cho qua.

## B. Root cause

Frozen concrete `CADICBenchmarkAdapterV1` không định nghĩa close(). Extractor `CADICViTFeatureExtractor` mới là chủ forward hook và có close(). Cached wrapper r3 thiếu lifecycle method, nên `CompactCADIC.__getattr__ -> CachedCADICAdapter.__getattr__ -> RegisteredAdapter.__getattr__` chuyển lookup đến concrete adapter không có method. Cùng lỗi tồn tại ở finally của runner controls, không chỉ resource worker.

Đây là phần lifecycle tôi bỏ sót khi bàn giao. Unit parity và registry state tests trước đó không chạy tới finally của actual worker/control runner, nên40tests không bắt được. Không quy lỗi cho input người dùng/Kaggle.

## C. Sửa và acceptance

`CachedCADICAdapter.close()` explicit và idempotent: gọi concrete close nếu có, nếu không gọi extractor.close để gỡ ViT hook. Chỉ đánh dấu closed sau cleanup thành công; exception cleanup vẫn được báo, có thể retry. Không đổi bank/weights/RNG hoặc pair-cache math. Resource report chỉ thêm cleanup_complete sau finally thành công; parent còn yêu cầu worker exit0. Supporting gate không chấp nhận thiếu cleanup marker. Control runner ghi lifecycle_completion.json sau close; chỉ hoàn tất khi có marker này. Marker ràng buộc run digest.

New source/notebook revision `lifecycle-r4`; scientific backend vẫn `cadic_exact_pair_tile_cache_v2`. Existing source/protocol locks không được rebind; bootstrap từ chối evidence có source digest khác ngay trước restore, không để chạy lâu rồi mới fail lock. R4 cần fresh preflight với original W6/completed7.2, không import r3 failed như learned resume.

## D. Verification trọn chu trình

4tests lifecycle tái hiện lỗi trước sửa (4errors), rồi PASS sau sửa:

1. Compact/registered adapter close đúng extractor một lần, repeat close không làm đổi bank.
2. Cleanup error không bị swallow; retry sau failure hoạt động.
3. Actual registered adapter + observer + W7Engine train3tasks, save states, final row, profile/storage, write completion và close; lifecycle marker tồn tại.
4. Actual resource worker CLI chạy trong subprocess, actual registry/adapter/cache/Compact/state/scoring/parity/finally. Dùng16RGBnormal fixtures, bank10k, deterministic extractor4D thay ViT để kiểm lifecycle, worker exit0 và resource.status=passed/cleanup_complete true. Không gọi đây là canonical768D/T4 benchmark.

Thêm notebook positive success/export/restore test cho cả preflight/controls. Cùng40tests cũ tạo45tests. Unit fixtures được đánh dấu fixture, không là research AD evidence. Kiểm pack standalone subprocess/lifecycle và source hashes trước bàn giao. R3 đã có T4 actual768D algorithm/parity evidence; R4 T4 cleanup/global preflight vẫn pending, không claim guarantee mọi future run.

## E. Dùng gói mới

Import preflight mới, source_r4.zip + W6 + completed7.2 + MVTec + ViT (5inputs). Không gắn failed outputs, không sửa device thànhCPU, không nới watchdog hoặc rerunW6. Save Version / Save & Run All; sau PASS tải evidence để review trước controls. Rename r3 archive trước khi lưu ZIP mới, giữ lịch sử negative/failure.

Actual cost r3 đến handoff482.26s, riêng resource train122.40s. Đây là số đo của lần đã đọc, không là bảo đảm elapsed của mọi Kaggle session hoặc full matched baseline. Không chọn method/hyperparameters bằng anomaly metrics trong lượt sửa này.
