# Nested Learning for CAD — bàn giao W1–W6 theo spec v3

Ngày cập nhật: 2026-10-13. Worktree: `D:/src/DATN/hope_cad_v2_2`, branch `hope_cad_v2_2`, base commit `d78faa69c066094deb9bb9decc59df84b5d470db`. Các thay đổi trong đợt này chưa commit/push/merge. Source bundle chứa cả file mới và có manifest SHA256; base commit một mình không đại diện cho implementation mới.

## 1. Phạm vi và trạng thái nghiệm thu

Giữ nguyên ba spec v3, mathematical method ID, recurrence W3, objective CMS, exact 16-support unroll, masks, teacher, scoring, periods, finite search và thresholds của gate. Đây là sửa implementation/integration cho đúng contract đã chốt, không là method revision sau khi nhìn anomaly-test metrics.

| Phần | Implementation/evidence hiện có | Điều kiện còn thiếu |
|---|---|---|
| W1 | Protocol/roles, explicit CADIC route, full Eq.9 và locks có tests; normal-train manifest local đã được byte-verify | Không có matched CADIC full-run mới trong lần bàn giao này. Kết quả cũ của Khang giữ scope riêng; cần matched benchmark artifacts ở W7 |
| W2 | Notebook bootstrap/CLI, source/runtime locks, cache invalidation, real RGB preflight, atomic export và session resume đã triển khai | Chạy clean-session Kaggle với assets thật và tải evidence về |
| W3 | Core numerical implementation giữ nguyên; các gates numerical/upstream/T4 lịch sử giữ nguyên scope | Không suy ra CAD utility từ correctness tests |
| W4 | Search/fit/stress resume, candidate selection, C4 kế thừa β_M, selected-initialization receipt | Real-prefix selection và 1024-event traces của source mới phải chạy trên GPU đủ bộ nhớ |
| W5 | RGB adapter và static controls đã sửa; synthetic full-RGB resume qua event 8 có tests | Real RGB/cache parity và resume preflight trên runtime Kaggle |
| W6 | Calibration diagnostics, probe, trained/static/teacher-mean controls, score/history/visual interventions và resource logs đã nối vào pipeline | Chưa nghiệm thu khoa học; phải review selected checkpoints, gate traces và mechanism evidence thật |
| W7 | Không tự chạy trong notebook này | Sau W6 review: matched baselines, locked-recipe refit seeds 1/2, full task matrix, image-level predictions/paired bootstrap và uncertainty vẫn là công việc tiếp theo |

Các suite lịch sử 74/74 và 77/77 được giữ ở `results/hope_cad_v2/w1_w6_local/verification.json` và `preflight_revision_verification.json`; không dùng chúng để chứng nhận source mới. Final suite cho revision sau audit xem `results/hope_cad_v2/pre_run_fixes/verification.json`. Regression mới kiểm tra numerical failure được lưu trước khi search trả về, interruption/resume không thử lại candidate đã fail, exact optimizer/RNG disk resume và thời gian bootstrap được trừ khỏi budget. Các tests dùng numerical/protocol/integration fixtures và actual notebook bootstrap (ZIP/extracted source, hash-drift rejection); không thay thế experiments trên MVTec. `--synthetic-smoke` không xuất recipe đủ điều kiện cho scientific runner. Nghiệm thu W6 không được suy ra chỉ từ `all_methods_selected` hoặc chạy hết notebook cells.

**Real-data smoke đã PASS local:** producer trích xuất 85 ảnh (50 support + 20 query + 15 validation), không trích probe/test; cache khoảng 1.024 GB. `real_rgb_preflight/preflight.json` xác nhận 2 ảnh × 5 views có maximum absolute error bằng 0, exact state/RNG/score resume ở 9 events, commits `(9,1)`, ngắt sau event 5. Peak allocated 471,757,312 bytes, peak reserved 559,939,584 bytes trên RTX 3050 Ti 4 GiB; 108.65 s gồm byte verification/loading/preflight. Host peak local chưa có số đo (`null`). Đây là deployment smoke với random initialization, **không** là canonical outer-calibration resource profile. Local Pixi dùng torch 2.12.0+cu126, timm 1.0.30, safetensors 0.7.0; Kaggle có runtime riêng và phải chạy lại parity/gates. Lỗi import DLL Windows ban đầu được xử lý bằng đúng môi trường Pixi và thứ tự import của runner, không sửa các dependency/spec của nhánh chính.

## 2. Đối chiếu các lỗi audit và implementation

| Lỗi/khoảng trống | File/function xử lý | Evidence/giới hạn |
|---|---|---|
| C0/C1 vẫn update CMS online | `training/hope_cms_session_v2.py: CMSSession.update/restore` | Stream cursor tiến theo ảnh; CMS clock/weights/accumulators/commits giữ nguyên. Tests qua task boundary và resume |
| Cache có identity nhưng không enforce | `training/hope_feature_contract_v3.py: load_context/validate_bundle`, `prepare_prefix_features.py` | So actual baseline lock/config/extractor/backbone/preprocess/mask/source/roles; exact allowed identities; tensor shape/dtype/hash/unit teacher. Cache cũ không tương thích phải rebuild |
| Source/runtime drift khi resume | `training/hope_runtime_v3.py`, `run_w1_w6.py`, `calibrate_features.py`, `benchmark_adapter.py`, `run_hope_v3.py` | Resume cùng pipeline/search/session yêu cầu source/runtime/data/recipe identity bằng nhau; mismatch fail trước cập nhật |
| Random initialization được dùng như selected model | `training/hope_initialization_v3.py: selection_receipt/validate_initialization` | CLI scientific cần `selection.pt` gốc và selected recipe; động cần gate 1024 events, static cần ordinary-fit recipe. Canonical export phải reset clock/cursor/accumulators |
| Atomic save và full-search resume thiếu | `hope_runtime_v3.atomic_torch_save`, `hope_search_journal_v3.SearchJournal`, `hope_benchmark_engine_v3` | Temporary file cùng filesystem rồi replace; optimizer/RNG/cursor/partial accumulator được giữ. Completed/failed candidates không chạy lại để tìm kết quả tốt hơn |
| W6 thiếu diagnostics/probe | `calibrate_features.py`, `run_mechanism.py`, `hope_mechanism_v3.py` | Real canonical calibration ghi steps 0/1/5/25 và selected; probe chỉ sau selection. Tests xác nhận diagnostics giữ parameters/RNG và controls static không adapt |
| Precision/dependencies chưa thống nhất | `hope_runtime_v3.configure_runtime/runtime_identity`, notebook | Tắt TF32/AMP ở reference, ghi package/GPU/runtime; notebook enforce đúng pins thay vì chỉ install khi thiếu |
| C0/C1 summary gây hiểu nhầm | `calibrate_features.py` | `selected_beta=null`, ghi `selected_outer_lr`; β nội bộ của static control chỉ là placeholder không dùng online |
| Non-finite visual exception thoát khỏi candidate failure bookkeeping | `visual_memory._finite`, `HOPECADVisualBlock.forward`, `fit`, `run_stress_gate`, `search`, `derive_a1` | Numerical failure phát `FloatingPointError`; lỗi shape/config vẫn `ValueError`. Failure callback ghi candidate/selection atomically trước khi trả về search, kể cả A1. Không thay phép tính recurrence hoặc thresholds |
| Preflight chưa chạy real-feature outer/optimizer path | `preflight_outer.py: verify_outer_resume` | Production `fit` chạy smoke hai bước exact-16, Adam/clipping, validation và diagnostics; reload checkpoint bước 1 rồi so bước 2 với uninterrupted weights/optimizer/RNG/losses/reads. Smoke không tham gia selection |

Tách `source_identity` toàn runner khỏi `learning_source_identity` là có chủ ý: resume một run vẫn kiểm toàn source; selected weights còn kiểm các dependency đã tạo ra chúng. Thêm reporting CLI ở bước sau không tự ép phải refit nếu learning dependencies không đổi. Source/provenance gốc vẫn được giữ, không viết lại history.

## 3. Đường chạy của notebook mới

Notebook chính: `notebooks/hope_cad_v2_2_kaggle.ipynb`. Notebook kiểm tra ngắn: `notebooks/hope_cad_v2_2_preflight_kaggle.ipynb`. Logic thực thi nằm trong `scripts/hope_cad_v2/run_w1_w6.py`; hai notebook dùng cùng bootstrap và CLI. Preflight thêm `--preflight-only`, chạy cả outer smoke mới rồi dừng trước full calibration/search/probe, ghi `preflight_passed` và xuất đúng evidence schema để main tiếp tục bằng `--resume`. Ước lượng preflight 15–30 phút, 30–45 phút nếu cần chuẩn bị data/cache; revision có thêm smoke cần đo lại trên T4, các khoảng này không là cam kết runtime.

1. Đọc ZIP source hoặc thư mục đã được Kaggle giải nén. Xác minh từng SHA256 trong `bundle_manifest.json`; từ chối bundle cũ không có runner mới.
2. Enforce [timm 1.0.29](https://pypi.org/project/timm/1.0.29/) và [safetensors 0.8.0](https://pypi.org/project/safetensors/0.8.0/); giữ PyTorch của Kaggle nhưng ghi đầy đủ và khóa runtime khi resume. CUDA device phải có ít nhất 12 GiB. T4x2 sử dụng **một T4** cho một state/stream; không cộng bộ nhớ hai GPU thành một GPU.
3. Tìm MVTec root hoặc giải nén archive; xác minh ViT checkpoint có SHA256 `2c4839589adbbf701573dd155e1814f8adb145cd4d235bb86a93269369e111ec`.
4. Chạy correctness suite và canonical exact-16 resource profile. Chuẩn bị matched-prefix CADIC lock/normal-train manifest, không chạy full CADIC benchmark ở đây.
5. Xác minh cache input nếu có; cache cũ bị từ chối sẽ được tạo lại từ đúng prefix. W6 luôn dùng cache đã xác minh trong evidence directory, không trỏ trở lại input cache cũ.
6. Real RGB preflight: so frozen features với cache trên 2 support images × 5 views; chạy 9 support images, ngắt ở event 5 và resume qua slow commit 8. So tensors, pending gradients, scores và RNG với uninterrupted run. Boundary sau event 6 là **mô phỏng trong cùng prefix**, không truy cập task tương lai. Model preflight là random/unselected, không phải utility experiment.
7. Real-prefix outer smoke: M, β=1e-3, full canonical dimensions, exact 16-support graph, Adam lr=1e-4/clipping 1.0. Giữ schedule 200 episodes nhưng chạy `smoke=True`, cap hai bước và kiểm tra validation ở 0/2 chỉ để verification. Tải checkpoint bước 1 từ đĩa, chạy lại bước 2, yêu cầu weights/optimizer/RNG/training losses/validation losses/reads khớp exact. Lưu diagnostics ở 0/1. Không xuất deployment initialization hoặc selected receipt; calibration chính bắt đầu fresh, không kế thừa smoke weights. `--local-cpu-audit` chỉ đo CPU, không đủ để thỏa Kaggle gate. Sau đó selection seed 0 theo thứ tự `M C0 C1 C2 C3 C4 A1`. M/C2/C3 search β trên grid; C0/C1 search ordinary outer LR; C4 dùng β_M và fit/step riêng; A1 reuse learned C1 weights với β_M, không thêm optimizer steps, có gate riêng.
8. Candidate diagnostics và checkpoint ghi trong lúc fit. Gate chỉ loại candidates không ổn định, không rank bằng stress/probe. Fit fail không được cứu bằng checkpoint tốt trước đó; không retry candidate failed khi resume.
9. Sau selection, trích xuất cache `normal_probe` riêng, đo mechanism controls/interventions. Chỉ `ready_for_artifact_review` khi engineering pipeline đầy đủ. Không gọi W7 và không đọc official anomaly test trong các stages này.

`METHODS` mặc định giữ đủ bảy methods. Không giảm support length, steps, periods, feature size hoặc precision để vượt resource gate. Nếu không đủ một phiên/quota thì resume phiên sau; việc chưa có kết quả của một control không cho phép loại control đó khỏi claim cuối.

## 4. Việc người dùng cần làm trên Kaggle

1. Upload notebook **preflight** nêu trên trước. Sau khi preflight PASS và saved output được kiểm tra, dùng notebook chính.
2. Tạo version mới cho input **`hope_cad_v2_2_source_bundle`** từ `ultils/hope_cad_v2_2_source_bundle.zip` của worktree này; notebook phải dùng version mới. Kaggle tự giải nén ZIP cũng được.
3. Giữ hai inputs MVTec AD và ViT-B/8 ImageNet-21k đang có. Cache `mvtec_prefix_features_final.pt` cũ là optional; có thể bỏ để tránh tải lại một cache chắc chắn chưa có contract mới. Không cần tự sửa hashes để ép cache pass.
4. Chọn GPU T4, bật Internet cho bước cài đúng pins. Nếu mount có nhiều checkpoint/data roots, điền các biến `*_OVERRIDE` ở cell setup theo đường dẫn thật. Không sửa backbone SHA đã khóa.
5. Bắt đầu lượt chạy bằng **Save Version → Save & Run All**, bật lưu outputs trong hộp lưu version nếu có tuỳ chọn. Đây là fresh background run; không bấm Run All interactive rồi chạy lại Save & Run All để tránh tiêu quota hai lần. `SESSION_HOURS=10.5` tính từ code cell đầu tiên, gồm bootstrap/data preparation/evidence import. Notebook truyền monotonic start sang runner, không tự cộng thêm 10,5 giờ sau bootstrap. Nếu hoàn tất sớm thì xuất ngay; nếu tới budget thì dừng/xuất theo kế hoạch. Budget dành khoảng đệm trước giới hạn GPU 12 giờ được công bố, không đổi scientific recipe và không bảo đảm chống mọi platform termination. Quota thực tế vẫn cần đủ. Có thể cần nhiều phiên để hoàn thành tối đa 16 fits và 11 stress gates.
6. Sau khi background run kết thúc, mở đúng saved version, kiểm tra tab Output có **`hope_cad_v2_2_w1_w6_evidence.zip`**, `hope_w1_w6/handoff.json` và checkpoints/cache bên dưới `hope_w1_w6`. FileLink trong interactive chỉ là link tới server; saved version Output mới là nơi kiểm tra bản lưu. Có thể tải ZIP về máy sau khi ngủ dậy.
7. Khi preflight có `preflight_passed`, vào notebook chính → **Add Input → Notebook Output**, chọn output của đúng preflight version đã PASS. Giữ source bundle, dataset và backbone; notebook khôi phục evidence và tự `--resume`, bỏ qua các gate/cache đã đạt. Nếu status là `paused_for_session_limit`, attach output của đúng phiên vừa tạm dừng để tiếp tục. Attach ZIP dưới dạng Dataset cũng được. Không attach nhiều phiên evidence cùng lúc trừ khi đặt `RESUME_OVERRIDE` rõ ràng.
8. Nếu status là `stopped`, tải cả failure evidence và gửi lại để phân tích. Nếu là `ready_for_artifact_review`, cũng tải ZIP và đưa về repo để review. Cả hai chưa phải lệnh cho phép chạy W7.

Nếu runtime/packages/GPU hoặc source đổi giữa các phiên, resume fail có chủ ý. Không sửa `pipeline_lock.json`/`selection_lock.json` hay xoá failure rồi chạy tiếp như cùng run. Cần đánh giá mismatch trước. Nếu session bị cưỡng bức tắt mà chưa kịp export, checkpoint chỉ tồn tại nếu nền tảng giữ working output; code không thể khôi phục file đã bị xoá. Phải kiểm tra saved version và Output thực tế sau mỗi phiên.

Revision sau audit thay source hashes. Phải upload source ZIP mới và cả hai notebook mới, chạy preflight mới; không attach saved evidence/preflight từ revision cũ như một run cùng source. Main và preflight của revision mới dùng chung source hash và resume bình thường. Tránh chọn nhầm ZIP cũ cùng tên trong `results/hope_cad_v2/`; file bàn giao chính thức nằm trong `ultils/`.

`paused_for_session_limit` giờ trả exit code 0 và kết thúc notebook bình thường, để lượt lưu background có thể hoàn tất. `handoff.json` vẫn ghi `engineering_gates_passed=false`, W6 incomplete và W7 chưa được phép. Scientific failure vẫn trả nonzero và báo lỗi sau khi xuất evidence; không đổi failed thành passed. `preflight_passed` cũng chỉ là setup/correctness/resource/integration acceptance, không phải selection/mechanism acceptance.

Source checkout và extracted MVTec được đặt trong `/tmp` để saved outputs không bao gồm bản sao dataset/source. `/kaggle/working` giữ evidence directory và ZIP. MVTec sẽ được giải nén lại khi đổi phiên nếu input chỉ là archive; verified frozen cache được khôi phục từ saved output. Workflow này không sử dụng Google Drive.

## 5. Artifacts phải bàn giao

`artifact_manifest.json` bao phủ từng file trong ZIP. Các nhóm quan trọng:

- `pipeline_lock.json`, `commands.json`, `logs/`, `handoff.json`: source/runtime/methods, lệnh và trạng thái thật.
- `verification.json`, `canonical_resource_profile.json`, `preflight/preflight.json` và preflight attempt states: correctness, tài nguyên, cache/RGB/resume parity.
- `preflight_outer/preflight_outer.json`, smoke schedule, diagnostics và các `.pt`: real-feature exact-16 outer/Adam disk resume. `reportable=false`, `smoke=true`; không dùng những weights này làm initialization W6/W7. Operational pause giữa smoke có thể phải chạy lại smoke; không thay hoặc lặp các candidates đã thất bại của selection chính.
- `baseline_lock/`: manifest normal-train, prefix roles, resolved configs và route lock. Không coi lock là baseline score.
- `prefix_features.pt`, `cache_decision.json` nếu nhận input cache; `probe_features.pt`: hai phạm vi cache độc lập.
- `selection/<method>/selection_lock.json`, `schedule.json` (independently fitted methods), per-candidate fit/gate resumes và completion payloads, `selection.pt`, `selection.json`, `deployment_initialization.pt`, gate JSON, diagnostics và resource report.
- `mechanism.json`: selected receipts, teacher statistics, global/position teacher-mean, frozen masked-feature predictor, trained static controls, history/reset/freeze/visual sensitivity, output variance và contraction diagnostics.

A1 không có training steps 0/1/5/25 hay optimizer mới: bằng chứng representation learning kế thừa từ C1; parent receipts và `additional_optimizer_steps=0` được ghi. A2 trong W6 là paired inference intervention: single row-scan readout giữ scale đã học (các routes khác vẫn tính, không claim speedup), tắt từng guard trên cùng selected model. Đây không phải retrained A2 ablation để đưa vào main performance table. A3 ở W6 chỉ dùng prefix histories; wrong-history từ tasks đã thấy thuộc các analyses sau này.

## 6. Quy tắc đọc kết quả và chuyển sang W7

- `implemented`: có code path và contract tương ứng.
- `tested`: đúng phạm vi tests/artifacts đã thực sự chạy; fixture PASS không đồng nghĩa real-data PASS.
- `experimentally validated`: chỉ ghi sau khi có actual selected initialization/gate/probe evidence và review kết quả; không ghi từ log bị mất artifacts.
- `not implemented/pending`: locked-recipe refits và đầy đủ W7 uncertainty/benchmark orchestration vẫn phải làm ở chặng sau.

Long gate PASS chỉ chứng minh stability trong scope prefix đã đăng ký. Nếu score/history/visual contribution gần như không đổi, hoặc trained static/teacher-mean giải thích hết tín hiệu, báo mechanism inconclusive/negative theo từng giả thuyết. Không tự tăng β, đổi decoder/guards, dùng future tasks, hoặc xem anomaly-test metrics để sửa selection. Chỉ sau review artifacts mới quyết định chuyển changes sang `hope_cad_v2_1` và triển khai W7.

## 7. Tái tạo source bundle

Trong đúng worktree `hope_cad_v2_2`, sau khi source đã ổn định:

```powershell
pixi run python -B scripts/hope_cad_v2/build_w1_w6_notebook.py
pixi run python -B scripts/hope_cad_v2/package_w1_w6.py --output ultils/hope_cad_v2_2_source_bundle.zip
```

Packager chỉ lấy explicit source roots/extensions, notebook và spec; không đưa data, model checkpoint, results, credentials hoặc `.git` vào source ZIP. Nó kiểm lại CRC và từng SHA256 sau khi ghi. Source ZIP cần được cập nhật khi source đổi; notebook trên Kaggle không tự cập nhật từ GitHub.
