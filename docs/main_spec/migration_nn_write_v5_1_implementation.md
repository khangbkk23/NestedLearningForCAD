# Nested Learning for CAD — bàn giao G4 NN-write bridge

Ngày21/10/2026. [Spec](migration_nn_write_v5_1.md), [contract](migration_nn_write_v5_1_contract.json). G3 đã review PASS; G4 có implementation/local evidence riêng, **canonical bridge chờ user chạy**. Không suy từ các tests rằng benchmark hoặc ADperformance đã đạt.

## Codespace

- `training/support_migration_v5_1/contract.py`: verify additive source và pin G1/G2/G3 receipts; dựng execution/suite/job identity mới.
- `data.py`: one-arrival normal delivery, lazy tar extraction; từ chối duplicate/out-of-order/forbidden rows.
- `bridge.py`: hai strict owners, full state/event exact sau mỗi arrival trước ảnh kế tiếp; durable attempt ledger và budget, không finalize/publish partial.
- `lineage.py`: publish candidate384 theo staging/rename; raw parent manifest và inherited hashes; new owner cho active/task mới; inherited task00 resolver scoped qua original loader; wrappers cho frozen train/final/matrix evaluation.
- `worker.py`: canonical T4 load376/pending checks, feature extraction và bridge, strict reload384 trước publication.
- `notebook_support/support_migration_v5_1/`: bootstrap/entry/build cùng prerequisite receipts được pin; failure export, source ZIP và notebook mới.
- `tests/support_migration_v5_1/`: synthetic identities/causal delivery/bridge/ownership/rollback/budget/publication/resume/final15/matrix120/tamper. Test counts, logs và hashes thật nằm trong delivery/review `.agents`.

Không sửa engine/model/candidate/spec cũ. Parent376 giữ nguyên; legacy task00 wrapper không rewrite sang new owner. Execution wrappers giữ original strict validators, không nới source/runtime/identity. API continuation được test local, nhưng notebook batch này chỉ expose bridge; full benchmark interface sẽ giao sau khi review canonical bridge.

## Các bước Kaggle

1. Import `notebooks/hope_cad_v5_1_nn_write_bridge_kaggle.ipynb`, dùng kernel mới.
2. Attach source mới: `.agents/artifacts/deliveries/hope_support_v5_1_nn_write_bridge_r2/hope_cad_support_v5_1_nn_write_bridge_source_r2.zip` (hoặc Dataset giải nén). Tháo source NN-validation/profiler hoặc bridge r1 cũ. R2 chỉ bổ sung metadata indexing để tar khác thứ tự vẫn giao đúng arrival; không đọc pixels của ảnh tương lai.
3. Attach **original** `.agents/artifacts/runs/support_v5_1_benchmark_NESTED_seed0/session_01/hope_cad_support_v5_1_benchmark_evidence.zip`, paused376. Không attach output validation195 hoặc profiler; G1/G2/G3 receipts đã được đóng gói/pin.
4. Giữ MVTec AD raw và checkpoint `vit_base_patch8_224.augreg_in21k.safetensors` đang dùng. Tổng4 inputs. Không W6/pilot learned states.
5. GPU T4×2; PHASE='bridge', overridesNone nếu discovery rõ. Save Version → Save & Run All. Budget cố định60 phút gồm180s conservative export reserve, không ETA. Hai actors chạy trên cuda:0; GPU thứ hai không tự pool VRAM.
6. Tải `hope_cad_support_v5_1_nn_write_bridge_evidence.zip` cùng log vào `D:/src/DATN/.agents/artifacts/runs/support_v5_1_nn_write_bridge_01/session_01/`. Không giải nén vào code hoặc ghi đè evidence.

## Nghiệm thu

`bridge_complete_pending_review` cần review đúng parent/source/runtime, pending load,8 pairs, checkpoint384/trace/lineage và inherited owner. Candidate384 không tự là full benchmark hoàn tất. Incomplete/failure giữ artifacts, parent376 vẫn hợp lệ; không tự tăng budget/retry hoặc dùng shadow reference continuation. Publication chưa có marker canonical thì không accepted; stage không là output benchmark.

Nếu canonical bridge đạt, reviewer mới chuẩn bị continuation từ384; không rerun W6/pilot/G3. Final/matrix cells tiếp tục dùng frozen metric/scoring protocol và per-checkpoint owner mapping. Không tuning ADtest hoặc VisA, không hứa model thắng CADIC từ speedup.
