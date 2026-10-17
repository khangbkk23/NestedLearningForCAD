# Recovery preflight r4: nghiệm thu và bước tiếp theo

Đã nghiệm thu ZIP `hope_cad_recovery_stage1_evidence.zip`, SHA256 `b657aa72d230b43e42cf137b3bb8aa877a46ca75898f1826b4b7b8b901b2b43e`,365053827bytes,156entries. Không sửa ZIP/source/notebooks sau nghiệm thu.

- Source `lifecycle-r4`, operations digest `ced707d247b1e84fc0fa9815e65c491615cae932df7af5922fb4ac3060b3ab47`,119frozen files giữ đúng hash.
- 130frozen +45recovery tests PASS trênT4/CUDA12.8.
- Independent recheck M/C1/A1 plain và interrupted/resumed terminal checkpoints: adapter/RNG/cursor exact, mỗi history9events.
- CADIC16normal images,12544patches,bank10000x768;2141replacements/403rejections; train119.9277s. Cache dùng đúng shared registry bank.
- Cloned full-budget reference pair/4normal-candidate operator/counters/features/Eq9 parity đạt, checkpoint roundtrip và read-only state/RNG đạt.
- cleanup_complete=true, worker/runner exit0, handoff recovery_preflight_passed. Elapsed tới handoff460.3349s (~7m40s), không phải thời gian toàn bộ controls.
- Gọi lại current strict validate_preflight trên receipts đã hash-verified: PASS. Parent M subset47entries vẫn đúng pinned hashes.

## Chạy controls

1. Import `notebooks/hope_cad_recovery_stage1_controls_kaggle.ipynb` hiện hành (source r4). Không đổi source hoặc packaged main_spec sau PASS.
2. Attach5inputs: source_r4; completed W6; chính preflight r4 ZIP/Output vừa nghiệm thu; MVTec; ViT. Remove W7.2 khỏi inputs (M parent đã nằm trong accepted recovery output). Không attach failed recovery outputs hoặc upload cả thư mục chứa nhiều phiên khác nhau.
3. Giữ PHASE='controls', JOBS=['C1_seed0','A1_seed0','CADIC10000_seed0'], SESSION_HOURS=10.5 nếu quota cho phép. GPU T4 x2. Save Version / Save & Run All.
4. Bootstrap phải restore recovery output đúng study digest `b732d8566d56362e87f6f2c6a4acafddd62f716df8c0f05585d7e269fd3e7cd7`; source digest giữ nguyên. Runner đọc M saved checkpoints để normal diagnostics, rồi train/final/diagnostics C1,A1 và matched CADIC. Không refit W6 hoặc rescore M official test.
5. Complete: recovery_stage1_complete_pending_review, đủcomparison+3controls+M/C1/A1normal diagnostics+lifecycle receipts. Pause: paused_at_atomic_boundary, giữ latest ZIP để resume cùng source/config. Session hours là budget, không là ETA bảo đảm full3jobs. Không suy fullCADIC time từ16prefix images.
6. Tải latest ZIP/log về thư mục riêng `docs/recovery_stage1_kaggle/controls/` và review. Giữ preflight và các failure cũ; không overwrite. Chưa tự chuyển W7.3/VisA hoặc thay method bằng test outputs.

Đây là technical acceptance cho controls, không là bằng chứng model ngang/hơnCADIC hoặc giai đoạn1 đã hoàn tất. Machine report: `results/recovery_stage1/preflight_r4_review.json`.
