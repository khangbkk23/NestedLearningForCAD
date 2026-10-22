# Nested Learning for CAD — source hợp nhất v7

Đọc README và docs/README.md rồi architecture/method/benchmark/operations/results theo nhiệm vụ. Một package tại src/hope_cad; không tạo lại namespace phiên bản v2/v5/v5_1 hoặc launcher riêng cho mỗi delivery.

- Method hiện hành là hope_cad_v5_1_causal_scaled_utility. Tên workspace/package release không là scientific revision. resources/method.json và study.json giữ contract; lịch sử design identity ở lineage.json, source identity hiện hành bao gồm docs/code/tests.
- Giữ nguyên operations/precision/causality/clock/bank/Eq9 trong cleanup. Thay khoa học cần contract/method revision và acceptance riêng; không tune từ anomaly test. MVTec đã development-exposed; chưa VisA confirmation.
- User quyết định. Codex cộng tác viên chính; Claude/model khác cố vấn task-scoped read-only mặc định. Không tự API/rental/SSH/GPU/push. Không sửa sibling worktrees.
- Tài liệu tự đủ trong repo: docs/README.md định tuyến current docs/results và docs/research/ chứa full spec/plan prospective. Không cần context ngoài repo hoặc chatlog để hiểu/implement; raw artifacts cần cấp riêng theo hashes trong docs/results.md.
- Editable install trước test; kiểm clean source installation, gradients, parity, source/runtime/data locks, atomic rollback và resume. Local tests không thay real5090 preflight hoặc AD benchmark.
- Production entrypoint hope-cad / python -m hope_cad. Optional baselines ở integrations; fixture adapters ở testing, không dùng làm comparator thật. Output ngoài source hoặc runs/results/logs; data/checkpoints không commit.
- Historical-query slow consolidation chưa implement. Không tự bổ sung proposal vào method khi chỉ được giao tổ chức code.

- Cohort tiếp theo theo `docs/pipeline_alignment_spec.md` và JSON cạnh nó: full-train 3.629, giữ recipe hiện hành, H32 là candidate riêng; runtime 3.599 chưa đổi. Đọc docs/research/README.md và pipeline trước implementation, không overwrite frozen resources hoặc old evidence.
