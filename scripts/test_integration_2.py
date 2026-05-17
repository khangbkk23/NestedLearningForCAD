"""
test_integration.py
-------------------
Kiểm tra toàn bộ pipeline Meta-NATH từ đầu đến cuối.
Chạy không cần GPU, không cần backbone weights thật
(tự fallback về _FallbackBackbone nếu HF không có sẵn).

Cách chạy:
    python test_integration.py

Output mong đợi: tất cả [PASS], không có [FAIL] hay Exception.
"""

import sys
import os
import traceback
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thêm project root vào path — chỉnh lại nếu cấu trúc thư mục khác
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Màu terminal ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

_results = []

def check(name: str, condition: bool, detail: str = ""):
    status = f"{GREEN}[PASS]{RESET}" if condition else f"{RED}[FAIL]{RESET}"
    msg = f"  {status} {name}"
    if detail:
        msg += f"  ← {detail}"
    print(msg)
    _results.append((name, condition))

def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {YELLOW}{title}{RESET}")
    print(f"{'─'*60}")

# ── Dummy batch ───────────────────────────────────────────────────────────────
BATCH   = 4
D       = 768
N_PATCH = 256   # 16×16 với ViT-B/14 (DINOv2) @ 224px
IMG_H   = IMG_W = 224
device  = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n{'='*60}")
print(f"  Meta-NATH Integration Test")
print(f"  Device : {device}")
print(f"  Batch  : {BATCH} × 3 × {IMG_H} × {IMG_W}")
print(f"{'='*60}")

dummy_images = torch.randn(BATCH, 3, IMG_H, IMG_W, device=device)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Import tất cả module không crash
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 1 · Imports")

try:
    from models.titans_memory import TITANSMemory, TTTEngine
    check("titans_memory import", True)
except Exception as e:
    check("titans_memory import", False, str(e)); sys.exit(1)

try:
    from models.acc_gating import ACCGating
    check("acc_gating import", True)
except Exception as e:
    check("acc_gating import", False, str(e)); sys.exit(1)

try:
    from models.cadic_coreset import CADICCoreset
    check("cadic_coreset import", True)
except Exception as e:
    check("cadic_coreset import", False, str(e)); sys.exit(1)

try:
    from models.meta_nath_core import MetaNATHCore
    check("meta_nath_core import", True)
except Exception as e:
    check("meta_nath_core import", False, str(e)); sys.exit(1)

try:
    from training.meta_nath_engine import MetaNATHEngine
    check("meta_nath_engine import", True)
except Exception as e:
    check("meta_nath_engine import", False, str(e)); sys.exit(1)

try:
    from models.null_space_proj import NSP2Config, NullSpaceProjector
    from models.cbp import CBPConfig, CBPMonitor
    from training.consolidation_engine import Phase3Config, NestedBackboneConsolidator
    check("phase3 modules import", True)
except Exception as e:
    check("phase3 modules import", False, str(e)); sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — TITANSMemory
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 2 · TITANSMemory (Delta Rule)")

mem = TITANSMemory(d=D, device=device)
check("M khởi tạo bằng zero",    mem.M.sum().item() == 0.0)
check("M shape đúng [d×d]",      tuple(mem.M.shape) == (D, D))

z = torch.randn(BATCH, D, device=device)
pred = mem.retrieve(z)
check("retrieve() shape [B, d]",  tuple(pred.shape) == (BATCH, D))

surprise = mem.update(k=z, v=z)
check("update() trả về float",   isinstance(surprise, float))
check("M đã thay đổi sau update", mem.M.abs().sum().item() > 0.0)
check("M trong [-5, 5]",          mem.M.abs().max().item() <= 5.0 + 1e-6)

sd = mem.state_dict()
mem2 = TITANSMemory(d=D, device=device)
mem2.load_state_dict(sd)
check("state_dict round-trip",    torch.allclose(mem.M, mem2.M))

mem.reset()
check("reset() → M = 0",         mem.M.sum().item() == 0.0)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — TTTEngine
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 3 · TTTEngine")

engine = TTTEngine(d=D, device=device)
z_cls  = torch.randn(BATCH, D, device=device)

z_updated, surprise_val = engine.process(z_cls)
check("process() → z_updated shape",  tuple(z_updated.shape) == (BATCH, D))
check("process() → surprise float",   isinstance(surprise_val, float))
check("step counter tăng",            engine._step == 1)

# Chạy nhiều bước — memory phải tích luỹ dần
for _ in range(9):
    engine.process(z_cls)
check("10 bước không crash",          engine._step == 10)
check("|M|_F > 0 sau 10 bước",        engine.memory.M.norm().item() > 0.0)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — ACCGating
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 4 · ACCGating")

gating = ACCGating(tau=0.25)

# Case 1: z_updated == z_original → cos_sim=1, h_term≈0 → ACC ≈ 1 → APPROVED
z_same = torch.randn(BATCH, D, device=device)
approved_same, acc_same = gating.should_consolidate(z_same, z_same)
check("z_same → APPROVED",            approved_same, f"ACC={acc_same:.4f}")

# Case 2: z_updated hoàn toàn ngẫu nhiên, khác xa z_original → thường REJECTED
z_a = torch.randn(BATCH, D, device=device)
z_b = torch.randn(BATCH, D, device=device)
approved_diff, acc_diff = gating.should_consolidate(z_a, z_b)
check("z_random → kết quả trả về được", isinstance(approved_diff, bool),
      f"ACC={acc_diff:.4f} → {'APPROVED' if approved_diff else 'REJECTED'}")

check("running_avg_acc() hoạt động",  isinstance(gating.running_avg_acc(), float))
check("approval_rate() trong [0,1]",  0.0 <= gating.approval_rate() <= 1.0)

try:
    ACCGating(tau=0.0)
    check("tau=0 raise ValueError", False)
except ValueError:
    check("tau=0 raise ValueError", True)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — CADICCoreset
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 5 · CADICCoreset")

coreset = CADICCoreset(max_size=10, d=D, n_patch=N_PATCH, store_images=False, device=device)
check("khởi tạo rỗng",           len(coreset) == 0)
check("is_full = False",         not coreset.is_full)

# Thêm 5 entries
for i in range(5):
    cls_e    = torch.randn(D, device=device)
    patch_e  = torch.randn(N_PATCH, D, device=device)
    coreset.update(cls_e, patch_e, task_id=0)

check("5 entries sau 5 update()", len(coreset) == 5)

# update_batch
cls_batch   = torch.randn(BATCH, D, device=device)
patch_batch = torch.randn(BATCH, N_PATCH, D, device=device)
n_up = coreset.update_batch(cls_batch, patch_batch, task_id=1)
check("update_batch() trả về int",  isinstance(n_up, int))

# compute_anomaly_score — batch API
patch_test = torch.randn(1, N_PATCH, D, device=device)
s_img, s_pix = coreset.compute_anomaly_score(patch_test, b=2)
check("s_img shape [B]",            tuple(s_img.shape) == (1,))
check("s_pix shape [B,N_patch]",    tuple(s_pix.shape) == (1, N_PATCH))
check("s_pix >= 0",                 s_pix.min().item() >= 0.0)
check("s_img <= max s_pix",         s_img.max().item() <= s_pix.max().item() + 1e-6)

# Đầy coreset → thay thế
for i in range(20):
    coreset.update(torch.randn(D, device=device), torch.randn(N_PATCH, D, device=device))
check("Coreset không vượt max_size", len(coreset) <= 10)

# state_dict round-trip
sd_c = coreset.state_dict()
coreset2 = CADICCoreset(max_size=10, d=D, n_patch=N_PATCH, device=device)
coreset2.load_state_dict(sd_c)
check("state_dict round-trip size",  len(coreset2) == len(coreset))

cls_repl = [torch.randn(D, device=device) for _ in range(len(coreset2))]
patch_repl = [torch.randn(N_PATCH, D, device=device) for _ in range(len(coreset2))]
coreset2.replace_all_embeddings(cls_repl, patch_repl)
check("replace_all_embeddings giữ size", len(coreset2) == len(cls_repl))
check("replace_all_embeddings refresh patch bank", tuple(coreset2.get_all_patch_embs().shape) == (len(cls_repl) * N_PATCH, D))

# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — MetaNATHCore (pipeline đầy đủ)
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 6 · MetaNATHCore — Full Pipeline")

try:
    model = MetaNATHCore(
        d=D,
        tau_acc=0.25,
        max_coreset_size=50,
        n_patch=N_PATCH,
        store_images=False,
        device=device,
    )
    check("MetaNATHCore khởi tạo", True)
except Exception as e:
    check("MetaNATHCore khởi tạo", False, str(e))
    traceback.print_exc(); sys.exit(1)

# Backbone phải ở eval mode
check("backbone.training = False", not model.backbone.training)

# Backbone phải frozen
n_grad = sum(1 for p in model.backbone.parameters() if p.requires_grad)
check("Backbone frozen (0 grad params)", n_grad == 0, f"grad_params={n_grad}")

# Gọi model.train() → backbone vẫn phải eval
model.train()
check("Sau train() backbone vẫn eval", not model.backbone.training)
check("Sau train() backbone vẫn frozen",
      all(not p.requires_grad for p in model.backbone.parameters()))

# Forward pass
model.eval()
try:
    out = model(dummy_images, task_id=0, update_coreset=True)
    check("forward() không crash",      True)
    check("z_cls shape [B, d]",         tuple(out["z_cls"].shape) == (BATCH, D))
    check("z_updated shape [B, d]",     tuple(out["z_updated"].shape) == (BATCH, D))
    check("z_patches shape [B, N, d]",  tuple(out["z_patches"].shape) == (BATCH, N_PATCH, D))
    check("patch_grid inferred 16×16",  out["patch_grid"] == (16, 16))
    check("surprise là float",          isinstance(out["surprise"], float))
    check("acc_score là float",         isinstance(out["acc_score"], float))
    check("approved là bool",           isinstance(out["approved"], bool))
    check("coreset_updated là bool",    isinstance(out["coreset_updated"], bool))

    status = f"ACC={out['acc_score']:.4f} → {'APPROVED ✓' if out['approved'] else 'REJECTED ✗'}"
    print(f"\n  {YELLOW}► Gating result: {status}{RESET}")

except Exception as e:
    check("forward() không crash", False, str(e))
    traceback.print_exc(); sys.exit(1)

# Chạy thêm vài batch để lấp đầy coreset
for i in range(5):
    model(torch.randn(BATCH, 3, IMG_H, IMG_W, device=device), task_id=i % 3)
check("5 forward() liên tiếp không crash", True)

# Normal-only coreset update through MetaNATHEngine
try:
    model.coreset = CADICCoreset(max_size=50, d=D, n_patch=N_PATCH, store_images=False, device=device)
    engine_for_filter = MetaNATHEngine(model=model, device=device)
    mixed_batch = {
        "img": dummy_images.detach().cpu(),
        "anomaly": torch.tensor([0, 1, 0, 1]),
    }
    filter_metrics = engine_for_filter.train_task([mixed_batch], task_id=99, epochs=1, verbose=False)
    check("normal-only update count", filter_metrics["normal_update_samples"] == 2)
    check("synthetic anomalies skipped", filter_metrics["skipped_anomaly_count"] == 2)
    check("coreset không nhận quá normal samples", len(model.coreset) <= 2)
except Exception as e:
    check("normal-only coreset filter", False, str(e))
    traceback.print_exc()

# score_image() — chỉ chạy nếu coreset đã có entries
if len(model.coreset) > 0:
    try:
        single_img = torch.randn(1, 3, IMG_H, IMG_W, device=device)
        result = model.score_image(single_img, b=2)
        check("score_image() không crash",        True)
        check("s_img là float",                   isinstance(result["s_img"], float))
        check("anomaly_map shape [H, W]",         tuple(result["anomaly_map"].shape) == (IMG_H, IMG_W))
        print(f"\n  {YELLOW}► Anomaly score: s_img={result['s_img']:.6f}{RESET}")
    except Exception as e:
        check("score_image() không crash", False, str(e))
        traceback.print_exc()
else:
    print(f"  {YELLOW}[SKIP] score_image() — coreset rỗng (tất cả batch bị REJECTED){RESET}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — Checkpoint save/load
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 7 · Checkpoint")

try:
    sd_full = model.full_state_dict()
    check("full_state_dict() có đủ keys",
          all(k in sd_full for k in ["backbone", "ttt_engine", "gating", "coreset"]))

    # Save → load → so sánh M matrix
    M_before = model.ttt_engine.memory.M.clone()
    model2 = MetaNATHCore(d=D, n_patch=N_PATCH, store_images=False, device=device)
    model2.load_full_state_dict(sd_full)
    M_after = model2.ttt_engine.memory.M

    check("M matrix khớp sau load",      torch.allclose(M_before, M_after))
    check("Coreset size khớp sau load",   len(model2.coreset) == len(model.coreset))
    check("backbone vẫn frozen sau load",
          all(not p.requires_grad for p in model2.backbone.parameters()))
except Exception as e:
    check("Checkpoint round-trip", False, str(e))
    traceback.print_exc()

try:
    refresh_model = MetaNATHCore(d=D, n_patch=N_PATCH, store_images=True, device=device)
    refresh_model(dummy_images, task_id=0, update_coreset=True)
    refresh_stats = refresh_model.refresh_coreset_embeddings(batch_size=2)
    check("refresh_coreset_embeddings chạy được", refresh_stats["refreshed"])
    check("refresh giữ coreset size", refresh_stats["entries"] == len(refresh_model.coreset))
except Exception as e:
    check("refresh_coreset_embeddings", False, str(e))
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 8 — Phase 3 utilities
# ─────────────────────────────────────────────────────────────────────────────
section("TEST 8 · Phase 3 Utilities")

try:
    projector = NullSpaceProjector(d=D, config=NSP2Config(enabled=True, min_null_dim=64), device=device)
    nsp_stats = projector.fit(torch.randn(8, D, device=device))
    grad_vec = torch.randn(D, device=device)
    grad_mat = torch.randn(4, D, device=device)
    check("NSP2 fit trả stats", "null_dim" in nsp_stats)
    check("NSP2 project vector shape", tuple(projector.project(grad_vec).shape) == (D,))
    check("NSP2 project matrix shape", tuple(projector.project(grad_mat).shape) == (4, D))

    lin = torch.nn.Linear(D, 4).to(device)
    out = lin(torch.randn(2, D, device=device)).sum()
    out.backward()
    cbp_stats = CBPMonitor(CBPConfig(enabled=False)).scan_and_maybe_reset(lin, projector=projector)
    check("CBP monitor trả stats", "dead_neuron_ratio" in cbp_stats)

    small_projector = NullSpaceProjector(
        d=16,
        config=NSP2Config(enabled=True, min_null_dim=8, recycling_enabled=True, fallback_null_dims=(8, 4, 2)),
        device=device,
    )
    recycling_stats = small_projector.fit(torch.randn(32, 16, device=device))
    check("Subspace Recycling log có recycled", "recycled" in recycling_stats)
    check("Subspace Recycling giữ null_dim tối thiểu", recycling_stats["null_dim"] >= 8)

    reset_lin = torch.nn.Linear(4, 4).to(device)
    reset_out = reset_lin(torch.randn(2, 4, device=device)).sum()
    reset_out.backward()
    reset_projector = NullSpaceProjector(
        d=4,
        config=NSP2Config(enabled=True, min_null_dim=2),
        device=device,
    )
    reset_projector.fit(torch.randn(8, 4, device=device))
    reset_stats = CBPMonitor(
        CBPConfig(enabled=True, monitor_only=False, threshold=2.0)
    ).scan_and_maybe_reset(reset_lin, projector=reset_projector)
    check("CBP reset path chạy thật", reset_stats["reset_units"] > 0)
    check("Phase3Config khởi tạo", isinstance(Phase3Config(), Phase3Config))
    check("NestedBackboneConsolidator class callable", callable(NestedBackboneConsolidator))
except Exception as e:
    check("Phase 3 utilities", False, str(e))
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# Tổng kết
# ─────────────────────────────────────────────────────────────────────────────
total  = len(_results)
passed = sum(1 for _, ok in _results if ok)
failed = total - passed

print(f"\n{'='*60}")
if failed == 0:
    print(f"  {GREEN}ALL {total} TESTS PASSED ✓{RESET}")
    print(f"  Pipeline sẵn sàng kết nối DataLoader.")
else:
    print(f"  {RED}{failed}/{total} TESTS FAILED ✗{RESET}")
    print(f"  Các test thất bại:")
    for name, ok in _results:
        if not ok:
            print(f"    {RED}✗ {name}{RESET}")
print(f"{'='*60}\n")

sys.exit(0 if failed == 0 else 1)
