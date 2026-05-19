"""
mechanism_smoke.py
------------------
Run a CPU/GPU-friendly smoke test for the Meta-NATH mechanisms.

The test does not require real dataset files. If the HuggingFace backbone is not
available, MetaNATHCore falls back to its lightweight local backbone.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "models").is_dir() and (parent / "training").is_dir():
            return parent
    raise RuntimeError("Could not locate project root from script path.")


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

RESULTS: list[tuple[str, bool]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    ok = bool(condition)
    status = f"{GREEN}[PASS]{RESET}" if ok else f"{RED}[FAIL]{RESET}"
    msg = f"  {status} {name}"
    if detail:
        msg += f"  <- {detail}"
    print(msg)
    RESULTS.append((name, ok))


def section(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {YELLOW}{title}{RESET}")
    print(f"{'-' * 60}")


BATCH = 4
D = 768
N_PATCH = 256
IMG_H = IMG_W = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n{'=' * 60}")
print("  Meta-NATH Mechanism Smoke Test")
print(f"  Device : {DEVICE}")
print(f"  Batch  : {BATCH} x 3 x {IMG_H} x {IMG_W}")
print(f"{'=' * 60}")

dummy_images = torch.randn(BATCH, 3, IMG_H, IMG_W, device=DEVICE)

section("TEST 1 - Imports")

try:
    from models.titans_memory import TITANSMemory, TTTEngine
    check("titans_memory import", True)
except Exception as exc:
    check("titans_memory import", False, str(exc))
    sys.exit(1)

try:
    from models.acc_gating import ACCGating
    check("acc_gating import", True)
except Exception as exc:
    check("acc_gating import", False, str(exc))
    sys.exit(1)

try:
    from models.cadic_coreset import CADICCoreset
    check("cadic_coreset import", True)
except Exception as exc:
    check("cadic_coreset import", False, str(exc))
    sys.exit(1)

try:
    from models.meta_nath_core import MetaNATHCore
    check("meta_nath_core import", True)
except Exception as exc:
    check("meta_nath_core import", False, str(exc))
    sys.exit(1)

try:
    from training.meta_nath_engine import MetaNATHEngine
    check("meta_nath_engine import", True)
except Exception as exc:
    check("meta_nath_engine import", False, str(exc))
    sys.exit(1)

try:
    from models.null_space_proj import NSP2Config, NullSpaceProjector
    from models.cbp import CBPConfig, CBPMonitor
    from training.consolidation_engine import Phase3Config, NestedBackboneConsolidator
    check("phase3 modules import", True)
except Exception as exc:
    check("phase3 modules import", False, str(exc))
    sys.exit(1)

section("TEST 2 - TITANSMemory")

mem = TITANSMemory(d=D, device=DEVICE)
check("M initializes to zero", mem.M.sum().item() == 0.0)
check("M shape is [d, d]", tuple(mem.M.shape) == (D, D))

z = torch.randn(BATCH, D, device=DEVICE)
pred = mem.retrieve(z)
check("retrieve() shape [B, d]", tuple(pred.shape) == (BATCH, D))

surprise = mem.update(k=z, v=z)
check("update() returns float", isinstance(surprise, float))
check("M changes after update", mem.M.abs().sum().item() > 0.0)
check("M stays clamped", mem.M.abs().max().item() <= 5.0 + 1e-6)

state = mem.state_dict()
mem2 = TITANSMemory(d=D, device=DEVICE)
mem2.load_state_dict(state)
check("state_dict round-trip", torch.allclose(mem.M, mem2.M))

mem.reset()
check("reset() clears M", mem.M.sum().item() == 0.0)

section("TEST 3 - TTTEngine")

engine = TTTEngine(d=D, device=DEVICE)
z_cls = torch.randn(BATCH, D, device=DEVICE)

z_updated, surprise_val = engine.process(z_cls)
check("process() z_updated shape", tuple(z_updated.shape) == (BATCH, D))
check("process() surprise float", isinstance(surprise_val, float))
check("step counter increments", engine._step == 1)

for _ in range(9):
    engine.process(z_cls)
check("10 steps do not crash", engine._step == 10)
check("|M|_F > 0 after 10 steps", engine.memory.M.norm().item() > 0.0)

section("TEST 4 - ACCGating")

gating = ACCGating(tau=0.25)
z_same = torch.randn(BATCH, D, device=DEVICE)
approved_same, acc_same = gating.should_consolidate(z_same, z_same)
check("identical embeddings are approved", approved_same, f"ACC={acc_same:.4f}")

z_a = torch.randn(BATCH, D, device=DEVICE)
z_b = torch.randn(BATCH, D, device=DEVICE)
approved_diff, acc_diff = gating.should_consolidate(z_a, z_b)
check(
    "random embeddings return a bool decision",
    isinstance(approved_diff, bool),
    f"ACC={acc_diff:.4f} -> {'APPROVED' if approved_diff else 'REJECTED'}",
)

check("running_avg_acc() returns float", isinstance(gating.running_avg_acc(), float))
check("approval_rate() in [0, 1]", 0.0 <= gating.approval_rate() <= 1.0)

try:
    ACCGating(tau=0.0)
    check("tau=0 raises ValueError", False)
except ValueError:
    check("tau=0 raises ValueError", True)

section("TEST 5 - CADICCoreset")

coreset = CADICCoreset(max_size=10, d=D, n_patch=N_PATCH, store_images=False, device=DEVICE)
check("starts empty", len(coreset) == 0)
check("is_full is False", not coreset.is_full)

for _ in range(5):
    coreset.update(
        torch.randn(D, device=DEVICE),
        torch.randn(N_PATCH, D, device=DEVICE),
        task_id=0,
    )
check("5 entries after 5 update() calls", len(coreset) == 5)

cls_batch = torch.randn(BATCH, D, device=DEVICE)
patch_batch = torch.randn(BATCH, N_PATCH, D, device=DEVICE)
n_updated = coreset.update_batch(cls_batch, patch_batch, task_id=1)
check("update_batch() returns int", isinstance(n_updated, int))

patch_test = torch.randn(1, N_PATCH, D, device=DEVICE)
s_img, s_pix = coreset.compute_anomaly_score(patch_test, b=2)
check("s_img shape [B]", tuple(s_img.shape) == (1,))
check("s_pix shape [B, N_patch]", tuple(s_pix.shape) == (1, N_PATCH))
check("s_pix is non-negative", s_pix.min().item() >= 0.0)
check("s_img is bounded by max pixel score", s_img.max().item() <= s_pix.max().item() + 1e-6)

for _ in range(20):
    coreset.update(
        torch.randn(D, device=DEVICE),
        torch.randn(N_PATCH, D, device=DEVICE),
    )
check("Coreset does not exceed max_size", len(coreset) <= 10)

state_c = coreset.state_dict()
coreset2 = CADICCoreset(max_size=10, d=D, n_patch=N_PATCH, device=DEVICE)
coreset2.load_state_dict(state_c)
check("state_dict round-trip size", len(coreset2) == len(coreset))

cls_repl = [torch.randn(D, device=DEVICE) for _ in range(len(coreset2))]
patch_repl = [torch.randn(N_PATCH, D, device=DEVICE) for _ in range(len(coreset2))]
coreset2.replace_all_embeddings(cls_repl, patch_repl)
check("replace_all_embeddings keeps size", len(coreset2) == len(cls_repl))
check(
    "replace_all_embeddings refreshes patch bank",
    tuple(coreset2.get_all_patch_embs().shape) == (len(cls_repl) * N_PATCH, D),
)

section("TEST 6 - MetaNATHCore full pipeline")

try:
    model = MetaNATHCore(
        d=D,
        tau_acc=0.25,
        max_coreset_size=50,
        n_patch=None,
        store_images=False,
        device=DEVICE,
    )
    check("MetaNATHCore initializes", True)
except Exception as exc:
    check("MetaNATHCore initializes", False, str(exc))
    traceback.print_exc()
    sys.exit(1)

check("backbone.training is False", not model.backbone.training)
check("backbone is frozen", sum(1 for p in model.backbone.parameters() if p.requires_grad) == 0)

model.train()
check("train() keeps backbone in eval mode", not model.backbone.training)
check("train() keeps backbone frozen", all(not p.requires_grad for p in model.backbone.parameters()))

model.eval()
try:
    out = model(dummy_images, task_id=0, update_coreset=True)
    actual_n_patch = out["z_patches"].shape[1]
    actual_grid = out["patch_grid"]
    check("forward() does not crash", True)
    check("z_cls shape [B, d]", tuple(out["z_cls"].shape) == (BATCH, D))
    check("z_updated shape [B, d]", tuple(out["z_updated"].shape) == (BATCH, D))
    check("z_patches shape [B, N, d]", tuple(out["z_patches"].shape) == (BATCH, actual_n_patch, D))
    check("patch grid is square", actual_grid[0] * actual_grid[1] == actual_n_patch)
    check("surprise is float", isinstance(out["surprise"], float))
    check("acc_score is float", isinstance(out["acc_score"], float))
    check("approved is bool", isinstance(out["approved"], bool))
    check("coreset_updated is bool", isinstance(out["coreset_updated"], bool))

    status = f"ACC={out['acc_score']:.4f} -> {'APPROVED' if out['approved'] else 'REJECTED'}"
    print(f"\n  {YELLOW}> Gating result: {status}{RESET}")
except Exception as exc:
    check("forward() does not crash", False, str(exc))
    traceback.print_exc()
    sys.exit(1)

for task_id in range(5):
    model(torch.randn(BATCH, 3, IMG_H, IMG_W, device=DEVICE), task_id=task_id % 3)
check("5 consecutive forward() calls do not crash", True)

try:
    model.coreset = CADICCoreset(max_size=50, d=D, n_patch=actual_n_patch, store_images=False, device=DEVICE)
    engine_for_filter = MetaNATHEngine(model=model, device=DEVICE)
    mixed_batch = {
        "img": dummy_images.detach().cpu(),
        "anomaly": torch.tensor([0, 1, 0, 1]),
    }
    filter_metrics = engine_for_filter.train_task([mixed_batch], task_id=99, epochs=1, verbose=False)
    check("normal-only update count", filter_metrics["normal_update_samples"] == 2)
    check("synthetic anomalies skipped", filter_metrics["skipped_anomaly_count"] == 2)
    check("coreset receives at most normal samples", len(model.coreset) <= 2)
except Exception as exc:
    check("normal-only coreset filter", False, str(exc))
    traceback.print_exc()

if len(model.coreset) > 0:
    try:
        result = model.score_image(torch.randn(1, 3, IMG_H, IMG_W, device=DEVICE), b=2)
        check("score_image() does not crash", True)
        check("s_img is float", isinstance(result["s_img"], float))
        check("anomaly_map shape [H, W]", tuple(result["anomaly_map"].shape) == (IMG_H, IMG_W))
        print(f"\n  {YELLOW}> Anomaly score: s_img={result['s_img']:.6f}{RESET}")
    except Exception as exc:
        check("score_image() does not crash", False, str(exc))
        traceback.print_exc()
else:
    print(f"  {YELLOW}[SKIP] score_image(): coreset is empty because all batches were rejected{RESET}")

section("TEST 7 - Checkpoint")

try:
    full_state = model.full_state_dict()
    check(
        "full_state_dict() has required keys",
        all(key in full_state for key in ["backbone", "ttt_engine", "gating", "coreset"]),
    )

    memory_before = model.ttt_engine.memory.M.clone()
    model2 = MetaNATHCore(d=D, n_patch=None, store_images=False, device=DEVICE)
    model2.load_full_state_dict(full_state)
    memory_after = model2.ttt_engine.memory.M

    check("M matrix matches after load", torch.allclose(memory_before, memory_after))
    check("Coreset size matches after load", len(model2.coreset) == len(model.coreset))
    check("backbone remains frozen after load", all(not p.requires_grad for p in model2.backbone.parameters()))
except Exception as exc:
    check("checkpoint round-trip", False, str(exc))
    traceback.print_exc()

try:
    refresh_model = MetaNATHCore(d=D, n_patch=None, store_images=True, device=DEVICE)
    refresh_model(dummy_images, task_id=0, update_coreset=True)
    refresh_stats = refresh_model.refresh_coreset_embeddings(batch_size=2)
    check("refresh_coreset_embeddings runs", refresh_stats["refreshed"])
    check("refresh keeps coreset size", refresh_stats["entries"] == len(refresh_model.coreset))
except Exception as exc:
    check("refresh_coreset_embeddings", False, str(exc))
    traceback.print_exc()

section("TEST 8 - Phase 3 utilities")

try:
    projector = NullSpaceProjector(d=D, config=NSP2Config(enabled=True, min_null_dim=64), device=DEVICE)
    nsp_stats = projector.fit(torch.randn(8, D, device=DEVICE))
    grad_vec = torch.randn(D, device=DEVICE)
    grad_mat = torch.randn(4, D, device=DEVICE)
    check("NSP2 fit returns stats", "null_dim" in nsp_stats)
    check("NSP2 projects vector shape", tuple(projector.project(grad_vec).shape) == (D,))
    check("NSP2 projects matrix shape", tuple(projector.project(grad_mat).shape) == (4, D))

    lin = torch.nn.Linear(D, 4).to(DEVICE)
    lin(torch.randn(2, D, device=DEVICE)).sum().backward()
    cbp_stats = CBPMonitor(CBPConfig(enabled=False)).scan_and_maybe_reset(lin, projector=projector)
    check("CBP monitor returns stats", "dead_neuron_ratio" in cbp_stats)

    small_projector = NullSpaceProjector(
        d=16,
        config=NSP2Config(enabled=True, min_null_dim=8, recycling_enabled=True, fallback_null_dims=(8, 4, 2)),
        device=DEVICE,
    )
    recycling_stats = small_projector.fit(torch.randn(32, 16, device=DEVICE))
    check("Subspace Recycling reports recycled flag", "recycled" in recycling_stats)
    check("Subspace Recycling preserves requested null_dim", recycling_stats["null_dim"] >= 8)

    reset_lin = torch.nn.Linear(4, 4).to(DEVICE)
    reset_lin(torch.randn(2, 4, device=DEVICE)).sum().backward()
    reset_projector = NullSpaceProjector(
        d=4,
        config=NSP2Config(enabled=True, min_null_dim=2),
        device=DEVICE,
    )
    reset_projector.fit(torch.randn(8, 4, device=DEVICE))
    reset_stats = CBPMonitor(
        CBPConfig(enabled=True, monitor_only=False, threshold=2.0)
    ).scan_and_maybe_reset(reset_lin, projector=reset_projector)
    check("CBP reset path executes", reset_stats["reset_units"] > 0)
    check("Phase3Config initializes", isinstance(Phase3Config(), Phase3Config))
    check("NestedBackboneConsolidator is callable", callable(NestedBackboneConsolidator))
except Exception as exc:
    check("Phase 3 utilities", False, str(exc))
    traceback.print_exc()

total = len(RESULTS)
passed = sum(1 for _, ok in RESULTS if ok)
failed = total - passed

print(f"\n{'=' * 60}")
if failed == 0:
    print(f"  {GREEN}ALL {total} TESTS PASSED{RESET}")
    print("  Mechanism paths are ready for the full demo workflow.")
else:
    print(f"  {RED}{failed}/{total} TESTS FAILED{RESET}")
    print("  Failed checks:")
    for name, ok in RESULTS:
        if not ok:
            print(f"    {RED}x {name}{RESET}")
print(f"{'=' * 60}\n")

sys.exit(0 if failed == 0 else 1)
