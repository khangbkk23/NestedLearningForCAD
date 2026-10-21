# exps/npim_cad/preflight.py
"""EXP-NPIM-00 Stage A/B preflight.

Checks, in order:

1. repository identity and dirty-tree state (read-only Git inspection);
2. environment and device facts;
3. frozen checkpoint existence and SHA-256 identity;
4. pre-attention hook identity and architecture preconditions;
5. block-8 ``Y`` parity against the immutable historical caches;
6. the decisive target-leakage oracle: perturbing only the raw 8x8 patch ``p``
   must leave ``key_p`` unchanged, bitwise for the CPU oracle, while ``Y`` may
   change and deep neighbour/CLS tokens are demonstrably contaminated;
7. autograd dependency of ``key_p`` on target-patch pixels (exactly zero);
8. vectorised leave-one-out vs loop oracle, patch-edge handling and numerical
   degeneracy;
9. normal-only data availability (writer stream + 59 normal witnesses), with no
   anomaly label or mask ever opened;
10. optional bounded CUDA smoke with VRAM accounting.

Writes one exclusive JSON manifest.  It does not train, score anomalies or
modify existing files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.benchmark_protocol_v1 import _ImageDataset  # noqa: E402
from exps.task_p.p0_normal_support import require  # noqa: E402
from models.hope_cad.npim_context import (  # noqa: E402
    GRID,
    PATCHES,
    NPIMContextConfig,
    NPIMFeatureExtractor,
    build_keys,
    build_keys_loop,
    global_leave_one_out,
    make_projection,
    neighbour_table,
    sha256_file,
    sha256_tensor,
)

CP = ROOT / "checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth"
CP_SHA = "755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415"
CP_BYTES = 343_282_995
CACHE_DIR = ROOT / "results/hope_cad/frozen_features_v1/clean"
PROTOCOL = ROOT / "exps/npim_cad/protocol.md"
P0C = ROOT / "results/hope_cad/task_p/p0c/protocol_resolved.json"
MVTEC = ROOT / "data/mvtec"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
PARITY_ROWS = (0, 104, 208)
VRAM_BUDGET_BYTES = int(3.5 * 1024**3)


def write_json_exclusive(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def image_tensor(relative_path: str) -> torch.Tensor:
    row = {"relative_path": relative_path, "label": 0}
    dataset = _ImageDataset(MVTEC.resolve(), [row], 224, MEAN, STD, training=True)
    return dataset[0]["images"][None]


def patch_region(index: int) -> tuple[int, int]:
    return divmod(index, GRID)


def tensor_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    a, b = a.detach().to(torch.float64), b.detach().to(torch.float64)
    difference = (a - b).abs()
    denominator = b.abs().clamp_min(1e-12)
    reference_scale = b.abs().max().clamp_min(1e-12)
    return {
        "shape": list(a.shape),
        "bitwise_equal": bool(torch.equal(a, b)),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "scaled_max_abs": float((difference.max() / reference_scale).item()),
        "relative_l2": float(((a - b).norm() / b.norm().clamp_min(1e-12)).item()),
        "max_elementwise_relative": float((difference / denominator).max().item()),
    }


def leakage_experiment(extractor: NPIMFeatureExtractor, projection: torch.Tensor, image: torch.Tensor) -> Dict[str, Any]:
    """Perturb only model-input patch p; keys for p must not move."""

    results: Dict[str, Any] = {}
    for label, p in (("interior", 14 * GRID + 14), ("corner", 0), ("edge", 13 * GRID + 0)):
        row, col = patch_region(p)
        perturbed = image.clone()
        # +0.5 in normalised units is a large, unambiguous visual perturbation.
        perturbed[0, :, row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8] += 0.5
        z0, y0 = extractor.encode(image)
        z1, y1 = extractor.encode(perturbed)
        k0 = build_keys(z0[0], projection)
        k1 = build_keys(z1[0], projection)
        other = torch.arange(PATCHES) != p
        neighbours = neighbour_table()[p]
        results[label] = {
            "target_patch": int(p),
            "neighbour_count": len(neighbours),
            "neighbours": neighbours,
            "self_in_neighbours": p in neighbours,
            "key_p": tensor_stats(k1[p], k0[p]),
            "key_p_bitwise_unchanged": bool(torch.equal(k1[p], k0[p])),
            "Z_other_patches": tensor_stats(z1[0][other], z0[0][other]),
            "Y_other_patches": tensor_stats(y1[0][other], y0[0][other]),
            "Y_target_patch": tensor_stats(y1[0][p], y0[0][p]),
            "context_contamination_diagnostic_y_other_max_abs": float(
                (y1[0][other] - y0[0][other]).abs().max().item()
            ),
        }
    return results


def autograd_experiment(
    extractor: NPIMFeatureExtractor,
    projection: torch.Tensor,
    image: torch.Tensor,
    target_patch: int = 14 * GRID + 14,
) -> Dict[str, Any]:
    """d key_p / d pixels(p) must be exactly zero; other pixels may receive grad."""

    model = extractor.model
    captured: Dict[str, torch.Tensor] = {}

    def capture_pre(_module, args):
        captured["z"] = args[0]

    hook = model.blocks[0].register_forward_pre_hook(capture_pre)
    differentiable = image.clone().detach().requires_grad_(True)
    try:
        with torch.enable_grad():
            _ = model(differentiable)
            z = captured["z"][:, 1:, :]
            keys = build_keys(z[0], projection)
            loss = keys[target_patch].sum()
            gradient, = torch.autograd.grad(loss, differentiable)
    finally:
        hook.remove()

    row, col = patch_region(target_patch)
    target_gradient = gradient[0, :, row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8]
    outside = gradient[0].clone()
    outside[:, row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8] = 0.0
    return {
        "target_patch": int(target_patch),
        "target_patch_gradient_max_abs": float(target_gradient.abs().max().item()),
        "target_patch_gradient_is_exactly_zero": bool(float(target_gradient.abs().max().item()) == 0.0),
        "outside_patch_gradient_max_abs": float(outside.abs().max().item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
    }


def parity_experiment(extractor: NPIMFeatureExtractor) -> Dict[str, Any]:
    per_row: Dict[str, Any] = {}
    passed = True
    for category in ("bottle", "carpet", "hazelnut"):
        cache_path = CACHE_DIR / f"{category}_train.pt"
        cache = torch.load(cache_path, mmap=True, weights_only=True)
        rows = [r for r in PARITY_ROWS if r < cache["patches"].shape[0]]
        for index in rows:
            relative = cache["relative_paths"][index]
            raw = MVTEC / relative
            raw_sha = sha256_file(raw)
            image = image_tensor(relative)
            _, y = extractor.encode(image)
            reference = cache["patches"][index].to(torch.float32)
            stats = tensor_stats(y[0].cpu(), reference)
            raw_sha_ok = raw_sha == cache["per_image_source_sha256"][index]
            # The repository's own cache marker uses relative-L2 <= 3.0517578125e-05
            # and scaled max-abs <= 1.220703125e-04.  Bitwise equality is not
            # required across timm 1.0.30 (this run) vs 1.0.26 (cache writer).
            row_ok = stats["bitwise_equal"] or (
                stats["relative_l2"] <= 3.0517578125e-05
                and stats["scaled_max_abs"] <= 1.220703125e-04
            )
            passed = passed and raw_sha_ok and row_ok
            per_row[f"{category}:{index}"] = {
                "relative_path": relative,
                "raw_sha256_matches_cache": bool(raw_sha_ok),
                "cache_checkpoint_sha256": cache["metadata"]["checkpoint_sha256"],
                "Y_vs_cache": stats,
                "row_ok": bool(row_ok),
            }
    return {
        "rows": per_row,
        "all_rows_ok": bool(passed),
        "tolerance": {
            "relative_l2": 3.0517578125e-05,
            "scaled_max_abs": 1.220703125e-04,
            "bitwise_required": False,
        },
        "parity_note": "tolerance-defined parity to the immutable block-8 cache; "
                       "bitwise differences are batch/timm-version numerics",
        "cache_checkpoint_pin": "755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415",
    }


def oracle_experiment(projection: torch.Tensor) -> Dict[str, Any]:
    generator = torch.Generator().manual_seed(2711)
    Z = torch.randn(PATCHES, 768, generator=generator)
    vectorised = build_keys(Z, projection)
    loop = build_keys_loop(Z, projection)
    difference = (vectorised - loop).abs().max().item()

    # Explicit target-exclusion probe: replacing Z[p] must not change LOO[p].
    p = 13 * GRID + 5
    Z_modified = Z.clone()
    Z_modified[p] = 1.0e6
    loo_delta = (global_leave_one_out(Z_modified)[p] - global_leave_one_out(Z)[p]).abs().max().item()

    # Numerical degeneracy: all-zero context yields finite zero keys.
    zero_keys = build_keys(torch.zeros(PATCHES, 768), projection)
    return {
        "vectorised_vs_loop_max_abs": float(difference),
        "vectorised_vs_loop_ok": bool(difference <= 1e-5),
        "leave_one_out_ignores_target_max_abs": float(loo_delta),
        "leave_one_out_ignores_target_ok": bool(float(loo_delta) == 0.0),
        "zero_context_keys_are_zero": bool(torch.equal(zero_keys, torch.zeros_like(zero_keys))),
        "zero_context_keys_finite": bool(torch.isfinite(zero_keys).all().item()),
        "projection_sha256": sha256_tensor(projection),
    }


def data_experiment() -> Dict[str, Any]:
    protocol = json.loads(P0C.read_text())
    updates = protocol["updates"]
    probes = protocol["probes"]
    require(len(updates) == 350, "writer stream must contain 350 normal images")
    require(len(probes) == 59, "expected 59 normal witness probes")
    update_paths = {row["relative_path"] for row in updates}
    probe_paths = {row["relative_path"] for row in probes}
    require(not (update_paths & probe_paths), "writer/probe filename overlap")
    update_hashes = {row["raw_png_sha256"] for row in updates}
    probe_hashes = {row["raw_png_sha256"] for row in probes}
    require(not (update_hashes & probe_hashes), "writer/probe content overlap")

    missing, bad = [], []
    for row in updates + probes:
        path = MVTEC / row["relative_path"]
        if not path.is_file():
            missing.append(row["relative_path"])
        elif sha256_file(path) != row["raw_png_sha256"]:
            bad.append(row["relative_path"])
    require(not missing and not bad, f"raw normal images missing/changed: {missing[:3]} {bad[:3]}")
    require(all(row["relative_path"].split("/")[1:3] == ["train", "good"] for row in updates + probes),
            "normal-only role violated")
    boundaries = protocol["arrival_boundaries"]
    counts = {
        "bottle": sum(1 for row in updates if row["category"] == "bottle"),
        "carpet": sum(1 for row in updates if row["category"] == "carpet"),
        "hazelnut": sum(1 for row in updates if row["category"] == "hazelnut"),
    }
    return {
        "writer_images": len(updates),
        "probe_images": len(probes),
        "writer_category_counts": counts,
        "arrival_boundaries": boundaries,
        "writer_probe_disjoint_paths": True,
        "writer_probe_disjoint_content": True,
        "raw_images_and_hashes_verified": True,
        "roles": "train/good only; no anomaly label or mask opened",
        "protocol_source": str(P0C.relative_to(ROOT)),
        "source_manifest_sha256": sha256_file(ROOT / "results/hope_cad/frozen_features_v1/source_manifest.json"),
    }


def cuda_smoke(cfg: NPIMContextConfig, image: torch.Tensor) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"ran": False, "reason": "torch.cuda.is_available() is False"}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    extractor = NPIMFeatureExtractor(cfg, device="cuda")
    started = time.perf_counter()
    z, y = extractor.encode(image.to("cuda"))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    stats = {
        "ran": True,
        "device": torch.cuda.get_device_name(0),
        "elapsed_seconds_one_image": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "vram_budget_bytes": VRAM_BUDGET_BYTES,
        "within_budget": bool(torch.cuda.max_memory_allocated() <= VRAM_BUDGET_BYTES),
        "Z_shape": list(z.shape),
        "Y_shape": list(y.shape),
    }
    extractor.close()
    del extractor
    torch.cuda.empty_cache()
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="EXP-NPIM-00 preflight")
    parser.add_argument("--out", required=True, help="new exclusive JSON output path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cuda-smoke", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.out)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite preflight output: {output_path}")

    started = time.time()
    torch.set_num_threads(8)
    manifest: Dict[str, Any] = {
        "experiment": "EXP-NPIM-00",
        "stage": "A/B preflight",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_sha256": sha256_file(PROTOCOL),
        "protocol_path": str(PROTOCOL.relative_to(ROOT)),
        "git": {
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "head": git("rev-parse", "HEAD"),
            "status_short": git("status", "--short"),
            "diff_stat": git("diff", "--stat"),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "device_requested": args.device,
        },
    }
    try:
        import timm
        manifest["environment"]["timm"] = timm.__version__
    except Exception:  # pragma: no cover
        manifest["environment"]["timm"] = None

    # -- checkpoint identity -------------------------------------------------
    if not CP.is_file():
        manifest["decision"] = "BLOCKED_INPUT_LINEAGE"
        manifest["blocker"] = f"missing frozen checkpoint: {CP}"
        write_json_exclusive(output_path, manifest)
        print(json.dumps({"decision": manifest["decision"], "blocker": manifest["blocker"]}, indent=2))
        return 2
    actual_sha = sha256_file(CP)
    manifest["checkpoint"] = {
        "path": str(CP.relative_to(ROOT)),
        "sha256": actual_sha,
        "bytes": CP.stat().st_size,
        "expected_sha256": CP_SHA,
        "expected_bytes": CP_BYTES,
        "sha_matches": actual_sha == CP_SHA,
        "bytes_match": CP.stat().st_size == CP_BYTES,
        "provisioning_source": "timm/vit_base_patch8_224.augreg_in21k@3c68ba3fef589ef43b862ec19254ecb1146506c3",
    }
    if not (actual_sha == CP_SHA and CP.stat().st_size == CP_BYTES):
        manifest["decision"] = "BLOCKED_INPUT_LINEAGE"
        manifest["blocker"] = "checkpoint identity mismatch"
        write_json_exclusive(output_path, manifest)
        print(json.dumps({"decision": manifest["decision"], "blocker": manifest["blocker"]}, indent=2))
        return 2

    # -- extractor + architecture -------------------------------------------
    cfg = NPIMContextConfig(checkpoint_path=str(CP), checkpoint_sha256=CP_SHA)
    extractor = NPIMFeatureExtractor(cfg, device=args.device)
    manifest["architecture"] = extractor.architecture_manifest

    # -- parity, leakage, oracle --------------------------------------------
    manifest["block8_parity"] = parity_experiment(extractor)
    projection = make_projection(cfg.key_dim, cfg.embed_dim, cfg.projection_seed)
    image = image_tensor("bottle/train/good/000.png")
    manifest["leakage"] = leakage_experiment(extractor, projection, image)
    manifest["autograd"] = autograd_experiment(extractor, projection, image)
    manifest["context_oracle"] = oracle_experiment(projection)
    manifest["data"] = data_experiment()

    if args.cuda_smoke:
        try:
            manifest["cuda_smoke"] = cuda_smoke(cfg, image)
        except Exception as exc:  # report, do not hide
            manifest["cuda_smoke"] = {"ran": False, "error": repr(exc)}

    # -- decision ------------------------------------------------------------
    leakage_ok = all(
        entry["key_p_bitwise_unchanged"] and entry["Z_other_patches"]["bitwise_equal"]
        for entry in manifest["leakage"].values()
    )
    parity_ok = manifest["block8_parity"]["all_rows_ok"]
    oracle_ok = all(
        manifest["context_oracle"][key]
        for key in (
            "vectorised_vs_loop_ok",
            "leave_one_out_ignores_target_ok",
            "zero_context_keys_are_zero",
            "zero_context_keys_finite",
        )
    )
    autograd_ok = manifest["autograd"]["target_patch_gradient_is_exactly_zero"]
    manifest["gates"] = {
        "G0_source_lineage": True,
        "G1_target_leakage": bool(leakage_ok and autograd_ok),
        "G2_geometry_and_Y_parity": bool(parity_ok),
        "context_numerical_oracle": bool(oracle_ok),
    }
    if not leakage_ok:
        manifest["decision"] = "BLOCKED_LEAKAGE_FREE_CONTEXT"
        manifest["blocker"] = "target-patch perturbation changed key_p or another Z"
    elif not parity_ok:
        manifest["decision"] = "BLOCKED_INPUT_LINEAGE"
        manifest["blocker"] = "block-8 Y did not reproduce the immutable cache"
    elif not (oracle_ok and autograd_ok):
        manifest["decision"] = "BLOCKED_LEAKAGE_FREE_CONTEXT"
        manifest["blocker"] = "context numerical/autograd oracle failed"
    else:
        manifest["decision"] = "READY_FOR_SYNTHETIC_MECHANICS"

    extractor.close()
    manifest["elapsed_seconds"] = time.time() - started
    write_json_exclusive(output_path, manifest)
    print(json.dumps(
        {
            "decision": manifest["decision"],
            "checkpoint_sha256": actual_sha,
            "block8_parity": manifest["block8_parity"]["all_rows_ok"],
            "leakage_ok": leakage_ok,
            "autograd_zero": autograd_ok,
            "oracle_ok": oracle_ok,
            "out": str(output_path),
        },
        indent=2,
    ))
    return 0 if manifest["decision"] == "READY_FOR_SYNTHETIC_MECHANICS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
