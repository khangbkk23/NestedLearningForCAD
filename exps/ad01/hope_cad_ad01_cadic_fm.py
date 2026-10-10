# exps/ad01/hope_cad_ad01_cadic_fm.py
"""Bounded CADIC forgetting measurement on an isolated evaluation copy.

The archived 15-task run has 16 saved states but never had its forgetting
matrix computed: the benchmark's `fm` phase was never invoked, so the project
has no forgetting number for any method.

A full 15x15 triangular matrix would require 120 state/task evaluations over
1725 test images. On CPU that is roughly 20 hours, which is not a reasonable
use of this task's budget, so this module computes a *bounded* slice instead:

    states  : after bottle (0), after carpet (3), after hazelnut (5), final (14)
    tasks   : bottle, carpet, hazelnut

That is 12 state/task pairs over 310 test images and directly serves the AD-01
question, because the four-arm experiment uses exactly those three tasks. The
slice is reported as bounded; it is not the full continual matrix.

Every archived artifact is treated as read-only. The evaluation copy holds
hard-linked states whose content is byte-identical to the originals, and the
original file identities are verified before and after.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from exps.ad01.hope_cad_ad01_phase0 import ReferenceRun, _protocol

ROOT = Path(__file__).resolve().parents[2]

REFERENCE_RUN = (
    ROOT / "results/benchmarks/mvtec_1x15_v1/cadic_compatible_v1/pf10k_20260927T155859Z"
)
COPY_ROOT = ROOT / "results/hope_cad/ad01_phase0/fm_cadic_pf10k"
EVAL_TASKS = ("bottle", "carpet", "hazelnut")


def state_identity(path: Path) -> dict[str, Any]:
    """Cheap but strong identity: size, mtime, inode, and head/tail hashes."""
    import hashlib

    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(1 << 20))
        stream.seek(max(0, stat.st_size - (1 << 20)))
        digest.update(stream.read(1 << 20))
    return {
        "name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
        "head_tail_sha256": digest.hexdigest(),
    }


def snapshot_states(run_dir: Path) -> list[dict[str, Any]]:
    return [state_identity(path) for path in sorted((run_dir / "states").glob("*.pt"))]


def build_evaluation_copy(force: bool = False) -> dict[str, Any]:
    """Create an isolated, read-only-by-convention evaluation copy.

    Config and manifest files are copied (they are small and must not be
    mutated). State files are hard-linked so the copy consumes no additional
    disk while remaining byte-identical; the link count is recorded so the
    reader can confirm the originals were never rewritten in place.
    """
    if not REFERENCE_RUN.is_dir():
        raise FileNotFoundError(REFERENCE_RUN)
    COPY_ROOT.mkdir(parents=True, exist_ok=True)

    before = snapshot_states(REFERENCE_RUN)
    source_state_dir = REFERENCE_RUN / "states"
    target_state_dir = COPY_ROOT / "states"
    target_state_dir.mkdir(parents=True, exist_ok=True)

    for name in (
        "protocol_resolved.yaml",
        "method_resolved.yaml",
        "manifest_train.json",
        "run.json",
        "git.json",
        "environment.json",
    ):
        source = REFERENCE_RUN / name
        if source.is_file():
            (COPY_ROOT / name).write_bytes(source.read_bytes())

    linked, copied = [], []
    for source in sorted(source_state_dir.glob("*.pt")):
        target = target_state_dir / source.name
        if target.exists() and not force:
            linked.append(source.name)
            continue
        if target.exists():
            target.unlink()
        try:
            os.link(source, target)
            linked.append(source.name)
        except OSError:
            target.write_bytes(source.read_bytes())
            copied.append(source.name)

    link_counts = {
        path.name: path.stat().st_nlink for path in sorted(target_state_dir.glob("*.pt"))
    }
    after = snapshot_states(REFERENCE_RUN)
    if before != after:
        raise RuntimeError("original state files changed while building the copy")

    report = {
        "reference_run": str(REFERENCE_RUN.relative_to(ROOT)),
        "copy_root": str(COPY_ROOT.relative_to(ROOT)),
        "hard_linked": linked,
        "copied": copied,
        "hard_link_counts": link_counts,
        "originals_unchanged": True,
        "originals": before,
        "note": (
            "state files are hard links; they are opened read-only and never "
            "written by this module"
        ),
    }
    (COPY_ROOT / "copy_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


@torch.no_grad()
def score_task(adapter, protocol_obj, task_id: int, b: int) -> dict[str, Any]:
    """Score one official test task and return image scores, labels and masks."""
    loader = protocol_obj.build_test_loader(task_id)
    image_scores, labels, pixel_scores, masks = [], [], [], []
    for batch in loader:
        features = adapter.extractor.extract_patch_features(batch["images"].to(adapter.device))
        images, pixel = adapter.coreset.score(features, b=b)
        image_scores.append(images.cpu())
        pixel_scores.append(pixel.cpu())
        labels.append(batch["labels"].clone())
        masks.append(batch["masks"].clone())
    return {
        "image_scores": torch.cat(image_scores).numpy().astype(np.float64),
        "labels": torch.cat(labels).numpy().astype(np.int64),
        "pixel_scores": torch.cat(pixel_scores).numpy().astype(np.float64),
        "masks": torch.cat(masks).numpy().astype(bool),
        "paths": None,
    }


def metrics_for(result: dict[str, Any]) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = result["labels"]
    masks = result["masks"]
    native_scores, native_masks = [], []
    for index in range(result["pixel_scores"].shape[0]):
        mask = masks[index]
        resized = torch.nn.functional.interpolate(
            torch.from_numpy(result["pixel_scores"][index]).reshape(1, 1, 28, 28),
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        native_scores.append(resized.reshape(-1))
        native_masks.append(mask.reshape(-1))

    # The 28x28 diagnostic must compare a 28x28 score against a 28x28 mask;
    # using the native mask here would be a shape mismatch, not a metric.
    masks28 = torch.nn.functional.interpolate(
        torch.from_numpy(masks).float().unsqueeze(1), size=(28, 28), mode="nearest"
    ).squeeze(1).numpy().astype(bool)

    return {
        "i_auroc": float(roc_auc_score(labels, result["image_scores"])),
        "p_aupr_native": float(
            average_precision_score(np.concatenate(native_masks), np.concatenate(native_scores))
        ),
        "p_aupr_grid28": float(
            average_precision_score(
                masks28.reshape(-1), result["pixel_scores"].reshape(-1)
            )
        ),
        "n_images": int(labels.size),
        "n_normal": int((labels == 0).sum()),
        "n_defect": int((labels == 1).sum()),
    }


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Bounded CADIC forgetting slice")
    parser.add_argument("--stage", choices=["copy", "fm", "all"], default="all")
    parser.add_argument("--force-copy", action="store_true")
    args = parser.parse_args()

    if args.stage in ("copy", "all"):
        report = build_evaluation_copy(force=args.force_copy)
        print(json.dumps({k: v for k, v in report.items() if k != "originals"}, indent=2))

    if args.stage == "copy":
        return

    reference = ReferenceRun.load(REFERENCE_RUN)
    copy_run = ReferenceRun.load(COPY_ROOT)
    order = list(reference.protocol["task_order"])
    b = int(reference.method["scoring"]["image_neighbors_b"])

    states = {}
    for name in ("task_00.pt", "task_03.pt", "task_05.pt", "final.pt"):
        path = COPY_ROOT / "states" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        states[name] = path

    protocol_obj = _protocol(reference)
    results: dict[str, Any] = {"slice": {}, "states": list(states), "tasks": list(EVAL_TASKS)}
    started = time.perf_counter()

    def learned_after(state_name: str) -> str:
        """Task whose learning produced this state."""
        if state_name == "final.pt":
            return f"all {len(order)} (final)"
        index = int(state_name.removeprefix("task_").removesuffix(".pt"))
        return order[index]

    for state_name, state_path in states.items():
        adapter = copy_run.load_adapter("cpu")
        adapter.load_state_dict(
            torch.load(state_path, map_location="cpu", weights_only=False)
        )
        for task in EVAL_TASKS:
            task_id = order.index(task)
            begin = time.perf_counter()
            scored = score_task(adapter, protocol_obj, task_id, b)
            entry = metrics_for(scored)
            entry["seconds"] = time.perf_counter() - begin
            results["slice"][f"{state_name}|{task}"] = entry
            print(
                f"[FM] after {learned_after(state_name):<18} on {task:<9} "
                f"I-AUROC={entry['i_auroc']:.4f} P-AUPR(native)={entry['p_aupr_native']:.4f} "
                f"P-AUPR(28)={entry['p_aupr_grid28']:.4f} {entry['seconds']:.1f}s",
                flush=True,
            )

    results["seconds_total"] = time.perf_counter() - started

    matrix = {}
    for task in EVAL_TASKS:
        row = {}
        for state_name in states:
            key = f"{state_name}|{task}"
            row[state_name] = results["slice"][key]["i_auroc"]
        matrix[task] = row
    results["i_auroc_matrix"] = matrix

    after = snapshot_states(REFERENCE_RUN)
    manifest = json.loads((COPY_ROOT / "copy_manifest.json").read_text())
    results["originals_unchanged_after_fm"] = after == manifest["originals"]

    (COPY_ROOT / "fm_slice.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "slice"}, indent=2))


if __name__ == "__main__":
    main()
