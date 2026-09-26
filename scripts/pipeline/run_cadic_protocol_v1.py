"""Build the isolated normal-only CADIC-compatible patch coreset."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "conf").is_dir() and (parent / "training").is_dir():
            return parent
    raise RuntimeError("could not find repository root")


ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.load_dataset import ContinualStreamingManager
from models.cadic_patch_coreset_v1 import CADICPatchCoresetConfig, CADICPatchCoresetV1
from models.feature_extractors.cadic_vit_v1 import CADICViTConfig, CADICViTFeatureExtractor
from utils.global_seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "conf/cadic_exact_v1.yaml")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unavailable"


def _load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _extractor_config(config: Dict[str, Any]) -> CADICViTConfig:
    raw = dict(config.get("extractor", {}))
    for key in ("mean", "std"):
        if key in raw:
            raw[key] = tuple(raw[key])
    checkpoint = Path(str(raw.get("checkpoint_path", "")))
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    raw["checkpoint_path"] = str(checkpoint)
    return CADICViTConfig(**raw)


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    train_cfg = config.setdefault("training", {})
    set_seed(int(train_cfg.get("seed", 42)))

    requested_device = str(train_cfg.get("device", "cuda"))
    device = requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu"
    if not bool(config.get("protocol", {}).get("exact_claim_allowed", False)):
        print("[CADIC v1] protocol is explicitly marked compatible-foundation; exact claim disabled")

    extractor = CADICViTFeatureExtractor(_extractor_config(config), device=device)
    memory_cfg = config.get("memory", {})
    budget = int(args.budget or memory_cfg.get("budget", 10000))
    coreset = CADICPatchCoresetV1(
        CADICPatchCoresetConfig(
            budget=budget,
            dim=int(config["extractor"].get("embed_dim", 768)),
            dtype=str(memory_cfg.get("dtype", "float32")),
            distance=str(memory_cfg.get("distance", "euclidean")),
            chunk_size=int(memory_cfg.get("chunk_size", 2048)),
            image_neighbors=int(memory_cfg.get("image_neighbors_b", 9)),
        ),
        device=device,
    )

    dataset_cfg = config.setdefault("dataset", {})
    dataset_cfg["cadic_parity"] = True
    dataset_cfg["synthetic_anomaly"] = False
    dataset_cfg["load_test_during_stream"] = False
    if args.max_tasks is not None:
        order = dataset_cfg.get("class_order", [])
        dataset_cfg["class_order"] = list(order)[: int(args.max_tasks)]
    stream = ContinualStreamingManager(config)
    records = []
    stream_start = time.perf_counter()

    while True:
        train_loader, test_loader, task_info = stream.get_next_task()
        if train_loader is None:
            break
        if test_loader is not None:
            raise RuntimeError("CADIC parity stream unexpectedly loaded official test data")
        task_start = time.perf_counter()
        streamed = 0
        eligible = 0
        accepted = replaced = rejected = 0
        feature_seconds = update_seconds = 0.0
        for batch in train_loader:
            images = batch["img"].to(device)
            labels = batch["anomaly"]
            if bool(torch.any(labels != 0)):
                raise RuntimeError("CADIC parity stream contains a non-normal sample")
            streamed += int(images.shape[0])
            eligible += int(images.shape[0])
            start = time.perf_counter()
            patches = extractor.extract_patch_features(images)
            feature_seconds += time.perf_counter() - start
            start = time.perf_counter()
            update = coreset.update(patches)
            update_seconds += time.perf_counter() - start
            accepted += int(update["accepted"])
            replaced += int(update["replaced"])
            rejected += int(update["rejected"])
        records.append({
            "task_id": int(task_info["task_id"]),
            "category": str(task_info["category"]),
            "train_files_on_disk": int(task_info["train_file_count"]),
            "train_samples_streamed": streamed,
            "eligible_normal_samples": eligible,
            "drop_last": bool(task_info["train_drop_last"]),
            "accepted_patch_vectors": accepted,
            "replaced_patch_vectors": replaced,
            "rejected_patch_vectors": rejected,
            "feature_seconds": feature_seconds,
            "coreset_update_seconds": update_seconds,
            "task_seconds": time.perf_counter() - task_start,
            "coreset": coreset.stats(),
        })
        if not args.quiet:
            print(json.dumps(records[-1], indent=2))

    output_dir = args.output_dir or (ROOT / str(config.get("logging", {}).get("results_dir", "results")) / f"CADIC_v1_{budget}")
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "schema": "cadic_protocol_state_v1",
        "git_commit": _git_commit(),
        "resolved_config": config,
        "extractor": extractor.protocol_metadata(),
        "coreset": coreset.state_dict(),
        "tasks": records,
        "runtime": {"stream_seconds": time.perf_counter() - stream_start},
        "memory": coreset.stats(),
    }
    torch.save(state, output_dir / "cadic_state.pt")
    (output_dir / "cadic_train_records.json").write_text(
        json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"CADIC v1 foundation state written to {output_dir / 'cadic_state.pt'}")


if __name__ == "__main__":
    main()
