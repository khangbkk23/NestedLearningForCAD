import os
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conf.config import load_config
from models.cbp import CBPConfig
from models.null_space_proj import NSP2Config
from training.checkpointing import CheckpointManager
from training.consolidation_engine import NestedBackboneConsolidator, Phase3Config
from training.run_experiment import build_model
from utils.global_seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 N2B-NC consolidation")
    parser.add_argument("--config", type=str, default="conf/config_phase3.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_suffix", type=str, default="phase3")
    return parser.parse_args()


def _phase3_config(config: Dict[str, Any]) -> Phase3Config:
    cfg = config.get("phase3", {})
    return Phase3Config(
        top_k_anchors=int(cfg.get("top_k_anchors", 32)),
        unfreeze_last_blocks=int(cfg.get("unfreeze_last_blocks", 2)),
        drift_threshold=float(cfg.get("drift_threshold", 0.05)),
        grad_clip=float(cfg.get("grad_clip", 1.0)),
        distill_weight=float(cfg.get("distill_weight", 1.0)),
        lejepa_weight=float(cfg.get("lejepa_weight", 0.1)),
        lr=float(cfg.get("lr", 1e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        steps=int(cfg.get("steps", 1)),
        refresh_coreset=bool(cfg.get("refresh_coreset", True)),
        refresh_batch_size=int(cfg.get("refresh_batch_size", 32)),
    )


def _nsp2_config(config: Dict[str, Any]) -> NSP2Config:
    cfg = config.get("nsp2", {})
    return NSP2Config(
        enabled=bool(cfg.get("enabled", False)),
        energy_threshold=float(cfg.get("energy_threshold", 0.99)),
        min_null_dim=int(cfg.get("min_null_dim", 64)),
    )


def _cbp_config(config: Dict[str, Any]) -> CBPConfig:
    cfg = config.get("cbp", {})
    return CBPConfig(
        enabled=bool(cfg.get("enabled", False)),
        monitor_only=bool(cfg.get("monitor_only", True)),
        threshold=float(cfg.get("threshold", 0.01)),
        reinit_std=float(cfg.get("reinit_std", 0.02)),
    )


def _make_output_dir(config: Dict[str, Any], output_dir: str | None, suffix: str) -> Path:
    if output_dir:
        path = Path(output_dir)
    else:
        logging_cfg = config.get("logging", {})
        root = Path(logging_cfg.get("results_dir", "results"))
        exp_name = logging_cfg.get("experiment_name", "MetaNATH_Phase3")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = root / f"{exp_name}_{timestamp}_{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("training", {}).get("seed", 42)))

    output_dir = _make_output_dir(config, args.output_dir, args.run_suffix)
    with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    training_cfg = config.get("training", {})
    requested_device = str(training_cfg.get("device", "cuda"))
    device = requested_device if torch.cuda.is_available() or requested_device == "cpu" else "cpu"

    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_full_state_dict(checkpoint["model_state_dict"])

    consolidator = NestedBackboneConsolidator(
        core_model=model,
        phase3_config=_phase3_config(config),
        nsp2_config=_nsp2_config(config),
        cbp_config=_cbp_config(config),
        device=device,
    )
    log = consolidator.execute_global_consolidation()

    with open(output_dir / "consolidation_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    checkpoint_manager = CheckpointManager(
        run_dir=str(output_dir),
        checkpoint_mode=str(config.get("logging", {}).get("checkpoint_mode", "phase3_full")),
        checkpoint_policy=str(config.get("logging", {}).get("checkpoint_policy", "last_only")),
        save_models=bool(config.get("logging", {}).get("save_models", True)),
    )
    saved_paths = checkpoint_manager.save_task(
        model=model,
        config=config,
        task_id=int(checkpoint.get("task_id", -1)),
        category=str(checkpoint.get("category", "phase3")),
        eval_metrics={},
    )

    summary = {
        "output_dir": str(output_dir),
        "source_checkpoint": str(args.checkpoint),
        "saved_checkpoints": saved_paths,
        **log,
    }
    with open(output_dir / "phase3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Phase 3 completed. Summary: {summary}")


if __name__ == "__main__":
    main()
