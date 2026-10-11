"""Run HOPE-CAD v3 through the shared continual benchmark engine."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from dataset.benchmark_manifest_v1 import canonical_digest
from dataset.benchmark_protocol_v2 import PrefixContinualProtocol, file_hash, validate_prefix_manifest
from models.hope_cad_v2.benchmark_adapter import HOPEBenchmarkAdapterV3
from scripts.hope_cad_v2.artifacts import provenance, write_json
from scripts.hope_cad_v2.run_matched_cadic import baseline_identity
from training.benchmark_artifacts_v1 import BenchmarkArtifacts
from training.benchmark_config_v1 import load_configs
from training.benchmark_config_v2 import validate_configs
from training.benchmark_engine_v1 import BenchmarkEngineV1


def _plain(value):
    return json.loads(json.dumps(value))


def _load_context(args):
    protocol, baseline_method = load_configs(args.protocol, args.baseline_method)
    resolved = validate_configs(protocol, baseline_method)
    manifest = json.loads((args.lock_dir / "manifest_train.json").read_text(encoding="utf-8"))
    validate_prefix_manifest(manifest, protocol, verify_bytes=True)
    expected = baseline_identity(
        protocol,
        baseline_method,
        resolved,
        manifest["digest"],
        seed=args.seed,
        smoke=False,
        max_tasks=None,
        max_train_images=None,
    )
    expected_lock = {"identity": expected, "digest": canonical_digest(expected)}
    actual_lock = json.loads((args.lock_dir / "baseline_lock.json").read_text(encoding="utf-8"))
    if actual_lock != expected_lock:
        raise ValueError("shared baseline lock is stale or differs from the requested protocol")
    return protocol, baseline_method, resolved, manifest, actual_lock


def _build_adapter(args, baseline_method, lock):
    initialization = torch.load(args.initialization, map_location="cpu", weights_only=False)
    protocol_identity = {
        "protocol_id": lock["identity"]["protocol"]["id"],
        "manifest_digest": lock["identity"]["manifest_digest"],
        "baseline_lock_digest": lock["digest"],
    }
    return HOPEBenchmarkAdapterV3(
        initialization,
        checkpoint_path=baseline_method["extractor"]["checkpoint_path"],
        checkpoint_sha256=baseline_method["extractor"]["checkpoint_sha256"],
        checkpoint_identity=baseline_method["extractor"]["checkpoint_identity"],
        device=args.device,
        protocol_identity=protocol_identity,
    ), initialization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path,
                        default=Path("conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml"))
    parser.add_argument("--baseline-method", type=Path,
                        default=Path("conf/benchmarks/methods/cadic_matched_prefix_v2.yaml"))
    parser.add_argument("--lock-dir", type=Path, required=True,
                        help="prepared shared-protocol run directory; baseline is not executed")
    parser.add_argument("--initialization", type=Path, required=True,
                        help="selected W6 deployment_initialization.pt")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "train", "eval", "fm"), default="prepare")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.seed != 0:
        parser.error("the locked MVTec main run currently requires seed 0")
    if not args.initialization.is_file():
        parser.error("selected deployment initialization does not exist")
    protocol, baseline_method, resolved, manifest, lock = _load_context(args)
    adapter, initialization = _build_adapter(args, baseline_method, lock)
    method_metadata = _plain(adapter.method_metadata())
    run_identity = {
        "schema": "hope_cad_benchmark_lock_v3",
        "design_id": "hope_cad_v3_masked_teacher_cms",
        "protocol_id": protocol["id"],
        "baseline_lock_digest": lock["digest"],
        "manifest_digest": manifest["digest"],
        "initialization_sha256": initialization["initial_cms_sha256"],
        "initialization_file_sha256": file_hash(args.initialization),
        "method_metadata": method_metadata,
    }
    run_lock = {"identity": run_identity, "digest": canonical_digest(run_identity)}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.run_dir / "hope_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != run_lock:
        raise ValueError("HOPE run lock differs; refuse source/init/protocol drift")
    if args.phase != "prepare" and not lock_path.exists():
        raise ValueError("prepare and lock the HOPE run before executing")

    artifacts = BenchmarkArtifacts(args.run_dir)
    evidence = provenance()
    if args.phase == "prepare":
        if not lock_path.exists():
            write_json(lock_path, run_lock)
            artifacts.initialize(protocol, method_metadata, manifest, evidence, False, False)
            write_json(args.run_dir / "shared_baseline_lock.json", lock)
            write_json(args.run_dir / "initialization_provenance.json", initialization["provenance"])
            write_json(args.run_dir / "run.json", {
                "status": "prepared_not_run",
                "reportable": False,
                "design_id": run_identity["design_id"],
                "protocol_id": protocol["id"],
                "baseline_lock_digest": lock["digest"],
                "initialization_sha256": initialization["initial_cms_sha256"],
            })
        print(f"HOPE run locked: {lock_path}")
        adapter.close()
        return 0

    dataset = PrefixContinualProtocol(protocol, manifest, baseline_method, args.seed)
    engine = BenchmarkEngineV1(protocol, adapter, artifacts, args.device,
                               fail_on_state_mutation=True)
    if args.phase == "train":
        if any(artifacts.states.iterdir()) and not args.resume:
            raise ValueError("existing task states require explicit --resume")
        outcome = engine.train(dataset, resume=args.resume)
    else:
        if args.phase == "eval":
            final_state = torch.load(artifacts.states / "final.pt", map_location=args.device,
                                     weights_only=False)
            adapter.load_state_dict(final_state)
            outcome = engine.evaluate_final(dataset)
        else:
            outcome = engine.evaluate_forgetting(dataset)
    evidence = provenance()
    write_json(args.run_dir / f"{args.phase}_evidence.json", {
        "status": "phase_complete",
        "reportable": False,
        "source": evidence,
        "run_lock_digest": run_lock["digest"],
        "method_metadata": method_metadata,
        "timings": engine.times,
        "memory": adapter.memory_stats(),
    })
    adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())