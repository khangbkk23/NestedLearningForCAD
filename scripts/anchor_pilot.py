"""B323 launcher: ``er_bytes`` and ``anchor`` on the B319 recipe, one GPU each.

``--check`` verifies recipe and inputs without CUDA. ``--parallel`` starts one
child per arm, each pinned to one physical GPU. ``--qualify`` stops both arms at
the registered prefix (second boundary); the full run is refused until the caps
are approved (card B323, section 7).
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

SLOTS = {"er_bytes": "0", "anchor": "1"}
FIXED = {
    "experiment": "B323",
    "base_recipe": "configs/b319.json",
    "seed": 0,
    "arms": ["er_bytes", "anchor"],
    "er_bytes": {"reservoir_capacity": 74},
    "anchor": {"weight": 1.0},
    "comparison": {"candidate": "anchor", "control": "er_bytes"},
    "gate": {
        "final_pixel_AP_gain_min": 0.005,
        "category_pixel_AP_delta_min": -0.01,
        "acquisition_macro_drop_max": 0.01,
        "final_image_AUROC_drop_max": 0.005,
    },
    "prefix_reference": {
        "arm": "anchor",
        "boundary": 209,
        "model_sha256": "84c3e40145656edfbc149e718b9514c0b8f3abb2038977c163677acb8fa8bfd8",
        "source": "B319 er_uniform boundary 1, reproduced by b322-s0-20261115-01",
    },
    "qualification": {"integration_arrivals": 433},
}


def load_pilot(path, digest):
    """Return the validated B323 config; the scientific contract is fixed here."""
    pilot = json.loads(Path(path).read_text(encoding="utf8"))
    changed = [key for key, value in FIXED.items() if pilot.get(key) != value]
    if changed:
        raise ValueError("Changed B323 registered contract: " + ", ".join(changed))
    if digest(ROOT / pilot["base_recipe"]) != pilot["base_recipe_sha256"]:
        raise ValueError("B319 base recipe bytes differ from the registered SHA")
    seconds = pilot.get("caps", {}).get("per_arm_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds < 1:
        raise ValueError("B323 per-arm cap must be a positive integer")
    if pilot.get("caps_status") not in ("PROVISIONAL_PENDING_PROFILE", "APPROVED"):
        raise ValueError("Unknown B323 caps status")
    return pilot


def arm_overrides(pilot, arm, qualify):
    """Config keys added to the B319 recipe for one arm."""
    if arm not in pilot["arms"]:
        raise ValueError("Unregistered B323 arm")
    extra = {"experiment": "B323", "base_recipe_sha256": pilot["base_recipe_sha256"], "qualification": bool(qualify)}
    if arm == "er_bytes":
        extra["reservoir_capacity"] = pilot["er_bytes"]["reservoir_capacity"]
    else:
        extra["anchor"] = dict(pilot["anchor"])
    if qualify:
        extra["integration_arrivals"] = pilot["qualification"]["integration_arrivals"]
    return extra


def gpu_uuids():
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def run_parallel(args, pilot):
    """One child per arm, each on its own physical GPU; both always run to their end."""
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    gpus = gpu_uuids()
    if len(gpus) < 2:
        (root / "result.json").write_text(json.dumps({"status": "BLOCKED_NEEDS_TWO_GPUS", "gpus": gpus}, indent=2) + "\n")
        print("B323_BLOCKED two GPUs are required; found", gpus, flush=True)
        return 2
    (root / "logs").mkdir()
    processes = {}
    for arm in pilot["arms"]:
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--config", str(args.config),
                   "--data", str(args.data), "--arm", arm, "--output", str(root / arm)]
        if args.weights:
            command += ["--weights", str(args.weights)]
        if args.qualify:
            command.append("--qualify")
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=SLOTS[arm], B323_PHYSICAL_GPU_SLOT=SLOTS[arm],
                   NRCAD_LOCAL_ROOT=str(root / "local" / arm))
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf8", errors="replace", bufsize=1)
        processes[arm] = process
        print("B323_ARM_START", arm, "gpu_slot", SLOTS[arm], gpus[int(SLOTS[arm])], flush=True)

    def pump(arm, process):
        with (root / "logs" / (arm + ".log")).open("x", encoding="utf8") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f"[{arm}] {line}", end="", flush=True)

    threads = [threading.Thread(target=pump, args=item, daemon=True) for item in processes.items()]
    for thread in threads:
        thread.start()
    codes = {arm: process.wait() for arm, process in processes.items()}
    for thread in threads:
        thread.join(timeout=30)
    status = "COMPLETE_REVIEW_PENDING" if not any(codes.values()) else "FAILED"
    result = {"status": status, "experiment": "B323", "qualification": bool(args.qualify),
              "exit_codes": codes, "gpus": gpus, "slots": SLOTS}
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print("B323_PARALLEL_RESULT", json.dumps(result), flush=True)
    return 0 if status == "COMPLETE_REVIEW_PENDING" else 1


def main():
    import train
    from shadow_pilot import check_bundled_design

    parser = argparse.ArgumentParser(description="B323: byte-matched ER versus boundary function anchor")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/b323.json")
    parser.add_argument("--data", type=Path, default=train.local_paths.dataset())
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, help="New result directory (required to train)")
    parser.add_argument("--arm", choices=("er_bytes", "anchor"))
    parser.add_argument("--parallel", action="store_true", help="Run both arms, one GPU each")
    parser.add_argument("--qualify", action="store_true", help="Stop at the registered prefix (second boundary)")
    parser.add_argument("--check", action="store_true", help="Verify recipe and inputs only")
    args = parser.parse_args()
    args.config = args.config.resolve()
    try:
        pilot = load_pilot(args.config, train.digest)
        base = types.SimpleNamespace(config=ROOT / pilot["base_recipe"], data=args.data, weights=args.weights)
        config, full, versions, _ = train.load_inputs(base)
    except (OSError, ValueError, RuntimeError, train.importlib.metadata.PackageNotFoundError) as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.check:
        print(json.dumps(dict(status="INPUTS_VERIFIED", experiment="B323", arms=pilot["arms"],
                              caps_status=pilot["caps_status"], runtime=versions), indent=2))
        return 0
    if not args.qualify and pilot["caps_status"] != "APPROVED":
        parser.error("B323 caps are provisional; qualification and approval come first")
    if platform.system() != "Linux":
        parser.error("Training requires Linux GPU")
    if args.output is None or args.output.exists():
        parser.error("Choose a new --output directory")
    if args.parallel == bool(args.arm):
        parser.error("Choose exactly one of --parallel or --arm")
    from nested_replay_cad import provenance

    check_bundled_design(provenance, "B323", "configs/b323.json")
    if args.parallel:
        return run_parallel(args, pilot)
    train.local_paths.process_environment()
    config.update(arm_overrides(pilot, args.arm, args.qualify))
    config["caps"] = {**config["caps"], **pilot["caps"]}
    output = args.output.resolve()
    run = types.SimpleNamespace(arm=args.arm)
    return train.execute(run, config, full, versions, args.config, output, output)


if __name__ == "__main__":
    raise SystemExit(main())
