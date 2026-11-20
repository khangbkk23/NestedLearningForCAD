"""B324 launcher: uint8 input replay and update schedules on the B319 recipe.

``--check`` verifies recipe and inputs without CUDA. ``--job`` runs the two
arms of a registered job in parallel, one physical GPU each. ``--arm`` with
``--seed`` runs one arm (used by ``--job`` for its children). Job ``q0`` is a
prefix identity check and may run while the caps are provisional; every other
job needs approved caps (card B324, section 7).
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

ARMS = ("q0", "raw1", "flat", "nested", "iid")
FLAT = {"mode": "flat", "extra_per_arrival": 2, "samples": 4}
NESTED = {"mode": "nested", "every": 64, "updates": 128, "samples": 5, "tail": 90, "final_arrival": 3629}
SCHEDULES = {"q0": {"mode": "single"}, "raw1": {"mode": "single"}, "flat": FLAT, "nested": NESTED, "iid": FLAT}
# Registered caps (card section 4): fast step = 1 current + up to 4 stored samples.
SUPPORT_CAP = {"q0": 18145, "raw1": 18145, "flat": 54435, "iid": 54435, "nested": 54435}
REPLAY_ENCODER_CAP = {"q0": 14516, "raw1": 14516, "flat": 43548, "iid": 43548, "nested": 50806}


def contract():
    """The registered scientific contract: configs/b324.json minus the caps fields."""
    return {
        "experiment": "B324",
        "base_recipe": "configs/b319.json",
        "seeds": [0, 1],
        "replay": {"format": "uint8_chw_392", "bytes_per_sample": 460992, "budget_bytes": 155123712, "fifo": 8},
        "arms": {
            "q0": {"reservoir_capacity": 24, "updates_per_arrival": 1, "total_updates": 3629},
            "raw1": {"reservoir_capacity": 328, "updates_per_arrival": 1, "total_updates": 3629},
            "flat": {"reservoir_capacity": 328, "updates_per_arrival": 3, "total_updates": 10887},
            "nested": {"reservoir_capacity": 328, "updates_per_arrival": 1,
                       "consolidation": {"every_arrivals": 64, "updates": 128, "samples_per_update": 5,
                                         "tail_updates": 90},
                       "total_updates": 10887},
            "iid": {"reservoir_capacity": 328, "updates_per_arrival": 3, "shuffled_stream": True,
                    "observers": "final_only", "total_updates": 10887},
        },
        "jobs": {
            "q0": {"arms": ["q0", "nested"], "seed": 0, "integration_arrivals": 433},
            "s1": {"arms": ["raw1", "iid"], "seed": 0},
            "s2a": {"arms": ["flat", "nested"], "seed": 0},
            "s2b": {"arms": ["flat", "nested"], "seed": 1},
            "r1": {"arms": ["q0", "raw1"], "seed": 1},
        },
        "identity_reference": {
            "arm": "q0", "seed": 0,
            "model_sha256": {"209": "84c3e40145656edfbc149e718b9514c0b8f3abb2038977c163677acb8fa8bfd8",
                             "433": "2d4401de4b9d4a664116fc1003189d9cdbaaf26fdd7331da7c1d1eaf4ad462fc"},
            "source": "B319 er_uniform boundaries 1 and 2, reproduced by b322-s0-20261115-01",
        },
        "s1_prediction": {"macro_Facq_max": 0.01, "final_macro_pixel_AP_min": 0.62},
        "s2_gate": {"final_pixel_AP_gain_min": 0.005, "category_pixel_AP_delta_min": -0.01,
                    "final_image_AUROC_drop_max": 0.005, "adds_nothing_abs_mean_gain_below": 0.005},
        "r1_rule": {"final_pixel_AP_gain_min": 0.005, "raw1_macro_Facq_max": 0.01},
    }


def load_pilot(path, digest):
    pilot = json.loads(Path(path).read_text(encoding="utf8"))
    changed = [key for key, value in contract().items() if pilot.get(key) != value]
    if changed:
        raise ValueError("Changed B324 registered contract: " + ", ".join(changed))
    if digest(ROOT / pilot["base_recipe"]) != pilot["base_recipe_sha256"]:
        raise ValueError("B319 base recipe bytes differ from the registered SHA")
    seconds = pilot.get("caps", {}).get("per_arm_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds < 1:
        raise ValueError("B324 per-arm cap must be a positive integer")
    if pilot.get("caps_status") not in ("PROVISIONAL_PENDING_PROFILE", "APPROVED"):
        raise ValueError("Unknown B324 caps status")
    from nested_replay_cad import coverage

    nested = coverage.validate_schedule(dict(NESTED))
    for arm, spec in pilot["arms"].items():
        if coverage.expected_updates(SCHEDULES[arm] if arm != "nested" else nested, 3629) != spec["total_updates"]:
            raise ValueError("Registered total updates differ from the schedule: " + arm)
        if (spec["reservoir_capacity"] + pilot["replay"]["fifo"]) * pilot["replay"]["bytes_per_sample"] \
                > pilot["replay"]["budget_bytes"]:
            raise ValueError("Replay memory exceeds the byte budget: " + arm)
    return pilot


def arm_overrides(pilot, arm, seed, job=None):
    """Config keys added to the B319 recipe for one arm of B324."""
    if arm not in ARMS or seed not in pilot["seeds"]:
        raise ValueError("Unregistered B324 arm or seed")
    if job is not None and (arm not in pilot["jobs"][job]["arms"] or seed != pilot["jobs"][job]["seed"]):
        raise ValueError("Arm or seed not registered for this job")
    spec = pilot["arms"][arm]
    extra = {
        "experiment": "B324",
        "base_recipe_sha256": pilot["base_recipe_sha256"],
        "seed": seed,
        "replay_format": pilot["replay"]["format"],
        "replay_budget_bytes": pilot["replay"]["budget_bytes"],
        "reservoir_capacity": spec["reservoir_capacity"],
        "schedule": dict(SCHEDULES[arm]),
        "total_updates": spec["total_updates"],
        "support_forwards_cap": SUPPORT_CAP[arm],
        "replay_encoder_cap": REPLAY_ENCODER_CAP[arm],
        "job": job,
    }
    if spec.get("shuffled_stream"):
        extra.update(shuffled_stream=True, observe_at=[3629])
    if job is not None and pilot["jobs"][job].get("integration_arrivals"):
        extra["integration_arrivals"] = pilot["jobs"][job]["integration_arrivals"]
    return extra


def gpu_uuids():
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def run_job(args, pilot):
    spec = pilot["jobs"][args.job]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    gpus = gpu_uuids()
    if len(gpus) < 2:
        (root / "result.json").write_text(json.dumps({"status": "BLOCKED_NEEDS_TWO_GPUS", "gpus": gpus}, indent=2) + "\n")
        print("B324_BLOCKED two GPUs are required; found", gpus, flush=True)
        return 2
    (root / "logs").mkdir()
    processes, slots = {}, {}
    for slot, arm in enumerate(spec["arms"]):
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--config", str(args.config),
                   "--data", str(args.data), "--arm", arm, "--seed", str(spec["seed"]), "--job", args.job,
                   "--output", str(root / arm)]
        if args.weights:
            command += ["--weights", str(args.weights)]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(slot), B324_PHYSICAL_GPU_SLOT=str(slot),
                   NRCAD_LOCAL_ROOT=str(root / "local" / arm))
        processes[arm] = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True, encoding="utf8",
                                          errors="replace", bufsize=1)
        slots[arm] = slot
        print("B324_ARM_START", args.job, arm, "seed", spec["seed"], "gpu_slot", slot, gpus[slot], flush=True)

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
    result = {"status": status, "experiment": "B324", "job": args.job, "seed": spec["seed"],
              "exit_codes": codes, "gpus": gpus, "slots": slots}
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print("B324_PARALLEL_RESULT", json.dumps(result), flush=True)
    return 0 if status == "COMPLETE_REVIEW_PENDING" else 1


def main():
    import train
    from shadow_pilot import check_bundled_design

    parser = argparse.ArgumentParser(description="B324: uint8 input replay and update schedules")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/b324.json")
    parser.add_argument("--data", type=Path, default=train.local_paths.dataset())
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, help="New result directory (required to train)")
    parser.add_argument("--job", choices=("q0", "s1", "s2a", "s2b", "r1"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
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
        print(json.dumps(dict(status="INPUTS_VERIFIED", experiment="B324", caps_status=pilot["caps_status"],
                              runtime=versions), indent=2))
        return 0
    if args.job is None:
        parser.error("Choose a registered --job")
    if args.job != "q0" and pilot["caps_status"] != "APPROVED":
        parser.error("B324 caps are provisional; only job q0 may run before approval")
    if platform.system() != "Linux":
        parser.error("Training requires Linux GPU")
    if args.output is None or args.output.exists():
        parser.error("Choose a new --output directory")
    from nested_replay_cad import provenance

    check_bundled_design(provenance, "B324", "configs/b324.json")
    if args.arm is None:
        return run_job(args, pilot)
    if args.seed is None:
        parser.error("--arm needs --seed")
    train.local_paths.process_environment()
    config.update(arm_overrides(pilot, args.arm, args.seed, args.job))
    config["caps"] = {**config["caps"], **pilot["caps"]}
    output = args.output.resolve()
    run = types.SimpleNamespace(arm=args.arm)
    return train.execute(run, config, full, versions, args.config, output, output)


if __name__ == "__main__":
    raise SystemExit(main())
