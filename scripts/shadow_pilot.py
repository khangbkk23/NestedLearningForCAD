"""B322 launcher: the B319 er_uniform arm with passive EMA shadow readouts.

``--check`` verifies the recipe and inputs without CUDA. Training is refused
until the B322 caps are approved after a measured profile (card section 8).
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

FIXED = {
    "experiment": "B322",
    "base_recipe": "configs/b319.json",
    "seed": 0,
    "arm": "er_uniform",
    "shadow": {"taus": [512, 128], "primary": 512, "secondary": 128},
    "gate": {
        "final_pixel_AP_gain_min": 0.005,
        "Fmax_reduction_min": 0.01,
        "acquisition_macro_drop_max": 0.01,
        "acquisition_category_drop_max": 0.03,
        "final_image_AUROC_drop_max": 0.005,
    },
    "historical_screen": {
        "pixel_AP_gain_min": 0.005,
        "category_pixel_AP_delta_min": -0.01,
        "category_image_AUROC_delta_min": -0.005,
    },
}


def load_pilot(path, digest):
    """Return the validated B322 config; the scientific contract is fixed here."""
    pilot = json.loads(Path(path).read_text(encoding="utf8"))
    changed = [key for key, value in FIXED.items() if pilot.get(key) != value]
    if changed:
        raise ValueError("Changed B322 registered contract: " + ", ".join(changed))
    if digest(ROOT / pilot["base_recipe"]) != pilot["base_recipe_sha256"]:
        raise ValueError("B319 base recipe bytes differ from the registered SHA")
    seconds = pilot.get("caps", {}).get("per_arm_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds < 1:
        raise ValueError("B322 per-arm cap must be a positive integer")
    if pilot.get("caps_status") not in ("PROVISIONAL_PENDING_PROFILE", "APPROVED"):
        raise ValueError("Unknown B322 caps status")
    return pilot


def check_bundled_design(provenance, experiment="B322", recipe="configs/b322.json"):
    """Freeze card and recipe. In a bundle without git, compare them with SOURCE.json.

    The packager records the attempt plan in ``design_hashes`` too, so the strict
    equality of ``provenance.check_design`` cannot hold there; card and recipe
    are compared as a subset instead.
    """
    try:
        return provenance.check_design(ROOT, experiment, recipe)
    except ValueError:
        origin = ROOT / "SOURCE.json"
        if not origin.is_file():
            raise
        info = json.loads(origin.read_text(encoding="utf8"))
        recorded = info.get("design_hashes") or {}
        for name in (f"experiments/{experiment}/card.md", recipe):
            if recorded.get(name) != provenance.digest(ROOT / name):
                raise ValueError(f"Bundled {experiment} design differs from SOURCE.json: " + name)
        if info.get("git", {}).get("dirty") is not False or not info.get("design_commit"):
            raise ValueError(f"Bundled {experiment} source is not a committed design")
        return info["design_commit"]


def main():
    import train

    parser = argparse.ArgumentParser(description="B322: B319 er_uniform with passive EMA shadows")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/b322.json")
    parser.add_argument("--data", type=Path, default=train.local_paths.dataset())
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, help="New result directory (required to train)")
    parser.add_argument("--check", action="store_true", help="Verify recipe and inputs only")
    args = parser.parse_args()
    try:
        pilot = load_pilot(args.config, train.digest)
        base = types.SimpleNamespace(
            config=ROOT / pilot["base_recipe"], data=args.data, weights=args.weights
        )
        config, full, versions, _ = train.load_inputs(base)
    except (OSError, ValueError, RuntimeError, train.importlib.metadata.PackageNotFoundError) as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.check:
        print(json.dumps(dict(status="INPUTS_VERIFIED", experiment="B322", arm=pilot["arm"],
                              shadow=pilot["shadow"], caps_status=pilot["caps_status"],
                              runtime=versions), indent=2))
        return 0
    if pilot["caps_status"] != "APPROVED":
        parser.error("B322 caps are provisional; profile and approval come first")
    if platform.system() != "Linux":
        parser.error("Training requires Linux GPU")
    if args.output is None or args.output.exists():
        parser.error("Choose a new --output directory")
    from nested_replay_cad import provenance

    check_bundled_design(provenance)
    train.local_paths.process_environment()
    config.update(experiment="B322", shadow=pilot["shadow"],
                  base_recipe_sha256=pilot["base_recipe_sha256"])
    config["caps"] = {**config["caps"], **pilot["caps"]}
    output = args.output.resolve()
    run = types.SimpleNamespace(arm=pilot["arm"])
    return train.execute(run, config, full, versions, args.config.resolve(), output, output)


if __name__ == "__main__":
    raise SystemExit(main())
