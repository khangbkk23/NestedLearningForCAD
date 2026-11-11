"""Run B319 from local MVTec data and frozen DINOv2 weights."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nested_replay_cad import paths as local_paths, provenance


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def load_inputs(args):
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if (config["method"] != "B319" or config["seed"] != 0
            or config["arms"] != ["er_uniform", "a0_tail", "a0_tail_ctx"] or config["hard_mining_final"] != .9
            or config["hard_mining_ramp_arrivals"] != 26):
        raise ValueError("Expected the B319 seed0 recipe")
    manifest_path = ROOT / config["full_manifest_path"]
    if digest(manifest_path) != config["full_manifest_sha256"]:
        raise ValueError("Training manifest SHA mismatch")
    full = json.loads(manifest_path.read_text())
    from nested_replay_cad.data import selection
    chosen = selection(full)
    dataset = args.data.resolve()
    weights = (args.weights or local_paths.weights()).resolve()
    missing = [str(path) for path in (dataset, weights) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))
    if digest(weights) != config["encoder"]["sha256"]:
        raise ValueError("DINOv2 checkpoint SHA mismatch")
    expected = {"torch":"2.12.0+cu126", "torchvision":"0.27.0+cu126",
                "numpy":"2.4.5", "timm":"1.0.29", "Pillow":"12.2.0", "safetensors":"0.7.0"}
    versions = {name:importlib.metadata.version(name) for name in expected}
    if platform.python_version() != "3.14.4" or versions != expected:
        raise RuntimeError("Install the root locked Pixi environment")
    from nested_replay_cad.core.data.manifest import build_manifest
    actual = build_manifest(dataset)
    if isinstance(actual, str):
        actual = json.loads(actual)
    if actual != full:
        raise ValueError("MVTec train/good tree differs from the frozen 3629-image manifest")
    config.update(project_root=str(ROOT), dataset_path=str(dataset),
                  selection_path="", cohort_path="", categories=chosen["categories"],
                  snapshots=chosen["boundaries"], integration_arrivals=3629)
    config["encoder"] = {**config["encoder"], "path":str(weights)}
    return config, full, versions, config_path


def main():
    parser = argparse.ArgumentParser(description="B319: 3629 normal arrivals, 15 tasks, full1725 test")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/b319.json")
    parser.add_argument("--data", type=Path, default=local_paths.dataset())
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, help="Explicit result directory; default is .local/runs/b319/<run-id>/outputs")
    parser.add_argument("--run-id", help="Stable run ID when using the default output layout")
    parser.add_argument("--arm", choices=("all", "er_uniform", "a0_tail", "a0_tail_ctx"), default="all")
    parser.add_argument("--check", action="store_true", help="Verify inputs/runtime without training or CUDA initialization")
    args = parser.parse_args()
    if not args.check and platform.system() != "Linux":
        parser.error("Training requires Linux GPU; use --check for input validation")
    try:
        config, full, versions, config_path = load_inputs(args)
    except (OSError, ValueError, RuntimeError, importlib.metadata.PackageNotFoundError) as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.check:
        print(json.dumps(dict(status="INPUTS_VERIFIED", arm=args.arm, seed=0,
                              arrivals=3629, test_images_expected=1725, runtime=versions), indent=2))
        return 0
    run_folder = local_paths.run_root(args.run_id) if args.output is None else None
    output = args.output.resolve() if args.output is not None else run_folder / "outputs"
    if output.exists():
        parser.error("Output already exists; choose a new run directory")
    local_paths.process_environment()
    record_folder=run_folder or (output.parent if output.name=='outputs' else output)
    record=None if os.environ.get('NRCAD_PARENT_RUN_ID') else provenance.RunRecord(
        ROOT,record_folder,output,config,config_path,args.run_id or record_folder.name,args.arm)
    try:
        code=execute(args,config,full,versions,config_path,output,record_folder)
    except BaseException as error:
        if record:record.finish('cancelled' if isinstance(error,KeyboardInterrupt) else 'failed',repr(error))
        raise
    if record:record.finish('complete_review_pending' if code==0 else 'failed',None if code==0 else 'worker exit '+str(code))
    return code


def execute(args,config,full,versions,config_path,output,record_folder):
    output.mkdir(parents=True,exist_ok=True)
    if args.arm == "all":
        (output/'recipe.json').write_text(json.dumps(config,indent=2)+'\n')
        for arm in config['arms']:
            print('B319_ARM_START',arm,flush=True)
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'--config',str(config_path),
                     '--data',config['dataset_path'],'--weights',config['encoder']['path'],
                     '--output',str(output/arm),'--arm',arm]
            log_root=record_folder/'logs';log_root.mkdir(parents=True,exist_ok=True)
            env=os.environ.copy();env['NRCAD_PARENT_RUN_ID']=record_folder.name
            with (log_root/(arm+'.log')).open('x',encoding='utf8') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in process.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                code=process.wait()
            if code:
                print('B319_ARM_FAILED',arm,code,flush=True)
                return code
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/summarize.py'),'--run',str(output)],check=True)
        return 0
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from nested_replay_cad.data import prepare
    prepare(config["dataset_path"], full, output)
    config.update(selection_path=str(output/"full15_selection.json"),cohort_path=str(output/"full15_cohort.json"),arms=[args.arm],arm=args.arm)
    source_paths=[*sorted((ROOT/"src").rglob("*.py")),*sorted((ROOT/"scripts").rglob("*.py")),
                  *sorted((ROOT/"configs").glob("*.json")),ROOT/"pixi.toml",ROOT/"pixi.lock"]
    config["source_sha256"]={p.relative_to(ROOT).as_posix():digest(p) for p in source_paths}
    config["recipe_sha256"]=digest(config_path);config["runtime"]=versions
    (output/"config.json").write_text(json.dumps(config,indent=2)+'\n')
    (output/"arms").mkdir()
    from nested_replay_cad.trainer import arm_trial
    try:
        passed=arm_trial(output/"arms"/args.arm,config,args.arm,time.monotonic()+config["caps"]["per_arm_seconds"])
    finally:
        provenance.write_json(output/'runtime.json',provenance.runtime_info())
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
