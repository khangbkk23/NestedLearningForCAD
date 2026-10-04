# scripts/benchmarks/replaycad/pipeline.py
"""ReplayCAD orchestration and deterministic benchmark-owned metadata."""
from __future__ import annotations
import json, os, re, shlex, shutil, subprocess, time
from pathlib import Path
from scripts.benchmarks.common.cli import TASK_ORDER, tasks_arg
from scripts.benchmarks.common.config import load_yaml, resolve_root
from scripts.benchmarks.common.artifacts import atomic_json, stage_marker
from scripts.benchmarks.common.environment import probe_python
from scripts.benchmarks.common.subprocess_runner import run_command

def cfg_paths(method, project_root):
    root_raw = method.get("replaycad_root", "")
    cache_raw = method.get("cache_root", "")
    root = resolve_root(root_raw, project_root) if root_raw and not root_raw.startswith("${") else Path(os.environ.get("REPLAYCAD_ROOT", "/__missing_replaycad_root__")).expanduser()
    cache = resolve_root(cache_raw, project_root) if cache_raw and not cache_raw.startswith("${") else Path(os.environ.get("REPLAYCAD_CACHE_ROOT", str(project_root / "results" / "replaycad_cache"))).expanduser()
    dataset = method.get("dataset_root", "")
    dataset = resolve_root(dataset, project_root) if dataset and not dataset.startswith("${") else Path(os.environ.get("MVTEC_ROOT", "/__missing_mvtec_root__")).expanduser()
    def exe(k):
        raw = method.get(k, "")
        envname = {"detector_python":"REPLAYCAD_DETECTOR_PYTHON", "ldm_python":"REPLAYCAD_LDM_PYTHON"}[k]
        return Path(os.environ.get(envname, "/__missing_replaycad_python__")).expanduser() if raw.startswith("${") else Path(raw).expanduser()
    return root, cache, dataset, exe("detector_python"), exe("ldm_python")

def selected_from_run(run_dir, cli_tasks=None):
    meta = json.loads((Path(run_dir) / "run.json").read_text())
    return tasks_arg(cli_tasks) if cli_tasks else list(meta.get("task_order", TASK_ORDER))

def expected_checkpoint(class_name, method, smoke=False, max_steps=None):
    overrides = method.get("condition_learning", {}).get("checkpoint_steps", {})
    if class_name in overrides: return int(overrides[class_name])
    if smoke and max_steps: return max(0, int(max_steps) - 1)
    defaults = {"bottle":2499,"cable":5999,"capsule":14999,"carpet":9999,"grid":24999,"hazelnut":19999,"leather":3499,"metal_nut":19999,"pill":1499,"screw":19999,"tile":4999,"toothbrush":1499,"transistor":4999,"wood":5999}
    return defaults.get(class_name, 19999)

def status(name, state, reason=None, **extra):
    return {"name": name, "status": state, **({"reason": reason} if reason else {}), **extra}

def preflight(method, protocol, *, project_root, run_dir, selected):
    root, cache, dataset, detector_py, ldm_py = cfg_paths(method, project_root)
    statuses = []
    statuses.append(status("replaycad_source", "PASS" if root.is_dir() else "FAIL", None if root.is_dir() else "REPLAYCAD_ROOT_missing", path=str(root)))
    expected = [root/"run.py", root/"trainer"/"invad_trainer.py", root/"configs"/"invad"/"invad_mvtec.py", root/"textual_inversion-main"/"main.py", root/"textual_inversion-main"/"scripts"/"txt2img_with_mask.py"]
    for p in expected: statuses.append(status(f"author_file:{p.relative_to(root) if root.exists() else p.name}", "PASS" if p.is_file() else "FAIL", None if p.is_file() else "expected_author_file_missing", path=str(p)))
    for name, exe in (("detector_python", detector_py), ("ldm_python", ldm_py)):
        statuses.append(status(name, "PASS" if exe.is_file() else "FAIL", None if exe.is_file() else "python_executable_missing", path=str(exe)))
    if detector_py.is_file(): statuses.append({"name":"detector_imports", **probe_python(detector_py, ("torch", "numpy", "timm", "cv2", "einops", "trainer", "data"), root, env={"PYTHONPATH":str(root)})})
    else: statuses.append(status("detector_imports", "MISSING", "detector_python_missing"))
    if ldm_py.is_file(): statuses.append({"name":"ldm_imports", **probe_python(ldm_py, ("torch", "numpy", "PIL", "omegaconf", "ldm"), root/"textual_inversion-main", env={"PYTHONPATH":str(root/"textual_inversion-main")})})
    else: statuses.append(status("ldm_imports", "MISSING", "ldm_python_missing"))
    try:
        import torch
        statuses.append(status("cuda", "PASS" if torch.cuda.is_available() else "MISSING", "cuda_unavailable" if not torch.cuda.is_available() else None, gpu_count=torch.cuda.device_count()))
    except Exception as exc: statuses.append(status("cuda", "MISSING", str(exc)))
    cats = list(protocol.get("task_order", TASK_ORDER)); root_ok = dataset.is_dir()
    statuses.append(status("canonical_mvtec_root", "PASS" if root_ok else "FAIL", None if root_ok else "dataset_root_missing", path=str(dataset)))
    for cls in cats:
        base = dataset / cls
        good = list((base/"train"/"good").glob("*.png")) if (base/"train"/"good").is_dir() else []
        test = list((base/"test").glob("*/*.png")) if (base/"test").is_dir() else []
        gt = list((base/"ground_truth").glob("*/*.png")) if (base/"ground_truth").is_dir() else []
        statuses += [status(f"dataset:{cls}:train_good", "PASS" if good else "FAIL", "official_train_good_missing" if not good else None, count=len(good)), status(f"dataset:{cls}:test", "PASS" if test else "FAIL", "official_test_missing" if not test else None, count=len(test)), status(f"dataset:{cls}:ground_truth", "PASS" if gt else "FAIL", "ground_truth_missing" if not gt else None, count=len(gt))]
    sam = method.get("paths", {}).get("sam_root", "")
    sam = Path(os.environ.get("REPLAYCAD_SAM_ROOT", sam)).expanduser() if str(sam).startswith("${") else Path(sam).expanduser()
    statuses.append(status("sam_masks", "PASS" if sam.is_dir() and any(sam.rglob("*.png")) else "MISSING", "sam_masks_missing", path=str(sam)))
    ckpt = root / method.get("paths", {}).get("ldm_checkpoint", "textual_inversion-main/models/model.ckpt")
    statuses.append(status("ldm_checkpoint", "PASS" if ckpt.is_file() else "FAIL", "required_ldm_checkpoint_missing", path=str(ckpt)))
    for asset_name, rel in (("stable_diffusion_checkpoint",method.get("paths",{}).get("stable_diffusion_checkpoint","")), ("bert_assets","textual_inversion-main/models/bert/bert-base-uncased"), ("clip_assets","textual_inversion-main/clip-vit-large-patch14")):
        ap=root/rel if rel else root/"__missing_asset__"
        statuses.append(status(asset_name,"PASS" if ap.exists() else "FAIL","required_pretrained_asset_missing",path=str(ap)))
    enc = root / method.get("paths", {}).get("invad_assets", "model/pretrain")
    enc_ok = enc.is_file() or (enc.is_dir() and any(enc.rglob("*.pth")))
    statuses.append(status("invad_pretrained_assets", "PASS" if enc_ok else "FAIL", "required_detector_asset_missing", path=str(enc)))
    cache.mkdir(parents=True, exist_ok=True)
    statuses += [status("cache_root", "PASS", path=str(cache)), status("existing_conditions", "PASS" if any((Path(run_dir)/"author_outputs"/"conditions").rglob("*.pt")) else "MISSING"), status("existing_generated", "PASS" if any((Path(run_dir)/"author_outputs"/"generated").rglob("*.png")) else "MISSING"), status("existing_detector", "PASS" if any((Path(run_dir)/"author_outputs"/"invad").rglob("*.pth")) else "MISSING")]
    return statuses, {"root": str(root), "cache": str(cache), "dataset": str(dataset), "detector_python": str(detector_py), "ldm_python": str(ldm_py)}

def command_plan(method, selected, run_dir, *, smoke=False, condition_max_steps=None, generation_max_samples=None, detector_max_epochs=None):
    root, cache, dataset, detector_py, ldm_py = cfg_paths(method, Path.cwd())
    workspace=Path(run_dir)/"replay"/"workspace"
    commands=[]; conditions=Path(run_dir)/"author_outputs"/"conditions"; generated=Path(run_dir)/"author_outputs"/"generated"
    for cls in selected:
        steps = condition_max_steps if smoke and condition_max_steps else method.get("condition_learning", {}).get("max_steps", 20000)
        config_name="add_mask.yaml" if cls in {"screw","metal_nut","hazelnut"} else ("add_mask_3_direction.yaml" if cls=="grid" else "add_mask_no_flip.yaml")
        commands.append({"stage":"conditions", "class":cls, "argv":[str(ldm_py), "textual_inversion-main/main.py", "--base", f"textual_inversion-main/configs/latent-diffusion/{config_name}", "-t", "--actual_resume", "textual_inversion-main/models/model.ckpt", "-n", "LD_mvtec_addmask", "--gpus", "0", "--data_root", f"data/mvtec_generate/{cls}", "--init_word", "screw", "--max_steps", str(steps), "--logdir", str(conditions/cls/"logs")], "cwd":str(workspace), "output":str(conditions/cls)})
        gen=method.get("replay_generation",{}); niter=gen.get("per_class_n_iter",{}).get(cls,gen.get("n_iter",25))
        commands.append({"stage":"generate", "class":cls, "argv":[str(ldm_py), "textual_inversion-main/scripts/txt2img_with_mask.py", "--n_samples", str(generation_max_samples or gen.get("n_samples",8)), "--n_iter", str(niter)], "cwd":str(workspace), "output":str(generated/cls)})
    commands.append({"stage":"detector", "argv":[str(detector_py), "run.py", "-c", "configs/invad/invad_mvtec.py", "-m", "train", "data.meta", "replay_meta.json"], "cwd":str(workspace)})
    return commands

def prepare_workspace(run_dir, root, dataset, sam_root, selected):
    """Create only cache-owned symlinks; canonical MVTec and author trees stay read-only."""
    ws=Path(run_dir)/"replay"/"workspace"; ws.mkdir(parents=True,exist_ok=True)
    for name, target in (("textual_inversion-main",Path(root)/"textual_inversion-main"),("run.py",Path(root)/"run.py"),("configs",Path(root)/"configs"),("model",Path(root)/"model"),("trainer",Path(root)/"trainer"),("data_src",Path(root)/"data")):
        link=ws/name
        if not link.exists() and target.exists(): link.symlink_to(target, target_is_directory=target.is_dir())
    if sam_root:
        link=ws/"SAM"
        if not link.exists() and Path(sam_root).exists(): link.symlink_to(Path(sam_root), target_is_directory=True)
    data=ws/"data"/"mvtec_generate"; data.mkdir(parents=True,exist_ok=True)
    for cls in selected:
        target=data/cls
        source=Path(dataset)/cls/"train"/"good"
        if not target.exists() and source.is_dir(): target.symlink_to(source, target_is_directory=True)
    return ws

def metadata(run_dir, dataset_root, selected, generated_root):
    dataset_root=Path(dataset_root); generated_root=Path(generated_root)
    if generated_root.exists() and generated_root.resolve().is_relative_to(dataset_root.resolve()):
        raise ValueError("generated ReplayCAD data must be outside the canonical MVTec root")
    view=Path(run_dir)/"replay"/"dataset_view"; mv=view/"mvtec"; mv.mkdir(parents=True,exist_ok=True)
    train_by={}; test_by={}; rows=[]
    for cls in selected:
        class_link=mv/cls
        if not class_link.exists() and (dataset_root/cls).is_dir(): class_link.symlink_to((dataset_root/cls).resolve(), target_is_directory=True)
        train=[]; gen=generated_root/cls
        for p in sorted(gen.glob("*.png")):
            target=mv/"generate"/cls/"samples"; target.mkdir(parents=True,exist_ok=True); link=target/p.name
            if not link.exists(): link.symlink_to(p.resolve())
            train.append({"img_path":link.relative_to(mv).as_posix(),"mask_path":"","cls_name":cls,"specie_name":"generated","anomaly":0})
        if not train:
            for p in sorted((dataset_root/cls/"train"/"good").glob("*.png")):
                train.append({"img_path":p.relative_to(dataset_root).as_posix(),"mask_path":"","cls_name":cls,"specie_name":"good","anomaly":0})
        tests=[]
        testroot=dataset_root/cls/"test"
        for defect in sorted(testroot.iterdir()) if testroot.is_dir() else ():
            for p in sorted(defect.glob("*.png")):
                label=int(defect.name != "good"); mask=f"{cls}/ground_truth/{defect.name}/{p.stem}_mask.png" if label else ""
                tests.append({"img_path":p.relative_to(dataset_root).as_posix(),"mask_path":mask,"cls_name":cls,"specie_name":defect.name,"anomaly":label})
        train_by[cls]=train; test_by[cls]=tests; rows.extend(train); rows.extend(tests)
    out=Path(run_dir)/"replay"/"replay_metadata.json"; atomic_json(out,{"train":train_by,"test":test_by}); return out, rows
