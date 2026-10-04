# scripts/benchmarks/replaycad/run.py
"""Subprocess runner for the upstream ReplayCAD stages."""
from __future__ import annotations
import argparse, json, os, shutil, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from scripts.benchmarks.common.artifacts import ensure_run_dir, atomic_json, read_json, stage_marker, command_file
from scripts.benchmarks.common.cli import tasks_arg
from scripts.benchmarks.common.config import load_yaml
from scripts.benchmarks.common.environment import write_environment, git_info
from scripts.benchmarks.common.resources import resource_snapshot, directory_bytes
from scripts.benchmarks.common.subprocess_runner import run_command, CommandError
from scripts.benchmarks.replaycad.pipeline import preflight, command_plan, cfg_paths, expected_checkpoint, metadata, prepare_workspace
from training.replaycad_normalizer import normalize_run

def args(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--run-dir",required=True); p.add_argument("--stage",choices=["preflight","conditions","generate","metadata","detector","evaluate","normalize","all"],default="all"); p.add_argument("--device",default="cuda"); p.add_argument("--resume",action="store_true"); p.add_argument("--tasks"); p.add_argument("--dry-run",action="store_true"); p.add_argument("--smoke",action="store_true"); p.add_argument("--condition-max-steps",type=int); p.add_argument("--generation-max-samples",type=int); p.add_argument("--detector-max-epochs",type=int); p.add_argument("--execution-mode",choices=["native","continual_boundaries"]); return p.parse_args(argv)

def load_run(run):
    run=Path(run).expanduser().resolve(); meta=read_json(run/"run.json",{}); protocol=load_yaml(run/"protocol_resolved.yaml"); method=load_yaml(run/"method_resolved.yaml"); return run,meta,protocol,method

def detector_child_env(exe):
    """Environment needed by author-side CUDA extensions, without source edits."""
    exe=Path(exe)
    nvidia_root=next(iter((exe.parent.parent/"lib").glob("python*/site-packages/nvidia")), None)
    cuda_include=[str(exe.parent.parent/"include")]
    cuda_lib=[str(exe.parent.parent/"lib")]
    if nvidia_root:
        cuda_include.extend(str(p) for p in sorted(nvidia_root.glob("*/include")))
        cuda_lib.extend(str(p) for p in sorted(nvidia_root.glob("*/lib")))
    return {
        "PATH":str(exe.parent)+os.pathsep+os.environ.get("PATH",""),
        "CUDA_HOME":str(exe.parent.parent),
        "CUDA_PATH":str(exe.parent.parent),
        "CUDACXX":str(exe.parent/"nvcc"),
        "CPATH":os.pathsep.join(cuda_include)+os.pathsep+os.environ.get("CPATH",""),
        "LIBRARY_PATH":os.pathsep.join(cuda_lib)+os.pathsep+os.environ.get("LIBRARY_PATH",""),
        "LD_LIBRARY_PATH":os.pathsep.join(cuda_lib)+os.pathsep+os.environ.get("LD_LIBRARY_PATH","") ,
    }

def ensure_preflight(run, protocol, method, selected):
    statuses,paths=preflight(method,protocol,project_root=ROOT,run_dir=run,selected=selected); atomic_json(run/"preflight.json",{"statuses":statuses,"paths":paths}); return statuses,paths

def _condition_command(cmd, cls, run, paths, smoke, max_steps):
    argv=list(cmd["argv"]); # Preserve author entry point; only resolve path/cwd and cap smoke.
    if smoke and max_steps:
        for i,x in enumerate(argv):
            if x=="--max_steps" and i+1<len(argv): argv[i+1]=str(max_steps)
    return argv

def run_conditions(run, method, selected, *, paths, a, commands):
    outroot=run/"author_outputs"/"conditions"; outroot.mkdir(parents=True,exist_ok=True)
    sam_raw=method.get("paths",{}).get("sam_root",""); sam=os.environ.get("REPLAYCAD_SAM_ROOT","") if str(sam_raw).startswith("${") else sam_raw
    ws=prepare_workspace(run,paths["root"],paths["dataset"],sam,selected)
    for cmd in commands:
        if cmd["stage"]!="conditions" or cmd["class"] not in selected: continue
        cls=cmd["class"]; target=outroot/cls; target.mkdir(exist_ok=True)
        if a.resume and (target/"selection.json").is_file(): continue
        if a.dry_run: continue
        # Upstream main writes relative logs; use a class-owned working directory and
        # an explicit data root. No upstream source is copied or modified.
        argv=_condition_command(cmd,cls,run,paths,a.smoke,a.condition_max_steps)
        command_error = None
        try: run_command(argv,cwd=cmd["cwd"],log_path=target/"condition.log")
        except CommandError as exc:
            # The released MVTec DataModule has no test_dataloader(), so the
            # author process can exit nonzero after it has already emitted the
            # requested condition checkpoint. Preserve the log and accept the
            # checkpoint only when both expected files are present.
            command_error = str(exc)
        expected=expected_checkpoint(cls,method,a.smoke,a.condition_max_steps)
        pairs=list(target.rglob(f"embeddings_gs-{expected}.pt"))
        pairs=[p for p in pairs if p.parent.name=="checkpoints"]
        mask=[p.parent/f"mask_linear-{expected}.pt" for p in pairs]
        if not pairs or not mask or not mask[0].is_file():
            atomic_json(target/"status.json",{"status":"failed","reason":"expected_condition_checkpoint_missing","expected_step":expected}); raise RuntimeError(f"{cls}: expected_condition_checkpoint_missing at step {expected}")
        shutil.copy2(pairs[0],target/pairs[0].name); shutil.copy2(mask[0],target/mask[0].name)
        result={"status":"pass","expected_step":expected,"embedding":str(target/pairs[0].name),"mask":str(target/mask[0].name)}
        if command_error: result.update({"author_exit":"nonzero_after_checkpoint","author_error":command_error})
        atomic_json(target/"selection.json",result)
        atomic_json(target/"status.json",result)

def run_generate(run, method, selected, paths, a, commands):
    cache_root=Path(paths["cache"])/run.name/"generated"; cache_root.mkdir(parents=True,exist_ok=True)
    outroot=run/"author_outputs"/"generated"
    if outroot.is_dir() and not outroot.is_symlink() and not any(outroot.iterdir()): outroot.rmdir()
    if not outroot.exists(): outroot.symlink_to(cache_root.resolve(),target_is_directory=True)
    outroot=cache_root
    ws=Path(run)/"replay"/"workspace"
    for cmd in commands:
        if cmd["stage"]!="generate" or cmd["class"] not in selected: continue
        cls=cmd["class"]; out=outroot/cls; out.mkdir(exist_ok=True)
        if a.resume and (out/"manifest.json").is_file(): continue
        if a.dry_run: continue
        sel=run/"author_outputs"/"conditions"/cls/"selection.json"
        if not sel.is_file(): raise RuntimeError(f"{cls}: condition selection missing")
        selection=json.loads(sel.read_text()); step=int(selection["expected_step"])
        gen=method.get("replay_generation",{}); niter=1 if a.smoke else gen.get("per_class_n_iter",{}).get(cls,gen.get("n_iter",25))
        argv=[str(paths["ldm_python"]), "textual_inversion-main/scripts/txt2img_with_mask.py", "--ddim_eta", "0.0", "--n_samples", str(a.generation_max_samples or gen.get("n_samples",8)), "--n_iter", str(niter), "--embedding_path", str(run/"author_outputs"/"conditions"/cls/selection["embedding"]), "--class_layer_path", str(run/"author_outputs"/"conditions"/cls/selection["mask"]), "--prompt", "a photo of *", "--outdir", str(out)]
        argv += ["--conference_mask_path", f"SAM/data/mvtec_conference/{cls}"]
        run_command(argv,cwd=cmd["cwd"],log_path=run/"logs"/f"generate_{cls}.log")
        files=sorted(
            p for p in out.iterdir() if p.is_file()
            and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            and not p.stem.endswith("_mask")
        )
        gen=method.get("replay_generation",{}); configured=int(gen.get("n_samples",8)); niter=int(gen.get("per_class_n_iter",{}).get(cls,gen.get("n_iter",25))); bytes_=sum(p.stat().st_size for p in out.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        atomic_json(out/"manifest.json",{"class":cls,"configured_n_samples":configured,"configured_n_iter":niter,"actual_generated_images":len(files),"generated_bytes":bytes_,"deterministic":False,"seed":method.get("seed")})

def run_detector(run, method, paths, a):
    if a.dry_run: return
    root=Path(paths["root"]); exe=Path(paths["detector_python"]); out=run/"author_outputs"/"invad"; out.mkdir(parents=True,exist_ok=True)
    if a.resume and (out/"ckpt.pth").is_file(): return
    sam_raw=method.get("paths",{}).get("sam_root",""); sam=os.environ.get("REPLAYCAD_SAM_ROOT","") if str(sam_raw).startswith("${") else sam_raw
    prepare_workspace(run,root,paths["dataset"],sam,[])
    ws=run/"replay"/"workspace"; view=run/"replay"/"dataset_view"/"mvtec"; data=ws/"data"; data.mkdir(parents=True,exist_ok=True)
    link=data/"mvtec"
    if not link.exists(): link.symlink_to(view.resolve(),target_is_directory=True)
    shutil.copy2(run/"replay"/"replay_metadata.json",view/"replay_meta.json")
    cfg=root/method.get("detector",{}).get("config","configs/invad/invad_mvtec.py")
    argv=[str(exe),str(ws/"run.py"),"-c","configs/invad/invad_mvtec.py","-m","train","data.meta=replay_meta.json"]
    if a.smoke and a.detector_max_epochs:
        argv += [f"trainer.epoch_full={a.detector_max_epochs}",f"epoch_full={a.detector_max_epochs}"]
        # A one-sample smoke replay would otherwise be dropped by the
        # author's normal ``drop_last=True`` batch of 32, yielding zero
        # optimizer steps and no checkpoint.
        argv += ["trainer.data.batch_size_per_gpu=1", "trainer.data.batch_size=1"]
    if a.device=="cpu": raise RuntimeError("ReplayCAD InvAD author code requires CUDA")
    # Author code builds StyleGAN2 extensions at import time.  Its subprocess
    # must see tools from the selected detector environment (not only the
    # host wrapper environment), in particular that environment's ``ninja``.
    # Point the extension builder at the same CUDA toolkit that supplies the
    # detector environment's nvcc and headers.  This is a subprocess
    # environment fix; author source and CUDA kernels remain untouched.
    child_env=detector_child_env(exe)
    run_command(argv,cwd=ws,env=child_env,log_path=out/"detector.log")
    checkpoints=sorted(ws.rglob("ckpt.pth"), key=lambda p:p.stat().st_mtime)
    if checkpoints:
        candidate=checkpoints[-1].parent/"metric.txt"
        if candidate.is_file(): shutil.copy2(candidate,out/"metric.txt")
    for candidate in ws.rglob("*.pth"):
        target=out/candidate.name
        if not target.exists(): shutil.copy2(candidate,target)

def run_evaluate(run, method, paths, a):
    if a.dry_run: return
    ws=Path(run)/"replay"/"workspace"; out=Path(run)/"author_outputs"/"invad"; checkpoints=sorted(ws.rglob("ckpt.pth"), key=lambda p:p.stat().st_mtime)
    # Training writes a (possibly empty/ghost) metric.txt too.  It is not an
    # evaluation completion marker, so resume must use an explicit marker.
    if a.resume and (out/"evaluation_complete.json").is_file(): return
    if not checkpoints: raise RuntimeError("final configured ReplayCAD checkpoint ckpt.pth is missing")
    final=checkpoints[-1]; base=ws/"runs"; rel=final.parent.relative_to(base) if final.is_relative_to(base) else final.parent.name
    argv=[str(paths["detector_python"]),str(ws/"run.py"),"-c","configs/invad/invad_mvtec.py","-m","test",f"trainer.resume_dir={rel}","data.meta=replay_meta.json"]
    run_command(argv,cwd=ws,env=detector_child_env(paths["detector_python"]),log_path=out/"evaluate.log")
    candidate=final.parent/"metric.txt"
    if candidate.is_file(): shutil.copy2(candidate,out/"metric.txt")
    atomic_json(out/"evaluation_complete.json",{"checkpoint":str(final),"resume_dir":str(rel),"metric_path":str(out/"metric.txt")})

def main(argv=None):
    a=args(argv); run,meta,protocol,method=load_run(a.run_dir); ensure_run_dir(run,overwrite=True)
    if a.smoke:
        # Keep the explicit smoke mode bounded when callers do not provide
        # caps.  These are compatibility probes and are never reportable.
        a.condition_max_steps = a.condition_max_steps or 501
        a.generation_max_samples = a.generation_max_samples or 1
        a.detector_max_epochs = a.detector_max_epochs or 1
    selected=tasks_arg(a.tasks) if a.tasks else list(meta.get("task_order",protocol.get("task_order",[])))
    if meta.get("reportable") and (a.smoke or any(x is not None for x in (a.condition_max_steps,a.generation_max_samples,a.detector_max_epochs))): raise SystemExit("smoke caps cannot be applied to reportable runs")
    if a.execution_mode: meta["execution_mode"]=a.execution_mode
    if a.smoke: meta.update({"smoke":True,"reportable":False})
    atomic_json(run/"run.json",meta); write_environment(run,ROOT,{"upstream":git_info((preflight(method,protocol,project_root=ROOT,run_dir=run,selected=selected)[1])["root"]),"execution_mode":meta.get("execution_mode","native")})
    statuses,paths=ensure_preflight(run,protocol,method,selected); commands=command_plan(method,selected,run,smoke=a.smoke,condition_max_steps=a.condition_max_steps,generation_max_samples=a.generation_max_samples,detector_max_epochs=a.detector_max_epochs); command_file(run,commands)
    if a.stage=="preflight": print(json.dumps({"statuses":statuses,"commands":commands},indent=2)); return 0
    if meta.get("execution_mode","native")=="continual_boundaries":
        # The released InvAD trainer only emits epoch checkpoints. Treating those
        # as task boundaries would change the scientific procedure, so this mode
        # remains an explicit compatibility gate until a verified resume bridge is
        # available.
        atomic_json(run/"normalized_metrics"/"continual_matrix.json",{"matrix_available":False,"reason":"author_checkpoints_are_epoch_states_not_task_boundary_states","execution_mode":"continual_boundaries"})
        atomic_json(run/"normalized_metrics"/"forgetting.json",{"availability":False,"reason":"author_checkpoints_are_epoch_states_not_task_boundary_states","execution_mode":"continual_boundaries"})
        stage_marker(run,"continual_boundaries","UNSUPPORTED",reason="author_checkpoints_are_epoch_states_not_task_boundary_states")
        raise SystemExit("continual_boundaries is unsupported until optimizer/model continuation is verified; no native metrics were relabelled")
    if a.dry_run: print(json.dumps({"run_dir":str(run),"commands":commands},indent=2)); return 0
    if any(s.get("status")=="FAIL" for s in statuses) and a.stage in ("all","conditions","generate","detector","evaluate"): raise SystemExit("ReplayCAD preflight has blocking failures; inspect preflight.json")
    if a.stage in ("all","conditions"):
        t=time.monotonic(); run_conditions(run,method,selected,paths=paths,a=a,commands=commands); stage_marker(run,"conditions","PASS",wall_seconds=time.monotonic()-t)
    if a.stage in ("all","generate"):
        t=time.monotonic(); run_generate(run,method,selected,paths,a,commands); stage_marker(run,"generate","PASS",wall_seconds=time.monotonic()-t)
    if a.stage in ("all","metadata"):
        t=time.monotonic(); generated=Path(paths["cache"])/run.name/"generated"; out,rows=metadata(run,paths["dataset"],selected,generated); atomic_json(run/"replay"/"manifest.json",{"classes":selected,"rows":len(rows),"generated":str(generated)}); stage_marker(run,"metadata","PASS",rows=len(rows),path=str(out),wall_seconds=time.monotonic()-t)
    if a.stage in ("all","detector"):
        t=time.monotonic(); run_detector(run,method,paths,a); stage_marker(run,"detector","PASS",wall_seconds=time.monotonic()-t)
    if a.stage in ("all","evaluate"):
        t=time.monotonic(); run_evaluate(run,method,paths,a); stage_marker(run,"evaluate","PASS",note="native evaluator output remains in author_outputs",wall_seconds=time.monotonic()-t)
    if a.stage in ("all","normalize"):
        configured_epoch=a.detector_max_epochs if a.smoke and a.detector_max_epochs else method.get("detector",{}).get("final_epoch",200)
        t=time.monotonic(); normalize_run(run,classes=selected,final_epoch=int(configured_epoch),source="reproduced",semantics_match=True); stage_marker(run,"normalize","PASS",wall_seconds=time.monotonic()-t)
        if meta.get("execution_mode","native")=="native": atomic_json(run/"normalized_metrics"/"continual_matrix.json",{"matrix_available":False,"reason":"author_code_does_not_emit_task_boundary_states","matrix":None})
    atomic_json(run/"resource"/"memory.json",{"peak_gpu_vram_bytes":None,"peak_cpu_ram_bytes":None,"note":"not measurable without active author process"}); disk=resource_snapshot(run); disk["generated_replay"]=directory_bytes(Path(paths["cache"])/run.name/"generated"); atomic_json(run/"resource"/"disk.json",disk)
    summary={"method":"replaycad","protocol":protocol.get("id"),"execution_mode":meta.get("execution_mode","native"),"reportable":meta.get("reportable",False),"smoke":meta.get("smoke",False),"normalized_metrics":read_json(run/"normalized_metrics"/"final_macro.json",{}),"forgetting":read_json(run/"normalized_metrics"/"forgetting.json",{})}
    atomic_json(run/"summary.json",summary); (run/"summary.md").write_text("# ReplayCAD benchmark run\n\n"+json.dumps(summary,indent=2)+"\n")
    print(run); return 0
if __name__=="__main__": raise SystemExit(main())
