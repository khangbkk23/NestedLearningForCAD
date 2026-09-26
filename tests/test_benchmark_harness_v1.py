from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import pytest

from dataset.benchmark_manifest_v1 import build_training_manifest, manifest_digest
from dataset.benchmark_protocol_v1 import MVTecContinualProtocol
from models.fake_benchmark_adapter_v1 import FakeBenchmarkAdapter
from training.benchmark_artifacts_v1 import BenchmarkArtifacts
from training.benchmark_engine_v1 import BenchmarkEngineV1
from training.benchmark_metrics_v1 import forgetting_matrix, pixel_aupr

TASKS=["bottle"]
def make_mvtec(root):
    for name in ["bottle"]:
        train=root/name/"train"/"good"; test=root/name/"test"/"good"; defect=root/name/"test"/"scratch"; gt=root/name/"ground_truth"/"scratch"
        for p in [train,test,defect,gt]: p.mkdir(parents=True,exist_ok=True)
        for i in range(2): Image.fromarray(np.full((12,12,3),i*40+30,np.uint8)).save(train/f"{i:03}.png")
        Image.fromarray(np.full((12,12,3),30,np.uint8)).save(test/"000.png")
        Image.fromarray(np.full((12,12,3),240,np.uint8)).save(defect/"000.png")
        Image.fromarray(np.pad(np.ones((4,4),np.uint8),4)*255).save(gt/"000_mask.png")

def configs(root):
    protocol={"id":"test_protocol","version":1,"dataset":{"name":"MVTec AD","root":str(root)},"task_order":TASKS,
      "training":{"normal_only":True,"dev_mode":"disabled","drop_last":False},"evaluation":{"final_per_task":True,"primary":["i_auroc","p_aupr"],"aggregation":"macro_tasks","pixels":"all","pooled":"secondary_only"},"forgetting":{"enabled":True,"formula":"mean_prior_max_minus_final"},"leakage":{"official_test_feedback":"forbidden"}}
    method={"id":"fake","version":1,"adapter":"fake","exact_parity_claim":False,"preprocessing":{"image_size":12,"resize":"direct_square_bilinear_pil","mean":[.5]*3,"std":[.5]*3},"runtime":{"batch_size":2,"dtype":"float32"}}
    return protocol,method

def test_manifest_is_portable_and_train_only(tmp_path):
    make_mvtec(tmp_path); p,m=configs(tmp_path); a=build_training_manifest(p,3,"abc")
    assert all("/train/good/" in "/"+e["relative_path"] for e in a["entries"])
    assert a["digest"]==manifest_digest(a)
    p2,m2=configs(tmp_path/"other"); (tmp_path/"other").mkdir(); make_mvtec(tmp_path/"other")
    assert a["digest"]==build_training_manifest(p2,3,"abc")["digest"]

def test_metrics_and_forgetting():
    assert pixel_aupr(np.array([[[0.,1.],[0.,1.]]]),np.array([[[0,1],[0,1]]]))==1.0
    assert forgetting_matrix([[.9,.5],[.7,.8]])["fm"]==pytest.approx(.1)

def test_fake_engine_no_training_test_access(tmp_path):
    make_mvtec(tmp_path); p,m=configs(tmp_path); manifest=build_training_manifest(p,0,"abc")
    proto=MVTecContinualProtocol(p,manifest,m,0); art=BenchmarkArtifacts(tmp_path/"run"); adapter=FakeBenchmarkAdapter()
    engine=BenchmarkEngineV1(p,adapter,art,fail_on_state_mutation=True)
    rows=engine.train(proto); assert rows[0]["train_samples"]==2 and (art.states/"final.pt").is_file()
    adapter.load_state_dict(torch.load(art.states/"final.pt",weights_only=False)); per,macro=engine.evaluate_final(proto)
    assert set(per)=={"bottle"} and macro["task_count"]==1
    fm=engine.evaluate_forgetting(proto); assert fm["matrix"]

def test_setup_and_run_scripts_end_to_end(tmp_path):
    make_mvtec(tmp_path); p=tmp_path/"p.yaml"; m=tmp_path/"m.yaml"
    protocol,method=configs(tmp_path)
    import yaml; p.write_text(yaml.safe_dump(protocol)); m.write_text(yaml.safe_dump(method))
    env=dict(__import__("os").environ); env["MVTEC_ROOT"]=str(tmp_path)
    # The setup command is tested using a temporary config and the smoke gate.
    out=tmp_path/"out"; cmd=[sys.executable,"scripts/benchmarks/0_setup_benchmark.py","--protocol",str(p),"--method",str(m),"--seed","0","--output-root",str(out),"--smoke"]
    result=subprocess.run(cmd,capture_output=True,text=True,env=env); assert result.returncode==0,result.stderr
    run=Path(result.stdout.strip()); result=subprocess.run([sys.executable,"scripts/benchmarks/1_run_benchmark.py","--run-dir",str(run),"--phase","all","--max-tasks","1"],capture_output=True,text=True,env=env)
    assert result.returncode==0,result.stderr; assert (run/"summary.json").is_file()
