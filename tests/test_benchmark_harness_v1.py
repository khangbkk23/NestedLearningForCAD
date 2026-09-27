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
from models.cadic_patch_coreset_v1 import CADICPatchCoresetConfig, CADICPatchCoresetV1

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
    assert forgetting_matrix([[.9,.5],[.7,.8]])["fm"]==pytest.approx(.2)

def test_cadic_closest_pair_blockwise_matches_direct_and_nonzero_row():
    features = torch.tensor([[0., 0.], [100., 100.], [200., 200.],
                             [10., 10.], [11., 11.], [300., 300.]])
    coreset = CADICPatchCoresetV1(CADICPatchCoresetConfig(
        budget=6, dim=2, chunk_size=2, pair_chunk_size=2, query_chunk_size=2))
    coreset.update(features)
    value, row = coreset._closest_pair()
    direct = torch.cdist(features, features, p=2,
                         compute_mode="donot_use_mm_for_euclid_dist")
    direct.fill_diagonal_(float("inf"))
    expected_value, flat = direct.reshape(-1).min(dim=0)
    expected_row = int(flat.item() // features.shape[0])
    assert row == expected_row == 3
    assert value.item() == pytest.approx(expected_value.item())
    assert coreset.distance_profile()["pair_max_shape"] == [2, 2]

def test_cadic_nearest_two_dimensional_chunking_matches_direct_and_ties():
    bank = torch.tensor([[0., 0.], [2., 0.], [0., 2.], [10., 0.]])
    query = torch.tensor([[1., 0.], [8., 0.], [0., 1.], [9., 0.], [1., 0.]])
    coreset = CADICPatchCoresetV1(CADICPatchCoresetConfig(
        budget=4, dim=2, chunk_size=2, query_chunk_size=2, pair_chunk_size=2))
    coreset.features = bank.clone()
    values, indices = coreset._nearest(query, bank)
    reference = torch.cdist(query, bank, p=2,
                            compute_mode="donot_use_mm_for_euclid_dist")
    ref_values, ref_indices = reference.min(dim=1)
    assert torch.equal(indices, ref_indices)
    assert torch.allclose(values, ref_values.double())
    assert indices[0].item() == 0  # equal-distance tie uses the lowest index
    profile = coreset.distance_profile()
    assert profile["nearest_max_shape"] == [2, 2]
    assert profile["nearest_max_elements"] <= 2 * 2

def test_cadic_image_score_large_distance_is_finite_and_support_ties_stable():
    coreset = CADICPatchCoresetV1(CADICPatchCoresetConfig(
        budget=3, dim=2, chunk_size=2, query_chunk_size=1, pair_chunk_size=2,
        image_neighbors=2))
    coreset.features = torch.tensor([[0., 0.], [0., 0.], [1., 0.]])
    pixels, indices, images = coreset.pixel_scores(torch.tensor([[[1e20, 0.]]]))
    assert torch.isfinite(pixels).all() and torch.isfinite(images).all()
    assert indices.shape == (1, 1)
    assert coreset._topk_indices(coreset.features[0], 2).tolist() == [0, 1]

def test_cadic_load_state_is_clone_independent():
    source = CADICPatchCoresetV1(CADICPatchCoresetConfig(budget=2, dim=2))
    source.update(torch.tensor([[1., 2.], [3., 4.]]))
    state = source.state_dict()
    target = CADICPatchCoresetV1(CADICPatchCoresetConfig(budget=2, dim=2))
    target.load_state_dict(state)
    target.features[0, 0] = 999.
    assert state["features"][0, 0].item() == 1.

def test_cadic_forgetting_uses_historical_maximum_and_separate_metrics():
    image = [[.6, .5, .4], [.8, .7, .3], [.7, .2, .9]]
    pixel = [[.4, .3, .2], [.5, .6, .1], [.2, .1, .8]]
    assert forgetting_matrix(image)["fm"] == pytest.approx(.3)
    assert forgetting_matrix(pixel)["fm"] == pytest.approx(.4)

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
