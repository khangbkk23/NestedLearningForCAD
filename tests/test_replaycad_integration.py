# tests/test_replaycad_integration.py
from __future__ import annotations
import json
from pathlib import Path
import pytest

from scripts.benchmarks.common.cli import TASK_ORDER, tasks_arg
from scripts.benchmarks.common.metrics import forgetting_matrix
from scripts.benchmarks.common.result_schema import unavailable, load_normalized
from scripts.benchmarks.replaycad.pipeline import metadata, expected_checkpoint, command_plan, preflight
from training.replaycad_normalizer import parse_metric_lines, normalize_native

def test_task_order_is_canonical():
    assert tasks_arg("capsule,bottle,cable") == ["bottle", "cable", "capsule"]
    assert tasks_arg(None) == TASK_ORDER

def test_unavailable_is_not_zero():
    x=unavailable("p_auroc","not_reported")
    assert x["value"] is None and not x["availability"]

def test_forgetting_future_cells_and_reference():
    result=forgetting_matrix([[.9,None],[.8,.7]])
    assert result["fm"] == pytest.approx(.1)

def test_metric_fixture_percent_conversion_and_final_epoch(tmp_path):
    p=tmp_path/"metric.txt"; p.write_text("0 90 40 70 20 80 30\n1 95 50 75 25 85 35\n")
    native=parse_metric_lines(p,["bottle","cable","capsule"])
    assert native["final_epoch"] == 1
    assert native["per_task"]["bottle"]["i_auroc"] == .95
    assert normalize_native(native)["metrics"]["p_aupr"]["value"] == (.50+.25+.35)/3

def test_metadata_is_deterministic_and_nested(tmp_path):
    root=tmp_path/"mvtec"; cls=root/"bottle"; (cls/"train"/"good").mkdir(parents=True); (cls/"test"/"good").mkdir(parents=True); (cls/"test"/"crack").mkdir(parents=True); (cls/"ground_truth"/"crack").mkdir(parents=True)
    for p in [cls/"train"/"good"/"2.png", cls/"train"/"good"/"1.png", cls/"test"/"good"/"1.png", cls/"test"/"crack"/"1.png", cls/"ground_truth"/"crack"/"1_mask.png"]: p.write_bytes(b"x")
    generated=tmp_path/"generated"/"bottle"; generated.mkdir(parents=True); (generated/"z.png").write_bytes(b"z")
    out, rows=metadata(tmp_path/"run",root,["bottle"],tmp_path/"generated")
    data=json.loads(out.read_text()); assert set(data)=={"train","test"}; assert data["train"]["bottle"][0]["img_path"].startswith("generate/")
    assert not (root/"bottle"/"train"/"good"/"replay_meta.json").exists()

def test_expected_condition_checkpoint_is_explicit():
    assert expected_checkpoint("bottle", {"condition_learning":{"checkpoint_steps":{}}}) == 2499
    assert expected_checkpoint("bottle", {}, smoke=True, max_steps=4) == 3

def test_old_cadic_result_loader(tmp_path):
    run=tmp_path/"old"; run.mkdir(); (run/"summary.json").write_text(json.dumps({"macro_final_i_auroc":.8,"macro_final_p_aupr":.3}))
    data=load_normalized(run)
    assert data["legacy"] is True and data["metrics"]["i_auroc"]["value"] is None

def test_author_commands_use_external_cwd_and_cache_outputs(tmp_path):
    method={"replay_generation":{"n_samples":8,"n_iter":25,"per_class_n_iter":{"bottle":25}},"condition_learning":{"max_steps":20000}}
    cmds=command_plan(method,["bottle"],tmp_path,smoke=True,condition_max_steps=3,generation_max_samples=1)
    condition=next(x for x in cmds if x["stage"]=="conditions")
    assert condition["cwd"].endswith("replay/workspace")
    assert "--max_steps" in condition["argv"] and "3" in condition["argv"]
    assert str(tmp_path/"author_outputs") in condition["output"]

def test_preflight_reports_missing_upstream_independently(tmp_path):
    statuses, paths=preflight({"replaycad_root":"/definitely/missing","cache_root":str(tmp_path/"cache"),"dataset_root":"/definitely/missing","detector_python":"/definitely/missing","ldm_python":"/definitely/missing"},{"task_order":["bottle"]},project_root=tmp_path,run_dir=tmp_path/"run",selected=["bottle"])
    by_name={x["name"]:x for x in statuses}
    assert by_name["replaycad_source"]["status"]=="FAIL"
    assert by_name["detector_python"]["status"]=="FAIL"
    assert by_name["dataset:bottle:train_good"]["status"]=="FAIL"
