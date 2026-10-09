# scripts/exps/hope_outer_learning_pilot_v1.py
"""Bounded normal calibration with isolated functional image events."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import datetime
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback

import pandas as pd
import torch
from torch.nn import functional as F

from exps.hope_outer_learning_pilot_v1 import (
    CENTER_IDS, CENTERS, EXPECTED_CACHE_SHA, EXPECTED_CHECKPOINT_SHA,
    atomic_json, atomic_torch, commit, copy_parameters, detached_state, event,
    file_sha, initial_state, inventory, make_schedule, mask_rgb, nontriviality,
    paired_bootstrap, parameters_from_fixture, parameters_sha, predict_with_product,
    propose, query_rows, read, restore_parameters, ridge_fit, ridge_read, parameter_storage_bytes,
    schedule_sha, serialize_parameters, spatial, state_fingerprint, support_sequence,
    teacher_error, tensor_sha, validate_bottle_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/hope_cad/outer_learning_pilot_v1/shared_gpu_attempt"
SOURCE_CACHE = ROOT / "results/hope_cad/memory_learning_gate/features/class_bottle.pt"
SOURCE_FIXTURE = ROOT / "results/hope_cad/memory_learning_gate/p0_initial_fixture.pt"
CHECKPOINT = ROOT / "checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth"
ARMS = ("RAND_FROZEN", "RAND_P1", "META_FROZEN", "META_P1", "STATIC_NORMAL_CONTROL", "STATIC_META")


class HardGate(RuntimeError):
    pass


class Budget:
    def __init__(self, out, device):
        self.device = device
        self.start = time.perf_counter()
        authorization = json.loads((out / "authorization.json").read_text())
        start_utc = datetime.datetime.fromisoformat(authorization["budget_start_utc"])
        self.offset = (datetime.datetime.now(datetime.timezone.utc)-start_utc).total_seconds()
        self.limit = authorization["aggregate_wall_cap_seconds"]
        self.memory_limit = authorization["memory_cap_bytes"]

    def elapsed(self):
        return self.offset + time.perf_counter()-self.start

    def check(self):
        if self.elapsed() > self.limit:
            raise HardGate("aggregate wall-clock budget exceeded")
        if torch.cuda.max_memory_allocated(self.device) > self.memory_limit:
            raise HardGate("allocated CUDA memory budget exceeded")


def sync(device):
    torch.cuda.synchronize(device)
    return time.perf_counter()


def write_table(out, name, rows):
    path = out / name
    frame = pd.DataFrame(rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)
    reopened = pd.read_parquet(path)
    if len(reopened) != len(frame) or list(reopened) != list(frame):
        raise HardGate("table round-trip differs")


def verify_integrity():
    record = json.loads((ROOT / "results/hope_cad/outer_learning_pilot_v1/stage0_verification.json").read_text())
    sources = {row["path"]: row["current_sha256"] for row in record["sources"]}
    for path, sha in sources.items():
        if file_sha(ROOT / path) != sha:
            raise HardGate("verified source identity changed: " + path)
    previous = ROOT / "results/hope_cad/outer_learning_stage0"
    artifacts = json.loads((previous / "artifact_validation.json").read_text())
    for name, sha in artifacts["files"].items():
        if file_sha(previous / name) != sha:
            raise HardGate("verified synthetic artifact identity changed: " + name)
    return {"sources":sources,"synthetic_artifacts_verified":len(artifacts["files"]),
            "prior_tests_verified":133,"prior_real_integrations_deselected":3}


def oracle_parity(fixture, parameters, image):
    from exps.hope_image_synchronous_memory import (
        ImageSynchronousMemory, aggregate_image_statistics,
    )
    oracle = ImageSynchronousMemory(fixture["smt"], "P1", device=parameters.a0.device)
    before = oracle.state_fingerprint()
    snapshot = oracle.snapshot_state()
    quantities = oracle.generate_update_quantities(image, snapshot)
    stats = aggregate_image_statistics(quantities)
    expected = oracle.propose_event(image)
    state = initial_state(parameters)
    proposed = propose(parameters, state, image)
    pairs = [("spatial", proposed.spatial, quantities.spatial),
             ("queries", proposed.queries, quantities.queries),
             ("keys", proposed.keys, quantities.keys), ("values", proposed.values, quantities.values),
             ("gates", proposed.gates, quantities.gates), ("C", proposed.C, stats.C),
             ("D", proposed.D, stats.D), ("transition", proposed.transition, expected.transition),
             ("pre_read", proposed.pre_read, expected.causal_output)]
    pairs += [(name, proposed.weights[name], expected.weights[name]) for name in state.weights]
    details=[]
    for name, actual, reference in pairs:
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        details.append({"name":name,"max_abs":float((actual-reference).abs().max()),
                        "bitwise_equal":bool(torch.equal(actual,reference))})
    if before != oracle.state_fingerprint():
        raise HardGate("oracle proposal mutated state")
    updated = commit(parameters, state, proposed)
    oracle.commit_event(expected)
    after_commit_identity = oracle.state_fingerprint()
    post = read(parameters, updated, image)
    reference = oracle.evaluate_read_only(image)["memory"]
    torch.testing.assert_close(post,reference,rtol=1e-5,atol=1e-6)
    details.append({"name":"post_support_read","max_abs":float((post-reference).abs().max()),
                    "bitwise_equal":bool(torch.equal(post,reference))})
    if updated.counters.completed != 1 or int(oracle.completed_events) != 1:
        raise HardGate("real-input oracle event count differs")
    if after_commit_identity != oracle.state_fingerprint():
        raise HardGate("oracle query evaluation mutated state")
    for name, expected_count in (("memory_update_count", 1), ("auxiliary_update_count", 1), ("online_update_count", 2)):
        if int(getattr(oracle.smt, name)) != expected_count:
            raise HardGate("real-input oracle update counters differ")
    return {"passed":True,"device":str(parameters.a0.device),"comparisons":details,
            "rtol":1e-5,"atol":1e-6,"completed_events":1}


def extract_masked(out, payload, rows, device, budget):
    from dataset.benchmark_protocol_v1 import _ImageDataset
    from models.feature_extractors.cadic_vit_v1 import CADICViTConfig, CADICViTFeatureExtractor
    if file_sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise HardGate("ViT checkpoint SHA differs")
    if file_sha(SOURCE_CACHE) != EXPECTED_CACHE_SHA:
        raise HardGate("clean cache changed before inference")
    config = CADICViTConfig(checkpoint_path=str(CHECKPOINT), checkpoint_identity="sha256:"+EXPECTED_CHECKPOINT_SHA)
    from scripts.hope_cad.probe_real_features import feature_metadata
    if payload["metadata"] != feature_metadata(CHECKPOINT, EXPECTED_CHECKPOINT_SHA):
        raise HardGate("preprocessing/feature metadata differs")
    view_path=out/"masked_features.pt"
    wanted=list(range(60,80))+list(range(92,100))+list(range(150,170))
    provenance={"source_cache_sha256":EXPECTED_CACHE_SHA,"checkpoint_sha256":EXPECTED_CHECKPOINT_SHA,
                "identities":wanted,"raw_sha256":{str(i):rows[i]["raw_sha256"] for i in wanted},
                "centers":[list(c) for c in CENTERS],"view_contract":"ONE_SHARED_16_REGION_VIEW"}
    if view_path.exists():
        cached=torch.load(view_path,map_location="cpu",weights_only=True)
        if cached["provenance"] != provenance:
            raise HardGate("masked cache provenance differs")
        for i in wanted:
            if tensor_sha(cached["features"][i]) != cached["feature_hashes"][str(i)]:
                raise HardGate("masked feature hash differs")
        print("validated existing immutable masked views",flush=True)
        return cached["features"],cached["diagnostics"]
    started=sync(device)
    extractor=CADICViTFeatureExtractor(config,device=device)
    load_seconds=sync(device)-started
    if extractor.training or extractor.model.training or any(p.requires_grad for p in extractor.parameters()):
        raise HardGate("backbone is not frozen/eval")
    ds=_ImageDataset(ROOT/"data/mvtec",[{"relative_path":r["relative_path"]} for r in rows],224,
                     config.mean,config.std,True)
    features, diagnostics={},[]
    extraction_start=sync(device)
    for pos,i in enumerate(wanted):
        budget.check()
        clean_rgb=ds[i]["images"].unsqueeze(0)
        masked_rgb,mask=mask_rgb(clean_rgb)
        if mask.sum()!=9216 or not torch.equal(masked_rgb[:,:,mask],torch.zeros_like(masked_rgb[:,:,mask])):
            raise HardGate("masked rectangle or fill differs")
        if not torch.equal(clean_rgb[:,:,~mask],masked_rgb[:,:,~mask]):
            raise HardGate("unmasked input changed")
        for r,c in CENTERS:
            if not bool(mask[(r-1)*8:(r+2)*8,(c-1)*8:(c+2)*8].all()):
                raise HardGate("center-to-pixel mapping differs")
        extracted=extractor.extract_patch_features(masked_rgb).detach().cpu()[0]
        if extracted.shape!=(784,768) or extracted.dtype!=torch.float32 or extracted.requires_grad or not bool(torch.isfinite(extracted).all()):
            raise HardGate("masked feature geometry differs")
        if pos==0:
            repeat=extractor.extract_patch_features(masked_rgb).detach().cpu()[0]
            if not torch.equal(extracted,repeat):
                raise HardGate("masked inference is not deterministic")
            direct=extractor.extract_patch_features(clean_rgb).detach().cpu()[0]
            from scripts.exps.hope_image_synchronous_memory import feature_equivalence
            equivalent=feature_equivalence(direct,payload["patches"][i])
            if not equivalent["passed"]:
                raise HardGate("clean direct/cache equivalence failed")
            atomic_json(out/"extractor_check.json", {"passed":True,"cache_equivalence":equivalent,
                "repeated_masked_bitwise_equal":True,"metadata":extractor.protocol_metadata(),
                "checkpoint_load_seconds":load_seconds,"clean_identity":i,
                "backbone_parameter_bytes":sum(p.numel()*p.element_size() for p in extractor.parameters())})
        clean=payload["patches"][i]
        unit_clean=F.normalize(clean,dim=-1,eps=1e-8)
        unit_masked=F.normalize(extracted,dim=-1,eps=1e-8)
        features[i]=extracted
        diagnostics.append({"identity":i,"role":rows[i]["role"],"clean_input_sha256":tensor_sha(clean_rgb),
                            "masked_input_sha256":tensor_sha(masked_rgb),"masked_feature_sha256":tensor_sha(extracted),
                            "center_cosine_mean":float((unit_clean[list(CENTER_IDS)]*unit_masked[list(CENTER_IDS)]).sum(-1).mean()),
                            "whole_grid_cosine_mean":float((unit_clean*unit_masked).sum(-1).mean()),
                            "raw_center_gap":float((clean[list(CENTER_IDS)]-extracted[list(CENTER_IDS)]).norm(dim=-1).mean()),
                            "raw_whole_grid_gap":float((clean-extracted).norm(dim=-1).mean())})
        if pos%8==0 or pos==len(wanted)-1:
            print(f"masked feature extraction {pos+1}/{len(wanted)}",flush=True)
    extraction_seconds=sync(device)-extraction_start
    extractor.close()
    del extractor
    gc.collect(); torch.cuda.empty_cache()
    data={"provenance":provenance,"features":features,"diagnostics":diagnostics,
          "feature_hashes":{str(i):tensor_sha(v) for i,v in features.items()},
          "extraction_seconds":extraction_seconds,"checkpoint_load_seconds":load_seconds}
    atomic_torch(view_path,data)
    reopened=torch.load(view_path,map_location="cpu",weights_only=True)
    if reopened["feature_hashes"] != data["feature_hashes"]:
        raise HardGate("masked cache reopen differs")
    atomic_json(out/"feature_timing.json",{"checkpoint_load_seconds":load_seconds,
                                         "masked_extraction_seconds":extraction_seconds,"masked_images":48})
    write_table(out,"masked_view_diagnostics.parquet",diagnostics)
    return features,diagnostics


def save_fit_checkpoint(path,p,optimizer,step,arm,schedule_hash,manifest_hash):
    payload={"schema":"normal_outer_fit_v1","arm":arm,"step":step,
             "parameters":serialize_parameters(p),"parameters_sha256":parameters_sha(p),
             "optimizer":optimizer.state_dict(),"schedule_sha256":schedule_hash,
             "manifest_sha256":manifest_hash,"cpu_rng_state":torch.get_rng_state(),
             "cuda_rng_state":torch.cuda.get_rng_state(p.a0.device) if p.a0.is_cuda else None,
             "configuration":{"lr":1e-4,"betas":[.9,.999],"eps":1e-8,"weight_decay":0,
                              "trainable_names":["A0","Wq"],"inner_h":.02}}
    # Optimizer tensors are detached; no episode graph enters the checkpoint.
    payload["optimizer"]={**payload["optimizer"],"state":{
        key:{name:v.detach().cpu().clone() if torch.is_tensor(v) else v for name,v in state.items()}
        for key,state in payload["optimizer"]["state"].items()}}
    atomic_torch(path,payload)
    reopened=torch.load(path,map_location="cpu",weights_only=True)
    if reopened["parameters_sha256"]!=parameters_sha(restore_parameters(reopened["parameters"])) or reopened["step"]!=step:
        raise HardGate("fit checkpoint identity differs")


def fit_arm(out,arm,initial,clean,masked,targets,schedule,manifest_hash,budget):
    device=initial.a0.device
    checkpoint_dir=out/"checkpoints";checkpoint_dir.mkdir(exist_ok=True)
    p=copy_parameters(initial,trainable=True)
    optimizer=torch.optim.Adam([p.a0,p.wq],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
    trace,validation,panel=[],[],[]
    history_file=out/f"{arm.lower()}_training.parquet"
    validation_file=out/f"{arm.lower()}_validation.parquet"
    panel_file=out/f"{arm.lower()}_train_panel.parquet"
    resume_step=0
    checkpoints=sorted(checkpoint_dir.glob(f"{arm}_step*.pt"))
    if checkpoints:
        saved=torch.load(checkpoints[-1],map_location="cpu",weights_only=True)
        if saved["arm"]!=arm or saved["schedule_sha256"]!=schedule_sha(schedule) or saved["manifest_sha256"]!=manifest_hash:
            raise HardGate("resume provenance differs")
        p=restore_parameters(saved["parameters"],device,trainable=True)
        if parameters_sha(p)!=saved["parameters_sha256"]:
            raise HardGate("resume parameters differ")
        optimizer=torch.optim.Adam([p.a0,p.wq],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        optimizer.load_state_dict(saved["optimizer"])
        resume_step=int(saved["step"])
        torch.set_rng_state(saved["cpu_rng_state"])
        torch.cuda.set_rng_state(saved["cuda_rng_state"],device)
        trace=pd.read_parquet(history_file).query("step <= @resume_step").to_dict("records")
        validation=pd.read_parquet(validation_file).query("step <= @resume_step").to_dict("records")
        panel=pd.read_parquet(panel_file).query("step <= @resume_step").to_dict("records")
        print(f"resume {arm} at committed outer step {resume_step}",flush=True)

    @torch.no_grad()
    def evaluate(step):
        plain=copy_parameters(p)
        for kind,episodes,rows in (("validation",schedule["validation"],validation),("train_panel",schedule["train_panel"],panel)):
            for episode in episodes:
                budget.check();qid=episode["query"]
                if arm=="META_P1":
                    state,_=support_sequence(plain,[clean[i].unsqueeze(0).to(device) for i in episode["support"]])
                else:
                    state=initial_state(plain)
                before=state_fingerprint(state)
                prediction=read(plain,state,masked[qid].unsqueeze(0).to(device))
                target=targets[qid].to(device)
                rows.append({"arm":arm,"step":step,"kind":kind,"identity":qid,
                             "error":float(teacher_error(prediction,target)),
                             "prediction_rms":float(prediction[list(CENTER_IDS)].square().mean().sqrt()),
                             "completed_events":state.counters.completed})
                if before!=state_fingerprint(state):
                    raise HardGate("fit validation wrote state")
        write_table(out,validation_file.name,validation)
        write_table(out,panel_file.name,panel)

    if resume_step==0:
        evaluate(0)
    fit_started=sync(device)
    try:
        for episode in schedule["episodes"][resume_step:]:
            budget.check();step=episode["step"]
            started=sync(device);optimizer.zero_grad(set_to_none=True)
            if arm=="META_P1":
                state,product=support_sequence(p,[clean[i].unsqueeze(0).to(device) for i in episode["support"]])
            else:
                state=initial_state(p);product=torch.eye(p.dim,device=device)
            qid=episode["query"]
            prediction=read(p,state,masked[qid].unsqueeze(0).to(device))
            loss=teacher_error(prediction,targets[qid].to(device))
            loss.backward()
            if not torch.isfinite(loss) or any(v.grad is None or not bool(torch.isfinite(v.grad).all()) for v in (p.a0,p.wq)):
                raise HardGate("invalid or missing meta gradient")
            if any(v.grad is not None for v in (*p.auxiliary.values(),p.conv_weight,p.conv_bias)):
                raise HardGate("unauthorized gradient ownership")
            row={"arm":arm,"step":step,"query_identity":qid,"loss":float(loss.detach()),
                 "gradient_a0_norm":float(p.a0.grad.norm()),"gradient_wq_norm":float(p.wq.grad.norm()),
                 "a0_gradient_zero_fraction":float((p.a0.grad==0).float().mean()),
                 "wq_gradient_zero_fraction":float((p.wq.grad==0).float().mean()),
                 "prediction_rms":float(prediction.detach()[list(CENTER_IDS)].square().mean().sqrt()),
                 "support_completed_events":state.counters.completed,
                 "product_frobenius":float(product.norm()),"product_delta_norm":float((product-torch.eye(p.dim,device=device)).norm())}
            optimizer.step();p.validate()
            row.update({"a0_norm":float(p.a0.norm()),"wq_norm":float(p.wq.norm()),
                        "a0_drift":float((p.a0-initial.a0).norm()),"wq_drift":float((p.wq-initial.wq).norm())})
            del state,product,prediction,loss
            row["step_seconds"]=sync(device)-started
            row["peak_cuda_allocated_bytes"]=torch.cuda.max_memory_allocated(device)
            trace.append(row);budget.check()
            if step%10==0 or step==1:
                print(f"{arm} outer step {step}/200 loss={row['loss']:.7g} wall={budget.elapsed():.1f}s",flush=True)
                write_table(out,history_file.name,trace)
            if step==10:
                projected=(sum(r["step_seconds"] for r in trace)/10)*(400-step)+budget.elapsed()+600
                atomic_json(out/f"{arm.lower()}_runtime_projection.json",{"first10_seconds":sum(r["step_seconds"] for r in trace),
                    "projected_aggregate_seconds":projected,"budget_seconds":budget.limit,
                    "evaluation_replay_reserve_seconds":600})
                if projected>budget.limit:
                    raise HardGate("first ten steps project beyond aggregate budget")
            if step in (25,50,100,200):
                evaluate(step)
                write_table(out,history_file.name,trace)
                save_fit_checkpoint(checkpoint_dir/f"{arm}_step{step:03}.pt",p,optimizer,step,arm,schedule_sha(schedule),manifest_hash)
    except Exception:
        committed=trace[-1]["step"] if trace else resume_step
        write_table(out,history_file.name,trace)
        save_fit_checkpoint(checkpoint_dir/f"{arm}_step{committed:03}.pt",p,optimizer,committed,arm,schedule_sha(schedule),manifest_hash)
        raise
    elapsed=sync(device)-fit_started
    del optimizer
    return copy_parameters(p),trace,validation,panel,elapsed


def sequential_oracle_check(fixture, p, clean, identities):
    """Verify recurrence against equal-order writes, without reassociating GEMMs."""
    from exps.hope_image_synchronous_memory import ImageSynchronousMemory
    oracle_source = deepcopy(fixture["smt"])
    oracle_source["base_q.weight"] = p.wq.detach().cpu().clone()
    for suffix in ("weight", "initial_weight"):
        oracle_source["memories.memory." + suffix] = p.a0.detach().cpu().clone()
    oracle = ImageSynchronousMemory(oracle_source, "P1", device=p.a0.device)
    state = initial_state(p)
    product = torch.eye(p.dim, device=p.a0.device, dtype=p.a0.dtype)
    maximum = 0.0
    with torch.no_grad():
        for i in identities:
            image = clean[i].unsqueeze(0).to(p.a0.device)
            state, proposal = event(p, state, image)
            expected = oracle.propose_event(image)
            for name in state.weights:
                difference = float((state.weights[name] - expected.weights[name]).abs().max())
                maximum = max(maximum, difference)
                if not torch.equal(state.weights[name], expected.weights[name]):
                    raise HardGate("equal-order online oracle is not bitwise equivalent")
            if not torch.equal(proposal.transition, expected.transition):
                raise HardGate("online transition oracle differs")
            oracle.commit_event(expected)
            product = product @ proposal.transition
        if state.counters.completed != len(identities) or int(oracle.completed_events) != len(identities):
            raise HardGate("online oracle counters differ")
        composed = p.a0 @ product
        residual = state.weights["memory"] - composed
    return {"passed":True,"bitwise_equal_all_five_maps":True,"events":len(identities),
            "equal_order_oracle_max_abs":maximum,
            "reassociated_product_max_abs":float(residual.abs().max()),
            "reassociated_product_relative_l2":float(residual.norm()/(state.weights["memory"].norm()+1e-12)),
            "reassociation_is_oracle_criterion":False}


def evaluate_all(out,initial,meta,static,clean,masked,targets,schedule,budget,*,fixture=None,reference_source=None):
    device=initial.a0.device; started=sync(device)
    context_products={}; online_rows=[]
    # These references were specified and hashed before any target evaluation.
    for label,groups in (("pooled4",schedule["pooled4"]),("pooled50",schedule["pooled50"]),
                         ("distinct50",[schedule["wrong_history50"]]),("actual50",[schedule["online"]])):
        if reference_source is not None:
            context_products[label] = reference_source[label].to(device)
            continue
        products=[]
        for number,ids in enumerate(groups):
            state=initial_state(initial);product=torch.eye(initial.dim,device=device)
            for pos,i in enumerate(ids):
                budget.check();t=sync(device)
                state,proposal=event(initial,state,clean[i].unsqueeze(0).to(device))
                product=product@proposal.transition
                if label=="actual50":
                    row={"event":pos+1,"identity":i,"update_seconds":sync(device)-t,
                         "completed_events":state.counters.completed,"memory_updates":state.counters.memory,
                         "auxiliary_updates":state.counters.auxiliary,"online_updates":state.counters.online}
                    for name,v in state.weights.items():
                        row[name+"_norm"]=float(v.norm());row[name+"_max_abs"]=float(v.abs().max())
                    online_rows.append(row)
            products.append(product)
            if state.counters.completed!=len(ids):
                raise HardGate("history replay counters differ")
            print(f"reference {label} {number+1}/{len(groups)} complete",flush=True)
        context_products[label]=torch.stack(products).double().mean(0).float()
    atomic_torch(out/"reference_products.pt",{k:v.detach().cpu() for k,v in context_products.items()})
    write_table(out,"online_events.parquet",online_rows)
    # Both parameter pairs traverse actual events; a product shortcut is not an online training arm.
    online_states={}
    oracle_checks={}
    for arm,p in (("RAND_P1",initial),("META_P1",meta)):
        state=initial_state(p)
        for i in schedule["online"]:
            budget.check();state,_=event(p,state,clean[i].unsqueeze(0).to(device))
        state=detached_state(state)
        # Sequential (A T1) T2 and A (T1 T2) are different FP32 reductions.
        # The locked sequential oracle, with identical operation ordering, is
        # the correctness criterion; product differences are diagnostics only.
        if fixture is not None:
            oracle_checks[arm] = sequential_oracle_check(fixture,p,clean,schedule["online"])
        if state.counters.completed!=50 or inventory(p,state)["graph_bearing"]:
            raise HardGate("online event or graph lifecycle differs")
        online_states[arm]=state
        atomic_torch(out/f"{arm.lower()}_online_state.pt",{"weights":{k:v.detach().cpu() for k,v in state.weights.items()},
                      "counters":asdict(state.counters),"state_sha256":state_fingerprint(state),"parameters_sha256":parameters_sha(p)})
    atomic_json(out/"online_oracle_equivalence.json",oracle_checks)
    distinct_online_state, _ = support_sequence(meta,[clean[i].unsqueeze(0).to(device) for i in schedule["wrong_history50"]])
    distinct_online_state = detached_state(distinct_online_state)
    with torch.no_grad():
        h=torch.cat([query_rows(initial,masked[i].unsqueeze(0).to(device))[list(CENTER_IDS)] for i in range(60,80)])
        y=torch.cat([targets[i][list(CENTER_IDS)].to(device) for i in range(60,80)])
        fitted=ridge_fit(h,y)
    atomic_torch(out/"static_ridge.pt",{"coefficient":fitted[0].cpu(),"input_mean":fitted[1].cpu(),
                 "target_mean":fitted[2].cpu(),"penalty":fitted[3],"fit_identities":list(range(60,80))})
    results,interventions=[],[]
    validation_by_query={item["query"]:item for item in schedule["validation"]}
    with torch.no_grad():
        for split,ids in (("meta_val",range(92,100)),("normal_probe",range(150,170))):
            for i in ids:
                budget.check();image=masked[i].unsqueeze(0).to(device);target=targets[i].to(device)
                states={}
                for arm,p in (("RAND_P1",initial),("META_P1",meta)):
                    if split=="meta_val":
                        states[arm],_=support_sequence(p,[clean[j].unsqueeze(0).to(device) for j in validation_by_query[i]["support"]])
                    else:
                        states[arm]=online_states[arm]
                for arm in ARMS:
                    t=sync(device)
                    if arm=="STATIC_NORMAL_CONTROL":
                        prediction=ridge_read(query_rows(initial,image),fitted)
                    else:
                        p=meta if arm.startswith("META") else static if arm=="STATIC_META" else initial
                        state=states[arm] if arm.endswith("_P1") else initial_state(p)
                        before=state_fingerprint(state)
                        prediction=read(p,state,image)
                        if before!=state_fingerprint(state):
                            raise HardGate("evaluation mutated memory")
                    centered=prediction-prediction.mean(0)
                    results.append({"arm":arm,"split":split,"identity":i,"error":float(teacher_error(prediction,target)),
                                    "prediction_rms":float(prediction.square().mean().sqrt()),
                                    "centered_variance":float(centered.square().mean()),
                                    "target_cosine":float(F.cosine_similarity(prediction[list(CENTER_IDS)],target[list(CENTER_IDS)],dim=-1).mean()),
                                    "read_seconds":sync(device)-t,"read_only_verified":True})
                actual=read(meta,states["META_P1"],image)
                if split=="normal_probe":
                    variants={"actual_history":context_products["actual50"],"distinct_history":context_products["distinct50"],
                              "pooled_history":context_products["pooled50"],"reset":torch.eye(meta.dim,device=device)}
                else:
                    _,actual_product=support_sequence(meta,[clean[j].unsqueeze(0).to(device) for j in validation_by_query[i]["support"]])
                    distinct_state,distinct_product=support_sequence(meta,[clean[j].unsqueeze(0).to(device) for j in validation_by_query[i]["distinct_support"]])
                    variants={"actual_history":actual_product,"distinct_history":distinct_product,
                              "pooled_history":context_products["pooled4"],"reset":torch.eye(meta.dim,device=device)}
                for label,product in variants.items():
                    if label == "actual_history":
                        prediction = actual
                    elif label == "distinct_history":
                        prediction = read(meta,distinct_online_state if split == "normal_probe" else distinct_state,image)
                    else:
                        prediction=predict_with_product(meta,image,product)
                    interventions.append({"split":split,"identity":i,"history":label,"error":float(teacher_error(prediction,target)),
                                          "relative_l2_to_actual":float((prediction-actual).norm()/(actual.norm()+1e-12)),
                                          "cosine_to_actual":float(F.cosine_similarity(prediction.flatten(),actual.flatten(),dim=0)),
                                          "rms_ratio_to_actual":float(prediction.square().mean().sqrt()/(actual.square().mean().sqrt()+1e-12))})
    write_table(out,"six_arm_evaluation.parquet",results)
    write_table(out,"history_interventions.parquet",interventions)
    frame=pd.DataFrame(results);intervention_frame=pd.DataFrame(interventions)
    deltas={}
    for split in ("meta_val","normal_probe"):
        pivot=frame[frame.split==split].pivot(index="identity",columns="arm",values="error")
        comparisons={"Delta_paired":("META_P1","META_FROZEN"),"Delta_static":("META_P1","STATIC_META"),
                     "Delta_rand":("RAND_P1","RAND_FROZEN"),"Delta_ridge":("META_P1","STATIC_NORMAL_CONTROL")}
        deltas[split]={key:paired_bootstrap(pivot[a].values,pivot[b].values) for key,(a,b) in comparisons.items()}
        meta_gain=pivot.META_FROZEN-pivot.META_P1;rand_gain=pivot.RAND_FROZEN-pivot.RAND_P1
        deltas[split]["Interaction"]=paired_bootstrap((rand_gain-meta_gain).values,torch.zeros(len(pivot)).numpy())
        histories=intervention_frame[intervention_frame.split==split].pivot(index="identity",columns="history",values="error")
        for key,other in (("Delta_history","distinct_history"),("Delta_pool","pooled_history")):
            deltas[split][key]=paired_bootstrap(histories.actual_history.values,histories[other].values)
        deltas[split]["relative_paired_error_reduction"]=float((pivot.META_FROZEN-pivot.META_P1).mean()/pivot.META_FROZEN.mean())
    atomic_json(out/"causal_deltas.json",deltas)
    storage={}
    for arm in ARMS:
        p=meta if arm.startswith("META") else static if arm=="STATIC_META" else initial
        if arm=="STATIC_NORMAL_CONTROL":
            storage[arm]={"persistent_bytes":sum(v.numel()*v.element_size() for v in fitted[:3])+
                          parameter_storage_bytes(p),
                          "optimizer_bytes_deployed":0,"fixed_basis":True}
        elif arm.endswith("_P1"):
            storage[arm]=inventory(p,online_states[arm])
        else:
            storage[arm]={"persistent_bytes":parameter_storage_bytes(p),
                          "optimizer_bytes_deployed":0}
    storage["evaluator_only_products_bytes"]=sum(v.numel()*v.element_size() for v in context_products.values())
    atomic_json(out/"storage.json",storage)
    return deltas,results,interventions,sync(device)-started


def complete_evaluation(source,device):
    """Resume only missing evaluation from verified terminal fitted parameters."""
    out=source/"evaluation_completion"
    out.mkdir(exist_ok=True)
    budget=Budget(source,device)
    torch.set_num_threads(2);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    integrity=verify_integrity()
    saved_summary=json.loads((source/"summary.json").read_text())
    if saved_summary["steps_meta_p1"]!=200 or saved_summary["steps_static_meta"]!=200 or saved_summary["preflight_status"]!="PASS":
        raise HardGate("evaluation resume lacks complete preflight/training")
    schedule=json.loads((source/"episode_schedule.json").read_text())
    manifest_hash=file_sha(source/"stream_manifest.parquet")
    initial_fixture=torch.load(SOURCE_FIXTURE,map_location="cpu",weights_only=False)
    initial=parameters_from_fixture(initial_fixture,device)
    arms={}
    provenance={"source_directory":str(source),"new_optimizer_steps":0,"schedule_sha256":schedule_sha(schedule),
                "manifest_sha256":manifest_hash,"terminal_checkpoints":{}}
    for arm in ("META_P1","STATIC_META"):
        path=source/"checkpoints"/f"{arm}_step200.pt"
        saved=torch.load(path,map_location="cpu",weights_only=True)
        if saved["step"]!=200 or saved["arm"]!=arm or saved["schedule_sha256"]!=schedule_sha(schedule) or saved["manifest_sha256"]!=manifest_hash:
            raise HardGate("terminal checkpoint provenance differs")
        p=restore_parameters(saved["parameters"],device)
        if parameters_sha(p)!=saved["parameters_sha256"]:
            raise HardGate("terminal parameter fingerprint differs")
        arms[arm]=p
        provenance["terminal_checkpoints"][arm]={"path":str(path),"sha256":file_sha(path),"parameters_sha256":parameters_sha(p)}
    if file_sha(SOURCE_CACHE)!=EXPECTED_CACHE_SHA:
        raise HardGate("evaluation clean cache differs")
    payload=torch.load(SOURCE_CACHE,map_location="cpu",weights_only=False,mmap=True)
    rows=validate_bottle_manifest(ROOT/"data/mvtec",payload,EXPECTED_CACHE_SHA)
    original_rows=pd.read_parquet(source/"stream_manifest.parquet").to_dict("records")
    if rows!=original_rows:
        raise HardGate("evaluation raw-image manifest differs")
    masked_payload=torch.load(source/"masked_features.pt",map_location="cpu",weights_only=True)
    masked=masked_payload["features"]
    for i,value in masked.items():
        if tensor_sha(value)!=masked_payload["feature_hashes"][str(i)]:
            raise HardGate("evaluation masked feature differs")
    clean=payload["patches"]
    targets={i:F.normalize(clean[i],dim=-1,eps=1e-8) for i in masked}
    references=torch.load(source/"reference_products.pt",map_location="cpu",weights_only=True)
    provenance["reference_products_sha256"]=file_sha(source/"reference_products.pt")
    atomic_json(out/"resume_provenance.json",provenance)
    result={"active_task_id":"OL-04","preflight_status":"PASS","steps_meta_p1":200,"steps_static_meta":200,
            "training_status":"COMPLETE","evaluation_status":"PENDING","new_optimizer_steps":0,
            "stage0_verified":True,"data_manifest_valid":True,"production_core_unchanged":True,
            "future_categories_accessed":False,"confirmation_accessed":False,"anomaly_evaluation_performed":False,
            "scientific_decision":"NOT_EVALUATED","ol05_authorized":False}
    try:
        deltas,_,_,seconds=evaluate_all(out,initial,arms["META_P1"],arms["STATIC_META"],clean,masked,targets,schedule,budget,
                                      fixture=initial_fixture,reference_source=references)
        valframe=pd.read_parquet(source/"meta_p1_validation.parquet")
        starting=float(valframe[valframe.step==0].error.mean())
        terminal=float(valframe[valframe.step==200].error.mean())
        screen={"terminal_meta_validation_improvement":1-terminal/starting,
                "normal_probe_paired_relative_improvement":deltas["normal_probe"]["relative_paired_error_reduction"],
                "positive_validation_images":deltas["meta_val"]["Delta_paired"]["positive_images"],
                "post_online_probe_delta":deltas["normal_probe"]["Delta_paired"]["estimate"]}
        passes=screen["terminal_meta_validation_improvement"]>=.05 and screen["normal_probe_paired_relative_improvement"]>=.01 and screen["positive_validation_images"]>=6 and screen["post_online_probe_delta"]>0
        uncertain=any(deltas[s]["Delta_paired"]["ci_low"]<=0<=deltas[s]["Delta_paired"]["ci_high"] for s in ("meta_val","normal_probe"))
        history_supported=all(deltas["normal_probe"][key]["ci_low"]>0 for key in ("Delta_history","Delta_pool","Delta_static"))
        result.update({"status":"COMPLETE","evaluation_status":"COMPLETE","functional_screen":screen,
                       "screen_all_margins_pass":passes,"causal_deltas":deltas,
                       "scientific_decision":"INCONCLUSIVE" if passes and uncertain else "FUNCTIONAL_GO" if passes else "SCIENTIFIC_NO_GO",
                       "normal_acquisition":"SUPPORTED" if passes and not uncertain else "INCONCLUSIVE" if passes else "NOT_SUPPORTED",
                       "history_specificity":"SUPPORTED" if history_supported else "INCONCLUSIVE",
                       "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(device),
                       "cpu_max_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                       "evaluation_seconds":seconds,"wall_seconds_including_implementation":budget.elapsed(),
                       "original_runtime_failure_preserved":True})
        budget.check()
        for path,sha in integrity["sources"].items():
            if file_sha(ROOT/path)!=sha:
                raise HardGate("protected source changed during evaluation")
    except Exception as exc:
        result.update({"status":"FAILED","evaluation_status":"FAILED","failure_reason":str(exc)})
        atomic_json(out/"failure.json",{"reason":str(exc),"traceback":traceback.format_exc()})
        print(traceback.format_exc(),flush=True)
    atomic_json(out/"summary.json",result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def run(out,device):
    out.mkdir(parents=True,exist_ok=True)
    budget=Budget(out,device)
    summary={"active_task_id":"OL-04","stage0_verified":False,"data_manifest_valid":False,
             "preflight_status":"PENDING","training_status":"BLOCKED","steps_meta_p1":0,"steps_static_meta":0,
             "scientific_decision":"NOT_EVALUATED","production_core_unchanged":True,
             "confirmation_accessed":False,"future_categories_accessed":False,"anomaly_evaluation_performed":False,
             "ol05_authorized":False,"shared_gpu_explicitly_authorized":True,"device":str(device)}
    integrity=None
    try:
        integrity=verify_integrity();summary["stage0_verified"]=True
        atomic_json(out/"stage0_integrity.json",integrity)
        torch.set_num_threads(2);torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
        torch.cuda.set_device(device)
        total=torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0,8*1024**3/total),device)
        atomic_json(out/"resource_start.json",{"gpu":torch.cuda.get_device_name(device),
                   "free_total_bytes":list(torch.cuda.mem_get_info(device)),
                   "shared_use_authorized":True,"torch":torch.__version__,"cuda":torch.version.cuda,
                   "budget_seconds":5400,"allocated_cap_bytes":8*1024**3})
        schedule=make_schedule();atomic_json(out/"episode_schedule.json",schedule)
        atomic_json(out/"config_resolved.yaml",{"seed":0,"dtype":"float32","device":str(device),"CMS":"DISABLED","P2":"DISABLED",
                   "inner_h":.02,"alpha_image":1,"trainable_parameters":["A0","Wq"],"parameter_count":1179648,
                   "support_length":4,"query_images_per_episode":1,"outer_steps_each":200,
                   "optimizer":{"name":"Adam","lr":1e-4,"betas":[.9,.999],"eps":1e-8,"weight_decay":0},
                   "arms":list(ARMS),"validation_steps":[0,25,50,100,200],"schedule_sha256":schedule_sha(schedule),
                   "mask_centers":[list(c) for c in CENTERS],"mask_view":"ONE_SHARED_16_REGION_VIEW",
                   "mask_size_pixels":[24,24],"mask_fill":"ImageNet mean RGB / normalized zero",
                   "outer_loss":"0.5 * mean_centers(sum_channels((A_support q_masked - unit_clean_teacher)^2))",
                   "pooled50_products":8,"allocated_memory_cap_bytes":8*1024**3,"aggregate_wall_cap_seconds":5400,
                   "shared_gpu_user_override":True,"feature_metadata":None})
        cache_sha=file_sha(SOURCE_CACHE)
        payload=torch.load(SOURCE_CACHE,map_location="cpu",weights_only=False,mmap=True)
        rows=validate_bottle_manifest(ROOT/"data/mvtec",payload,cache_sha)
        write_table(out,"stream_manifest.parquet",rows)
        manifest_hash=file_sha(out/"stream_manifest.parquet")
        atomic_json(out/"manifest_identity.json",{"sha256":manifest_hash,"roles":{r:sum(v["role"]==r for v in rows) for r in set(v["role"] for v in rows)},
                   "feature_metadata":payload["metadata"],"cache_sha256":cache_sha,"source_cache_path":str(SOURCE_CACHE),
                   "raw_images":170,"raw_content_disjoint":True})
        summary["data_manifest_valid"]=True
        print("approved bottle manifest verified: 170 unique train/good identities",flush=True)
        fixture=torch.load(SOURCE_FIXTURE,map_location="cpu",weights_only=False)
        initial=parameters_from_fixture(fixture,device)
        atomic_json(out/"initial_state_manifest.json",{"fixture_sha256":file_sha(SOURCE_FIXTURE),
                   "fixture_initialization_hash":fixture["initialization_hash"],"parameters_sha256":parameters_sha(initial),
                   "inventory":inventory(initial,initial_state(initial))})
        masked,view_diagnostics=extract_masked(out,payload,rows,device,budget)
        clean=payload["patches"]
        targets={i:F.normalize(clean[i],dim=-1,eps=1e-8) for i in masked}
        preflight=nontriviality(torch.stack([clean[i] for i in range(92,100)]),
                               torch.stack([masked[i] for i in range(92,100)]),
                               torch.stack([targets[i] for i in range(60,80)]))
        atomic_json(out/"mask_nontriviality.json",preflight)
        if not preflight["passed"]:
            raise HardGate("masked-view nontriviality heuristic failed")
        parity=oracle_parity(fixture,initial,clean[0].unsqueeze(0).to(device))
        atomic_json(out/"real_forward_parity.json",parity)
        smoke=copy_parameters(initial,trainable=True)
        before=[tensor_sha(v) for v in smoke.auxiliary.values()]
        state,product=support_sequence(smoke,[clean[i].unsqueeze(0).to(device) for i in schedule["episodes"][0]["support"]])
        prediction=read(smoke,state,masked[60].unsqueeze(0).to(device))
        loss=teacher_error(prediction,targets[60].to(device));loss.backward()
        if any(v.grad is None or not bool(torch.isfinite(v.grad).all()) or float(v.grad.norm())==0 for v in (smoke.a0,smoke.wq)):
            raise HardGate("real backward smoke failed")
        if product.requires_grad or before!=[tensor_sha(v) for v in smoke.auxiliary.values()]:
            raise HardGate("real frozen-policy isolation failed")
        export=detached_state(state)
        if inventory(smoke,export)["graph_bearing"]:
            raise HardGate("deployment graph leak")
        smoke_results={"passed":True,"optimizer_steps":0,"loss":float(loss.detach()),
                       "a0_gradient_norm":float(smoke.a0.grad.norm()),"wq_gradient_norm":float(smoke.wq.grad.norm()),
                       "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(device),
                       "completed_support_events":state.counters.completed}
        atomic_json(out/"real_backward_smoke.json",smoke_results)
        del smoke,state,product,prediction,loss,export;gc.collect();torch.cuda.empty_cache();budget.check()
        summary["preflight_status"]="PASS"
        atomic_json(out/"summary.json",summary)
        print("all preflight gates PASS; automatically starting bounded outer fits",flush=True)
        meta,meta_trace,meta_validation,meta_panel,meta_seconds=fit_arm(out,"META_P1",initial,clean,masked,targets,schedule,manifest_hash,budget)
        summary["steps_meta_p1"]=len(meta_trace);atomic_json(out/"summary.json",summary)
        static,static_trace,static_validation,static_panel,static_seconds=fit_arm(out,"STATIC_META",initial,clean,masked,targets,schedule,manifest_hash,budget)
        summary["steps_static_meta"]=len(static_trace)
        if len(meta_trace)!=200 or len(static_trace)!=200:
            raise HardGate("optimizer step count is incomplete")
        deltas,eval_rows,interventions,eval_seconds=evaluate_all(out,initial,meta,static,clean,masked,targets,schedule,budget,fixture=fixture)
        valframe=pd.DataFrame(meta_validation)
        initial_error=float(valframe[valframe.step==0].error.mean())
        terminal_error=float(valframe[valframe.step==200].error.mean())
        screen={"terminal_meta_validation_improvement":1-terminal_error/initial_error,
                "normal_probe_paired_relative_improvement":deltas["normal_probe"]["relative_paired_error_reduction"],
                "positive_validation_images":deltas["meta_val"]["Delta_paired"]["positive_images"],
                "post_online_probe_delta":deltas["normal_probe"]["Delta_paired"]["estimate"]}
        passes=screen["terminal_meta_validation_improvement"]>=.05 and screen["normal_probe_paired_relative_improvement"]>=.01 and screen["positive_validation_images"]>=6 and screen["post_online_probe_delta"]>0
        uncertain=any(deltas[s]["Delta_paired"]["ci_low"]<=0<=deltas[s]["Delta_paired"]["ci_high"] for s in ("meta_val","normal_probe"))
        history_supported=all(deltas["normal_probe"][key]["ci_low"]>0 for key in ("Delta_history","Delta_pool","Delta_static"))
        summary.update({"training_status":"COMPLETE","status":"COMPLETE","scientific_decision":"INCONCLUSIVE" if passes and uncertain else "FUNCTIONAL_GO" if passes else "SCIENTIFIC_NO_GO",
                        "functional_screen":screen,"screen_all_margins_pass":passes,"causal_deltas":deltas,
                        "normal_acquisition":"SUPPORTED" if passes and not uncertain else "INCONCLUSIVE" if passes else "NOT_SUPPORTED",
                        "history_specificity":"SUPPORTED" if history_supported else "INCONCLUSIVE",
                        "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(device),
                        "cpu_max_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                        "wall_seconds_including_implementation":budget.elapsed(),
                        "timing":{"meta_fit_seconds":meta_seconds,"static_fit_seconds":static_seconds,"causal_evaluation_seconds":eval_seconds},
                        "dense_anomaly_map_validated":False})
        budget.check()
    except Exception as exc:
        summary.update({"status":"BLOCKED" if summary["preflight_status"]!="PASS" else "FAILED",
                        "preflight_status":"FAIL" if summary["preflight_status"]!="PASS" else "PASS",
                        "training_status":"BLOCKED" if summary["preflight_status"]!="PASS" else "INCOMPLETE_BUDGET" if "budget" in str(exc) else "FAILED",
                        "failure_reason":str(exc),"failure_type":type(exc).__name__,
                        "scientific_decision":"NOT_EVALUATED","wall_seconds_including_implementation":budget.elapsed()})
        atomic_json(out/"failure.json",{"reason":str(exc),"traceback":traceback.format_exc()})
        print(traceback.format_exc(),flush=True)
    finally:
        for arm, key in (("META_P1", "steps_meta_p1"), ("STATIC_META", "steps_static_meta")):
            path = out / f"{arm.lower()}_training.parquet"
            if path.exists():
                summary[key] = len(pd.read_parquet(path))
        if integrity is not None:
            changed=[path for path,sha in integrity["sources"].items() if file_sha(ROOT/path)!=sha]
            if changed:
                summary["production_core_unchanged"]=False
                summary["failure_reason"]="protected sources changed: "+", ".join(changed)
                summary["status"]="FAILED"
        atomic_json(out/"summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--device",default="cuda:0",choices=("cuda:0",))
    parser.add_argument("--evaluation-only",action="store_true")
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable in the selected environment")
    summary=(complete_evaluation if args.evaluation_only else run)(args.output_dir.resolve(),torch.device(args.device))
    raise SystemExit(0 if summary.get("status")=="COMPLETE" else 1)


if __name__=="__main__":
    main()
