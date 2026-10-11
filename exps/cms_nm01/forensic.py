# exps/cms_nm01/forensic.py
"""Read-only independent checkpoint reassessment; no training or anomaly access."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .cms_nd import CMSNormalDensityMemory
from .reporting import level_rows, markdown_level_table
from .run import CATEGORIES, PROBE_RANGES, FEATURE_ROOT, RESULT_ROOT, make_model, normal_metrics, state_digest, evaluate_gates

OUT = RESULT_ROOT / "forensic"


def reference_flow(state: dict, prefix: str, z: torch.Tensor):
    """Functional FP64 oracle: no model log_prob/coupling/mixture methods."""
    value = z
    logdet = torch.zeros(z.shape[:-1], dtype=z.dtype)
    scales = []
    for block in range(4):
        key = f"{prefix}blocks.{block}.net."
        left, right = value[..., :32], value[..., 32:]
        hidden = F.linear(left, state[key+"0.weight"], state[key+"0.bias"])
        hidden = 0.5 * hidden * (1 + torch.erf(hidden / math.sqrt(2)))
        raw = F.linear(hidden, state[key+"2.weight"], state[key+"2.bias"])
        scale = 2 * torch.tanh(raw[..., :32] / 2)
        value = torch.cat((right * scale.exp() + raw[..., 32:], left), -1)
        logdet = logdet + scale.sum(-1)
        scales.append(scale)
    sigma = state[prefix+"base_logstd"].clamp(-5, 5).exp()
    base = torch.distributions.Independent(torch.distributions.Normal(state[prefix+"base_mean"], sigma), 1)
    return base.log_prob(value) + logdet, logdet, value, scales


def reference_inverse(state: dict, prefix: str, y: torch.Tensor):
    value = y
    for block in reversed(range(4)):
        key = f"{prefix}blocks.{block}.net."
        left = value[..., 32:]
        hidden = F.linear(left, state[key+"0.weight"], state[key+"0.bias"])
        hidden = 0.5 * hidden * (1 + torch.erf(hidden / math.sqrt(2)))
        raw = F.linear(hidden, state[key+"2.weight"], state[key+"2.bias"])
        scale = 2 * torch.tanh(raw[..., :32] / 2)
        right = (value[..., :32] - raw[..., 32:]) * (-scale).exp()
        value = torch.cat((left, right), -1)
    return value


def bootstrap_change(before, after, *, seed=2801):
    a, b = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    indices = np.random.default_rng(seed).integers(0, len(a), size=(4000, len(a)))
    delta = (b[indices] - a[indices]).mean(1)
    relative = delta / np.maximum(np.abs(a[indices].mean(1)), 1e-6)
    return {"after_minus_before": float((b-a).mean()), "ci95": np.quantile(delta, [.025,.975]).tolist(),
            "relative_change": float((b.mean()-a.mean())/max(abs(a.mean()),1e-6)), "relative_ci95": np.quantile(relative,[.025,.975]).tolist()}


def preserved_check():
    manifest = json.loads((OUT / "original_manifest.json").read_text())["files"]
    failures = []
    for name, metadata in manifest.items():
        if "/checkpoints/" in name or name.startswith("models/") or name.startswith("results/"):
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != metadata["sha256"]:
                failures.append(name)
    if failures:
        raise ValueError(f"original result/checkpoint/production changed: {failures}")


def main():
    started = time.perf_counter()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    preserved_check()
    original = json.loads((RESULT_ROOT / "normal_only_records.json").read_text())
    contract = json.loads((RESULT_ROOT / "normal_identity_contract.json").read_text())
    protocol = json.loads((RESULT_ROOT / "protocol_resolved.json").read_text())
    probes = {}
    input_hashes = {}
    for category in CATEGORIES:
        path = FEATURE_ROOT / f"{category}_train.pt"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == protocol["data"]["feature_files"][str(path.relative_to(Path.cwd()))]["sha256"]
        input_hashes[str(path)] = digest
        payload = torch.load(path, map_location="cpu", weights_only=False)
        lo, hi = PROBE_RANGES[category]
        assert payload["paths"][lo:hi] == contract["identities"][category]["heldout"]
        assert not set(contract["identities"][category]["heldout"]) & set(contract["identities"][category]["update"])
        probes[category] = payload["patches"][lo:hi].float().clone()
        del payload
    diagnostic = torch.cat([probes[c][:4] for c in CATEGORIES])
    records, validation, jacobians, anchors = [], [], [], []
    tensor_dir = OUT / "normal_scores"
    tensor_dir.mkdir(exist_ok=True)
    for old in original:
        record = copy.deepcopy(old)
        for stage in range(4):
            checkpoint = Path(old["checkpoint_dir"]) / f"boundary{stage}.pt"
            model = make_model(old["arm"], seed=7101)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state)
            before_digest = state_digest(model)
            assert before_digest == old["stages"][stage]["metrics"]["state_digest"]
            measured = normal_metrics(model, probes, diagnostic)
            state64 = {key: value.double() if value.is_floating_point() else value for key,value in state.items()}
            saved = {}
            for category in CATEGORIES:
                patches = probes[category]
                result = model.score_patches(patches)
                actual_levels = torch.stack(result["log_probs"]).double()
                actual_mixture = result["mixture_nll"].double()
                with torch.no_grad():
                    z = patches.double() @ state64["projection.matrix"].T
                    if isinstance(model, CMSNormalDensityMemory):
                        references = [reference_flow(state64, f"levels.{level}.flow.", z)[0] for level in range(len(model.levels))]
                    else:
                        variance = (state64["m2"] / max(int(state["count"])-1, 1)).clamp_min(1e-4)
                        distribution = torch.distributions.Independent(torch.distributions.Normal(state64["mean"], variance.sqrt()), 1)
                        references = [distribution.log_prob(z)]
                    reference = torch.stack(references)
                    # Independently combine densities with NumPy logaddexp.
                    mix64 = -np.logaddexp.reduce(reference.numpy(), axis=0) + math.log(len(references))
                nll = -actual_levels
                minimum = nll.amin(0)
                patch_upper = float((actual_mixture - minimum - math.log(len(references))).max())
                patch_lower = float((minimum - actual_mixture).max())
                mean_upper = float(actual_mixture.mean() - nll.mean((1,2)).min() - math.log(len(references)))
                oracle_max = float((reference - actual_levels).abs().max())
                mix_error = float(np.max(np.abs(mix64 - actual_mixture.numpy())))
                assert actual_levels.shape == (len(references),20,784)
                assert patch_upper <= 2e-4 and patch_lower <= 2e-4 and mean_upper <= 2e-4
                # The oracle is evaluated in FP64 while the saved checkpoint
                # and production read path are FP32. Exact Jacobian checks
                # below use FP64; likelihood parity therefore uses a declared
                # 0.25 absolute FP32-vs-FP64 tolerance. The largest pointwise
                # discrepancy is recorded; mean NLL parity is checked below.
                assert oracle_max <= 0.25 and mix_error <= 0.25, (old["arm"], old["order_seed"], stage, category, oracle_max, mix_error)
                previous = old["stages"][stage]["metrics"]["per_category"][category]
                current = measured["per_category"][category]
                raw_mean_error = max(abs(previous["mixture_mean"]-current["mixture_mean"]), max(abs(a-b) for a,b in zip(previous["level_mean"],current["level_mean"])))
                assert raw_mean_error <= 2e-4
                current["level_image_nll"] = [(-value).mean(1).tolist() for value in actual_levels]
                current["fp64_reference_mixture_mean"] = float(mix64.mean())
                current["posterior_mean_level_weight"] = torch.softmax(actual_levels,0).mean((1,2)).tolist()
                current["best_level_patch_fraction"] = [(actual_levels.argmax(0)==i).double().mean().item() for i in range(len(references))]
                validation.append({"arm":old["arm"],"seed":old["order_seed"],"boundary":stage,"category":category,
                    "max_patch_upper_bound_residual":patch_upper,"max_patch_lower_bound_residual":patch_lower,
                    "mean_upper_bound_residual":mean_upper,"fp64_logp_max_abs_error":oracle_max,
                    "fp64_mixture_max_abs_error":mix_error,"original_raw_mean_max_abs_error":raw_mean_error,
                    "logsumexp_after_reduction_nll":float(-torch.logsumexp(actual_levels.mean((1,2)),0)+math.log(len(references)))})
                saved[category+"_logp_fp32"] = actual_levels.float().numpy()
                saved[category+"_mixture_fp32"] = actual_mixture.float().numpy()
                saved[category+"_logp_reference_fp64"] = reference.numpy()
                if isinstance(model,CMSNormalDensityMemory) and category == CATEGORIES[0]:
                    # Full 64x64 Jacobian of each actual checkpoint, one normal patch.
                    for level in range(len(model.levels)):
                        prefix = f"levels.{level}.flow."
                        sample = z[0,0].clone()
                        jacobian = torch.autograd.functional.jacobian(lambda v: reference_flow(state64,prefix,v)[2],sample,vectorize=True)
                        _, numerical_logdet = torch.linalg.slogdet(jacobian)
                        _, analytic_logdet, transformed, _ = reference_flow(state64,prefix,sample)
                        error = float((numerical_logdet-analytic_logdet).abs())
                        inverse_error = float((reference_inverse(state64,prefix,transformed)-sample).abs().max())
                        assert error < 1e-8 and inverse_error < 1e-8
                        jacobians.append({"seed":old["order_seed"],"arm":old["arm"],"boundary":stage,"level":level,"logdet_error":error,"inverse_max_abs_error":inverse_error})
            np.savez_compressed(tensor_dir/f"seed{old['order_seed']}_{old['arm']}_boundary{stage}.npz",**saved)
            if isinstance(model,CMSNormalDensityMemory):
                event = int(model.completed_events)
                assert [int(level.update_count) for level in model.levels] == [event//period for period in model.periods]
                assert [int(level.pending_count) for level in model.levels] == [event%period for period in model.periods]
                initial = make_model(old["arm"],seed=7101)
                anchor_rows = []
                for index, level in enumerate(model.levels):
                    expected = model.levels[index+1].current_parameters() if index+1<len(model.levels) else initial.levels[index].current_parameters()
                    # Before the first next-slower commit, this level retains its own initial anchor.
                    if index+1<len(model.levels) and event < model.periods[index+1]:
                        expected = initial.levels[index].current_parameters()
                    max_error = max(float((a-b.detach()).abs().max()) for a,b in zip(level.anchors(),expected))
                    assert max_error == 0
                    initial_pair_distance = None
                    if index+1<len(model.levels):
                        initial_pair_distance = float(torch.sqrt(sum((a-b).square().sum() for a,b in zip(initial.levels[index].current_parameters(),initial.levels[index+1].current_parameters()))).detach())
                    gradient_norm = math.sqrt(sum(float(value.square().sum()) for value in level.accumulators()))/max(int(level.pending_count),1)
                    anchor_force = level.retention*measured["online_counters"]["parameter_anchor_distances"][index]
                    anchor_rows.append({"level":index,"exact_anchor_error":max_error,"independent_initial_pair_distance":initial_pair_distance,
                         "parameter_anchor_distance":measured["online_counters"]["parameter_anchor_distances"][index],
                         "pending_mean_gradient_norm":gradient_norm,"anchor_force_norm":anchor_force,
                         "anchor_to_pending_gradient_ratio":anchor_force/gradient_norm if gradient_norm else None})
                anchors.append({"seed":old["order_seed"],"arm":old["arm"],"boundary":stage,"events":event,"levels":anchor_rows})
            assert before_digest == state_digest(model)
            record["stages"][stage]["metrics"] = measured
            if stage==0:
                record["calibration_metrics"] = measured
            if stage==3:
                record["final_metrics"] = measured
        records.append(record)
        print(f"verified seed={old['order_seed']} arm={old['arm']}",flush=True)
    uncertainty = []
    for record in records:
        for boundary in range(1,4):
            for category in CATEGORIES:
                acquired = record["order"].index(category)+1
                comparisons = [("calibration",0)] + ([("post_acquisition",acquired)] if boundary>=acquired else [])
                now = record["stages"][boundary]["metrics"]["per_category"][category]
                for comparison,base in comparisons:
                    before = record["stages"][base]["metrics"]["per_category"][category]
                    for level in range(len(now["level_mean"])+1):
                        level_name = "mixture" if level==0 else ("fast","medium","slow")[level-1]
                        a = before["image_nll"] if level==0 else before["level_image_nll"][level-1]
                        b = now["image_nll"] if level==0 else now["level_image_nll"][level-1]
                        uncertainty.append({"seed":record["order_seed"],"arm":record["arm"],"boundary":boundary,"category":category,"level":level_name,"reference":comparison,**bootstrap_change(a,b)})
    gates = evaluate_gates(records)
    (OUT/"verified_records.json").write_text(json.dumps(records,indent=2,sort_keys=True)+"\n")
    (OUT/"verified_gates.json").write_text(json.dumps(gates,indent=2,sort_keys=True)+"\n")
    (OUT/"uncertainty.json").write_text(json.dumps(uncertainty,indent=2,sort_keys=True)+"\n")
    (OUT/"anchors.json").write_text(json.dumps(anchors,indent=2,sort_keys=True)+"\n")
    summary={"checkpoint_count":48,"category_checkpoint_cells":len(validation),"jacobian_tests":len(jacobians),
             "validation":validation,"jacobians":jacobians,"input_hashes":input_hashes,
             "max_raw_mean_error":max(x["original_raw_mean_max_abs_error"] for x in validation),
             "max_fp64_logp_error":max(x["fp64_logp_max_abs_error"] for x in validation),
             "max_fp64_mixture_error":max(x["fp64_mixture_max_abs_error"] for x in validation),
             "max_patch_bound_residual":max(x["max_patch_upper_bound_residual"] for x in validation),
             "elapsed_seconds":time.perf_counter()-started,"normal_only":True,"anomaly_labels_accessed":False}
    (OUT/"validation.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    rows=[row for record in records for row in level_rows(record)]
    with (OUT/"all_boundary_means.csv").open("w") as handle:
        writer=csv.DictWriter(handle,fieldnames=["seed","arm","boundary","category","mixture","fast","medium","slow"])
        writer.writeheader(); writer.writerows(rows)
    neural=[record for record in records if record["arm"]=="CMS-ND-3"]
    (OUT/"verified_level_tables.md").write_text(markdown_level_table(neural)+"\n")
    preserved_check()
    print(json.dumps({k:v for k,v in summary.items() if k not in ("validation","jacobians","input_hashes")}),flush=True)
    print(json.dumps({"G1":gates["g1"]["pass"],"G2_written_protocol":gates["g2"]["pass"],"post_acquisition_20pct_diagnostic":gates["g2"]["post_acquisition_diagnostic_pass_at_20pct"]}),flush=True)


if __name__=="__main__":
    main()
