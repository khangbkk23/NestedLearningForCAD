"""Read-only measurement analysis; never fits, reads features, or evaluates a model.

The frozen runner completed its measurements but its JSON summary contained numpy.bool_.
This separate finalizer preserves the runner/protocol and recovers the exact registered summary.
"""
import io
import json
from pathlib import Path

import numpy as np
import torch

from exps.r1.run_r1_conditional_pairing import OUT, ROOT, ARMS, CATEGORIES, sha, summarize, tensor_bytes


def native(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [native(v) for v in value]
    return value


def save(name, value):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + ".part")
    tmp.write_text(json.dumps(native(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


def serialized_bytes(value):
    b = io.BytesIO(); torch.save(value, b); return b.tell()


def main():
    p = json.loads((OUT / "protocol_resolved.json").read_text())
    assert sha(OUT / "protocol_resolved.json") == (OUT / "protocol.sha256").read_text().strip()
    assert sha(ROOT / "exps/r1/run_r1_conditional_pairing.py") == p["runner_sha256"]
    assert all(sha(ROOT / f) == h for f, h in p["preserved_sha256"].items())
    errors = json.loads((OUT / "execution_error.json").read_text())
    assert errors["type"] == "TypeError" and "JSON serializable" in errors["error"]
    per_image = json.loads((OUT / "per_image_results.json").read_text())
    boundaries = json.loads((OUT / "boundaries.json").read_text())
    trace = json.loads((OUT / "training_trace.json").read_text())
    chain = json.loads((OUT / "state_chain.json").read_text())
    access = json.loads((OUT / "access_log.json").read_text())
    assert len(per_image) == 180 and len(trace) == 3500 and len(chain["blocks"]) == 35
    assert [b["event"] for b in boundaries] == [100, 200, 300, 350]
    assert [set(b["categories"]) for b in boundaries] == [set(CATEGORIES[:1]), set(CATEGORIES[:2]), set(CATEGORIES), set(CATEGORIES)]
    assert all(b["optimizer_steps"] == {a: b["event"] * 5 for a in ("shuffled", "neural")} for b in boundaries)
    assert all(r["parameter_delta_l2"] > 0 and r["output_delta_rms_l2"] > 0 for r in trace)
    assert all(r["state_unchanged"] and r["parameter_delta_l2"] == 0 and r["output_delta_rms_l2"] == 0 for r in chain["no_write"])
    assert all(a["after"] == b["before"] for a, b in zip(chain["blocks"], chain["blocks"][1:]))
    assert chain["blocks"][0]["before"] == chain["initial"]
    update_paths = [r["path"] for r in access if r["role"] == "update"]
    assert update_paths == [r["relative_path"] for r in p["updates"]]
    allowed_probes = {r["relative_path"] for r in p["probes"]}
    assert {r["path"] for r in access if r["role"] == "probe"} == allowed_probes
    # Reconstruct exact expected access order, including only already-arrived category probes.
    expected = []; arrived = []
    for block in p["blocks"]:
        cat = block["category"]
        if cat not in arrived: arrived.append(cat)
        expected.extend({"path": r["relative_path"], "role": "update"} for r in block["rows"])
        if block["arrival_end"] in p["boundaries"]:
            expected.extend({"path": r["relative_path"], "role": "probe"} for r in p["probes"] if r["class_name"] in arrived)
    assert expected == access
    # Recompute the registered bootstrap and effect, not an alternative endpoint.
    result = native(summarize(per_image, p))
    for arm in ARMS:
        category_means = [sum(r["losses"][arm] for r in per_image if r["boundary"] == 350 and r["category"] == c) / 20 for c in CATEGORIES]
        assert abs(sum(category_means) / 3 - result["macro"][arm]["mse"]) < 1e-14
    result["gates"]["integrity"] = True
    result["status"] = "R1_PASS_PREDICTIVE_INFO" if all(result["gates"].values()) else "R1_FAIL"
    save("summary.json", result)
    # Secondary descriptive contrasts from the same fixed endpoints and bootstrap seed.
    rng = np.random.default_rng(p["bootstrap_seed"])
    indices = rng.integers(0, 20, size=(10000, 3, 20))
    extra = {}
    for k, cat in enumerate(CATEGORIES):
        rows = [r for r in per_image if r["boundary"] == 350 and r["category"] == cat]
        d = np.array([r["losses"]["neural"] for r in rows])
        extra[cat] = {}
        for arm in ARMS[:3]:
            c = np.array([r["losses"][arm] for r in rows])
            samples = 1 - d[indices[:, k]].mean(1) / c[indices[:, k]].mean(1)
            extra[cat][arm] = {"relative_neural_gain": float(1-d.mean()/c.mean()),
                              "ci95": np.quantile(samples, [.025, .975]).tolist()}
    retention = {}
    for cat in CATEGORIES:
        seq = [(b["event"], b["categories"][cat]) for b in boundaries if cat in b["categories"]]
        retention[cat] = {arm: {"acquired_mse": seq[0][1][arm]["mse"],
                               "final_mse": seq[-1][1][arm]["mse"],
                               "final_minus_acquired": seq[-1][1][arm]["mse"] - seq[0][1][arm]["mse"],
                               "relative_deterioration": seq[-1][1][arm]["mse"] / seq[0][1][arm]["mse"] - 1}
                          for arm in ARMS}
    states = torch.load(OUT / "final_states.pt", weights_only=True, map_location="cpu")
    assert states["protocol_sha256"] == sha(OUT / "protocol_resolved.json")
    for a in ("shuffled", "neural"):
        assert {int(v["step"]) for v in states["optimizers"][a]["state"].values()} == {1750}
    serialization = {a: {"model_checkpoint_bytes": serialized_bytes(states["models"][a]),
                         "optimizer_checkpoint_bytes": serialized_bytes(states["optimizers"][a]) if a in states["optimizers"] else 0}
                     for a in ARMS}
    inference_payload = {a: tensor_bytes(states["models"][a]) for a in ARMS}
    inference_payload["ridge"] = tensor_bytes([states["models"]["ridge"][k] for k in ("w", "b")])
    causal = {}
    for a in ("shuffled", "neural"):
        rows = [r for r in trace if r["arm"] == a]
        causal[a] = {key: {"min": min(r[key] for r in rows), "median": float(np.median([r[key] for r in rows])),
                          "max": max(r[key] for r in rows)} for key in ("parameter_delta_l2", "output_delta_rms_l2")}
        causal[a]["step_count"] = len(rows)
        causal[a]["block_training"] = [{"block": b, "before": next(r["pre_loss"] for r in rows if r["block"] == b),
                                       "after": [r["post_loss"] for r in rows if r["block"] == b][-1]} for b in range(35)]
    save("forensic_analysis.json", {"per_category_neural_contrasts": extra, "retention": retention,
         "causal": causal, "serialization": serialization, "inference_tensor_payload_bytes": inference_payload,
         "derived_effects": {"neural_to_ridge_final_error_ratio": result["macro"]["neural"]["mse"] / result["macro"]["ridge"]["mse"],
             "ridge_relative_improvement_over_mean": 1 - result["macro"]["ridge"]["mse"] / result["macro"]["mean"]["mse"]},
         "checks": {"expected_access_order": True, "no_future_probes": True, "all_updates_change_readout": True,
                    "no_write": True, "state_continuity": True, "equal_optimizer_steps": True,
                    "independent_macro_recompute": True, "frozen_protocol_and_runner": True},
         "serializer_recovery": {"original_error": errors, "training_rerun": False, "feature_reads": False,
                                 "model_inference": False, "protocol_changed": False,
                                 "fix": "numpy scalar to Python scalar in analysis output only"}})
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
