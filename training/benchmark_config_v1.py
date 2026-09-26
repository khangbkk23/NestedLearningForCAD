"""Safe config resolution and explicit protocol compatibility checks."""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
import yaml
from dataset.benchmark_protocol_v1 import CANONICAL_MVTEC_TASKS


def expand_environment(value):
    if isinstance(value, dict):
        return {k: expand_environment(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_environment(v) for v in value]
    if isinstance(value, str):
        for name in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value):
            if name not in os.environ:
                raise ValueError(f"Missing required environment variable: {name}")
            value = value.replace("${" + name + "}", os.environ[name])
    return value


def apply_overrides(protocol, method, overrides):
    protocol, method = copy.deepcopy(protocol), copy.deepcopy(method)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError("Override must be key=value")
        value = yaml.safe_load(raw)
        if not (value is None or type(value) in (str, int, float, bool)):
            raise ValueError("Override values must be primitives")
        keys = key.split(".")
        if keys[0] in {"protocol", "method"}:
            target = protocol if keys.pop(0) == "protocol" else method
        else:
            target = method
        for part in keys[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown override path: {key}")
            target = target[part]
        if not keys or keys[-1] not in target:
            raise ValueError(f"Unknown override key: {key}")
        target[keys[-1]] = value
    return expand_environment(protocol), expand_environment(method)


def load_configs(protocol_path, method_path, overrides=()):
    protocol = yaml.safe_load(Path(protocol_path).read_text())
    method = yaml.safe_load(Path(method_path).read_text())
    if not isinstance(protocol, dict) or not isinstance(method, dict):
        raise ValueError("Configs must be mappings")
    return apply_overrides(protocol, method, overrides)


def validate_configs(protocol, method, smoke=False):
    for config in (protocol, method):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", config["id"]):
            raise ValueError("Unsafe config id")
        if config["version"] != 1:
            raise ValueError("Unsupported config version")
    if protocol["dataset"]["name"] != "MVTec AD":
        raise ValueError("Only MVTec AD is supported in v1")
    if any(k in protocol for k in ("extractor", "backbone", "memory", "phase3", "replay")):
        raise ValueError("Method values cannot be placed in protocol config")
    if (protocol["training"] != {"normal_only": True, "dev_mode": "disabled", "drop_last": False}
            or protocol["leakage"]["official_test_feedback"] != "forbidden"):
        raise ValueError("Training/leakage rules are immutable in this harness")
    evaluation = protocol["evaluation"]
    if (evaluation["final_per_task"] is not True or evaluation["aggregation"] != "macro_tasks"
            or evaluation["pixels"] != "all" or evaluation["primary"] != ["i_auroc", "p_aupr"]
            or evaluation["pooled"] != "secondary_only" or protocol["forgetting"]["enabled"] is not True
            or protocol["forgetting"]["formula"] != "mean_prior_max_minus_final"):
        raise ValueError("Unsupported evaluation protocol")
    tasks = protocol["task_order"]
    if (not tasks or len(tasks) != len(set(tasks)) or any(t not in CANONICAL_MVTEC_TASKS for t in tasks)
            or (not smoke and tasks != CANONICAL_MVTEC_TASKS)):
        raise ValueError("Reportable protocol requires all 15 canonical tasks in alphabetical order")
    if method["adapter"] not in {"cadic", "metanath_legacy", "fake"}:
        raise ValueError("Unknown adapter")
    if method["adapter"] == "fake" and not smoke:
        raise ValueError("Fake adapter is structural smoke only")
    if method.get("exact_parity_claim", False):
        raise ValueError("No verified CADIC exact-parity claim is supported")
    pre = method["preprocessing"]
    if (pre["resize"] != "direct_square_bilinear_pil" or len(pre["mean"]) != 3
            or len(pre["std"]) != 3 or any(s <= 0 for s in pre["std"])
            or type(pre["image_size"]) is not int or pre["image_size"] < 1):
        raise ValueError("Invalid preprocessing")
    if type(method["runtime"]["batch_size"]) is not int or method["runtime"]["batch_size"] < 1:
        raise ValueError("Invalid batch size")
    if method["runtime"]["dtype"] != "float32":
        raise ValueError("v1 requires float32 runtime")
    if method["adapter"] == "cadic":
        extractor, memory = method["extractor"], method["memory"]
        if (pre["image_size"] != 224 or extractor["model_name"] != "vit_base_patch8_224"
                or extractor["pretraining"] != "ImageNet-21k" or extractor["block_index"] != 8
                or extractor["layer_number"] != 9 or extractor["layer_indexing"] != "one_based"
                or extractor["feature_normalization"] != "none" or memory["distance"] != "euclidean"
                or type(memory["budget"]) is not int or memory["budget"] < 2
                or type(memory["chunk_size"]) is not int or memory["chunk_size"] < 1
                or type(method["scoring"]["image_neighbors_b"]) is not int
                or method["scoring"]["image_neighbors_b"] < 2):
            raise ValueError("Invalid CADIC-compatible declaration")
        if not smoke and memory["budget"] not in memory.get("allowed_budgets", [2500, 5000, 10000]):
            raise ValueError("CADIC-compatible budget must be one of the declared ladder values")
        if not method.get("unresolved_assumptions"):
            raise ValueError("CADIC compatibility assumptions must be declared")
    if not Path(protocol["dataset"]["root"]).is_dir():
        raise FileNotFoundError(protocol["dataset"]["root"])
