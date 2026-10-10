# exps/hope_outer_learning_pilot_v1.py
"""Device-neutral functional memory and bottle-only calibration contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
from torch.nn import functional as F

from exps.ol03.hope_outer_learning_p1_v1 import CENTERS, Counters, MEMORIES


ROLES = (("meta_train_support", 0, 60), ("meta_train_query", 60, 80),
         ("meta_val_support", 80, 92), ("meta_val_query", 92, 100),
         ("online_train", 100, 150), ("normal_probe", 150, 170))
CENTER_IDS = tuple(r * 28 + c for r, c in CENTERS)
EXPECTED_CACHE_SHA = "25c17dd008468a3746564cd75b01f0cc1507ff518014c59dc73be2491a39d44b"
EXPECTED_CHECKPOINT_SHA = "ae3012808a9b406a19b799381bd26b253634ad26125937c26d02dbcbbc85dd92"
EXPECTED_INIT_HASH = "bc8e66aac3e672a484b9d1331157f8c87e3cb08b346908498d13e15e1e024ae6"


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(tensor):
    plain = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(plain.shape), plain.dtype)).encode() + plain.numpy().tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)
    if json.loads(path.read_text()) != value:
        raise ValueError("JSON round-trip differs")


def atomic_torch(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


@dataclass(frozen=True)
class Parameters:
    a0: torch.Tensor
    wq: torch.Tensor
    auxiliary: dict
    conv_weight: torch.Tensor
    conv_bias: torch.Tensor

    @property
    def dim(self):
        return self.a0.shape[0]

    def validate(self):
        d = self.dim
        for name, value in {"a0": self.a0, "wq": self.wq, **self.auxiliary}.items():
            rows = 1 if name in ("eta", "alpha") else d
            if value.shape != (rows, d) or value.device != self.a0.device or value.dtype != self.a0.dtype:
                raise ValueError("parameter geometry differs")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("non-finite parameter")
        if set(self.auxiliary) != {"k", "v", "eta", "alpha"}:
            raise ValueError("auxiliary schema differs")
        if self.conv_weight.shape != (d, d, 4) or self.conv_bias.shape != (d,):
            raise ValueError("convolution geometry differs")
        for value in (*self.auxiliary.values(), self.conv_weight, self.conv_bias):
            if value.requires_grad or value.grad_fn is not None or not bool(torch.isfinite(value).all()):
                raise ValueError("frozen parameters carry a graph or invalid value")


@dataclass(frozen=True)
class State:
    weights: dict
    counters: Counters = Counters()


@dataclass(frozen=True)
class Proposal:
    source: State
    weights: dict
    counters: Counters
    spatial: torch.Tensor
    queries: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    gates: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor
    transition: torch.Tensor
    pre_read: torch.Tensor


def parameters_from_fixture(fixture, device="cpu", *, trainable=False):
    source = fixture["smt"]
    from exps.hope_image_synchronous_memory import fingerprint
    if fixture["initialization_hash"] != EXPECTED_INIT_HASH or fingerprint(source) != EXPECTED_INIT_HASH:
        raise ValueError("original fixture identity differs")
    for name in MEMORIES:
        if not torch.equal(source[f"memories.{name}.weight"], source[f"memories.{name}.initial_weight"]):
            raise ValueError("fixture contains previously adapted state")
    if any(int(source[name]) != 0 for name in ("memory_update_count", "auxiliary_update_count", "online_update_count")):
        raise ValueError("fixture counters are not initial")
    p = Parameters(source["memories.memory.initial_weight"].to(device).detach().clone().requires_grad_(trainable),
                   source["base_q.weight"].to(device).detach().clone().requires_grad_(trainable),
                   {name: source[f"memories.{name}.initial_weight"].to(device).detach().clone()
                    for name in ("k", "v", "eta", "alpha")},
                   source["local_conv.weight"].to(device).detach().clone(),
                   source["local_conv.bias"].to(device).detach().clone())
    p.validate()
    return p


def copy_parameters(p, *, trainable=False, device=None):
    device = p.a0.device if device is None else device
    return Parameters(p.a0.detach().to(device).clone().requires_grad_(trainable),
                      p.wq.detach().to(device).clone().requires_grad_(trainable),
                      {name: v.detach().to(device).clone() for name, v in p.auxiliary.items()},
                      p.conv_weight.detach().to(device).clone(), p.conv_bias.detach().to(device).clone())


def initial_state(p):
    p.validate()
    return State(MappingProxyType({"memory": p.a0.clone(), **{k: v.clone() for k, v in p.auxiliary.items()}}))


def validate_state(p, state):
    state.counters.validate()
    if set(state.weights) != set(MEMORIES):
        raise ValueError("state schema differs")
    for name, value in state.weights.items():
        if value.shape != (1 if name in ("eta", "alpha") else p.dim, p.dim):
            raise ValueError("state geometry differs")
        if value.device != p.a0.device or value.dtype != p.a0.dtype or not bool(torch.isfinite(value).all()):
            raise ValueError("invalid state")
        if name != "memory" and (value.requires_grad or value.grad_fn is not None):
            raise ValueError("auxiliary state must be graph-free")


def spatial(p, image):
    if image.ndim != 3 or image.shape[0] != 1 or image.shape[1] < 1 or image.shape[2] != p.dim:
        raise ValueError("one patch image required")
    if image.requires_grad or image.device != p.a0.device or image.dtype != p.a0.dtype:
        raise ValueError("frozen input precision or placement differs")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("non-finite input")
    return F.conv1d(F.pad(image.transpose(1, 2), (1, 2)), p.conv_weight, p.conv_bias).transpose(1, 2)[0]


def query_rows(p, image):
    return F.normalize(F.linear(spatial(p, image), p.wq), dim=-1, eps=1e-8)


def read(p, state, image):
    validate_state(p, state)
    return F.linear(query_rows(p, image), state.weights["memory"])


def propose(p, state, image):
    validate_state(p, state)
    snapshot = {name: value.clone() for name, value in state.weights.items()}
    s = spatial(p, image)
    q = F.normalize(F.linear(s, p.wq), dim=-1, eps=1e-8)
    k = F.normalize(F.linear(s, snapshot["k"]), dim=-1, eps=1e-8)
    v = F.linear(s, snapshot["v"])
    g = torch.sigmoid(F.linear(s, snapshot["eta"]))
    wk = k * (g / len(k))
    zero = torch.zeros_like(snapshot["k"])
    C = zero + k.T @ wk
    D = zero + (k - v).T @ wk
    T = torch.eye(p.dim, device=s.device, dtype=s.dtype) - 0.02 * (C + D)
    weights = MappingProxyType({name: value @ T for name, value in snapshot.items()})
    return Proposal(state, weights, state.counters.next_image(), s, q, k, v, g, C, D, T,
                    F.linear(q, snapshot["memory"]))


def commit(p, state, proposal):
    if proposal.source is not state or proposal.counters != state.counters.next_image():
        raise ValueError("stale proposal or inconsistent event")
    next_state = State(proposal.weights, proposal.counters)
    validate_state(p, next_state)
    if proposal.transition.requires_grad or proposal.transition.grad_fn is not None:
        raise ValueError("transition depends on unauthorized trainable parameters")
    for name in MEMORIES:
        if not torch.equal(proposal.weights[name], state.weights[name] @ proposal.transition):
            raise ValueError("shared transition differs")
    return State(MappingProxyType({name: value.clone() for name, value in proposal.weights.items()}),
                 proposal.counters)


def event(p, state, image):
    proposal = propose(p, state, image)
    return commit(p, state, proposal), proposal


def support_sequence(p, images):
    state = initial_state(p)
    product = torch.eye(p.dim, device=p.a0.device, dtype=p.a0.dtype)
    for image in images:
        state, proposal = event(p, state, image)
        product = product @ proposal.transition
    return state, product


def detached_state(state):
    return State(MappingProxyType({k: v.detach().clone() for k, v in state.weights.items()}), state.counters)


def state_fingerprint(state):
    return hashlib.sha256(json.dumps({"tensors": {k: tensor_sha(v) for k, v in state.weights.items()},
                                     "counters": asdict(state.counters)}, sort_keys=True).encode()).hexdigest()


def predict_with_product(p, image, product):
    # The content right-action remains differentiable; no auxiliary policy is learned.
    return F.linear(query_rows(p, image), p.a0 @ product)


def teacher_error(prediction, target, centers=CENTER_IDS):
    if target.requires_grad or target.grad_fn is not None:
        raise ValueError("teacher must be immutable")
    return 0.5 * (prediction[list(centers)] - target[list(centers)]).square().sum(-1).mean()


def make_schedule(seed=0):
    generator = torch.Generator().manual_seed(seed)
    episodes = [{"step": step + 1, "support": torch.randperm(60, generator=generator)[:4].tolist(),
                 "query": 60 + int(torch.randint(20, (1,), generator=generator))} for step in range(200)]
    val = [{"query": q, "support": (torch.randperm(12, generator=generator)[:4] + 80).tolist(),
            "distinct_support": torch.randperm(60, generator=generator)[:4].tolist()} for q in range(92, 100)]
    train_panel = [{"query": q, "support": torch.randperm(60, generator=generator)[:4].tolist()} for q in range(60, 80)]
    wrong = torch.randperm(60, generator=generator)[:50].tolist()
    pool50 = [torch.randperm(60, generator=generator)[:50].tolist() for _ in range(8)]
    pool4 = [list(range(start, start + 4)) for start in range(0, 60, 4)]
    return {"seed": seed, "episodes": episodes, "validation": val, "train_panel": train_panel,
            "online": list(range(100, 150)), "wrong_history50": wrong,
            "pooled50": pool50, "pooled4": pool4}


def schedule_sha(schedule):
    return hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()


def validate_bottle_manifest(data_root, payload, cache_sha):
    if cache_sha != EXPECTED_CACHE_SHA or payload["class_name"] != "bottle":
        raise ValueError("bottle cache identity differs")
    expected = [f"bottle/train/good/{i:03}.png" for i in range(170)]
    if payload["relative_paths"] != expected or payload["patches"].shape != (170, 784, 768):
        raise ValueError("cache path order or geometry differs")
    if payload["patches"].dtype != torch.float32 or not bool(torch.isfinite(payload["patches"]).all()):
        raise ValueError("invalid cached features")
    meta = payload["metadata"]
    if (meta["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA or meta["layer_number"] != 9
            or meta["block_index"] != 8 or meta["feature_normalization"] != "none" or not meta["cls_removed"]):
        raise ValueError("feature identity differs")
    root = Path(data_root).resolve()
    allowed = root / "bottle/train/good"
    rows = []
    for role, start, end in ROLES:
        for i in range(start, end):
            path = root / expected[i]
            if not path.is_file() or path.resolve().parent != allowed or path.is_symlink():
                raise ValueError("approved normal image path missing or redirected")
            rows.append({"identity": i, "relative_path": expected[i], "raw_path": str(path),
                         "raw_sha256": file_sha(path), "cache_index": i,
                         "feature_sha256": tensor_sha(payload["patches"][i]), "role": role,
                         "source_cache_sha256": cache_sha,
                         "first_accessible_stage": "calibration" if i < 100 else "online" if i < 150 else "read_only_probe"})
    if len({row["raw_sha256"] for row in rows}) != 170:
        raise ValueError("duplicate raw image content across approved roles")
    return rows


def mask_rgb(normalized):
    if normalized.shape != (1, 3, 224, 224):
        raise ValueError("RGB view geometry differs")
    mask = torch.zeros((224, 224), dtype=torch.bool, device=normalized.device)
    for r, c in CENTERS:
        mask[(r - 1)*8:(r + 2)*8, (c - 1)*8:(c + 2)*8] = True
    return normalized.masked_fill(mask[None, None], 0), mask


def nontriviality(clean, masked, train_targets):
    teacher = F.normalize(clean, dim=-1, eps=1e-8)[:, list(CENTER_IDS)]
    student = F.normalize(masked, dim=-1, eps=1e-8)[:, list(CENTER_IDS)]
    cosine = (teacher * student).sum(-1)
    center_mean = teacher.mean(0, keepdim=True)
    variance = (teacher - center_mean).square().sum(-1).mean()
    fixed_mean = train_targets[:, list(CENTER_IDS)].mean(0)
    fractions = float((cosine >= 0.999).float().mean())
    result = {"identical_center_fraction": fractions,
              "target_mean_squared_distance": float(variance),
              "teacher_mean_validation_error": float(0.5 * (teacher - fixed_mean).square().sum(-1).mean()),
              "teacher_mean_fit_identities": list(range(60, 80)),
              "cosine_min": float(cosine.min()), "cosine_median": float(cosine.median()),
              "cosine_max": float(cosine.max()), "normalized_gap_mean": float((teacher-student).norm(dim=-1).mean()),
              "per_center_target_squared_distance": ((teacher-center_mean).square().sum(-1).mean(0)).tolist(),
              "masked_student_centered_variance": float((student-student.mean(0)).square().sum(-1).mean()),
              "passed": fractions <= 0.95 and float(variance) >= 1e-4}
    return result


def ridge_fit(rows, targets):
    H, Y = rows.double(), targets.double()
    hm, ym = H.mean(0), Y.mean(0)
    hc, yc = H-hm, Y-ym
    gram = hc.T @ hc
    penalty = max(1e-12, 0.001 * float(gram.trace()) / H.shape[-1])
    coefficient = torch.linalg.solve(gram + penalty * torch.eye(H.shape[-1], device=H.device, dtype=H.dtype), hc.T @ yc)
    return coefficient.to(rows.dtype), hm.to(rows.dtype), ym.to(rows.dtype), penalty


def ridge_read(rows, fitted):
    coefficient, hm, ym, _ = fitted
    return (rows-hm) @ coefficient + ym


def paired_bootstrap(a, b, *, seed=0, repeats=2000):
    """Utility delta from paired image errors; patches are never resampled."""
    delta = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    if len(delta) < 2 or not np.isfinite(delta).all():
        raise ValueError("finite paired image errors required")
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(len(delta), size=(repeats, len(delta)))].mean(1)
    return {"estimate": float(delta.mean()), "ci_low": float(np.quantile(means, .025)),
            "ci_high": float(np.quantile(means, .975)), "n_images": len(delta), "repeats": repeats,
            "positive_images": int((delta > 0).sum())}


def serialize_parameters(p):
    return {"a0":p.a0.detach().cpu().clone(), "wq":p.wq.detach().cpu().clone(),
            "auxiliary": {k:v.detach().cpu().clone() for k,v in p.auxiliary.items()},
            "conv_weight":p.conv_weight.detach().cpu().clone(), "conv_bias":p.conv_bias.detach().cpu().clone()}


def restore_parameters(payload, device="cpu", *, trainable=False):
    return Parameters(payload["a0"].to(device).detach().clone().requires_grad_(trainable),
                      payload["wq"].to(device).detach().clone().requires_grad_(trainable),
                      {k:v.to(device).detach().clone() for k,v in payload["auxiliary"].items()},
                      payload["conv_weight"].to(device).detach().clone(), payload["conv_bias"].to(device).detach().clone())


def parameters_sha(p):
    values = serialize_parameters(p)
    hashes = {k:tensor_sha(v) for k,v in values.items() if isinstance(v, torch.Tensor)}
    hashes.update({k:tensor_sha(v) for k,v in values["auxiliary"].items()})
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def inventory(p, state):
    fast = sum(v.numel()*v.element_size() for v in state.weights.values()) + 32
    reset = sum(v.numel()*v.element_size() for v in (p.a0, *p.auxiliary.values()))
    upstream = sum(v.numel()*v.element_size() for v in (p.wq, p.conv_weight, p.conv_bias))
    return {"fast_bytes":fast,"reset_bytes":reset,"fixed_upstream_bytes":upstream,
            "total_deployed_bytes":fast+reset+upstream,
            "key_count":len(state.weights)+4,
            "graph_bearing":any(v.requires_grad or v.grad_fn is not None for v in state.weights.values()),
            "counters":asdict(state.counters)}


def parameter_storage_bytes(p):
    """Count every retained reset/upstream tensor, including unused auxiliaries."""
    return sum(v.numel()*v.element_size() for v in
               (p.a0,p.wq,p.conv_weight,p.conv_bias,*p.auxiliary.values()))
