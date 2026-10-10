# exps/hope_outer_learning_p1_v1.py
"""CPU functional image-synchronous memory, with explicit episode ownership.

Rows represent patches: C = keys.T @ (gates * keys / N), and
D = (keys - values).T @ (gates * keys / N). All five maps right-multiply
the same pre-image transition. Only initial content and the query reader
may require gradients. Auxiliary generation is graph-free for this subset;
this restriction is not a general differentiable update-policy implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import torch
from torch.nn import functional as F


MEMORIES = ("k", "v", "eta", "alpha", "memory")
ARMS = ("RAND_FROZEN", "RAND_P1", "META_FROZEN", "META_P1",
        "STATIC_NORMAL_CONTROL", "STATIC_META")
CENTERS = tuple((row, col) for row in (3, 10, 17, 24) for col in (3, 10, 17, 24))


@dataclass(frozen=True)
class Configuration:
    dim: int
    h: float = 0.02
    normalization_eps: float = 1e-8

    def __post_init__(self):
        if self.dim < 1 or self.h != 0.02 or self.normalization_eps != 1e-8:
            raise ValueError("unsupported functional configuration")


@dataclass(frozen=True)
class Counters:
    memory: int = 0
    auxiliary: int = 0
    online: int = 0
    completed: int = 0

    def validate(self):
        if any(type(value) is not int for value in asdict(self).values()):
            raise ValueError("counters must be integer image metadata")
        if self.completed < 0 or (self.memory, self.auxiliary, self.online) != (
            self.completed, self.completed, 2 * self.completed
        ):
            raise ValueError("inconsistent image counters")

    def next_image(self):
        return Counters(self.memory + 1, self.auxiliary + 1,
                        self.online + 2, self.completed + 1)


@dataclass(frozen=True)
class SlowParameters:
    a0: torch.Tensor
    wq: torch.Tensor
    auxiliary: Mapping[str, torch.Tensor]
    conv_weight: torch.Tensor
    conv_bias: torch.Tensor
    config: Configuration

    def validate(self):
        d = self.config.dim
        _check(self.a0, (d, d), self.a0.dtype)
        _check(self.wq, (d, d), self.a0.dtype)
        if set(self.auxiliary) != {"k", "v", "eta", "alpha"}:
            raise ValueError("auxiliary schema differs")
        for name, value in self.auxiliary.items():
            _check(value, ((1 if name in ("eta", "alpha") else d), d), self.a0.dtype)
            if value.requires_grad:
                raise ValueError("auxiliary initialization must be frozen")
        _check(self.conv_weight, (d, d, 4), self.a0.dtype)
        _check(self.conv_bias, (d,), self.a0.dtype)
        if self.conv_weight.requires_grad or self.conv_bias.requires_grad:
            raise ValueError("spatial convolution must be frozen")


@dataclass(frozen=True)
class FunctionalState:
    weights: Mapping[str, torch.Tensor]
    counters: Counters = Counters()


@dataclass(frozen=True)
class Projection:
    spatial: torch.Tensor
    queries: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    gates: torch.Tensor


@dataclass(frozen=True)
class Proposal:
    weights: Mapping[str, torch.Tensor]
    counters: Counters
    projection: Projection
    C: torch.Tensor
    D: torch.Tensor
    transition: torch.Tensor
    pre_image_read: torch.Tensor
    source_identity: str
    parameter_identity: str
    input_identity: str


@dataclass(frozen=True)
class SupportResult:
    state: FunctionalState
    product: torch.Tensor
    events: tuple[Proposal, ...]


def _check(value, shape, dtype):
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.dtype not in (torch.float32, torch.float64)
            or value.dtype != dtype or tuple(value.shape) != shape):
        raise ValueError("CPU tensor geometry or precision differs")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("non-finite tensor")


def _fingerprint(values: Mapping[str, torch.Tensor], metadata=None) -> str:
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for key, value in sorted(values.items()):
        if value.device.type != "cpu":
            raise ValueError("CPU tensor required")
        plain = value.detach().contiguous()
        digest.update(key.encode())
        digest.update(str((tuple(plain.shape), plain.dtype)).encode())
        digest.update(plain.numpy().tobytes())
    return digest.hexdigest()


def state_fingerprint(state: FunctionalState) -> str:
    return _fingerprint(state.weights, asdict(state.counters))


def parameter_fingerprint(parameters: SlowParameters) -> str:
    return _fingerprint({"a0": parameters.a0, "wq": parameters.wq,
                         "conv_weight": parameters.conv_weight,
                         "conv_bias": parameters.conv_bias, **parameters.auxiliary},
                        asdict(parameters.config))


def synthetic_parameters(dim=4, *, seed=0, dtype=torch.float64,
                         requires_grad=True) -> SlowParameters:
    """Generate a private CPU fixture; no dataset or checkpoint is loaded."""
    config = Configuration(dim)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def uniform(shape, bound):
        return torch.empty(shape, dtype=dtype, device="cpu").uniform_(
            -bound, bound, generator=generator
        )

    a0 = uniform((dim, dim), (6 / (2 * dim)) ** 0.5).requires_grad_(requires_grad)
    wq = uniform((dim, dim), dim ** -0.5).requires_grad_(requires_grad)
    aux = {name: uniform((rows, dim), (6 / (rows + dim)) ** 0.5)
           for name, rows in (("k", dim), ("v", dim), ("eta", 1), ("alpha", 1))}
    result = SlowParameters(a0, wq, MappingProxyType(aux),
                            uniform((dim, dim, 4), (4 * dim) ** -0.5),
                            uniform((dim,), (4 * dim) ** -0.5), config)
    result.validate()
    return result


def validate_state(parameters: SlowParameters, state: FunctionalState):
    parameters.validate()
    state.counters.validate()
    if set(state.weights) != set(MEMORIES):
        raise ValueError("fast memory schema differs")
    d = parameters.config.dim
    for name, value in state.weights.items():
        _check(value, ((1 if name in ("eta", "alpha") else d), d), parameters.a0.dtype)
        if name != "memory" and value.requires_grad:
            raise ValueError("auxiliary fast state must be graph-free")


def initial_functional_state(parameters: SlowParameters) -> FunctionalState:
    parameters.validate()
    # clone preserves the link to the allowed content initialization.
    weights = {"memory": parameters.a0.clone(),
               **{name: value.clone() for name, value in parameters.auxiliary.items()}}
    return FunctionalState(MappingProxyType(weights))


def _spatial(parameters: SlowParameters, image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3 or image.shape[0] != 1 or image.shape[1] < 1:
        raise ValueError("one nonempty patch image is required")
    _check(image, (1, image.shape[1], parameters.config.dim), parameters.a0.dtype)
    if image.requires_grad:
        raise ValueError("input features must be frozen")
    return F.conv1d(F.pad(image.transpose(1, 2), (1, 2)),
                    parameters.conv_weight, parameters.conv_bias).transpose(1, 2)[0]


def project_from_snapshot(parameters: SlowParameters, state: FunctionalState,
                          image: torch.Tensor) -> Projection:
    validate_state(parameters, state)
    spatial = _spatial(parameters, image)
    eps = parameters.config.normalization_eps
    return Projection(spatial, F.normalize(F.linear(spatial, parameters.wq), dim=-1, eps=eps),
                      F.normalize(F.linear(spatial, state.weights["k"]), dim=-1, eps=eps),
                      F.linear(spatial, state.weights["v"]),
                      torch.sigmoid(F.linear(spatial, state.weights["eta"])))


def read_query_without_update(parameters: SlowParameters, state: FunctionalState,
                              query: torch.Tensor) -> torch.Tensor:
    validate_state(parameters, state)
    q = F.normalize(F.linear(_spatial(parameters, query), parameters.wq),
                    dim=-1, eps=parameters.config.normalization_eps)
    return F.linear(q, state.weights["memory"])


def propose_functional_event(parameters: SlowParameters, state: FunctionalState,
                             image: torch.Tensor) -> Proposal:
    validate_state(parameters, state)
    snapshot = FunctionalState(MappingProxyType({name: value.clone()
                                                for name, value in state.weights.items()}),
                               state.counters)
    projection = project_from_snapshot(parameters, snapshot, image)
    n = image.shape[1]
    weighted_keys = projection.keys * (projection.gates / n)
    # Adding zero matches the oracle's single-partition reduction order.
    zero = torch.zeros_like(snapshot.weights["k"])
    C = zero + projection.keys.T @ weighted_keys
    D = zero + (projection.keys - projection.values).T @ weighted_keys
    transition = torch.eye(parameters.config.dim, dtype=image.dtype) - parameters.config.h * (C + D)
    weights = {name: value @ transition for name, value in snapshot.weights.items()}
    return Proposal(MappingProxyType(weights), state.counters.next_image(), projection,
                    C, D, transition, F.linear(projection.queries, snapshot.weights["memory"]),
                    state_fingerprint(state), parameter_fingerprint(parameters),
                    _fingerprint({"image": image}))


def commit_functional_event(parameters: SlowParameters, state: FunctionalState,
                            proposal: Proposal) -> FunctionalState:
    """Validate every component before returning the new logical state."""
    validate_state(parameters, state)
    if proposal.source_identity != state_fingerprint(state):
        raise ValueError("stale or unrelated snapshot")
    if proposal.parameter_identity != parameter_fingerprint(parameters):
        raise ValueError("proposal uses different slow parameters")
    if proposal.counters != state.counters.next_image():
        raise ValueError("proposal changes image counters")
    candidate = FunctionalState(proposal.weights, proposal.counters)
    validate_state(parameters, candidate)
    d = parameters.config.dim
    for value in (proposal.C, proposal.D, proposal.transition):
        _check(value, (d, d), parameters.a0.dtype)
        if value.requires_grad:
            raise ValueError("transition must not depend on selected meta parameters")
    expected = torch.eye(d, dtype=parameters.a0.dtype) - parameters.config.h * (proposal.C + proposal.D)
    if not torch.equal(expected, proposal.transition):
        raise ValueError("proposal changes the transition")
    for name in MEMORIES:
        if not torch.equal(proposal.weights[name], state.weights[name] @ proposal.transition):
            raise ValueError("proposal violates shared right action")
    _check(proposal.pre_image_read, (proposal.projection.keys.shape[0], d), parameters.a0.dtype)
    # Own tensor storage without breaking the content meta-gradient.
    return FunctionalState(MappingProxyType({name: value.clone()
                                             for name, value in proposal.weights.items()}),
                           proposal.counters)


def functional_image_event(parameters: SlowParameters, state: FunctionalState,
                           image: torch.Tensor) -> tuple[FunctionalState, Proposal]:
    proposal = propose_functional_event(parameters, state, image)
    return commit_functional_event(parameters, state, proposal), proposal


def functional_support_sequence(parameters: SlowParameters,
                                support: Sequence[torch.Tensor], *,
                                state: FunctionalState | None = None) -> SupportResult:
    current = initial_functional_state(parameters) if state is None else state
    validate_state(parameters, current)
    product = torch.eye(parameters.config.dim, dtype=parameters.a0.dtype)
    events = []
    for image in support:
        current, event = functional_image_event(parameters, current, image)
        product = product @ event.transition
        events.append(event)
    return SupportResult(current, product, tuple(events))


def functional_state_to_detached_snapshot(state: FunctionalState) -> FunctionalState:
    state.counters.validate()
    return FunctionalState(MappingProxyType({name: value.detach().clone()
                                             for name, value in state.weights.items()}),
                           state.counters)


def detached_parameters(parameters: SlowParameters) -> SlowParameters:
    return SlowParameters(parameters.a0.detach().clone(), parameters.wq.detach().clone(),
                          MappingProxyType({name: value.detach().clone()
                                            for name, value in parameters.auxiliary.items()}),
                          parameters.conv_weight.detach().clone(),
                          parameters.conv_bias.detach().clone(), parameters.config)


def tensor_inventory(state: FunctionalState) -> dict:
    return {"tensor_bytes": sum(value.numel() * value.element_size() for value in state.weights.values()) + 32,
            "key_count": len(state.weights) + 4,
            "schema": {name: [list(value.shape), str(value.dtype)] for name, value in state.weights.items()},
            "counters": asdict(state.counters),
            "graph_bearing": any(value.requires_grad or value.grad_fn is not None
                                 for value in state.weights.values())}


def checkpoint_payload(parameters: SlowParameters, state: FunctionalState,
                       *, input_identities: Sequence[str], order: Sequence[int]) -> dict:
    validate_state(parameters, state)
    identities, order = tuple(input_identities), tuple(order)
    if len(set(identities)) != len(identities) or sorted(order) != list(range(len(identities))):
        raise ValueError("input identities or order are inconsistent")
    if state.counters.completed > len(order):
        raise ValueError("checkpoint has more events than input identities")
    plain = detached_parameters(parameters)
    snapshot = functional_state_to_detached_snapshot(state)
    return {"schema": "functional_p1_cpu_v1", "config": asdict(parameters.config),
            "parameters": {"a0": plain.a0, "wq": plain.wq,
                           "auxiliary": dict(plain.auxiliary),
                           "conv_weight": plain.conv_weight, "conv_bias": plain.conv_bias},
            "weights": dict(snapshot.weights), "counters": asdict(snapshot.counters),
            "parameter_hash": parameter_fingerprint(parameters),
            "state_hash": state_fingerprint(snapshot), "input_identities": identities,
            "order": order, "cpu_rng_state": torch.get_rng_state().clone()}


def restore_checkpoint(payload: Mapping, *, expected_identities: Sequence[str],
                       expected_order: Sequence[int], requires_grad=False,
                       restore_rng=False) -> tuple[SlowParameters, FunctionalState]:
    if payload["schema"] != "functional_p1_cpu_v1":
        raise ValueError("checkpoint schema differs")
    if tuple(payload["input_identities"]) != tuple(expected_identities) or tuple(payload["order"]) != tuple(expected_order):
        raise ValueError("checkpoint input identities differ")
    values = payload["parameters"]
    parameters = SlowParameters(values["a0"].detach().clone().requires_grad_(requires_grad),
                                values["wq"].detach().clone().requires_grad_(requires_grad),
                                MappingProxyType({name: value.detach().clone()
                                                  for name, value in values["auxiliary"].items()}),
                                values["conv_weight"].detach().clone(),
                                values["conv_bias"].detach().clone(), Configuration(**payload["config"]))
    state = FunctionalState(MappingProxyType({name: value.detach().clone()
                                             for name, value in payload["weights"].items()}),
                            Counters(**payload["counters"]))
    validate_state(parameters, state)
    identities, order = tuple(payload["input_identities"]), tuple(payload["order"])
    if (len(set(identities)) != len(identities)
            or sorted(order) != list(range(len(identities)))
            or state.counters.completed > len(order)):
        raise ValueError("checkpoint input identities or event count are inconsistent")
    if parameter_fingerprint(parameters) != payload["parameter_hash"] or state_fingerprint(state) != payload["state_hash"]:
        raise ValueError("checkpoint tensor integrity differs")
    # Continuation is detached; a new gradient episode must reset to A0.
    if restore_rng:
        torch.set_rng_state(payload["cpu_rng_state"].clone())
    return parameters, state


def save_checkpoint(path: Path, payload: Mapping):
    torch.save(dict(payload), path)


def load_checkpoint(path: Path, **kwargs):
    return restore_checkpoint(torch.load(path, map_location="cpu", weights_only=True), **kwargs)


def read_with_transition(parameters: SlowParameters, query: torch.Tensor,
                         transition: torch.Tensor) -> torch.Tensor:
    _check(transition, (parameters.config.dim, parameters.config.dim), parameters.a0.dtype)
    state = initial_functional_state(parameters)
    weights = {**state.weights, "memory": parameters.a0 @ transition}
    return read_query_without_update(parameters, FunctionalState(MappingProxyType(weights)), query)


def evaluate_arm(arm: str, parameters: SlowParameters, query: torch.Tensor, *,
                 adapted_state: FunctionalState | None = None,
                 support_transition: torch.Tensor | None = None,
                 static_parameters: SlowParameters | None = None,
                 static_predictor: Callable[[torch.Tensor], torch.Tensor] | None = None) -> torch.Tensor:
    """Pure prediction interface; it neither fits parameters nor writes memory."""
    if arm not in ARMS:
        raise ValueError("unsupported comparison arm")
    if arm == "STATIC_NORMAL_CONTROL":
        if static_predictor is None:
            raise ValueError("an explicit frozen predictor is required")
        return static_predictor(query)
    if arm == "STATIC_META":
        if static_parameters is None:
            raise ValueError("independent static parameters are required")
        return read_query_without_update(static_parameters, initial_functional_state(static_parameters), query)
    if arm.endswith("FROZEN"):
        return read_query_without_update(parameters, initial_functional_state(parameters), query)
    if adapted_state is not None and support_transition is None:
        return read_query_without_update(parameters, adapted_state, query)
    if adapted_state is None and support_transition is not None:
        return read_with_transition(parameters, query, support_transition)
    raise ValueError("provide exactly one adapted state or support transition")


def masked_synthetic_rgb(clean_normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One shared sixteen-hole view; input and fill use normalized RGB units."""
    if clean_normalized.ndim != 4 or clean_normalized.shape[1:] != (3, 224, 224):
        raise ValueError("normalized RGB geometry must be 224 by 224")
    if clean_normalized.device.type != "cpu" or not bool(torch.isfinite(clean_normalized).all()):
        raise ValueError("finite CPU RGB tensor required")
    mask = torch.zeros((224, 224), dtype=torch.bool)
    for row, col in CENTERS:
        mask[(row - 1) * 8:(row + 2) * 8, (col - 1) * 8:(col + 2) * 8] = True
    return torch.where(mask[None, None], torch.zeros_like(clean_normalized), clean_normalized), mask


def mock_patch_backbone(normalized_rgb: torch.Tensor) -> torch.Tensor:
    """Independent patch means for coordinate mechanics, without attention."""
    if normalized_rgb.ndim != 4 or normalized_rgb.shape[1:] != (3, 224, 224):
        raise ValueError("mock input geometry differs")
    return F.avg_pool2d(normalized_rgb, 8, 8).flatten(2).transpose(1, 2)
