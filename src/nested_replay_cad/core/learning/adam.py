"""Functional reference slow Adam: zero-decay, bias-corrected FP32 proposals."""

from __future__ import annotations

from types import MappingProxyType

from nested_replay_cad.core.contracts import AdamState, NamedTensors, PhiState
from nested_replay_cad.core.errors import (
    NonFiniteState,
    ParameterLayoutMismatch,
    SchemaError,
)
from nested_replay_cad.core.learning.parameters import validate_detached_gradient
from nested_replay_cad.core.policy.layout import phi_shapes, validate_parameters
from nested_replay_cad.core.runtime.tensors import reference_torch


def adam_propose(
    phi: PhiState, adam: AdamState, gradient: NamedTensors
) -> tuple[PhiState, AdamState]:
    validate_adam_state(phi, adam)
    torch = reference_torch()
    shapes = phi_shapes()
    validate_detached_gradient(gradient, shapes)
    step = adam.step + 1
    values = {}
    first = {}
    second = {}
    with torch.no_grad():
        for name in sorted(shapes):
            g = gradient[name]
            m = 0.9 * adam.m[name] + (1 - 0.9) * g
            v = 0.999 * adam.v[name] + (1 - 0.999) * (g * g)
            value = phi.parameters[name].detach() - 0.001 * (m / (1 - 0.9**step)) / (
                (v / (1 - 0.999**step)).sqrt() + 1e-8
            )
            if not all(torch.isfinite(x).all().item() for x in (m, v, value)):
                raise NonFiniteState("nonfinite_Adam_candidate")
            first[name] = m.detach()
            second[name] = v.detach()
            values[name] = value.detach()
    return PhiState(MappingProxyType(values)), AdamState(
        MappingProxyType(first), MappingProxyType(second), step
    )


def validate_adam_state(phi: PhiState, adam: AdamState) -> None:
    """Validate slow state before expensive query work, without making a step."""
    torch = reference_torch()
    shapes = phi_shapes()
    if not isinstance(phi, PhiState) or not isinstance(adam, AdamState):
        raise SchemaError("invalid_slow_state")
    validate_parameters(phi.parameters, shapes)
    validate_detached_gradient(adam.m, shapes)
    validate_detached_gradient(adam.v, shapes)
    if type(adam.step) is not int or adam.step < 0:
        raise SchemaError("invalid_Adam_step")
    storages = [
        v.untyped_storage().data_ptr()
        for group in (phi.parameters, adam.m, adam.v)
        for v in group.values()
    ]
    if len(set(storages)) != len(storages):
        raise ParameterLayoutMismatch("aliased_phi_or_Adam_moments")
    if any((v < 0).any().item() for v in adam.v.values()):
        raise SchemaError("negative_Adam_second_moment")
    if adam.step == 0 and any(
        v.any().item() for group in (adam.m, adam.v) for v in group.values()
    ):
        raise SchemaError("nonzero_Adam_moments_at_step0")


def adam_initial(phi: PhiState) -> AdamState:
    torch = reference_torch()
    if not isinstance(phi, PhiState):
        raise SchemaError("invalid_initial_phi")
    validate_parameters(phi.parameters, phi_shapes())
    return AdamState(
        MappingProxyType({n: torch.zeros_like(v) for n, v in phi.parameters.items()}),
        MappingProxyType({n: torch.zeros_like(v) for n, v in phi.parameters.items()}),
        0,
    )
