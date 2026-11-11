"""Canonical parameter layout validation; not the fast/meta gradient engine."""

from __future__ import annotations

from nested_replay_cad.core.contracts import NamedTensors
from nested_replay_cad.core.errors import ParameterLayoutMismatch


def validate_detached_gradient(
    gradient: NamedTensors, shapes: dict[str, tuple[int, ...]]
) -> None:
    from nested_replay_cad.core.policy.layout import validate_parameters

    validate_parameters(gradient, shapes)
    if any(
        value.requires_grad or value.grad_fn is not None for value in gradient.values()
    ):
        raise ParameterLayoutMismatch("gradient_must_be_detached")
