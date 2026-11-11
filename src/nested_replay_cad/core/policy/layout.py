"""Explicit named state, independent of module constructors/default RNG."""

from nested_replay_cad.core.contracts import NamedTensors
from nested_replay_cad.core.errors import NonFiniteState, ParameterLayoutMismatch
from nested_replay_cad.core.runtime.tensors import reference_torch


def phi_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "hidden.weight": (16, 4),
        "hidden.bias": (16,),
        "out.weight": (1, 16),
        "out.bias": (1,),
    }


def validate_parameters(
    parameters: NamedTensors, shapes: dict[str, tuple[int, ...]]
) -> None:
    from collections.abc import Mapping

    torch = reference_torch()
    if not isinstance(parameters, Mapping) or set(parameters) != set(shapes):
        raise ParameterLayoutMismatch("parameter_keyset_mismatch")
    storages = set()
    for name, shape in shapes.items():
        value = parameters[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or value.layout != torch.strided
            or tuple(value.shape) != shape
            or not value.is_contiguous()
            or value.storage_offset() != 0
            or value.untyped_storage().nbytes() != value.numel() * 4
        ):
            raise ParameterLayoutMismatch(
                "parameter_shape_dtype_layout_mismatch:" + name
            )
        pointer = value.untyped_storage().data_ptr()
        if pointer in storages:
            raise ParameterLayoutMismatch("aliased_parameter_storage")
        storages.add(pointer)
        if not torch.isfinite(value).all().item():
            raise NonFiniteState("nonfinite_parameter:" + name)
