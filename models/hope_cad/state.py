# models/hope_cad/state.py
"""Small, generic utilities for persistent HOPE tensor state.

This module deliberately knows nothing about Self-Modifying Titans, CMS, or
anomaly detection. It provides value-style snapshots for tests and future
online-memory modules while leaving ordinary nn.Module parameters and transient
forward tensors under their owners' control.
"""
from collections.abc import Mapping, MutableMapping
from typing import Any
import torch

StateTree = Any

def snapshot_persistent_state(state: StateTree) -> StateTree:
    """Return an independent tensor snapshot of a nested state tree.

    Mappings, lists, and tuples are traversed recursively. Tensor leaves are
    cloned, preserving dtype, device, and requires_grad metadata. Scalar
    metadata is retained by value/reference because it is not mutable tensor
    state. The result is suitable for comparison or deterministic restore.
    """

    if isinstance(state, torch.Tensor):
        return state.detach().clone().requires_grad_(state.requires_grad)
    if isinstance(state, Mapping):
        return type(state)((key, snapshot_persistent_state(value)) for key, value in state.items())
    if isinstance(state, list):
        return [snapshot_persistent_state(value) for value in state]
    if isinstance(state, tuple):
        return tuple(snapshot_persistent_state(value) for value in state)
    return state

def persistent_states_equal(left: StateTree, right: StateTree,*, rtol: float = 0.0, atol: float = 0.0) -> bool:
    """Compare two state trees, optionally with an explicit tensor tolerance."""

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
            return False
        if rtol == 0.0 and atol == 0.0:
            return torch.equal(left, right)
        return bool(torch.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False))

    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if list(left.keys()) != list(right.keys()):
            return False
        return all(
            persistent_states_equal(left[key], right[key], rtol=rtol, atol=atol)
            for key in left
        )

    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(
            persistent_states_equal(a, b, rtol=rtol, atol=atol)
            for a, b in zip(left, right)
        )

    try:
        result = left == right
    except Exception:
        return False
    return bool(result) if isinstance(result, (bool, torch.Tensor)) else False


def persistent_tensor_bytes(state: StateTree) -> int:
    """Return logical payload bytes for all tensor leaves in a state tree."""

    if isinstance(state, torch.Tensor):
        return int(state.numel() * state.element_size())
    if isinstance(state, Mapping):
        return sum(persistent_tensor_bytes(value) for value in state.values())
    if isinstance(state, (list, tuple)):
        return sum(persistent_tensor_bytes(value) for value in state)
    return 0


def validate_finite_state(state: StateTree) -> bool:
    """Validate that every tensor leaf contains only finite values.

    Returns True for a valid state and raises ValueError with a useful path for
    the first non-finite tensor. Empty/non-tensor metadata is valid.
    """

    def visit(value: StateTree, path: str) -> None:
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all().item():
                raise ValueError(f"non-finite persistent state tensor at {path}")
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(state, "state")
    return True


def restore_persistent_state(target: StateTree, snapshot: StateTree) -> None:
    """Copy a snapshot into an existing mutable tensor state tree in place.

    Structure and tensor leaves must match. Future nn.Module owners can use
    native load_state_dict for module buffers/parameters; this helper is for
    nested online-state containers that need deterministic reset in tests.
    """

    if isinstance(target, torch.Tensor) or isinstance(snapshot, torch.Tensor):
        if not isinstance(target, torch.Tensor) or not isinstance(snapshot, torch.Tensor):
            raise TypeError("target and snapshot tensor structures differ")
        if target.shape != snapshot.shape or target.dtype != snapshot.dtype:
            raise ValueError("target and snapshot tensor metadata differ")
        target.copy_(snapshot.to(device=target.device))
        return

    if isinstance(target, MutableMapping) or isinstance(snapshot, Mapping):
        if not isinstance(target, MutableMapping) or not isinstance(snapshot, Mapping):
            raise TypeError("target and snapshot mapping structures differ")
        if list(target.keys()) != list(snapshot.keys()):
            raise ValueError("target and snapshot mapping keys differ")
        for key in target:
            restore_persistent_state(target[key], snapshot[key])
        return

    if isinstance(target, list) or isinstance(snapshot, list):
        if not isinstance(target, list) or not isinstance(snapshot, list) or len(target) != len(snapshot):
            raise ValueError("target and snapshot list structures differ")
        for current, saved in zip(target, snapshot):
            restore_persistent_state(current, saved)
        return

    if isinstance(target, tuple) or isinstance(snapshot, tuple):
        if not isinstance(target, tuple) or not isinstance(snapshot, tuple) or len(target) != len(snapshot):
            raise ValueError("target and snapshot tuple structures differ")
        for current, saved in zip(target, snapshot):
            restore_persistent_state(current, saved)
        return

    if target != snapshot:
        raise ValueError("target and snapshot metadata differ")