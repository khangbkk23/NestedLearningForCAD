"""Lazy CPU tensor contracts; import alone never loads torch or discovers CUDA."""

from __future__ import annotations

import platform
from types import ModuleType

from nested_replay_cad.core.errors import ScopeBlocked


def reference_torch() -> ModuleType:
    import torch

    if platform.python_version() != "3.14.4" or torch.__version__ != "2.12.0+cu126":
        raise ScopeBlocked("tensor_component_requires_reference_python_and_torch")
    return torch
