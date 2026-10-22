"""Strict non-mutating checkpoint helpers; booleans are not numeric clocks."""
import math
import torch


def keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f'{label}: exact keys required')


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label}: invalid integer')
    return value


def scalar(value, label, minimum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{label}: finite scalar required')
    if minimum is not None and value < minimum:
        raise ValueError(f'{label}: below lower bound')
    return float(value)


def tensor(value, shape, dtype, label):
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape) or value.dtype != dtype:
        raise ValueError(f'{label}: tensor shape/dtype changed')
    if value.requires_grad or not bool(torch.isfinite(value).all()):
        raise ValueError(f'{label}: checkpoint must be detached and finite')
    return value


def tree_finite(value):
    if torch.is_tensor(value):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite checkpoint tensor')
    elif isinstance(value, dict):
        for item in value.values():
            tree_finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            tree_finite(item)
    elif type(value) is float and not math.isfinite(value):
        raise ValueError('nonfinite checkpoint scalar')


def topology(value, template, label):
    """State dicts include non-tensor primitive `_extra_state` metadata."""
    if torch.is_tensor(template):
        tensor(value, template.shape, template.dtype, label)
    elif isinstance(template, dict):
        keys(value, template, label)
        for k in template:
            topology(value[k], template[k], label + '.' + k)
    elif isinstance(template, (list, tuple)):
        if type(value) is not type(template) or len(value) != len(template):
            raise ValueError(label + ': sequence topology changed')
        for a, b in zip(value, template):
            topology(a, b, label)
    elif type(value) is not type(template) or value != template:
        raise ValueError(label + ': fixed metadata changed')
