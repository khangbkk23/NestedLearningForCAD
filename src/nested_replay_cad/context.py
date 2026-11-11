"""CTX-RN-01 exploratory detached residual-context transform.

Only raw columns0/1 are transformed, using the same causal active replay set.
No queries, category IDs, labels, future records or learned state are accessed.
"""

CONTEXT_MODE = "ctx-rn01-residual-sigma-floor1e-3-v1"
SIGMA_FLOOR = 1e-3


def normalize_context(contexts):
    """Return fresh contiguous CPU FP32 contexts; never modify the input."""
    import torch

    if (
        not isinstance(contexts, torch.Tensor)
        or contexts.device.type != "cpu"
        or contexts.dtype != torch.float32
        or contexts.layout != torch.strided
        or contexts.ndim != 2
        or contexts.shape[1] != 4
        or contexts.shape[0] > 4
        or not contexts.is_contiguous()
        or contexts.requires_grad
        or contexts.grad_fn is not None
        or contexts.is_conj()
        or contexts.is_neg()
    ):
        raise ValueError("CTX-RN-01 requires detached contiguous CPU FP32[m<=4,4]")
    if not torch.isfinite(contexts).all().item():
        raise ValueError("CTX-RN-01 nonfinite contexts")
    # Match the inherited context producer's declared FP32 boundary allowance.
    # No clamp is applied to the raw context or preserved columns.
    if not ((contexts >= -2e-6) & (contexts <= 1 + 2e-6)).all().item():
        raise ValueError("CTX-RN-01 raw contexts outside inherited0..1 contract")
    with torch.no_grad():
        result = contexts.clone().contiguous()
        if len(result):
            residual = contexts[:, :2]
            centered = residual - residual.mean(dim=0, keepdim=True)
            sigma = centered.square().mean(dim=0, keepdim=True).sqrt()
            standardized = centered / sigma.clamp_min(SIGMA_FLOOR)
            result[:, :2] = 0.5 + 0.25 * standardized.clamp(-2.0, 2.0)
    if (
        result.requires_grad
        or result.grad_fn is not None
        or not torch.isfinite(result).all().item()
    ):
        raise ValueError("CTX-RN-01 invalid detached output")
    return result


normalize_contexts = normalize_context
