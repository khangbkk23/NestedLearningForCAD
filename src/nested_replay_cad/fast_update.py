"""Forward-bound first conditional VJP candidate, not production adoption.

FP32/FP64 optimizer primals and branch masks are observed in source dtype;
chain-rule arithmetic is accumulated in FP64. This is a smooth surrogate
derivative convention, not a literal derivative of quantized arithmetic.
"""

import math


def clipping_coefficient(gradients, maximum=0.1):
    import torch

    norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g, 2) for g in gradients.values()]), 2
    )
    plus = norm + 1e-6
    raw = maximum / plus
    return raw.clamp(max=1.0), norm


def _tensor_primals(g, state, c, betas, correction2, eps, clip_threshold):
    import torch

    beta1, beta2 = betas
    if any(
        state[k].dtype != g.dtype
        or state[k].shape != g.shape
        or state[k].device != g.device
        for k in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
    ):
        raise ValueError("Moment geometry/dtype/device")
    h = g * c
    mean = (state["exp_avg"] * beta1).add(h, alpha=1 - beta1)
    variance = (state["exp_avg_sq"] * beta2).addcmul(h, h, value=1 - beta2)
    old = state["max_exp_avg_sq"]
    maximum_variance = torch.maximum(old, variance)
    root = maximum_variance.sqrt()
    denominator = (root / math.sqrt(correction2)).add(eps)
    ratio = h / denominator
    rms = torch.linalg.vector_norm(ratio, 2) / math.sqrt(h.numel())
    scaled_rms = rms / clip_threshold
    scaled = float(scaled_rms.detach()) > 1.0
    return dict(
        h=h,
        mean=mean,
        variance=variance,
        maximum_variance=maximum_variance,
        root=root,
        denominator=denominator,
        ratio=ratio,
        rms=rms,
        scale=scaled_rms if scaled else h.new_ones(()),
        rms_scaled=scaled,
        ams_gt=variance > old,
        ams_eq=variance == old,
    )


def forward_primals(
    gradients,
    moments,
    step,
    *,
    betas=(0.9, 0.999),
    eps=1e-10,
    maximum=0.1,
    clip_threshold=1.0,
):
    """Independent per-tensor primals; do not mutate gradients or moments.

    Callers may inspect masks/counts without changing the algorithm. Tensor values
    retain their graph unless caller uses no_grad; conditional_vjp uses no_grad.
    """
    import torch

    if type(step) is not int or step < 0 or maximum <= 0 or clip_threshold <= 0:
        raise ValueError("Invalid update constants")
    if not gradients or set(gradients) != set(moments):
        raise ValueError("Gradient/moment population")
    first = next(iter(gradients.values()))
    if first.dtype not in (torch.float32, torch.float64):
        raise ValueError("FP32/FP64 gradients required")
    if any(
        g.dtype != first.dtype or g.device != first.device for g in gradients.values()
    ):
        raise ValueError("Mixed dtype/device")
    c, n = clipping_coefficient(gradients, maximum)
    plus = n + 1e-6
    raw = maximum / plus
    beta1, beta2 = betas
    correction2 = 1 - beta2 ** (step + 1)
    rows = {}
    for name, g in gradients.items():
        rows[name] = _tensor_primals(
            g, moments[name], c, betas, correction2, eps, clip_threshold
        )
    return dict(
        clip=c,
        norm=n,
        norm_plus=plus,
        clip_raw=raw,
        clip_derivative_active=bool((raw <= 1).detach()),
        tensors=rows,
    )


def functional_step(
    parameters,
    gradients,
    moments,
    step,
    *,
    lr=0.002,
    betas=(0.9, 0.999),
    eps=1e-10,
    weight_decay=1e-4,
    maximum=0.1,
    clip_threshold=1.0,
    destination=None,
):
    import torch

    if set(parameters) != set(gradients):
        raise ValueError("Parameter population")
    if destination is not None:
        if torch.is_grad_enabled() or set(destination) != set(parameters):
            raise ValueError("No-grad distinct destination required")
        if any(
            destination[n].untyped_storage().data_ptr()
            == p.untyped_storage().data_ptr()
            for n, p in parameters.items()
        ):
            raise ValueError("Destination aliases parent")
        if any(
            destination[n].shape != p.shape
            or destination[n].dtype != p.dtype
            or destination[n].device != p.device
            for n, p in parameters.items()
        ):
            raise ValueError("Destination geometry/dtype/device")
    c, norm = clipping_coefficient(gradients, maximum)
    correction1 = 1 - betas[0] ** (step + 1)
    correction2 = 1 - betas[1] ** (step + 1)
    result = {}
    for name, p in parameters.items():
        r = _tensor_primals(
            gradients[name], moments[name], c, betas, correction2, eps, clip_threshold
        )
        step_size = (
            float((lr / correction1 / r["scale"]).detach())
            if r["rms_scaled"]
            else lr / correction1
        )
        exact = torch.addcdiv(
            p * (1 - lr * weight_decay), r["mean"], r["denominator"], value=-step_size
        )
        if torch.is_grad_enabled():
            # Exact forward bytes, explicitly declared smooth conditional proxy.
            proxy = (
                p * (1 - lr * weight_decay)
                - (lr / correction1) * r["mean"] / r["denominator"] / r["scale"]
            )
            proposed = exact.detach() + (proxy - proxy.detach())
        else:
            proposed = exact
        if destination is None:
            result[name] = proposed
        else:
            destination[name].copy_(proposed)
    return result, dict(global_gradient_norm=norm)


def conditional_vjp(
    gradients,
    moments,
    step,
    query_gradients,
    *,
    lr=0.002,
    betas=(0.9, 0.999),
    eps=1e-10,
    maximum=0.1,
    clip_threshold=1.0,
):
    import torch

    if set(query_gradients) != set(gradients):
        raise ValueError("Query population")
    if any(
        g.requires_grad or q.requires_grad
        for g, q in zip(gradients.values(), query_gradients.values())
    ):
        raise ValueError("Conditional first VJP requires detached gradients/query")
    beta1, beta2 = betas
    correction1 = 1 - beta1 ** (step + 1)
    correction2 = 1 - beta2 ** (step + 1)
    adjoints = {}
    branches = {}
    with torch.no_grad():
        coefficient, total = clipping_coefficient(gradients, maximum)
        norm_plus = total + 1e-6
        raw = maximum / norm_plus
        clip_derivative_active = bool(raw <= 1)
        for name in gradients:
            row = _tensor_primals(
                gradients[name],
                moments[name],
                coefficient,
                betas,
                correction2,
                eps,
                clip_threshold,
            )
            h = row["h"].double()
            mean = row["mean"].double()
            root = row["root"].double()
            denominator = row["denominator"].double()
            ratio = row["ratio"].double()
            rms = float(row["rms"])
            scale = float(row["scale"])
            query = query_gradients[name].double()
            active = row["ams_gt"].double() + 0.5 * row["ams_eq"].double()
            derivative_denominator = torch.zeros_like(h)
            positive = root > 0
            derivative_denominator[positive] = (
                active[positive]
                * (1 - beta2)
                * h[positive]
                / (math.sqrt(correction2) * root[positive])
            )
            diagonal = (
                1 - beta1
            ) / denominator - mean * derivative_denominator / denominator.square()
            scale_derivative = torch.zeros_like(h)
            if row["rms_scaled"]:
                scale_derivative = (
                    ratio
                    / (h.numel() * rms * clip_threshold)
                    * (
                        1 / denominator
                        - h * derivative_denominator / denominator.square()
                    )
                )
            adjoints[name] = (-lr / correction1) * (
                query * diagonal / scale
                - (query * mean / denominator).sum() * scale_derivative / scale**2
            )
            branches[name] = dict(
                amsgrad_equal=int(row["ams_eq"].sum()),
                zero_variance=int((row["maximum_variance"] == 0).sum()),
                rms=rms,
                rms_scaled=row["rms_scaled"],
            )
        norm = float(total)
        plus = float(norm_plus)
        c = float(coefficient)
        dot = math.fsum(
            float((adjoints[n] * gradients[n].double()).sum()) for n in gradients
        )
        coupling = (
            maximum * dot / (norm * plus**2) if clip_derivative_active and norm else 0.0
        )
        for name in gradients:
            adjoints[name] = c * adjoints[name] - coupling * gradients[name].double()
    if any(not torch.isfinite(v).all() for v in adjoints.values()):
        raise ValueError("Nonfinite conditional VJP")
    return adjoints, dict(
        global_gradient_norm=norm,
        global_clip_active=float(raw) < 1,
        clipping_coefficient=c,
        clip_derivative_active=clip_derivative_active,
        clip_exact_boundary=float(raw) == 1.0,
        tensors=branches,
        derivative_convention="source-dtype forward primals/branches; FP64 chain-rule first conditional VJP",
    )
