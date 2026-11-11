"""Approved revision helpers under qualification; not yet the active model route.

Lazy imports preserve metadata-only use. Wire model/config/codec identity together
before production use; importing this module does not change existing models.
"""

from nested_replay_cad.core.contracts import PhiState
from nested_replay_cad.core.errors import NonFiniteState, SchemaError, ScopeBlocked
from nested_replay_cad.core.policy.layout import phi_shapes, validate_parameters
from nested_replay_cad.core.runtime.tensors import reference_torch


def _logits(t, params, contexts):
    hidden_pre = t.nn.functional.linear(
        contexts, params["hidden.weight"], params["hidden.bias"]
    )
    if not t.isfinite(hidden_pre).all().item():
        raise NonFiniteState("nonfinite_policy_hidden_activation")
    hidden = t.tanh(hidden_pre)
    out_pre = t.nn.functional.linear(
        hidden, params["out.weight"], params["out.bias"]
    ).squeeze(-1)
    if not t.isfinite(out_pre).all().item():
        raise NonFiniteState("nonfinite_policy_output_activation")
    return 2 * t.tanh(out_pre)


def policy_weights_first_vjp(phi, contexts):
    """Fixed detached contexts; first derivative to phi only, no policy Hessian.

    Each unweighted logit Jacobian is computed in FP32. Pair coefficient and
    Jacobian difference use FP64, each pair is cast to FP32 before ordered sum.
    This is a numerical route, not exact arithmetic or a higher-order oracle.
    """
    t = reference_torch()
    if not isinstance(phi, PhiState):
        raise SchemaError("invalid_conditional_phi")
    validate_parameters(phi.parameters, phi_shapes())
    c = contexts
    if (
        not isinstance(c, t.Tensor)
        or c.device.type != "cpu"
        or c.dtype != t.float32
        or c.layout != t.strided
        or c.ndim != 2
        or c.shape[1] != 4
        or c.shape[0] > 4
        or c.requires_grad
        or c.grad_fn is not None
        or not c.is_contiguous()
    ):
        raise SchemaError("invalid_detached_policy_context")
    if not t.isfinite(c).all().item():
        raise NonFiniteState("nonfinite_policy_context")
    if not ((c >= -2e-6) & (c <= 1 + 2e-6)).all().item():
        raise SchemaError("policy_context_out_of_range")
    names = tuple(sorted(phi.parameters))

    class PairwisePolicy(t.autograd.Function):
        @staticmethod
        def forward(ctx, c, *parameters):
            ctx.save_for_backward(c, *parameters)
            if len(c) == 0:
                return t.empty(0, dtype=t.float32, device="cpu")
            logits = _logits(t, dict(zip(names, parameters)), c)
            weights = 0.1 / len(c) + 0.9 * t.softmax(logits, dim=0)
            if not t.isfinite(weights).all().item():
                raise NonFiniteState("nonfinite_policy_weights")
            return weights

        @staticmethod
        def backward(ctx, upstream):
            if t.is_grad_enabled():
                raise ScopeBlocked("policy_first_VJP_only_higher_order_not_qualified")
            if not t.isfinite(upstream).all().item():
                raise NonFiniteState("nonfinite_policy_upstream")
            c, *saved = ctx.saved_tensors
            with t.enable_grad():
                params = {
                    n: p.detach().clone().requires_grad_() for n, p in zip(names, saved)
                }
                values = tuple(params.values())
                gradients = tuple(t.zeros_like(p) for p in values)
                if len(c) > 1:
                    logits = _logits(t, params, c)
                    probabilities = t.softmax(logits, dim=0)
                    jacobians = [
                        t.autograd.grad(logits[i], values, retain_graph=True)
                        for i in range(len(c))
                    ]
                    for i in range(len(c)):
                        for j in range(i + 1, len(c)):
                            coefficient = (
                                0.9
                                * probabilities[i].double()
                                * probabilities[j].double()
                                * (upstream[i].double() - upstream[j].double())
                            ).detach()
                            pair = tuple(
                                (coefficient * (a.double() - b.double())).to(a.dtype)
                                for a, b in zip(jacobians[i], jacobians[j])
                            )
                            gradients = tuple(a + b for a, b in zip(gradients, pair))
            if any(not t.isfinite(g).all().item() for g in gradients):
                raise NonFiniteState("nonfinite_policy_gradient")
            return (None, *gradients)

    return PairwisePolicy.apply(c, *(phi.parameters[n] for n in names))
