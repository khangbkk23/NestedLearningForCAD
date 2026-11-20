"""Passive EMA shadow readouts for B322.

A shadow is a second copy of the decoder parameters that follows the trained
decoder and never feeds back into training. Definition (card B322, section 3):

    s_0 = theta_0
    s_t = a_t * s_{t-1} + (1 - a_t) * theta_t        after optimizer step t
    a_t = min(1 - 1/tau, 1 - 1/(t + 1))

``tau`` counts optimizer steps. ``tau == 1`` gives ``s_t == theta_t``. While
``t + 1 <= tau`` the shadow is the plain mean of theta_0..theta_t.
"""
import hashlib


def coefficient(step, tau):
    """Return a_t for optimizer step ``step`` (1-based) and horizon ``tau``."""
    for value in (step, tau):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Shadow step and tau must be positive integers")
    return min(1.0 - 1.0 / tau, 1.0 - 1.0 / (step + 1))


def validate_taus(taus):
    if (
        not isinstance(taus, (list, tuple))
        or not taus
        or len(set(taus)) != len(taus)
        or any(isinstance(t, bool) or not isinstance(t, int) or t < 1 for t in taus)
    ):
        raise ValueError("Shadow taus must be distinct positive integers")
    return tuple(taus)


class Shadows:
    """FP32 parameter copies updated in place; no RNG, gradient or mode change."""

    def __init__(self, torch, decoder, taus):
        self.torch = torch
        self.taus = validate_taus(taus)
        names = [name for name, _ in decoder.named_parameters()]
        if list(decoder.buffers()) or sorted(decoder.state_dict()) != sorted(names):
            raise ValueError("Shadow requires a decoder whose state is parameters only")
        self.step = 0
        self.params = {
            tau: {
                name: value.detach().clone()
                for name, value in decoder.named_parameters()
            }
            for tau in self.taus
        }

    def update(self, decoder):
        """Call once, right after each optimizer step."""
        torch = self.torch
        self.step += 1
        with torch.no_grad():
            for tau, params in self.params.items():
                a = coefficient(self.step, tau)
                for name, value in decoder.named_parameters():
                    params[name].mul_(a).add_(value.detach(), alpha=1.0 - a)

    def forward(self, decoder, tau, first, second):
        """Run ``decoder`` with the shadow parameters; the module is not modified."""
        return self.torch.func.functional_call(decoder, self.params[tau], (first, second))

    def digest(self, tau):
        value = hashlib.sha256()
        for name, tensor in sorted(self.params[tau].items()):
            value.update(name.encode())
            value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return value.hexdigest()

    def nbytes(self, tau):
        return sum(t.numel() * t.element_size() for t in self.params[tau].values())
