"""Boundary-consolidated slow level used as a function anchor (B323).

Two parameter levels with different update frequencies:

    fast  theta_F : one optimizer step per arrival
    slow  theta_S : a hard copy of theta_F, taken once per task boundary

Between boundaries theta_S is frozen. For a replay sample admitted before the
last boundary, the fast decoder is pulled towards the slow decoder's output on
the same frozen feature:

    D(x) = mean_g mean_positions [ 1 - cos( de_F,g(x), de_S,g(x) ) ]

``de_*`` are the fused decoder outputs (two groups, channel dimension 1). The
slow output is computed without gradient in eval mode, so it is deterministic
and consumes no random numbers; the module's mode is restored afterwards.
Nothing is cached between steps: the only extra state is one parameter copy.
"""
import hashlib


class SlowLevel:
    def __init__(self, torch, decoder):
        names = [name for name, _ in decoder.named_parameters()]
        if list(decoder.buffers()) or sorted(decoder.state_dict()) != sorted(names):
            raise ValueError("Slow level requires a decoder whose state is parameters only")
        self.torch = torch
        self.params = None
        self.boundary = 0
        self.updates = 0

    @property
    def active(self):
        return self.params is not None

    def consolidate(self, decoder, arrival):
        """Copy the fast parameters; call once, right after a boundary observer."""
        if isinstance(arrival, bool) or not isinstance(arrival, int) or arrival <= self.boundary:
            raise ValueError("Slow level boundaries must increase")
        self.params = {name: value.detach().clone() for name, value in decoder.named_parameters()}
        self.boundary = arrival
        self.updates += 1

    def covers(self, admitted_arrival):
        """True when a replay sample was admitted at or before the last boundary."""
        return self.active and admitted_arrival <= self.boundary

    def distance(self, decoder, feature, prediction):
        """Anchor distance between the given fast ``prediction`` and the slow output."""
        torch = self.torch
        if not self.active:
            raise ValueError("Slow level has not been consolidated")
        training = decoder.training
        decoder.eval()
        try:
            with torch.no_grad():
                _, slow = torch.func.functional_call(
                    decoder, self.params, (feature[None, 0], feature[None, 1])
                )
        finally:
            decoder.train(training)
        if len(slow) != len(prediction) or not prediction:
            raise ValueError("Anchor output groups differ")
        total = 0.0
        for fast_group, slow_group in zip(prediction, slow):
            if fast_group.shape != slow_group.shape:
                raise ValueError("Anchor output shapes differ")
            similarity = torch.nn.functional.cosine_similarity(fast_group, slow_group.detach(), dim=1)
            total = total + (1.0 - similarity).mean()
        return total / len(prediction)

    def digest(self):
        if not self.active:
            return None
        value = hashlib.sha256()
        for name, tensor in sorted(self.params.items()):
            value.update(name.encode())
            value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return value.hexdigest()

    def nbytes(self):
        return 0 if not self.active else sum(t.numel() * t.element_size() for t in self.params.values())
