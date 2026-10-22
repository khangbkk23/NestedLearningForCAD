"""Out-of-place CMS state. No task-boundary reset or implicit flush exists."""
from dataclasses import dataclass
import hashlib
import torch


def tensor_digest(groups):
    digest = hashlib.sha256()
    for group in groups:
        for name, value in sorted(group.items()):
            value = value.detach().contiguous().cpu()
            digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CMSState:
    weights: tuple[dict[str, torch.Tensor], ...]
    accumulators: tuple[dict[str, torch.Tensor], ...]
    pending: tuple[int, ...]
    commits: tuple[int, ...]
    clock: int = 0

    def detached(self):
        return CMSState(
            tuple({k: v.detach().clone().requires_grad_(True) for k, v in w.items()} for w in self.weights),
            tuple({k: v.detach().clone() for k, v in w.items()} for w in self.accumulators),
            self.pending, self.commits, self.clock,
        )

    def checkpoint(self):
        return dict(schema="cms_state_v3", clock=self.clock, pending=list(self.pending),
                    commits=list(self.commits),
                    weights=[{k: v.detach().cpu().clone() for k, v in w.items()} for w in self.weights],
                    accumulators=[{k: v.detach().cpu().clone() for k, v in w.items()} for w in self.accumulators])


def finite_tensors(values, context):
    if any(not bool(torch.isfinite(v).all()) for v in values):
        raise FloatingPointError(f"non-finite {context}")


def vector_norm(values):
    # Diagnostic precision only; never feeds back into an update.
    return torch.cat([v.detach().reshape(-1).double() for v in values]).norm().item()
