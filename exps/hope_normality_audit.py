# exps/hope_normality_audit.py
"""Evaluator-only representation alignment and fixed-teacher diagnostics."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from exps.hope_image_synchronous_memory import EPS, comparison


@torch.no_grad()
def cosine_summary(a, b):
    values = F.cosine_similarity(a.double(), b.double(), dim=-1, eps=EPS)
    quantiles = torch.quantile(values, values.new_tensor([0, .01, .25, .5, .75, .99, 1]))
    return {**dict(zip(("min", "p01", "q25", "median", "q75", "p99", "max"), map(float, quantiles))),
            "mean": float(values.mean()), "std": float(values.std(unbiased=False))}


@torch.no_grad()
def relational_alignment(a, b):
    """Centered linear CKA and off-diagonal cosine-kernel correlation."""
    a, b = a.detach().double(), b.detach().double()
    ac, bc = a - a.mean(0), b - b.mean(0)
    ga, gb = ac @ ac.T, bc @ bc.T
    cka = (ga * gb).sum() / (ga.norm() * gb.norm()).clamp_min(EPS)
    na, nb = F.normalize(a, dim=-1, eps=EPS), F.normalize(b, dim=-1, eps=EPS)
    ids = torch.triu_indices(len(a), len(a), offset=1, device=a.device)
    ra, rb = (na @ na.T)[ids[0], ids[1]], (nb @ nb.T)[ids[0], ids[1]]
    ra, rb = ra - ra.mean(), rb - rb.mean()
    correlation = (ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(EPS)
    return {"linear_cka": float(cka), "cosine_kernel_correlation": float(correlation)}


@torch.no_grad()
def interimage_cosine_correlation(a, b, image_ids):
    """Relational correlation using only pairs from different images."""
    a, b = F.normalize(a.double(), dim=-1, eps=EPS), F.normalize(b.double(), dim=-1, eps=EPS)
    mask = torch.triu(image_ids[:, None] != image_ids[None, :], diagonal=1)
    ra, rb = (a @ a.T)[mask], (b @ b.T)[mask]
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(EPS))


@torch.no_grad()
def direction_overlap(queries, keys, delta, *, dimensions=64):
    """Dominant uncentered read/write directions in the memory input basis."""
    q, k, delta = queries.double(), keys.double(), delta.double()
    r = min(dimensions, q.shape[1], len(q), len(k))
    _, vq = torch.linalg.eigh(q.T @ q)
    _, vk = torch.linalg.eigh(k.T @ k)
    uq, uk = vq[:, -r:], vk[:, -r:]
    return {"dominant_dimensions": r,
            "dominant_subspace_overlap": float((uq.T @ uk).square().sum() / r),
            "key_energy_in_query_subspace": float((k @ uq).square().sum() / k.square().sum().clamp_min(EPS)),
            "write_energy_in_query_subspace": float((delta @ uq).square().sum() / delta.square().sum().clamp_min(EPS)),
            "write_energy_in_key_subspace": float((delta @ uk).square().sum() / delta.square().sum().clamp_min(EPS)),
            "isotropic_overlap_reference": r / q.shape[1]}


@dataclass(frozen=True)
class FrozenTeacherMap:
    coefficient: torch.Tensor
    input_mean: torch.Tensor
    target_mean: torch.Tensor
    ridge: float

    @torch.no_grad()
    def predict(self, rows):
        return (rows.detach().double() - self.input_mean) @ self.coefficient + self.target_mean

    @torch.no_grad()
    def error(self, rows, target):
        target = target.detach().double()
        return float((self.predict(rows) - target).square().sum() / target.square().sum().clamp_min(EPS))


@torch.no_grad()
def fit_teacher_map(inputs, targets, *, ridge_fraction=.001):
    """One fixed ridge decoder; callers must enforce disjoint normal fit/probe IDs."""
    x, y = inputs.detach().double(), targets.detach().double()
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    gram = xc.T @ xc / len(x)
    ridge = ridge_fraction * float(gram.trace()) / gram.shape[0]
    if ridge <= 0:
        raise ValueError("nonzero fit geometry required")
    coefficient = torch.linalg.solve(gram + ridge * torch.eye(gram.shape[0], device=x.device, dtype=x.dtype),
                                     xc.T @ yc / len(x))
    return FrozenTeacherMap(coefficient.detach().clone(), xm.clone(), ym.clone(), ridge)


@torch.no_grad()
def fixed_teacher(image):
    """State-independent frozen ViT target; never a mutable-memory target."""
    return F.normalize(image.detach().double(), dim=-1, eps=1e-8).clone()


@torch.no_grad()
def objective_gradient(weight, quantities):
    k, v = quantities.keys.double(), quantities.values.double()
    w = quantities.gates.double() / quantities.count
    return ((weight.double() @ k.T - weight.double() @ v.T) * w.T) @ k + (weight.double() @ k.T * w.T) @ k


@torch.no_grad()
def output_change(reference, current):
    return comparison(reference, current)
