from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TitansMatchStats:
    """Container for TITANS nearest-neighbor matching diagnostics."""

    topk_similarity: torch.Tensor
    novelty: torch.Tensor
    surprise: torch.Tensor


class TITANSMemory(nn.Module):
    """
    TITANS-style memory matcher for normal embedding prototypes.

    The module stores a rolling memory bank of normal embeddings and computes a
    ``Surprise Scalar`` per sample from nearest-neighbor mismatch.

    Surprise is defined from novelty:
      novelty = 1 - mean(top-k cosine similarity)
      surprise = clamp(0.5 * novelty, 0, 1)

    A value near 0 means highly familiar (likely normal), and near 1 means
    highly surprising (likely anomalous).
    """

    def __init__(
        self,
        embedding_dim: int,
        bank_size: int = 8192,
        k_neighbors: int = 8,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if bank_size <= 0:
            raise ValueError("bank_size must be > 0")
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be > 0")

        self.embedding_dim = int(embedding_dim)
        self.bank_size = int(bank_size)
        self.k_neighbors = int(k_neighbors)
        self.eps = float(eps)

        self.register_buffer("bank", torch.zeros(bank_size, embedding_dim))
        self.register_buffer("occupied", torch.zeros(bank_size, dtype=torch.bool))
        self.register_buffer("write_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("running_center", torch.zeros(embedding_dim))
        self.register_buffer("num_updates", torch.zeros(1, dtype=torch.long))

    @property
    def num_items(self) -> int:
        """Number of valid items currently stored in the memory bank."""
        return int(self.occupied.sum().item())

    def clear(self) -> None:
        """Reset memory bank content."""
        with torch.no_grad():
            self.bank.zero_()
            self.occupied.zero_()
            self.write_ptr.zero_()
            self.running_center.zero_()
            self.num_updates.zero_()

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor) -> None:
        """
        Insert new normal embeddings into TITANS memory.

        Args:
            embeddings: Tensor with shape ``(B, D)``.
        """
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (B, D)")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim}, got {embeddings.shape[1]}"
            )

        if embeddings.numel() == 0:
            return

        normed = F.normalize(embeddings.detach(), dim=-1, eps=self.eps)
        batch_size = normed.shape[0]

        ptr = int(self.write_ptr.item())
        for row in range(batch_size):
            self.bank[ptr].copy_(normed[row])
            self.occupied[ptr] = True
            ptr = (ptr + 1) % self.bank_size

        self.write_ptr[0] = ptr

        batch_center = normed.mean(dim=0)
        updates = int(self.num_updates.item())
        momentum = 0.995 if updates > 0 else 0.0
        self.running_center.mul_(momentum).add_(batch_center, alpha=1.0 - momentum)
        self.num_updates[0] = updates + 1

    def _valid_bank(self) -> torch.Tensor:
        valid = self.bank[self.occupied]
        if valid.numel() == 0:
            return torch.empty(0, self.embedding_dim, device=self.bank.device)
        return valid

    @torch.no_grad()
    def match(self, embeddings: torch.Tensor) -> TitansMatchStats:
        """
        Match embeddings against memory and return novelty diagnostics.

        Args:
            embeddings: Tensor with shape ``(B, D)``.
        """
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (B, D)")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim}, got {embeddings.shape[1]}"
            )

        normed = F.normalize(embeddings.detach(), dim=-1, eps=self.eps)
        valid_bank = self._valid_bank()

        if valid_bank.shape[0] == 0:
            novelty = torch.ones(normed.shape[0], device=normed.device)
            surprise = novelty.clone()
            empty_topk = torch.zeros(normed.shape[0], 1, device=normed.device)
            return TitansMatchStats(
                topk_similarity=empty_topk,
                novelty=novelty,
                surprise=surprise,
            )

        similarity = normed @ valid_bank.t()
        k = min(self.k_neighbors, similarity.shape[1])
        topk_similarity, _ = torch.topk(similarity, k=k, dim=1, largest=True)

        novelty = 1.0 - topk_similarity.mean(dim=1)
        surprise = (0.5 * novelty).clamp_(0.0, 1.0)

        return TitansMatchStats(
            topk_similarity=topk_similarity,
            novelty=novelty,
            surprise=surprise,
        )

    @torch.no_grad()
    def compute_surprise(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute only the scalar surprise score in ``[0, 1]``."""
        return self.match(embeddings).surprise

    @torch.no_grad()
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`compute_surprise` for module-style usage."""
        return self.compute_surprise(embeddings)

    def state_summary(self) -> Dict[str, float]:
        """Return lightweight state metadata for logging."""
        return {
            "num_items": float(self.num_items),
            "bank_size": float(self.bank_size),
            "k_neighbors": float(self.k_neighbors),
        }

    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, bank_size={self.bank_size}, "
            f"k_neighbors={self.k_neighbors}, num_items={self.num_items}"
        )
