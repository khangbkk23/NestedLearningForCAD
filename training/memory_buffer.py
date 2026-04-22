from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class SlowMemoryEntry:
    """Single memory entry for slow normal-buffer retention."""

    image: torch.Tensor
    embedding: torch.Tensor
    utility: float
    metadata: Optional[Dict[str, Any]] = None


class SlowMemory:
    """
    Diversity-aware normal image buffer using utility-based pruning.

    Utility score follows a k-center intuition: samples that are far from the
    current memory manifold receive higher utility and are preferred.
    """

    def __init__(
        self,
        capacity: int,
        embedding_dim: int,
        utility_temperature: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")

        self.capacity = int(capacity)
        self.embedding_dim = int(embedding_dim)
        self.utility_temperature = float(utility_temperature)
        self.eps = float(eps)
        self._entries: List[SlowMemoryEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> List[SlowMemoryEntry]:
        return self._entries

    def clear(self) -> None:
        """Remove all buffer entries."""
        self._entries.clear()

    def _stack_embeddings(self) -> torch.Tensor:
        if not self._entries:
            return torch.empty(0, self.embedding_dim)
        emb = torch.stack([entry.embedding for entry in self._entries], dim=0)
        return F.normalize(emb, dim=-1, eps=self.eps)

    def utility_score(self, embedding: torch.Tensor) -> float:
        """
        Compute utility for one candidate embedding.

        Higher score means the sample adds more geometric diversity.
        """
        if embedding.ndim != 1:
            raise ValueError("embedding must have shape (D,)")
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim}, got {embedding.shape[0]}"
            )

        emb = F.normalize(embedding, dim=-1, eps=self.eps)
        existing = self._stack_embeddings()

        if existing.numel() == 0:
            return 1.0

        cosine = torch.matmul(existing, emb)
        min_cos = float(cosine.max().item())
        novelty = 1.0 - min_cos
        scaled = novelty / max(self.utility_temperature, self.eps)
        return float(scaled)

    def add_batch(
        self,
        images: torch.Tensor,
        embeddings: torch.Tensor,
        metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> None:
        """
        Add a batch of normal samples and keep only top-K diverse entries.

        Args:
            images: Tensor ``(B, C, H, W)``.
            embeddings: Tensor ``(B, D)``.
            metadata: Optional list of length B with auxiliary fields.
        """
        if images.ndim < 2:
            raise ValueError("images must have batch dimension")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (B, D)")
        if images.shape[0] != embeddings.shape[0]:
            raise ValueError("images and embeddings must share batch size")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim}, got {embeddings.shape[1]}"
            )

        batch_size = images.shape[0]
        if metadata is None:
            metadata = [None] * batch_size
        if len(metadata) != batch_size:
            raise ValueError("metadata length must match batch size")

        with torch.no_grad():
            normed_embeddings = F.normalize(embeddings.detach().cpu(), dim=-1, eps=self.eps)
            images_cpu = images.detach().cpu()

            for idx in range(batch_size):
                emb = normed_embeddings[idx]
                util = self.utility_score(emb)
                self._entries.append(
                    SlowMemoryEntry(
                        image=images_cpu[idx],
                        embedding=emb,
                        utility=util,
                        metadata=metadata[idx],
                    )
                )

            self._prune_to_capacity()

    def _kcenter_select(self, embeddings: torch.Tensor, k: int) -> List[int]:
        """
        Select representative indices via k-center greedy on cosine distance.
        """
        num_items = embeddings.shape[0]
        if k >= num_items:
            return list(range(num_items))

        selected: List[int] = []
        first_idx = int(torch.argmax(torch.norm(embeddings, dim=1)).item())
        selected.append(first_idx)

        min_dist = torch.full((num_items,), float("inf"))

        for _ in range(1, k):
            last_vec = embeddings[selected[-1]].unsqueeze(0)
            cosine = F.cosine_similarity(embeddings, last_vec.expand_as(embeddings), dim=1)
            distance = 1.0 - cosine
            min_dist = torch.minimum(min_dist, distance)
            next_idx = int(torch.argmax(min_dist).item())
            if next_idx in selected:
                break
            selected.append(next_idx)

        if len(selected) < k:
            for idx in range(num_items):
                if idx not in selected:
                    selected.append(idx)
                if len(selected) == k:
                    break

        return selected[:k]

    def _recompute_utility(self) -> None:
        if not self._entries:
            return

        embeddings = self._stack_embeddings()
        if embeddings.shape[0] == 1:
            self._entries[0].utility = 1.0
            return

        sims = embeddings @ embeddings.t()
        sims.fill_diagonal_(-1.0)
        nearest = sims.max(dim=1).values
        utilities = (1.0 - nearest).clamp_min(0.0)

        for idx, util in enumerate(utilities.tolist()):
            self._entries[idx].utility = float(util)

    def _prune_to_capacity(self) -> None:
        if len(self._entries) <= self.capacity:
            self._recompute_utility()
            return

        embeddings = self._stack_embeddings()
        keep_indices = self._kcenter_select(embeddings, self.capacity)
        keep_set = set(keep_indices)

        self._entries = [
            entry for idx, entry in enumerate(self._entries) if idx in keep_set
        ]

        self._recompute_utility()

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Randomly sample entries from slow memory.

        Returns:
            Dict with ``images``, ``embeddings``, and ``utility`` tensors.
        """
        if len(self._entries) == 0:
            return {
                "images": torch.empty(0),
                "embeddings": torch.empty(0, self.embedding_dim),
                "utility": torch.empty(0),
            }

        k = min(int(batch_size), len(self._entries))
        indices = torch.randperm(len(self._entries))[:k].tolist()

        images = torch.stack([self._entries[idx].image for idx in indices], dim=0)
        embeddings = torch.stack([self._entries[idx].embedding for idx in indices], dim=0)
        utility = torch.tensor([self._entries[idx].utility for idx in indices], dtype=torch.float32)

        return {
            "images": images,
            "embeddings": embeddings,
            "utility": utility,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Serializable state for checkpointing."""
        return {
            "capacity": self.capacity,
            "embedding_dim": self.embedding_dim,
            "utility_temperature": self.utility_temperature,
            "entries": [
                {
                    "image": entry.image,
                    "embedding": entry.embedding,
                    "utility": entry.utility,
                    "metadata": entry.metadata,
                }
                for entry in self._entries
            ],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state created by :meth:`state_dict`."""
        self.capacity = int(state["capacity"])
        self.embedding_dim = int(state["embedding_dim"])
        self.utility_temperature = float(state.get("utility_temperature", 1.0))

        self._entries = []
        for item in state.get("entries", []):
            self._entries.append(
                SlowMemoryEntry(
                    image=item["image"],
                    embedding=item["embedding"],
                    utility=float(item["utility"]),
                    metadata=item.get("metadata"),
                )
            )
