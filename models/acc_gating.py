"""
acc_gating.py
-------------
ACC Gating decides whether a Test-Time Training update is safe enough
to consolidate into Slow Memory / Coreset.

Math (instruction_CAD.md):
    ACC = cos_sim(z_updated, z_original) - H_term
    H_term = ||z_updated - z_original|| / d
    Approve if ACC > tau (default tau = 0.25)

Theory references:
    - "Social Welfare Optimization..." (2512.07453v2)
    - "Agent Behavior in Continual Learning" (2512.07462v2)
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ACCGating:
    """
    Autonomous Consolidation Controller.

    Decides whether a TTT step produced embeddings that are stable enough
    to enter Slow Memory (CADIC Coreset).

    If z_updated drifts too far from z_original, the memory likely learned
    noise or anomaly evidence, so the update is not consolidated.

    This class is not an nn.Module because it has no learnable parameters.
    """

    def __init__(self, tau: float = 0.25):
        """
        Args:
            tau: ACC threshold (instruction_CAD, constraint 8).
                 Default is 0.25.
                 Higher tau is more selective and updates the Coreset less often.
                 Lower tau approves more updates.
        """
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in the open interval (0, 1); got {tau}")
        self.tau = tau
        self._history: list[float] = []   # Recent ACC scores for debugging.

    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_acc(
        self,
        z_updated: torch.Tensor,
        z_original: torch.Tensor,
    ) -> float:
        """
        Compute the ACC score.

        Args:
            z_updated:  [batch, d] after TTT update
            z_original: [batch, d] before TTT update

        Returns:
            acc_score: float
        """
        # Mean cosine similarity across the batch.
        cos_sim = F.cosine_similarity(z_updated, z_original, dim=-1).mean()

        # H_term measures dimension-normalized embedding drift.
        h_term = (z_updated - z_original).norm(dim=-1).mean() / z_updated.shape[-1]

        acc = (cos_sim - h_term).item()
        return acc

    # ------------------------------------------------------------------
    @torch.no_grad()
    def should_consolidate(
        self,
        z_updated: torch.Tensor,
        z_original: torch.Tensor,
    ) -> Tuple[bool, float]:
        """
        Decide whether to approve this batch for Slow Memory.

        Args:
            z_updated:  [batch, d]
            z_original: [batch, d]

        Returns:
            approved:  bool, True when ACC > tau
            acc_score: float for logging
        """
        acc = self.compute_acc(z_updated, z_original)
        self._history.append(acc)

        approved = acc > self.tau

        logger.debug(
            f"[ACCGating] ACC={acc:.4f} tau={self.tau} "
            f"{'APPROVED' if approved else 'REJECTED'}"
        )

        return approved, acc

    # ------------------------------------------------------------------
    def running_avg_acc(self, window: int = 50) -> float:
        """
        Average ACC over the most recent window.

        Useful for monitoring whether the model is stable or drifting.
        """
        if not self._history:
            return 0.0
        recent = self._history[-window:]
        return sum(recent) / len(recent)

    def approval_rate(self, window: int = 100) -> float:
        """Approval ratio over the most recent window (0.0 to 1.0)."""
        if not self._history:
            return 0.0
        recent = self._history[-window:]
        # Count only scores that would pass the current threshold.
        approved_count = sum(1 for acc in recent if acc > self.tau)
        return approved_count / len(recent)

    def reset_history(self) -> None:
        self._history.clear()

    def state_dict(self) -> dict:
        return {"tau": self.tau, "history": self._history[-1000:]}

    def load_state_dict(self, sd: dict) -> None:
        self.tau = sd["tau"]
        self._history = sd.get("history", [])
