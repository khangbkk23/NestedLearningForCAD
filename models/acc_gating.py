"""
acc_gating.py
-------------
ACC Gating — bộ lọc Game Theory quyết định có nên
consolidate (đưa vào Slow Memory / Coreset) hay không.

Toán học (instruction_CAD.md §2):
    ACC = cos_sim(z_updated, z_original) - H_term
    H_term = ||z_updated - z_original|| / d
    Approve nếu ACC > τ  (default τ = 0.25)

Nền tảng lý thuyết:
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

    Quyết định xem một bước TTT có tạo ra embedding đủ
    ổn định để đưa vào Slow Memory (CADIC Coreset) hay không.

    Nếu z_updated quá khác z_original → memory vừa học điều gì
    "lạ" (có thể là noise/anomaly) → KHÔNG consolidate.

    Không kế thừa nn.Module vì không có learnable parameters.
    """

    def __init__(self, tau: float = 0.25):
        """
        Args:
            tau: ngưỡng ACC (instruction_CAD §5, constraint 8).
                 Giá trị mặc định = 0.25.
                 Tăng τ → chọn lọc hơn (ít update Coreset hơn).
                 Giảm τ → dễ approve hơn (update Coreset thường xuyên hơn).
        """
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau phải trong khoảng (0, 1), nhận được {tau}")
        self.tau = tau
        self._history: list[float] = []   # log ACC scores để debug

    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_acc(
        self,
        z_updated: torch.Tensor,
        z_original: torch.Tensor,
    ) -> float:
        """
        Tính ACC score.

        Args:
            z_updated:  [batch, d] — sau TTT update
            z_original: [batch, d] — trước TTT update (z_cls gốc từ backbone)

        Returns:
            acc_score: float
        """
        # Cosine similarity trung bình trên batch
        cos_sim = F.cosine_similarity(z_updated, z_original, dim=-1).mean()

        # H_term: đo mức độ "drift" bình thường hoá theo chiều d
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
        Quyết định chính: có approve cho Slow Memory không?

        Args:
            z_updated:  [batch, d]
            z_original: [batch, d]

        Returns:
            approved:  bool — True nếu ACC > τ
            acc_score: float — giá trị ACC để logging
        """
        acc = self.compute_acc(z_updated, z_original)
        self._history.append(acc)

        approved = acc > self.tau

        logger.debug(
            f"[ACCGating] ACC={acc:.4f} τ={self.tau} "
            f"→ {'APPROVED ✓' if approved else 'REJECTED ✗'}"
        )

        return approved, acc

    # ------------------------------------------------------------------
    def running_avg_acc(self, window: int = 50) -> float:
        """
        ACC trung bình trên window bước gần nhất.
        Dùng để monitor xem model đang ổn định hay drift.
        """
        if not self._history:
            return 0.0
        recent = self._history[-window:]
        return sum(recent) / len(recent)

    def approval_rate(self, window: int = 100) -> float:
        """Tỉ lệ approved trong window bước gần nhất (0.0 → 1.0)."""
        if not self._history:
            return 0.0
        recent = self._history[-window:]
        # Chỉ đếm những lần approved
        approved_count = sum(1 for acc in recent if acc > self.tau)
        return approved_count / len(recent)

    def reset_history(self) -> None:
        self._history.clear()

    def state_dict(self) -> dict:
        return {"tau": self.tau, "history": self._history[-1000:]}

    def load_state_dict(self, sd: dict) -> None:
        self.tau = sd["tau"]
        self._history = sd.get("history", [])
