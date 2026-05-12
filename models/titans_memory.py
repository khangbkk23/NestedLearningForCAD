"""
titans_memory.py
----------------
TITANS Fast Memory — Phase 1 của Meta-NATH CAD.

Triết lý: Không phải MLP, không phải neural net.
TITANS ở đây chỉ là một ma trận M [d×d] được cập nhật theo
Delta Rule hoàn toàn trong torch.no_grad().

Toán học (instruction_CAD.md §2):
    M_t = (1 - α) * M_{t-1} + η_t * (v_t - M_{t-1} k_t) k_t^T
    η_t = η₀ / (1 + surprise_t)
    clamp M ∈ [-5.0, 5.0]
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. TITANSMemory — chỉ lưu ma trận M
# ---------------------------------------------------------------------------

class TITANSMemory:
    """
    Associative memory là một ma trận M duy nhất kích thước [d, d].

    Không kế thừa nn.Module vì M được cập nhật thủ công (no autograd).
    Nếu cần save/load checkpoint, dùng state_dict() / load_state_dict().
    """

    def __init__(self, d: int = 768, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.d = d
        self.device = device
        # Buffer chính — không tham gia autograd
        self.M: torch.Tensor = torch.zeros(d, d, device=device)

    # ------------------------------------------------------------------
    def retrieve(self, k: torch.Tensor) -> torch.Tensor:
        """
        Truy xuất bộ nhớ: z = M @ k^T => shape [batch, d].

        Args:
            k: [batch, d] — key vector (thường là z_cls từ DINOv3)

        Returns:
            pred: [batch, d]
        """
        return (self.M @ k.T).T   # [batch, d]

    # ------------------------------------------------------------------
    def surprise(self, v: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        Tính surprise scalar: norm của residual trung bình trên batch.

        Returns:
            scalar tensor (không gradient)
        """
        residual = v - pred                          # [batch, d]
        return residual.norm(dim=-1).mean()          # scalar

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: float = 0.9,
        eta0: float = 0.01,
        clamp: float = 5.0,
    ) -> float:
        """
        Cập nhật M theo Delta Rule (bắt buộc gọi trong no_grad context).

        Args:
            k:     [batch, d] — key
            v:     [batch, d] — value (thường = k.detach() vì self-supervised)
            alpha: decay rate (default 0.9)
            eta0:  base learning rate (default 0.01)
            clamp: giá trị clamp tuyệt đối cho M (default 5.0)

        Returns:
            surprise_scalar (float) — để TTTEngine log và điều chỉnh η
        """
        pred            = self.retrieve(k)                      # [batch, d]
        surprise_scalar = self.surprise(v, pred).item()

        eta_t  = eta0 / (1.0 + surprise_scalar)
        delta  = v - pred                                        # [batch, d]
        # Outer product trung bình trên batch: [d, d]
        update = torch.einsum("bi,bj->ij", delta, k) / k.shape[0]

        self.M.data = (1.0 - alpha) * self.M.data + eta_t * update
        self.M.data.clamp_(-clamp, clamp)

        return surprise_scalar

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"M": self.M.cpu().clone(), "d": self.d}

    def load_state_dict(self, sd: dict) -> None:
        self.M = sd["M"].to(self.device)
        self.d = sd["d"]

    def reset(self) -> None:
        """Reset M về zero (dùng khi bắt đầu task mới nếu muốn)."""
        self.M.zero_()


# ---------------------------------------------------------------------------
# 2. TTTEngine — Phase 1 orchestrator
# ---------------------------------------------------------------------------

class TTTEngine:
    """
    Test-Time Training Engine (Phase 1 — Nhịp Phản xạ).

    Nhận z_cls từ DINOv3, cập nhật TITANSMemory, trả về z_updated
    và flag `approved` cho Phase 2 (ACCGating quyết định).

    Không kế thừa nn.Module — toàn bộ tính toán nằm trong no_grad.
    """

    def __init__(
        self,
        d: int = 768,
        alpha: float = 0.9,
        eta0: float = 0.01,
        clamp: float = 5.0,
        device: str | None = None,
    ):
        self.alpha  = alpha
        self.eta0   = eta0
        self.clamp  = clamp
        self.memory = TITANSMemory(d=d, device=device)
        self._step  = 0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def process(self, z_cls: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Chạy một bước TTT trên batch z_cls.

        Pipeline:
            1. Retrieve:  pred = M @ z_cls^T
            2. Compute surprise
            3. Update M (Delta Rule)
            4. Produce z_updated = M_updated @ z_cls^T

        Args:
            z_cls: [batch, d] — CLS token từ DINOv3 (đã detach)

        Returns:
            z_updated:       [batch, d]
            surprise_scalar: float
        """
        self._step += 1

        # k = v = z_cls (self-supervised: memory nhớ chính nó)
        k = z_cls
        v = z_cls

        surprise_scalar = self.memory.update(
            k=k, v=v,
            alpha=self.alpha,
            eta0=self.eta0,
            clamp=self.clamp,
        )

        # Retrieve sau khi update để lấy z_updated
        z_updated = self.memory.retrieve(z_cls)    # [batch, d]

        if self._step % 100 == 0:
            logger.debug(
                f"[TTTEngine] step={self._step} "
                f"surprise={surprise_scalar:.4f} "
                f"|M|_F={self.memory.M.norm().item():.4f}"
            )

        return z_updated, surprise_scalar

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "memory": self.memory.state_dict(),
            "alpha":  self.alpha,
            "eta0":   self.eta0,
            "step":   self._step,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.memory.load_state_dict(sd["memory"])
        self.alpha  = sd["alpha"]
        self.eta0   = sd["eta0"]
        self._step  = sd["step"]

    def reset_memory(self) -> None:
        """Xóa Fast Memory (dùng khi muốn reset giữa các experiment)."""
        self.memory.reset()
        self._step = 0
