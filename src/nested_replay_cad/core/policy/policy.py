"""A0-R1 bounded replay policy over detached four-feature contexts."""

from __future__ import annotations

from nested_replay_cad.core.contracts import PhiState


class ReplayPolicy:
    def weights(self, phi: PhiState, contexts_detached: Tensor) -> Tensor:
        """Detached context, bounded logits and fixed floor; gradient to phi kept."""
        from nested_replay_cad.core.policy.numerical import policy_weights_first_vjp

        return policy_weights_first_vjp(phi, contexts_detached)
