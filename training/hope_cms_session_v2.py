"""Transactional feature-level deployment state used by W4 and the future W5 adapter."""
from copy import deepcopy
import torch
from .hope_outer_learning_v2 import capture_rng, restore_rng


class CMSSession:
    def __init__(self, model, identity):
        if not identity:
            raise ValueError('session requires source/data/config identity')
        self.model = deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.identity = deepcopy(identity)
        self.state = self.model.cms.initial_state()
        self.cursor = 0

    def update(self, feature_factory):
        """Factory receives next view. Failure rolls RNG, cursor and CMS back atomically."""
        rng = capture_rng()
        try:
            features, target, mask = feature_factory(self.state.clock % 4)
            with torch.no_grad():
                x = self.model.encode(features)
            proposal = self.model.cms.event(x, target, mask, self.model.decoder, self.state)
        except Exception:
            restore_rng(rng)
            raise
        self.state = proposal.state
        self.cursor += 1
        return proposal

    def update_transport(self, factories):
        # A transport batch has the semantics of ordered single-image transactions.
        return [self.update(factory) for factory in factories]

    @torch.no_grad()
    def predict(self, features):
        rng = capture_rng()
        try:
            return self.model.predict(features, self.state)
        finally:
            restore_rng(rng)

    def checkpoint(self):
        return dict(schema='hope_cms_session_v3', identity=deepcopy(self.identity),
                    metadata=self.model.metadata(), model=deepcopy(self.model.state_dict()),
                    cms=self.state.checkpoint(), cursor=self.cursor, rng=capture_rng())

    def restore(self, payload):
        if (payload.get('schema') != 'hope_cms_session_v3' or payload['identity'] != self.identity
                or payload['metadata'] != self.model.metadata() or type(payload['cursor']) is not int):
            raise ValueError('deployment checkpoint identity mismatch')
        # Validate everything in an isolated clone before replacing live state/RNG.
        candidate = deepcopy(self.model)
        candidate.load_state_dict(payload['model'], strict=True)
        state = candidate.cms.load_state(payload['cms'])
        if state.clock != payload['cursor']:
            raise ValueError('deployment cursor/clock mismatch')
        old_rng = capture_rng()
        try:
            restore_rng(payload['rng'])
        except Exception:
            restore_rng(old_rng)
            raise
        self.model, self.state, self.cursor = candidate, state, payload['cursor']
