"""Transactional one-image credit/admission with strict temporary-owner restore."""
from copy import deepcopy
from dataclasses import asdict
import torch
from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetConfig
from models.hope_cad_v2.state import finite_tensors, tensor_digest
from .contract import CONTRACT, design_digest, canonical, digest
from .geometry import (ForkableCoreset, AnchorReservoir, FixedContextProjection,
                       risk, anchor_risk, context_statistics, causal_scale)
from .learner import UtilityModel, NestedUtilityLearner
from .validation import keys, integer, scalar, tensor, tree_finite

MODES = ('CADIC', 'GEOMETRIC', 'FIRST_ORDER', 'NESTED', 'FAST_ONLY',
         'NO_FAST_ADAPTATION', 'NO_SELF_MODIFICATION', 'NO_ANCHOR_OBJECTIVE')
COUNTERS = ('proposals', 'learned_decisions', 'rejected_learned', 'accepted_proposals',
            'noop_proposals', 'labels_completed', 'censored')
PENDING_KEYS = ('decision_cursor', 'identity', 'accepted', 'other', 'main_sha256',
                'other_sha256', 'anchor0', 'anchor1', 'current0', 'current1', 'c', 'gamma',
                'tokens', 'stats', 'prediction', 'decision_d', 'learner_sha256', 'warmup')


def bank_digest(features):
    return tensor_digest(({'features': features},))


class SupportSession:
    def __init__(self, mode='NESTED', *, device='cpu', seed=0, bank_config=None, model_config=None):
        if mode not in MODES:
            raise ValueError('unregistered support policy')
        integer(seed, 'seed')
        self.mode, self.device, self.seed = mode, torch.device(device), seed
        g = CONTRACT['geometry']
        self.bank_config = bank_config or CADICPatchCoresetConfig(
            budget=g['bank_budget'], dim=g['raw_dim'], chunk_size=256,
            query_chunk_size=256, pair_chunk_size=256, image_neighbors=9)
        self.model_config = dict(model_config or {})
        if mode == 'NO_SELF_MODIFICATION':
            self.model_config['self_modifying'] = False
        self.bank = ForkableCoreset(self.bank_config, device=self.device)
        self.anchors = AnchorReservoir(seed=seed)
        self.learned, self.learner = mode not in ('CADIC', 'GEOMETRIC'), None
        if self.learned:
            devices = list(range(torch.cuda.device_count())) if self.device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                # Initialization is private; observed data never changes the initial template.
                torch.random.default_generator.manual_seed(seed)
                model = UtilityModel(**self.model_config).to(self.device).eval()
            learning_mode = mode if mode in ('FIRST_ORDER', 'FAST_ONLY', 'NO_FAST_ADAPTATION') else 'NESTED'
            self.learner = NestedUtilityLearner(model, learning_mode)
        self.projection = FixedContextProjection(self.bank_config.dim, self.model_config.get('context_dim', 32),
                                                  device=self.device) if self.learned else None
        self.cursor, self.pending, self.finalized, self.seen_ids = 0, None, False, []
        self.counters = {k: 0 for k in COUNTERS}

    @property
    def anchor_weight(self):
        return 0. if self.mode == 'NO_ANCHOR_OBJECTIVE' else 1.

    @property
    def image_tokens(self):
        return self.model_config.get('height', 28) * self.model_config.get('width', 28)

    def _bank_view(self, features):
        bank = ForkableCoreset(self.bank_config, device=self.device)
        bank.features = features
        return bank

    def learner_hash(self):
        return tensor_digest(({k: v for k, v in self.learner.model.state_dict().items() if torch.is_tensor(v)},
                              self.learner.phi)) if self.learner else None

    def initialization_hashes(self):
        return dict(theta=tensor_digest(({k: p for k, p in self.learner.model.slow_parameters()},)) if self.learner else None,
                    phi=tensor_digest((self.learner.phi,)) if self.learner else None,
                    projection=tensor_digest(({'P': self.projection.matrix},)) if self.projection else None,
                    anchor_rng=digest(self.anchors.rng.getstate()))

    def _resolve(self, features):
        pending = self.pending
        if pending is None:
            return None
        if pending['decision_cursor'] + 1 != self.cursor or pending['main_sha256'] != bank_digest(self.bank.features) or pending['learner_sha256'] != self.learner_hash():
            raise ValueError('fixed-bank/learner credit chronology violated')
        minus = pending['other'] if pending['accepted'] else self.bank.features
        plus = self.bank.features if pending['accepted'] else pending['other']
        next0, next1 = risk(features, self._bank_view(minus)), risk(features, self._bank_view(plus))
        j0 = next0 + self.anchor_weight * pending['anchor0']
        j1 = next1 + self.anchor_weight * pending['anchor1']
        target = (j0 - j1) / pending['c']
        scalar(target, 'delayed target')
        record = dict(tokens=pending['tokens'].to(self.device), stats=pending['stats'].to(self.device),
            c=pending['c'], gamma=pending['gamma'], target=target, decision_d=pending['decision_d'],
            decision_identity=pending['identity'], decision_cursor=pending['decision_cursor'], Jminus=j0, Jplus=j1)
        details = self.learner.observe(record)
        self.counters['labels_completed'] += 1
        self.pending = None
        return dict(decision_identity=pending['identity'], decision_cursor=pending['decision_cursor'],
            reward_observed_at=self.cursor, target=target, Jminus=j0, Jplus=j1, next_minus=next0, next_plus=next1,
            anchor_minus=pending['anchor0'], anchor_plus=pending['anchor1'], c=pending['c'], gamma=pending['gamma'],
            decision_d=pending['decision_d'], prediction=pending['prediction'],
            residual=pending['prediction'] - target, accepted=pending['accepted'], learning=details)

    def observe(self, features, identity):
        if self.finalized:
            raise ValueError('true stream end cannot resume')
        if not isinstance(identity, str) or not identity or identity in self.seen_ids:
            raise ValueError('missing/duplicate normal identity')
        if not torch.is_tensor(features) or features.dtype != torch.float32 or features.requires_grad:
            raise ValueError('detached raw FP32 features required')
        if features.shape not in ((self.image_tokens, self.bank_config.dim), (1, self.image_tokens, self.bank_config.dim)):
            raise ValueError('one image with declared raw geometry required')
        finite_tensors((features,), 'normal raw features')
        x = features.detach().to(self.device).reshape(-1, self.bank_config.dim)
        # Existing bank tensors/pending are immutable; only a candidate fork is written.
        backup = dict(bank=self.bank, pending=self.pending, cursor=self.cursor, ids=list(self.seen_ids),
                      counters=dict(self.counters), anchors=self.anchors.state_dict(),
                      learner=self.learner.state_dict() if self.learner else None)
        try:
            credit = self._resolve(x)
            prequential = risk(x, self.bank) if self.bank.count else None
            candidate = self.bank.fork()
            result = candidate.update(x)
            record = dict(cursor=self.cursor, identity=identity, prequential_risk=prequential,
                          credit=credit, proposal=result, mode=self.mode)
            if not self.bank.is_full:
                self.bank = candidate
                record.update(action='bootstrap', accepted=True, prediction=None)
            elif torch.equal(candidate.features, self.bank.features):
                self.counters['noop_proposals'] += 1
                self.bank = candidate
                record.update(action='no_change', accepted=True, prediction=None)
            else:
                self.counters['proposals'] += 1
                changed_rows = int((candidate.features != self.bank.features).any(dim=-1).sum())
                if self.mode == 'CADIC':
                    accepted, pred, warm = True, None, False
                else:
                    a0 = anchor_risk(self.anchors.images, self.bank) if self.anchor_weight else 0.
                    a1 = anchor_risk(self.anchors.images, candidate) if self.anchor_weight else 0.
                    stats, known = context_statistics(x, self.bank, candidate, result, self.anchors.seen,
                                                       self.cursor, a0, a1, self.anchor_weight)
                    c, gamma = causal_scale(known['current0'], known['current1'], a0, a1, self.anchor_weight)
                    if self.learned:
                        tokens = self.projection(x)
                        pred, decision_d = self.learner.predict(tokens, stats, c, gamma)
                        scalar(pred, 'admission prediction')
                        scalar(decision_d, 'admission denominator', 1.)
                        warm = self.learner.labels < 32
                        accepted = warm or pred >= 0.
                        if not warm:
                            self.counters['learned_decisions'] += 1
                            self.counters['rejected_learned'] += int(not accepted)
                    else:
                        pred, warm, accepted = gamma, False, gamma >= 0.
                    record.update(current_minus=known['current0'], current_plus=known['current1'],
                                  anchor_minus=a0, anchor_plus=a1, c=c, gamma=gamma)
                old = self.bank
                if accepted:
                    self.bank = candidate
                    self.counters['accepted_proposals'] += 1
                if self.learned:
                    other = old.features if accepted else candidate.features
                    self.pending = dict(decision_cursor=self.cursor, identity=identity, accepted=bool(accepted),
                        other=other, main_sha256=bank_digest(self.bank.features), other_sha256=bank_digest(other),
                        anchor0=a0, anchor1=a1, current0=known['current0'], current1=known['current1'],
                        c=c, gamma=gamma, tokens=tokens.detach().cpu().clone(), stats=stats.detach().cpu().clone(),
                        prediction=pred, decision_d=decision_d, learner_sha256=self.learner_hash(), warmup=warm)
                    record['decision_d'] = decision_d
                record.update(action='accept' if accepted else 'reject', accepted=bool(accepted),
                              prediction=pred, warmup=warm, changed_rows=changed_rows)
            if self.mode != 'CADIC':
                self.anchors.update(x, identity)
            self.seen_ids.append(identity)
            self.cursor += 1
            record.update(bank_count=self.bank.count, learner_labels=self.learner.labels if self.learner else 0,
                          fast_updates=self.learner.fast_updates if self.learner else 0,
                          slow_updates=self.learner.slow_updates if self.learner else 0,
                          bank_sha256=bank_digest(self.bank.features))
            tree_finite(record)
            return record
        except BaseException:
            self.bank, self.pending, self.cursor = backup['bank'], backup['pending'], backup['cursor']
            self.seen_ids, self.counters = backup['ids'], backup['counters']
            self.anchors.load_state_dict(backup['anchors'], self.device)
            if self.learner:
                self.learner.load_state_dict(backup['learner'])
            raise

    @torch.no_grad()
    def score(self, features):
        if not self.bank.count or features.dtype != torch.float32 or features.requires_grad:
            raise ValueError('nonempty bank and raw FP32 scoring input required')
        finite_tensors((features,), 'score features')
        scores = self.bank.pixel_scores(features.to(self.device))
        finite_tensors(scores, 'raw NN/Eq9 scores')
        return scores

    def finalize_stream(self):
        if not self.finalized:
            if self.pending is not None:
                self.counters['censored'] += 1
                self.pending = None
            self.finalized = True

    def identity(self):
        return canonical(dict(design_digest=design_digest(), mode=self.mode, seed=self.seed,
                              bank_config=asdict(self.bank_config), model_config=self.model_config))

    def state_dict(self):
        return NestedUtilityLearner._cpu(dict(schema='hope_support_session_v5_1', identity=self.identity(),
            cursor=self.cursor, finalized=self.finalized, seen_ids=self.seen_ids, counters=self.counters,
            bank=self.bank.state_dict(), bank_sha256=bank_digest(self.bank.features), anchors=self.anchors.state_dict(),
            pending=self.pending, learner=self.learner.state_dict() if self.learner else None))

    def _load_validated(self, payload):
        keys(payload, self.state_dict(), 'session')
        if payload['schema'] != 'hope_support_session_v5_1' or payload['identity'] != self.identity():
            raise ValueError('session identity changed or predecessor checkpoint')
        tree_finite(payload)
        cursor = integer(payload['cursor'], 'image cursor')
        ids = payload['seen_ids']
        if type(payload['finalized']) is not bool or not isinstance(ids, list) or len(ids) != cursor or any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != cursor:
            raise ValueError('corrupt image identities/clock')
        keys(payload['counters'], COUNTERS, 'session counters')
        counters = payload['counters']
        for k, v in counters.items():
            integer(v, k)
        if not (counters['rejected_learned'] <= counters['learned_decisions'] <= counters['proposals'] <= cursor) or counters['accepted_proposals'] > counters['proposals'] or counters['noop_proposals'] > cursor or counters['censored'] > 1:
            raise ValueError('corrupt proposal counters')
        if counters['censored'] and not payload['finalized']:
            raise ValueError('censoring is allowed only at true stream end')
        bankstate = payload['bank']
        keys(bankstate, self.bank.state_dict(), 'bank')
        if bankstate['schema'] != self.bank.schema or bankstate['config'] != asdict(self.bank_config):
            raise ValueError('bank topology changed')
        rows = bankstate['features'].shape[0] if torch.is_tensor(bankstate['features']) and bankstate['features'].ndim == 2 else -1
        if not 0 <= rows <= self.bank_config.budget:
            raise ValueError('invalid bank occupancy')
        tensor(bankstate['features'], (rows, self.bank_config.dim), torch.float32, 'raw bank')
        for name in ('seen_features', 'accepted_features', 'replaced_features', 'rejected_features', 'update_batches'):
            integer(bankstate[name], name)
        if rows != bankstate['accepted_features'] or bankstate['seen_features'] != sum(bankstate[k] for k in ('accepted_features', 'replaced_features', 'rejected_features')) or bankstate['update_batches'] > cursor or bankstate['seen_features'] != bankstate['update_batches'] * self.image_tokens:
            raise ValueError('bank counters inconsistent')
        if bank_digest(bankstate['features']) != payload['bank_sha256'] or bool(cursor) != bool(rows):
            raise ValueError('bank checksum/empty-state mismatch')
        self.anchors.validate(payload['anchors'], raw_dim=self.bank_config.dim, patches=min(8, self.image_tokens),
                              seen_ids=ids, expected_seen=0 if self.mode == 'CADIC' else cursor)
        self.bank.load_state_dict(bankstate)
        self.anchors.load_state_dict(payload['anchors'], self.device)
        if self.learned:
            self.learner.load_state_dict(payload['learner'])
            if self.learner.labels != counters['labels_completed'] or self.learner.labels > max(0, cursor - 1):
                raise ValueError('label accounting inconsistent')
            replay=self.learner.records+(self.learner.saved_meta_episode['records'] if self.learner.saved_meta_episode else [])
            for r in replay:
                if r['decision_cursor'] >= cursor - 1 or ids[r['decision_cursor']] != r['decision_identity']:
                    raise ValueError('future/unobserved replay identity')
        elif payload['learner'] is not None or counters['labels_completed'] or counters['censored'] or counters['learned_decisions']:
            raise ValueError('fabricated baseline learning')
        pending = deepcopy(payload['pending'])
        if pending is not None:
            keys(pending, PENDING_KEYS, 'pending')
            integer(pending['decision_cursor'],'pending cursor')
            if not self.learned or payload['finalized'] or type(pending['accepted']) is not bool or type(pending['warmup']) is not bool or pending['decision_cursor'] != cursor - 1 or not cursor or pending['identity'] != ids[-1]:
                raise ValueError('invalid pending chronology/action')
            tensor(pending['other'], (rows, self.bank_config.dim), torch.float32, 'alternative bank')
            if rows != self.bank_config.budget or torch.equal(pending['other'], self.bank.features.cpu()):
                raise ValueError('pending must be a changed full-bank contrast')
            if pending['main_sha256'] != payload['bank_sha256'] or pending['other_sha256'] != bank_digest(pending['other']) or pending['learner_sha256'] != self.learner_hash():
                raise ValueError('pending fixed state changed')
            for k in ('current0', 'current1', 'anchor0', 'anchor1'):
                scalar(pending[k], k, 0.)
            expected_c, expected_gamma = causal_scale(pending['current0'], pending['current1'], pending['anchor0'], pending['anchor1'], self.anchor_weight)
            if pending['c'] != expected_c or pending['gamma'] != expected_gamma:
                raise ValueError('stored causal coefficients changed')
            scalar(pending['prediction'], 'pending prediction')
            scalar(pending['decision_d'], 'pending denominator', 1.)
            tensor(pending['tokens'], (1, self.image_tokens, self.model_config.get('context_dim', 32)), torch.float32, 'pending tokens')
            tensor(pending['stats'], (16,), torch.float32, 'pending stats')
            p, d = self.learner.predict(pending['tokens'].to(self.device), pending['stats'].to(self.device), pending['c'], pending['gamma'])
            if p != pending['prediction'] or d != pending['decision_d'] or pending['warmup'] != (self.learner.labels < 32) or pending['accepted'] != (pending['warmup'] or p >= 0.):
                raise ValueError('pending prediction/admission inconsistent')
            pending['other'] = pending['other'].to(self.device)
        if self.learned and counters['labels_completed'] + counters['censored'] + int(pending is not None) != counters['proposals']:
            raise ValueError('proposal credit/censor accounting inconsistent')
        self.cursor, self.finalized, self.seen_ids = cursor, payload['finalized'], list(ids)
        self.counters, self.pending = dict(counters), pending

    def load_state_dict(self, payload):
        other = SupportSession(self.mode, device=self.device, seed=self.seed,
                               bank_config=self.bank_config, model_config=self.model_config)
        other._load_validated(payload)
        self.__dict__.update(other.__dict__)

    def memory_profile(self):
        tensors = [self.bank.features, self.pending, self.anchors.images]
        if self.projection:
            tensors.append(self.projection.matrix)
        if self.learner:
            tensors.extend([self.learner.model.state_dict(), self.learner.phi, self.learner.records,
                            self.learner.block_initial, self.learner.saved_meta_episode,
                            self.learner.optimizer.state_dict() if self.learner.optimizer else None])
        seen, total = set(), 0
        def add(v):
            nonlocal total
            if torch.is_tensor(v):
                key = (str(v.device), v.untyped_storage().data_ptr())
                if key not in seen:
                    seen.add(key)
                    total += v.untyped_storage().nbytes()
            elif isinstance(v, dict):
                for item in v.values(): add(item)
            elif isinstance(v, (list, tuple)):
                for item in v: add(item)
        add(tensors)
        cache = self.bank.cache_profile()['runtime_cache_bytes']
        return dict(persistent_tensor_bytes=total, bank_bytes=self.bank.feature_bytes,
                    pending_alternative_bank_bytes=self.pending['other'].numel() * 4 if self.pending else 0,
                    derived_pair_cache_bytes=cache, tensor_plus_pair_cache_bytes=total + cache,
                    accounting='unique persistent tensors incl saved episode; separate pair cache; excludes shared backbone/Python/allocator/transaction and IO peaks')
