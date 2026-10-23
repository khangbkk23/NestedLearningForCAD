"""Persistent fast readout/CMS with exact quotient-aware 4+4 slow replay."""
from copy import deepcopy
from types import SimpleNamespace
import math
import torch
from torch import nn
from torch.func import functional_call
from hope_cad.models.primitives.cms import CMSLevel
from hope_cad.models.primitives.visual_block import HOPECADVisualBlock, VisualBlockConfig
from hope_cad.models.primitives.static_readout import encode as static_encode
from hope_cad.models.primitives.state import finite_tensors
from hope_cad.contract import CONTRACT, design_digest
from hope_cad.models.validation import keys, integer, scalar, tensor, tree_finite, topology
from hope_cad.runtime.measurement import timed

RECORD_KEYS = ('tokens', 'stats', 'c', 'gamma', 'target', 'decision_d',
               'decision_identity', 'decision_cursor', 'Jminus', 'Jplus')


class UtilityModel(nn.Module):
    def __init__(self, *, dim=32, context_dim=32, stats_dim=16, height=28, width=28,
                 memory_dim=32, head_dim=8, self_modifying=True):
        super().__init__()
        self.config = dict(dim=dim, context_dim=context_dim, stats_dim=stats_dim,
                           height=height, width=width, memory_dim=memory_dim,
                           head_dim=head_dim, self_modifying=self_modifying)
        self.visual = HOPECADVisualBlock(VisualBlockConfig(context_dim, dim, memory_dim,
                                                         head_dim, height, width), row_retention=.9)
        self.geometry = nn.Linear(stats_dim, dim)
        self.fast = nn.ModuleDict(dict(cms=CMSLevel(dim, 2 * dim), readout=nn.Linear(dim + 1, 1)))
        self.slow = CMSLevel(dim, 2 * dim)
        self.self_modifying = self_modifying
        nn.init.xavier_uniform_(self.geometry.weight)
        nn.init.zeros_(self.geometry.bias)
        nn.init.zeros_(self.fast['readout'].weight)
        nn.init.zeros_(self.fast['readout'].bias)
        self.fast.requires_grad_(False)
        if not self_modifying:
            for name in ('initial_key', 'initial_value', 'initial_eta', 'initial_alpha'):
                getattr(self.visual.memory, name).requires_grad_(False)

    @timed('utility_encode')
    def encode(self, tokens, stats):
        cfg = self.config
        if tokens.shape != (1, cfg['height'] * cfg['width'], cfg['context_dim']) or stats.shape != (cfg['stats_dim'],):
            raise ValueError('one projected image and declared geometry stats required')
        h = self.visual(tokens) if self.self_modifying else static_encode(
            SimpleNamespace(visual=self.visual, dynamics=SimpleNamespace(visual_updates=False)), tokens)
        return h.mean(dim=1, keepdim=True) + self.geometry(stats).reshape(1, 1, -1)

    def evaluate(self, tokens, stats, phi, c, gamma):
        if torch.is_autocast_enabled(tokens.device.type):
            raise ValueError('utility path forbids autocast')
        scalar(c, 'causal scale', 1.)
        scalar(gamma, 'geometry prior')
        encoded = self.encode(tokens, stats)
        cms = {k.removeprefix('cms.'): v for k, v in phi.items() if k.startswith('cms.')}
        h = self.slow(functional_call(self.fast['cms'], cms, (encoded,), strict=True)).reshape(-1)
        logc = torch.tensor(c, dtype=torch.float64, device=h.device).log1p().reshape(1).to(h.dtype)
        u = torch.cat((h, logc))
        # Both factors are promoted BEFORE the dot. The same cast u defines d.
        prediction = gamma + (phi['readout.weight'].reshape(-1).double() * u.double()).sum() + phi['readout.bias'].double().reshape(())
        denominator = u.double().square().sum() + 1.
        finite_tensors((prediction, denominator), 'utility scalar forward')
        if float(denominator.detach()) < 1.:
            raise FloatingPointError('invalid context energy')
        return prediction, denominator

    def initial_fast(self):
        return {k: v.detach().clone().requires_grad_(True) for k, v in self.fast.named_parameters()}

    def slow_parameters(self):
        return [(k, v) for k, v in self.named_parameters() if v.requires_grad]


def prediction(model, phi, record):
    return model.evaluate(record['tokens'], record['stats'], phi, record['c'], record['gamma'])


def fast_step(model, phi, record, lr=.05, *, exact):
    pred, denom = prediction(model, phi, record)
    return fast_step_from_prediction(phi,record['target'],pred,denom,lr,exact=exact)


@timed('fast_gradient_and_step')
def fast_step_from_prediction(phi,target,pred,denom,lr=.05,*,exact):
    """Consume the same forward used by factual validation, retaining its graph."""
    loss = .5 * (pred - target).square()
    grads = torch.autograd.grad(loss, tuple(phi.values()), create_graph=exact)
    finite_tensors((loss, *grads), 'utility fast gradients')
    increments = [g.double() / denom for g in grads]
    if not exact:
        increments = [v.detach() for v in increments]
    updated = {k: (v.double() - lr * increment).to(v.dtype)
               for (k, v), increment in zip(phi.items(), increments)}
    finite_tensors(updated.values(), 'utility fast proposal')
    return updated, loss, denom, grads


def meta_loss(model, initial, records, *, exact=True, fast_lr=.05):
    if len(records) != 8:
        raise ValueError('exactly four support and four query records required')
    phi = {k: v.detach().clone().requires_grad_(True) for k, v in initial.items()}
    for record in records[:4]:
        phi, _, _, _ = fast_step(model, phi, record, fast_lr, exact=exact)
    return torch.stack([.5 * (prediction(model, phi, r)[0] - r['target']).square()
                        for r in records[4:]]).mean()


def norm(values):
    result = math.sqrt(sum(float(v.detach().double().square().sum()) for v in values))
    if not math.isfinite(result):
        raise FloatingPointError('nonfinite norm')
    return result


class NestedUtilityLearner:
    def __init__(self, model, mode='NESTED'):
        if mode not in ('NESTED', 'FIRST_ORDER', 'FAST_ONLY', 'NO_FAST_ADAPTATION'):
            raise ValueError('invalid learning policy')
        self.model, self.mode = model, mode
        self.phi = model.initial_fast()
        self.labels = self.fast_updates = self.slow_updates = 0
        self.records, self.block_initial, self.last_slow, self.saved_meta_episode = [], None, None, None
        self.optimizer = None if mode in ('FAST_ONLY', 'NO_FAST_ADAPTATION') else torch.optim.Adam(
            [p for _, p in model.slow_parameters()], lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0.)

    def validate_record(self, record):
        keys(record, RECORD_KEYS, 'utility record')
        cfg = self.model.config
        dtype = next(self.model.parameters()).dtype
        tensor(record['tokens'], (1, cfg['height'] * cfg['width'], cfg['context_dim']), dtype, 'replay tokens')
        tensor(record['stats'], (cfg['stats_dim'],), dtype, 'replay stats')
        for field in ('c', 'decision_d'):
            scalar(record[field], field, 1.)
        for field in ('gamma', 'target', 'Jminus', 'Jplus'):
            scalar(record[field], field, 0. if field.startswith('J') else None)
        if record['target'] != (record['Jminus'] - record['Jplus']) / record['c']:
            raise ValueError('delayed target not derived from stored causal scale')
        if not isinstance(record['decision_identity'], str) or not record['decision_identity']:
            raise ValueError('missing decision identity')
        integer(record['decision_cursor'], 'decision cursor')

    @torch.no_grad()
    def predict(self, tokens, stats, c, gamma):
        p, d = self.model.evaluate(tokens, stats, self.phi, c, gamma)
        return float(p), float(d)

    def observe(self, record):
        self.validate_record(record)
        if self.records and record['decision_cursor'] <= self.records[-1]['decision_cursor']:
            raise ValueError('nonchronological replay')
        record = {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in record.items()}
        if not self.records:
            self.block_initial = {k: v.detach().clone() for k, v in self.phi.items()}
        before, decision_d = prediction(self.model, self.phi, record)
        if float(decision_d.detach()) != record['decision_d']:
            raise ValueError('credited context no longer matches factual decision denominator')
        label = self.labels + 1
        details = dict(labels=label, denominator=float(decision_d.detach()), prediction_before=float(before.detach()))
        if self.mode != 'NO_FAST_ADAPTATION':
            previous = self.phi
            updated, loss, _, grads = fast_step(self.model, previous, record, exact=False)
            details.update(fast_gradient_norm=norm(grads), fast_step_norm=norm(updated[k] - previous[k] for k in previous),
                           fast_group_step_norms={prefix: norm(updated[k] - previous[k] for k in previous if k.startswith(prefix))
                                                  for prefix in ('cms.', 'readout.')})
            self.phi = {k: v.detach().requires_grad_(True) for k, v in updated.items()}
            self.fast_updates += 1
        else:
            loss = .5 * (before - record['target']).square()
        finite_tensors((loss,), 'actual fast loss')
        details['fast_loss'] = float(loss.detach())
        # Sample functional response before any slow commit; no extra gradient/update.
        if label in CONTRACT['diagnostics']['functional_response_completed_labels']:
            with torch.no_grad():
                after = float(prediction(self.model, self.phi, record)[0])
            error = abs(float(before.detach()) - record['target'])
            details['response'] = dict(prediction_after=after, small_error=error <= 1e-12,
                response_gain=None if error <= 1e-12 else (error - abs(after - record['target'])) / error)
        self.labels = label
        self.records.append(record)
        if len(self.records) == 8:
            episode_model = self._cpu(self.model.state_dict()) if self.labels == 40 else None
            if self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
                outer = meta_loss(self.model, self.block_initial, self.records, exact=self.mode == 'NESTED')
                finite_tensors((outer,), 'meta loss')
                outer.backward()
                params = self.model.slow_parameters()
                if any(p.grad is None for _, p in params):
                    raise RuntimeError('disconnected active slow parameter')
                finite_tensors((p.grad for _, p in params), 'slow gradients')
                norms = {name: norm([p.grad]) for name, p in params}
                total = torch.nn.utils.clip_grad_norm_([p for _, p in params], 1., error_if_nonfinite=True)
                previous = {name: p.detach().clone() for name, p in params}
                self.optimizer.step()
                tree_finite(self.optimizer.state_dict())
                finite_tensors((p for _, p in params), 'slow parameters')
                self.slow_updates += 1
                self.last_slow = dict(update=self.slow_updates, loss=float(outer.detach()), gradient_norm=float(total),
                    group_norms=norms, step_norms={name: norm([p - previous[name]]) for name, p in params},
                    support_ids=[r['decision_identity'] for r in self.records[:4]],
                    query_ids=[r['decision_identity'] for r in self.records[4:]])
                details['slow'] = deepcopy(self.last_slow)
            if self.labels == 40:
                self.saved_meta_episode = dict(labels_inclusive=[33, 40], phi_start=self._cpu(self.block_initial),
                                               records=self._cpu(self.records), model=episode_model, mode=self.mode)
            self.records, self.block_initial = [], None
        details.update(fast_updates=self.fast_updates, slow_updates=self.slow_updates,
                       completed_label_blocks=self.labels // 8)
        return details

    @staticmethod
    def _cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: NestedUtilityLearner._cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [NestedUtilityLearner._cpu(v) for v in value]
        return deepcopy(value)

    def state_dict(self):
        return self._cpu(dict(schema='hope_support_learner_v5_1', design_digest=design_digest(),
            mode=self.mode, model=self.model.state_dict(), phi=self.phi,
            optimizer=self.optimizer.state_dict() if self.optimizer else None, labels=self.labels,
            fast_updates=self.fast_updates, slow_updates=self.slow_updates, records=self.records,
            block_initial=self.block_initial, last_slow=self.last_slow, saved_meta_episode=self.saved_meta_episode))

    def _load_validated(self, state):
        template = self.state_dict()
        keys(state, template, 'learner state')
        for field in ('schema', 'design_digest', 'mode'):
            if state[field] != template[field]:
                raise ValueError('learner identity changed')
        tree_finite(state)
        for name in ('labels', 'fast_updates', 'slow_updates'):
            integer(state[name], name)
        expected_fast = 0 if self.mode == 'NO_FAST_ADAPTATION' else state['labels']
        expected_slow = state['labels'] // 8 if self.optimizer else 0
        if state['fast_updates'] != expected_fast or state['slow_updates'] != expected_slow:
            raise ValueError('actual optimizer counters inconsistent')
        keys(state['model'], template['model'], 'model tensors')
        keys(state['phi'], template['phi'], 'fast tensors')
        for group in ('model', 'phi'):
            for name, value in state[group].items():
                topology(value, template[group][name], group + name)
                if group == 'model' and name.startswith('fast.') and not torch.equal(value.cpu(), template[group][name].cpu()):
                    raise ValueError('frozen fast initialization template changed')
        if not isinstance(state['records'], list) or len(state['records']) != state['labels'] % 8:
            raise ValueError('corrupt replay clock')
        if bool(state['records']) != (state['block_initial'] is not None):
            raise ValueError('partial block initial missing or fabricated')
        if state['block_initial'] is not None:
            keys(state['block_initial'], template['phi'], 'phi_start')
            for name, value in state['block_initial'].items():
                tensor(value, template['phi'][name].shape, template['phi'][name].dtype, name)
        for r in state['records']:
            self.validate_record(r)
        cursors = [r['decision_cursor'] for r in state['records']]
        ids = [r['decision_identity'] for r in state['records']]
        if cursors != sorted(set(cursors)) or len(set(ids)) != len(ids):
            raise ValueError('nonunique/nonchronological replay')
        if expected_slow:
            keys(state['last_slow'], ('update', 'loss', 'gradient_norm', 'group_norms', 'step_norms', 'support_ids', 'query_ids'), 'last slow trace')
            if state['last_slow']['update'] != expected_slow:
                raise ValueError('slow trace clock changed')
            for field in ('loss','gradient_norm'):
                scalar(state['last_slow'][field],field,0.)
            for field in ('group_norms','step_norms'):
                keys(state['last_slow'][field],dict(self.model.slow_parameters()),field)
                for v in state['last_slow'][field].values():scalar(v,field,0.)
            all_ids=state['last_slow']['support_ids']+state['last_slow']['query_ids']
            if len(state['last_slow']['support_ids'])!=4 or len(state['last_slow']['query_ids'])!=4 or len(set(all_ids))!=8 or any(not isinstance(v,str) or not v for v in all_ids):
                raise ValueError('invalid slow episode IDs')
        elif state['last_slow'] is not None:
            raise ValueError('fabricated slow trace')
        episode = state['saved_meta_episode']
        if (episode is not None) != (state['labels'] >= 40):
            raise ValueError('saved episode clock changed')
        if episode is not None:
            keys(episode, ('labels_inclusive', 'phi_start', 'records', 'model', 'mode'), 'saved episode')
            if episode['labels_inclusive'] != [33, 40] or episode['mode'] != self.mode or len(episode['records']) != 8:
                raise ValueError('saved episode identity changed')
            for r in episode['records']:
                self.validate_record(r)
            for group, expected in [('model', template['model']), ('phi_start', template['phi'])]:
                keys(episode[group], expected, 'saved episode ' + group)
                for k, v in episode[group].items():
                    topology(v, expected[k], k)
        self._validate_optimizer(state['optimizer'], expected_slow)
        device = next(self.model.parameters()).device
        self.model.load_state_dict(state['model'], strict=True)
        self.phi = {k: v.to(device).detach().clone().requires_grad_(True) for k, v in state['phi'].items()}
        if self.optimizer:
            self.optimizer.load_state_dict(deepcopy(state['optimizer']))
        self.labels, self.fast_updates, self.slow_updates = (state[k] for k in ('labels', 'fast_updates', 'slow_updates'))
        self.records = [dict(r, tokens=r['tokens'].to(device).detach().clone(), stats=r['stats'].to(device).detach().clone()) for r in state['records']]
        self.block_initial = {k: v.to(device).detach().clone() for k, v in state['block_initial'].items()} if state['block_initial'] is not None else None
        self.last_slow = deepcopy(state['last_slow'])
        self.saved_meta_episode = deepcopy(state['saved_meta_episode'])

    def _validate_optimizer(self, value, count):
        if self.optimizer is None:
            if value is not None:
                raise ValueError('inactive optimizer must be absent')
            return
        keys(value, ('state', 'param_groups'), 'Adam')
        expected = self.optimizer.state_dict()['param_groups']
        if value['param_groups'] != expected:
            raise ValueError('Adam parameter topology/hyperparameters changed')
        params = [p for _, p in self.model.slow_parameters()]
        indices = expected[0]['params']
        if set(value['state']) != (set(indices) if count else set()):
            raise ValueError('Adam slots/count mismatch')
        for idx, parameter in zip(indices, params):
            if count:
                item = value['state'][idx]
                keys(item, ('step', 'exp_avg', 'exp_avg_sq'), 'Adam slot')
                tensor(item['step'], (), torch.float32, 'Adam step')
                if float(item['step']) != count:
                    raise ValueError('Adam step clock inconsistent')
                for key in ('exp_avg', 'exp_avg_sq'):
                    tensor(item[key], parameter.shape, parameter.dtype, 'Adam ' + key)
                if bool((item['exp_avg_sq'] < 0).any()):
                    raise ValueError('negative Adam second moment')

    def load_state_dict(self, state):
        # Never partially mutate the live owner on a malformed checkpoint.
        other = deepcopy(self)
        other._load_validated(state)
        self.__dict__.update(other.__dict__)
