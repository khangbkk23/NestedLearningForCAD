"""Warm exact-16 meta-training and joint normal-validation selection under v4.

Feature providers accept (identity, mask_view), returning detached masked-ViT
features, detached unit teacher targets and the token mask. They must be built
from role-scoped manifests. This module never opens official test data.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
import random
import time
import numpy as np
import torch
from dataset.benchmark_manifest_v1 import canonical_digest
from dataset.benchmark_protocol_v2 import ROLES
from models.hope_cad_v2.cms import masked_teacher_loss
from models.hope_cad_v2.state import finite_tensors, tensor_digest

SHORT_STEPS = (0, 25, 50, 100, 200, 300, 400)
BETA_GRID = (1e-4, 1e-3, 1e-2)
STATIC_LR_GRID = (3e-5, 1e-4, 3e-4)


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state().clone(),
                cuda=[v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state['cuda']])


class PrefixFeatures:
    """Capability boundary: each read declares its role; probe is inaccessible here."""
    def __init__(self, role_manifest, provider):
        expected = canonical_digest({k: v for k, v in role_manifest.items() if k != 'digest'})
        if role_manifest.get('schema') != 'hope_prefix_roles_v3' or role_manifest.get('digest') != expected:
            raise ValueError('invalid role-scoped manifest')
        self.manifest = deepcopy(role_manifest)
        roles = self.manifest['roles']
        if set(roles) != set(ROLES):
            raise ValueError('invalid calibration roles')
        all_ids = [r['relative_path'] for role in ROLES for r in roles[role]]
        if len(set(all_ids)) != len(all_ids) or not 32 <= len(all_ids) <= 100:
            raise ValueError('invalid prefix size/role overlap')
        p = len(all_ids)
        sizes = (p // 2, p // 5, 15*p // 100)
        if [len(roles[r]) for r in ROLES] != [*sizes, p-sum(sizes)]:
            raise ValueError('invalid prefix split counts')
        if all_ids != sorted(all_ids) or any(r['task_name'] != self.manifest['first_task'] or r['task_id'] != 0 or r['split'] != 'train'
                                           for role in ROLES for r in roles[role]):
            raise ValueError('calibration may only see canonical first-task normal prefix')
        self.provider = provider
        self.reads = {r: 0 for r in ROLES}

    def ids(self, role):
        if role not in ROLES[:3]:
            raise ValueError('normal_probe is inaccessible before selection')
        return tuple(r['relative_path'] for r in self.manifest['roles'][role])

    def get(self, identity, view, role):
        if identity not in self.ids(role) or type(view) is not int or not 0 <= view < 4:
            raise ValueError('forbidden identity/role/view read')
        features, target, mask = self.provider(identity, view)
        if features.requires_grad or target.requires_grad:
            raise ValueError('frozen feature provider must detach outputs')
        finite_tensors((features, target), 'frozen features')
        self.reads[role] += 1
        return features, target, mask


def make_schedule(data, seed, steps=400):
    rng = random.Random(seed)
    episodes = [dict(support=rng.sample(data.ids('meta_support'), 16),
                     query=rng.choice(data.ids('meta_query')), query_view=i % 4) for i in range(steps)]
    # Separate fixed schedule; never depends on beta, checkpoint or anomaly metrics.
    validation_support = random.Random(seed + 104729).sample(data.ids('meta_support'), 16)
    result = dict(seed=seed, episodes=episodes, validation_support=validation_support,
                  roles_digest=data.manifest['digest'])
    result['digest'] = canonical_digest(result)
    return result


def validate_schedule(schedule, data):
    if (schedule['digest'] != canonical_digest({k: v for k, v in schedule.items() if k != 'digest'})
            or schedule['roles_digest'] != data.manifest['digest']):
        raise ValueError('episode schedule identity mismatch')
    supports = [e['support'] for e in schedule['episodes']] + [schedule['validation_support']]
    for identities in supports:
        if len(identities) != 16 or len(set(identities)) != 16 or not set(identities) <= set(data.ids('meta_support')):
            raise ValueError('support must contain 16 distinct permitted images')
    for i, episode in enumerate(schedule['episodes']):
        if episode['query'] not in data.ids('meta_query') or episode['query_view'] != i % 4:
            raise ValueError('query schedule mismatch')


def unroll_support(model, data, identities, *, meta):
    if len(identities) != 16 or len(set(identities)) != 16:
        raise ValueError('exactly 16 distinct support images required')
    state = model.cms.initial_state(meta=meta)
    for t, identity in enumerate(identities):
        features, target, mask = data.get(identity, t % 4, 'meta_support')
        if meta:
            x = model.encode(features)
        else:
            with torch.no_grad():
                x = model.encode(features)
        state = model.cms.event(x, target, mask, model.decoder, state, meta=meta).state
    expected = tuple(16 // p for p in model.cms.periods)
    if state.commits != expected or state.clock != 16:
        raise RuntimeError('exact unroll clock mismatch')
    return state


def episode_loss(model, data, episode):
    if model.dynamics.cms_updates:
        initial = model.cms.initial_state(meta=True)
        state = unroll_support(model, data, episode['support'], meta=True)
        features, target, mask = data.get(episode['query'], episode['query_view'], 'meta_query')
        encoded = model.encode(features)
        cold = masked_teacher_loss(model.decoder(model.cms(encoded, initial)[-1]), target, mask)
        adapted = masked_teacher_loss(model.decoder(model.cms(encoded, state)[-1]), target, mask)
        return (cold + adapted) / 2
    state = model.cms.initial_state(meta=True)
    losses = []
    for t, identity in enumerate(episode['support']):
        features, target, mask = data.get(identity, t % 4, 'meta_support')
        losses.append(masked_teacher_loss(model.predict(features, state), target, mask))
    features, target, mask = data.get(episode['query'], episode['query_view'], 'meta_query')
    losses.append(masked_teacher_loss(model.predict(features, state), target, mask))
    return torch.stack(losses).mean()


@torch.no_grad()
def validation_loss(model, data, state):
    losses = []
    for identity in data.ids('normal_validation'):
        for view in range(4):
            features, target, mask = data.get(identity, view, 'normal_validation')
            loss = masked_teacher_loss(model.predict(features, state), target, mask)
            finite_tensors((loss,), 'normal validation')
            losses.append(loss.double())
    return torch.stack(losses).mean().item()


def short_validation(model, data, support):
    return short_validation_components(model, data, support)['loss']


def short_validation_components(model, data, support):
    rng = capture_rng()
    try:
        loss0 = validation_loss(model, data, model.cms.initial_state(meta=False))
        state = (unroll_support(model, data, support, meta=False) if model.dynamics.cms_updates
                 else model.cms.initial_state(meta=False))
        loss16 = validation_loss(model, data, state) if model.dynamics.cms_updates else loss0
        return dict(loss=(loss0+loss16)/2, loss_at_0=loss0, loss_at_16=loss16)
    finally:
        restore_rng(rng)


def model_weights(model):
    return deepcopy(model.state_dict())


def fit(model, data, schedule, *, outer_lr=1e-4, max_steps=400, checkpoints=SHORT_STEPS,
        smoke=False, resume=None, checkpoint_callback=None, diagnostic_callback=None,
        failure_callback=None):
    """Fit fails as a whole on any nonfinite; no rescue of an earlier candidate.

    checkpoint_callback receives exact optimizer/RNG/schedule state after every
    completed step (including zero); it can persist without changing selection.
    """
    validate_schedule(schedule, data)
    if model.metadata()['design_id'] != 'hope_cad_v4_warm_joint_cms_row_retention':
        raise ValueError('v4 fit cannot use a v3 predictor')
    if not smoke and (max_steps != 400 or tuple(checkpoints) != SHORT_STEPS):
        raise ValueError('scientific selection requires frozen 400-step/checkpoint recipe')
    if max_steps > len(schedule['episodes']) or max_steps < 0:
        raise ValueError('insufficient episode schedule')
    if model.dynamics.method in ('C0', 'C1'):
        if outer_lr not in STATIC_LR_GRID:
            raise ValueError('static control LR outside frozen grid')
    elif outer_lr != 1e-4:
        raise ValueError('exact-meta outer LR is fixed')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=outer_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    result = dict(status='running', reportable=False, smoke=smoke, method=model.dynamics.method,
                  beta=model.cms.rates[0], outer_lr=outer_lr, schedule_digest=schedule['digest'],
                  role_digest=data.manifest['digest'], short_losses=[], training=[], shortlist=None,
                  first_failure=None, model_metadata=model.metadata())
    start = 0
    reads_at_start = dict(data.reads)
    elapsed_prior = 0.0
    if resume is not None:
        if (resume.get('schema') != 'hope_outer_checkpoint_v4'
                or resume['schedule'] != schedule or resume['metadata'] != model.metadata()
                or resume['outer_lr'] != outer_lr or resume['max_steps'] != max_steps
                or tuple(resume['checkpoints']) != tuple(checkpoints) or resume['smoke'] != smoke):
            raise ValueError('calibration resume contract mismatch')
        model.load_state_dict(resume['weights'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        result, start = deepcopy(resume['result']), resume['step'] + 1
        data.reads = dict(resume['feature_reads_total'])
        reads_at_start = dict(resume['feature_reads_at_start'])
        elapsed_prior = resume['elapsed_seconds']
        restore_rng(resume['rng'])
    started = time.perf_counter()
    step = start
    try:
        for step in range(start, max_steps + 1):
            if step:
                optimizer.zero_grad(set_to_none=True)
                loss = episode_loss(model, data, schedule['episodes'][step - 1])
                finite_tensors((loss,), 'outer loss')
                loss.backward()
                missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
                if missing:
                    raise RuntimeError(f'disconnected trainable groups: {missing}')
                finite_tensors((p.grad for p in params), 'outer gradient')
                norms = {name: p.grad.detach().double().norm().item() for name, p in model.named_parameters() if p.requires_grad}
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                post_clip_norm = torch.stack([p.grad.detach().double().square().sum() for p in params]).sum().sqrt().item()
                optimizer.step()
                finite_tensors(params, 'outer parameter proposal')
                finite_tensors((value for state in optimizer.state.values() for value in state.values()
                                if isinstance(value, torch.Tensor)), 'outer optimizer state')
                result['training'].append(dict(step=step, loss=loss.item(), gradient_norm=float(norm),
                                              post_clip_gradient_norm=post_clip_norm, group_norms=norms))
            if diagnostic_callback is not None and step in (0, 1, 5, 25):
                diagnostic_callback(step, model)
            if step in checkpoints:
                measured = short_validation_components(model, data, schedule['validation_support'])
                value = measured['loss']
                result['short_losses'].append(dict(step=step, **measured))
                old = result['shortlist']
                if old is None or (value, step) < (old['loss'], old['step']):
                    result['shortlist'] = dict(step=step, **measured, weights=model_weights(model))
            if checkpoint_callback is not None:
                checkpoint_callback(dict(schema='hope_outer_checkpoint_v4', step=step, schedule=deepcopy(schedule),
                                         metadata=model.metadata(), outer_lr=outer_lr, max_steps=max_steps,
                                         checkpoints=list(checkpoints), smoke=smoke, weights=model_weights(model),
                                         optimizer=deepcopy(optimizer.state_dict()), rng=capture_rng(), result=deepcopy(result),
                                         feature_reads_total=dict(data.reads),feature_reads_at_start=reads_at_start,
                                         elapsed_seconds=elapsed_prior+time.perf_counter()-started))
        if result['shortlist'] is None:
            raise ValueError('no short-validation checkpoint')
        result['status'] = 'fit_complete'
    except (FloatingPointError, RuntimeError) as error:
        result['status'] = 'fit_failed'
        result['shortlist'] = None
        result['first_failure'] = dict(step=step, reason=str(error))
    result['elapsed_seconds'] = elapsed_prior + time.perf_counter() - started
    result['feature_reads'] = {k:data.reads[k]-reads_at_start[k] for k in data.reads}
    if result['status'] == 'fit_failed' and failure_callback is not None:
        failure_callback(deepcopy(result))
    return result


def select_candidate(candidates):
    eligible = [c for c in candidates if c['status'] == 'fit_complete' and c.get('gate', {}).get('passed')]
    if not eligible:
        return dict(status='selection_failed', selected=None)
    selected = min(eligible, key=lambda c: (c['shortlist']['loss'], c['shortlist']['step'], c['beta']))
    return dict(status='selected', selected=selected)


def export_initialization(model, provenance):
    """Export learned initials, never support-terminal or stress-terminal states."""
    required = ('roles_digest', 'schedule_digest', 'source_sha256', 'spec_sha256', 'feature_identity')
    if any(not provenance.get(k) for k in required):
        raise ValueError('export requires complete source/spec/data/feature provenance')
    state = model.cms.initial_state(meta=False)
    return dict(schema='hope_deployment_initialization_v4', metadata=model.metadata(),
                slow_and_initial_weights=model_weights(model), cms=state.checkpoint(),
                initial_cms_sha256=tensor_digest(state.weights), cursor=0, rng=capture_rng(),
                provenance=deepcopy(provenance), selection_rule_id='prefix_warm_joint_stress_v4')
