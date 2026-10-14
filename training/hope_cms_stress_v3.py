"""Preregistered 1024-event prefix-only screen. No retuning and no state rescue."""
import math
import time
import torch
from dataset.benchmark_manifest_v1 import canonical_digest
from models.hope_cad_v2.state import tensor_digest, vector_norm
from .hope_outer_learning_v2 import capture_rng, restore_rng, validation_loss

CONSTANTS = dict(horizon=1024, checkpoints=[0, 16, 64, 128, 256, 512, 1024],
                 max_weight_ratio=10.0, max_update_ratio=1.0,
                 max_loss_ratio=2.0, terminal_loss_ratio=1.25,
                 loss_floor=1e-8, weight_floor=1e-12)


def stress_schedule(data):
    pool = [(identity, role) for role in ('meta_support', 'meta_query') for identity in data.ids(role)]
    return [dict(identity=pool[t % len(pool)][0], role=pool[t % len(pool)][1], view=t % 4)
            for t in range(CONSTANTS['horizon'])]


def run_stress_gate(model, data, *, resume=None, checkpoint_callback=None, failure_callback=None):
    if not model.dynamics.cms_updates:
        raise ValueError('static C0/C1 do not receive a CMS stress gate')
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError('stress reference update precision is FP32')
    rng = capture_rng()
    started = time.perf_counter()
    state = model.cms.initial_state(meta=False)
    initial_hash = tensor_digest(state.weights)
    denominators = [max(vector_norm(w.values()), CONSTANTS['weight_floor']) for w in state.weights]
    schedule = stress_schedule(data)
    reads_before = dict(data.reads)
    report = dict(selection_rule_id='prefix_cms_stress_v3', constants=CONSTANTS.copy(),
                  model_metadata=model.metadata(),
                  schedule=schedule, schedule_digest=canonical_digest(schedule),
                  roles_digest=data.manifest['digest'], initial_cms_sha256=initial_hash,
                  passed=False, status='running', trace=[], losses={}, first_failure=None)
    prior_reads = {k: 0 for k in data.reads}
    elapsed_prior = 0.0
    if resume is not None:
        from training.hope_initialization_v3 import weights_digest
        if (resume.get('schema') != 'hope_stress_resume_v3'
                or resume['model_sha256'] != weights_digest(model.state_dict())
                or resume['report']['schedule_digest'] != report['schedule_digest']
                or resume['report']['roles_digest'] != report['roles_digest']
                or resume['report']['model_metadata'] != model.metadata()
                or resume['report']['constants'] != CONSTANTS):
            raise ValueError('stress resume identity mismatch')
        state = model.cms.load_state(resume['state'])
        from copy import deepcopy
        report = deepcopy(resume['report'])
        if len(report['trace']) != state.clock or not 0 <= state.clock <= CONSTANTS['horizon']:
            raise ValueError('stress resume cursor/trace mismatch')
        prior_reads, elapsed_prior = resume['feature_reads'], resume['elapsed_seconds']
    event = state.clock
    def checkpoint():
        if checkpoint_callback:
            from copy import deepcopy
            from training.hope_initialization_v3 import weights_digest
            checkpoint_callback(dict(schema='hope_stress_resume_v3', state=state.checkpoint(),
                model_sha256=weights_digest(model.state_dict()), report=deepcopy(report),
                elapsed_seconds=elapsed_prior+time.perf_counter()-started,
                feature_reads={k: prior_reads[k]+data.reads[k]-reads_before[k] for k in data.reads}))
    try:
        if not state.clock:
            report['losses'][0] = validation_loss(model, data, state)
            checkpoint()
        for event, row in enumerate(schedule[state.clock:], start=state.clock+1):
            features, target, mask = data.get(row['identity'], row['view'], row['role'])
            with torch.no_grad():
                x = model.encode(features)
            result = model.cms.event(x, target, mask, model.decoder, state, meta=False)
            # Diagnostics see the pre-reset accumulated gradient, including non-due levels.
            ratios = [dict(**d, R_w=d['weight_norm']/den, R_a=d['update_norm']/den)
                      for d, den in zip(result.diagnostics, denominators)]
            state = result.state
            report['trace'].append(dict(event=event, levels=ratios, commits=list(state.commits),
                                        pending=list(state.pending)))
            if any(not math.isfinite(d['R_w']) or not math.isfinite(d['R_a'])
                   or d['R_w'] > 10 or d['R_a'] > 1 for d in ratios):
                raise FloatingPointError('CMS weight/update ratio threshold failed')
            if event in CONSTANTS['checkpoints']:
                value = validation_loss(model, data, state)
                report['losses'][event] = value
                D = max(report['losses'][0], report['losses'][16], 1e-8)
                E = max(report['losses'][16], 1e-8)
                if value > 2 * D or (event == 1024 and value > 1.25 * E):
                    raise FloatingPointError('normal-validation drift threshold failed')
            if event % 16 == 0:
                checkpoint()
        report['passed'], report['status'] = True, 'passed'
    except (FloatingPointError, RuntimeError) as error:
        report['status'] = 'failed'
        report['first_failure'] = dict(event=event, reason=str(error))
    finally:
        restore_rng(rng)
    report.update(events_completed=state.clock, commits=list(state.commits),
                  elapsed_seconds=elapsed_prior+time.perf_counter()-started,
                  feature_reads={k: prior_reads[k]+data.reads[k]-reads_before[k] for k in data.reads},
                  source_initial_cms_sha256_after=tensor_digest(model.cms.initial_state().weights))
    if report['source_initial_cms_sha256_after'] != initial_hash:
        raise RuntimeError('stress mutated learned initialization')
    if not report['passed'] and failure_callback is not None:
        from copy import deepcopy
        failure_callback(deepcopy(report))
    return report
