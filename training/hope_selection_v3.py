"""Finite candidate orchestration; C4 inherits M beta and never searches its own."""
from copy import deepcopy
import random
import numpy as np
import torch
from .hope_outer_learning_v2 import (BETA_GRID, STATIC_LR_GRID, capture_rng, restore_rng,
                                     fit, select_candidate)
from .hope_cms_stress_v3 import run_stress_gate


def search(model_factory, data, schedule, *, method='M', seed=0, selected_m=None,
           smoke=False, fit_options=None, journal=None):
    if seed != 0:
        raise ValueError('hyperparameter selection is seed 0 only; final seeds refit a locked recipe')
    if method not in ('M', 'C0', 'C1', 'C2', 'C3', 'C4'):
        raise ValueError('method has no registered independent selection recipe')
    if method == 'C4' and (not selected_m or selected_m.get('status') != 'selected'
                           or selected_m['selected']['method'] != 'M'):
        raise ValueError('C4 selection requires a successful M selection')
    if method == 'C4' and (selected_m.get('smoke') != smoke or selected_m.get('common_seed') != seed
                           or selected_m.get('schedule_digest') != schedule['digest']
                           or selected_m['selected'].get('role_digest') != data.manifest['digest']):
        raise ValueError('C4 must inherit M from the same real/smoke prefix, seed and schedule')
    static = method in ('C0', 'C1')
    values = STATIC_LR_GRID if static else (BETA_GRID if method != 'C4' else (selected_m['selected']['beta'],))
    candidates, rng = [], capture_rng()
    try:
        for value in values:
            beta, lr = (1e-3 if static else value), (value if static else 1e-4)
            saved = journal.load_candidate(method, beta, lr) if journal else None
            if saved is not None:
                candidates.append(saved)
                continue
            # Common random numbers: same initialization and materialized episodes.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = model_factory(method, beta,
                                  value if method == 'C4' else None)
            options = dict(fit_options or {})
            if journal:
                options.update(journal.fit_options(method, beta, lr))
                options['failure_callback'] = lambda result: journal.save_candidate(method, beta, lr, result)
            candidate = fit(model, data, schedule, outer_lr=lr, smoke=smoke, **options)
            if candidate['status'] == 'fit_complete':
                model.load_state_dict(candidate['shortlist']['weights'], strict=True)
                gate_options = journal.gate_options(method, beta, lr) if journal else {}
                if journal:
                    gate_options['failure_callback'] = lambda report: journal.save_candidate(
                        method, beta, lr, {**candidate, 'gate': report})
                candidate['gate'] = (dict(passed=True, status='not_applicable_static') if static
                                     else run_stress_gate(model, data, **gate_options))
            if journal:
                journal.save_candidate(method, beta, lr, candidate)
            candidates.append(candidate)
        if static:
            eligible = [c for c in candidates if c['status'] == 'fit_complete']
            # Frozen spec does not distinguish exact equal-loss/equal-step LR ties;
            # stable registered grid order is deterministic and is recorded here.
            selected = min(eligible, key=lambda c: (c['shortlist']['loss'], c['shortlist']['step'])) if eligible else None
            outcome = dict(status='selected' if selected else 'selection_failed', selected=selected)
        else:
            outcome = select_candidate(candidates)
        return dict(**outcome, selection_rule_id='prefix_cms_stress_v3', candidates=candidates,
                    common_seed=seed, schedule_digest=schedule['digest'], smoke=smoke,
                    baseline_anomaly_feedback_used=False)
    finally:
        restore_rng(rng)
