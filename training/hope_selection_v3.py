"""Finite candidate orchestration; C4 inherits M beta and never searches its own."""
from copy import deepcopy
import random
import numpy as np
import torch
from .hope_outer_learning_v2 import (BETA_GRID, STATIC_LR_GRID, capture_rng, restore_rng,
                                     fit, select_candidate)
from .hope_cms_stress_v3 import run_stress_gate


def search(model_factory, data, schedule, *, method='M', seed=0, selected_m=None,
           smoke=False, fit_options=None):
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
            # Common random numbers: same initialization and materialized episodes.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = model_factory(method, 1e-3 if static else value,
                                  value if method == 'C4' else None)
            candidate = fit(model, data, schedule, outer_lr=value if static else 1e-4,
                            smoke=smoke, **(fit_options or {}))
            if candidate['status'] == 'fit_complete':
                model.load_state_dict(candidate['shortlist']['weights'], strict=True)
                candidate['gate'] = (dict(passed=True, status='not_applicable_static') if static
                                     else run_stress_gate(model, data))
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
