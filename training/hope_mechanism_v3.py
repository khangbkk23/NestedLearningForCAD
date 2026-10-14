"""Post-selection normal-probe measurements. Never selects or repairs a method."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
import time
import torch
from torch.nn import functional as F
from models.hope_cad_v2.state import tensor_digest, finite_tensors
from models.hope_cad_v2.visual_memory import GuardedStep, stability_matched_step
from training.hope_outer_learning_v2 import capture_rng, restore_rng, unroll_support
from scripts.hope_cad_v2.diagnostics import visual_diagnostics


@torch.no_grad()
def score_probe(model, state, probe):
    rows, all_scores, predictions = [], [], []
    for identity, views in probe:
        n = views[0][0].shape[1]
        scores = torch.empty(n, device=views[0][0].device)
        covered = torch.zeros(n, device=scores.device, dtype=torch.int32)
        image_predictions = torch.empty_like(views[0][1][0])
        for features, target, mask in views:
            prediction = model.predict(features, state)
            residual = (prediction-target).square().sum(-1)[0]
            finite_tensors((prediction, residual), 'mechanism probe prediction')
            scores[mask] = residual[mask]
            image_predictions[mask] = prediction[0, mask]
            covered[mask] += 1
        if not bool((covered == 1).all()):
            raise ValueError('probe views must cover every patch exactly once')
        rows.append(dict(identity=identity, normal_loss=0.5*scores.double().mean().item(),
                         image_score=scores.max().item()))
        all_scores.append(scores.cpu())
        predictions.append(image_predictions.double().cpu())
    stacked = torch.stack(all_scores)
    prediction_stack = torch.stack(predictions)
    return dict(mean_normal_loss=sum(r['normal_loss'] for r in rows)/len(rows), images=rows,
                prediction_position_variance=prediction_stack.var(0, unbiased=False).mean().item(),
                prediction_spatial_variance=prediction_stack.var(1, unbiased=False).mean().item(),
                score_variance=stacked.double().var(unbiased=False).item()), stacked


def adapt_pool(model, data):
    state = model.cms.initial_state()
    if not model.dynamics.cms_updates:
        return state
    for role in ('meta_support', 'meta_query'):
        for identity in data.ids(role):
            x, target, mask = data.get(identity, state.clock % 4, role)
            with torch.no_grad():
                encoded = model.encode(x)
            state = model.cms.event(encoded, target, mask, model.decoder, state).state
    return state


@torch.no_grad()
def teacher_controls(data, probe):
    targets = [data.get(identity, 0, role)[1].double().cpu()[0]
               for role in ('meta_support', 'meta_query') for identity in data.ids(role)]
    targets = torch.stack(targets)
    position_mean = targets.mean(0)
    global_mean = targets.mean((0, 1))
    reports = {}
    for control in ('global_teacher_mean', 'position_teacher_mean', 'frozen_masked_features'):
        losses = []
        for identity, views in probe:
            total = 0.0
            for features, target, mask in views:
                prediction = (F.normalize(features, dim=-1, eps=1e-8) if control == 'frozen_masked_features'
                              else (global_mean if control == 'global_teacher_mean' else position_mean).to(target)[None])
                prediction = prediction.expand_as(target)
                total += (prediction[:, mask]-target[:, mask]).double().square().sum(-1).mean().item()/8
            losses.append(dict(identity=identity, normal_loss=total))
        reports[control] = dict(mean_normal_loss=sum(r['normal_loss'] for r in losses)/len(losses), images=losses)
    probe_targets = torch.stack([views[0][1].double().cpu()[0] for _, views in probe])
    reports['teacher_statistics'] = dict(train_unique_images=len(targets), probe_unique_images=len(probe),
        train_position_variance=targets.var(0, unbiased=False).mean().item(),
        probe_position_variance=probe_targets.var(0, unbiased=False).mean().item(),
        fitting_roles=['meta_support','meta_query'], probe_used_for_fitting=False)
    return reports


def mechanism_report(models, data, schedule, probe):
    if not probe:
        raise ValueError('normal probe is required after selection')
    before = {name: tensor_digest((dict(model.named_parameters()),)) for name, model in models.items()}
    rng = capture_rng()
    started = time.perf_counter()
    report = dict(schema='hope_mechanism_evidence_v3', reportable=False,
                  selection_feedback=False, official_test_accessed=False,
                  normal_probe_identities=[identity for identity, _ in probe],
                  hypotheses_status='normal-only diagnostics; CAD utility and H3 require evidence review',
                  methods={}, interventions={}, controls={})
    try:
        report['controls'] = teacher_controls(data, probe)
        selected_x, _, _ = data.get(data.ids('meta_support')[0], 0, 'meta_support')
        for name, model in models.items():
            initial = model.cms.initial_state()
            supported = (unroll_support(model, data, schedule['validation_support'], meta=False)
                         if model.dynamics.cms_updates else initial)
            stream = adapt_pool(model, data)
            measurements = {}
            for key, state in (('initial_frozen_history', initial), ('after_16_support', supported),
                               ('after_support_query_stream', stream), ('reset_history', model.cms.initial_state())):
                measurements[key], scores = score_probe(model, state, probe)
                if key == 'initial_frozen_history':
                    initial_scores = scores
                measurements[key]['score_delta_from_initial_rms'] = (scores-initial_scores).double().square().mean().sqrt().item()
                measurements[key]['clock'] = state.clock
                measurements[key]['commits'] = list(state.commits)
            measurements['selected_visual_diagnostics'] = visual_diagnostics(model, selected_x)
            report['methods'][name] = measurements
            if name == 'M':
                reference, reference_scores = score_probe(model, stream, probe)
                visual_off = deepcopy(model)
                visual_off.dynamics = replace(visual_off.dynamics, visual_updates=False)
                measured, scores = score_probe(visual_off, stream, probe)
                measured['score_delta_rms'] = (scores-reference_scores).double().square().mean().sqrt().item()
                report['interventions']['visual_updates_off_same_learned_weights'] = measured
                # A2 measurements are explicitly paired inference interventions,
                # not retrained ablations or alternative deployed architectures.
                single = deepcopy(model)
                with torch.no_grad():
                    single.visual.memory.fusion[1:].zero_()
                measured, scores = score_probe(single, stream, probe)
                measured.update(score_delta_rms=(scores-reference_scores).double().square().mean().sqrt().item(),
                                 policy='row-forward readout only, retain its learned scale; all routes still execute; no speed claim')
                report['interventions']['single_scan_readout'] = measured
                for mode in ('no_spectral_cap', 'no_injection_cap'):
                    def altered(key, delta, raw_eta, alpha):
                        original = stability_matched_step(key, delta, raw_eta, alpha)
                        if mode == 'no_injection_cap':
                            executed = torch.minimum(raw_eta, original.spectral_limit)
                        else:
                            safe = torch.where(original.injection_limit > 0, original.injection_limit, torch.ones_like(raw_eta))
                            executed = original.injection_limit * (-torch.expm1(-raw_eta/safe))
                        return GuardedStep(executed, original.injection_limit, original.spectral_limit)
                    try:
                        with patch('models.hope_cad_v2.visual_memory.stability_matched_step', side_effect=altered):
                            measured, scores = score_probe(model, stream, probe)
                        measured.update(status='finite', score_delta_rms=(scores-reference_scores).double().square().mean().sqrt().item())
                    except (ValueError, RuntimeError, FloatingPointError) as error:
                        measured = dict(status='unstable_intervention', error=str(error))
                    report['interventions'][mode] = measured
        after = {name: tensor_digest((dict(model.named_parameters()),)) for name, model in models.items()}
        if after != before:
            raise RuntimeError('mechanism diagnostics mutated selected learned parameters')
        report.update(status='complete_pending_scientific_review', engineering_passed=True,
                      mutable_state_isolated=True, utility_claim='inconclusive_pending_review',
                      required_review='Compare trained/static/mean controls and score interventions; no invented gain cutoff.')
    finally:
        restore_rng(rng)
    report['elapsed_seconds'] = time.perf_counter()-started
    report['feature_reads'] = dict(data.reads)
    return report
