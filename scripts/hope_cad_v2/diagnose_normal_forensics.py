"""Bounded post-selection normal-only forensics. No fitting or method selection.

The diagnostic protocol is a separate preregistration made after observing W6.
This runner never changes the frozen v3 algorithm or its saved initializations.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from dataset.benchmark_protocol_v2 import file_hash
from dataset.hope_mask_views import patch_masks
from models.hope_cad_v2.cms import masked_teacher_loss
from models.hope_cad_v2.state import tensor_digest
from training.hope_feature_contract_v3 import validate_bundle
from training.hope_initialization_v3 import predictor_from_initialization, validate_initialization
from training.hope_mechanism_v3 import score_probe
from training.hope_runtime_v3 import atomic_json, configure_runtime, resource_usage, runtime_identity

PROTOCOL = ROOT / 'docs/main_spec/normal_only_diagnostic_protocol_v1.json'


def digest_stream(stream):
    digest = hashlib.sha256()
    while chunk := stream.read(8 * 1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def verify_archive(archive, expected):
    if file_hash(archive) != expected:
        raise ValueError('unexpected W6 evidence bytes')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        manifest = json.loads(z.read('artifact_manifest.json'))['files']
        if len(names) != len(set(names)) or set(names) != {*manifest, 'artifact_manifest.json'}:
            raise ValueError('evidence manifest membership mismatch')
        for name, digest in manifest.items():
            with z.open(name) as stream:
                if digest_stream(stream) != digest:
                    raise ValueError('evidence hash mismatch: ' + name)
        source = json.loads(z.read('pipeline_lock.json'))['source_sha256']
        if any(not (ROOT / p).is_file() or file_hash(ROOT / p) != digest for p, digest in source.items()):
            raise ValueError('original source/spec has changed')
    return source


def load_tensor(z, name):
    with z.open(name) as stream:
        return torch.load(io.BytesIO(stream.read()), map_location='cpu', weights_only=False)


def context_from_archive(z):
    return dict(method=yaml.safe_load(z.read('baseline_lock/method_resolved.yaml')),
                protocol=yaml.safe_load(z.read('baseline_lock/protocol_resolved.yaml')),
                manifest=json.loads(z.read('baseline_lock/manifest_train.json')),
                roles=json.loads(z.read('baseline_lock/calibration_roles.json')),
                lock=json.loads(z.read('baseline_lock/baseline_lock.json')))


def retained_level(previous, proposed, level):
    """Freeze only applied weights; retain the original event schedule exactly."""
    weights = tuple(previous.weights[i] if i == level else proposed.weights[i]
                    for i in range(len(proposed.weights)))
    return replace(proposed, weights=weights).detached()


def gradient_alignment(model, state, features, target, mask):
    """No outer optimizer; autograd.grad does not populate saved parameter grads."""
    with torch.no_grad():
        x = model.encode(features)
    working = state.detached()
    outputs = model.cms(x, working)
    predictions = [model.decoder(h) for h in outputs]
    losses = [masked_teacher_loss(p, target, mask) for p in predictions]
    first = tuple(working.weights[0].values())
    local = torch.autograd.grad(losses[0], first, retain_graph=True)
    final = torch.autograd.grad(losses[-1], first, retain_graph=True)
    dot = sum((a.double() * b.double()).sum().item() for a, b in zip(local, final))
    norm_local = sum(a.double().square().sum().item() for a in local) ** .5
    norm_final = sum(a.double().square().sum().item() for a in final) ** .5
    event = model.cms.event(x, target, mask, model.decoder, state, meta=False)
    with torch.no_grad():
        after = masked_teacher_loss(model.predict(features, event.state), target, mask).item()
    return dict(L_local_first=losses[0].item(), L_final=losses[-1].item(),
                cosine=dot / max(norm_local * norm_final, 1e-30), dot=dot,
                gradient_norm_local=norm_local, gradient_norm_final=norm_final,
                hypothetical_final_loss=after, hypothetical_delta=after-losses[-1].item(),
                prediction_norm_all_patch_mean=predictions[-1].detach().norm(dim=-1).mean().item(),
                teacher_norm_all_patch_mean=target.norm(dim=-1).mean().item(),
                encoded_norm_all_patch_mean=x.norm(dim=-1).mean().item())


@torch.no_grad()
def spatial_readout(model, features):
    v, c = model.visual, model.visual.config
    projected = v.projection(features)
    z = v.in_projection(v.input_norm(projected)).transpose(1, 2).reshape(1, c.memory_dim, c.height, c.width)
    z = v.depthwise(z).flatten(2).transpose(1, 2)
    inspected = v.memory.inspect(z)
    rms = inspected.directional_outputs.double().square().mean(-1).sqrt()[0]
    grid = rms.reshape(4, c.height, c.width)
    border = torch.zeros((c.height, c.width), dtype=torch.bool, device=features.device)
    border[:2] = True
    border[-2:] = True
    border[:, :2] = True
    border[:, -2:] = True
    center = grid[:, 12:16, 12:16].mean((1, 2))
    edge = grid[:, border].mean(-1)
    branch = v.out_projection(inspected.output).double().square().mean().sqrt().item()
    base = projected.double().square().mean().sqrt().item()
    chunks = inspected.scan_groups[0][1]
    return dict(center_rms_by_direction=center.cpu().tolist(), border_rms_by_direction=edge.cpu().tolist(),
                center_to_border_ratio_by_direction=(center/edge.clamp_min(1e-30)).cpu().tolist(),
                directional_rms=rms.cpu().tolist(), branch_to_input_rms=branch/max(base,1e-12),
                alpha_first_chunk=chunks[0].alpha.double().mean().item(),
                alpha_last_chunk=chunks[-1].alpha.double().mean().item(),
                final_map_rms=[p.double().square().mean().sqrt().item() for p in inspected.final_state.tensors()])


def compact_scores(model, state, probe, ledger):
    measured, _ = score_probe(model, state, probe)
    ledger['probe_image_views'] += sum(len(views) for _, views in probe)
    measured.update(cms_clock=state.clock, scheduled_commits=list(state.commits))
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    configure_runtime()
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('use a new output directory; preserve previous forensic evidence')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = dict(schema='hope_normal_forensics_evidence_v1', diagnostic_id=protocol['diagnostic_id'],
                  status='running', reportable=False, official_test_accessed=False,
                  additional_outer_optimizer_steps=0, selection_feedback=False,
                  original_recipe_modified=False, probe_exposure=protocol['probe_exposure'],
                  protocol=protocol, protocol_sha256=file_hash(PROTOCOL),
                  runner_sha256=file_hash(Path(__file__)), device=args.device,
                  runtime=runtime_identity(), arms={}, spatial=[], gradients=[],
                  reads=dict(calibration_image_views=0, probe_image_views=0, gradient_image_views=0, spatial_image_views=0))
    def persist():
        report['elapsed_seconds'] = time.perf_counter()-started
        atomic_json(report, args.output_dir/'report.json')
    persist()
    try:
        report['original_source_sha256'] = verify_archive(args.evidence, protocol['evidence_sha256'])
        report['evidence_sha256'] = protocol['evidence_sha256']
        print('Verified W6 evidence and original source.', flush=True)
        with zipfile.ZipFile(args.evidence) as z:
            context = context_from_archive(z)
            calibration = validate_bundle(load_tensor(z, 'prefix_features.pt'), context)
            probes = validate_bundle(load_tensor(z, 'probe_features.pt'), context, role='probe')
            roles = context['roles']['roles']
            fixtures = [r['relative_path'] for r in roles[protocol['fixture_role']]][:protocol['fixture_count']]
            schedule = [r['relative_path'] for role in ('meta_support','meta_query') for r in roles[role]]
            if len(schedule) != 70 or len(probes['images']) != protocol['probe_count']:
                raise ValueError('unexpected diagnostic data counts')
            if any('/train/good/' not in p for p in (*schedule,*probes['images'])):
                raise ValueError('diagnostic may only access normal train images')
            masks = patch_masks(device=args.device)
            probe = [(identity, [(r['tensors'][f'masked_{view}'].to(args.device),
                                  r['tensors']['teacher'].to(args.device), masks[view]) for view in range(4)])
                     for identity,r in sorted(probes['images'].items())]
            report['data_identities'] = dict(stream=schedule, fixtures=fixtures,
                                            probe=[identity for identity,_ in probe], roles_digest=context['roles']['digest'])
            def get(identity, view):
                row=calibration['images'][identity]['tensors']
                return row[f'masked_{view}'].to(args.device),row['teacher'].to(args.device),masks[view]
            for arm in protocol['arms']:
                parent=arm if arm in ('A1','C0','C1') else 'M'
                payload=load_tensor(z,f'selection/{parent}/deployment_initialization.pt')
                receipt=validate_initialization(payload,context)
                model=predictor_from_initialization(payload,args.device)
                before=tensor_digest((dict(model.named_parameters()),))
                if arm=='M_visual_off':
                    model.dynamics=replace(model.dynamics,visual_updates=False)
                frozen=0 if arm=='M_freeze_first' else 1 if arm=='M_freeze_second' else None
                state=model.cms.initial_state()
                arm_report=dict(parent=parent, selected_recipe_digest=receipt['digest'],
                                freeze_level=frozen, counter_scope='scheduled commits; actual weight applications listed separately',
                                checkpoints={}, events=[], actual_weight_applications=[0]*len(state.weights))
                report['arms'][arm]=arm_report
                snapshots={}
                for t in range(len(schedule)+1):
                    if t in protocol['checkpoints']:
                        if model.dynamics.cms_updates or t==0:
                            arm_report['checkpoints'][str(t)]=compact_scores(model,state,probe,report['reads'])
                        else:
                            arm_report['checkpoints'][str(t)]=deepcopy(arm_report['checkpoints']['0'])
                        arm_report['checkpoints'][str(t)]['stream_events_seen']=t
                        print(arm,'event',t,'probe loss',arm_report['checkpoints'][str(t)]['mean_normal_loss'],flush=True)
                        persist()
                    if arm in protocol['gradient_methods'] and t in protocol['gradient_states']:
                        snapshots[t]=state.detached()
                    if t==len(schedule):break
                    if not model.dynamics.cms_updates:continue
                    features,target,mask=get(schedule[t],t%4)
                    report['reads']['calibration_image_views']+=1
                    with torch.no_grad():encoded=model.encode(features)
                    old=state
                    result=model.cms.event(encoded,target,mask,model.decoder,state,meta=False)
                    state=result.state if frozen is None else retained_level(old,result.state,frozen)
                    model.cms.validate_state(state)
                    for i,d in enumerate(result.diagnostics):
                        arm_report['actual_weight_applications'][i]+=int(d['committed'] and i!=frozen)
                    arm_report['events'].append(dict(event=t+1,identity=schedule[t],view=t%4,
                        pre_event_local_losses=[loss.item() for loss in result.losses],
                        scheduled_commits=list(state.commits),pending=list(state.pending)))
                if arm in protocol['gradient_methods']:
                    for t,snapshot in snapshots.items():
                        for identity in fixtures:
                            for view in range(4):
                                features,target,mask=get(identity,view)
                                values=gradient_alignment(model,snapshot,features,target,mask)
                                report['gradients'].append(dict(method=arm,history_events=t,identity=identity,view=view,**values))
                                report['reads']['gradient_image_views']+=1
                    print(arm,'gradient diagnostics complete',flush=True)
                if arm=='M':
                    for identity in fixtures:
                        for view in range(4):
                            features,_,_=get(identity,view)
                            report['spatial'].append(dict(identity=identity,view=view,**spatial_readout(model,features)))
                            report['reads']['spatial_image_views']+=1
                    print('M spatial diagnostics complete',flush=True)
                if before!=tensor_digest((dict(model.named_parameters()),)):
                    raise RuntimeError('diagnostics changed learned parameters')
                arm_report['learned_parameters_unchanged']=True
                persist()
                del model,payload,snapshots
        if any(file_hash(ROOT/p)!=digest for p,digest in report['original_source_sha256'].items()):
            raise RuntimeError('original source changed during diagnostics')
        report.update(status='complete_pending_architecture_review', saved_initializations_unchanged=True,
                      original_source_unchanged=True, resources=resource_usage(args.device))
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        persist()
        raise
    persist()
    print('Evidence:',args.output_dir/'report.json',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
