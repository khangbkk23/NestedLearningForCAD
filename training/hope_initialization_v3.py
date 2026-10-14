"""Selected-initialization receipts; random smoke exports cannot enter scientific runs."""
import torch
from dataset.benchmark_manifest_v1 import canonical_digest
from dataset.benchmark_protocol_v2 import file_hash
from models.hope_cad_v2.state import tensor_digest
from training.hope_feature_contract_v3 import expected_feature_identity, validate_feature_identity
from training.hope_runtime_v3 import learning_source_identity


def weights_digest(weights):
    tensors = {k: v for k, v in weights.items() if isinstance(v, torch.Tensor)}
    other = {k: v for k, v in weights.items() if not isinstance(v, torch.Tensor)}
    return canonical_digest(dict(tensors=tensor_digest((tensors,)), other=other))


def selection_receipt(outcome, selection_path):
    selected = outcome.get('selected')
    if outcome.get('status') != 'selected' or outcome.get('smoke') or not selected:
        raise ValueError('a real successful selection is required')
    if not selected.get('gate', {}).get('passed'):
        raise ValueError('selected candidate did not pass its registered gate')
    from training.hope_cms_stress_v3 import CONSTANTS
    from training.hope_outer_learning_v2 import BETA_GRID, STATIC_LR_GRID
    method, gate = selected['method'], selected['gate']
    if (outcome.get('selection_rule_id') != 'prefix_cms_stress_v3'
            or outcome.get('common_seed') != 0 or selected.get('status') != 'fit_complete'
            or selected['shortlist']['step'] not in (0, 25, 50, 100, 200)):
        raise ValueError('selection is not the registered scientific recipe')
    if method in ('C0', 'C1'):
        if gate.get('status') != 'not_applicable_static' or selected['outer_lr'] not in STATIC_LR_GRID:
            raise ValueError('invalid static-control selection evidence')
    elif method in ('M', 'C2', 'C3', 'C4', 'A1'):
        commits = [1024] if method == 'C3' else ([1024, 1024] if method == 'C4' else [1024, 128])
        if (gate.get('status') != 'passed' or gate.get('constants') != CONSTANTS
                or gate.get('events_completed') != 1024 or gate.get('commits') != commits
                or len(gate.get('trace', [])) != 1024
                or gate.get('roles_digest') != selected['role_digest']
                or selected['beta'] not in BETA_GRID):
            raise ValueError('incomplete or incompatible 1024-event gate evidence')
        if method != 'A1' and selected['outer_lr'] != 1e-4:
            raise ValueError('invalid dynamic-control outer learning rate')
        if method == 'C4' and not outcome.get('inherited_m_selection_sha256'):
            raise ValueError('C4 must bind the selected M parent')
        if method == 'A1' and (set(outcome.get('inherited', {})) != {'M', 'C1'}
                               or outcome.get('additional_optimizer_steps') != 0):
            raise ValueError('A1 must inherit M/C1 without additional fitting')
    else:
        raise ValueError('unregistered selected method')
    record = dict(schema='hope_selected_recipe_v3', status='selected', seed=0,
                  method=selected['method'], beta=selected['beta'], outer_lr=selected['outer_lr'],
                  selected_step=selected['shortlist']['step'],
                  weights_sha256=weights_digest(selected['shortlist']['weights']),
                  selection_file_sha256=file_hash(selection_path),
                  gate_sha256=canonical_digest(selected['gate']),
                  schedule_digest=outcome['schedule_digest'],
                  roles_digest=selected['role_digest'],
                  selection_rule_id='prefix_cms_stress_v3')
    if method == 'C4':
        record['inherited_m_selection_sha256'] = outcome['inherited_m_selection_sha256']
    if method == 'A1':
        record['inherited'] = outcome['inherited']
    record['digest'] = canonical_digest(record)
    return record


def validate_initialization(initialization, context, *, selection_path=None):
    if initialization.get('schema') != 'hope_deployment_initialization_v3':
        raise ValueError('invalid deployment initialization schema')
    provenance = initialization.get('provenance', {})
    validate_feature_identity(provenance.get('feature_identity', {}), expected_feature_identity(context))
    if any(provenance.get('source_sha256',{}).get(k) != v for k,v in learning_source_identity().items()):
        raise ValueError('initialization source/spec hashes differ from the current implementation')
    receipt = provenance.get('selection_receipt', {})
    if (receipt.get('schema') != 'hope_selected_recipe_v3' or receipt.get('status') != 'selected'
            or receipt.get('digest') != canonical_digest({k: v for k, v in receipt.items() if k != 'digest'})
            or receipt.get('weights_sha256') != weights_digest(initialization['slow_and_initial_weights'])
            or receipt.get('method') != initialization['metadata']['dynamics']['method']
            or receipt.get('beta') != initialization['metadata']['dynamics']['rates'][0]
            or not initialization['metadata'].get('canonical')
            or provenance.get('roles_digest') != context['roles']['digest']
            or receipt.get('roles_digest') != context['roles']['digest']
            or receipt.get('schedule_digest') != provenance.get('schedule_digest')
            or initialization.get('cursor') != 0 or initialization['cms']['clock'] != 0
            or any(initialization['cms']['pending']) or any(initialization['cms']['commits'])
            or tensor_digest(tuple(initialization['cms']['weights'])) != initialization['initial_cms_sha256']):
        raise ValueError('missing/invalid selected recipe or non-initial deployment state')
    # CLI consumers also require the original selection payload, not only its label.
    if selection_path is not None:
        if file_hash(selection_path) != receipt['selection_file_sha256']:
            raise ValueError('selection payload hash mismatch')
        outcome = torch.load(selection_path, map_location='cpu', weights_only=False)
        if selection_receipt(outcome, selection_path) != receipt:
            raise ValueError('initialization differs from selection evidence')
    return receipt


def predictor_from_initialization(initialization, device='cpu'):
    from models.hope_cad_v2.predictor import HOPEFeaturePredictor
    from models.hope_cad_v2.visual_block import VisualBlockConfig
    metadata = initialization['metadata']
    method = metadata['dynamics']['method']
    beta = metadata['dynamics']['rates'][0]
    model = HOPEFeaturePredictor(VisualBlockConfig(**metadata['visual_config']), method, beta,
                                 beta if method == 'C4' else None)
    if model.metadata() != metadata:
        raise ValueError('initialization topology/dynamics mismatch')
    model.load_state_dict(initialization['slow_and_initial_weights'], strict=True)
    return model.to(device).eval()
