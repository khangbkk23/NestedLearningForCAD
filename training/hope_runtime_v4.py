"""v4 identities; frozen-feature producers retain their historical v3 contract."""
from training.hope_runtime_v3 import (ROOT, configure_runtime, runtime_identity,
                                    atomic_torch_save, atomic_json, resource_usage,
                                    source_identity as historical_source_identity)
from dataset.benchmark_protocol_v2 import file_hash

DESIGN_ID = 'hope_cad_v4_warm_joint_cms_row_retention'
SELECTION_ID = 'prefix_warm_joint_stress_v4'
SPECS = tuple('docs/main_spec/'+name for name in (
    'architecture_decision_v4.md', 'instruction_CAD_v4.md', 'planning_CAD_v4.md'))


def source_identity():
    files = historical_source_identity()
    files.update({p: file_hash(ROOT/p) for p in SPECS})
    files['docs/main_spec/revision_contract_v4.json'] = file_hash(ROOT/'docs/main_spec/revision_contract_v4.json')
    return files


def learning_source_identity():
    files = set(SPECS) | {'docs/main_spec/revision_contract_v4.json'}
    files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'models/hope_cad_v2').glob('*.py')
                 if p.name not in ('benchmark_adapter.py', 'benchmark_adapter_v4.py'))
    files.update('training/'+p for p in (
        'hope_outer_learning_v4.py', 'hope_selection_v4.py', 'hope_cms_stress_v4.py',
        'hope_search_journal_v4.py', 'hope_feature_contract_v3.py', 'hope_initialization_v4.py',
        'hope_runtime_v3.py', 'hope_runtime_v4.py', 'hope_outer_learning_v2.py',
        'hope_cms_session_v2.py'))
    files.update(('scripts/hope_cad_v2/calibrate_features_v4.py', 'scripts/hope_cad_v2/diagnostics_v4.py'))
    return {p: file_hash(ROOT/p) for p in sorted(files)}
