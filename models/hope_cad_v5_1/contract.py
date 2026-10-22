"""Immutable design inputs and additive executable acceptance identity."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


SPEC_NAMES = ('architecture_decision_v5_1.md', 'instruction_CAD_v5_1.md',
              'planning_CAD_v5_1.md', 'revision_contract_v5_1.json',
              'implementation_acceptance_v5_1.md')
CONTRACT = json.loads((ROOT / 'docs/main_spec/revision_contract_v5_1.json').read_text(encoding='utf-8'))
DESIGN_ID = CONTRACT['design_id']
PILOT_JOBS = tuple(CONTRACT['pilot']['jobs'])


def executable_contract():
    return dict(schema='hope_support_executable_contract_v5_1', design=CONTRACT,
                spec_sha256={name: file_sha(ROOT / 'docs/main_spec' / name) for name in SPEC_NAMES},
                score_probe=dict(first_online_images=8, patches_per_image=784,
                                 reduction='max_abs_raw_patch_NN', after_terminal_finalize=True),
                counter_semantics='actual_successful_optimizer_updates; labels//8 is schedule only')


def design_digest():
    return digest(executable_contract())
