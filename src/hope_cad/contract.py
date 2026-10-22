"""Unchanged scientific contract, explicit historical design identity.

The historical digest names the accepted method. Current source and consolidated
documentation have their own complete source digest; they are not old spec bytes.
"""
from pathlib import Path
import hashlib
import json
from .paths import ROOT, RESOURCES


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


CONTRACT = json.loads((RESOURCES / 'method.json').read_text(encoding='utf-8'))
DESIGN_ID = CONTRACT['design_id']
PILOT_JOBS = tuple(CONTRACT['pilot']['jobs'])


def executable_contract():
    lineage = json.loads((RESOURCES / 'lineage.json').read_text(encoding='utf-8'))
    if file_sha(RESOURCES / 'method.json') != lineage['method_json_sha256']:
        raise ValueError('Scientific method JSON changed; a new method revision is required')
    return dict(schema='hope_support_executable_contract_v5_1', design=CONTRACT,
                spec_sha256=lineage['historical_spec_sha256'],
                score_probe=dict(first_online_images=8, patches_per_image=784,
                                 reduction='max_abs_raw_patch_NN', after_terminal_finalize=True),
                counter_semantics='actual_successful_optimizer_updates; labels//8 is schedule only')


def design_digest():
    value = digest(executable_contract())
    lineage = json.loads((RESOURCES / 'lineage.json').read_text(encoding='utf-8'))
    if value != lineage['historical_design_digest']:
        raise ValueError('Historical method identity does not reproduce')
    return value
