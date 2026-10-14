"""Verify frozen features against the current shared protocol, not self-declared hashes."""
from pathlib import Path
import json
import torch
from dataset.benchmark_manifest_v1 import canonical_digest
from dataset.benchmark_protocol_v2 import calibration_roles, validate_prefix_manifest, file_hash
from models.hope_cad_v2.state import tensor_digest
from training.benchmark_config_v1 import load_configs
from training.benchmark_config_v2 import validate_configs
from training.hope_runtime_v3 import ROOT

PRODUCERS = ('dataset/benchmark_protocol_v1.py', 'dataset/benchmark_protocol_v2.py',
             'dataset/hope_mask_views.py', 'models/feature_extractors/cadic_vit_v1.py',
             'models/hope_cad_v2/teacher_head.py', 'scripts/hope_cad_v2/prepare_prefix_features.py',
             'training/hope_feature_contract_v3.py')


def load_context(protocol_path, method_path, lock_dir):
    from scripts.hope_cad_v2.run_matched_cadic import baseline_identity
    protocol, method = load_configs(protocol_path, method_path)
    resolved = validate_configs(protocol, method)
    manifest = json.loads((Path(lock_dir) / 'manifest_train.json').read_text(encoding='utf-8'))
    validate_prefix_manifest(manifest, protocol, verify_bytes=True)
    lock = json.loads((Path(lock_dir) / 'baseline_lock.json').read_text(encoding='utf-8'))
    identity = baseline_identity(protocol, method, resolved, manifest['digest'],
                                 seed=manifest['seed'], smoke=False, max_tasks=None, max_train_images=None)
    if lock != dict(identity=identity, digest=canonical_digest(identity)):
        raise ValueError('stale baseline lock: data/config/extractor/route/source mismatch')
    return dict(protocol=protocol, method=method, manifest=manifest, lock=lock,
                roles=calibration_roles(manifest))


def expected_feature_identity(context):
    method, manifest, lock = context['method'], context['manifest'], context['lock']
    extractor = {k: v for k, v in method['extractor'].items() if k != 'checkpoint_path'}
    return dict(contract_version=1, design_id='hope_cad_v3_masked_teacher_cms',
                backbone_sha256=extractor['checkpoint_sha256'],
                backbone_identity=extractor['checkpoint_identity'],
                extractor_sha256=canonical_digest(extractor),
                preprocessing_sha256=canonical_digest(method['preprocessing']),
                mask_sha256=file_hash(ROOT / 'dataset/hope_mask_views.py'),
                source_sha256={p: file_hash(ROOT / p) for p in PRODUCERS},
                manifest_digest=manifest['digest'], roles_digest=context['roles']['digest'],
                baseline_lock_digest=lock['digest'])


def validate_feature_identity(actual, expected):
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ValueError('frozen feature identity mismatch: ' + ', '.join(mismatches))


def validate_bundle(bundle, context, *, role='calibration'):
    if role not in ('calibration', 'probe'):
        raise ValueError('unregistered feature role')
    if bundle.get('schema') != 'hope_frozen_prefix_features_v3' or bundle.get('cache_role') != role:
        raise ValueError('old or wrong-role feature cache; rebuild with the current producer')
    validate_feature_identity(bundle.get('feature_identity', {}), expected_feature_identity(context))
    if bundle.get('role_manifest') != context['roles']:
        raise ValueError('cache role manifest differs from byte-verified protocol')
    roles = ('normal_probe',) if role == 'probe' else ('meta_support', 'meta_query', 'normal_validation')
    permitted = {r['relative_path'] for key in roles for r in context['roles']['roles'][key]}
    if set(bundle.get('images', {})) != permitted:
        raise ValueError('cache must contain precisely the permitted role identities')
    for row in bundle['images'].values():
        tensors = row['tensors']
        if set(tensors) != {'masked_0', 'masked_1', 'masked_2', 'masked_3', 'teacher'}:
            raise ValueError('invalid cache tensor keys')
        if row['sha256'] != tensor_digest((tensors,)):
            raise ValueError('cache tensor checksum mismatch')
        for tensor in tensors.values():
            if (tensor.shape != (1, 784, 768) or tensor.dtype != torch.float32
                    or tensor.requires_grad or not bool(torch.isfinite(tensor).all())):
                raise ValueError('invalid cached tensor shape/dtype/graph/finiteness')
        if not bool(((tensors['teacher'].norm(dim=-1) - 1).abs() < 1e-5).all()):
            raise ValueError('teacher must be unit-normalized')
    return bundle
