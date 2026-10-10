"""Strict prefix protocol validator; never weakens the v1 schema."""
import copy
import re
import random
from pathlib import Path
from dataset.benchmark_protocol_v1 import CANONICAL_MVTEC_TASKS
from dataset.benchmark_protocol_v2 import CANONICAL_VISA_TASKS, file_hash
from models.benchmark_registry_v2 import route_descriptor
from .benchmark_config_v1 import validate_configs as validate_v1


def validate_configs(protocol, method, smoke=False):
    for cfg in (protocol, method):
        if cfg.get('version') != 2 or not re.fullmatch(r'[A-Za-z0-9_.-]+', cfg.get('id', '')):
            raise ValueError('expected safe v2 config identity')
    name = protocol['dataset']['name']
    if name not in ('MVTec AD', 'VisA'):
        raise ValueError('unsupported prefix dataset')
    tasks = CANONICAL_MVTEC_TASKS if name == 'MVTec AD' else CANONICAL_VISA_TASKS
    expected_id = 'mvtec_1x15_prefix_cal_v2' if name == 'MVTec AD' else 'visa_1x12_prefix_cal_v2'
    order_seed = protocol.get('order_seed')
    if order_seed is not None:
        if order_seed not in (100, 200):
            raise ValueError('only preregistered order seeds 100/200 are allowed')
        tail = list(tasks[1:])
        random.Random(order_seed).shuffle(tail)
        tasks = [tasks[0], *tail]
        expected_id += f'_order{order_seed}'
    if protocol['task_order'] != tasks:
        raise ValueError('task order differs from registered canonical/first-task-fixed order')
    if protocol['id'] != expected_id:
        raise ValueError('protocol ID does not identify frozen recipe')
    if protocol['training'] != dict(normal_only=True, dev_mode='first_task_prefix', drop_last=False):
        raise ValueError('prefix normal-only training is immutable')
    if protocol.get('calibration') != dict(prefix_cap=100, minimum_prefix=32, support_length=16,
                                         split='contiguous_50_20_15_remainder', online_exclude=['normal_validation', 'normal_probe']):
        raise ValueError('prefix calibration policy mismatch')
    if any(protocol['leakage'].get(k) != 'forbidden' for k in
           ('official_test_feedback', 'test_checkpoint_selection', 'test_early_stopping', 'test_parameter_tuning', 'future_task_calibration')):
        raise ValueError('forbidden data access')
    route = route_descriptor(method['adapter'])
    if method['memory']['update_unit'] != route['effective_update_unit']:
        raise ValueError('effective CADIC update unit mismatch')
    if method.get('track') != 'matched_prefix' or (not smoke and method['memory']['budget'] != 10000):
        raise ValueError('main matched CADIC requires matched_prefix, 10k budget')
    if method['scoring'] != dict(image_neighbors_b=9,
            image_score='s_star*(1-exp(s_star)/sum_exp(support_distances_around_c_star))'):
        raise ValueError('matched baseline requires the locked b=9 full Eq.9 score')
    # Reuse unchanged v1 checks on a copy, only for shared CADIC math/metrics.
    p, m = copy.deepcopy(protocol), copy.deepcopy(method)
    p['version'] = m['version'] = 1
    p['dataset']['name'], p['task_order'] = 'MVTec AD', CANONICAL_MVTEC_TASKS
    p['training']['dev_mode'] = 'disabled'
    m['adapter'] = 'cadic'
    validate_v1(p, m, smoke=smoke)
    if method['preprocessing']['mean'] != [0.485, 0.456, 0.406] or method['preprocessing']['std'] != [0.229, 0.224, 0.225]:
        raise ValueError('ImageNet preprocessing constants are immutable')
    checkpoint = Path(method['extractor']['checkpoint_path'])
    expected = method['extractor'].get('checkpoint_sha256', '')
    if not re.fullmatch('[0-9a-f]{64}', expected) or file_hash(checkpoint) != expected:
        raise ValueError('real frozen checkpoint and matching SHA256 required, including smoke')
    return dict(reportable=not smoke, effective_adapter=route,
                checkpoint_sha256=expected, exact_parity_claim=False)
