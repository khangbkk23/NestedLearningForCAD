"""Source and reviewed-prerequisite locks for one exact-parent execution migration."""
import json
from training.support_throughput_v5_1.contract import ROOT,sha,digest,safe,contract as base_contract
from training.support_nn_write_v5_1.contract import verify_source as verify_G3

SCHEMA='hope_support_v5_1_nn_write_bridge_evidence_v1'
SOURCE_SCHEMA='hope_support_v5_1_nn_write_bridge_source_v1'
CONTRACT_PATH=ROOT/'docs/main_spec/migration_nn_write_v5_1_contract.json'

def contract():
    c=json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))
    if c['schema']!='hope_support_nn_write_migration_contract_v1' or c['job']!='NESTED_seed0' or (c['start_cursor'],c['end_cursor_exclusive'],c['max_unique_arrivals'],c['max_owner_transitions'])!=(376,384,8,16):raise ValueError('Unknown bridge contract')
    return c

def prerequisites():
    c=contract();values={}
    for name,h in c['review_receipts_sha256'].items():
        p=ROOT/'notebook_support/support_migration_v5_1/receipts'/(name+'.json')
        if sha(p)!=h:raise ValueError('Prerequisite receipt changed: '+name)
        values[name]=json.loads(p.read_text(encoding='utf-8'))
    if not values['G2local']['passed'] or values['G2local']['tests']!=94 or not values['G3audit']['passed'] or not values['G3audit']['metrics']['summary']['all_engineering_gates_passed']:raise ValueError('Missing reviewed G2/G3 gates')
    for name in ['G1','G2','G3']:
        if values[name]['status']!='finalized' or values[name]['reportable'] is not False:raise ValueError('Prerequisite review status differs')
    return dict(c['review_receipts_sha256'])

def verify_source():
    old=verify_G3();c=contract();prerequisites()
    if old['source_digest']!=c['G3_source_digest']:raise ValueError('Wrong reviewed G3 source')
    m=json.loads((ROOT/'bridge_manifest.json').read_text(encoding='utf-8'))
    if m['schema']!=SOURCE_SCHEMA or m['contract_sha256']!=sha(CONTRACT_PATH):raise ValueError('Bridge source identity differs')
    for n,h in m['files'].items():
        if sha(safe(ROOT,n))!=h:raise ValueError('Bridge payload drift: '+n)
    oldmap=json.loads((ROOT/'nn_write_manifest.json').read_text(encoding='utf-8'))['files']
    if any(m['files'].get(n)!=h for n,h in oldmap.items()):raise ValueError('Bridge source replaced reviewed G3 payload')
    return dict(source_digest=digest(m['files']),files=len(m['files']),contract_sha256=sha(CONTRACT_PATH),G3_source_digest=old['source_digest'])

def make_identity(info,source,runtime,policy):
    """No data reads or owner creation; full JSON identities with a distinct execution source."""
    parent=info['job'];suite=info['suite']
    if digest(parent['identity'])!=parent['digest'] or digest(suite['identity'])!=suite['digest']:raise ValueError('Corrupt inherited identity')
    if runtime!=suite['identity']['runtime']:raise ValueError('Runtime differs from parent')
    execution=dict(schema='nn_write_execution_identity_v1',engineering_id=policy['id'],parent_manifest_sha256=info['manifest_sha256'],
        parent_suite_digest=suite['digest'],parent_job_digest=parent['digest'],new_source=source,runtime=runtime,
        method_digest=suite['identity']['design_digest'],study_digest=suite['identity']['study_digest'],prerequisites=policy['review_receipts_sha256'])
    execution_digest=digest(execution)
    si=dict(suite['identity'],schema='support_v5_1_migrated_suite_v1',source_digest=source['source_digest'],parent_suite_digest=suite['digest'],execution_digest=execution_digest)
    new_suite=dict(identity=si,digest=digest(si))
    ji=dict(parent['identity'],suite_digest=new_suite['digest'],fresh_initialization=False,parent_job_digest=parent['digest'],execution_digest=execution_digest)
    return dict(execution=dict(identity=execution,digest=execution_digest),suite=new_suite,job=dict(identity=ji,digest=digest(ji)))
