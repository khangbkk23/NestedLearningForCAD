"""Local final artifact QA; does not fit or inspect anomaly data."""
from pathlib import Path
import hashlib
import json
import re
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dataset.benchmark_protocol_v2 import file_hash
from training.hope_runtime_v4 import source_identity, atomic_json

report=dict(status='running',reportable=False,W6_scientific_acceptance=False,W7_authorized=False,checks={})
inheritance=json.loads((ROOT/'docs/main_spec/v4_inheritance_manifest.json').read_text())
for name,digest in inheritance['v3_parent_sha256'].items():
    assert file_hash(ROOT/name)==digest, name
report['checks']['original_v3_specs_unchanged']=True
for name in ('architecture_decision','instruction_CAD','planning_CAD'):
    old=(ROOT/f'docs/main_spec/{name}_v3.md').read_text(encoding='utf-8')
    new=(ROOT/f'docs/main_spec/{name}_v4.md').read_text(encoding='utf-8')
    assert len(new)>=len(old)
    # All major numbered sections retained (some names updated for the revision).
    headings=lambda text:re.findall(r'^#{2,3} ([0-9]+(?:\.[0-9]+)?|[AB][0-9]*)(?=[. — ])',text,re.M)
    assert set(headings(old))<=set(headings(new)), name
    assert 'prefix_warm_joint_stress_v4' in new and 'hope_cad_v4_warm_joint_cms_row_retention' in new
    assert 'logit}(0.9)' not in new
    assert 'max 200 steps' not in new and 'cap 200 steps' not in new
    report['checks'][f'{name}_full_inheritance']=dict(parent_bytes=len(old.encode()),new_bytes=len(new.encode()))
verification=json.loads((ROOT/'results/hope_cad_v2/v4_verification_local.json').read_text())
assert verification['success'] and verification['complete_suite'] and not verification['failures'] and not verification['errors']
assert verification['source_sha256']==source_identity(), 'source drift after suite'
report['checks']['full_suite']=dict(tests=verification['tests_run'],skipped=verification['skipped'],seconds=verification['elapsed_seconds'])
for path in (ROOT/'scripts/hope_cad_v2').glob('*v4.py'):
    compile(path.read_text(encoding='utf-8'),str(path),'exec')
for path in (ROOT/'training').glob('*v4.py'):
    compile(path.read_text(encoding='utf-8'),str(path),'exec')
for name in ('hope_cad_v4_preflight_kaggle.ipynb','hope_cad_v4_normal_kaggle.ipynb'):
    book=json.loads((ROOT/'notebooks'/name).read_text())
    assert len({c['id'] for c in book['cells']})==len(book['cells'])
    for index,cell in enumerate(book['cells']):
        if cell['cell_type']=='code': compile(''.join(cell['source']),f'{name}:{index}','exec')
    text='\n'.join(''.join(c['source']) for c in book['cells'])
    assert 'hope_normal_evidence_v4' in text and 'hope_normal_v4_evidence_v3' not in text
    assert "'--parent-evidence'" in text and 'V3 learned evidence cannot resume v4' in text
report['checks']['notebook_and_CLI_syntax']=True
parent=ROOT/'docs/w6_kaggle/hope_cad_v2_2_w1_w6_evidence.zip'
with zipfile.ZipFile(parent) as archive:
    features=json.loads(archive.read('prefix_features.json'))
    identity=features['feature_identity']
    from training.hope_feature_contract_v3 import PRODUCERS
    for name in PRODUCERS:
        assert identity['source_sha256'][name]==file_hash(ROOT/name), name
report['checks']['historical_frozen_feature_producer_hashes_match']=len(PRODUCERS)
bundle=ROOT/'ultils/hope_cad_v4_source_bundle.zip'
with zipfile.ZipFile(bundle) as archive:
    manifest=json.loads(archive.read('bundle_manifest.json'))
    assert manifest['schema']=='hope_normal_source_bundle_v4'
    assert archive.testzip() is None
    for name,digest in manifest['files'].items():
        assert hashlib.sha256(archive.read(name)).hexdigest()==digest
        assert file_hash(ROOT/name)==digest
    report['checks']['source_bundle']=dict(files=len(manifest['files']),sha256=file_hash(bundle),bytes=bundle.stat().st_size)
report['status']='passed'
report['scope']='local source/math/packaging QA; new canonical T4/RGB/real W6 pending'
atomic_json(report,ROOT/'results/hope_cad_v2/v4_delivery_validation.json')
evidence_paths=[ROOT/'results/hope_cad_v2'/name for name in (
    'v4_verification_local.json','v4_verification_local.log',
    'v4_delivery_validation.json','v4_reduced_profile_local.json',
    'v4_notebook_verification_local.json')]
target=ROOT/'ultils/v4_local_verification_evidence.zip'
manifest=dict(schema='hope_local_verification_evidence_v4',reportable=False,
              source_bundle_sha256=file_hash(bundle),W6_scientific_acceptance=False,
              files={p.name:file_hash(p) for p in evidence_paths})
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for p in evidence_paths: archive.write(p,p.name)
    archive.writestr('artifact_manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    for name,digest in manifest['files'].items(): assert hashlib.sha256(archive.read(name)).hexdigest()==digest
print(json.dumps(report,indent=2))
