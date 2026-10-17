"""Machine-readable design freeze. No runtime hyperparameter search."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2]
CONTRACT=json.loads((ROOT/'docs/main_spec/revision_contract_v5.json').read_text(encoding='utf-8'))
DESIGN_ID=CONTRACT['design_id']
PILOT_JOBS=tuple(CONTRACT['pilot']['jobs'])

def canonical(value):return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def file_sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def design_digest():return digest(CONTRACT)
