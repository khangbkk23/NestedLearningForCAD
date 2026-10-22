"""Pinned source/parent validation and safe evidence discovery (stdlib only)."""
from pathlib import Path, PurePosixPath
import hashlib, json, zipfile

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / 'docs/main_spec/throughput_v5_1_contract.json'
SCHEMA = 'hope_support_v5_1_throughput_evidence_v1'
SOURCE_SCHEMA = 'hope_support_v5_1_throughput_source_v1'
PASSES = ('U1', 'P1', 'U2')

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def contract():
    c = json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))
    if c['contract_revision'] != 2 or c['profile']['passes'] != list(PASSES):
        raise ValueError('Unknown engineering contract')
    return c

def safe(root, name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or p.as_posix() != name or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe payload path: ' + name)
    result = Path(root) / name
    if not result.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Payload escapes root')
    return result

class Evidence:
    def __init__(self, path):
        self.path = Path(path)
        self.archive = None
        if self.path.is_dir():
            raw = safe(self.path, 'artifact_manifest.json').read_bytes()
        else:
            self.archive = zipfile.ZipFile(self.path)
            names = self.archive.namelist()
            if len(names) != len(set(names)):
                self.close(); raise ValueError('Duplicate evidence ZIP entries')
            raw = self.archive.read('artifact_manifest.json')
        self.manifest = json.loads(raw)
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if not isinstance(self.manifest.get('files'), dict):
            self.close(); raise ValueError('Missing evidence file map')
        for n in self.manifest['files']:
            safe(Path('/tmp/verification'), n)
        if self.archive and set(self.archive.namelist()) != set(self.manifest['files']) | {'artifact_manifest.json'}:
            self.close(); raise ValueError('ZIP inventory differs')

    def close(self):
        if self.archive is not None:
            self.archive.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def read(self, name):
        if name not in self.manifest['files']:
            raise ValueError('Unregistered evidence payload: ' + name)
        if self.archive:
            raw = self.archive.read(name)
        else:
            p = safe(self.path, name)
            if p.is_symlink(): raise ValueError('Symlink evidence payload')
            raw = p.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.manifest['files'][name]:
            raise ValueError('Evidence SHA mismatch: ' + name)
        return raw

    def verify_all(self):
        for name in self.manifest['files']:
            self.read(name)
        return len(self.manifest['files'])

    def json(self, name): return json.loads(self.read(name))

def select_evidence(inputs, *, schema, manifest_sha=None, override=None):
    paths = [Path(override)] if override else [p.parent for p in Path(inputs).rglob('artifact_manifest.json')]
    if not override: paths += list(Path(inputs).rglob('*.zip'))
    choices = []
    for p in paths:
        try:
            with Evidence(p) as e:
                if e.manifest.get('schema') == schema and (manifest_sha is None or e.manifest_sha256 == manifest_sha):
                    choices.append((e.manifest_sha256, p))
        except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
            if override: raise ValueError('Invalid explicit evidence override: '+str(p)) from exc
    identities = {h for h, _ in choices}
    if not choices:
        if override or manifest_sha is not None: raise ValueError('Required reviewed evidence not found; set explicit override')
        return None
    if len(identities) != 1:
        raise ValueError('Multiple different evidence versions; remove older outputs or set override')
    return sorted(choices, key=lambda x: str(x[1]))[0][1]

def verify_source(root=ROOT):
    root = Path(root); c = json.loads((root / CONTRACT_PATH.relative_to(ROOT)).read_text(encoding='utf-8'))
    base = json.loads((root/'bundle_manifest.json').read_text(encoding='utf-8'))
    expected = c['frozen_reference']
    if len(base['files']) != expected['payloads'] or digest(base['files']) != expected['source_digest']:
        raise ValueError('Frozen166 source identity mismatch')
    for name, h in base['files'].items():
        if sha(safe(root, name)) != h: raise ValueError('Frozen base source drift: ' + name)
    delivery = json.loads((root/'throughput_manifest.json').read_text(encoding='utf-8'))
    if delivery['schema'] != SOURCE_SCHEMA or delivery['contract_sha256'] != sha(root/CONTRACT_PATH.relative_to(ROOT)):
        raise ValueError('Profiler delivery/contract mismatch')
    for name, h in delivery['files'].items():
        if sha(safe(root, name)) != h: raise ValueError('Profiler source drift: ' + name)
    if any(delivery['files'].get(n) != h for n,h in base['files'].items()):
        raise ValueError('Profiler delivery omitted or replaced frozen base')
    return dict(source_digest=digest(delivery['files']), files=len(delivery['files']),
                base_source_digest=expected['source_digest'], contract_sha256=sha(CONTRACT_PATH if root==ROOT else root/CONTRACT_PATH.relative_to(ROOT)))

def verify_parent(e, c):
    p = c['parent']
    if e.manifest_sha256 != p['artifact_manifest_sha256']:
        raise ValueError('Wrong parent evidence version')
    if not e.path.is_dir() and sha(e.path) != p['archive_sha256']:
        raise ValueError('Wrong reviewed parent ZIP bytes')
    count = e.verify_all()
    for entry, h in [(p['active_entry'],p['active_sha256']), (p['profile_start_entry'],p['profile_start_sha256'])]:
        if hashlib.sha256(e.read(entry)).hexdigest() != h: raise ValueError('Parent checkpoint changed')
    suite = e.json('suite_lock.json'); job = e.json('runs/NESTED_seed0/job_lock.json')
    if suite['digest'] != digest(suite['identity']) or job['digest'] != digest(job['identity']):
        raise ValueError('Corrupt parent lock digest')
    if suite['digest'] != p['suite_digest'] or job['digest'] != p['job_digest']:
        raise ValueError('Parent suite/job differs')
    for key, value in [('source_digest',c['frozen_reference']['source_digest']),('study_digest',c['study_digest']),('design_digest',c['method_digest'])]:
        if suite['identity'][key] != value: raise ValueError('Parent identity mismatch: ' + key)
    rows = job['identity']['rows']; f = c['profile']; window = rows[f['start_cursor']:f['end_cursor_exclusive']]
    if len(rows) != 3599 or digest(window) != f['window_rows_canonical_sha256']:
        raise ValueError('Registered replay manifest changed')
    if any(r.get('split')!='train' or '/train/good/' not in r['relative_path'] for r in rows):
        raise ValueError('Normal-only rights violated')
    if e.json('handoff.json')['status'] != 'paused_at_atomic_boundary': raise ValueError('Parent is not reviewed pause')
    return dict(rows=rows, suite=suite, job=job, verified_payloads=count,
                manifest_sha256=e.manifest_sha256, representation='directory' if e.path.is_dir() else 'ZIP')
