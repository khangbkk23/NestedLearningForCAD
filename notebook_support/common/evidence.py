"""Hash-verified evidence transport; compatible payload semantics, no W6 imports."""
from pathlib import Path,PurePosixPath
import hashlib,json,shutil,zipfile
from dataset.benchmark_protocol_v2 import file_hash
from training.runtime import atomic_json

def destination(root, name):
    relative = PurePosixPath(name)
    if (not isinstance(name,str) or relative.is_absolute() or relative.as_posix()!=name
            or '\\' in name or ':' in name or '..' in relative.parts):
        raise ValueError('unsafe archive path')
    path = Path(root)/name
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('archive path escapes target')
    return path

def restore_archive(source, root, *, schema, select=None):
    root = Path(root)
    source = Path(source)
    if source.is_dir():
        manifest = json.loads((source/'artifact_manifest.json').read_text())
        if manifest.get('schema')!=schema:
            raise ValueError('extracted evidence schema mismatch')
        for name,expected in manifest['files'].items():
            path = destination(root,name)
            original = destination(source,name)
            if select is not None and not select(name):
                continue
            if not original.is_file() or original.is_symlink() or file_hash(original)!=expected:
                raise ValueError('extracted evidence checksum mismatch')
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists() and file_hash(path)!=expected:
                raise ValueError('restore would overwrite different local evidence')
            shutil.copyfile(original,path)
        return manifest
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('duplicate archive entries')
        manifest = json.loads(archive.read('artifact_manifest.json'))
        if manifest.get('schema') != schema or set(names) != set(manifest['files'])|{'artifact_manifest.json'}:
            raise ValueError('archive schema/file inventory mismatch')
        for name,expected in manifest['files'].items():
            path = destination(root,name)
            if select is not None and not select(name):
                continue
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary = path.with_name(path.name+'.import_tmp')
            h = hashlib.sha256()
            with archive.open(name) as reader, temporary.open('wb') as writer:
                for block in iter(lambda:reader.read(8*1024*1024),b''):
                    h.update(block)
                    writer.write(block)
            if h.hexdigest()!=expected:
                temporary.unlink(missing_ok=True)
                raise ValueError(f'archive checksum mismatch: {name}')
            if path.exists() and file_hash(path)!=expected:
                temporary.unlink(missing_ok=True)
                raise ValueError('restore would overwrite different local evidence')
            temporary.replace(path)
    return manifest

def export_archive(root, output, *, schema='hope_w7_evidence_v1'):
    root, output = Path(root),Path(output)
    paths = sorted(p for p in root.rglob('*') if p.is_file()
                   and p!=output and not p.name.endswith(('.tmp','.import_tmp')))
    manifest = dict(schema=schema,
                    files={p.relative_to(root).as_posix():file_hash(p) for p in paths})
    temporary = output.with_name(output.name+'.tmp')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=2) as archive:
        for path in paths:
            archive.write(path,path.relative_to(root).as_posix())
        archive.writestr('artifact_manifest.json',json.dumps(manifest,sort_keys=True))
    # Streaming verification does not require a multi-GB file in RAM.
    with zipfile.ZipFile(temporary) as archive:
        for name,expected in manifest['files'].items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:
                    raise RuntimeError('export verification failed')
    temporary.replace(output)
    return dict(path=str(output),bytes=output.stat().st_size,sha256=file_hash(output),files=len(paths))

def read_manifest(source):
    source=Path(source)
    if source.is_dir():
        raw=(source/'artifact_manifest.json').read_bytes()
    else:
        with zipfile.ZipFile(source) as z:
            names=z.namelist()
            if len(names)!=len(set(names)):raise ValueError('duplicate archive paths')
            raw=z.read('artifact_manifest.json')
            if set(names)!=set(json.loads(raw)['files'])|{'artifact_manifest.json'}:
                raise ValueError('evidence inventory differs from manifest')
    return json.loads(raw),hashlib.sha256(raw).hexdigest()
