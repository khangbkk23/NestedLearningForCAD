"""Checkpoint publisher for NANM state (NANM-RECOVERY-01; new code — the server publisher is lost).

publish(obj, path):
  1. serialize to memory with torch.save (size known before any disk write);
  2. storage guard: refuse if the write would push the WSL VHDX host drive (D:) below the reserve;
  3. write to a unique hidden temp file in the target directory, flush + fsync;
  4. os.replace(temp, path)  — the only step that makes the new bytes visible at `path`;
  5. fsync the directory (recorded as False where the filesystem rejects it, e.g. WSL DrvFS / 9p);
  6. publish `<name>.sha256` the same way; read back and verify the digest.
load(path): refuses unless sha256(file) equals the sidecar.

A reader racing a publish sees either the complete old or the complete new payload at `path`; between steps 4
and 6 the sidecar may still name the old digest, so load() fails loudly rather than returning unverified data.

Guarantees are filesystem-dependent. Step 4 atomicity and step 5 durability are POSIX/ext4 properties; on WSL
DrvFS (NTFS over 9p) they are not documented equivalents — see docs/recovery/storage_preflight.md.
"""
from __future__ import annotations

import errno
import hashlib
import io
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch

GiB = 2 ** 30
D_RESERVE_BYTES = 30 * GiB          # NANM-RECOVERY-01 Phase 2 reserve on the VHDX host drive (do not change silently)
VHDX_HOST_MOUNT = Path('/mnt/d')    # D:\ hosts D:\wsl\Ubuntu_new\ext4.vhdx on the owner's laptop


class StorageGuardError(RuntimeError):
    pass


@dataclass(frozen=True)
class PublishRecord:
    path: Path
    sha256: str
    nbytes: int
    dir_fsynced: bool


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def _on_root_fs(d: Path) -> bool:
    return os.stat(d).st_dev == os.stat('/').st_dev


def storage_guard(d: Path, nbytes: int, reserve: int = D_RESERVE_BYTES) -> None:
    """Refuse writes that would leave less than `reserve` free on the physical drive that absorbs them.
    Writes to the WSL root ext4 grow the VHDX on D:, whose real free space is read through /mnt/d
    (`df /` inside WSL reports the virtual disk, not D:)."""
    if _on_root_fs(d) and VHDX_HOST_MOUNT.is_dir():
        free, where = shutil.disk_usage(VHDX_HOST_MOUNT).free, f'{VHDX_HOST_MOUNT} (VHDX host)'
    else:
        free, where = shutil.disk_usage(d).free, str(d)
    if free - nbytes < reserve:
        raise StorageGuardError(f'write of {nbytes} B to {d} would leave {free - nbytes} B free on {where} '
                                f'(< reserve {reserve} B)')


def _fsync_dir(d: Path) -> bool:
    try:
        fd = os.open(d, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    except OSError:
        return False
    try:
        os.fsync(fd)
        return True
    except OSError as e:
        if e.errno in (errno.EINVAL, errno.ENOTSUP, errno.EBADF, errno.EACCES):
            return False
        raise
    finally:
        os.close(fd)


def _write_atomic(path: Path, data: bytes) -> bool:
    tmp = path.with_name(f'.{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}')
    try:
        with open(tmp, 'xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return _fsync_dir(path.parent)


def publish(obj, path, reserve: int = D_RESERVE_BYTES) -> PublishRecord:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    digest = sha256_bytes(data)
    side = f'{digest}  {path.name}\n'.encode()
    storage_guard(path.parent, len(data) + len(side), reserve)
    ok1 = _write_atomic(path, data)
    ok2 = _write_atomic(path.with_name(path.name + '.sha256'), side)
    got = sha256_file(path)
    if got != digest:
        raise RuntimeError(f'read-back digest mismatch for {path}: {got} != {digest}')
    return PublishRecord(path, digest, len(data), ok1 and ok2)


def expected_sha256(path) -> str:
    path = Path(path)
    line = path.with_name(path.name + '.sha256').read_text().split()
    if len(line) != 2 or line[1] != path.name or len(line[0]) != 64:
        raise RuntimeError(f'malformed sidecar for {path}')
    return line[0]


def load(path, weights_only: bool = True):
    path = Path(path)
    want = expected_sha256(path)
    data = path.read_bytes()
    got = sha256_bytes(data)
    if got != want:
        raise RuntimeError(f'checksum mismatch for {path}: {got} != {want}')
    return torch.load(io.BytesIO(data), weights_only=weights_only)
