"""Spec3§5.1 train-only identity metadata; no tensors or test-tree inventory.

Archive fields identify the expected source. Scanning an extracted tree does
not authenticate the archive or establish its receipt as verified.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath

from nested_replay_cad.core.contracts import ArrivalIndex, RecordID, RecordMeta
from nested_replay_cad.core.errors import (
    BudgetStop,
    GateBlocked,
    IdentityMismatch,
    IllegalDataAccess,
    SchemaError,
    ScopeBlocked,
)
from nested_replay_cad.core.identity import canonical_json

ARCHIVE_SHA256 = "cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d"
DATASET_ID = "MVTecAD:mirror:" + ARCHIVE_SHA256
PREPROCESS_ID = "A0-R1:RGB:PIL-bilinear-224:FP32-div255:ImageNet:no-crop-v1"
COUNTS = (
    ("bottle", 209),
    ("cable", 224),
    ("capsule", 219),
    ("carpet", 280),
    ("grid", 264),
    ("hazelnut", 391),
    ("leather", 245),
    ("metal_nut", 220),
    ("pill", 267),
    ("screw", 320),
    ("tile", 230),
    ("toothbrush", 60),
    ("transistor", 213),
    ("wood", 247),
    ("zipper", 240),
)
EXPECTED_SOURCE = {
    "archive_sha256": ARCHIVE_SHA256,
    "archive_bytes": 5264982680,
    "scope": "expected_reference_archive; not_verified_by_tree_scan",
}


@dataclass(frozen=True)
class ManifestProtocol:
    dataset_id: str
    task_counts: tuple[tuple[str, int], ...]

    def validate(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.isascii():
            raise SchemaError("invalid_manifest_dataset_id")
        if (
            self.dataset_id != DATASET_ID
            and re.fullmatch(r"synthetic:[a-z0-9_-]+:v[1-9][0-9]*", self.dataset_id)
            is None
        ):
            raise ScopeBlocked("custom_real_dataset_requires_registered_protocol")
        if not isinstance(self.task_counts, tuple) or not self.task_counts:
            raise SchemaError("invalid_task_counts")
        names = []
        for row in self.task_counts:
            if not isinstance(row, tuple) or len(row) != 2:
                raise SchemaError("invalid_task_count_entry")
            task, count = row
            if (
                not isinstance(task, str)
                or re.fullmatch(r"[a-z][a-z0-9_]*", task, flags=re.ASCII) is None
            ):
                raise SchemaError("invalid_task_name")
            if type(count) is not int or count < 1:
                raise SchemaError("invalid_task_count")
            names.append(task)
        if names != sorted(set(names)):
            raise SchemaError("task_order_must_be_unique_alphabetical")
        if self.dataset_id == DATASET_ID and self.task_counts != COUNTS:
            raise IdentityMismatch("reference_task_counts_are_locked")


REFERENCE = ManifestProtocol(DATASET_ID, COUNTS)


@dataclass(frozen=True)
class ManifestRecord:
    meta: RecordMeta
    relative_path: str
    bytes: int
    split: str
    task_audit_only: str
    shape: tuple[int, int, int]  # original height,width,channels; orchestrator only


@dataclass(frozen=True)
class TrainManifest:
    protocol: ManifestProtocol
    records: tuple[ManifestRecord, ...]
    duplicate_content_aliases: tuple[tuple[RecordID, ...], ...]
    canonical: str  # canonical JSON + one LF, no root path/BOM

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()

    def require_execution_ready(self) -> None:
        if self.duplicate_content_aliases:
            raise GateBlocked("duplicate_content_aliases_require_reviewed_disposition")


def _component(value: str) -> None:
    # Conservative portable v1 subset; no Windows aliases/ADS or ambiguous POSIX paths.
    if (
        not value
        or value in {".", ".."}
        or not value.isascii()
        or any(c in value for c in "\\/:\x00")
        or value.endswith((".", " "))
        or any(ord(c) < 32 for c in value)
        or any(c in value for c in '<>"|?*')
    ):
        raise IllegalDataAccess("invalid_manifest_path_component")
    if value.split(".", 1)[0].upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *{f"COM{i}" for i in range(1, 10)},
        *{f"LPT{i}" for i in range(1, 10)},
    }:
        raise IllegalDataAccess("reserved_manifest_path_component")


def checked_root(dataset_root: str | Path) -> Path:
    raw = Path(dataset_root).absolute()
    try:
        for parent in (*raw.parents, raw):
            if parent.is_symlink() or parent.is_junction():
                raise IllegalDataAccess("symlink_or_junction_dataset_root")
        root = raw.resolve(strict=True)
        if not root.is_dir():
            raise IllegalDataAccess("dataset_root_must_be_directory")
        return root
    except OSError as exc:
        raise IllegalDataAccess("dataset_root_unavailable") from exc


def checked_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or relative != PurePosixPath(relative).as_posix():
        raise IllegalDataAccess("noncanonical_manifest_path")
    parts = relative.split("/")
    for part in parts:
        _component(part)
    path = root
    try:
        for part in parts:
            path = path / part
            if path.is_symlink() or path.is_junction():
                raise IllegalDataAccess("symlink_or_junction_input")
        if not path.resolve(strict=True).is_relative_to(root):
            raise IllegalDataAccess("input_escapes_dataset_root")
        return path
    except OSError as exc:
        raise IllegalDataAccess("manifest_path_unavailable") from exc


def read_verified_bytes(
    root: Path, relative: str, *, max_bytes: int | None = None
) -> bytes:
    """Read regular bytes, detect ordinary concurrent replacement; not hostile-FS sandbox."""
    path = checked_path(root, relative)
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
        raise SchemaError("positive_training_read_cap_required")
    try:
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise IllegalDataAccess("input_must_be_regular_file")
        if max_bytes is not None and before.st_size > max_bytes:
            raise BudgetStop("training_file_byte_cap")
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                raise IdentityMismatch("input_replaced_before_read")
            if max_bytes is not None and opened.st_size > max_bytes:
                raise BudgetStop("training_file_byte_cap")
            raw = handle.read() if max_bytes is None else handle.read(max_bytes + 1)
            if max_bytes is not None and len(raw) > max_bytes:
                raise BudgetStop("training_file_byte_cap")
            after = os.fstat(handle.fileno())
        checked_path(root, relative)
        current = path.stat(follow_symlinks=False)
        # Windows Python can expose creation-time via path stat but change-time
        # via fstat. Compare ctime only within the same API, never across them.
        signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if (
            signature(before) != signature(opened)
            or signature(opened) != signature(after)
            or signature(after) != signature(current)
            or before.st_ctime_ns != current.st_ctime_ns
            or opened.st_ctime_ns != after.st_ctime_ns
            or len(raw) != after.st_size
        ):
            raise IdentityMismatch("input_changed_during_read")
        return raw
    except OSError as exc:
        raise IllegalDataAccess("input_unreadable") from exc


def _image_class():
    from PIL import Image

    if Image.__version__ != "12.2.0":
        raise ScopeBlocked("data_requires_reference_Pillow_12.2.0")
    return Image


def _shape(raw: bytes) -> tuple[int, int, int]:
    Image = _image_class()
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format != "PNG":
                raise SchemaError("training_input_must_be_png_bytes")
            width, height = image.size
            channels = len(image.getbands())
            if width < 1 or height < 1:
                raise SchemaError("invalid_image_shape")
            return height, width, channels
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise SchemaError("invalid_png_header") from exc


def _entries(path: Path) -> tuple[Path, ...]:
    try:
        bounded = []
        with os.scandir(path) as stream:
            for entry in stream:
                if len(bounded) >= 20000:
                    raise BudgetStop("training_directory_entry_cap")
                bounded.append(Path(entry.path))
        entries = tuple(bounded)
    except OSError as exc:
        raise IllegalDataAccess("training_directory_unavailable") from exc
    folded = set()
    for entry in entries:
        _component(entry.name)
        if entry.name.casefold() in folded:
            raise IllegalDataAccess("case_colliding_directory_entries")
        folded.add(entry.name.casefold())
    return entries


def _source(protocol: ManifestProtocol) -> dict:
    return (
        EXPECTED_SOURCE.copy()
        if protocol.dataset_id == DATASET_ID
        else {"scope": "synthetic_fixture; not_MVTec_or_archive_provenance"}
    )


def _training_paths(
    root: Path, protocol: ManifestProtocol
) -> tuple[tuple[str, str], ...]:
    entries = _entries(root)
    names = {entry.name for entry in entries}
    tasks = {name for name, _ in protocol.task_counts}
    if not tasks.issubset(names):
        raise IdentityMismatch("missing_training_category")
    for entry in entries:
        # lstat only; do not descend into or follow an unregistered directory.
        if entry.name not in tasks and stat.S_ISDIR(entry.lstat().st_mode):
            raise IdentityMismatch("unexpected_dataset_directory")
    paths = []
    for task, expected in protocol.task_counts:
        folder = checked_path(root, task + "/train/good")
        children = sorted(_entries(folder), key=lambda p: p.name)
        if len(children) != expected:
            raise IdentityMismatch("training_category_count_mismatch:" + task)
        for entry in children:
            if entry.suffix != ".png":
                raise SchemaError("unexpected_train_good_entry")
            paths.append((task, f"{task}/train/good/{entry.name}"))
    return tuple(paths)


def _assemble(
    protocol: ManifestProtocol, records: tuple[ManifestRecord, ...]
) -> TrainManifest:
    by_hash: dict[str, list[RecordID]] = {}
    for row in records:
        by_hash.setdefault(row.meta.raw_sha256, []).append(row.meta.opaque_id)
    aliases = tuple(tuple(ids) for _, ids in sorted(by_hash.items()) if len(ids) > 1)
    value = {
        "schema": "a0_manifest_v1",
        "dataset_id": protocol.dataset_id,
        "source": _source(protocol),
        "preprocess_id": PREPROCESS_ID,
        "task_order": [name for name, _ in protocol.task_counts],
        "task_counts": {name: count for name, count in protocol.task_counts},
        "count": len(records),
        "records": [
            {
                "opaque_id": row.meta.opaque_id,
                "arrival": row.meta.arrival,
                "file_sha256": row.meta.raw_sha256,
                "relative_path": row.relative_path,
                "bytes": row.bytes,
                "split": row.split,
                "task_audit_only": row.task_audit_only,
                "shape": list(row.shape),
            }
            for row in records
        ],
        "duplicate_content_aliases": [list(group) for group in aliases],
    }
    return TrainManifest(protocol, records, aliases, canonical_json(value) + "\n")


def build_manifest(
    dataset_root: str | Path,
    *,
    protocol: ManifestProtocol = REFERENCE,
    checkpoint=lambda: None,
    max_file_bytes: int | None = None,
) -> str:
    """Metadata only; default full3629 reference. Synthetic protocols have distinct IDs."""
    if not isinstance(protocol, ManifestProtocol):
        raise SchemaError("invalid_manifest_protocol")
    protocol.validate()
    root = checked_root(dataset_root)
    checkpoint()
    records = []
    for task, relative in _training_paths(root, protocol):
        checkpoint()
        raw = read_verified_bytes(root, relative, max_bytes=max_file_bytes)
        file_sha = hashlib.sha256(raw).hexdigest()
        opaque = hashlib.sha256(
            canonical_json([protocol.dataset_id, relative, file_sha]).encode("utf-8")
        ).hexdigest()
        records.append(
            ManifestRecord(
                RecordMeta(RecordID(opaque), ArrivalIndex(len(records) + 1), file_sha),
                relative,
                len(raw),
                "train/good",
                task,
                _shape(raw),
            )
        )
    result = _assemble(protocol, tuple(records)).canonical
    checkpoint()
    return result
