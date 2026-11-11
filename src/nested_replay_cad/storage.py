"""Evaluator/output resource guards, independent of learner state."""

import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

OUTPUT_CAP = 18 * 1024**3


FREE_RESERVE = 2 * 1024**3


CHUNK_TARGET_BYTES = 1024**3


def census(root):
    root = Path(root).resolve()
    return sum(
        p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink()
    )


class DiskGuard:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.last = -float("inf")
        self.peak_observed = 0

    def check(self, force=False, additional_bytes=0):
        now = time.monotonic()
        if not force and not additional_bytes and now - self.last < 30:
            return
        used = census(self.root)
        free = shutil.disk_usage(self.root).free
        self.peak_observed = max(self.peak_observed, used)
        self.last = now
        observation = dict(
            working_bytes=used,
            free_bytes=free,
            additional_reserved_bytes=additional_bytes,
            output_cap_bytes=OUTPUT_CAP,
            free_reserve_bytes=FREE_RESERVE,
        )
        if force or additional_bytes:
            print("FULLTEST_DISK_OBSERVATION", json.dumps(observation), flush=True)
        if used + additional_bytes > OUTPUT_CAP:
            raise RuntimeError(
                "Working output cap including raw/ZIP duplicates "
                + json.dumps(observation)
            )
        if free < FREE_RESERVE + additional_bytes:
            raise RuntimeError(
                "Filesystem free-space/export reserve " + json.dumps(observation)
            )
        return dict(
            working_bytes=used,
            filesystem_free_bytes=free,
            peak_observed_working_bytes=self.peak_observed,
            additional_reserved_bytes=additional_bytes,
        )


def tensor_bytes(values):
    return sum(v.numel() * v.element_size() for v in values)


def archive_tensors(destination, relative_root, working, stem, deadline):
    """Close and verify bounded ZIP chunks before removing raw duplicates."""
    destination = Path(destination).resolve()
    relative_root = Path(relative_root).resolve()
    working = Path(working).resolve()
    guard = DiskGuard(working)
    files = sorted(destination.rglob("*.safetensors"))
    groups = []
    group = []
    size = 0
    for p in files:
        if p.is_symlink() or not p.resolve().is_relative_to(destination):
            raise ValueError("Unsafe tensor path")
        if group and size + p.stat().st_size > CHUNK_TARGET_BYTES:
            groups.append(group)
            group = []
            size = 0
        group.append(p)
        size += p.stat().st_size
    if group:
        groups.append(group)
    result = dict(archives=[], files=[])
    for i, group in enumerate(groups, 1):
        raw = sum(p.stat().st_size for p in group)
        guard.check(force=True, additional_bytes=int(raw * 1.01) + 1024**2)
        archive_path = working / (stem + f"_part{i:03}.zip")
        inventory = []
        with zipfile.ZipFile(
            archive_path, "x", zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            for p in group:
                if time.monotonic() > deadline - 120:
                    raise TimeoutError("Chunk export reserve")
                h = hashlib.sha256()
                with p.open("rb") as stream:
                    while block := stream.read(1024**2):
                        h.update(block)
                name = p.relative_to(relative_root).as_posix()
                inventory.append(
                    dict(
                        path=name,
                        sha256=h.hexdigest(),
                        bytes=p.stat().st_size,
                        archive=archive_path.name,
                    )
                )
                archive.write(p, name)
        with zipfile.ZipFile(archive_path) as archive:
            for item in inventory:
                h = hashlib.sha256()
                with archive.open(item["path"]) as stream:
                    while block := stream.read(1024**2):
                        h.update(block)
                if h.hexdigest() != item["sha256"]:
                    raise ValueError("Chunk readback hash")
        h = hashlib.sha256()
        with archive_path.open("rb") as stream:
            while block := stream.read(1024**2):
                h.update(block)
        result["archives"].append(
            dict(
                path=archive_path.name,
                bytes=archive_path.stat().st_size,
                sha256=h.hexdigest(),
            )
        )
        result["files"].extend(inventory)
        progress = destination / "CHUNK_PROGRESS.json"
        staged = progress.with_suffix(".json.partial")
        staged.write_text(json.dumps(result, indent=2) + "\n")
        os.replace(staged, progress)
        for p in group:
            p.unlink()
        guard.check(force=True)
        print(
            "FULLTEST_CHUNK_VERIFIED",
            archive_path.name,
            len(group),
            archive_path.stat().st_size,
            flush=True,
        )
    return result
