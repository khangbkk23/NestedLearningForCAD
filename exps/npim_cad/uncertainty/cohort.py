# exps/npim_cad/uncertainty/cohort.py
"""Normal-only calibration cohort for EXP-NPIM-03.

The only eligible source is the 59 P0c `probes` (all train/good).  They are
content-disjoint from the 350 writer images and the P1 DEV/sealed identities
(prior NPIM-01 audit), and were previously used only for NPIM-00 normal
telemetry.  No P1 DEV image or mask is opened here.

Split rule (frozen): within each category, sort by raw_png_sha256; first
floor(n/2) are fit, remaining are the normal-only witness set.  Category is used
only for causal eligibility and this split, never as a scorer input.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

from exps.npim_cad.npim01_contract import MVTEC, sha256_file

ROOT = Path(__file__).resolve().parents[3]
P0C_PROTOCOL = ROOT / "results/hope_cad/task_p/p0c/protocol_resolved.json"
P0C_SHA = "0090f697518587e51eda91a3bd5def44b5ceb2c572b49eb75f5b9a0dfd4fb182"
CATEGORIES = ("bottle", "carpet", "hazelnut")
BOUNDARY_CATEGORIES: Dict[int, Tuple[str, ...]] = {
    100: ("bottle",),
    200: ("bottle", "carpet"),
    300: ("bottle", "carpet", "hazelnut"),
    350: ("bottle", "carpet", "hazelnut"),
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ProbeRecord:
    index: int
    relative_path: str
    sha256: str
    category: str
    split: str  # "fit" | "witness"


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_probes() -> List[ProbeRecord]:
    if not P0C_PROTOCOL.is_file():
        raise FileNotFoundError(P0C_PROTOCOL)
    if sha256_file(P0C_PROTOCOL) != P0C_SHA:
        raise RuntimeError("P0c protocol SHA-256 mismatch")
    protocol = json.loads(P0C_PROTOCOL.read_text())
    probes = protocol.get("probes")
    updates = protocol.get("updates")
    if not isinstance(probes, list) or len(probes) != 59:
        raise RuntimeError("expected exactly 59 P0c probes")
    if not isinstance(updates, list) or len(updates) != 350:
        raise RuntimeError("expected the 350-image writer stream")
    writer_paths = {row["relative_path"] for row in updates}
    writer_hashes = {row["raw_png_sha256"] for row in updates}
    records: List[ProbeRecord] = []
    seen_paths, seen_hashes = set(), set()
    counts = {category: 0 for category in CATEGORIES}
    for index, row in enumerate(probes):
        if row.get("role") != "probe":
            raise RuntimeError(f"probe {index} has role {row.get('role')!r}")
        relative = row["relative_path"]
        parts = Path(relative).parts
        if len(parts) != 4 or parts[1:3] != ("train", "good") or parts[0] not in CATEGORIES:
            raise RuntimeError(f"probe {index} is not a normal train/good image: {relative}")
        if relative in writer_paths or row["raw_png_sha256"] in writer_hashes:
            raise RuntimeError(f"probe {index} overlaps the writer stream")
        if relative in seen_paths or row["raw_png_sha256"] in seen_hashes:
            raise RuntimeError(f"duplicate probe identity/content at {index}")
        raw = MVTEC / relative
        if not raw.is_file():
            raise RuntimeError(f"missing probe image: {relative}")
        if sha256_file(raw) != row["raw_png_sha256"]:
            raise RuntimeError(f"probe content changed: {relative}")
        seen_paths.add(relative)
        seen_hashes.add(row["raw_png_sha256"])
        counts[row["category"]] += 1
        records.append(ProbeRecord(index, relative, row["raw_png_sha256"], row["category"], "unassigned"))
    if counts != {"bottle": 19, "carpet": 20, "hazelnut": 20}:
        raise RuntimeError(f"unexpected probe category counts: {counts}")
    # Deterministic split per category by content hash.
    split_records: List[ProbeRecord] = []
    for category in CATEGORIES:
        group = sorted([record for record in records if record.category == category], key=lambda record: record.sha256)
        half = len(group) // 2
        for position, record in enumerate(group):
            split_records.append(
                ProbeRecord(record.index, record.relative_path, record.sha256, record.category,
                            "fit" if position < half else "witness")
            )
    split_records.sort(key=lambda record: record.index)
    return split_records


def eligible(boundary: int) -> Tuple[str, ...]:
    if boundary not in BOUNDARY_CATEGORIES:
        raise ValueError(f"unknown boundary {boundary}")
    return BOUNDARY_CATEGORIES[boundary]


def fit_records(records: Sequence[ProbeRecord], boundary: int) -> List[ProbeRecord]:
    return [record for record in records if record.split == "fit" and record.category in eligible(boundary)]


def witness_records(records: Sequence[ProbeRecord], boundary: int) -> List[ProbeRecord]:
    return [record for record in records if record.split == "witness" and record.category in eligible(boundary)]


def load_normal_image(record: ProbeRecord) -> torch.Tensor:
    path = MVTEC / record.relative_path
    if sha256_file(path) != record.sha256:
        raise RuntimeError(f"probe content changed: {record.relative_path}")
    with Image.open(path) as source:
        image = source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        array = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(array)
    mean = torch.tensor(MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(STD, dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


def audit(records: Sequence[ProbeRecord]) -> Dict[str, Any]:
    counts = {
        (category, split): sum(1 for record in records if record.category == category and record.split == split)
        for category in CATEGORIES
        for split in ("fit", "witness")
    }
    split_payload = [
        {"relative_path": record.relative_path, "sha256": record.sha256,
         "category": record.category, "split": record.split}
        for record in records
    ]
    return {
        "source_protocol": str(P0C_PROTOCOL.relative_to(ROOT)),
        "source_protocol_sha256": P0C_SHA,
        "probe_count": len(records),
        "category_split_counts": {f"{category}:{split}": count for (category, split), count in counts.items()},
        "fit_witness_by_boundary": {
            str(boundary): {
                "categories": list(eligible(boundary)),
                "fit": len(fit_records(records, boundary)),
                "witness": len(witness_records(records, boundary)),
            }
            for boundary in BOUNDARY_CATEGORIES
        },
        "split_digest": _canonical_digest(split_payload),
        "all_train_good": True,
        "writer_disjoint": True,
        "dev_sealed_disjoint_prior_audit": "verified in NPIM-01; probes are train/good, DEV/sealed are test/",
        "prior_exposure": "the 59 probes were used read-only for NPIM-00 normal telemetry; they are not fresh confirmation",
    }
