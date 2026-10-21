# exps/npim_cad/npim01_contract.py
"""EXP-NPIM-01 frozen contract: 12 NPIM-00 states + the admitted 60-image P1 DEV whitelist.

This module is verification-only.  It never opens a numeric annotation mask; it
only checks that each admitted mask path exists.  Mask bytes are read exclusively
by the E2 metrics stage after the E1 prediction payload is sealed.

Authoritative stream (confirmed by the User after the G0 conflict):
100 bottle -> 100 carpet -> 100 hazelnut -> 50 bottle-return normal images.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

from models.hope_cad.npim_cad import NPIMConfig, NPIMDetector
from models.hope_cad.npim_state import load_state_file

ROOT = Path(__file__).resolve().parents[2]
P1_PROTOCOL = ROOT / "results/hope_cad/task_p/p1/protocol_resolved.json"
P0C_PROTOCOL = ROOT / "results/hope_cad/task_p/p0c/protocol_resolved.json"
PILOT = ROOT / "results/hope_cad/npim_00/npim00_pilot_20261021_cpu"
CHECKPOINT = ROOT / "checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth"
PROTOCOL = ROOT / "exps/npim_cad/protocol_npim01.md"
MVTEC = ROOT / "data/mvtec"

P1_SHA = "11c1cc03b20cb1f79a6e8b7e472bb6f44758b43046b61043368ea11b4ce1e08f"
P0C_SHA = "0090f697518587e51eda91a3bd5def44b5ceb2c572b49eb75f5b9a0dfd4fb182"
CP_SHA = "755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415"
CP_BYTES = 343_282_995
COHORT_DIGEST = "63d1dbb48e9b6dd4665e50a0fa2702616834adddf95fd9889323804c21875447"
WRITER_DIGEST = "9549f24aae822ea8cccdbe63dd9731790633cf5eba80b37506c11c2944df3c24"
PILOT_MANIFEST_SHA = "e310d1e4edc4bf16b43dc5ecc0342fac72eb02f9fd6eed414ec5047cd6ab486e"
PILOT_EVENTS_SHA = "984bd53a78ce7e2e3c67b0d93eadc9a0c97580b9baf6ab98ae9877e3af07c9aa"

BOUNDARIES: tuple[int, ...] = (100, 200, 300, 350)
ADMITTED: Dict[int, tuple[str, ...]] = {
    100: ("bottle",),
    200: ("bottle", "carpet"),
    300: ("bottle", "carpet", "hazelnut"),
    350: ("bottle", "carpet", "hazelnut"),
}
CATEGORIES = ("bottle", "carpet", "hazelnut")
LABELS = (0, 1)
ARMS: Dict[str, Dict[str, Any]] = {
    "C1": {"r": 145, "periods": (1,), "lambdas": (0.1,), "weights": (1.0,)},
    "C2": {"r": 64, "periods": (1, 1, 1), "lambdas": (0.1, 1.0, 10.0), "weights": (1.0 / 3.0,) * 3},
    "C3": {"r": 64, "periods": (1, 16, 128), "lambdas": (0.1, 1.0, 10.0), "weights": (1.0 / 3.0,) * 3},
}

STATE_SHA256: Dict[tuple[str, int], str] = {
    ("C1", 100): "159f834563391df8ccfe2675496f7bfcfecd1c79f0b3c4e123461ecdc135f1f6",
    ("C1", 200): "c3f780dc88f180b887255f9ee6d6677c58c4875e7ff8d0dc067ef204322683dc",
    ("C1", 300): "8376713e0cd952736067d4234dc3bbf2a4bcdff27df1e6c4fbdd2c5fb34de993",
    ("C1", 350): "0422b417df7cf24c57674ef8ef33925e35698ebb98779edf1e98de8430b50c3e",
    ("C2", 100): "6df5e6cfd413be03a347e369dd37adcf38429ed7385458ed0b6c5bca689d84e3",
    ("C2", 200): "5428672189575f9749f3c46c325658e9f40932522a80834d7c914c72c7edd7ea",
    ("C2", 300): "00aa754b13867302d0e146a36ccd08add2e7c4c557988a1d36ff4e19d7d88766",
    ("C2", 350): "599ff8fc4a749c25fe4938afa001a9a6740d11ec649b830705b7fe4644b0380a",
    ("C3", 100): "df8699c56f0a76c76f83d9c4f548c0c07ec8158a135842b00d910c6406f554e5",
    ("C3", 200): "8af71237794ce4b19894a18eb96d045538ca91c3e2216f32421094a4c4d05417",
    ("C3", 300): "02ca0adbc4d60576d5b6dbc36205094a0e662ce4c762ff0a60a14dbf17162e7c",
    ("C3", 350): "be03f5c06ea9054d5d95661d74b542eaada0d11a2afab57b48772b31272c3d1b",
}

STATE_HASH: Dict[tuple[str, int], str] = {
    ("C1", 100): "c6917e021ad46e5167fa928cb3e7dad0b05d5987af8d447e15f5b50e1054aebe",
    ("C1", 200): "8dea3c124ec85ee6ecc59ef3e6c031b6e6963d17239c70999b77d81e5a4f7e20",
    ("C1", 300): "b5fc53f0bbf1a8fdcf6abeeea509f244bf9da5783a112f9e26a1d660aa254e83",
    ("C1", 350): "cac7cb82d8ed3c54683b1a7be5c0898a667018737615194671fce8f8c3b90222",
    ("C2", 100): "f7a7c23ac3f0232361eefe7fd958e0ddc5c17baea94dc4af908234a93a34204e",
    ("C2", 200): "69041a65ae25e4f7ee46312cce18472596b20cee5d058105c5f1214c402c6463",
    ("C2", 300): "269a9bf4b742143a2de1cfd8a1a6afb0fae3ac3d1c114fca4b151ab576a1e9c4",
    ("C2", 350): "2097d2d417ef3a963dff2cf0f593d8577ca172757a1dc97d25d7e210cf07359e",
    ("C3", 100): "4f0952385b21bddae5da70c02368d68739dd1586a75139f99530ae62f90f1bef",
    ("C3", 200): "5bc1a87335cfecc2af8b2c60bd4301c04091df4b242d9c9181af3ae04b50a0fc",
    ("C3", 300): "80554fade42761da69421a2054c00b519c1342ddd4ee1c9dbfeb07145fbc680c",
    ("C3", 350): "f3651beba2344d412445cfa35325b31ecbd6201c8124f57ea8d3b031f611fb97",
}

EXPECTED_READY: Dict[tuple[str, int], tuple[bool, ...]] = {
    ("C1", 100): (True,), ("C1", 200): (True,), ("C1", 300): (True,), ("C1", 350): (True,),
    ("C2", 100): (True, True, True), ("C2", 200): (True, True, True),
    ("C2", 300): (True, True, True), ("C2", 350): (True, True, True),
    ("C3", 100): (True, True, False), ("C3", 200): (True, True, True),
    ("C3", 300): (True, True, True), ("C3", 350): (True, True, True),
}


class ContractError(RuntimeError):
    """Raised when a pinned NPIM-01 artifact or access rule is violated."""


@dataclass(frozen=True)
class DevRecord:
    index: int
    relative_path: str
    sha256: str
    category: str
    label: int
    hw: tuple[int, int]
    mask_path: str | None
    mask_sha256: str | None


@dataclass(frozen=True)
class Contract:
    records: tuple[DevRecord, ...]
    protocol_sha256: str
    p1_sha256: str
    p0c_sha256: str
    checkpoint_sha256: str
    cohort_digest: str
    writer_digest: str
    state_sha256: Mapping[tuple[str, int], str]
    state_hash: Mapping[tuple[str, int], str]
    source_sha256: Mapping[str, str]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def arm_config(arm: str) -> NPIMConfig:
    spec = ARMS[arm]
    return NPIMConfig(
        r=int(spec["r"]),
        periods=tuple(int(value) for value in spec["periods"]),
        lambdas=tuple(float(value) for value in spec["lambdas"]),
        weights=tuple(float(value) for value in spec["weights"]),
        projection_seed=20261021,
    )


def admitted_categories(boundary: int) -> tuple[str, ...]:
    if boundary not in ADMITTED:
        raise ContractError(f"unknown NPIM-01 boundary: {boundary}")
    return ADMITTED[boundary]


def source_hashes() -> Dict[str, str]:
    paths = [
        "models/hope_cad/npim_context.py",
        "models/hope_cad/npim_proximal.py",
        "models/hope_cad/npim_state.py",
        "models/hope_cad/npim_cad.py",
        "exps/npim_cad/npim01_contract.py",
        "exps/npim_cad/npim01_metrics.py",
        "exps/npim_cad/npim01_evaluate.py",
        "exps/npim_cad/protocol_npim01.md",
    ]
    return {relative: sha256_file(ROOT / relative) for relative in paths if (ROOT / relative).is_file()}


def _load_json(path: Path, expected_sha: str | None = None) -> Any:
    if not path.is_file():
        raise ContractError(f"missing required protocol artifact: {path}")
    actual = sha256_file(path)
    if expected_sha is not None and actual != expected_sha:
        raise ContractError(f"SHA-256 mismatch for {path}: {actual} != {expected_sha}")
    return json.loads(path.read_text())


def _dev_records(p1: Mapping[str, Any]) -> tuple[DevRecord, ...]:
    evaluations = p1.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != 60:
        raise ContractError("P1 protocol must contain exactly 60 development_eval records")
    records: List[DevRecord] = []
    seen_paths, seen_hashes = set(), set()
    for index, row in enumerate(evaluations):
        if row.get("role") != "development_eval":
            raise ContractError(f"record {index} is not development_eval")
        relative = row["relative_path"]
        parts = Path(relative).parts
        if len(parts) != 4 or parts[1] != "test" or parts[0] != row["category"]:
            raise ContractError(f"record {index} has an invalid development path: {relative}")
        if row["category"] not in CATEGORIES:
            raise ContractError(f"record {index} has an unknown category")
        if row["label"] not in LABELS:
            raise ContractError(f"record {index} has an invalid label")
        if row["label"] == 1 and not row.get("mask_path"):
            raise ContractError(f"record {index} is anomalous but has no mask path")
        if row["label"] == 0 and row.get("mask_path"):
            raise ContractError(f"record {index} is normal but has a mask path")
        raw = MVTEC / relative
        if not raw.is_file():
            raise ContractError(f"missing admitted DEV image: {relative}")
        if sha256_file(raw) != row["sha256"]:
            raise ContractError(f"admitted DEV image content changed: {relative}")
        if relative in seen_paths or row["sha256"] in seen_hashes:
            raise ContractError(f"duplicate DEV identity/content: {relative}")
        seen_paths.add(relative)
        seen_hashes.add(row["sha256"])
        hw = row.get("hw")
        if not isinstance(hw, list) or len(hw) != 2 or min(int(hw[0]), int(hw[1])) <= 0:
            raise ContractError(f"record {index} has invalid native geometry")
        mask_path = row.get("mask_path")
        if mask_path is not None:
            # Existence only; mask bytes are never read by the contract.
            if not (MVTEC / mask_path).is_file():
                raise ContractError(f"missing admitted mask path (not opened): {mask_path}")
        records.append(
            DevRecord(
                index=index,
                relative_path=relative,
                sha256=row["sha256"],
                category=row["category"],
                label=int(row["label"]),
                hw=(int(hw[0]), int(hw[1])),
                mask_path=mask_path,
                mask_sha256=row.get("mask_sha256"),
            )
        )
    counts = {
        (category, label): sum(1 for row in records if row.category == category and row.label == label)
        for category in CATEGORIES
        for label in LABELS
    }
    expected = {(category, label): 10 for category in CATEGORIES for label in LABELS}
    if counts != expected:
        raise ContractError(f"DEV cohort is not 10 normal + 10 anomalous per category: {counts}")
    return tuple(records)


def _writer_digest(p1: Mapping[str, Any]) -> str:
    updates = p1.get("updates")
    if not isinstance(updates, list) or len(updates) != 350:
        raise ContractError("P1 protocol must contain exactly 350 writer updates")
    runs = []
    previous = None
    for row in updates:
        if row["category"] != previous:
            runs.append([int(row["event"]), row["category"]])
            previous = row["category"]
    expected_runs = [[1, "bottle"], [101, "carpet"], [201, "hazelnut"], [301, "bottle"]]
    if runs != expected_runs:
        raise ContractError(f"writer stream order is not the authoritative 100/100/100/50 order: {runs}")
    return canonical_digest(
        [
            {"event": int(row["event"]), "relative_path": row["relative_path"],
             "sha256": row["sha256"], "category": row["category"]}
            for row in updates
        ]
    )


def load_contract(*, verify_state_files: bool = True) -> Contract:
    """Verify all pinned provenance. Never reads a mask byte."""

    if not PROTOCOL.is_file():
        raise ContractError(f"missing NPIM-01 protocol: {PROTOCOL}")
    protocol_sha = sha256_file(PROTOCOL)
    p1 = _load_json(P1_PROTOCOL, P1_SHA)
    p0c_sha = sha256_file(P0C_PROTOCOL)
    if p0c_sha != P0C_SHA:
        raise ContractError(f"P0c protocol SHA-256 mismatch: {p0c_sha}")
    if p1.get("p0c_protocol_sha256") != P0C_SHA:
        raise ContractError("P1 protocol does not pin the expected P0c protocol")
    if p1.get("representation", {}).get("checkpoint_sha256") != CP_SHA:
        raise ContractError("P1 protocol checkpoint pin mismatch")
    if not CHECKPOINT.is_file():
        raise ContractError(f"missing frozen checkpoint: {CHECKPOINT}")
    checkpoint_sha = sha256_file(CHECKPOINT)
    if checkpoint_sha != CP_SHA or CHECKPOINT.stat().st_size != CP_BYTES:
        raise ContractError("frozen checkpoint identity mismatch")

    records = _dev_records(p1)
    cohort_digest = canonical_digest(
        [
            {"relative_path": row.relative_path, "sha256": row.sha256, "category": row.category,
             "label": row.label, "hw": list(row.hw), "role": "development_eval",
             "mask_path": row.mask_path, "mask_sha256": row.mask_sha256}
            for row in records
        ]
    )
    if cohort_digest != COHORT_DIGEST:
        raise ContractError(f"DEV cohort identity digest mismatch: {cohort_digest}")
    writer_digest = _writer_digest(p1)
    if writer_digest != WRITER_DIGEST:
        raise ContractError(f"writer stream digest mismatch: {writer_digest}")

    writer_paths = {row["relative_path"] for row in p1["updates"]}
    writer_hashes = {row["sha256"] for row in p1["updates"]}
    dev_paths = {row.relative_path for row in records}
    dev_hashes = {row.sha256 for row in records}
    if writer_paths & dev_paths or writer_hashes & dev_hashes:
        raise ContractError("writer and DEV identities/content overlap")
    sealed = set(p1.get("sealed_identity_paths", []))
    if sealed & dev_paths:
        raise ContractError("DEV and sealed identity sets overlap")
    if sha256_file(PILOT / "manifest.json") != PILOT_MANIFEST_SHA:
        raise ContractError("NPIM-00 pilot manifest changed")
    if sha256_file(PILOT / "events.jsonl") != PILOT_EVENTS_SHA:
        raise ContractError("NPIM-00 event log changed")

    if verify_state_files:
        for (arm, boundary), expected in STATE_SHA256.items():
            path = PILOT / "states" / f"{arm}_{boundary}.pt"
            if not path.is_file():
                raise ContractError(f"missing NPIM-00 state: {path}")
            actual = sha256_file(path)
            if actual != expected:
                raise ContractError(f"state file SHA-256 mismatch for {arm}@{boundary}: {actual}")

    return Contract(
        records=records,
        protocol_sha256=protocol_sha,
        p1_sha256=P1_SHA,
        p0c_sha256=P0C_SHA,
        checkpoint_sha256=CP_SHA,
        cohort_digest=COHORT_DIGEST,
        writer_digest=WRITER_DIGEST,
        state_sha256=dict(STATE_SHA256),
        state_hash=dict(STATE_HASH),
        source_sha256=source_hashes(),
    )


def load_state_detectors(device: str | torch.device = "cpu") -> Dict[tuple[str, int], NPIMDetector]:
    """Load all 12 pinned states and verify their reloaded state hashes."""

    detectors: Dict[tuple[str, int], NPIMDetector] = {}
    projections: Dict[str, torch.Tensor] = {}
    for (arm, boundary), expected in STATE_HASH.items():
        detector = NPIMDetector(arm_config(arm), device=device)
        detector.load_state_dict(load_state_file(PILOT / "states" / f"{arm}_{boundary}.pt"))
        actual = detector.state_hash()
        if actual != expected:
            raise ContractError(f"reloaded state hash mismatch for {arm}@{boundary}: {actual}")
        ready = tuple(level.ready for level in detector.levels)
        if ready != EXPECTED_READY[(arm, boundary)]:
            raise ContractError(f"unexpected ready flags for {arm}@{boundary}: {ready}")
        projection = detector.projection.detach().to("cpu", copy=True)
        if arm not in projections:
            projections[arm] = projection
        elif not torch.equal(projections[arm], projection):
            raise ContractError(f"projection differs across {arm} boundary states")
        detectors[(arm, boundary)] = detector
    return detectors
