"""EXP-CSG-01 — distance-preserving quantized normal-support memory.

Frozen protocol constants, state container, byte accounting and decision bands.
The frozen stream, feature cache and exact Euclidean NN utility are reused from
the sealed CSG-00 implementation without modifying it.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from exps.csg00 import contract as C0

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------

ROOT = C0.ROOT
RUN_ID = "csg01_g0g2_20261022"
RUN_DIR = ROOT / "results" / "hope_cad" / "csg01" / RUN_ID
G0_LOCK_PATH = RUN_DIR / "G0_LOCK.json"
G2_SEAL_PATH = RUN_DIR / "G2_SEAL.json"
TASK_RECORD = ROOT / "agents" / "tasks" / "task_csg01_distance_preserving_compression.md"
REPORT_PATH = ROOT / "agents" / "reports" / "task_csg01_distance_preserving_compression.md"

CHECKPOINT = C0.CHECKPOINT
CHECKPOINT_SHA256 = C0.CHECKPOINT_SHA256
CHECKPOINT_BYTES = C0.CHECKPOINT_BYTES
CACHE_DIR = C0.CACHE_DIR
CACHE_SHA256 = C0.CACHE_SHA256
P0C_PROTOCOL = C0.P0C_PROTOCOL
P0C_PROTOCOL_SHA256 = C0.P0C_PROTOCOL_SHA256
AGENTS_MD = C0.AGENTS_MD
AGENTS_MD_PIN = C0.AGENTS_MD_PIN
SOURCE_MANIFEST = C0.SOURCE_MANIFEST
SOURCE_MANIFEST_DIGEST = C0.SOURCE_MANIFEST_DIGEST

# Sealed CSG-00 evidence used as historical context / provenance anchors.
CSG00_RUN = ROOT / "results" / "hope_cad" / "csg00" / "csg00_g0g2_20261021_r3"
CSG00_LOCK = CSG00_RUN / "G0_LOCK.json"
CSG00_SEAL = CSG00_RUN / "G2_SEAL.json"
CSG00_REPORT = ROOT / "agents" / "reports" / "task_csg00_compact_geometry.md"

# ---------------------------------------------------------------------------
# Frozen geometry and stream
# ---------------------------------------------------------------------------

GRID = C0.GRID
PATCHES = C0.PATCHES
DIM = C0.DIM
BIN_SIDE = C0.BIN_SIDE
BINS_PER_SIDE = C0.BINS_PER_SIDE
BINS = C0.BINS
TOKENS_PER_BIN = C0.TOKENS_PER_BIN
CATEGORIES = C0.CATEGORIES
BOUNDARIES = C0.BOUNDARIES
PROBE_BLOCKS = C0.PROBE_BLOCKS
ELIGIBLE_PREFIX = C0.ELIGIBLE_PREFIX

# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------

SELECTION_COUNT = 2_500                     # Arms B, C, D and their shared identities
QUOTA_BASE = SELECTION_COUNT // BINS        # 51
QUOTA_REMAINDER = SELECTION_COUNT - QUOTA_BASE * BINS  # 1

# E budget derived by the G0 empty-schema byte match (locked at 632).
SIZE_MATCHED_COUNT = 632
SIZE_MATCHED_BASE = SIZE_MATCHED_COUNT // BINS   # 12
SIZE_MATCHED_REMAINDER = SIZE_MATCHED_COUNT - SIZE_MATCHED_BASE * BINS  # 44

RANDOM_SEED = 20261022
FPS_INIT = "descriptor closest to the bin mean (medoid); ties -> lowest candidate index"
FPS_TIE = "argmax over min-distance-to-selected; ties -> lowest candidate index"
FPS_EXCLUDE = "selected candidates are masked out of subsequent argmax"
EXTRA_QUOTA_RULE = ("all bins have exactly 16*t causal descriptors, so the largest-candidate-count "
                    "tie always resolves to the lowest bin index; the extra descriptor is bin 0 "
                    "for B/C/D, and bins 0..43 receive +1 for E")

QUANT_DESCRIPTION = (
    "per-vector affine min/max: xmin=min(x), xmax=max(x), step=(xmax-xmin)/255; "
    "q=clamp(rint((x-xmin)/step),0,255) with round-half-to-even (numpy.rint); "
    "xhat=xmin+step*q; xmin and step stored as float32 and used in decode; "
    "constant vectors use step=0, q=0, xhat=xmin"
)
QUANT_TOL = 1.0e-5

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

BOOTSTRAP_REPLICATES_T350 = 10_000
BOOTSTRAP_REPLICATES_OTHER = 2_000
BOOTSTRAP_SEED = 20261022
BOOTSTRAP_CI = (2.5, 97.5)
TOP_FRACTIONS = (0.01, 0.05)
DTYPE_MAIN = "float32"
SCORE_ORIENTATION = "high = more anomalous; unsquared Euclidean distance"

# ---------------------------------------------------------------------------
# Locked decision gates (feasibility heuristics, not Pixel-AP guarantees)
# ---------------------------------------------------------------------------

GATES = {
    "quantization_C_vs_B": {
        "image_max_spearman_min_each_category": 0.99,
        "top5_precision_min": 0.95,
        "observed_error_within_delta": True,
        "tail5_mean_abs_error_max_factor": 2.0,
        "note": "every observed |s_C-s_B| must be <= delta_bin + QUANT_TOL; a violation "
                "beyond numerical tolerance is an integrity failure, not a tuning signal",
    },
    "selection_B_vs_A": {
        "image_max_spearman_min_each_category": 0.80,
        "top5_precision_min": 0.65,
        "report_epsilon_usefulness": True,
    },
    "memory_efficiency": {
        "C_vs_D_byte_ratio": [0.95, 1.05],
        "C_vs_E_byte_ratio": [0.95, 1.05],
        "C_D_same_cardinality_and_schema": True,
        "meaningful_advantage_min": 0.05,
        "category_loss_max": 0.10,
        "primary_endpoint": "image-max Spearman against Arm A (macro)",
        "co_primary_endpoint": "top-5% patch precision against Arm A (macro)",
        "meaningful_advantage_rule": "C - competitor >= 0.05 on primary or co-primary, "
                                     "paired whole-image bootstrap 95% CI lower > 0, and no "
                                     "category loss > 0.10 on the chosen endpoint",
    },
}

DECISION_ENUM = (
    "CSG01_BLOCKED",
    "CSG01_NEGATIVE_QUANTIZATION",
    "CSG01_NEGATIVE_SELECTION_OR_BUDGET",
    "CSG01_INCONCLUSIVE",
    "CSG01_GO_REQUEST_FRESH_REAL_ANOMALY_GATE",
)

STOP_CRITERIA = [
    "G0 integrity, stream identity, feature provenance or historical-hash failure",
    "any G1 test failure",
    "theoretical inequality violation beyond QUANT_TOL (implementation/mathematical failure)",
    "non-finite score, selected identity or quantization value",
    "C/D or C/E persistent-byte ratio outside [0.95, 1.05]",
    "FPS selection infeasible within the preregistered resource cap",
    "CPU/CUDA score parity max abs difference > 1.0e-3",
    "any anomaly, test, ground-truth, sealed or consumed-DEV payload access",
]

# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

MAGIC = b"CSG01ST1"
SCHEMA = "csg01_state_v1"


def b_layout(n: int = SELECTION_COUNT):
    return (
        ("counts", "int32", (BINS,)),
        ("offsets", "int32", (BINS,)),
        ("desc", "float32", (n, DIM)),
    )


def q_layout(n: int = SELECTION_COUNT):
    return (
        ("counts", "int32", (BINS,)),
        ("offsets", "int32", (BINS,)),
        ("q", "uint8", (n, DIM)),
        ("xmin", "float32", (n,)),
        ("step", "float32", (n,)),
    )


def e_layout(n: int = SIZE_MATCHED_COUNT):
    return (
        ("counts", "int32", (BINS,)),
        ("offsets", "int32", (BINS,)),
        ("desc", "float32", (n, DIM)),
    )


def layout_for(arm: str):
    if arm in {"B", "E"}:
        return b_layout(SELECTION_COUNT if arm == "B" else SIZE_MATCHED_COUNT)
    if arm in {"C", "D"}:
        return q_layout(SELECTION_COUNT)
    raise ValueError(f"unknown arm {arm}")


def _array_bytes(arrays: Mapping[str, np.ndarray], layout) -> bytes:
    parts = []
    for name, dtype, shape in layout:
        arr = np.ascontiguousarray(np.asarray(arrays[name], dtype=np.dtype(dtype)))
        if tuple(arr.shape) != tuple(shape):
            raise ValueError(f"state array {name}: {arr.shape} != {shape}")
        parts.append(arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes(order="C"))
    return b"".join(parts)


def build_state_blob(arm: str, boundary: int, params: Mapping[str, Any],
                     arrays: Mapping[str, np.ndarray], layout) -> bytes:
    payload = _array_bytes(arrays, layout)
    header = {
        "schema": SCHEMA,
        "arm": arm,
        "boundary": int(boundary),
        "params": dict(params),
        "layout": [{"name": n, "dtype": d, "shape": list(s)} for n, d, s in layout],
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    hjson = C0.canonical_json(header).encode("utf-8")
    return MAGIC + struct.pack("<QQ", len(hjson), len(payload)) + hjson + payload


def write_state(path: Path | str, arm: str, boundary: int, params: Mapping[str, Any],
                arrays: Mapping[str, np.ndarray], exclusive: bool = True) -> Dict[str, Any]:
    layout = layout_for(arm)
    blob = build_state_blob(arm, boundary, params, arrays, layout)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb" if exclusive else "wb") as fh:
        fh.write(blob)
    return {
        "path": str(path),
        "bytes": len(blob),
        "payload_bytes": int(json.loads(blob[24:24 + struct.unpack("<Q", blob[8:16])[0]]
                                       .decode("utf-8"))["payload_bytes"]),
        "header_bytes": struct.unpack("<Q", blob[8:16])[0],
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def read_state(path: Path | str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    raw = Path(path).read_bytes()
    if raw[:8] != MAGIC:
        raise ValueError(f"{path}: bad magic")
    hlen, plen = struct.unpack("<QQ", raw[8:24])
    header = json.loads(raw[24:24 + hlen].decode("utf-8"))
    payload = raw[24 + hlen:]
    if len(payload) != plen or header["payload_bytes"] != plen:
        raise ValueError(f"{path}: payload length mismatch")
    if hashlib.sha256(payload).hexdigest() != header["payload_sha256"]:
        raise ValueError(f"{path}: payload hash mismatch")
    arrays = {}
    offset = 0
    for item in header["layout"]:
        dtype = np.dtype(item["dtype"]).newbyteorder("<")
        shape = tuple(item["shape"])
        nbytes = int(np.prod(shape)) * dtype.itemsize
        arrays[item["name"]] = np.frombuffer(payload[offset:offset + nbytes], dtype=dtype) \
            .reshape(shape).astype(np.dtype(item["dtype"]), copy=True)
        offset += nbytes
    if offset != plen:
        raise ValueError(f"{path}: trailing payload")
    return header, arrays


def empty_arrays(arm: str) -> Dict[str, np.ndarray]:
    out = {}
    for name, dtype, shape in layout_for(arm):
        out[name] = np.zeros(shape, dtype=np.dtype(dtype))
    return out


def dry_run_bytes(arm: str) -> Dict[str, Any]:
    arrays = empty_arrays(arm)
    blob = build_state_blob(arm, 350, arms_params()[arm], arrays, layout_for(arm))
    return {"arm": arm, "bytes": len(blob)}


def choose_size_matched_count() -> Dict[str, Any]:
    """Choose the FP32 count whose serialized state is closest to Arm C's state."""
    c_bytes = dry_run_bytes("C")["bytes"]
    candidates = []
    for n in range(256, 1025):
        layout = e_layout(n)
        arrays = {name: np.zeros(shape, dtype=np.dtype(dtype)) for name, dtype, shape in layout}
        blob = build_state_blob("E", 350, arms_params()["E"], arrays, layout)
        candidates.append((abs(len(blob) - c_bytes), n, len(blob)))
    candidates.sort()
    _, n, e_bytes = candidates[0]
    return {
        "c_bytes": c_bytes,
        "chosen_count": n,
        "e_bytes": e_bytes,
        "ratio": e_bytes / c_bytes,
        "method": "minimize |serialized_E(n) - serialized_C(2500)| over integer n",
    }


def quota_vector(n_total: int) -> np.ndarray:
    base, rem = divmod(int(n_total), BINS)
    q = np.full((BINS,), base, dtype=np.int32)
    q[:rem] += 1
    return q


# ---------------------------------------------------------------------------
# Protocol manifest
# ---------------------------------------------------------------------------


def arms_params() -> Dict[str, Any]:
    return {
        "A": {
            "name": "FULL_SPATIAL_NN_REFERENCE",
            "support": "all causal writer-prefix descriptors in the bin",
            "score": "min unsquared Euclidean distance in the query bin",
            "role": "empirical reference, not a Pixel-AP upper bound",
        },
        "B": {
            "name": "SELECTED_FP32_SUPPORT",
            "selection": "deterministic farthest-first (FPS)",
            "count": SELECTION_COUNT,
            "quotas": quota_vector(SELECTION_COUNT).tolist(),
            "fps_init": FPS_INIT,
            "fps_tie": FPS_TIE,
            "fps_exclude": FPS_EXCLUDE,
            "extra_quota_rule": EXTRA_QUOTA_RULE,
            "storage": "float32 descriptors + int32 per-bin counts/offsets",
        },
        "C": {
            "name": "SELECTED_UINT8_SUPPORT",
            "selection": "exact same identities and order as Arm B",
            "count": SELECTION_COUNT,
            "quantizer": QUANT_DESCRIPTION,
            "storage": "uint8[768] + float32 xmin + float32 step + counts/offsets",
        },
        "D": {
            "name": "RANDOM_UINT8_SUPPORT",
            "selection": "deterministic hash-random priority",
            "count": SELECTION_COUNT,
            "quotas": quota_vector(SELECTION_COUNT).tolist(),
            "seed": RANDOM_SEED,
            "priority": "splitmix64((seed + global_descriptor_id * GOLDEN) mod 2^64); "
                        "smallest priority first; ties by global descriptor id",
            "quantizer": QUANT_DESCRIPTION,
            "storage": "same schema as C",
        },
        "E": {
            "name": "BYTE_MATCHED_SELECTED_FP32",
            "selection": "same deterministic FPS policy as B with fewer descriptors",
            "count": SIZE_MATCHED_COUNT,
            "quotas": quota_vector(SIZE_MATCHED_COUNT).tolist(),
            "storage": "float32 descriptors + counts/offsets",
            "note": "C vs E is a storage-efficiency trade-off, not a pure quantization ablation",
        },
    }


def protocol_manifest() -> Dict[str, Any]:
    return {
        "schema": "csg01_protocol_v1",
        "run_id": RUN_ID,
        "feature": {
            "backbone": "vit_base_patch8_224 (timm), ImageNet-21k augreg",
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_bytes": CHECKPOINT_BYTES,
            "capture": "block 8 output (zero-based) = layer 9 (one-based), before final "
                       "backbone normalization; CLS removed",
            "grid": [GRID, GRID],
            "dim": DIM,
            "dtype": "float32",
            "post_normalization": "none",
            "projection": "none",
            "preprocessing": "PIL direct square bilinear 224x224, uint8/255, "
                             "mean (0.485,0.456,0.406), std (0.229,0.224,0.225)",
        },
        "data": {
            "stream_source": str(P0C_PROTOCOL.relative_to(ROOT)),
            "stream_source_sha256": P0C_PROTOCOL_SHA256,
            "writer_events": 350,
            "writer_blocks": [["bottle", 100], ["carpet", 100], ["hazelnut", 100],
                              ["bottle", 50]],
            "witnesses": 59,
            "witness_blocks": [["bottle", 19], ["carpet", 20], ["hazelnut", 20]],
            "boundaries": list(BOUNDARIES),
            "eligible_witness_prefix": dict(ELIGIBLE_PREFIX),
            "role": "MVTec AD v1 train/good normal identities only",
        },
        "spatial": {
            "bin": "b(p)=(row//4)*7+(col//4), p=28*row+col",
            "bins": BINS,
            "tokens_per_bin": TOKENS_PER_BIN,
            "eligibility": "all primary arms use only descriptors from the query patch's bin",
            "global_diagnostic": "report-only; never a primary arm",
        },
        "arms": arms_params(),
        "byte_budget": choose_size_matched_count(),
        "scoring": {
            "patch": "min unsquared Euclidean distance inside the query bin",
            "image": "max over 784 patch scores",
            "orientation": SCORE_ORIENTATION,
            "precision": "CPU float64 canonical; CUDA float32 candidates + float64 winner recompute",
            "ties": "deterministic first-index minimum; stable top-k order",
        },
        "evaluation": {
            "boundaries": list(BOUNDARIES),
            "witness_eligibility": "category must have appeared by the boundary",
            "categories": ["bottle", "carpet", "hazelnut"],
            "bootstrap": {
                "t350_replicates": BOOTSTRAP_REPLICATES_T350,
                "other_boundaries_replicates": BOOTSTRAP_REPLICATES_OTHER,
                "seed": BOOTSTRAP_SEED,
                "resampling": "whole images, category-stratified, paired",
                "ci": list(BOOTSTRAP_CI),
            },
            "top_fractions": list(TOP_FRACTIONS),
            "comparisons": [["B", "A"], ["C", "A"], ["C", "B"], ["D", "A"],
                            ["E", "A"], ["C", "D"], ["C", "E"]],
        },
        "gates": GATES,
        "stop_criteria": STOP_CRITERIA,
    }


# ---------------------------------------------------------------------------
# Historical hashes
# ---------------------------------------------------------------------------


def historical_paths() -> List[Path]:
    paths = [
        AGENTS_MD,
        ROOT / "AGENTS.local.md",
        ROOT / "agents" / "tasks" / "CSG00_current_task.md",
        TASK_RECORD,
        P0C_PROTOCOL,
        ROOT / "results" / "hope_cad" / "task_p" / "p1" / "protocol_resolved.json",
        CHECKPOINT,
        SOURCE_MANIFEST,
        CACHE_DIR / "bottle_train.pt",
        CACHE_DIR / "carpet_train.pt",
        CACHE_DIR / "hazelnut_train.pt",
        CSG00_REPORT,
        CSG00_LOCK,
        CSG00_SEAL,
        ROOT / "agents" / "reports" / "task_p_p1_anomaly_signal_validation.md",
        ROOT / "agents" / "reports" / "task_npim_01_real_anomaly_dev.md",
        ROOT / "agents" / "reports" / "task_npim_02_forensic.md",
        ROOT / "agents" / "reports" / "task_npim_03_uncertainty.md",
        ROOT / "agents" / "reports" / "task_cs_lrnsm_01_compact_support.md",
    ]
    return [p for p in paths if p.is_file()]


def historical_hashes() -> Dict[str, str]:
    return {str(p.relative_to(ROOT)): C0.sha256_file(p) for p in historical_paths()}


def source_hashes() -> Dict[str, str]:
    return {str(p.relative_to(ROOT)): C0.sha256_file(p)
            for p in sorted((ROOT / "exps" / "csg01").glob("*.py"))}
