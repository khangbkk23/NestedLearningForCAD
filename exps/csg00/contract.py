"""CSG-00 protocol constants, integrity helpers and state serialization.

Everything in this module is frozen by ``G0_LOCK.json``.  Changing any constant
after the lock invalidates the run; a new run id and a new lock are required.

Scientific scope: static geometric compression feasibility on frozen normal
features only.  No anomaly image, mask, label or sealed identity is opened.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

RUN_ID = "csg00_g0g2_20261021_r3"
RUN_DIR = ROOT / "results" / "hope_cad" / "csg00" / RUN_ID
G0_LOCK_PATH = RUN_DIR / "G0_LOCK.json"
G2_SEAL_PATH = RUN_DIR / "G2_SEAL.json"

CHECKPOINT = ROOT / "checkpoints" / "cadic" / "vit_base_patch8_224_augreg_in21k_state_dict.pth"
CHECKPOINT_SHA256 = "755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415"
CHECKPOINT_BYTES = 343_282_995

CACHE_DIR = ROOT / "results" / "hope_cad" / "frozen_features_v1" / "clean"
CACHE_SHA256 = {
    "bottle": "ef4f3180d2381b64b4cdafea447b9df254b7522b9338f24992f4a857114ad6b3",
    "carpet": "f3b1957d7b9d12add45c4ff9c33a5f9f5e237ff7ce6dbdb7693c81527c89b179",
    "hazelnut": "7bd97accb4a927285881d8c8b61cbc5db8106e57e2ceee1fe623f5903f62d180",
}

P0C_PROTOCOL = ROOT / "results" / "hope_cad" / "task_p" / "p0c" / "protocol_resolved.json"
P0C_PROTOCOL_SHA256 = "0090f697518587e51eda91a3bd5def44b5ceb2c572b49eb75f5b9a0dfd4fb182"
SUPPLIED_TASK = ROOT / "agents" / "tasks" / "CSG00_current_task.md"
CANONICAL_TASK = ROOT / "agents" / "tasks" / "task_csg00_compact_geometry.md"
AGENTS_MD = ROOT / "AGENTS.md"
AGENTS_MD_PIN = "b88393815586295209c2704a6e16262730f2aba4000a6904b7c5c6d2b65ed622"

SOURCE_MANIFEST = ROOT / "results" / "hope_cad" / "frozen_features_v1" / "source_manifest.json"
SOURCE_MANIFEST_DIGEST = "060579457d344992eef159b1d4f1a86c34dd9283b97b70d3c3a386006b0f4dcc"

# ---------------------------------------------------------------------------
# Frozen geometry
# ---------------------------------------------------------------------------

GRID = 28
PATCHES = GRID * GRID                     # 784
DIM = 768                                 # full-dimensional, no projection
BIN_SIDE = 4
BINS_PER_SIDE = GRID // BIN_SIDE           # 7
BINS = BINS_PER_SIDE * BINS_PER_SIDE      # 49
TOKENS_PER_BIN = BIN_SIDE * BIN_SIDE      # 16

K_MODES = 2
Q_TANGENT = 4
TANGENT_MULT = 2.0                        # R_j = 2*sqrt(lambda_j + eps)
ORTHO_MULT = 1.0                          # rho = 1*sqrt(trace_perp_cov)
COV_NORMALIZATION = "1/n (population covariance); per-sample eigenvalues"
N_MIN_MODE = 16                           # fewer assigned samples -> mode dropped

# eps is tied to the observed feature scale and FP32 resolution, causally using
# only the first 100 writer events (bottle) available at boundary t=100.
FEATURE_SCALE_S = 0.530551116359072        # mean per-dim std, float64, t=100 writer prefix
FP32_UNIT_ROUNDOFF = 2.0 ** -23            # 1.1920928955078125e-07
EPS_SCALE_FACTOR = 32.0
EPS_VAR = (EPS_SCALE_FACTOR * FP32_UNIT_ROUNDOFF * FEATURE_SCALE_S) ** 2
# locked value: 4.096138393940893e-12

# Deterministic fitting rule (offline static feasibility, not online learning)
FIT_SEED = 20261021
KMEANS_RESTARTS = 4
KMEANS_MAX_ITERS = 100
KMEANS_TOL = 1e-6

# Arm C byte-matched spatial reservoir
RESERVOIR_CAPACITY = 10                    # descriptors per spatial bin
RESERVOIR_SEED = 20261021
RESERVOIR_RULE = "Algorithm R with stateless counter-based SplitMix64 RNG"

# Causal stream
CATEGORIES = ("bottle", "carpet", "hazelnut")
BOUNDARIES = (100, 200, 300, 350)
EXPECTED_EVENT_CATEGORIES = (
    ("bottle", 100),
    ("carpet", 100),
    ("hazelnut", 100),
    ("bottle", 50),
)

# Witness eligibility: protocol probe order is bottle(19), carpet(20), hazelnut(20),
# so the eligible prefix length is a pure function of the boundary.
PROBE_BLOCKS = (("bottle", 19), ("carpet", 20), ("hazelnut", 20))
ELIGIBLE_PREFIX = {100: 19, 200: 39, 300: 59, 350: 59}

# ---------------------------------------------------------------------------
# Analysis / bootstrap
# ---------------------------------------------------------------------------

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261021
BOOTSTRAP_CI = (2.5, 97.5)
TOP_FRACTIONS = (0.01, 0.05)
SCORE_HIST_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, float("inf"))

GAP_LAMBDAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
GAP_EXTRAPOLATIONS = (-0.25, 1.25)
GAP_CROSS_LAMBDAS = (0.25, 0.5, 0.75)
GAP_ACCEPT_RATIO = 0.25
SUPPORT_SAMPLES_PER_MODE = 64
SUPPORT_SAMPLE_SEED = 20261021

# ---------------------------------------------------------------------------
# Pre-registered decision bands (locked before G2)
# ---------------------------------------------------------------------------

DECISION_BANDS = {
    "primary_pooled": {
        "boundary": 350,
        "delta_cov": "coverage_B - coverage_C at t=350, equal-category macro; "
                     "coverage_M = mean(s_M <= s_A) over eligible witness patches; "
                     "whole-image bootstrap stratified by category",
        "delta_rho": "rho_B - rho_C at t=350, equal-category macro; image-max Spearman "
                     "against arm A; whole-image bootstrap stratified by category",
        "gfa_B": "fraction of between-mode/cross-category feature-space gap probes "
                 "with A-score >= bin-median witness A-score that arm B accepts "
                 f"with s_B <= {GAP_ACCEPT_RATIO} * A-score; pooled over boundaries "
                 "and per boundary",
        "ovl_B": "fraction of t=350 eligible witness patches with s_B == 0",
    },
    "go_requires": [
        "delta_cov CI lower > 0.05",
        "delta_rho CI lower > -0.05",
        "pooled gfa_B <= 0.25",
        "max-boundary gfa_B <= 0.40",
        "ovl_B <= 0.95",
        "no degenerate constant arm-B image-score vector in any eligible category",
    ],
    "negative_if_any": [
        "pooled gfa_B > 0.25",
        "any boundary gfa_B > 0.40",
        "ovl_B > 0.95",
        "delta_cov CI upper < 0.05",
        "delta_rho CI upper < -0.10",
        "arm-B image-score vector constant in an eligible category",
    ],
    "inconclusive": "neither GO nor NEGATIVE conditions are met",
    "blocked": "any integrity, feature, data, serialization or test failure",
    "note": "These are decision bands for authorizing a separately gated real-anomaly "
            "experiment. They are not anomaly efficacy thresholds and cannot certify "
            "Pixel AP, AUROC or continual retention.",
}

STOP_CRITERIA = [
    "G0 integrity check fails (hash, stream identity, train/good separation, ownership)",
    "any G1 test fails",
    "a mode has n < 16 and every mode in a bin is dropped (bin has no ready support)",
    "non-finite state or score",
    "serialized B/C byte ratio outside [0.95, 1.05] with the preregistered capacity",
    "any anomaly, test, ground-truth, sealed or consumed-DEV path is opened",
    "any historical artifact hash changes",
    "CUDA/CPU parity max absolute score difference > 1.0e-3 on the common support",
]

# ---------------------------------------------------------------------------
# State serialization
# ---------------------------------------------------------------------------

STATE_MAGIC = b"CSG00ST1"
STATE_SCHEMA = "csg00_state_v1"

B_STATE_LAYOUT = (
    ("ready", "uint8", (BINS, K_MODES)),
    ("count", "int64", (BINS, K_MODES)),
    ("n_total", "int64", (BINS,)),
    ("mu", "float32", (BINS, K_MODES, DIM)),
    ("U", "float32", (BINS, K_MODES, DIM, Q_TANGENT)),
    ("lam", "float32", (BINS, K_MODES, Q_TANGENT)),
    ("R", "float32", (BINS, K_MODES, Q_TANGENT)),
    ("rho", "float32", (BINS, K_MODES)),
)

C_STATE_LAYOUT = (
    ("res", "float32", (BINS, RESERVOIR_CAPACITY, DIM)),
    ("count", "int64", (BINS,)),
    ("seen", "int64", (BINS,)),
)


def b_payload_bytes() -> int:
    return sum(int(np.prod(shape)) * np.dtype(dt).itemsize for _, dt, shape in B_STATE_LAYOUT)


def c_payload_bytes() -> int:
    return sum(int(np.prod(shape)) * np.dtype(dt).itemsize for _, dt, shape in C_STATE_LAYOUT)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_tensor(tensor) -> str:
    import torch

    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(t.numpy().tobytes(order="C"))
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_digest(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode())


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def splitmix64(x: int) -> int:
    """Deterministic 64-bit SplitMix64 finalizer."""
    mask = (1 << 64) - 1
    x = (x + 0x9E3779B97F4A7C15) & mask
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & mask


def mix_keys(seed: int, *keys: int) -> int:
    x = int(seed) & ((1 << 64) - 1)
    for key in keys:
        x = splitmix64(x ^ (int(key) & ((1 << 64) - 1)))
    return x


def counter_uniform(seed: int, *keys: int) -> float:
    """Uniform double in [0,1) from a stateless counter-based RNG."""
    return (mix_keys(seed, *keys) >> 11) / float(1 << 53)


def rng_from(seed: int, *keys: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(mix_keys(seed, *keys)))


def patch_bin_indices() -> np.ndarray:
    """Return bin index b(p) for every canonical patch p = 28*row + col."""
    p = np.arange(PATCHES, dtype=np.int64)
    row = p // GRID
    col = p % GRID
    return (row // BIN_SIDE) * BINS_PER_SIDE + (col // BIN_SIDE)


def bin_positions() -> Tuple[np.ndarray, ...]:
    bins = patch_bin_indices()
    return tuple(np.nonzero(bins == b)[0].astype(np.int64) for b in range(BINS))


def assert_normal_only(relpath: str, whitelist: Iterable[str]) -> None:
    """Default-deny guard for every image/feature identity read by this package."""
    allowed = set(whitelist)
    if relpath not in allowed:
        raise PermissionError(f"CSG-00 default-deny: identity not whitelisted: {relpath!r}")
    parts = relpath.split("/")
    if len(parts) != 4 or parts[1:3] != ["train", "good"]:
        raise PermissionError(f"CSG-00 normal-only guard: not train/good: {relpath!r}")
    if not parts[3].endswith(".png"):
        raise PermissionError(f"CSG-00 normal-only guard: unexpected suffix: {relpath!r}")
    if any(tok in relpath.lower() for tok in ("test", "ground_truth", "mask", "anomaly", "sealed")):
        raise PermissionError(f"CSG-00 normal-only guard: forbidden token: {relpath!r}")


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


def write_state(path: Path | str, arm: str, boundary: int, params: Mapping[str, Any],
                arrays: Mapping[str, np.ndarray], layout: Sequence[Tuple[str, str, Tuple[int, ...]]],
                exclusive: bool = True) -> Dict[str, Any]:
    """Write a deterministic, self-describing state file.

    File = MAGIC(8) | header_len(uint64) | payload_len(uint64) | header_json | payload.
    The header has no timestamps, so identical content produces identical bytes.
    """
    path = Path(path)
    payload_parts = []
    for name, dtype, shape in layout:
        arr = np.ascontiguousarray(np.asarray(arrays[name], dtype=np.dtype(dtype)))
        if tuple(arr.shape) != tuple(shape):
            raise ValueError(f"state array {name}: shape {arr.shape} != {shape}")
        payload_parts.append(arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes(order="C"))
    payload = b"".join(payload_parts)
    header = {
        "schema": STATE_SCHEMA,
        "arm": arm,
        "boundary": int(boundary),
        "params": dict(params),
        "layout": [{"name": n, "dtype": d, "shape": list(s)} for n, d, s in layout],
        "payload_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
    }
    header_json = canonical_json(header).encode("utf-8")
    blob = STATE_MAGIC + struct.pack("<QQ", len(header_json), len(payload)) + header_json + payload
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if exclusive else "wb"
    with open(path, mode) as fh:
        fh.write(blob)
    return {
        "path": str(path),
        "bytes": len(blob),
        "header_bytes": len(header_json),
        "payload_bytes": len(payload),
        "sha256": sha256_bytes(blob),
        "payload_sha256": header["payload_sha256"],
    }


def read_state(path: Path | str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    path = Path(path)
    raw = path.read_bytes()
    if raw[:8] != STATE_MAGIC:
        raise ValueError(f"{path}: bad magic")
    header_len, payload_len = struct.unpack("<QQ", raw[8:24])
    header = json.loads(raw[24:24 + header_len].decode("utf-8"))
    payload = raw[24 + header_len:]
    if len(payload) != payload_len or header["payload_bytes"] != payload_len:
        raise ValueError(f"{path}: payload length mismatch")
    if sha256_bytes(payload) != header["payload_sha256"]:
        raise ValueError(f"{path}: payload hash mismatch")
    arrays: Dict[str, np.ndarray] = {}
    offset = 0
    for item in header["layout"]:
        dtype = np.dtype(item["dtype"]).newbyteorder("<")
        shape = tuple(item["shape"])
        nbytes = int(np.prod(shape)) * dtype.itemsize
        block = payload[offset:offset + nbytes]
        arrays[item["name"]] = np.frombuffer(block, dtype=dtype).reshape(shape).astype(
            np.dtype(item["dtype"]), copy=True
        )
        offset += nbytes
    if offset != payload_len:
        raise ValueError(f"{path}: trailing payload bytes")
    return header, arrays


# ---------------------------------------------------------------------------
# Frozen stream
# ---------------------------------------------------------------------------


def load_stream() -> Dict[str, Any]:
    """Load and verify the frozen 350-event normal writer stream and 59 witnesses."""
    data = json.loads(P0C_PROTOCOL.read_text())
    updates = data["updates"]
    probes = data["probes"]

    if len(updates) != 350:
        raise RuntimeError(f"writer stream must contain 350 events, got {len(updates)}")
    if len(probes) != 59:
        raise RuntimeError(f"expected 59 normal witnesses, got {len(probes)}")

    # Exact category blocks.
    blocks = []
    start = 0
    cats = [row["category"] for row in updates]
    for i in range(1, len(cats) + 1):
        if i == len(cats) or cats[i] != cats[start]:
            blocks.append((cats[start], i - start))
            start = i
    if tuple(blocks) != EXPECTED_EVENT_CATEGORIES:
        raise RuntimeError(f"unexpected writer category blocks: {blocks}")

    # Distinct identities by path and content.
    paths = [row["relative_path"] for row in updates]
    hashes = [row["raw_png_sha256"] for row in updates]
    if len(set(paths)) != 350 or len(set(hashes)) != 350:
        raise RuntimeError("writer events are not 350 distinct identities/contents")

    # Witness identities and order.
    probe_paths = [row["relative_path"] for row in probes]
    probe_cats = [row["category"] for row in probes]
    expected_probe_cats = ["bottle"] * 19 + ["carpet"] * 20 + ["hazelnut"] * 20
    if probe_cats != expected_probe_cats:
        raise RuntimeError("witness probe order is not bottle(19), carpet(20), hazelnut(20)")
    if len(set(probe_paths)) != 59:
        raise RuntimeError("witness identities are not distinct")

    if set(paths) & set(probe_paths):
        raise RuntimeError("writer/witness path overlap")
    if set(hashes) & set(row["raw_png_sha256"] for row in probes):
        raise RuntimeError("writer/witness content overlap")

    for row in updates + probes:
        assert_normal_only(row["relative_path"], set(paths) | set(probe_paths))

    # Category schedule per boundary.
    schedule = {}
    seen = []
    for t in BOUNDARIES:
        seen = sorted({row["category"] for row in updates[:t]})
        expected = sorted({row["category"] for row in updates[:t]})
        eligible = 0
        for cat, count in PROBE_BLOCKS:
            if cat in seen:
                eligible += count
        if eligible != ELIGIBLE_PREFIX[t]:
            raise RuntimeError(f"boundary {t}: eligible witness prefix mismatch")
        schedule[t] = {
            "categories_seen": seen,
            "eligible_witness_prefix": eligible,
            "eligible_witnesses_by_category": {
                cat: (count if cat in seen else 0) for cat, count in PROBE_BLOCKS
            },
        }

    return {
        "protocol": data,
        "updates": updates,
        "probes": probes,
        "writer_paths": paths,
        "probe_paths": probe_paths,
        "schedule": schedule,
        "stream_digest": canonical_digest(
            [{"event": e + 1, "category": r["category"], "relative_path": r["relative_path"],
              "raw_png_sha256": r["raw_png_sha256"]} for e, r in enumerate(updates)]
        ),
        "witness_digest": canonical_digest(
            [{"relative_path": r["relative_path"], "raw_png_sha256": r["raw_png_sha256"]}
             for r in probes]
        ),
    }


# ---------------------------------------------------------------------------
# Fitting / byte-budget derivation (dry run, no feature values involved)
# ---------------------------------------------------------------------------


def reservoir_capacity_for_byte_match() -> Dict[str, Any]:
    """Choose the integer per-bin reservoir capacity whose payload is closest to B.

    This is an empty-schema dry run: it uses only declared shapes and dtypes, not
    any feature value or outcome.
    """
    target = b_payload_bytes()
    candidates = []
    for n in range(1, 65):
        payload = BINS * (n * DIM * 4 + 8 + 8)
        candidates.append((abs(payload - target), n, payload))
    candidates.sort()
    _, capacity, payload = candidates[0]
    return {
        "b_payload_bytes": target,
        "chosen_capacity": capacity,
        "c_payload_bytes_at_capacity": payload,
        "payload_ratio": payload / target,
        "method": "minimize |payload_C(n) - payload_B| over integer n",
    }


def protocol_manifest() -> Dict[str, Any]:
    """Complete locked protocol description (no outcome data)."""
    return {
        "schema": "csg00_protocol_v1",
        "run_id": RUN_ID,
        "feature": {
            "backbone": "vit_base_patch8_224 (timm), ImageNet-21k augreg",
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_bytes": CHECKPOINT_BYTES,
            "capture": "block 8 output (zero-based) = layer 9 (one-based), before final "
                       "backbone normalization; CLS removed",
            "patch_grid": [GRID, GRID],
            "dim": DIM,
            "dtype": "float32",
            "post_extraction_normalization": "none",
            "preprocessing": "PIL direct square bilinear resize to 224x224, uint8/255, "
                             "mean (0.485,0.456,0.406), std (0.229,0.224,0.225)",
            "spatial_index": "p = 28*row + col, row/col zero-based",
            "projection": "none (d=768 end-to-end)",
        },
        "data": {
            "stream_source": str(P0C_PROTOCOL.relative_to(ROOT)),
            "stream_source_sha256": P0C_PROTOCOL_SHA256,
            "writer_events": 350,
            "writer_blocks": [list(x) for x in EXPECTED_EVENT_CATEGORIES],
            "witnesses": 59,
            "witness_blocks": [list(x) for x in PROBE_BLOCKS],
            "boundaries": list(BOUNDARIES),
            "eligible_witness_prefix": ELIGIBLE_PREFIX,
            "role": "MVTec AD v1 train/good only; no anomaly/test/mask/sealed/consumed-DEV identity",
        },
        "spatial_eligibility": {
            "bin_index": "b(p) = (row//4)*7 + (col//4)",
            "bins": BINS,
            "tokens_per_bin": TOKENS_PER_BIN,
            "rule": "all three primary arms may use only descriptors from the same bin b(p) "
                    "as the query patch",
            "diagnostic_only": "global unrestricted NN is report-only and never substituted "
                               "into primary results",
        },
        "arms": {
            "A": {
                "name": "FULL_SPATIAL_NN_REFERENCE",
                "rule": "S_A(z,b) = min over all writer-prefix descriptors in bin b of ||z-x||_2",
                "role": "empirical normal geometry reference; NOT a Pixel-AP upper bound",
            },
            "B": {
                "name": "COMPACT_BOUNDED_MODE_768",
                "k_modes": K_MODES,
                "q_tangent": Q_TANGENT,
                "support": "S_bk = {mu + U a + v : |a_j| <= R_j, U^T v = 0, ||v|| <= rho}",
                "distance": "sqrt(sum_j max(|a_j|-R_j,0)^2 + max(||e||-rho,0)^2), "
                            "min over ready modes",
                "R_formula": "R_j = 2*sqrt(lambda_j + eps)",
                "rho_formula": "rho = 1.0*sqrt(max(trace(Cov_mode) - sum_j lambda_j, 0))",
                "lambda": "per-sample covariance eigenvalue (1/n convention)",
                "eps_var": EPS_VAR,
                "eps_derivation": "(32 * 2^-23 * 0.530551116359072)^2, feature scale from the "
                                  "first 100 writer events only",
                "n_min_mode": N_MIN_MODE,
                "fitting": "deterministic two-mode k-means++ (4 restarts, best inertia), then "
                           "per-cluster covariance eigendecomposition (top q); offline static "
                           "refit from the causal writer prefix at each boundary",
                "label": "OFFLINE STATIC FEASIBILITY, not online continual learning",
            },
            "C": {
                "name": "BYTE_MATCHED_SPATIAL_RESERVOIR",
                "capacity_per_bin": RESERVOIR_CAPACITY,
                "rule": "uniform Algorithm R reservoir per bin over arriving writer descriptors",
                "rng": RESERVOIR_RULE,
                "seed": RESERVOIR_SEED,
            },
        },
        "spatial_state_budget": reservoir_capacity_for_byte_match(),
        "scoring": {
            "patch_score": "per arm, min distance inside the query patch's own bin",
            "image_score_primary": "max over all 784 patch scores",
            "image_score_secondary_descriptive": "mean of the 7 largest patch scores",
            "orientation": "high = more anomalous",
            "tie_treatment": "deterministic first-index minimum; stable (score,image,patch) sort",
        },
        "statistics": {
            "image_level": ["paired Spearman", "Kendall tau-b", "B-vs-A", "C-vs-A"],
            "patch_sets": [f"top {f:.0%} precision and IoU vs A" for f in TOP_FRACTIONS],
            "bootstrap": f"{BOOTSTRAP_REPLICATES} whole-image replicates, category-stratified, "
                         f"paired, seed {BOOTSTRAP_SEED}, percentile CI {BOOTSTRAP_CI}",
            "caution": "patch-level iid bootstrap is never used as if patches were independent",
        },
        "decision_bands": DECISION_BANDS,
        "stop_criteria": STOP_CRITERIA,
        "authorization": {
            "authorized": ["G0", "G1", "G2 normal-only feasibility"],
            "not_authorized": ["G3 real-anomaly execution", "G4 continual learning",
                               "ViT training", "historical artifact edits", "git commit/push"],
        },
    }
