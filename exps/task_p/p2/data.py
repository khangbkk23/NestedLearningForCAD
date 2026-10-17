"""Identity-gated access to immutable normal training and P1 development caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from exps.task_p.p1_data import CP_SHA, sha, write_json

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/hope_cad/task_p/p2"
P1 = ROOT / "results/hope_cad/task_p/p1"
FEATURES = ROOT / "results/hope_cad/frozen_features_v1"
P1_SHA = "11c1cc03b20cb1f79a6e8b7e472bb6f44758b43046b61043368ea11b4ce1e08f"
ARMS = ("A", "B", "C", "D", "D_noinner")
SEEDS = (2711, 2712, 2713)
ORDERS = {"canonical": ("bottle", "carpet", "hazelnut"),
          "alternate": ("bottle", "hazelnut", "carpet")}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest_tensor(features):
    return hashlib.sha256(str((tuple(features.shape), features.dtype)).encode()
                          + features.contiguous().numpy().tobytes()).hexdigest()


def tree_digest(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for element in item:
                visit(element)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(element) for element in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(element) for element in value)
    return 0


def p1_protocol():
    require(sha(P1 / "protocol_resolved.json") == P1_SHA, "P1 protocol changed")
    return json.loads((P1 / "protocol_resolved.json").read_text())


def stream(protocol, order):
    rows = protocol["updates"]
    phases = {category: [row for row in rows[:300] if row["category"] == category]
              for category in ORDERS[order]}
    ordered = [row for category in ORDERS[order] for row in phases[category]] + rows[300:]
    return [dict(row, event=index + 1) for index, row in enumerate(ordered)]


class NormalReader:
    def __init__(self, rows):
        self.rows, self.completed = rows, 0
        self.cache, self.category = None, None
        self.access = []

    def get(self, row):
        require(self.completed < len(self.rows) and row == self.rows[self.completed],
                "future, historical reread or unauthorized normal input")
        require(row["role"] == "normal_train" and "/train/good/" in row["relative_path"], "non-normal update")
        category = row["category"]
        if self.category != category:
            self.cache = torch.load(FEATURES / "clean" / f"{category}_train.pt", weights_only=True, mmap=True)
            self.category = category
            require(self.cache["view"] == "clean" and self.cache["metadata"]["checkpoint_sha256"] == CP_SHA,
                    "cache representation drift")
        index = row["cache_index"]
        require(self.cache["relative_paths"][index] == row["relative_path"]
                and self.cache["per_image_source_sha256"][index] == row["sha256"], "normal identity mismatch")
        features = self.cache["patches"][index].clone()
        require(features.shape == (784, 768) and features.dtype == torch.float32
                and bool(torch.isfinite(features).all()), "invalid normal features")
        digest = digest_tensor(features)
        require(digest == self.cache["per_image_feature_sha256"][index], "normal feature row digest changed")
        self.completed += 1
        self.access.append({"event": self.completed, "source": row["relative_path"], "feature_sha256": digest})
        return features


class Replay:
    def __init__(self, seed, capacity=24):
        self.capacity, self.seen = capacity, 0
        self.generator = torch.Generator().manual_seed(seed + 9000)
        self.images, self.identities = [], []

    def sample(self):
        if not self.images:
            return None, None
        index = int(torch.randint(len(self.images), (1,), generator=self.generator))
        return self.images[index], self.identities[index]

    def insert(self, features, identity):
        self.seen += 1
        if len(self.images) < self.capacity:
            index = len(self.images)
            self.images.append(features.clone())
            self.identities.append(identity)
            return
        index = int(torch.randint(self.seen, (1,), generator=self.generator))
        if index < self.capacity:
            self.images[index] = features.clone()
            self.identities[index] = identity

    def state(self):
        return {"capacity": self.capacity, "seen": self.seen, "images": self.images,
                "identities": self.identities, "rng": self.generator.get_state()}


def verify_inheritance():
    protocol = p1_protocol()
    p0c = ROOT / "results/hope_cad/task_p/p0c"
    require(sha(p0c / "protocol_resolved.json") == (p0c / "protocol.sha256").read_text().strip(),
            "P0c protocol changed")
    checks = {}
    for name, expected in protocol["source_sha256"].items():
        checks[name] = {"expected": expected, "actual": sha(ROOT / name)}
    prior = json.loads((P1 / "input_verification.json").read_text())
    for name, check in prior["checks"].items():
        target = ROOT / name
        if name.startswith("checkpoints/") and not target.exists():
            checks[name] = {"expected": check["actual"], "actual": None,
                            "disclosure": "checkpoint missing; immutable complete feature caches used, no extraction"}
        else:
            checks[name] = {"expected": check["actual"], "actual": sha(target)}
    for row in protocol["evaluations"]:
        key = hashlib.sha256(row["relative_path"].encode()).hexdigest()
        marker = json.loads((P1 / "evaluation_features" / (key + ".json")).read_text())
        require(marker["source"] == row and marker["protocol_sha256"] == P1_SHA
                and marker["checkpoint_sha256"] == CP_SHA, "P1 evaluation marker drift")
        path = P1 / "evaluation_features" / (key + ".pt")
        checks[str(path.relative_to(ROOT))] = {"expected": marker["file_sha256"], "actual": sha(path)}
    failures = {name: check for name, check in checks.items()
                if check["actual"] != check["expected"] and "disclosure" not in check}
    require(not failures, f"inherited integrity mismatch: {failures}")
    preserved = {}
    directories = (FEATURES, ROOT / "results/hope_cad/task_p/p0", p0c, P1,
                   ROOT / "results/hope_cad/task_p/p0b", ROOT / "results/hope_cad/r1_conditional_pairing")
    for directory in directories:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                preserved[str(path.relative_to(ROOT))] = sha(path)
    source_paths = [ROOT / "AGENTS.md", ROOT / "dataset/validate_mvtec.py"]
    source_paths += list((ROOT / "exps/task_p").glob("*.py"))
    source_paths += list((ROOT / "agents/tasks").glob("*.md"))
    source_paths += list((ROOT / "agents/reports").glob("*.md"))
    for path in source_paths:
        if path.is_file() and path.name not in ("current_task_p2.md", "task_p_p2a_neural_anomaly_foundation.md",
                                               "task_p_p2b_nested_learning_mechanism.md"):
            preserved[str(path.relative_to(ROOT))] = sha(path)
    return {"passed": True, "checks": checks, "preserved": preserved,
            "sealed_payload_reads": 0, "p1_protocol_sha256": P1_SHA}


def evaluation_features(row, protocol):
    require(row in protocol["evaluations"] and row["relative_path"] not in protocol["sealed_identity_paths"],
            "unauthorized evaluation source")
    key = hashlib.sha256(row["relative_path"].encode()).hexdigest()
    marker = json.loads((P1 / "evaluation_features" / (key + ".json")).read_text())
    path = P1 / "evaluation_features" / (key + ".pt")
    require(marker["source"] == row and marker["protocol_sha256"] == P1_SHA
            and sha(path) == marker["file_sha256"], "evaluation cache drift")
    features = torch.load(path, weights_only=True, map_location="cpu")
    require(features.shape == (784, 768) and features.dtype == torch.float32
            and bool(torch.isfinite(features).all()), "invalid evaluation features")
    return features
