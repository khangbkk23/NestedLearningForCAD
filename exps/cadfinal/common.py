# exps/cadfinal/common.py
"""CAD-RESET-FINAL-01 shared definitions: paths, cohorts, streams, create-exclusive I/O.

Opens no mask and no test image. VisA-DEV / VisA-FINAL is a fixed, outcome-free split of the official
VisA 1cls test set: per class and image label, the 10 images with the smallest SHA-256 of their relative
path form DEV; the rest form FINAL.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from exps.calm02 import data as CD
from exps.task_p.p0_normal_support import require

ROOT = CD.ROOT
OUT = ROOT / "results/hope_cad/cad_final_v1"
CACHE = Path.home() / "hope_cad_cache/cad_final_v1"
TASK = "agents/tasks/task_cad_reset_final_01.md"
SEEDS = (2711, 2712, 2713)
DEV_PER_LABEL = 10
BUDGETS = {"B1": 2_500 * 768 * 4, "B2": 10_000 * 768 * 4}       # CADIC 2,500 / 10,000 FP32 patches
MEMORY_ARMS = {"CALM_L1": "L1_FIXED_U8", "CADIC_COMPAT": "A_CADIC_FP32"}
ORDERS = {"mvtec": CD.MVTEC_ORDER, "visa": CD.VISA_ORDER}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def text_sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=1, sort_keys=True, allow_nan=True)
    return sha(path)


def read(path):
    return json.loads(Path(path).read_text())


def r15_train_paths():
    """The exact 3,099 RH-01b R15 training identities (mvtec-relative paths), per category."""
    proto = read(ROOT / "results/hope_cad/rh01b/PROTOCOL.json")
    out = {c: [r["relative_path"] for r in proto["train"][c]] for c in CD.MVTEC_ORDER}
    require(sum(len(v) for v in out.values()) == 3099, "R15 identity count changed")
    return out


def benchmark_identities(bench):
    """Tasks in stream order with train rows (the matched training identities) and all test rows."""
    ids = CD.identities(bench)
    if bench == "mvtec":
        keep = r15_train_paths()
        for t in ids["tasks"]:
            allowed = set(keep[t["task"]])
            t["train"] = [r for r in t["train"] if r["relative_path"] in allowed]
            require(len(t["train"]) == len(allowed), f"R15 identity missing on disk: {t['task']}")
    return ids


def visa_split(ids=None):
    """{task: {"dev": [relative_path], "final": [relative_path]}}; outcome-free."""
    ids = ids or CD.identities("visa")
    out = {}
    for t in ids["tasks"]:
        dev = []
        for lab in (0, 1):
            rows = sorted((r for r in t["test"] if r["label"] == lab), key=lambda r: text_sha(r["relative_path"]))
            require(len(rows) > DEV_PER_LABEL, f"too few test images: {t['task']}")
            dev += [r["relative_path"] for r in rows[:DEV_PER_LABEL]]
        d = set(dev)
        out[t["task"]] = {"dev": sorted(dev), "final": sorted(r["relative_path"] for r in t["test"] if r["relative_path"] not in d)}
    return out


def cohort_rows(bench, cohort, ids=None):
    """Per task, the indices (into the task's test rows) of the evaluated cohort."""
    ids = ids or benchmark_identities(bench)
    if bench == "mvtec":
        require(cohort in ("full", "no_p1dev"), "MVTec cohorts: full | no_p1dev")
        drop = set(CD.p1_development_paths()) if cohort == "no_p1dev" else set()
        return {t["task"]: [i for i, r in enumerate(t["test"]) if r["relative_path"] not in drop] for t in ids["tasks"]}
    require(cohort in ("dev", "final"), "VisA cohorts: dev | final")
    sp = visa_split()
    return {t["task"]: [i for i, r in enumerate(t["test"]) if r["relative_path"] in set(sp[t["task"]][cohort])]
            for t in ids["tasks"]}
