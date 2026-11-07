"""Create-exclusive protocol lock and tamper refusal."""
from __future__ import annotations

from pathlib import Path

from exps.nl_sgm_01a import common as C
from exps.task_p.p0_normal_support import require

SOURCES = [
    "AGENTS.md", "agents/tasks/task_nl_sgm_01a.md", "agents/plans/nl_sgm_01a_plan.md",
    "exps/nl_sgm_01a/common.py", "exps/nl_sgm_01a/exact.py", "exps/nl_sgm_01a/codecs.py",
    "exps/nl_sgm_01a/rules.py", "exps/nl_sgm_01a/metrics.py", "exps/nl_sgm_01a/stats.py",
    "exps/nl_sgm_01a/lock.py", "exps/nl_sgm_01a/bench.py",
    "exps/calm01/closest_pair.py", "exps/calm01/storage.py", "exps/calm01/arms.py", "exps/calm01/calm.py",
    "exps/csg01/select.py", "exps/csg01/quantize.py", "exps/ad02/hope_cad_ad02_partition.py",
    "exps/task_p/p0_normal_support.py", "exps/task_p/p1_readout.py", "exps/cadfinal/common.py",
    "exps/calm02/data.py", "results/hope_cad/rh01b/PROTOCOL.json",
    "results/hope_cad/calm02/features/mvtec/MANIFEST.json",
]


def source_hashes(root=C.ROOT, sources=SOURCES):
    return {s: C.sha(Path(root) / s) for s in sources}


def write_lock(path, payload, root=C.ROOT, sources=SOURCES):
    body = dict(payload, sources=source_hashes(root, sources), utc=C.now())
    return C.write_json(path, body)                          # mode "x": refuses to overwrite


def verify_lock(path, root=C.ROOT):
    lock = C.read_json(path)
    for s, h in lock["sources"].items():
        require(C.sha(Path(root) / s) == h, f"LOCK mismatch: {s}")
    return lock
