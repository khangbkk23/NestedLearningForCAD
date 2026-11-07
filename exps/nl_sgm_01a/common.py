"""NL-SGM-01A shared definitions: cohorts, custody guards, feature access, create-exclusive I/O.

Normal-only. Reads the CALM-02 MVTec ``{task}_train.pt`` cache (train/good) and R15 identity lists only.
Never opens ``*_test.pt``, ``/test/`` paths, ``ground_truth`` or VisA.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch

from exps.cadfinal import common as CF
from exps.calm02 import data as CD
from exps.task_p.p0_normal_support import require

ROOT = CD.ROOT
OUT = ROOT / "results/hope_cad/nl_sgm_01a"
FEATURES = ROOT / "results/hope_cad/calm02/features/mvtec"
MANIFEST_SHA = "fcfaf66c"                       # prefix recorded by NL-SGM-00 / FREEZE_mvtec.json
R15_DIGEST = "2fa385fe"                         # prefix of the sorted R15 path digest (PROTOCOL_LOCK_v1)
ORDER = CD.MVTEC_ORDER
PATCHES, DIM, SIDE = 784, 768, 28
PROBES_PER_CATEGORY, PER_ROLE = 12, 6
SEED = 20261106
B1, B2 = 7_680_000, 30_720_000
OLD = tuple(range(0, 7))                        # a(c) <= 6: bottle .. leather (Amendment 01 item 4)
LATE = tuple(range(7, 15))                      # a(c) >= 7: metal_nut .. zipper


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


sha = CF.sha
text_sha = CF.text_sha


def write_json(path, value):
    """Create-exclusive JSON write (refuses to overwrite)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=1, sort_keys=True, allow_nan=True, default=_default)
    return sha(path)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def read_json(path):
    return json.loads(Path(path).read_text())


def guard_path(p):
    """Normal-only custody: refuse any path outside MVTec train/good."""
    s = str(p)
    require("/test/" not in s and "ground_truth" not in s and "visa" not in s.lower()
            and not s.endswith("_test.pt"), f"custody violation: {s}")
    return p


# ------------------------------------------------------------------ cohorts

def probe_key(path):
    return text_sha("NL-SGM-01A/probe/v1|" + path)


def role_key(path):
    return text_sha("NL-SGM-01A/role/v1|" + path)


def cohorts(per_role=PER_ROLE):
    """{category: {"writers": [...sorted], "sel": [...], "eval": [...]}} from the R15 identities."""
    r15 = CF.r15_train_paths()
    out = {}
    for c in ORDER:
        paths = sorted(r15[c])
        for p in paths:
            guard_path(p)
            require(p.startswith(f"{c}/train/good/"), f"non-train identity {p}")
        probes = sorted(paths, key=probe_key)[:2 * per_role]
        ranked = sorted(probes, key=role_key)
        sel, ev = sorted(ranked[:per_role]), sorted(ranked[per_role:])
        writers = [p for p in paths if p not in set(probes)]
        require(not set(writers) & set(probes) and not set(sel) & set(ev), "cohort overlap")
        out[c] = {"writers": writers, "sel": sel, "eval": ev}
    return out


def r15_digest():
    r15 = CF.r15_train_paths()
    return text_sha("\n".join(sorted(p for c in ORDER for p in r15[c])))


# ------------------------------------------------------------------ features

class Features:
    """Memory-mapped per-category train features; rows addressed by relative path."""

    def __init__(self):
        self._obj, self._index = {}, {}

    def _load(self, c):
        if c not in self._obj:
            path = guard_path(FEATURES / f"{c}_train.pt")
            obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            require(obj["features"].shape[1:] == (PATCHES, DIM) and obj["features"].dtype == torch.float32,
                    "unexpected feature layout")
            self._obj[c] = obj
            self._index[c] = {r["relative_path"]: i for i, r in enumerate(obj["rows"])}
        return self._obj[c]

    def rows(self, c):
        return self._load(c)["rows"]

    def image(self, path):
        c = path.split("/")[0]
        guard_path(path)
        obj = self._load(c)
        return obj["features"][self._index[c][path]]

    def images(self, paths):
        if not paths:
            return torch.zeros(0, PATCHES, DIM)
        c = paths[0].split("/")[0]
        obj = self._load(c)
        idx = torch.tensor([self._index[c][guard_path(p)] for p in paths])
        return obj["features"][idx]

    def drop(self, c=None):
        for k in ([c] if c else list(self._obj)):
            self._obj.pop(k, None); self._index.pop(k, None)


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def current_rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024 ** 2
    return float("nan")


def set_threads(n=6):
    torch.set_num_threads(n)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
