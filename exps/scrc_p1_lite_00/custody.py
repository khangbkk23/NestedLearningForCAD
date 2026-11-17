"""SCRC-P1-LITE-00 data custody (prereg §13): phase-gated ``open`` audit hook, access log, guarded feature files.

Phases
  PRE  preparation / tests: train-side data only; no test feature, manifest, mask, label, VisA, historical eval output.
  S    Phase S: additionally the 15 MVTec ``*_test.pt`` feature files (``features`` key only, via ``guarded_features``);
       MVTec train features are no longer needed and are refused.
  L    Phase L: additionally the AM03 ``DATASET_MANIFEST.json`` and ``data/mvtec/<c>/ground_truth/**`` masks, and the
       ``rows`` key of the test feature files (order check).
Always refused: any path containing ``visa`` (case-insensitive), ``FINAL``, MVTec RGB test images (``/test/``),
historical evaluation outputs. Scope: absolute paths under the repository or ``/mnt``; code files are exempt (as P0).
The hook cannot be removed once installed, so ``install`` is idempotent and the phase is a module variable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from exps.scrc_p1_lite_00 import protocol as P

ROOT = str(P.ROOT)
TEST_FEATURES = frozenset(str(P.ROOT / f"results/hope_cad/calm02/features/mvtec/{c}_test.pt") for c in P.ORDER)
MANIFEST = str(P.ROOT / "results/hope_cad/nl_sgm_01a_am03/DATASET_MANIFEST.json")
GT_PREFIX = str(P.ROOT / "data/mvtec") + "/"
ALWAYS = ("/test/", "FINAL", "/eval/", "/cells/", "METRICS.json", "_TABLE.csv", "MATCHED_COMPARISONS",
          "STAGE_METRIC_MATRIX", "split_csv", "image_anno", "/Masks/")
LABEL = ("_test.pt", "DATASET_MANIFEST", "ground_truth", "/masks", "_mask", "Anomaly")
PHASES = ("PRE", "S", "L")

_state = {"phase": "PRE", "installed": False}
ACCESS = []               # (phase, path) in first-open order
_seen = set()


def decide(path, phase):
    """(allowed, reason) for an absolute path under the given phase."""
    s = os.path.abspath(os.fsdecode(path))
    if not (s.startswith(ROOT) or s.startswith("/mnt/")) or s.endswith((".py", ".pyc", ".so")):
        return True, "out_of_scope"
    if "visa" in s.lower():
        return False, "visa"
    if phase in ("S", "L") and s in TEST_FEATURES:
        return True, "test_features"
    if phase == "L" and (s == MANIFEST or (s.startswith(GT_PREFIX) and "/ground_truth/" in s and "/test/" not in s
                                           and s.count("/ground_truth/") == 1)):
        return True, "labels_phase_L"
    for pat in ALWAYS:
        if pat in s:
            return False, f"always:{pat}"
    if phase in ("S", "L") and s.endswith("_train.pt"):
        return False, "train_features_not_needed"
    for pat in LABEL:
        if pat in s:
            return False, f"label:{pat}"
    return True, "allowed"


def _audit(event, args):
    if event != "open" or not args:
        return
    p = args[0]
    if not isinstance(p, (str, bytes, os.PathLike)):
        return
    ok, why = decide(p, _state["phase"])
    if why == "out_of_scope":
        return
    s = os.path.abspath(os.fsdecode(p))
    if not ok:
        raise PermissionError(f"SCRC-P1-LITE-00 custody violation ({why}, phase {_state['phase']}): {s}")
    key = (_state["phase"], s)
    if key not in _seen:
        _seen.add(key)
        ACCESS.append(key)


def install(phase="PRE"):
    if not _state["installed"]:
        sys.addaudithook(_audit)
        _state["installed"] = True
    set_phase(phase)


def set_phase(phase):
    P.require(phase in PHASES, f"unknown phase {phase}")
    _state["phase"] = phase


def phase():
    return _state["phase"]


def access_log():
    return [{"phase": ph, "path": os.path.relpath(s, ROOT) if s.startswith(ROOT) else s} for ph, s in ACCESS]


class FeatureFile:
    """A CALM-02 feature dict whose label-bearing keys (``rows`` holds relative paths that encode the defect type)
    are reachable only in phase L."""

    def __init__(self, obj, path):
        self._obj, self.path = obj, str(path)

    @property
    def features(self):
        return self._obj["features"]

    def __getitem__(self, k):
        if k != "features" and _state["phase"] != "L":
            raise PermissionError(f"key '{k}' of {self.path} is label-bearing; phase {_state['phase']} may read "
                                  f"'features' only")
        return self._obj[k]


def guarded_features(path):
    import torch
    obj = torch.load(Path(path), map_location="cpu", weights_only=False, mmap=True)
    return FeatureFile(obj, path)
