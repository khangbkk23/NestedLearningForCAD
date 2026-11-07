#!/usr/bin/env bash
# NL-RET-01A Phases 3-6 (CPU): analysis lock -> tables -> validate (G3/G4, freeze) -> posthoc align (G5) -> decide.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
R=results/hope_cad/nl_ret_01a
x() { echo "== $(date -u +%FT%TZ) $*"; "$@"; }
[ -f $R/ANALYSIS_LOCK.json ] || x $PY - <<'PYEOF'
from exps.cadfinal import common as K
from exps.dinomaly.common import sha
from exps.nl_ret_01a import common as C
src = ["exps/nl_ret_01a/features.py", "exps/nl_ret_01a/validate.py", "exps/nl_ret_01a/test_nl_ret_01a.py",
       "exps/nl_ret_01a/common.py", "exps/nl_ret_01a/extract.py", "exps/nl_ret_01a/prereg.py"]
K.write_json(C.OUT / "ANALYSIS_LOCK.json", {"sources": {s: sha(C.ROOT / s) for s in src},
             "audit_lock_sha256": sha(C.OUT / "AUDIT_LOCK.json"), "utc": K.now(),
             "note": "written before FEATURE_TABLE/OUTCOME_TABLE exist"})
PYEOF
[ -f $R/FEATURE_TABLE.csv ] || x $PY -m exps.nl_ret_01a.features
[ -f $R/CROSS_CLASS_VALIDATION.json ] || x $PY -m exps.nl_ret_01a.validate validate
[ -f $R/REAL_AP_ALIGNMENT_DESCRIPTIVE.json ] || x $PY -m exps.nl_ret_01a.validate align
[ -f $R/FINAL_VERDICT.json ] || x $PY -m exps.nl_ret_01a.validate decide
echo "== $(date -u +%FT%TZ) analysis done"
