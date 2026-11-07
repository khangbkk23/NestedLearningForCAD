#!/usr/bin/env bash
# NL-RET-02 G2-G5 (CPU): analysis lock -> tables -> validate (freeze) -> posthoc real-AP alignment -> decision.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
R=results/hope_cad/nl_ret_02
x() { echo "== $(date -u +%FT%TZ) $*"; "$@"; }
[ -f $R/ANALYSIS_LOCK.json ] || x $PY - <<'PYEOF'
from exps.cadfinal import common as K
from exps.dinomaly.common import sha
from exps.nl_ret_02 import common as C
src = ["exps/nl_ret_02/tables.py", "exps/nl_ret_02/validate.py", "exps/nl_ret_02/test_nl_ret_02.py",
       "exps/nl_ret_02/common.py", "exps/nl_ret_02/prereg.py", "exps/nl_ret_02/probe.py", "exps/nl_ret_01a/validate.py"]
K.write_json(C.OUT / "ANALYSIS_LOCK.json", {"sources": {s: sha(C.ROOT / s) for s in src},
             "audit_lock_sha256": sha(C.OUT / "AUDIT_LOCK.json"), "amendment_01_sha256": sha(C.OUT / "AMENDMENT_01.json"),
             "utc": K.now(), "note": "written before FEATURE_TABLE/OUTCOME_TABLE exist; no new feature has been paired "
             "with an outcome before this point"})
PYEOF
[ -f $R/FEATURE_TABLE.csv ] || x $PY -m exps.nl_ret_02.tables
[ -f $R/CROSS_CLASS_VALIDATION.json ] || x $PY -m exps.nl_ret_02.validate validate
[ -f $R/REAL_AP_ALIGNMENT_DESCRIPTIVE.json ] || x $PY -m exps.nl_ret_02.validate align
[ -f $R/FINAL_VERDICT.json ] || x $PY -m exps.nl_ret_02.validate decide
echo "== $(date -u +%FT%TZ) analysis done"
