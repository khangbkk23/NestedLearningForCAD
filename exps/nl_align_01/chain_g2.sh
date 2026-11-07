#!/usr/bin/env bash
# NL-ALIGN-01 G2 chain: runs only if G1_DECISION.json authorizes G2 (g2.verify enforces it again per command).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python
export PYTHONPATH=$PWD NLCAD_CAP_GIB=3.4
$PY -c "import json,sys; d=json.load(open('results/hope_cad/nl_align_01/g1/G1_DECISION.json')); sys.exit(0 if d['g2_authorized'] else 3)" \
  || { echo "G1 gate not passed: G2 not run"; exit 0; }
g() { echo "== $(date -u +%FT%TZ) $*"; $PY -m exps.nl_align_01.g2 "$@"; }
[ -f results/hope_cad/nl_align_01/g2/HYPERGRAD_AUDIT.json ] || { echo "== $(date -u +%FT%TZ) hypergradient audit"; $PY -m exps.nl_align_01.g2_audit; }
for s in 2711 2712 2713; do for a in NL SC RESET CLO RND; do g train --arm $a --seed $s; done; done
for s in 2711 2712 2713; do for a in NL SC RESET CLO RND; do g diag --arm $a --seed $s; done; done
[ -f results/hope_cad/nl_align_01/g2/G2_SEAL.json ] || g seal
[ -f results/hope_cad/nl_align_01/g2/G2_DECISION.json ] || g decide
echo "== $(date -u +%FT%TZ) G2 chain done"
