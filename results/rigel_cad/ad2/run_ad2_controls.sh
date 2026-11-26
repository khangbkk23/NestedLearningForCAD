#!/bin/bash
# MVTec AD 2 controls with the corrected test_public masks, through the official package (owner request 2026-11-26):
#   U = RIGEL-CAD unbounded, F = fast level only at 8 MiB (ablation). Final stage only, no submission maps.
cd /mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD
PY=~/miniconda3/envs/nested_learning/bin/python
for ARM in U F; do
  if [ $ARM = U ]; then X="--budget-mib none"; else X="--budget-mib 8 --fast-only"; fi
  echo "$(date +%FT%T) START $ARM"
  $PY -B -c "
import sys; sys.path.append('$HOME/miniconda3/envs/pytorch_env/lib/python3.12/site-packages')   # tifffile only
from rigel_cad.cli import main
main(['ad2', '--data-root', '/mnt/k/mvtec_ad_2', '--out', 'results/rigel_cad/ad2/controls/$ARM', '--cache-dir', '/mnt/k/NLCAD/cache/rigel_cad', '--seed', '2711', '--no-submission'] + '$X'.split())
"
  echo "$(date +%FT%T) END $ARM rc=$?"
done
