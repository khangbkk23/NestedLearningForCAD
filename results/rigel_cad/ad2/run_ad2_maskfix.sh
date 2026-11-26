#!/bin/bash
# RIGEL-CAD MVTec AD 2 re-run after the test_public mask fix (owner request 2026-11-26).
# Same stream, config and seed as run_ad2.sh; only the ground-truth lookup changed (good images -> all-normal mask).
# The submission maps do not depend on masks and were already verified byte-identical, so they are not rewritten.
cd /mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD
echo "$(date +%FT%T) START"
~/miniconda3/envs/nested_learning/bin/python -B -c "
import sys; sys.path.append('$HOME/miniconda3/envs/pytorch_env/lib/python3.12/site-packages')   # tifffile only
from rigel_cad.cli import main
main(['ad2', '--data-root', '/mnt/k/mvtec_ad_2', '--out', 'results/rigel_cad/ad2', '--cache-dir', '/mnt/k/NLCAD/cache/rigel_cad', '--budget-mib', '8', '--seed', '2711', '--per-stage', '--no-submission'])
"
echo "$(date +%FT%T) END rc=$?"
