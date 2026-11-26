#!/bin/bash
# RIGEL-CAD end-to-end MVTec AD 2 run from raw images (owner request 2026-11-26).
cd /mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD
echo "$(date +%FT%T) START"
~/miniconda3/envs/nested_learning/bin/python -B -c "
import sys; sys.path.append('$HOME/miniconda3/envs/pytorch_env/lib/python3.12/site-packages')   # tifffile only
from rigel_cad.cli import main
main(['ad2', '--data-root', '/mnt/k/mvtec_ad_2', '--out', 'results/rigel_cad/ad2', '--cache-dir', '/mnt/k/NLCAD/cache/rigel_cad', '--budget-mib', '8', '--seed', '2711', '--per-stage'])
"
echo "$(date +%FT%T) END rc=$?"
