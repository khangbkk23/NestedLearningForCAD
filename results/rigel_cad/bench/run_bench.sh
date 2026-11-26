#!/bin/bash
# RIGEL-CAD speed / memory benchmark from raw images (owner request 2026-11-26). One GPU job at a time.
cd /mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD
PY=~/miniconda3/envs/nested_learning/bin/python
for DS in ${DATASETS:-mpdd mvtec visa}; do
  echo "$(date +%FT%T) START $DS"
  $PY -B -m rigel_cad bench --dataset $DS --data-root data/$DS
  echo "$(date +%FT%T) END $DS rc=$?"
done
echo "$(date +%FT%T) START ad2"
$PY -B -c "
import sys; sys.path.append('$HOME/miniconda3/envs/pytorch_env/lib/python3.12/site-packages')
from rigel_cad.cli import main
main(['bench', '--dataset', 'ad2', '--data-root', '/mnt/k/mvtec_ad_2'])
"
echo "$(date +%FT%T) END ad2 rc=$?"
