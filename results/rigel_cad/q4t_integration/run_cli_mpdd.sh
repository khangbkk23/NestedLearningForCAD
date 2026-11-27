#!/bin/bash
# End-to-end CLI smoke test of the int4_triangular backend from raw MPDD images (no token cache).
cd /mnt/e/Development/Python/ContinualLearning/NestedLearningForCAD/NestedLearningForCAD
/home/dikhang_hcmut/miniconda3/envs/nested_learning/bin/python -B -m rigel_cad stream \
    --dataset mpdd --data-root data/mpdd --storage int4_triangular --budget-mib 8
