#!/usr/bin/env bash
# The 1024x16 challenge cell end to end. Usage: ./run.sh DATASET_DIR   (the Phase 2 public dataset, e.g. aicrowd/arc-whestbench-public-2026, revision v2-phase2)
set -euo pipefail
D=$1; HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
python get_nets.py "$D" 12
python ../arc_ladder/arc.py P SIMPLE2,AUGMENT2          # ARC's K2, unmodified
python ../estimator/lv2.py P 24                          # my estimator
python score_k2.py
python k3_scale_law.py --mlps 0,1,2,3,4,5,6,7,8,9,10,11 --kind SIMPLE --vmode none,k22                    # ARC's factored K3 without / with the scale law
python k3_scale_law.py --mlps 1,2,3,4,5 --kind AUGMENT --vmode none,k22 --dtype float32                   # AUGMENT K3 needs > 110 GB in float64
./graded_ablation.sh "$D" 12
