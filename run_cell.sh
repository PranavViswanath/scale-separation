#!/usr/bin/env bash
# Reproduce one cell end to end. Usage: ./run_cell.sh B 64 16 10   (cell name, width, depth, number of nets)
# Needs ARC's mlp_cumulant_propagation (commit 93d091a) importable as mlp_kprop, and one CUDA GPU.
set -euo pipefail
CELL=$1; N=$2; L=$3; T=$4
python nets.py "$CELL" "$N" "$L" "$T"      # random nets, fixed seeds
python mc.py "$CELL" 30                    # Monte Carlo truth, two independent halves (2^30 samples; use 28 for n >= 256)
python arc.py "$CELL"                      # ARC's ladder, unmodified
python lv2.py "$CELL" 24                     # scale-separation estimator: exact variance rule, plus old-rule and first-order controls
python ana.py "$CELL"                      # errors, CIs, ratios vs sampling and vs ARC's best
