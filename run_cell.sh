#!/usr/bin/env bash
# Reproduce one cell end to end. Usage: ./run_cell.sh B 64 16 10   (cell name, width, depth, number of nets)
# Needs ARC's mlp_cumulant_propagation (commit 93d091a) importable as mlp_kprop, and one CUDA GPU.
# Nets, truth and results are written to runs/.
set -euo pipefail
CELL=$1; N=$2; L=$3; T=$4
ROOT=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ROOT/runs" && cd "$ROOT/runs"
python "$ROOT/evaluation/nets.py" "$CELL" "$N" "$L" "$T"   # random nets, fixed seeds
python "$ROOT/evaluation/mc.py" "$CELL" 30                 # Monte Carlo truth, two independent halves (2^30 samples; use 28 for n >= 256)
python "$ROOT/arc_ladder/arc.py" "$CELL"                   # ARC's ladder, unmodified
python "$ROOT/estimator/lv2.py" "$CELL" 24                 # scale-separation estimator: exact variance rule, plus old-rule and first-order controls
python "$ROOT/evaluation/ana.py" "$CELL"                   # errors, CIs, ratios vs sampling and vs ARC's best
