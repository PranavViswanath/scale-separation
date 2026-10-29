#!/usr/bin/env bash
# Graded entry 336823 with and without scale separation, through the challenge's official local runner (whestbench 0.16.2, flopscope 0.12.3).
# Usage: ./graded_ablation.sh DATASET_DIR [n_mlps]. Needs the whest starter kit environment (the `whest` command).
# V30_DESCALE is the entry's kill switch for the scale separation; "off" is the same program with it set to 0 (its V29 base, bit for bit).
set -euo pipefail
D=$1; N=${2:-12}; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HERE/runs/on" "$HERE/runs/off"
cp "$HERE/graded_entry_336823.py" "$HERE/runs/on/estimator.py"
sed 's/_os.environ.get("V30_DESCALE", "1")/_os.environ.get("V30_DESCALE", "0")/' "$HERE/graded_entry_336823.py" > "$HERE/runs/off/estimator.py"
for S in on off; do
  whest run --estimator "$HERE/runs/$S/estimator.py" --runner local --dataset "$D" --split mini --n-mlps "$N" --format json > "$HERE/runs/graded_$S.json"
done
python - "$HERE/runs" <<'PY'
import json, sys
r = {s: {m["mlp_index"]: m for m in json.load(open(f"{sys.argv[1]}/graded_{s}.json"))["results"]["per_mlp"]} for s in ("on", "off")}
for i in sorted(r["on"]):
    a, b = r["on"][i], r["off"][i]
    print(f"mlp {i:2d}  final MSE with {a['final_layer_mse']:.3e}  without {b['final_layer_mse']:.3e}  ratio {b['final_layer_mse'] / a['final_layer_mse']:.2f}  util {a['flops_used'] / 2**41:.3f} / {b['flops_used'] / 2**41:.3f}")
PY
