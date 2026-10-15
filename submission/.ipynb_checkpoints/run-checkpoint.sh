#!/bin/bash
# The classifier plugged into the fit: run.sh <dataset_dir> <output_dir> -> <output_dir>/RESULT.json (fit.py, numpy + scipy; model.json from train.py).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$2"
python3 "$HERE/fit.py" "$1" "$2"
