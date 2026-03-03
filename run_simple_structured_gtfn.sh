#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="/home/raphael/Documents/Studium/Msc_thesis/gt4py/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

export SIMPLE_STRUCTURED_BACKEND="gtfn"
export GT4PY_BUILD_CACHE_LIFETIME="persistent"
export GT4PY_BUILD_CACHE_DIR="$ROOT_DIR/.cache_gtfn"
rm -rf "$ROOT_DIR/.cache_gtfn"


"$PYTHON_BIN" tests/next_tests/integration_tests/multi_feature_tests/ffront_tests/test_simple_structured.py \
  > out_gtfn.txt 2> err_gtfn.txt

echo "Done. Backend=gtfn, cache=$GT4PY_BUILD_CACHE_DIR/.gt4py_cache"