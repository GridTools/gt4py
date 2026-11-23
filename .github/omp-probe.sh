#!/usr/bin/env bash
# TEMP: macOS OpenMP-runtime probe for PR #3031, revert before merge.
set -u
uv lock --upgrade
uv python upgrade 3.12
./noxfile.py -p 3.12 -s "test_next-3.12(internal, cpu, nomesh)" "test_next-3.12(internal, cpu, atlas)" --install-only
ls .nox
PY=$(ls -d .nox/test_next-3-12-internal-cpu-nomesh)/bin/python
PYA=$(ls -d .nox/test_next-3-12-internal-cpu-atlas)/bin/python
SP=$($PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
SPA=$($PYA -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
BREW_OMP=$(brew --prefix libomp)/lib/libomp.dylib
export GT4PY_BUILD_CACHE_DIR=$PWD/omp-probe-cache
export GT4PY_BUILD_CACHE_LIFETIME=persistent

h() { echo; echo "##### PROBE $*"; }

h "install names"
echo "brew: $(otool -D "$BREW_OMP" | tail -1) realpath $(realpath "$BREW_OMP")"
ls "$SP/torch/lib"
otool -D "$SP/torch/lib/libomp.dylib"

h "openmp-like dylibs shipped in site-packages (nomesh, atlas)"
find "$SP" "$SPA" \( -name 'libomp*' -o -name 'libiomp*' -o -name 'libgomp*' \) -print

h "site-packages binaries linking an OpenMP runtime (nomesh, atlas)"
for sp in "$SP" "$SPA"; do
  find "$sp" \( -name '*.so' -o -name '*.dylib' \) -print | while read -r f; do
    l=$(otool -L "$f" 2>/dev/null | grep -iE 'omp' | grep -v "^$f")
    [ -n "$l" ] && echo "== ${f#$sp/}" && echo "$l"
  done
done

h "images loaded by import gt4py.next"
DYLD_PRINT_LIBRARIES=1 $PY -c "import gt4py.next" 2>&1 | grep -iE 'omp' || echo "(none)"
h "images loaded by import torch + op"
DYLD_PRINT_LIBRARIES=1 $PY -c "import torch; print(torch.ones(1<<20).sum())" 2>&1 | grep -iE 'omp|tensor' || echo "(none)"

T_GTFN="tests/next_tests/integration_tests/feature_tests/ffront_tests/test_basic.py"
T_TORCH="tests/next_tests/unit_tests/embedded_tests/test_nd_array_field_with_torch.py"
T_BENCH="tests/next_tests/benchmarks/benchmark_program_call.py"
run() { local name=$1; shift; h "pytest $name"; DYLD_PRINT_LIBRARIES=1 "$@" > "omp-probe-$name.log" 2>&1; echo "exit=$?"; grep -E 'libomp|OMP:|passed|failed|Fatal' "omp-probe-$name.log" | sort | uniq -c | head -30; }
P="$PY -m pytest -n 0 -p no:cacheprovider -v --benchmark-disable"
run gtfn_only   $P "$T_GTFN" -k "test_copy and gtfn"
run torch_only  $P "$T_TORCH"
run torch_gtfn  $P "$T_TORCH" "$T_GTFN" -k "torch or (test_copy and gtfn)"
run gtfn_torch  $P "$T_GTFN" "$T_TORCH" -k "torch or (test_copy and gtfn)"
run dace_only   $P "$T_BENCH" -k "benchmark_copy_01_arg_program and dace"
run torch_dace  $P "$T_TORCH" "$T_BENCH" -k "torch or (benchmark_copy_01_arg_program and dace)"
run atlas_gtfn  $PYA -m pytest -n 0 -p no:cacheprovider -v -m requires_atlas tests/next_tests/integration_tests/multi_feature_tests/iterator_tests/test_fvm_nabla.py -k "test_compute_zavgS and not zavgS2"

h "built extensions linking an OpenMP runtime"
find "$GT4PY_BUILD_CACHE_DIR" "$HOME/.dacecache" .dacecache \( -name '*.so' -o -name '*.dylib' \) -print 2>/dev/null | while read -r f; do
  echo "== $f"; otool -L "$f" | grep -iE 'omp'
done
exit 0
