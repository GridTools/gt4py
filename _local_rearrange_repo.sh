#!/bin/bash

set -e
set -x

LOCAL_FOLDER_ROOT="_local"

# ---- General ----
rm -Rf src/*.egg*
rm -Rf src/gt4py/*.egg*
mv src/gt4py src/orig_gt4py
mkdir -p src/gt4py/
mv tests orig_tests
mkdir -p tests

# ---- Eve ----
echo "[EVE] Move files"
mv src/eve src/gt4py/eve
mv orig_tests/eve_tests tests/eve_tests

# ---- Storage ----
echo "[STORAGE] Move files"
mv src/orig_gt4py/storage src/gt4py/storage
mkdir -p tests/storage_tests/unit_tests
mv orig_tests/test_unittest/test_storage.py tests/storage_tests/unit_tests/
mv orig_tests/test_unittest/test_layouts.py tests/storage_tests/unit_tests/
cp orig_tests/conftest.py tests/storage_tests/

# ---- GT4PY ----
echo "[GT4PY-cartesian] Move files"
mv src/orig_gt4py src/gt4py/cartesian
mv src/gtc src/gt4py/cartesian/gtc
mv orig_tests tests/cartesian_tests
mv tests/cartesian_tests/test_unittest tests/cartesian_tests/unit_tests
mv tests/cartesian_tests/test_integration tests/cartesian_tests/integration_tests
mv examples orig_examples
mkdir examples
mv orig_examples examples/cartesian


# ---- Postprocess ----
# Add new files
cp -r $LOCAL_FOLDER_ROOT/* .
git show functional:README.md > ./README.md
# git show functional:AUTHORS.md > ./AUTHORS.md
# git show functional:CODING_GUIDELINES.md > ./CODING_GUIDELINES.md
# git show functional:CONTRIBUTING.md > ./CONTRIBUTING.md

# Add changes to git
set +e
#git add -A .
git status
