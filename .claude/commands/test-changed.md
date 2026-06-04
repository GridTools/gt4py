---
description: Run the GT4Py tests for the subpackage(s) you changed (fast CPU loop)
---

Run the targeted test suite for whatever changed.

1. Inspect `git status`/`git diff --name-only` to find which subpackage(s) under `src/gt4py/`
   were touched: `cartesian`, `eve`, `next`, or `storage`.
2. Run the matching test root with `uv run pytest`:
   - `cartesian` → `tests/cartesian_tests/`
   - `eve` → `tests/eve_tests/`
   - `next` → `tests/next_tests/`
   - `storage` → `tests/storage_tests/`
     Example: `uv run pytest tests/eve_tests/ -x -q` (add `-n auto` for parallel).
3. Narrow further with `-k <pattern>` when only part of a subpackage is affected.
4. Report pass/fail with the failing test names; for thorough matrix coverage suggest the
   relevant `./noxfile.py -s "..."` session (note GPU sessions can't run here).
