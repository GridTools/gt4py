# gt4py.next — Agent Instructions

Conventions specific to `src/gt4py/next/` and `tests/next_tests/`, layered on
the repo-wide [root `AGENTS.md`](../../../AGENTS.md).

`gt4py.next` is the successor to `gt4py.cartesian`, targeting both Cartesian
and unstructured / mesh grids. A program is lowered through a fixed
toolchain:

```
field operators / programs (ffront)
  → FOAST / PAST          frontend ASTs
  → Iterator IR / GTIR    iterator/
  → OTF program source    otf/
  → backend               program_processors/runners/  (gtfn, dace, roundtrip)
```

## Module map (capabilities, not paths)

- `common` / `common.py` — the core data model: `Field`, `Dimension`,
  `Domain`, `UnitRange`, `Connectivity`, `GridType`.
- `ffront/` — the declarative frontend: `@field_operator`, `@program`,
  `@scan_operator`, `fbuiltins`, and lowering to FOAST/PAST.
- `iterator/` — Iterator IR (ITIR/GTIR), its transforms, and embedded
  execution of the IR.
- `embedded/` — pure-Python field execution (NumPy / CuPy / JAX); the
  debugging path and the `roundtrip` reference semantics.
- `otf/` — on-the-fly compilation toolchain (workflow steps, caching,
  argument descriptors).
- `program_processors/runners/` — the backends: `gtfn` (GridTools C++),
  `dace` (DaCe SDFG), `roundtrip` / `double_roundtrip` (pure Python).
- `type_system/` — `next` type specifications and the type inference the
  frontend runs.
- `errors/` — user-facing DSL error formatting.

## Conventions specific to next

- `__init__.py` deliberately re-exports the public API (`from … import *`
  with `__all__`) — a documented exception to the Google import style (see
  the module docstring). Export new public symbols there.
- Frontend code lives in two worlds: **traced** (Python inside
  `@field_operator` / `@program`, captured into FOAST/PAST — not executed as
  written) and **embedded** (executed directly on fields). Be explicit about
  which you are changing.
- Significant design decisions are in
  [`docs/development/ADRs/next/`](../../../docs/development/ADRs/next/). Start
  with `0001` (frontend design), `0011`/`0012` (OTF), `0019`
  (connectivities), and `0015` (test exclusion matrices).

## Testing (read before adding next tests)

Next tests run across a **backend matrix** (embedded NumPy/CuPy/JAX, `gtfn`,
`dace`, `roundtrip`). The matrix and the per-backend xfail/skip lists are the
**test-exclusion matrix** (ADR 15) in
[`tests/next_tests/definitions.py`](../../../tests/next_tests/definitions.py).

- Use the `cases` framework in
  `tests/next_tests/integration_tests/cases.py`: the `cartesian_case` /
  `unstructured_case` fixtures supply a backend + allocator; build inputs
  with `cases.allocate(...)` and check with `cases.verify` /
  `cases.verify_with_default_data`.
- If a test exercises a feature some backend can't handle, mark it with the
  matching `USES_*` / `CHECKS_SPECIFIC_ERROR` marker from `definitions.py`
  and add that marker to the backend's skip list — do **not** silently drop
  the backend. A new feature marker must be added to both places.
- Layout: `unit_tests/` (per-module, backend-free), `feature_tests/` (one DSL
  feature across the matrix), `multi_feature_tests/` (end-to-end programs).
- Run with `uv run pytest tests/next_tests/ -x -q`; matrix-level confidence
  comes from `uv run nox -s "test_next-<py>(...)"`. GPU and `dace` sessions
  may be unavailable locally and will skip.
