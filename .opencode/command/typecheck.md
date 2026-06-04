---
description: Type-check GT4Py and verify module-dependency boundaries
---

Check static typing and architectural boundaries.

1. Run `uv run mypy src/` (mypy runs on `src/` only — tests/docs/examples are excluded).
2. Run `uv run tach check` to verify module-dependency layering (`eve → _core → storage → {cartesian, next}`, per `tach.toml`).
3. Summarize any violations grouped by file, and propose minimal fixes consistent with
   `CODING_GUIDELINES.md`. Do not loosen type strictness to silence errors.
