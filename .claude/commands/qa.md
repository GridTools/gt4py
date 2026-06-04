---
description: Run GT4Py quality-assurance checks (pre-commit) and fix until clean
---

Run the project's quality-assurance hooks on the current changes and make them pass.

1. Run `pre-commit run` (staged changes) — or `pre-commit run -a` if asked for the whole repo.
2. This covers ruff lint+format, mypy (on `src/`), tach module boundaries, and file formatters.
3. If anything fails or auto-fixes files, review the changes, then re-run until everything passes.
4. Report a short summary of what was fixed and the final status.

Use `uv run ruff check --fix <paths>` / `uv run ruff format <paths>` for targeted fixes.
Always go through `uv` — never bare `pip`/`pytest`/`python`.
