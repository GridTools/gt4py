---
tags: [typing]
---

# Deprecated `typing` Aliases in `extended_typing`

- **Status**: valid
- **Authors**: Enrique G. Paredes (@egparedes)
- **Created**: 2026-08-10
- **Updated**: 2026-08-10

Why-statement: In the context of `gt4py.eve.extended_typing` acting as the single typing
import for the whole project, facing the fact that PEP 585 made `typing.List` and friends
deprecated spellings of the builtin generics, we decided to stop re-exporting the six
builtin-generic aliases and to reject them explicitly at runtime, to achieve a single
obvious spelling for these types. We considered simply deleting the re-exports and accept
that the guarantee is runtime-only, since a type checker still resolves the names through
the module's star imports.

## Context

`extended_typing` is a drop-in replacement for `typing`: it star-imports `typing` and then
`typing_extensions` (so the latter takes priority), and re-exports a number of names from
their non-deprecated homes. Downstream code does `from gt4py.eve.extended_typing import ...`
rather than importing `typing` directly.

Two groups of re-exports were historically bundled together, but they are not the same kind
of thing:

- **Builtin generics** — `Dict`, `FrozenSet`, `List`, `Set`, `Tuple`, `Type`. Since PEP 585
  the modern spelling is a *different* name: the builtin (`dict`, `list`, ...). The
  PascalCase names are deprecated.
- **Everything else** — `Sequence`, `Callable`, `Mapping`, `Match`, `deque`, ... Here the
  name is already the modern one; only its `typing` home is soft-deprecated. Re-exporting
  `collections.abc.Sequence` as `Sequence` is precisely what this module is for.

Drivers:

- The supported Python floor is now 3.12, so the builtin generics are available
  unconditionally and there is no compatibility reason to keep the aliases.
- Two spellings for the same type invite inconsistency, and the deprecated one is what new
  code tends to copy from its neighbours.
- Ruff's `UP006` cannot help: it rewrites `typing.List` but leaves both `xtyping.List` and
  `from gt4py.eve.extended_typing import List` untouched, even with `typing-modules`
  configured. Enabling the `UP` ruleset would therefore not migrate these use sites.

The decisive constraint is a mechanism that makes "just delete the re-export" ineffective.
The names are *not created* by the re-export block: `from typing import *` and
`from typing_extensions import *` already bind all six, and the block below merely shadows
them with the builtins. On top of that, the module defines a `__getattr__` that forwards any
missing name to `typing_extensions` and then `typing`. Deleting the block alone would leave
all six names in place, silently rebound from the builtins to the deprecated `typing`
objects — the opposite of the intent, and invisible at every use site.

## Decision

Remove the six builtin-generic aliases only, and make the removal effective:

1. Drop them from the module namespace after the star imports, and reject them in
   `__getattr__` with a message naming the replacement. Both `xtyping.List` and
   `from gt4py.eve.extended_typing import List` now fail loudly instead of resolving to
   `typing.List`.
2. Migrate the use sites to the builtins. A bare `typing.Type` means `type[Any]`, whereas a
   bare builtin `type` is stricter (it is not indexable and carries no arbitrary
   attributes), so unsubscripted occurrences become `type[Any]`. The other five aliases need
   no such care.
3. Keep forward-reference resolution unchanged. `eval_forward_ref` binds the name `typing`
   to this module, and resolving a reference through it has always normalized the deprecated
   spellings to the builtin generic. These names remain valid in *user-written* annotations,
   so both `List[int]` and `typing.List[int]` still resolve to `list[int]`.

The second group of re-exports is deliberately left alone.

## Consequences

- There is one spelling for these types in the codebase, and a wrong one fails immediately
  with a message that names the replacement, rather than degrading silently.
- The guarantee is runtime-only. A type checker still sees the names through the star
  imports, so `xtyping.List` type-checks and fails at import time. Closing that gap means
  replacing the star imports with an explicit `__all__`, which is a much larger change and
  is not attempted here.
- Downstream code importing these names from `extended_typing` breaks and must switch to the
  builtins. Code importing them from `typing` directly is unaffected.
- The distinction between the two groups of re-exports is now explicit, so future cleanups
  do not have to re-derive it.

## Alternatives considered

### Delete the re-export block only

- Good, because it is a one-line-per-name change with no use-site churn.
- Bad, because it does not remove anything: the star imports and `__getattr__` keep all six
  names alive, silently rebound from the builtins to the deprecated `typing` objects. It is
  a downgrade disguised as a cleanup.

### Keep the aliases and wait for a ruff rule to migrate the use sites

- Good, because it costs nothing now.
- Bad, because `UP006` does not fire on names sourced from `extended_typing` (verified), so
  the wait would be indefinite.

### Also drop the `collections.abc` / `collections` / `re` / `contextlib` re-exports

- Good, because it would shrink the module to a thinner shim over `typing`.
- Bad, because those names are not deprecated — only their `typing` home is — and providing
  them from their modern home is the module's purpose. It would also force every use site to
  import from two places instead of one, for no correctness gain.

### Replace the star imports with an explicit `__all__`

- Good, because it would make the removal visible to type checkers too, and would end this
  whole class of problem.
- Bad, because it is a large, risky change to the module's contract that is orthogonal to
  the deprecation at hand. Left as possible future work.

## References

- [PEP 585 - Type Hinting Generics In Standard Collections](https://peps.python.org/pep-0585/)
- [PEP 562 - Module `__getattr__` and `__dir__`](https://peps.python.org/pep-0562/)
