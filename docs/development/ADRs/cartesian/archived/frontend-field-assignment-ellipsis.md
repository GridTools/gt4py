# field[...] assignment syntax

Gt4Py version 1.2.1 was the last release to allow the following syntax:

```py
with computation(PARALLEL), interval(...):
    field[...] = 42   # support for [...] is dropped
    field = 42.42     # <- use this instead
```

Support for `[...]` as "the full field" has been removed because it is redundant with just `field = 42`, which is less verbose and widely adopted.

## Context

Both `field[...] = 42` and `field = 42` were allowed and doing the same thing. Having redundant syntax is generally a bad idea (more parsing overhead, possibility of inconsistent user code, maintenance cost). In this case we felt the maintenance cost as the syntax is - as far as we could gather - unused, yet in periodic updates had to deal with this extra syntax.

## Decision

The decision to use `field = 42` over `field[...] = 42` was captured in [this issue](https://github.com/GridTools/gt4py/issues/2406) and basically weighs the maintenance overhead against the usefulness/redundancy of the syntax. Since nobody was using the syntax (anymore) and we had to do maintenance work to support the syntax, it was decided to remove support for the syntax. Since we don't expect any usage, no deprecation period was chosen.

## Consequences

[This PR](https://github.com/GridTools/gt4py/pull/2820) drops support for the syntax in favor of just assigning to the field without any subscript. This avoids maintenance cost. We drop a custom error for any residual usage.
