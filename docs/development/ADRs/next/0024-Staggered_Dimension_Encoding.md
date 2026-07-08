---
tags: []
---

# Encoding Staggered Dimensions via a Name Prefix

- **Status**: valid
- **Authors**: Till Ehrengruber (@tehrengruber)
- **Created**: 2026-07-08
- **Updated**: 2026-07-08

A staggered dimension is encoded as its base dimension's name with the internal
`_Staggered` prefix (`common._STAGGERED_PREFIX`), rather than as a new attribute
on `Dimension`. Helpers `is_staggered`, `flip_staggered` and `as_non_staggered`
operate purely on that prefix.

## Context

Dimensions are identified by their **name** and appear throughout the toolchain
in more than one form: as a `common.Dimension` instance, but also as an
`itir.AxisLiteral` (and as plain strings in generated backend code). Adding a
"staggered" flag to `Dimension` would have required threading that flag through
every one of these representations and every place that constructs, compares, or
lowers a dimension — a large, invasive change touching the frontend, the IR, and
all backends.

## Decision

We take the pragmatic route of carrying the staggered marker *in the name
itself* via the `_Staggered` prefix. Because a name already round-trips cleanly
through all dimension representations, no format has to learn about staggering:
the information survives conversion to `AxisLiteral` and to backend tags for
free. The leading underscore marks the prefix as internal, signalling that it is
an implementation detail and not part of the user-facing API.
