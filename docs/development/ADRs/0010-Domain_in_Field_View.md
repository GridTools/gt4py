---
tags: [frontend]
---

# Domain in Field View

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2022-08-24
- **Updated**: 2022-08-24

In the context of the ICON port we needed multiple `out` arguments in calls to `field_operator`s from `program`s. However,
the domain was deduced relative to _the_ `out` argument. We decided to deduce the domain from the first field for now as described below.

## Context

The compute domain in field view is deduced relative to the shape of fields in the `out` argument. This is straight-forward for a single field.
For multiple fields we would need to check at call-time if all fields are sliced in the correct way.
Additionally, expressing absolute domains is desirable for some use-cases.
In summary, we need to revisit the domain handling in field view.

## Decision

For now, we implement multiple `out` fields (tuple) without runtime checks, but require that all fields are sliced in the same way.
Consequently, all fields have to have the same shape.

This restricting should be lifted, but as the domain handling is not final, we decided to implement this pragmatic solution for now.
