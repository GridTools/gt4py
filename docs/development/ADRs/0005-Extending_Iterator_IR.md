---
tags: [general]
---

# Extending Iterator IR

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2022-04-19
- **Updated**: 2022-04-19

When you should and when you shouldn't add an extension to Iterator IR.

## Decision

Iterator IR is the central structure of the Declarative GT4Py, it is the interface between frontend and backend, and the target of (most) optimizations.
IR elements and the interactions with each other need to be well understood and well documented.

Therefore, if it can be avoided, the IR should not be extended. Instead, required features can be implemented first in the backend where they are needed. If later we discover the features are needed in several backends, they can be promoted to the IR.

### Example

A recent example is the allocation of temporaries. We decided to add this feature only to the C++ backend for now, it's a performance optimization that we don't need in Python execution. While adding temporary allocation to the IR, seems straight-forward, it requires several new builtins, e.g. to express domain size relative to the domain passed by the user.
