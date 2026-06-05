---
tags: []
---

# [Runtime domains]

- **Status**: valid
- **Authors**: Till Ehrengruber (@tehrengruber)
- **Created**: 2025-09-01
- **Updated**: 2025-09-01

The mechanism for representing domains in the IR and accesing their values at runtime has been updated with the introduction of a new builtin on GTIR called `get_domain_range(field, dimension)`.

## History

In the early days the domain of a field was represented by a set of implicit (scalar) parameters that were added in the frontend (e.g. from PAST -> GTIR). This mechanism increased the complexity of the frontend and created artifical differences between the frontend representation (PAST) and GTIR.

## Decision

`get_domain_range(field, dimension) -> (start, stop)` takes a field and a dimension as an input and returns a tuple of integers containing the start and stop indices of given dimension in the domain of the field. Eventually we want a builtin that returns the entire domain of a field. However, this was withdrawn due to the effort required to design & implement it properly. We identified the following issues:

- The gtfn backend currently selects a backend, i.e. cartesian or unstructured, based on the type of the domain object on which a stencil is executed. Additionally, for unstructured all neighbor tables are stored in the domain. A field or sid, however, does and should not have a notion of a backend, so constructing a domain from a sid is not possible without hacks. We should therefore not encode the backend in the domain. The same applies for the connectivities.
- The domain in gtfn consists of a start index and a length, instead of a start and stop index. Initial work has started in GridTools for the cartesian domain (by @tehrengruber), but this has not been pursued any further because of the above issue.
