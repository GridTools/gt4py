# Enable/Disable OpenMP in DaCe-based backends

In the context of portability, facing compilers without out-of-the-box OpenMP support, we decided to add a environment variable allowing power users to enable/disable OpenMP at will. We considered auto-detection of OpenMP capabilities and decided against it.

## Context

The ability to avoid any OpenMP statements in generated code is motivated by compilers that don't ship with OpenMP by default (e.g. `apple-clang`).

## Decision

We decided to make OpenMP support user configurable through `GT4PY_CARTESIAN_ENABLE_OPENMP`. The default is `True` meaning we keep generating OpenMP-accelerated code by default.

We decided to make OpenMP support user configurable and not auto-detect the compiler's ability; see discussion below.

## Consequences

Users of GT4Py gain an easy way to turn OpenMP support on and off. We see this being used in two scenarios:

1. Supporting compilers that don't ship OpenMP by default (e.g. `apple-clang`).
2. It could be used in hyper-scaling scenarios where the load is distributed via MPI ranks and each rank only runs with one OpenMP thread. In such a scenario, users could now turn off OpenMP. More generally, turning OpenMP support off could be a tool in performance work to measure the overhead of running with OpenMP.

## Alternatives considered

### Auto-detecting compiler capability

We discussed auto-detecting the compiler's capability to handle OpenMP pragmas. While this seem appealing for the first use case (missing compiler support), it is inflexible with regards to the hyper-scaling use-case.

We might still add a compiler capability check in the future to harden against user-side misconfiguration (e.g. generating OpenMP pragmas without compiler support).
