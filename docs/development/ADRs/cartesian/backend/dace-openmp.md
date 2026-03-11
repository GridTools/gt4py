# Enable/Disable OpenMP in DaCe-based backends

In the context of portability, facing compilers without out-of-the-box OpenMP support, we decided to add an environment variable allowing power users to enable/disable OpenMP at will. We'll have cautious defaults based on compiler detection. Power users will be able to override our defaults by setting environment variables.

## Context

The ability to avoid any OpenMP statements in generated code is motivated by compilers that don't ship with OpenMP by default (e.g. `apple-clang`). Power users might want to disable OpenMP to measure overhead or in case of hyper-scaling scenarios.

## Decision

We decided to make OpenMP support user configurable through `GT4PY_CARTESIAN_ENABLE_OPENMP`. The default is set based on the compiler's out-of-the-box support for OpenMP, i.e. we detect the compiler and set the default according to our knowledge. For example, for `apple-clang` we turn it off by default and for `gcc`, we turn it on by default.

Power users will be able to override our decision by setting the environment variables `GT4PY_CARTESIAN_ENABLE_OPENMP`, `OPENMP_CPPFLAGS`, and `OPENMP_LDFLAGS`. This allows to e.g. enable OpenMP with `apple-clang` or disable OpenMP with `gcc`. The environment variables also allow power users to configure OpenMP support for not (yet) detected compilers.

## Consequences

Users of GT4Py gain a compiler-dependent way to control OpenMP flags and turn support on/off at will. We see this being used in three scenarios:

1. Supporting compilers that don't ship OpenMP by default (e.g. `apple-clang`).
2. It could be used in hyper-scaling scenarios where the load is distributed via MPI ranks and each rank only runs with one OpenMP thread. In such a scenario, users could now turn off OpenMP. More generally, turning OpenMP support off could be a tool in performance work to measure the overhead of running with OpenMP.
3. In case a new compiler emerges (we support `gcc`, `clang`, `apple-clang`, and `icx`), users have a way to enable/disable OpenMP support and configure the respective flags.

## Alternatives considered

### Auto-detecting compiler capability

After a brief time without compiler detection, we do now attempt to detect the compiler and infer its OpenMP support and flags as default.
