# Compiler support and detection

In the context of supporting multiple compilers, facing different compiler defaults and flag names, we decided to auto-detect compilers (CPU & GPU) to achieve a common baseline behavior across compilers by setting defaults per compiler. We understand that compiler detection isn't trivial and provide environment variables to override our defaults.

## Context

GT4Py cartesian was historically developed against the GNU C compiler. Going towards production and supporting users with macos laptops, drove the need to support other compilers such as `clang`, `apple-clang`, and `icx`. Those four compilers

1. support different features out of the box, e.g. `apple-clang` doesn't ship OpenMP support by default,
2. have different defaults, e.g. FMA support in `-O0` is inconsistent,
3. have different names for flags, e.g. OpenMP flags of `icx`.

GPU compilers were another driver. For one, they are known to not always use the same name for compiler flags as their host compilers. And in addition, supporting both NVIDIA and AMD cards, requires different library paths to be supplied.

## Decision

We chose to detect the CPU and GPU compilers as part of initialization. This allows us to make decisions per compiler (possibly per compiler version in the future) and have defaults that serve our users. We understand that our defaults are choices and thus continue the practice of providing environment variables to override the defaults.

## Consequences

With CPU and GPU compiler detection, we can set defaults for supported features (e.g. OpenMP on/off), have consistent defaults across supported compilers (e.g. FMA turned off in optimization level 0), and supply the compiler flags per compiler if they don't match (e.g. `-qopenmp` for `icx` and `-fopenmp` for `gcc` and `clang`).

We support environment variables to override our default choices, e.g. to run with OpenMP on `apple-clang` when OpenMP is manually installed.

One downside of the current implementation is that defaults are global in the sense that they apply for all backends. This can be challenging when different backends have conflicting needs. For example, GridTools-based backends have a hard requirement on OpenMP while DaCe-based backends can generate code that works with and without OpenMP. We decided to turn OpenMP support off by default on `apple-clang` because OpenMP isn't available out of the box. For users that run GridTools-based backends on `apple-clang`, this means that they have to set an environment variable to enable OpenMP support. We think this is appropriate because those users would anyway need to set environment variables to configure custom compiler and linker flags.

## Alternatives considered

The following alternatives were considered.

### Config file

This would be nice to configure defaults and overrides in a (possibly cascading) configuration file. That would move configuration out of code and allow to remove the need for environment variables. Environment variables can be a problem because they are global, e.g. they could interfere with other programs / processes. We currently mitigate that by prefixing GT4Py-only environment variables with `GT4Py_`.

The idea of a config file was discarded as this point in time because it would be a bigger refactor than continuing with configuration in code, as done previously. We might go to a set of possibly cascading config files in the future.

### Configuration per backend

To account for different needs depending on backend groups (e.g. OpenMP support discussed under consequences), we could have defaults not only per compiler but also per backend. We decided that it's not worth the effort and provide the alternative to configure via environment variables.
